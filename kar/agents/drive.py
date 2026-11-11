"""`kar run`: drive a team of CLI coding agents through a run, unattended.

The harness needs nothing from an agent except that it can run shell commands and edit files. The driver keeps
`lanes` agent sessions going at once; each one is the agent's own headless command, started fresh in the run's
workspace with a short prompt that names its lane and its command (`./lanes/<id>/kar`). The scheduler in the run
decides what each new session works on. While sessions run, this process serves the run's request queue
(server.py), so every agent can sit in whatever sandbox its CLI provides; the GPU work happens here.

Agents can differ per lane: `--agent claude,codex` alternates them. Presets cover CLIs we know; anything else works
through a command template:
    kar run RUN --agent-cmd 'mytool --prompt-file {prompt_file} --dir {ws}'
Placeholders: {prompt} (the text), {prompt_file} (a file holding it), {ws} (the workspace, the agent's working
folder), {lane}, {tool} (the lane's command), {writes} (the lane's folder, as a glob), {model}.

Each agent's own subagents are allowed where the CLI has them and keeps them inside the same permissions (Claude
Code: verified; Codex: subagents inherit the parent's sandbox, per its docs). Only the lane's command touches the GPU.
"""
import json
import os
import shlex
import signal
import string
import subprocess
import threading
import time

from kar.run import server
from kar.run.session import PROMPTS, Run
from kar.run.store import now

_GROK_ENV = {f"GROK_{src}_{what}_ENABLED": "false" for src in ("CLAUDE", "CURSOR")
             for what in ("AGENTS", "HOOKS", "MCPS", "RULES", "SKILLS")}

# Each preset: the command, flags for a model and a per-session dollar cap, extra environment, how the prompt goes
# in, and how to read token use from its output. "tested" says whether we ran it end to end in a team run.
PRESETS = {
    "claude": {
        # --safe-mode: none of the user's own CLAUDE.md, hooks, plugins or MCP servers. Subagents (Agent) inherit
        # these permission rules; tested: a subagent could run the lane's command and write only the lane folder.
        "cmd": ["claude", "-p", "{prompt}", "--safe-mode", "--output-format", "stream-json", "--verbose",
                "--permission-mode", "dontAsk", "--tools", "Bash,Read,Edit,Write,Glob,Grep,Agent",
                "--allowedTools", "Bash({tool})", "Bash({tool} *)", "Read(./**)", "Glob(./**)", "Grep(./**)",
                "Edit({writes})", "Write({writes})", "Agent", "--no-session-persistence"],
        "model": ["--model", "{model}"], "usd": ["--max-budget-usd", "{usd}"],
        "env": {"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "600000"},
        "usage": "claude", "tested": True,
    },
    "codex": {
        # Codex has no command allowlist: its sandbox is the boundary (writes only in the workspace, no network,
        # no GPU devices), which is why the harness's own state lives outside the workspace. The prompt goes in on
        # stdin. Native subagents are on by default and inherit the sandbox. Reports tokens, not dollars.
        "cmd": ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check", "--ephemeral",
                "--cd", "{ws}", "--json", "-"],
        "stdin": True, "model": ["-m", "{model}"], "env": {"KAR_WAIT_S": "240"}, "usage": "codex", "tested": False,
    },
    "grok": {
        # --trust: without it a headless run ignores AGENTS.md. The GROK_*_ENABLED switches stop it from loading the
        # user's ~/.claude and ~/.cursor agents, hooks, MCP servers, rules and skills; --deny MCPTool keeps any
        # configured MCP server out.
        "cmd": ["grok", "--prompt-file", "{prompt_file}", "--cwd", "{ws}", "--trust", "--no-auto-update",
                "--permission-mode", "dontAsk", "--allow", "Bash({tool})", "--allow", "Bash({tool} *)",
                "--allow", "Edit({writes})", "--allow", "Write({writes})", "--deny", "MCPTool",
                "--disable-web-search", "--output-format", "streaming-messages-json"],
        "model": ["-m", "{model}"], "env": {"GROK_MEMORY": "0", "GROK_DISABLE_AUTOUPDATER": "1", **_GROK_ENV},
        "usage": "claude", "tested": False,
    },
    "cursor": {
        # Not installed on the machine this was written on; flags from Cursor's CLI docs. --force runs commands
        # unless denied, so the boundary is Cursor's sandbox plus the harness's state living outside the workspace.
        "cmd": ["cursor-agent", "-p", "--workspace", "{ws}", "--trust", "--force", "--sandbox", "enabled",
                "--output-format", "json", "{prompt}"],
        "model": ["--model", "{model}"], "usage": "cursor", "tested": False,
    },
}


def _fill(args, values):
    out = []
    for a in args:
        for k, v in values.items():
            a = a.replace("{" + k + "}", str(v))
        out.append(a)
    return out


def command(agent, agent_cmd, prompt, prompt_file, ws, model=None, usd=None, tool="./kar", writes="./**", lane=""):
    """-> (argv, extra env, usage kind, stdin text or None)."""
    vals = {"prompt": prompt, "prompt_file": prompt_file, "ws": ws, "run": ws, "model": model or "", "tool": tool,
            "writes": writes, "lane": lane}
    if agent_cmd:
        return _fill(shlex.split(agent_cmd), vals), {}, None, None
    p = PRESETS[agent]
    cmd = _fill(p["cmd"], vals)
    if model and p.get("model"):
        cmd += _fill(p["model"], vals)
    if usd is not None and p.get("usd"):
        cmd += _fill(p["usd"], {**vals, "usd": f"{max(usd, 0.05):.2f}"})
    return cmd, dict(p.get("env", {})), p.get("usage"), prompt if p.get("stdin") else None


def usage_of(kind, log_path):
    """{usd, tokens, usage, turns, model} from an agent's output, when the CLI reports them."""
    out = {"usd": None, "tokens": None}
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return out
    objs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                objs.append(json.loads(ln))
            except ValueError:
                pass
    if not objs and text.strip().startswith("{"):
        try:
            objs.append(json.loads(text))
        except ValueError:
            pass
    if kind == "claude":            # Claude Code stream-json; Grok's streaming-messages-json has the same result line
        res = next((o for o in reversed(objs) if o.get("type") == "result" or "total_cost_usd" in o), None)
        if res:
            u = res.get("usage") or {}
            mu = res.get("modelUsage") or {}
            tok = sum(int(v) for m in mu.values() if isinstance(m, dict) for k, v in m.items()
                      if k.endswith("Tokens") and isinstance(v, (int, float)))
            if not tok:             # modelUsage includes subagents; usage alone does not
                tok = sum(int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                       "cache_read_input_tokens"))
            out.update(usd=res.get("total_cost_usd"), turns=res.get("num_turns"), usage=u, tokens=tok,
                       model=",".join(sorted(mu.keys())) or None, is_error=res.get("is_error"),
                       result=str(res.get("result", ""))[-500:])
    elif kind == "codex":
        tot = {}
        for o in objs:
            u = o.get("usage") if isinstance(o.get("usage"), dict) else None
            if o.get("type") == "turn.completed" and u:
                for k, v in u.items():
                    if isinstance(v, (int, float)):
                        tot[k] = tot.get(k, 0) + v
        if tot:
            out.update(usage=tot, tokens=int(sum(v for k, v in tot.items() if k != "cached_input_tokens")))
    elif kind == "cursor":
        res = next((o for o in reversed(objs) if o.get("type") == "result"), None)
        u = (res or {}).get("usage") or {}
        if u:
            out.update(usage=u, tokens=int(sum(v for v in u.values() if isinstance(v, (int, float)))))
    return out


def _prompt(run, lane):
    with open(os.path.join(PROMPTS, "session.md")) as f:
        return string.Template(f.read()).safe_substitute(lane=lane["id"], role=lane["role"],
                                                         tool=run.tool(lane["id"]))


def _log(msg):
    print(msg, flush=True)                   # the driver usually writes to a file someone is tailing


def session_budget(left, committed, free, spent, user_cap=None, floor=1.0, default_need=3.0):
    """How the agent budget left is shared among sessions: -> (start now?, this session's cap, a session's need).

    left: budget minus recorded spend; committed: the caps of sessions in flight (their spend is recorded only when
    they end); free: lanes without a session; spent: what each ended session cost.

    A session cut off by its cap before it hands anything back is money wasted (measured: a $15 run split four ways
    gave three builders $2.2 each; all three were cut mid-build). So a session's cap is never below what a session
    needs, which is learned from the run (1.5 x the median cost of its sessions so far; `default_need` before two
    have ended): when the money does not cover every free lane, fewer sessions run at once, each with enough. The
    last of the budget goes to one session if it is most of a session's need, else it is left unspent."""
    done = sorted(x for x in spent if x)
    need = max(floor, 1.5 * done[len(done) // 2]) if len(done) >= 2 else max(floor, default_need)
    avail = left - committed
    if free <= 0 or avail < floor:
        return False, 0.0, need
    n = free if avail / free >= need else int(avail // need)
    if n == 0:
        if committed > 0 or avail < 0.6 * need:      # wait for sessions in flight to return their unspent caps
            return False, 0.0, need
        n = 1
    cap = min(avail / n, 3 * need)
    if user_cap is not None:
        cap = min(cap, user_cap)
    return cap >= floor, cap, need


def drive(root, agents=("claude",), agent_cmd=None, model=None, session_usd=None, session_timeout_s=3600,
          log=_log, poll_s=2.0, min_session_usd=1.0):
    run = Run(root)
    released = run.release_dead_holders()
    if released:
        log(f"resuming: lanes {', '.join(released)} lost their driver; they go to the next free sessions")
    agents = list(agents) or ["claude"]
    names = [("custom:" + shlex.split(agent_cmd)[0]) if agent_cmd else a for a in agents]
    team = run.lanes_n()
    run.ledger.append("interventions", {"by": "user", "what": f"started kar run: a team of {team} with "
                                        f"{', '.join(sorted(set(names)))}" + (f", model {model}" if model else "")})
    stop = threading.Event()
    srv = threading.Thread(target=server.serve, args=(run.root, stop, log), daemon=True)
    srv.start()
    live = {}                                 # slot -> session
    slot_agent = {i: agents[i % len(agents)] for i in range(team)}

    def _term(signum, frame):                # `kill` on kar run must also end the agents (own process groups)
        raise SystemExit(128 + signum)

    def beat():
        with run.lock():
            st = run.state
            st["driver"] = {"pid": os.getpid(), "t": time.time()}
            run.save(st)

    old_term = signal.signal(signal.SIGTERM, _term)
    last_beat = last_judge = 0
    judging = threading.Lock()

    def judge_tick():
        if not judging.acquire(blocking=False):
            return
        try:
            from kar import judge as J
            rec = J.tick(Run(run.root))
            if rec:
                log(f"[{now()[11:19]}] judge check of {rec['cid']}: {rec.get('score', rec.get('status'))}")
        except Exception as e:  # noqa: BLE001 - a judge failure must not stop the team
            log(f"[{now()[11:19]}] judge check failed: {type(e).__name__}: {e}")
        finally:
            judging.release()
    done_since = None
    try:
        while True:
            if time.time() - last_beat > 10:
                beat()
                last_beat = time.time()
            if time.time() - last_judge > 300:
                threading.Thread(target=judge_tick, daemon=True).start()
                last_judge = time.time()
            for slot, s in list(live.items()):          # sessions that ended
                p = s["p"]
                killed = None
                if p.poll() is None and time.time() - s["t0"] > session_timeout_s:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    killed = f"killed after {session_timeout_s} s"
                if p.poll() is None:
                    continue
                s["log_f"].close()
                u = usage_of(s["kind"], s["logp"])
                run.ledger.append("cost", {"kind": "agent", "agent": s["name"], "lane": s["lane"], "session": s["n"],
                                           "s": round(time.time() - s["t0"], 1), "exit": p.returncode,
                                           "killed": killed, "log": os.path.relpath(s["logp"], run.root), **u})
                progressed = run.progress_key(s["lane"]) != s["before"]
                run.end_session(s["lane"], progressed)
                log(f"[{now()[11:19]}] {s['lane']} session {s['n']} ({s['name']}) ended: exit {p.returncode}"
                    + (f", {killed}" if killed else "") + f", {time.time() - s['t0']:.0f} s"
                    + (f", ${u['usd']:.2f}" if u.get("usd") is not None else "")
                    + (f", {u['tokens']:,} tokens" if u.get("tokens") else "")
                    + ("" if progressed else ", no step taken"))
                del live[slot]
            st = run.state
            if st["phase"] == "done":
                if not live:
                    log(f"STOP: {st['done_reason']}")
                    break
                done_since = done_since or time.time()
                if time.time() - done_since > 180:      # they get STOP from their next command; this is the backstop
                    for s in live.values():
                        if s["p"].poll() is None:
                            os.killpg(s["p"].pid, signal.SIGKILL)
                time.sleep(poll_s)
                continue
            for slot in range(team):                    # fill free slots
                if slot in live:
                    continue
                b = run.cfg["budget"]
                usd = session_usd
                if b.get("agent_usd") is not None:       # caps of sessions in flight plus this one fit the budget left
                    left = b["agent_usd"] - run.costs()["agent_usd"]
                    committed = sum(x["usd"] or 0 for x in live.values())
                    spent = [r.get("usd") for r in run.ledger.records("cost") if r.get("kind") == "agent"]
                    go, usd, need = session_budget(left, committed, team - len(live), spent, user_cap=session_usd,
                                                   floor=min_session_usd)
                    if not go:
                        if not live:
                            with run.lock():
                                st = run.state
                                if st["phase"] != "done":
                                    st.update(phase="done", done_reason=(
                                        f"agent budget spent: ${left:.2f} left, less than a session needs "
                                        f"(about ${need:.2f} on this run)"))
                                    run.save(st)
                        break
                agent = slot_agent[slot]
                name = names[slot % len(names)]
                holder = f"kar-run:{os.getpid()}:{slot}"
                # with money for one more session only, a survey or a plan is wasted: nothing could build on it
                # (measured: a $4.66 plan at the end of a run, cut off, with nothing left to build its ideas)
                tail = b.get("agent_usd") is not None and left < 2 * need
                lane, text = run.assign(holder, agent=name, roles=("build", "stack", "audit") if tail
                                        else ("survey", "plan", "build", "stack", "audit"))
                if lane is None:
                    if text.startswith("STOP"):
                        break
                    if not live:
                        with run.lock():
                            st = run.state
                            if st["phase"] != "done":
                                st.update(phase="done", done_reason=(
                                    f"agent budget: ${left:.2f} left covers one more session (about ${need:.2f} "
                                    f"each on this run) and no build, stack or fix is waiting; a new round needs "
                                    f"about ${2 * need:.0f}") if tail else "nothing left to hand out and no lane at work")
                                run.save(st)
                    break
                n = run.costs()["sessions"] + len(live) + 1
                lid = lane["id"]
                pf = run.p("agents", f"s{n:04d}_{lid}_prompt.md")
                prompt = _prompt(run, lane)
                with open(pf, "w") as f:
                    f.write(prompt)
                cmd, env, kind, stdin_text = command(agent, agent_cmd, prompt, pf, run.w(), model, usd,
                                                     tool=run.tool(lid), writes=f"./lanes/{lid}/**", lane=lid)
                logp = run.p("agents", f"s{n:04d}_{lid}.log")
                lf = open(logp, "wb")
                p = subprocess.Popen(cmd, cwd=run.w(), stdout=lf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                                     env={**os.environ, **env}, start_new_session=True)
                if stdin_text:
                    p.stdin.write(stdin_text.encode())
                    p.stdin.close()
                live[slot] = {"p": p, "lane": lid, "n": n, "t0": time.time(), "logp": logp, "log_f": lf,
                              "kind": kind, "name": name, "before": run.progress_key(lid), "usd": usd}
                log(f"[{now()[11:19]}] {lid} session {n} ({name}): {lane['role']}"
                    + (f" {lane['hyp']}" if lane.get("hyp") else "") + (f" ({lane['lens']})" if lane.get("lens")
                                                                       else "")
                    + f"; best {st['best']} {st['best_sol'] * 100:.1f}% of SOL; {len(live)} at work"
                    + (f"; cap ${usd:.2f}" if usd else ""))
            time.sleep(poll_s)
    finally:
        killed_any = False
        for s in live.values():
            if s["p"].poll() is None:
                os.killpg(s["p"].pid, signal.SIGKILL)
                s["p"].wait()
                killed_any = True
            try:
                s["log_f"].close()
            except OSError:
                pass
        if killed_any:
            run.ledger.append("interventions", {"by": "user", "what": "kar run was stopped; agent sessions in "
                                                "flight were killed (their token use is not recorded)"})
        with run.lock():
            st = run.state
            st["driver"] = None
            for lane in st["lanes"].values():
                if lane["status"] == "active" and str(lane.get("holder", "")).startswith(f"kar-run:{os.getpid()}"):
                    lane["holder"] = None
            run.save(st)
        signal.signal(signal.SIGTERM, old_term)
        stop.set()
        srv.join(timeout=10)
    return run


TASK_PROMPT = """You are writing a GPU kernel-optimization task, not a kernel. Read AGENTS.md and DESCRIPTION.md in this
folder, fill in the task files, and run `./kar-task validate` after each change until it prints TASK OK. Record every
choice the request left open under "Assumptions" in DESCRIPTION.md. Then end the session."""


def write_task(root, agent="claude", agent_cmd=None, model=None, usd=None, timeout_s=1800, sessions=2, log=print,
               backend=None):
    """Have an agent turn DESCRIPTION.md into task files; returns (ok, validate text, [session costs])."""
    from kar import newtask
    os.makedirs(os.path.join(root, ".kar-new"), exist_ok=True)
    pf = os.path.join(root, ".kar-new", "prompt.md")
    with open(pf, "w") as f:
        f.write(TASK_PROMPT)
    costs = []
    ok, text = False, ""
    for n in range(1, sessions + 1):
        cmd, env, kind, stdin_text = command(agent, agent_cmd, TASK_PROMPT, pf, root, model, usd, tool="./kar-task",
                                             writes="./**")
        logp = os.path.join(root, ".kar-new", f"s{n:02d}.log")
        log(f"[{now()[11:19]}] task-writing session {n} ({agent_cmd.split()[0] if agent_cmd else agent})")
        t0 = time.time()
        with open(logp, "wb") as lf:
            p = subprocess.Popen(cmd, cwd=root, stdout=lf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                                 env={**os.environ, **env}, start_new_session=True)
            if stdin_text:
                p.stdin.write(stdin_text.encode())
                p.stdin.close()
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        u = usage_of(kind, logp)
        costs.append({"session": n, "s": round(time.time() - t0, 1), **u})
        log(f"[{now()[11:19]}] session {n} ended: {time.time() - t0:.0f} s"
            + (f", ${u['usd']:.2f}" if u.get("usd") is not None else ""))
        ok, text = newtask.validate(root, backend=backend)
        if ok:
            break
    with open(os.path.join(root, ".kar-new", "costs.json"), "w") as f:
        json.dump(costs, f, indent=1)
    return ok, text, costs
