"""kar command line.

  kar init <task>                      validate the task, freeze it, prove the evaluator: reference, baseline and known-
                                       good kernels accepted, every canary rejected, baseline-vs-baseline within noise
  kar eval <task> <cand.py>...         lint, smoke and check candidates
  kar ab   <task> --cand X.py [...]    time candidates against a control (default: the baseline) in one job
  kar calibrate [--force]              measure this GPU (DRAM, L2, FP32, tensor cores, graph overhead); cached
  kar gap  <task> [--cand X.py]        speed-of-light gap table for a candidate (default: the baseline)
  kar profile <task> <cand.py>         one page: check, gap table, nsys per-kernel breakdown, ncu on the shapes
                                       with the most to gain, SASS audit

One prompt to a running team:
  kar go "what to optimize, on which GPU, any rules" [--from FOLDER] [--backend B] [--judge FILE] [--lanes N]
                                       doctor -> an agent writes the task (from FOLDER when given) -> validate ->
                                       you read what "correct" means -> evaluator proof -> the team runs

Tasks from a request:
  kar new <dir> "request" [--from FOLDER] [--agent NAME | --agent-cmd T]
                                       a task folder from plain words: skeleton + instructions (AGENTS.md) for any
                                       agent, or have an agent CLI write it; ./kar-task validate checks it. A person
                                       reviews reference.py (it defines correct) before `kar start`

Runs (a team of agents optimizes the task; any agent that can run shell commands and edit files):
  kar start <task> [--run DIR] [--lanes N]
                                       make a run folder: init, calibrate, profile the baseline. Then open agents in
                                       its workspace (RUN/ws) and tell each one "run ./kar next and follow it", with
                                       `kar serve` running; or use `kar run`
  kar run <task|run> --lanes N --agent NAME[,NAME...]
                                       drive a team of N agent sessions at once, unattended (presets: claude, codex,
                                       grok, cursor; a list alternates them by lane), or
          --agent-cmd 'TEMPLATE'       any CLI through a command template ({prompt}, {prompt_file}, {ws}, {lane},
                                       {tool}, {writes}, {model})
  kar serve <run>                      execute ./kar requests for agents sandboxed away from the GPU
  kar docs list | find "TERM" | get NAME [--section "TERM"]
                                       the GPU documentation agents read (fetched from an allowlist, cached)
  kar judge <run> [status|check CID|submit CID|request KIND CID WHY|approve ID|attach FILE]
                                       an optional external judge: quotas, noise, approvals (scored submits always
                                       need a person)
  kar doctor [--backend B]             is this machine ready (Python, torch, cuda-python, GPUs, nsys/ncu, agent
                                       CLIs, the backend), and what to fix
  kar status <run>                     where a run stands
  kar report <run>                     re-measure the best, check integrity, write REPORT.md

kernel-autoresearch by Gaurav Jain (gauravjain14): https://github.com/gauravjain14/kernel-autoresearch-plugin
"""
import argparse
import datetime
import json
import os
import sys

from kar import checks
from kar.runner import Runner
from kar.task import spec


def _backend(args):
    from kar import backends
    from kar import config
    spec = getattr(args, "backend", None)
    if not spec:
        return config.backend()
    return backends.parse_spec(spec, getattr(args, "backend_opt", None) or ())


def _out(args, task, cmd):
    if args.out:
        return args.out
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(os.getcwd(), "kar-runs", task.name, f"{cmd}-{ts}")


from kar.checks import _ci, _pct, print_ab  # noqa: E402,F401  (kept importable from here)


def cmd_init(args):
    task = spec.load(args.task)
    rep = checks.run_init(task, _out(args, task, "init"), gpu=args.gpu, reps=args.reps, backend=_backend(args))
    return 0 if rep["ok"] else (2 if rep.get("wrong_arch") else 1)


def cmd_eval(args):
    task = spec.load(args.task)
    r = Runner(task, _out(args, task, "eval"), gpu=args.gpu, backend=_backend(args))
    bad = 0
    for p in args.cand:
        rec = r.evaluate(os.path.basename(p), os.path.abspath(p))
        print(f"{os.path.basename(p)}: {rec['verdict']}" + (f" at {rec['stage']}: {rec['reason']}"
                                                            if rec["verdict"] != "accept" else ""))
        bad += rec["verdict"] != "accept"
    return 1 if bad else 0


def cmd_ab(args):
    task = spec.load(args.task)
    r = Runner(task, _out(args, task, "ab"), gpu=args.gpu, seed=args.seed, backend=_backend(args))
    control = ("control", os.path.abspath(args.control)) if args.control else ("baseline", task.path(task.baseline))
    cands = []
    for p in args.cand:
        rec = r.evaluate(os.path.basename(p), os.path.abspath(p))
        if rec["verdict"] != "accept":
            print(f"{os.path.basename(p)}: rejected at {rec['stage']}: {rec['reason']}")
            return 1
        cands.append((os.path.splitext(os.path.basename(p))[0], os.path.abspath(p)))
    out = r.ab(cands, control, reps=args.reps, copies=args.copies)
    print_ab(out)
    return 0 if out.get("status") == "ok" else 1


def cmd_calibrate(args):
    import tempfile
    from kar.task import spec as S
    task = S.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "toy", "rmsnorm"))
    r = Runner(task, args.out or tempfile.mkdtemp(prefix="kar-calibrate-"), gpu=args.gpu, backend=_backend(args))
    hw = r.calibrate(force=args.force)
    print(json.dumps(hw["device"], indent=1))
    print(json.dumps(hw["measured"], indent=1))
    return 0


def cmd_gap(args):
    task = spec.load(args.task)
    r = Runner(task, _out(args, task, "gap"), gpu=args.gpu, backend=_backend(args))
    if args.cand:
        rec = r.evaluate(os.path.basename(args.cand), args.cand)
        if rec["verdict"] != "accept":
            print(f"rejected at {rec['stage']}: {rec['reason']}")
            return 1
    g = r.gap(args.cand, reps=args.reps)
    print(g["markdown"])
    with open(os.path.join(r.out, "gap.md"), "w") as f:
        f.write(g["markdown"])
    return 0


def cmd_profile(args):
    task = spec.load(args.task)
    r = Runner(task, _out(args, task, "profile"), gpu=args.gpu, backend=_backend(args))
    res = checks.profile_page(r, task, os.path.abspath(args.cand), reps=args.reps, ncu_n=args.ncu)
    if res.get("rejected"):
        print(res["rejected"])
        return 1
    with open(os.path.join(r.out, "profile.md"), "w") as f:
        f.write(res["page"])
    print(res["page"])
    print(f"\nwritten: {os.path.join(r.out, 'profile.md')}")
    return 0


def _budget(args):
    return {k: v for k, v in {"gpu_h": args.gpu_h, "wall_h": args.wall_h, "submits": args.submits,
                              "sessions": args.sessions, "agent_usd": args.agent_usd,
                              "agent_mtok": args.agent_mtok}.items() if v is not None}


def _new_run(args):
    from kar.run.session import Run
    task = spec.load(args.task)
    root = args.run or os.path.join(os.getcwd(), "kar-runs", task.name,
                                    "run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return Run.create(args.task, root, target_sol=args.target_sol, budget=_budget(args), gpu=args.gpu,
                      lanes=args.lanes, survey=args.survey, audit=args.audit, backend=_backend(args))


def cmd_new(args):
    from kar import newtask
    root = newtask.create(args.dir, args.request, source=args.source)
    if not args.agent and not args.agent_cmd:
        print(f"task skeleton in {root}\nOpen it in any agent and say: \"read AGENTS.md and write this task\". "
              f"Check it with {os.path.join(root, 'kar-task')} validate.")
        return 0
    from kar.agents import drive
    ok, text, costs = drive.write_task(root, agent=args.agent or "claude", agent_cmd=args.agent_cmd, model=args.model,
                                       usd=args.session_usd)
    print(text)
    spent = sum(c.get("usd") or 0 for c in costs)
    print(f"\n{len(costs)} session(s), ${spent:.2f} as reported by the agent CLI")
    if ok:
        print(f"\nReview {os.path.join(root, 'reference.py')} and the Assumptions in DESCRIPTION.md, then:\n"
              f"  python3 -m kar init {root}\n  python3 -m kar run {root} --agent {args.agent or 'claude'}")
    return 0 if ok else 1


def _show_task(root):
    """What a person should look at before a team starts: what correct means, and every choice the writer made."""
    print("\n" + "=" * 100 + "\nThe task defines what correct means. Read it before the team starts:\n")
    for name, lines in (("DESCRIPTION.md", 60), ("reference.py", 80), ("shapes.py", 40), ("task.toml", 60)):
        p = os.path.join(root, name)
        if os.path.exists(p):
            with open(p) as f:
                body = f.read().splitlines()
            print(f"---- {name}" + (f" (first {lines} of {len(body)} lines)" if len(body) > lines else ""))
            print("\n".join(body[:lines]) + "\n")
    print("=" * 100)


def _detach(args, argv, log_dir):
    """Run this same command in the background, detached from the terminal or agent session that started it, with
    its output in a log file. -> exit code for the foreground command."""
    import subprocess
    import time
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"kar-{args.cmd}-{datetime.datetime.now():%Y%m%d-%H%M%S}.log")
    child = [a for a in argv if a != "--detach"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with open(log, "ab") as f:
        p = subprocess.Popen([sys.executable, "-m", "kar", *child], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env, cwd=os.getcwd())
    time.sleep(3)
    if p.poll() is not None:
        print(f"it stopped at once (exit {p.returncode}); its output:\n")
        with open(log) as f:
            print(f.read()[-4000:])
        return 1
    print(f"started in the background (pid {p.pid}); it keeps running after this session ends.\n"
          f"log: {log}\n"
          + ("" if args.cmd == "serve" else
             "The first minutes prove the evaluator on the GPU; the log then names the run folder "
             "(`run folder ...`).\nfollow it: kar status RUN      stop it: kar stop RUN      "
             "final result: kar report RUN"))
    return 0


def cmd_validate(args):
    from kar import newtask
    ok, text = newtask.validate(args.task, backend=_backend(args))
    print(text)
    return 0 if ok else 1


def cmd_stop(args):
    """Stop the team driving a run (the `kar go` or `kar run` process); its agent sessions stop with it."""
    import signal
    from kar.run.session import Run
    run = Run(args.run)
    d = (run.state or {}).get("driver") or {}
    pid = d.get("pid")
    if not pid:
        print("no team is driving this run")
        return 0
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        print(f"the team's process (pid {pid}) is no longer running")
        return 0
    print(f"stopped the team (pid {pid}). Continue later with: kar run {run.root}")
    return 0


def cmd_go(args):
    """One prompt to a running team: check the machine, write and validate the task, show it, start the team."""
    from kar import doctor, newtask
    from kar.agents import drive
    from kar.task import spec as S
    be = _backend(args)
    rows = doctor.checks(be)
    bad = [r for r in rows if r[0] == "FAIL"]
    if bad:
        for r in bad:
            print(f"FAIL {r[1]}: {r[2]}\n     fix: {r[3]}")
        return 1
    agents = [a.strip() for a in (args.agent or "claude").split(",") if a.strip()]
    if getattr(args, "detach", False):
        if not args.task or not args.yes:
            print("--detach starts the team in the background, so it needs a finished task and your yes: "
                  "kar go --task TASK_DIR --yes --detach")
            return 2
        S.load(os.path.abspath(args.task))
        return _detach(args, args.argv, os.path.join(os.getcwd(), "kar-runs", "logs"))
    if args.task:
        root = os.path.abspath(args.task)
        S.load(root)
    else:
        if not args.prompt:
            print("kar go \"what to optimize, on which GPU, any rules\" [--from FOLDER]   (or --task EXISTING_TASK)")
            return 2
        slug = "".join(c if c.isalnum() else "_" for c in args.prompt.lower())[:40].strip("_") or "task"
        root = os.path.abspath(args.task_dir or os.path.join(os.getcwd(), "kar-tasks",
                                                               f"{slug}-{datetime.datetime.now():%Y%m%d-%H%M%S}"))
        newtask.create(root, args.prompt, source=args.source)
        print(f"writing the task in {root} with {agents[0] if not args.agent_cmd else args.agent_cmd.split()[0]}")
        ok, text, costs = drive.write_task(root, agent=agents[0], agent_cmd=args.agent_cmd, model=args.model,
                                           usd=args.session_usd or 3.0, backend=be)
        print(text)
        if not ok:
            print(f"\nThe task is not ready. Fix it by hand or with any agent (`{root}/kar-task validate`), then: "
                  f"kar go --task {root}")
            return 1
    _show_task(root)
    if args.agent_usd is None:
        args.agent_usd = 25.0
        print("agent budget: $25 at list prices (pass --agent-usd to change it)")
    if not args.yes:
        ans = input(f"Start a team of {args.lanes or 4} on this task ({(be or {}).get('kind', 'local')} GPU, agents "
                    f"{', '.join(agents)}, ${args.agent_usd:.0f})? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print(f"Not started. When ready: kar go --task {root} --yes")
            return 0
    args.lanes = args.lanes or 4
    args.task = root
    run = _new_run(args)
    if args.judge:
        from kar import judge as J
        J.save_config(run, args.judge)
    print(f"\nrun folder {run.root}\nwatch it: kar status {run.root}   (agents' board: cd {run.w()} && ./kar board)\n")
    drive.drive(run.root, agents=agents, agent_cmd=args.agent_cmd, model=args.model, session_usd=args.session_usd,
                session_timeout_s=args.session_timeout)
    from kar.run.session import Run
    print(Run(run.root).status()[0])
    print(f"\nreport: kar report {run.root}")
    return 0


def cmd_start(args):
    run = _new_run(args)
    if getattr(args, "judge", None):
        from kar import judge as J
        J.save_config(run, args.judge)
    print(f"\nready: {run.root}\nEither drive a team: python3 -m kar run {run.root} --lanes 4 --agent claude\n"
          f"or by hand: python3 -m kar serve {run.root}, then open any agents in {run.w()} and tell each one: "
          f"\"run ./kar next and follow it\" (AGENTS.md has the rules)")
    return 0


def cmd_run(args):
    from kar.agents import drive
    from kar.run.session import Run
    if getattr(args, "detach", False):
        return _detach(args, args.argv, os.path.join(os.getcwd(), "kar-runs", "logs"))
    if os.path.exists(os.path.join(args.target, "run.json")):
        from kar.run.store import write_json
        run = Run(args.target)
        changed = []
        if _budget(args):
            run.cfg["budget"].update(_budget(args))
            changed.append(f"budget {run.cfg['budget']}")
        if args.target_sol is not None:
            run.cfg["target_sol"] = args.target_sol
            changed.append(f"target {args.target_sol}")
        if getattr(args, "backend", None):
            run.cfg["backend"] = _backend(args)
            changed.append(f"backend {run.cfg['backend']}")
        for k in ("lanes", "survey", "audit"):
            if getattr(args, k) is not None:
                run.cfg[k] = getattr(args, k)
                changed.append(f"{k} {getattr(args, k)}")
        if changed:
            write_json(run.p("run.json"), run.cfg)
            run.ledger.append("interventions", {"by": "user", "what": "changed " + ", ".join(changed)})
            with run.lock():
                st = run.state
                if st["phase"] == "done" and not run.over_budget() and not run.met(st["best_sol"]):
                    st.update(phase="running", cycle=st["cycle"] + 1, planned=False, done_reason=None, plateaus=0,
                              since_keep=0, target_met=None)
                    again = []
                    for h in st["hyps"]:     # a hazard still in the best comes first when a run continues
                        if (h.get("kind") == "fix" and h["status"] in ("abandoned", "dead", "unfixed")
                                and h.get("parent") == st["best"]):
                            h.update(status="open", lane=None, attempts=0)
                            again.append(h["id"])
                            run.ledger.append("hypotheses", {"id": h["id"], "status": "open", "why": "the run "
                                                             "continues and the hazard is still in the best"})
                    run.save(st)
                    run.ledger.append("interventions", {"by": "user", "what": "reopened the run" + (
                        f"; fixes queued again: {', '.join(again)}" if again else "")})
    else:
        args.task = args.target
        run = _new_run(args)
    if getattr(args, "judge", None):
        from kar import judge as J
        J.save_config(run, args.judge)
        run = Run(run.root)
    agents = [a.strip() for a in args.agent.split(",") if a.strip()]
    bad = [a for a in agents if a not in drive.PRESETS]
    if not args.agent_cmd and bad:
        print(f"unknown agent {', '.join(bad)}; presets: {', '.join(drive.PRESETS)}, or use --agent-cmd")
        return 2
    drive.drive(run.root, agents=agents, agent_cmd=args.agent_cmd, model=args.model,
                session_usd=args.session_usd, session_timeout_s=args.session_timeout)
    print(Run(run.root).status()[0])
    print(f"\nreport: python3 -m kar report {run.root}")
    return 0


def cmd_serve(args):
    from kar.run.server import serve
    from kar.run.session import find_root
    root = find_root(args.run)
    if getattr(args, "detach", False):
        return _detach(args, args.argv, os.path.join(root, "logs"))
    print(f"serving {root}; Ctrl-C to stop")
    import threading
    import time

    def judge_loop():                       # a by-hand team gets the judge's automatic checks too
        from kar import judge as J
        from kar.run.session import Run
        while True:
            time.sleep(300)
            try:
                rec = J.tick(Run(root))
                if rec:
                    print(f"judge check of {rec['cid']}: {rec.get('score', rec.get('status'))}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"judge check failed: {e}", flush=True)

    threading.Thread(target=judge_loop, daemon=True).start()
    try:
        serve(root, log=print)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_docs(args):
    from kar import docs
    text, code = docs.main(args.rest)
    print(text)
    return code


def cmd_judge(args):
    from kar import judge as J
    from kar.run.session import Run, find_root
    run = Run(find_root(args.run))
    a = args.rest or ["status"]
    try:
        if a[0] == "status":
            print(J.status_text(run))
        elif a[0] == "attach" and len(a) == 2:
            J.save_config(run, a[1])
            print(J.status_text(Run(run.root)))
        elif a[0] in ("check", "submit") and len(a) >= 2:
            cid = {"best": run.state["best"]}.get(a[1], a[1])
            rec = J.run_judge(run, a[0], cid, by="user", approved_by="user (command line)")
            print({k: rec.get(k) for k in ("kind", "cid", "score", "status", "dir")})
        elif a[0] == "request" and len(a) >= 4:
            print(J.request(run, a[1], {"best": run.state["best"]}.get(a[2], a[2]), "user", " ".join(a[3:])))
        elif a[0] == "approve" and len(a) == 2:
            rec = J.approve(run, a[1])
            print({k: rec.get(k) for k in ("kind", "cid", "score", "status", "dir")})
        else:
            print("kar judge RUN [status | check CID | submit CID | request check|submit CID WHY | approve ID | "
                  "attach FILE]")
            return 2
    except J.JudgeError as e:
        print(e)
        return 1
    return 0


def cmd_doctor(args):
    from kar import doctor
    return doctor.main(_backend(args) if args.backend else None)


def cmd_status(args):
    from kar.run.session import Run, find_root
    run = Run(find_root(args.run))
    print(run.status()[0])
    if getattr(args, "board", False):
        b = run.board()
        print("\n" + (b[0] if isinstance(b, tuple) else b))
    return 0


def cmd_report(args):
    from kar.run.session import Run, find_root
    run = Run(find_root(args.run))
    md, _ = run.report(reps=args.reps)
    print(md)
    print(f"written: {run.p('REPORT.md')}")
    return 0


def _run_opts(p):
    p.add_argument("--run", help="run folder (default kar-runs/<task>/run-<time>)")
    p.add_argument("--target-sol", type=float, help="stop at this fraction of speed of light (default: no target; the "
                   "team closes the gap to the roofline until the budget or a plateau ends the run)")
    p.add_argument("--gpu-h", type=float)
    p.add_argument("--wall-h", type=float)
    p.add_argument("--submits", type=int)
    p.add_argument("--sessions", type=int)
    p.add_argument("--agent-usd", type=float)
    p.add_argument("--agent-mtok", type=float, help="agent token budget in millions (for CLIs that report no cost)")
    p.add_argument("--gpu", type=int, default=0)
    _backend_opts(p)
    p.add_argument("--judge", help="a TOML file with a [judge] table: the external judge's check/submit commands")
    p.add_argument("--lanes", type=int, help="agent sessions at once (default 1)")
    p.add_argument("--survey", type=int, help="survey lenses per round (default: 0 for one lane, else up to 4)")
    p.add_argument("--audit", choices=("every", "milestones", "off"), help="which new bests auditors review")


def _backend_opts(p):
    p.add_argument("--backend", help="where GPU jobs run: local[:GPU] (default), modal:GPU[:MAX_BOXES] (e.g. "
                   "modal:H200:3), ssh:HOST[:GPU,GPU]")
    p.add_argument("--backend-opt", action="append", metavar="KEY=VALUE",
                   help="backend option, repeatable: image=REGISTRY_IMAGE, pip=a,b, apt=a,b, timeout_s=N, "
                        "python=/path, env.NAME=VALUE")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("task")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int)
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("eval")
    p.add_argument("task")
    p.add_argument("cand", nargs="+")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("ab")
    p.add_argument("task")
    p.add_argument("--cand", action="append", required=True)
    p.add_argument("--control")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int)
    p.add_argument("--copies", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("calibrate")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("gap")
    p.add_argument("task")
    p.add_argument("--cand")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int)
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("profile")
    p.add_argument("task")
    p.add_argument("cand")
    p.add_argument("--ncu", type=int, default=2, help="ncu on this many shapes, those with the most to gain")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int)
    p.add_argument("--out")
    _backend_opts(p)
    p = sub.add_parser("new")
    p.add_argument("dir")
    p.add_argument("request")
    p.add_argument("--from", dest="source", help="a project, challenge or benchmark folder the task is about (copied "
                   "into the task as source/ for the task writer to mirror)")
    p.add_argument("--agent")
    p.add_argument("--agent-cmd")
    p.add_argument("--model")
    p.add_argument("--session-usd", type=float, default=3.0)
    p = sub.add_parser("go", help="one prompt to a running team")
    p.add_argument("prompt", nargs="?", help="what to optimize, for which GPU, under which rules")
    p.add_argument("--from", dest="source", help="the project, challenge or benchmark folder the prompt is about")
    p.add_argument("--task", help="use this existing task folder instead of writing one")
    p.add_argument("--task-dir", help="where to write the task (default kar-tasks/<prompt>-<time>)")
    p.add_argument("--yes", action="store_true", help="start without asking (the task is still printed)")
    p.add_argument("--detach", action="store_true", help="start the team in the background and return (needs --task "
                   "and --yes); output goes to kar-runs/logs/")
    _run_opts(p)
    p.add_argument("--agent", help="agent preset(s), e.g. claude or claude,codex (default claude)")
    p.add_argument("--agent-cmd")
    p.add_argument("--model")
    p.add_argument("--session-usd", type=float)
    p.add_argument("--session-timeout", type=int, default=3600)
    p = sub.add_parser("start")
    p.add_argument("task")
    _run_opts(p)
    p = sub.add_parser("validate", help="check a task folder on the GPU (reference, baseline, shapes, op model)")
    p.add_argument("task")
    _backend_opts(p)
    p = sub.add_parser("stop", help="stop the team driving a run")
    p.add_argument("run")
    p = sub.add_parser("run")
    p.add_argument("target", help="a task folder (starts a new run) or a run folder (continues it)")
    p.add_argument("--detach", action="store_true", help="drive the team in the background and return")
    _run_opts(p)
    p.add_argument("--agent", default="claude")
    p.add_argument("--agent-cmd")
    p.add_argument("--model")
    p.add_argument("--session-usd", type=float)
    p.add_argument("--session-timeout", type=int, default=3600)
    p = sub.add_parser("doctor")
    _backend_opts(p)
    p = sub.add_parser("judge")
    p.add_argument("run")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("docs")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("serve")
    p.add_argument("run", nargs="?")
    p.add_argument("--detach", action="store_true", help="serve in the background and return")
    p = sub.add_parser("status")
    p.add_argument("run", nargs="?")
    p.add_argument("--board", action="store_true", help="also print the team's board")
    p = sub.add_parser("report")
    p.add_argument("run", nargs="?")
    p.add_argument("--reps", type=int)
    args = ap.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    from kar.run.session import RunError
    try:
        return _dispatch(args)
    except RunError as e:                    # a stated reason, not a crash: no traceback
        print(f"kar: {e}", file=sys.stderr)
        return 1


def _dispatch(args):
    return {"init": cmd_init, "eval": cmd_eval, "ab": cmd_ab, "calibrate": cmd_calibrate, "gap": cmd_gap,
            "profile": cmd_profile, "new": cmd_new, "start": cmd_start, "run": cmd_run, "serve": cmd_serve, "status": cmd_status,
            "report": cmd_report, "docs": cmd_docs, "judge": cmd_judge,
            "doctor": cmd_doctor, "go": cmd_go, "validate": cmd_validate, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
