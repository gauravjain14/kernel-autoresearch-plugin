"""A run: one task, one folder, a team of agents working at once.

Agents work in lanes. A lane is one piece of work with one role:

    survey  read the code and the profile through one lens (memory, instructions, latency, structure), measure,
            and hand back findings with numbers
    plan    turn the findings, the dead ends and the board into hypotheses
    build   implement one hypothesis and submit it (or abandon it with the numbers)
    stack   port wins that were judged against an older best onto the current best
    audit   read the new best for hazards the tests cannot reach, run the stress suite, and say ok or hazard

The scheduler (`assign`) hands out lanes: work in flight that lost its agent first, then stacking, auditing, the
survey and the plan for the next round, then the hypotheses in order of predicted gain. Many lanes run at once; GPU
jobs queue for the GPU, one at a time, so every timing stays clean. Agents see each other on `./kar board`, read
each other's lane notes, and message each other with `./kar post`.

Two folders. The run folder is the harness's own: run.json, state.json, the ledger, the frozen candidates, the GPU
jobs. Agents work in `ws/` inside it (the workspace: the task, published evidence, read-only copies of candidates,
and one folder per lane), so an agent's sandbox can be "write only here" whatever agent it is, and nothing an agent
writes can change what the judge uses.

Nothing important lives in an agent's context: state.json, the ledger and the evidence carry the run.
"""
import difflib
import hashlib
import math
import os
import shutil
import sys
import string
import time

from kar import checks
from kar.run import records
from kar.run.store import Ledger, locked, now, read_json, write_json
from kar.runner import Runner
from kar.task import spec

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(os.path.dirname(HERE))
PROMPTS = os.path.join(os.path.dirname(HERE), "agents", "prompts")

DEFAULTS = {
    "target_sol": None,            # stop when the fraction of speed of light reaches this; None: no target, the
                                   # team keeps closing the gap to the roofline until the budget or a plateau ends it
    "budget": {"gpu_h": 2.0, "wall_h": 12.0, "submits": 40, "sessions": None, "agent_usd": None, "agent_mtok": None},
    "lanes": 1,                    # agent sessions at once (the team size)
    "survey": None,                # survey lenses per round; None: none for one lane, else 2 (structure, memory)
    "audit": "milestones",         # auditor lanes read and stress-test: "every" new best, "milestones" (and the
                                   # best that meets the target), or "off"
    "try_reps": 16,                # reps of a quick `try` A/B (submit uses the task's bench reps)
    "tries": {"build": 12, "stack": 8, "survey": 10, "audit": 6, "plan": 0},
    "submit_failures": 3,          # failed gates before a hypothesis is closed
    "plateau": None,               # hypotheses in a row without a win; None: max(5, 2 x lanes)
    "milestone_pct": 5.0,          # cumulative score gain that makes a milestone
    "ncu_shapes": 2,
    "gpu": 0,
    "backend": None,               # where GPU jobs run: None = this machine's GPU `gpu`; or {"kind": "modal", ...}
    "lease_min": 30,               # a lane with no command for this long may be taken over by another agent
    "lane_sessions": 3,            # agent sessions one lane may take before it is closed as stalled
}

LENSES = {
    "structure": "the algorithm and the decomposition of the work: is this the right design at all? What would a "
                 "rewrite, a fusion, a different split of the work over blocks and warps, or a different pass "
                 "structure save, by its bound? Time the bound with `bound` probes.",
    "memory": "data movement: bytes moved against the compulsory bytes on every scored shape, access width and "
              "coalescing, bytes in flight per SM, L2 hit rate and reuse, how the output is written. Use the ncu "
              "memory sections.",
    "instructions": "the instruction stream: SASS instruction count and mix per element, address and integer "
                    "overhead, register use and spills, occupancy, warp stall reasons, fences and barriers. Use `sass` "
                    "and `ncu`.",
    "latency": "the small and launch-bound shapes: grid size against the SM count, waves and tail, the fixed cost of "
               "a call (launches, graph replay), serial phases, anything per call that is not the work itself.",
}
ROLES = ("survey", "plan", "build", "stack", "audit")
OPEN_LIKE = ("open", "building")


class RunError(Exception):
    pass


class BudgetSpent(Exception):
    pass


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _template(name, **kw):
    with open(os.path.join(PROMPTS, name)) as f:
        return string.Template(f.read()).safe_substitute(**kw)


def sol_of(rows, score_fn=None):
    """Fraction of speed of light: 1.0 means every shape at its floor (default: the weighted geometric mean of
    floor / time; with a task score, score(times) / score(floors)).

    A shape measured below its floor counts as exactly at it. The floor's write term depends on how much of the
    output L2 holds past the end of the call, which varies with the access pattern (calibration measured ~17.6 MB
    on the 4060 Ti; an agent's RMSNorm implied ~19.5 MB), so a floor can be a little high. Capping keeps that error
    from ever inflating the result."""
    from kar.sol import gap as G
    return G.sol(rows, score_fn)


def find_root(start=None):
    """The run folder from anywhere inside it (the run folder itself, its workspace, or a lane folder)."""
    d = os.path.abspath(start or os.environ.get("KAR_RUN") or os.getcwd())
    while True:
        if os.path.exists(os.path.join(d, "run.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RunError("not inside a kar run folder (no run.json here or above)")
        d = parent


from kar.run.roles import Roles  # noqa: E402  (the role commands need the helpers above)


class Run(Roles):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.cfg = read_json(self.p("run.json"))
        if self.cfg is None:
            raise RunError(f"{self.root} is not a run folder")
        if self.cfg.get("layout") != 2:
            raise RunError(f"{self.root} was made by an older kar (one agent at a time); start a new run")
        self.task = spec.load(self.cfg["task_root"])
        self.ledger = Ledger(self.p("ledger"))
        self.internal = False

    def p(self, *a):
        """A path in the harness's own folder."""
        return os.path.join(self.root, *a)

    def w(self, *a):
        """A path in the agents' workspace."""
        return os.path.join(self.root, "ws", *a)

    # ---- state ---------------------------------------------------------------------------------------------

    @property
    def state(self):
        return read_json(self.p("state.json"))

    def save(self, st):
        st["updated"] = now()
        write_json(self.p("state.json"), st)

    def lock(self):
        return locked(self.p(".lock"))

    def score_fn(self):
        return self.task.attr(self.task.score) if self.task.score else None

    def target(self):
        return self.cfg.get("target_sol")

    def target_text(self):
        t = self.target()
        return f"target {t * 100:.0f}%" if t is not None else "no target: closing the gap to speed of light"

    def met(self, sol):
        return self.target() is not None and sol >= self.target()

    def aim(self, st):
        """What a round of hypotheses should reach: the target, or with none, half the remaining gap to the floor."""
        t = self.target()
        return t if t is not None else st["best_sol"] + (1 - st["best_sol"]) / 2

    def lanes_n(self):
        return max(1, int(self.cfg.get("lanes") or 1))

    def lenses(self):
        n = self.cfg.get("survey")
        if n is None:
            n = 0 if self.lanes_n() == 1 else 2         # a 4-lens survey cost $13.7 of a $25 run; 2 found the win
        return list(LENSES)[:max(0, min(int(n), len(LENSES)))]

    def low_budget(self):
        """Less than a third of the agent budget left: no new surveys; the money goes to building, stacking, fixing."""
        b = self.cfg["budget"]
        if b.get("agent_usd"):
            return self.costs()["agent_usd"] > b["agent_usd"] * (2 / 3)
        if b.get("agent_mtok"):
            return self.costs()["tokens"] > b["agent_mtok"] * 1e6 * (2 / 3)
        return False

    def plateau_n(self):
        return self.cfg.get("plateau") or max(5, 2 * self.lanes_n())

    def max_hyps(self):
        return max(5, 2 * self.lanes_n())

    # ---- cost and budget -----------------------------------------------------------------------------------

    def costs(self):
        gpu = agent_usd = 0.0
        tokens = sessions = 0
        for c in self.ledger.records("cost"):
            if c["kind"] == "gpu":
                gpu += c.get("s", 0)
            elif c["kind"] == "agent":
                agent_usd += c.get("usd") or 0
                tokens += c.get("tokens") or 0
                sessions += 1
        st = self.state or {}
        wall = time.time() - st.get("started_ts", time.time())
        return {"gpu_s": gpu, "agent_usd": agent_usd, "tokens": tokens, "sessions": sessions, "wall_s": wall,
                "submits": st.get("submits", 0)}

    def session_cap(self):
        b = self.cfg["budget"]
        return b["sessions"] if b.get("sessions") is not None else max(60, 25 * self.lanes_n())

    def over_budget(self):
        b, c = self.cfg["budget"], self.costs()
        if b.get("gpu_h") is not None and c["gpu_s"] >= b["gpu_h"] * 3600:
            return f"GPU budget spent ({c['gpu_s'] / 3600:.2f} of {b['gpu_h']} GPU-hours)"
        if b.get("wall_h") is not None and c["wall_s"] >= b["wall_h"] * 3600:
            return f"wall-clock budget spent ({c['wall_s'] / 3600:.1f} of {b['wall_h']} h)"
        if b.get("submits") is not None and c["submits"] >= b["submits"]:
            return f"submit budget spent ({c['submits']} of {b['submits']})"
        if b.get("agent_usd") is not None and c["agent_usd"] >= b["agent_usd"]:
            return f"agent budget spent (${c['agent_usd']:.2f} of ${b['agent_usd']:.2f})"
        if b.get("agent_mtok") is not None and c["tokens"] >= b["agent_mtok"] * 1e6:
            return f"agent token budget spent ({c['tokens'] / 1e6:.1f} of {b['agent_mtok']} M tokens)"
        if c["sessions"] >= self.session_cap():
            return f"session budget spent ({c['sessions']} of {self.session_cap()} agent sessions)"
        return None

    def _guard(self, mode, label):
        if self.internal:
            return
        why = self.over_budget()
        if why:
            raise BudgetSpent(why)

    def _on_job(self, mode, label, res):
        self.ledger.append("cost", {"kind": "gpu", "mode": mode, "label": label, "s": res.get("wall_s", 0),
                                    "status": res.get("status"), "job": os.path.relpath(res.get("job_dir", ""),
                                                                                        self.root)})

    def runner(self, tag):
        out = self.p("gpu", f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{time.time_ns() % 10**6:06d}_{tag}")
        return Runner(self.task, out, gpu=self.cfg["gpu"], freeze=self.cfg["freeze"], guard=self._guard,
                      on_job=self._on_job, backend=self.cfg.get("backend"))

    # ---- creation ------------------------------------------------------------------------------------------

    @classmethod
    def create(cls, task_root, root, log=print, **opts):
        root = os.path.abspath(root)
        if os.path.exists(root) and os.listdir(root):
            raise RunError(f"{root} exists and is not empty")
        task = spec.load(task_root)
        cfg = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None and k not in ("budget", "tries")}}
        cfg["budget"] = {**DEFAULTS["budget"], **(opts.get("budget") or {})}
        cfg["tries"] = {**DEFAULTS["tries"], **(opts.get("tries") or {})}
        cfg["layout"] = 2
        os.makedirs(root)
        for d in ("ledger", "evidence", "candidates", "milestones", "gpu", "agents"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        for d in ("task", "help", "evidence", "candidates", "lanes", "queue", "notes", "survey", "tests"):
            os.makedirs(os.path.join(root, "ws", d), exist_ok=True)
        log(f"run folder {root}\n\n== init: proving the evaluator on {task.name}")
        rep = checks.run_init(task, os.path.join(root, "init"), gpu=cfg["gpu"], log=log, backend=cfg.get("backend"))
        if not rep["ok"]:
            raise RunError("init failed: " + "; ".join(rep["problems"]))
        cfg.update(task_root=task.root, task_name=task.name, freeze=rep["freeze"], gpu_name=rep["gpu"],
                   arch=rep["arch"], harness=HARNESS, created=now())
        write_json(os.path.join(root, "run.json"), cfg)
        run = cls(root)
        run._lay_out()
        st = {"v": 2, "phase": "running", "cycle": 1, "planned": False, "best": None, "best_sol": None,
              "best_score": None, "best_history": [], "hyps": [], "lanes": {}, "inbox": {}, "next_lane": 1,
              "next_cand": 0, "next_hyp": 1, "since_keep": 0, "plateaus": 0, "plateau_note": "", "submits": 0,
              "milestones": 0, "milestone_base": None, "survey": None, "stack": [], "audited": [], "to_audit": [],
              "plan_failures": {}, "target_met": None, "done_reason": None, "started": now(),
              "started_ts": time.time(), "driver": None}
        run.save(st)
        run.ledger.append("interventions", {"by": "user", "what": "created the run",
                                            "opts": {k: cfg.get(k) for k in ("target_sol", "budget", "gpu", "lanes",
                                                                         "survey", "audit")}})
        log("\n== calibrating the GPU")
        run.runner("calibrate").calibrate()
        log("\n== the baseline is the first best; profiling it")
        with run.lock():
            st = run.state
            cid = run._freeze_candidate(st, task.path(task.baseline), hyp=None, report=None, kind="baseline")
            st["best"] = cid
            run.internal = True
            prof = run._profile_best(cid)
            run.internal = False
            st.update(best_sol=prof["sol"], best_score=prof["score"], milestone_base=cid, baseline=cid,
                      best_history=[{"id": cid, "sol": prof["sol"], "score": prof["score"], "t": now()}])
            st["audited"].append(cid)            # the task's own baseline is not the agents' to audit
            run.save(st)
        log(f"\nbaseline at {prof['sol'] * 100:.1f}% of speed of light; {run.target_text()}")
        if run.met(prof["sol"]):
            log("the baseline already meets the target: there is nothing for an agent to do at this target "
                "(raise --target-sol, or choose a slower baseline)")
        return run

    def _shim(self, path, lane=None):
        """`./kar` for the workspace (lane None) or for one lane. Relocatable: it finds the workspace from where it
        lives."""
        py = sys.executable or shutil.which("python3") or "python3"
        up = "/../.." if lane else ""
        boot = ("import sys; sys.path.insert(0, sys.argv[1]); from kar.run.client import main; "
                "sys.exit(main(sys.argv[2], sys.argv[4:], lane=sys.argv[3] or None))")
        what = f"lane {lane}'s command" if lane else "the agents' entry to the harness"
        with open(path, "w") as f:
            f.write(f"#!/bin/sh\n# {what}. Talks to the harness process if one is serving this run, otherwise runs the "
                    "command directly.\n"
                    f"exec {py} -E -P -c '{boot}' '{HARNESS}' \"$(cd \"$(dirname \"$0\"){up}\" && pwd)\" "
                    f"'{lane or ''}' \"$@\"\n")
        os.chmod(path, 0o755)

    def _mcp_config(self):
        """The lane commands as MCP tools for an IDE or MCP-client agent opened in the workspace: a launcher and the
        project MCP files Claude Code (.mcp.json), Grok (.mcp.json) and Cursor (.cursor/mcp.json) read."""
        import json
        py = sys.executable or shutil.which("python3") or "python3"
        launcher = self.w("kar-mcp")
        boot = "import sys; sys.path.insert(0, sys.argv[1]); from kar.mcp import main; sys.exit(main(sys.argv[2:]))"
        with open(launcher, "w") as f:
            f.write("#!/bin/sh\n# The lane commands as MCP tools (stdio). kar_next hands out a lane.\n"
                    f"exec {py} -E -P -c '{boot}' '{HARNESS}' --ws \"$(cd \"$(dirname \"$0\")\" && pwd)\" \"$@\"\n")
        os.chmod(launcher, 0o755)
        cfg = {"mcpServers": {"kar": {"command": launcher, "args": []}}}
        for rel in (".mcp.json", os.path.join(".cursor", "mcp.json")):
            os.makedirs(os.path.dirname(self.w(rel)), exist_ok=True)
            with open(self.w(rel), "w") as f:
                json.dump(cfg, f, indent=1)

    def _lay_out(self):
        """The workspace: the task (minus the known-good kernels and canaries), the instructions, the client."""
        t = self.task
        hidden = {os.path.normpath(g) for g in t.good} | {os.path.normpath(c) for c in t.canaries}
        for rel in t.files():
            n = os.path.normpath(rel)
            if n in hidden or n.split(os.sep)[0] == "canaries":
                continue
            dst = self.w("task", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(t.path(rel), dst)
            os.chmod(dst, 0o444)
        toml = self.w("task", "task.toml")       # the agents' copy does not name the hidden files
        if os.path.exists(toml):
            with open(toml) as f:
                kept = [ln for ln in f if not ln.split("=")[0].strip() in ("good", "canaries")]
            os.chmod(toml, 0o644)
            with open(toml, "w") as f:
                f.writelines(kept)
            os.chmod(toml, 0o444)
        with open(self.w("task", "README.md"), "w") as f:
            f.write(records.task_readme(t))
        shutil.copy(os.path.join(PROMPTS, "AGENTS.md"), self.w("AGENTS.md"))
        shutil.copy(os.path.join(PROMPTS, "method.md"), self.w("METHOD.md"))
        shutil.copy(os.path.join(PROMPTS, "nvrtc_example.py"), self.w("help", "nvrtc_example.py"))
        shutil.copy(os.path.join(PROMPTS, "test_example.py"), self.w("help", "test_example.py"))
        shutil.copy(os.path.join(PROMPTS, "PITFALLS.md"), self.w("help", "PITFALLS.md"))
        with open(self.w("CLAUDE.md"), "w") as f:
            f.write("@AGENTS.md\n")
        self._shim(self.w("kar"))
        self._mcp_config()
        for name in ("facts.md", "dead_ends.md"):
            open(self.p("ledger", name), "a").close()
        self._publish_notes()

    # ---- publishing to the workspace -----------------------------------------------------------------------

    def _copy_ro(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.chmod(dst, 0o644)
        shutil.copy(src, dst)
        os.chmod(dst, 0o444)

    def _publish_candidate(self, cid):
        for name in ("kernel.py", "report.md"):
            src = self.p("candidates", cid, name)
            if os.path.exists(src):
                self._copy_ro(src, self.w("candidates", cid, name))

    def _publish_evidence(self, cid):
        for name in ("profile.md", "gap.json"):
            src = self.p("evidence", cid, name)
            if os.path.exists(src):
                self._copy_ro(src, self.w("evidence", cid, name))

    def _publish_notes(self):
        for name in ("facts.md", "dead_ends.md"):
            self._copy_ro(self.p("ledger", name), self.w("notes", name))

    def _freeze_candidate(self, st, path, hyp, report, kind="candidate", lane=None):
        cid = f"c{st['next_cand']:04d}"
        st["next_cand"] += 1
        d = self.p("candidates", cid)
        os.makedirs(d)
        shutil.copy(path, os.path.join(d, "kernel.py"))
        os.chmod(os.path.join(d, "kernel.py"), 0o444)
        if report and os.path.exists(report):
            shutil.copy(report, os.path.join(d, "report.md"))
        meta = {"id": cid, "kind": kind, "hyp": hyp, "lane": lane, "parent": st.get("best"),
                "sha": _sha(os.path.join(d, "kernel.py")),
                "source": os.path.relpath(path, self.w()) if path.startswith(self.w()) else "task baseline",
                "t": now()}
        write_json(os.path.join(d, "meta.json"), meta)
        self._publish_candidate(cid)
        return cid

    # ---- evidence ------------------------------------------------------------------------------------------

    def _profile_best(self, cid):
        r = self.runner(f"profile_{cid}")
        path = self.p("candidates", cid, "kernel.py")
        res = checks.profile_page(r, self.task, path, ncu_n=self.cfg["ncu_shapes"], label=cid,
                                  title=f"Profile of the best so far: {cid}")
        if res.get("rejected"):
            raise RunError(f"{cid} failed its own re-check while being profiled: {res['rejected']}")
        d = self.p("evidence", cid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "profile.md"), "w") as f:
            f.write(res["page"])
        g = res["gap"]
        eff = next(iter(g["ab"]["results"].values()))
        out = {"sol": sol_of(g["rows"], self.score_fn()), "score": 1 + eff["effect"],
               "score_ci": [1 + x for x in eff["ci"]],
               "rows": g["rows"], "total_pct": g["total_pct"]}
        write_json(os.path.join(d, "gap.json"), out)
        self._publish_evidence(cid)
        return out

    def _sol_rows(self, times, r):
        from kar.sol import gap as G
        hw = r.calibrate()
        fl = G.floors(self.task, hw, self.task.shape_list("scored"))
        w = {s["name"]: s["weight"] for s in self.task.shape_list("scored")}
        return [{"shape": k, "weight": w[k], "floor_us": fl[k]["floor_us"], "time_us": t} for k, t in times.items()]

    def _floor_text(self, times, r):
        rows = self._sol_rows(times, r)
        per = ", ".join(f"{x['shape']} {100 * x['floor_us'] / x['time_us']:.0f}%" for x in rows)
        return f"  % of floor: {per}; overall {sol_of(rows, self.score_fn()) * 100:.1f}% ({self.target_text()})"

    # ---- lanes ---------------------------------------------------------------------------------------------

    def _open_lane(self, st, role, holder, agent=None, **fields):
        lid = f"L{st['next_lane']:02d}"
        st["next_lane"] += 1
        lane = {"id": lid, "role": role, "status": "active", "holder": holder, "agent": agent, "seen": time.time(),
                "opened": now(), "closed": None, "sessions": 0, "idle": 0, "base": st["best"], "tries": 0, "fails": 0,
                "hyp": None, "lens": None, "items": None, "cycle": st["cycle"], "outcome": None, **fields}
        st["lanes"][lid] = lane
        d = self.w("lanes", lid)
        os.makedirs(d, exist_ok=True)
        self._shim(os.path.join(d, "kar"), lane=lid)
        base = self.p("candidates", lane["base"], "kernel.py")
        if role in ("build", "stack", "audit", "survey"):
            shutil.copy(base, os.path.join(d, "base.py"))
            os.chmod(os.path.join(d, "base.py"), 0o644)
        if role in ("build", "stack"):                 # edit it in place: no agent should retype a kernel
            shutil.copy(base, os.path.join(d, "cand.py"))
            os.chmod(os.path.join(d, "cand.py"), 0o644)
        with open(os.path.join(d, "NOTES.md"), "w") as f:
            f.write(f"WORKING: {role}" + (f" {lane['hyp']}" if lane["hyp"] else "")
                    + (f" ({lane['lens']})" if lane["lens"] else "") + "\n\n"
                    "(Line 1 is this lane's verdict for the other agents. Keep notes here with numbers.)\n")
        self.ledger.append("lanes", {k: lane[k] for k in ("id", "role", "holder", "agent", "base", "hyp", "lens",
                                                          "cycle")} | {"event": "open", "items": lane["items"]})
        return lane

    def _close_lane(self, st, lane, outcome):
        lane.update(status="closed", closed=now(), outcome=outcome)
        self.ledger.append("lanes", {"id": lane["id"], "event": "close", "outcome": outcome})
        self._verdict_line(lane, outcome)

    def _verdict_line(self, lane, outcome):
        p = self.w("lanes", lane["id"], "NOTES.md")
        try:
            with open(p) as f:
                rest = f.read().split("\n", 1)
        except OSError:
            rest = ["", ""]
        with open(p, "w") as f:
            f.write(f"CLOSED ({lane['role']}): {outcome[:300]}\n" + (rest[1] if len(rest) > 1 else ""))

    def _lane(self, st, lid):
        lane = st["lanes"].get(lid)
        if lane is None:
            raise RunError(f"no lane {lid} in this run")
        return lane

    def _post(self, st, to, frm, text):
        targets = [lid for lid, x in st["lanes"].items() if x["status"] == "active" and lid != frm] \
            if to == "all" else [to]
        for lid in targets:
            st["inbox"].setdefault(lid, []).append({"from": frm, "t": now(), "text": text})
        self.ledger.append("messages", {"from": frm, "to": to, "text": text})
        return targets

    def touch(self, lid):
        """Every command from a lane renews its lease and collects its messages."""
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            lane["seen"] = time.time()
            msgs = st["inbox"].pop(lid, [])
            self.save(st)
        if not msgs:
            return ""
        return "Messages for you:\n" + "\n".join(f"  [{m['t'][11:16]}] from {m['from']}: {m['text']}" for m in msgs) \
            + "\n\n"

    # ---- the scheduler -------------------------------------------------------------------------------------

    def _maybe_done(self, st):
        if st["phase"] == "done":
            return True
        why = self.over_budget()
        if not why and st.get("target_met") == st["best"] and not self._fixes_pending(st) and \
                (self.cfg["audit"] == "off" or st["best"] in st["audited"]):
            why = st.get("target_reason") or "target reached"
        if not why and self.met(st["best_sol"]) and st["best"] == st.get("baseline"):
            why = (f"the current best ({st['best']}) is already at {st['best_sol'] * 100:.1f}% of speed of light, at "
                   f"or above the {self.target_text()}")
        if why:
            st.update(phase="done", done_reason=why)
            return True
        return False

    @staticmethod
    def _fixes_pending(st):
        return any(h.get("kind") == "fix" and h["status"] in OPEN_LIKE for h in st["hyps"])

    def assign(self, holder, agent=None, roles=ROLES):
        """-> (lane or None, text). Hands the caller the most useful piece of work there is right now."""
        with self.lock():
            st = self.state
            if self._maybe_done(st):
                self.save(st)
                return None, f"STOP: the run is over ({st['done_reason']})."
            lane = self._pick(st, holder, agent, roles)
            self.save(st)
        if lane is None:
            return None, ("WAIT: nothing to hand out right now (the other lanes are finishing the work this needs: "
                          "a survey or a plan). Try again in a few minutes.")
        return lane, ""

    def _pick(self, st, holder, agent, roles):
        active = [x for x in st["lanes"].values() if x["status"] == "active"]
        lease = self.cfg["lease_min"] * 60
        for x in active:                                         # work in flight that lost its agent
            if x["role"] in roles and (x["holder"] is None or time.time() - x["seen"] > lease):
                self.ledger.append("lanes", {"id": x["id"], "event": "taken over", "from": x["holder"], "by": holder})
                x.update(holder=holder, agent=agent, seen=time.time())
                return x

        def busy(role, **kw):
            return any(x["role"] == role and all(x.get(k) == v for k, v in kw.items()) for x in active)

        if "stack" in roles and st["stack"] and not busy("stack"):
            items, st["stack"] = st["stack"], []
            return self._open_lane(st, "stack", holder, agent, items=items)
        if "audit" in roles and st["to_audit"] and not busy("audit"):
            st["to_audit"] = []          # audit what the team builds on now: the current best holds the queued code
            if st["best"] not in st["audited"]:
                return self._open_lane(st, "audit", holder, agent, base=st["best"])
        opens = [h for h in st["hyps"] if h["status"] == "open"]
        fixes = [h for h in opens if h.get("kind") == "fix"]
        if fixes and "build" in roles:                           # a hazard in the best comes before new work
            return self._build_lane(st, fixes[0], holder, agent)
        if st["planned"] and not opens and not busy("plan"):    # the round's hypotheses are all handed out
            # a new round needs something new to plan from: a new best, or a verdict on one of this round's
            # hypotheses. With every build still in flight, a second plan would repeat the first (measured: $1.83
            # for nothing), so a free lane waits.
            pb = st.get("plan_basis")
            closed = sum(1 for h in st["hyps"] if h["status"] not in ("open", "building"))
            building = any(h["status"] == "building" for h in st["hyps"])
            if pb is None or not building or st["best"] != pb["best"] or closed > pb["closed"]:
                st.update(cycle=st["cycle"] + 1, planned=False)
        if not st["planned"]:
            sv = st.get("survey")
            if self.lenses() and (sv is None or sv["cycle"] != st["cycle"]):
                fresh = (sv is None or sv["best"] != st["best"] or st.get("plateau_note")) and not (
                    sv is not None and self.low_budget())
                st["survey"] = {"cycle": st["cycle"], "best": st["best"], "files": [] if fresh else sv["files"],
                                "left": list(self.lenses()) if fresh else [], "reused_from": None if fresh else
                                sv["cycle"]}
                sv = st["survey"]
            if sv and sv["cycle"] == st["cycle"] and sv["left"] and "survey" in roles:
                return self._open_lane(st, "survey", holder, agent, lens=sv["left"].pop(0))
            surveying = (sv and sv["cycle"] == st["cycle"] and sv["left"]) or busy("survey", cycle=st["cycle"])
            if not surveying and not busy("plan") and "plan" in roles:
                return self._open_lane(st, "plan", holder, agent)
        if opens and "build" in roles:
            opens.sort(key=lambda h: (h.get("kind") != "fix", -h["predicted_gain_pct"]))
            return self._build_lane(st, opens[0], holder, agent)
        return None

    def _build_lane(self, st, h, holder, agent):
        h.update(status="building")
        lane = self._open_lane(st, "build", holder, agent, hyp=h["id"])
        h["lane"] = lane["id"]
        self.ledger.append("hypotheses", {"id": h["id"], "status": "building", "lane": lane["id"]})
        return lane

    def release_dead_holders(self):
        """After a driver died without cleaning up (kill -9, a crash, a reboot): lanes it held go back to the
        scheduler at once instead of waiting out their lease. -> the lane ids released."""
        out = []
        with self.lock():
            st = self.state
            for lane in st["lanes"].values():
                h = str(lane.get("holder") or "")
                if lane["status"] == "active" and h.startswith("kar-run:"):
                    try:
                        pid = int(h.split(":")[1])
                        os.kill(pid, 0)
                        alive = True
                    except (ValueError, IndexError, ProcessLookupError):
                        alive = False
                    except PermissionError:
                        alive = True
                    if not alive:
                        lane["holder"] = None
                        out.append(lane["id"])
            if out:
                self.ledger.append("interventions", {"by": "harness", "what": "released lanes held by a driver that "
                                                     "is no longer running: " + ", ".join(out)})
            self.save(st)
        return out

    def end_session(self, lid, progressed):
        """The driver calls this when a lane's agent session ends. A lane that keeps ending without progress, or
        that used all its sessions, is closed so the work goes back to the scheduler."""
        with self.lock():
            st = self.state
            lane = st["lanes"].get(lid)
            if lane is None or lane["status"] != "active":
                return
            lane["sessions"] += 1
            lane["idle"] = 0 if progressed else lane["idle"] + 1
            why = None
            if lane["idle"] >= 2:
                why = f"stalled: {lane['idle']} sessions in a row without a step"
            elif lane["sessions"] >= self.cfg["lane_sessions"]:
                why = f"stalled: used its {lane['sessions']} sessions"
            if why:
                self._stall(st, lane, why)
            lane["holder"] = None
            self.save(st)

    def _stall(self, st, lane, why):
        if lane["role"] == "build" and lane["hyp"] and self._hyp(st, lane["hyp"]).get("kind") == "fix":
            h = self._hyp(st, lane["hyp"])            # a hazard does not go away because a session ran out
            h.update(status="open", lane=None)
            self.ledger.append("hypotheses", {"id": h["id"], "status": "open", "why": f"back in the queue: {why}"})
        elif lane["role"] == "build" and lane["hyp"]:
            h = self._hyp(st, lane["hyp"])
            h.update(status="dead", why=why)
            self.ledger.append("hypotheses", {"id": h["id"], "status": "dead", "why": why})
            self.ledger.note("dead_ends.md", f"{h['id']} {h['title']} [stalled]: {why}")
            self._publish_notes()
        elif lane["role"] == "stack":
            for it in lane["items"] or []:
                self.ledger.note("dead_ends.md", f"{it['cid']} ({it.get('title', '')}) [not stacked]: {why}")
            self._publish_notes()
        elif lane["role"] == "plan":
            n = st["plan_failures"].get(str(st["cycle"]), 0) + 1
            st["plan_failures"][str(st["cycle"])] = n
            if n >= 2:
                st.update(phase="done", done_reason=f"no accepted plan in round {st['cycle']} after {n} planner "
                                                    f"lanes")
        elif lane["role"] == "audit":
            st["audited"].append(lane["base"])
        self._close_lane(st, lane, why)

    def _hyp(self, st, hid):
        return next(h for h in st["hyps"] if h["id"] == hid)

    # ---- briefs --------------------------------------------------------------------------------------------

    def _status_line(self, st):
        c, b = self.costs(), self.cfg["budget"]
        parts = [f"Task {self.task.name} on {self.cfg['gpu_name']}. Best so far: {st['best']} at "
                 f"{st['best_sol'] * 100:.1f}% of speed of light ({self.target_text()}), "
                 f"{st['best_score']:.3f}x the baseline."]
        used = [f"GPU {c['gpu_s'] / 3600:.2f}/{b['gpu_h']} h" if b.get("gpu_h") else None,
                f"submits {c['submits']}/{b['submits']}" if b.get("submits") else None,
                f"wall {c['wall_s'] / 3600:.1f}/{b['wall_h']} h" if b.get("wall_h") else None,
                f"agents ${c['agent_usd']:.2f}" + (f"/{b['agent_usd']:.0f}" if b.get("agent_usd") else "")]
        parts.append("Budget used: " + ", ".join(x for x in used if x) + ".")
        return " ".join(parts)

    def _kept_text(self):
        rows = [c for c in self.ledger.records("candidates") if c.get("decision") in ("KEEP", "KEEP-ON-BASE")]
        if not rows:
            return "Nothing yet: the best is the baseline."
        return "\n".join(f"- {c['id']} ({c.get('hyp')}: {c.get('title', '')}): {c['effect'] * 100:+.2f}% "
                         f"[{c['ci'][0] * 100:+.2f}, {c['ci'][1] * 100:+.2f}] over {c['parent']}"
                         + (" (judged on an older best; sent to stacking)" if c["decision"] == "KEEP-ON-BASE" else "")
                         for c in rows)

    def _dead_text(self, limit=25):
        with open(self.p("ledger", "dead_ends.md")) as f:
            lines = [x.rstrip() for x in f if x.strip()]
        return "\n".join(lines[-limit:]) if lines else "None yet."

    def _facts_text(self, limit=40):
        with open(self.p("ledger", "facts.md")) as f:
            lines = [x.rstrip() for x in f if x.strip()]
        return "\n".join(lines[-limit:]) if lines else "None yet."

    def _strategy(self, st):
        aim = self.aim(st)
        need = math.log(aim / st["best_sol"]) if st["best_sol"] < aim else 0
        if need <= 0:
            return "The target is reached."
        what = "the target" if self.target() is not None else \
            f"{aim * 100:.0f}% of speed of light (half the remaining gap to the floor, the bar for this round)"
        return (f"To reach {what} the best must get {math.expm1(need) * 100:.1f}% faster in score. If your "
                f"hypotheses together cannot add up to that even if every one pays, include at least one structural "
                f"change; the harness rejects a plan that cannot get there without one.")

    def board(self, lid=None):
        """What every agent sees: the best, every lane at work, the queue, recent decisions."""
        st, c, b = self.state, self.costs(), self.cfg["budget"]
        lines = [f"# Board: {self.task.name} on {self.cfg['gpu_name']}", "",
                 f"Best {st['best']}: {st['best_sol'] * 100:.1f}% of speed of light (target "
                 f"{self.target_text()}), {st['best_score']:.3f}x the baseline. Round {st['cycle']}"
                 + (", planned" if st["planned"] else ", planning") + f". Phase {st['phase']}"
                 + (f" ({st['done_reason']})" if st["done_reason"] else "") + ".",
                 f"Budget used: GPU {c['gpu_s'] / 3600:.2f}/{b['gpu_h']} h, submits {c['submits']}/{b['submits']}, "
                 f"agent sessions {c['sessions']}/{self.session_cap()}, agents ${c['agent_usd']:.2f}"
                 + (f"/{b['agent_usd']:.0f}" if b.get("agent_usd") else "") + ".", "", "## Lanes at work", ""]
        active = [x for x in st["lanes"].values() if x["status"] == "active"]
        tries = {}
        for t in self.ledger.records("tries"):
            tries[t.get("lane")] = t
        for x in active:
            what = {"build": lambda: f"{x['hyp']} {self._hyp(st, x['hyp'])['title']}",
                    "survey": lambda: f"lens {x['lens']}",
                    "plan": lambda: f"round {x['cycle']}",
                    "stack": lambda: "porting " + ", ".join(i["cid"] for i in x["items"] or []),
                    "audit": lambda: f"auditing {x['base']}"}[x["role"]]()
            last = tries.get(x["id"])
            lt = (f"; last try {last['effect'] * 100:+.1f}% [{last['ci'][0] * 100:+.1f}, {last['ci'][1] * 100:+.1f}]"
                  if last else "")
            me = " (you)" if x["id"] == lid else ""
            lines.append(f"- {x['id']}{me} {x['role']}: {what}; base {x['base']}, tries {x['tries']}, agent "
                         f"{x.get('agent') or x.get('holder') or '-'}{lt}. Notes: lanes/{x['id']}/NOTES.md")
        if not active:
            lines.append("(none)")
        opens = sorted((h for h in st["hyps"] if h["status"] == "open"),
                       key=lambda h: (h.get("kind") != "fix", -h["predicted_gain_pct"]))
        lines += ["", "## Queue (hypotheses not started)", ""]
        lines += [f"- {h['id']} {h['predicted_gain_pct']:.1f}% {h['title']}" + (" [fix]" if h.get("kind") == "fix"
                                                                                else "") for h in opens] or ["(empty)"]
        if st["stack"]:
            lines += ["", "Waiting for a stacker: " + ", ".join(f"{i['cid']} ({i['effect'] * 100:+.1f}% on "
                                                                  f"{i['base']})" for i in st["stack"])]
        lines += ["", "## Recent decisions", ""]
        recent = self.ledger.records("candidates")[-8:]
        for r in recent:
            eff = f" {r['effect'] * 100:+.2f}%" if r.get("effect") is not None else ""
            lines.append(f"- {r['id']} ({r.get('lane', '-')}, {r.get('hyp', '-')}): {r['decision']}{eff} vs "
                         f"{r.get('vs', r.get('parent'))}" + (f": {r['reason'][:120]}" if r.get("reason") else ""))
        closed = [x for x in st["lanes"].values() if x["status"] == "closed"][-6:]
        lines += [f"- lane {x['id']} {x['role']} closed: {(x['outcome'] or '')[:140]}" for x in closed]
        if not recent and not closed:
            lines.append("(none yet)")
        from kar import judge as J
        if J.config(self):
            lines += ["", "## Judge", "", J.status_text(self)]
        lines += ["", "Read any lane's notes (lanes/<id>/NOTES.md), any candidate (candidates/<id>/kernel.py) and any "
                      "profile (evidence/<id>/profile.md). Message a lane: `post <lane|all> \"...\"`."]
        return "\n".join(lines)

    # ---- read-only -----------------------------------------------------------------------------------------

    def status(self):
        st, c = self.state, self.costs()
        lines = [f"run {self.root}", f"task {self.task.name}, phase {st['phase']}, round {st['cycle']}, team of "
                 f"{self.lanes_n()}" + (f" ({st['done_reason']})" if st["done_reason"] else ""),
                 f"best {st['best']}: {st['best_sol'] * 100:.1f}% of speed of light (target "
                 f"{self.target_text()}), {st['best_score']:.3f}x the baseline",
                 f"submits {st['submits']}, milestones {st['milestones']}, GPU {c['gpu_s'] / 3600:.2f} h, wall "
                 f"{c['wall_s'] / 3600:.1f} h, agent sessions {c['sessions']}, agent ${c['agent_usd']:.2f}, "
                 f"tokens {c['tokens']}"]
        for x in st["lanes"].values():
            if x["status"] == "active":
                lines.append(f"  lane {x['id']} {x['role']}" + (f" {x['hyp']}" if x["hyp"] else "")
                             + (f" ({x['lens']})" if x["lens"] else "") + f", tries {x['tries']}, agent "
                             f"{x.get('agent') or '-'}")
        for bh in st.get("best_history", []):
            lines.append(f"  {bh['id']}  {bh['sol'] * 100:5.1f}% of SOL  {bh['score']:.3f}x  {bh['t']}")
        lines += self.hazard_lines(st)
        return "\n".join(lines), 0

    def hazard_lines(self, st):
        """What a reader of the result must know about correctness: hazards still in the best, and faster kernels
        that were replaced because they were wrong."""
        out = []
        for h in st["hyps"]:
            if h.get("kind") == "fix" and h["status"] in ("open", "building", "unfixed", "abandoned", "dead"):
                proof = (f"proven by registered test(s) {', '.join(h['tests'])} (the reference passes, the best "
                         f"fails)") if h.get("tests") else "found by reading the code; no test proves it"
                out.append(f"KNOWN HAZARD in the best {st['best']}, not fixed ({h['id']}, {proof}): "
                           f"{' '.join((h.get('mechanism') or '').split())[:400]}")
        for f in st.get("faster_but_wrong", []):
            out.append(f"{f['cid']} is {f['faster_pct']:.2f}% faster than its replacement {f['replaced_by']} but "
                       f"wrong on some inputs" + (f" (fails {', '.join(f['fails'])})" if f["fails"] else "")
                       + f": candidates/{f['cid']}/kernel.py, if those inputs cannot occur for you.")
        return out

    def diff(self, a, b):
        def lines(cid):
            with open(self.p("candidates", cid, "kernel.py")) as f:
                return f.readlines()
        return "".join(difflib.unified_diff(lines(a), lines(b), f"{a}/kernel.py", f"{b}/kernel.py"))

    # ---- the report (a human command) ----------------------------------------------------------------------

    def report(self, reps=None):
        """Re-measure the best from its frozen file, check the run's integrity, write REPORT.md."""
        st = self.state
        integ = {"ledger hash chains": ", ".join(f"{k} {v}" for k, v in self.ledger.verify().items())}
        best = st["best"]
        path = self.p("candidates", best, "kernel.py")
        meta = read_json(self.p("candidates", best, "meta.json")) or {}
        integ["best file matches the hash recorded when it was frozen"] = "yes" if meta.get("sha") == _sha(path) \
            else "NO"
        hits = records.scan_sources(self)
        integ["candidate sources naming files the agents were not given"] = "; ".join(hits) if hits else "none"
        bad = [x for x in self.ledger.records("cost") if x.get("status") in ("evaluator_changed", "task_changed")]
        integ["jobs refused because the evaluator or task changed"] = str(len(bad))
        hz = [a for a in self.ledger.records("audits") if a.get("verdict") == "hazard"]
        integ["hazards raised by auditors"] = "; ".join(f"{a['cid']}: {a['text'][:160]}" for a in hz) if hz else "none"
        fin = {"integrity": integ}
        self.internal = True
        try:
            r = self.runner("report")
            rec = r.evaluate(best, path)
            if rec["verdict"] != "accept":
                fin["error"] = f"{best} failed its re-check: {rec['stage']}: {rec['reason']}"
            else:
                sr = r.stress(best, path)
                integ["best passes the stress suite on re-check"] = "yes" if sr["verdict"] == "accept" else \
                    f"NO: {sr.get('reason')}"
                reg = self._registered()
                if reg:                      # the team's own real-use tests, run on the final kernel
                    tr = r.tests(best, path, [self.p(t["file"]) for t in reg])
                    integ[f"best passes the team's {len(reg)} registered test(s) on re-check"] = \
                        "yes" if tr["verdict"] == "accept" else f"NO: {tr.get('reason')}"
                reps = reps or 2 * self.task.bench["reps"]
                g = r.gap(path, reps=reps)
                eff = next(iter(g["ab"]["results"].values()))
                fin.update(best=best, score=1 + eff["effect"], score_ci=[1 + x for x in eff["ci"]],
                           sol=sol_of(g["rows"], self.score_fn()), rows=g["rows"], seed=g["ab"]["seed"], reps=reps,
                           copies=g["ab"]["copies"])
        finally:
            self.internal = False
        fin["hazards"] = self.hazard_lines(st)
        md = records.report_md(self, fin)
        with open(self.p("REPORT.md"), "w") as f:
            f.write(md)
        self.ledger.append("reports", {k: v for k, v in fin.items() if k != "rows"})
        return md, 0
