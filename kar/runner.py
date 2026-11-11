"""Evaluation pipeline over the backend: lint -> smoke -> check for one candidate, and same-job A/B timing.

A Runner is bound to one task, one output folder and one GPU. It freezes the evaluator and task hashes when it is
created (or takes them from an existing freeze.json), and every job it starts carries them.
"""
import json
import os
import random

from kar import backends
from kar.evaluate import lint as L
from kar.evaluate import stats
from kar.evaluate.manifest import evaluator_sha


class Runner:
    def __init__(self, task, out_dir, gpu=0, freeze=None, seed=None, guard=None, on_job=None, backend=None):
        """guard(mode, label) runs before every GPU job and may raise to refuse it (a budget); on_job(mode, label,
        result) runs after each one (cost accounting). backend: a kar.backends object, or a backend config dict;
        default the local GPU `gpu`."""
        self.task, self.out, self.gpu = task, os.path.abspath(out_dir), gpu
        self.backend = backend if hasattr(backend, "run") else backends.make(backend, gpu=gpu)
        self.guard, self.on_job = guard, on_job
        os.makedirs(self.out, exist_ok=True)
        self.freeze = freeze or {"evaluator_sha": evaluator_sha(), "task_sha": task.sha()}
        if freeze is None:
            with open(os.path.join(self.out, "freeze.json"), "w") as f:
                json.dump(self.freeze, f, indent=1)
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31)
        self.n = 0

    def job(self, mode, label, prefix_fn=None, **kw):
        self.n += 1
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)[:60]
        d = os.path.join(self.out, "jobs", f"{self.n:03d}_{mode}_{safe}")
        seed = (self.seed + 7907 * self.n) % 2**31
        job = {"mode": mode, "label": label, "task_root": self.task.root, "seed": seed, **self.freeze, **kw}
        lim = self.task.limits
        n_shapes = len(self.task.shape_list("scored")) + len(self.task.shape_list("nearby"))
        total = lim["start"] + 8 * lim["import"] + n_shapes * max(lim["shape"], lim["bench"]) + 60
        if self.guard:
            self.guard(mode, label)
        res = self.backend.run(job, d, limits=lim, total_s=total, prefix_fn=prefix_fn)
        if self.on_job:
            self.on_job(mode, label, res)
        return res

    def stress(self, label, path):
        """The stress suite (more seeds, concurrent streams, back-to-back calls) on an accepted kernel."""
        res = self.job("stress", label, path=os.path.abspath(path))
        return {**_brief(res), "verdict": "accept" if res.get("status") == "accept" else "reject"}

    def tests(self, label, path, test_paths):
        """The team's own tests (agent_test.py) on one kernel, in one job."""
        res = self.job("tests", label, path=os.path.abspath(path), tests=[os.path.abspath(t) for t in test_paths])
        return {**_brief(res), "cases": res.get("cases", []),
                "verdict": "accept" if res.get("status") == "accept" else "reject"}

    def lint(self, path):
        return L.lint(path, self.task.rules)

    def evaluate(self, label, path, lint=True):
        """lint -> smoke -> check. Returns a record with verdict 'accept' or 'reject' and the stage that decided."""
        rec = {"label": label, "path": path}
        if lint:
            rec["lint"] = self.lint(path)
            if not rec["lint"]["ok"]:
                return {**rec, "verdict": "reject", "stage": "lint", "reason": "; ".join(rec["lint"]["problems"])}
        path = os.path.abspath(path)
        sm = self.job("smoke", label, path=path)
        rec["smoke"] = _brief(sm)
        if sm.get("status") != "ok":
            return {**rec, "verdict": "reject", "stage": "smoke", "reason": sm.get("reason") or sm.get("status")}
        ck = self.job("check", label, path=path)
        rec["check"] = _brief(ck)
        if ck.get("status") != "accept":
            return {**rec, "verdict": "reject", "stage": "check", "reason": ck.get("reason") or ck.get("status")}
        return {**rec, "verdict": "accept", "stage": "check", "reason": ""}

    def ab(self, cands, control, reps=None, copies=None, unchecked=()):
        """Time candidates against a control in ONE job. cands: [(label, path)]; control: (label, path).

        The control gets `copies` copies like every candidate; with >= 2 copies the control-vs-control comparison
        measures placement noise, and the keep rule uses it."""
        reps = reps or self.task.bench["reps"]
        copies = copies or self.task.bench["copies"]
        seen = {control[0]}
        uniq = []
        for lb, p in cands:   # a candidate labelled like the control would merge with it (a known pitfall)
            while lb in seen:
                lb += "_cand"
            seen.add(lb)
            uniq.append((lb, p))
        cands = uniq
        variants = [{"label": lb, "path": os.path.abspath(p), "copies": copies} for lb, p in [control, *cands]]
        unchecked = [lb for lb, _ in cands if lb in set(unchecked)]
        res = self.job("bench", "ab_" + "_".join(lb for lb, _ in cands), variants=variants, reps=reps,
                       unchecked=unchecked)
        if res.get("status") != "ok":
            return {"status": res.get("status"), "reason": res.get("reason"), "job_dir": res.get("job_dir"),
                    "log_tail": res.get("log_tail")}
        b = res["bench"]
        ctrl = [f"{control[0]}#{c}" for c in range(copies)]
        noise = stats.compare(b, [ctrl[0]], [ctrl[1]]) if copies >= 2 else None
        out = {"status": "ok", "reps": reps, "copies": copies, "seed": self.seed, "empty_us": b["empty_us"],
               "noise": noise, "results": {}, "job_dir": res["job_dir"], "smi_before": res["smi_before"],
               "smi_after": res["smi_after"]}
        score_fn = self.task.attr(self.task.score) if self.task.score else None
        for lb, _ in cands:
            cmp = stats.compare(b, [f"{lb}#{c}" for c in range(copies)], ctrl, score_fn=score_fn)
            verdict, why = stats.decide(cmp, noise)
            out["results"][lb] = {**cmp, "verdict": verdict, "why": why}
        with open(os.path.join(self.out, f"ab_{self.n:03d}.json"), "w") as f:
            json.dump({**out, "bench": b}, f, indent=1)
        return out


    # ---- M1: calibration, speed-of-light gap table, profiles -------------------------------------------------

    def calibrate(self, force=False):
        """Measured GPU rates, cached per GPU model + driver under ~/.cache/kar/hw."""
        inf = self.backend.info()
        slug = "".join(c if c.isalnum() else "_" for c in f"{inf['name']}_{inf['arch']}_{inf['driver']}")
        path = os.path.join(HW_CACHE, slug + ".json")
        if os.path.exists(path) and not force:
            with open(path) as f:
                return json.load(f)
        res = self.job("calibrate", "calibrate")
        if res.get("status") != "ok":
            raise RuntimeError(f"calibration failed: {res.get('reason')} {res.get('log_tail', '')}")
        hw = {**res["hw"], "smi_before": res["smi_before"], "smi_after": res["smi_after"]}
        os.makedirs(HW_CACHE, exist_ok=True)
        with open(path, "w") as f:
            json.dump(hw, f, indent=1)
        return hw

    def gap(self, cand=None, reps=None, label=None):
        """Gap table for `cand` (default: the baseline itself), timed against the baseline in one job."""
        from kar.sol import gap as G
        hw = self.calibrate()
        base = ("baseline", self.task.path(self.task.baseline))
        label = label or (os.path.splitext(os.path.basename(cand))[0] if cand else "baseline_again")
        ab = self.ab([(label, cand or base[1])], base, reps=reps)
        if ab.get("status") != "ok":
            raise RuntimeError(f"A/B failed: {ab.get('reason')}")
        label, res = next(iter(ab["results"].items()))
        times = {k: v["a_us"] for k, v in res["shapes"].items()}
        btimes = {k: v["b_us"] for k, v in res["shapes"].items()}
        weights = {k: v["weight"] for k, v in res["shapes"].items()}
        scored = self.task.shape_list("scored")
        fl = G.floors(self.task, hw, scored)
        score_fn = self.task.attr(self.task.score) if self.task.score else None
        rows, total = G.table(fl, times, weights, baseline=btimes, score_fn=score_fn)
        md = G.markdown(rows, total, hw, title=f"Gap to speed of light: {label}", score_fn=score_fn)
        return {"hw": hw, "ab": ab, "rows": rows, "total_pct": total, "markdown": md}

    def nsys(self, label, path, reps=20):
        from kar.profile import nsys as N
        path = os.path.abspath(path)
        res = self.job("nsys", label, path=path, reps=reps,
                       prefix_fn=lambda d: N.command(os.path.join(d, "nsys")))
        rep = os.path.join(res["job_dir"], "nsys.nsys-rep")
        if res.get("status") != "ok" or not os.path.exists(rep):
            return {"status": res.get("status"), "reason": res.get("reason"), "log_tail": res.get("log_tail")}
        parsed = N.parse(N.export(rep, os.path.join(res["job_dir"], "nsys.sqlite")))
        return {"status": "ok", "parsed": parsed, "cubins": os.path.join(res["job_dir"], "cubins")}

    def ncu_full(self, label, path, shape, section_set="full"):
        """Every ncu section on one shape (`--set full`), as ncu's own text report: for a deeper look than the
        summary on the profile page."""
        from kar.profile import ncu as K
        path = os.path.abspath(path)
        res = self.job("ncu", f"{label}_{shape}_full", path=path, shape=shape,
                       prefix_fn=lambda d: K.command_set(os.path.join(d, "ncu"), section_set))
        rep = os.path.join(res["job_dir"], "ncu.ncu-rep")
        if not os.path.exists(rep):
            with open(os.path.join(res["job_dir"], "log.txt"), errors="replace") as f:
                log = f.read()
            hint = " (performance counters need permission: see ERR_NVGPUCTRPERM)" if "ERR_NVGPUCTRPERM" in log else ""
            return {"status": "failed", "reason": (res.get("reason") or "no report") + hint, "log_tail": log[-1500:]}
        return {"status": "ok", "text": K.details(rep), "rep": rep}

    def sass(self, label, path):
        """The SASS of every module image the kernel loads, plus the audit table."""
        from kar.profile import sass as SA
        ns = self.nsys(label, path, reps=2)
        if ns.get("status") != "ok":
            return ns
        return {"status": "ok", "text": SA.dump_dir(ns["cubins"]), "audit": SA.markdown(SA.audit_dir(ns["cubins"]))}

    def ncu(self, label, path, shape):
        from kar.profile import ncu as K
        path = os.path.abspath(path)
        metrics = list(K.METRICS)
        for _ in range(6):
            res = self.job("ncu", f"{label}_{shape}", path=path, shape=shape,
                           prefix_fn=lambda d: K.command(os.path.join(d, "ncu"), metrics))
            with open(os.path.join(res["job_dir"], "log.txt"), errors="replace") as f:
                log = f.read()
            bad = K.unknown_metrics(log)
            if bad:
                metrics = [m for m in metrics if m not in bad]
                continue
            break
        rep = os.path.join(res["job_dir"], "ncu.ncu-rep")
        if not os.path.exists(rep):
            tail = log[-1500:]
            hint = " (performance counters need permission: see ERR_NVGPUCTRPERM)" if "ERR_NVGPUCTRPERM" in log else ""
            return {"status": "failed", "reason": (res.get("reason") or "no report") + hint, "log_tail": tail}
        return {"status": "ok", "summary": K.summarize(K.raw_csv(rep)), "rep": rep}


HW_CACHE = os.path.expanduser("~/.cache/kar/hw")


def _brief(res):
    keep = ("status", "reason", "wall_s", "call_s", "import_s", "shape", "shapes", "device", "job_dir", "log_tail",
            "traceback")
    return {k: res[k] for k in keep if k in res}
