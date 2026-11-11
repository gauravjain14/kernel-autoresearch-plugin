"""An external judge, when the task has one: a leaderboard's CLI, a challenge's evaluation script, a CI job. Optional:
without one the team just keeps closing the gap to speed of light.

The judge is any command line, declared in task.toml or in a file given to `kar start --judge FILE`:

    [judge]
    check   = "mychallenge check {file} --problem P"   # an unscored evaluation (calibration, confirming a stack)
    submit  = "mychallenge submit {file} --problem P"  # a scored, public submission
    parse   = "judge.py:parse"       # in the task folder: parse(stdout, stderr, returncode) -> {"score": float, ...}
    filename = "submission.py"       # the name the judge wants the file under (default: kernel.py)
    higher_is_better = true          # false for a time
    quota   = { check = 20, submit = 10, window_h = 24 }
    policy  = { check = "auto", submit = "ask" }   # auto: the harness may run it; ask: a person approves each one
    min_gain_pct = 1.0               # auto checks only when the local score moved this much since the last check
    cwd = "."                        # relative to the task folder
    timeout_s = 1800

Placeholders: {file} (the frozen candidate, copied under `filename`), {cid}, {run}, {task}.

What the harness does with it, from our competition work:
- every judge run is logged with the candidate's sha in ledger/judge.jsonl, and nothing is ever judged twice by
  accident;
- the judge's noise comes from repeated runs of identical code (one reading is a draw);
- judge vs local is fitted from builds judged and measured both ways;
- automatic checks go only to stacks (the current best) that moved past the bar; agents never submit;
- a scored submit needs a person: `kar judge approve RUN ID`, or the plugin asks in the session.
"""
import math
import os
import re
import shlex
import shutil
import subprocess
import time

from kar.run.store import now, write_json

DEFAULTS = {"filename": "kernel.py", "higher_is_better": True, "quota": {"window_h": 24},
            "policy": {"check": "ask", "submit": "ask"}, "min_gain_pct": 1.0, "cwd": ".", "timeout_s": 1800}


class JudgeError(Exception):
    pass


def config(run):
    """The run's judge config (run.json "judge" overrides the task's [judge]), or None."""
    base = dict(getattr(run.task, "judge", None) or {})
    base.update(run.cfg.get("judge") or {})
    if not base.get("check") and not base.get("submit"):
        return None
    cfg = {**DEFAULTS, **base}
    cfg["quota"] = {**DEFAULTS["quota"], **(base.get("quota") or {})}
    cfg["policy"] = {**DEFAULTS["policy"], **(base.get("policy") or {})}
    return cfg


def _records(run):
    return run.ledger.records("judge")


def used(run, kind, cfg):
    window = cfg["quota"].get("window_h", 24) * 3600
    cut = time.time() - window
    return sum(1 for r in _records(run) if r.get("kind") == kind and r.get("ts", 0) >= cut
               and r.get("status") != "refused")


def left(run, kind, cfg):
    cap = cfg["quota"].get(kind)
    return None if cap is None else max(0, int(cap) - used(run, kind, cfg))


def _parse(run, cfg, out, err, code):
    if cfg.get("parse"):
        fn = run.task.attr(cfg["parse"])
        res = dict(fn(out, err, code) or {})
    else:                                   # a plain "score: 1.234" line
        m = re.findall(r"(?im)\bscore\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", out)
        res = {"score": float(m[-1])} if m else {}
    if code and "status" not in res:
        res["status"] = "error"
    res.setdefault("status", "ok" if res.get("score") is not None else "unparsed")
    return res


def run_judge(run, kind, cid, by, approved_by=None):
    """Run the judge on frozen candidate `cid`. -> the ledger record."""
    cfg = config(run)
    if cfg is None or not cfg.get(kind):
        raise JudgeError(f"no judge `{kind}` command is configured for this run")
    if cfg["policy"].get(kind, "ask") == "ask" and not approved_by:
        raise JudgeError(f"a judge {kind} needs a person's approval: `kar judge request {run.root} {kind} {cid}`, then "
                         f"`kar judge approve`")
    n_left = left(run, kind, cfg)
    if n_left is not None and n_left <= 0:
        raise JudgeError(f"no {kind} quota left ({cfg['quota'][kind]} per {cfg['quota']['window_h']} h)")
    src = run.p("candidates", cid, "kernel.py")
    if not os.path.exists(src):
        raise JudgeError(f"no candidate {cid}")
    sha = run._sha_of(src)
    d = run.p("judge", f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{cid}_{os.urandom(3).hex()}")
    os.makedirs(d)
    f = os.path.join(d, cfg["filename"])
    shutil.copy(src, f)
    vals = {"file": f, "cid": cid, "run": run.root, "task": run.task.root}
    cmd = [part.format(**vals) for part in shlex.split(cfg[kind])]
    cwd = os.path.normpath(os.path.join(run.task.root, cfg["cwd"]))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=int(cfg["timeout_s"]))
        out, err, code = p.stdout, p.stderr, p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        out, err, code = "", f"{type(e).__name__}: {e}", -1
    for name, text in (("stdout.txt", out), ("stderr.txt", err)):
        with open(os.path.join(d, name), "w") as fh:
            fh.write(text)
    res = _parse(run, cfg, out, err, code)
    rec = {"kind": kind, "cid": cid, "sha": sha, "by": by, "approved_by": approved_by, "ts": time.time(),
           "s": round(time.time() - t0, 1), "returncode": code, "dir": os.path.relpath(d, run.root),
           "local_score": _local_score(run, cid), **{k: v for k, v in res.items() if k != "raw"}}
    run.ledger.append("judge", rec)
    msg = (f"judge {kind} of {cid}: " + (f"score {res['score']}" if res.get("score") is not None else res["status"])
           + (f" (local estimate {rec['local_score']:.4g}x)" if rec.get("local_score") else ""))
    run.ledger.note("facts.md", msg)
    run._publish_notes()
    return rec


def _local_score(run, cid):
    for b in (run.state or {}).get("best_history", []):
        if b["id"] == cid:
            return b.get("score")
    return None


def noise(run):
    """Judge noise from repeated runs of identical code: pooled sd of log(score) per sha -> (sd, n_repeats)."""
    by_sha = {}
    for r in _records(run):
        if r.get("score") is not None and r.get("kind") in ("check", "submit"):
            by_sha.setdefault(r["sha"], []).append(math.log(abs(float(r["score"]))))
    ss, dof = 0.0, 0
    for xs in by_sha.values():
        if len(xs) > 1:
            m = sum(xs) / len(xs)
            ss += sum((x - m) ** 2 for x in xs)
            dof += len(xs) - 1
    return (math.sqrt(ss / dof), dof) if dof else (None, 0)


def calibration(run):
    """judge ~ a * local, fitted in log space over builds judged and measured both ways -> (a, n) or (None, 0)."""
    pts = [(math.log(r["local_score"]), math.log(abs(float(r["score"]))))
           for r in _records(run) if r.get("score") and r.get("local_score") and float(r["score"]) != 0]
    if not pts:
        return None, 0
    return math.exp(sum(j - l for l, j in pts) / len(pts)), len(pts)


def runs_needed(sd, gain):
    """Runs per arm to see a relative change `gain` at alpha .05, power .8 (two-sided z): 2 (1.96 + 0.84)^2 sd^2 / g^2."""
    if not sd or gain <= 0:
        return None
    return max(1, math.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / math.log1p(gain) ** 2))


# ---- requests and approvals (agents and the harness request; people approve) --------------------------------------

def request(run, kind, cid, by, why):
    rid = f"j{len(run.ledger.records('judge_requests')) + 1:03d}"
    run.ledger.append("judge_requests", {"id": rid, "kind": kind, "cid": cid, "by": by, "why": why, "event": "open"})
    return rid


def pending(run):
    state = {}
    for r in run.ledger.records("judge_requests"):
        state.setdefault(r["id"], {}).update(r)
    return [r for r in state.values() if r.get("event") == "open"]


def approve(run, rid, who="user"):
    req = next((r for r in pending(run) if r["id"] == rid), None)
    if req is None:
        raise JudgeError(f"no open judge request {rid}")
    run.ledger.append("judge_requests", {"id": rid, "event": "approved", "by": who})
    rec = run_judge(run, req["kind"], req["cid"], by=req["by"], approved_by=who)
    run.ledger.append("judge_requests", {"id": rid, "event": "done", "result": rec.get("score")})
    return rec


def tick(run):
    """Called by the harness process every few minutes: an automatic check of the current best when the policy
    allows it, quota remains, and the local score moved at least min_gain_pct since the last judged build."""
    cfg = config(run)
    if cfg is None or cfg["policy"].get("check") != "auto" or not cfg.get("check"):
        return None
    st = run.state
    if not st or st.get("phase") == "done" or st["best"] == st.get("baseline"):
        return None
    if left(run, "check", cfg) == 0:
        return None
    recs = [r for r in _records(run) if r.get("kind") == "check"]
    if any(r.get("sha") == run._sha_of(run.p("candidates", st["best"], "kernel.py")) for r in recs):
        return None
    last = recs[-1].get("local_score") if recs else 1.0
    cur = st.get("best_score") or 1.0
    if last and cur / last - 1 < cfg["min_gain_pct"] / 100:
        return None
    return run_judge(run, "check", st["best"], by="harness (auto)", approved_by="policy: check = auto")


def status_text(run):
    cfg = config(run)
    if cfg is None:
        return "No judge is configured: the team closes the gap to speed of light on its own measurements."
    lines = [f"Judge: check {'on' if cfg.get('check') else 'off'} (policy {cfg['policy'].get('check')}), submit "
             f"{'on' if cfg.get('submit') else 'off'} (policy {cfg['policy'].get('submit')})."]
    for kind in ("check", "submit"):
        n = left(run, kind, cfg)
        if cfg.get(kind) and n is not None:
            lines.append(f"  {kind}s left in the window: {n} of {cfg['quota'][kind]}")
    sd, dof = noise(run)
    if sd:
        lines.append(f"  noise: {sd * 100:.2f}% per run (from {dof} repeat degrees of freedom); runs per arm to see "
                     f"+1%: {runs_needed(sd, 0.01)}, +2%: {runs_needed(sd, 0.02)}")
    a, n = calibration(run)
    if a:
        lines.append(f"  judge ~ {a:.3f} x local score (from {n} builds judged and measured)")
    for r in _records(run)[-6:]:
        lines.append(f"  {r['t'][:16]} {r['kind']} {r['cid']}: " + (f"score {r['score']}" if r.get("score") is not None
                                                                     else r.get("status", "?"))
                     + (f" (local {r['local_score']:.4g}x)" if r.get("local_score") else ""))
    for p in pending(run):
        lines.append(f"  waiting for approval: {p['id']} {p['kind']} {p['cid']} ({p['by']}: {p['why'][:100]})")
    return "\n".join(lines)


def save_config(run, path):
    """Attach a judge config file (TOML with a [judge] table, or the table itself) to a run."""
    import tomllib
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    run.cfg["judge"] = raw.get("judge", raw)
    write_json(run.p("run.json"), run.cfg)
    run.ledger.append("interventions", {"by": "user", "what": f"attached judge config {path} at {now()}"})
