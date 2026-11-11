"""The two multi-step checks shared by the CLI and the run loop.

run_init      prove the evaluator on a task: reference, baseline and known-good kernels accepted, every canary
              rejected at its expected stage, baseline against itself inside the noise.
profile_page  one page for one candidate: check, gap to speed of light, nsys per-kernel breakdown, ncu on the shapes
              with most to gain, SASS audit.
"""
import json
import os

from kar.evaluate import canaries
from kar.runner import Runner


def _pct(x):
    return f"{x * 100:+.2f}%"


def _ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


def ab_text(out, title=""):
    if out.get("status") != "ok":
        s = f"  A/B failed: {out.get('status')}: {out.get('reason')}"
        return s + ("\n" + out["log_tail"] if out.get("log_tail") else "")
    lines = [title] if title else []
    for lb, r in out["results"].items():
        lines.append(f"  {lb}: {_pct(r['effect'])} {_ci(r['ci'])} -> {r['verdict']} {r['why']}")
        for name, p in r["shapes"].items():
            lines.append(f"      {name:22s} w{p['weight']:<4g} control {p['b_us']:9.2f} us  cand {p['a_us']:9.2f} us  "
                         f"{_pct(p['effect'])} {_ci(p['ci'])}")
    n = out.get("noise")
    if n:
        lines.append(f"  noise (control copy vs copy): {_pct(n['effect'])} {_ci(n['ci'])}")
    lines.append(f"  {out['reps']} reps x {out['copies']} copies, seed {out['seed']}, empty graph "
                 f"{out['empty_us']:.2f} us; clocks before/after {out['smi_before'].get('clocks.sm')} / "
                 f"{out['smi_after'].get('clocks.sm')}")
    return "\n".join(lines)


def print_ab(out, title=""):
    print(ab_text(out, title))


def run_init(task, out, gpu=0, reps=None, log=print, backend=None):
    """Returns the init report; report['ok'] says whether the task and evaluator passed."""
    scored, nearby = task.shape_list("scored"), task.shape_list("nearby")
    log(f"task {task.name}: {len(scored)} scored shapes, {len(nearby)} nearby, "
        f"exact reference: {'yes' if task.exact else 'no'}")
    problems = []
    if not nearby:
        problems.append("no nearby shapes: a kernel specialised to the scored shapes cannot be caught")
    if not task.exact:
        log("  note: no exact reference, so the precision (error-ratio) test is off")
    r = Runner(task, out, gpu=gpu, backend=backend)
    inf = r.backend.info()
    name, arch = inf["name"], inf["arch"]
    log(f"{r.backend.describe()}: {name} ({arch}); task arch {task.arch}")
    if task.arch != "any" and arch != task.arch:
        log(f"FAILED: task needs {task.arch}")
        return {"ok": False, "wrong_arch": True, "problems": [f"task needs {task.arch}, GPU is {arch}"]}
    log(f"frozen: evaluator {r.freeze['evaluator_sha'][:12]}, task {r.freeze['task_sha'][:12]}  ({out})")

    plan = [("reference", canaries.reference_candidate(task, os.path.join(out, "gen")), "accept", False),
            ("baseline", task.path(task.baseline), "accept", True)]
    plan += [(f"good:{os.path.basename(g)}", task.path(g), "accept", True) for g in task.good]
    for label, path, stage in canaries.materialize(task, os.path.join(out, "gen")):
        plan.append((label, path, "reject:" + stage, stage == "lint"))

    log(f"\n{'candidate':28s} {'expected':14s} result")
    records = []
    for label, path, expect, do_lint in plan:
        rec = r.evaluate(label, path, lint=do_lint)
        if rec["verdict"] == "accept" and (expect == "accept" or expect == "reject:stress"):
            sr = r.stress(label, path)              # what a run's judge also runs before it times anything
            if sr["verdict"] != "accept":
                rec.update(verdict="reject", stage="stress", reason=sr.get("reason") or "stress failed")
        want_accept = expect == "accept"
        ok = (rec["verdict"] == "accept") if want_accept else (
            rec["verdict"] == "reject" and rec["stage"] == expect.split(":")[1])
        rec["expected"], rec["as_expected"] = expect, ok
        records.append(rec)
        got = "accepted" if rec["verdict"] == "accept" else f"rejected at {rec['stage']}: {rec['reason']}"
        log(f"{label:28s} {expect:14s} {'ok  ' if ok else 'FAIL'} {got[:150]}")
        if not ok:
            problems.append(f"{label}: expected {expect}, got {got}")

    log("\nnoise: baseline against itself")
    base = ("baseline", task.path(task.baseline))

    def null_ok(v):
        return (v["ci"][0] <= 0 <= v["ci"][1] or abs(v["effect"]) <= 0.005) and v["verdict"] != "NOISY"

    ab = r.ab([("baseline_again", base[1])], base, reps=reps)
    log(ab_text(ab))
    if ab.get("status") == "ok" and not null_ok(ab["results"]["baseline_again"]):
        # a 95% interval misses a true zero 1 time in 20: measure again, and fail only if both miss
        log("  outside the noise allowance: measuring once more")
        first = ab["results"]["baseline_again"]
        ab = r.ab([("baseline_again", base[1])], base, reps=reps)
        log(ab_text(ab))
        if ab.get("status") == "ok" and not null_ok(ab["results"]["baseline_again"]):
            v = ab["results"]["baseline_again"]
            problems.append(f"baseline vs itself reads {_pct(first['effect'])} {_ci(first['ci'])} and then "
                            f"{_pct(v['effect'])} {_ci(v['ci'])} ({v['verdict']}): the bench resolves a difference "
                            f"that cannot exist")
    if ab.get("status") != "ok":
        problems.append("baseline-vs-baseline A/B failed")
    good_ok = [rec for rec in records if rec["label"].startswith("good:") and rec["verdict"] == "accept"]
    info = None
    if good_ok:
        log("\ninfo: known-good kernels against the baseline")
        info = r.ab([(rec["label"].split(":", 1)[1], rec["path"]) for rec in good_ok], base, reps=reps)
        log(ab_text(info))
    report = {"ok": not problems, "task": task.name, "out": out, "freeze": r.freeze, "gpu": name, "arch": arch,
              "seed": r.seed, "records": records, "noise_ab": ab, "good_ab": info, "problems": problems}
    with open(os.path.join(out, "init.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    log("\nINIT PASSED" if not problems else "\nINIT FAILED:\n  " + "\n  ".join(problems))
    return report


def _res(g):
    return next(iter(g["ab"]["results"].values()))


def profile_page(r, task, cand, reps=None, ncu_n=2, label=None, title=None):
    """{'page', 'gap', 'check'} for an accepted candidate, or {'rejected': reason}."""
    from kar.profile import ncu as K
    from kar.profile import nsys as N
    from kar.profile import sass as SA
    label = label or os.path.splitext(os.path.basename(cand))[0]
    rec = r.evaluate(label, cand)
    if rec["verdict"] != "accept":
        return {"rejected": f"rejected at {rec['stage']}: {rec['reason']}", "check": rec}
    g = r.gap(cand, reps=reps, label=label)
    times = {row["shape"]: row["time_us"] for row in g["rows"]}
    ns = r.nsys(label, cand)
    worst = max((x.get("mean_ratio", 0) for x in rec["check"].get("shapes", [])), default=0)
    parts = [f"# {title or 'Profile: ' + label} ({task.name})", "",
             f"Check: accepted on {len(rec['check'].get('shapes', []))} shapes (worst mean-error ratio "
             f"{worst:.2f}x). Against the baseline: {_pct(_res(g)['effect'])} {_ci(_res(g)['ci'])}.", "",
             g["markdown"], "## Where the time goes (nsys)", ""]
    parts.append(N.markdown(ns["parsed"], times, order=[row["shape"] for row in g["rows"]])
                 if ns.get("status") == "ok" else
                 f"nsys failed: {ns.get('reason')}\n```\n{ns.get('log_tail', '')}\n```\n")
    parts += ["## Hardware counters (ncu)", ""]
    for row in g["rows"][:ncu_n]:
        k = r.ncu(label, cand, row["shape"])
        parts.append(K.markdown(k["summary"], row["shape"]) if k.get("status") == "ok" else
                     f"ncu on {row['shape']} failed: {k.get('reason')}\n")
    parts += ["## SASS", ""]
    parts.append(SA.markdown(SA.audit_dir(ns["cubins"])) if ns.get("status") == "ok" else "no images\n")
    return {"page": "\n".join(parts), "gap": g, "check": rec}
