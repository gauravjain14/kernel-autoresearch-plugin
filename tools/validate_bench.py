"""GPU self-test of the A/B statistics on a real task.

Null runs: a module timed against itself many times. With honest CIs about 95% of runs contain zero, and the keep
rule must never say KEEP or SLOWER. Effect runs (GEMV only): the known-good kernel against a copy that spins SPIN
cycles per warp after its work. A long run (many reps) gives the true effect; standard runs must detect it and their
CIs should cover it.

    python3 tools/validate_bench.py tasks/toy/gemv --nulls 8 --spin 3000 --effect-runs 6
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kar.runner import Runner  # noqa: E402
from kar.task import spec  # noqa: E402

ANCHOR = "  bf16* y = reinterpret_cast<bf16*>(p.y);"
SPIN = "  { long long t0 = clock64(); while (clock64() - t0 < %d) { } }\n"


def spun(src_path, out_path, cycles):
    with open(src_path) as f:
        s = f.read()
    assert ANCHOR in s
    with open(out_path, "w") as f:
        f.write(s.replace(ANCHOR, SPIN % cycles + ANCHOR))
    return out_path


def row(tag, res):
    r = next(iter(res["results"].values()))
    n = res["noise"]
    return {"tag": tag, "effect": r["effect"], "ci": r["ci"], "verdict": r["verdict"],
            "noise": n["effect"] if n else None, "noise_ci": n["ci"] if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--nulls", type=int, default=8)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--effect-runs", type=int, default=6)
    ap.add_argument("--truth-reps", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    task = spec.load(a.task)
    r = Runner(task, a.out)
    rows = []
    mods = [("baseline", task.path(task.baseline))] + [(os.path.basename(g), task.path(g)) for g in task.good]
    for k in range(a.nulls):
        for label, path in mods:
            res = r.ab([(label + "_again", path)], (label, path))
            rows.append(row(f"null {label}", res))
            print(json.dumps(rows[-1]), flush=True)
    if a.spin and task.good:
        good = task.path(task.good[0])
        slow = spun(good, os.path.join(a.out, f"slow_spin{a.spin}.py"), a.spin)
        truth = r.ab([("fast", good)], ("slow", slow), reps=a.truth_reps)
        rows.append(row("truth", truth))
        print(json.dumps(rows[-1]), flush=True)
        for k in range(a.effect_runs):
            res = r.ab([("fast", good)], ("slow", slow))
            rows.append(row("effect", res))
            print(json.dumps(rows[-1]), flush=True)
    nulls = [x for x in rows if x["tag"].startswith("null")]
    cover = sum(x["ci"][0] <= 0 <= x["ci"][1] for x in nulls)
    wrong = [x for x in nulls if x["verdict"] in ("KEEP", "SLOWER", "MIXED")]
    print(f"\nnull runs: {cover}/{len(nulls)} CIs contain 0; {len(wrong)} wrong verdicts (KEEP/SLOWER/MIXED)")
    eff = [x for x in rows if x["tag"] == "effect"]
    if eff:
        t = next(x for x in rows if x["tag"] == "truth")
        hit = sum(x["verdict"] == "KEEP" for x in eff)
        cov = sum(x["ci"][0] <= t["effect"] <= x["ci"][1] for x in eff)
        print(f"truth {t['effect']:+.3%} {t['ci']}; effect runs: {hit}/{len(eff)} KEEP, {cov}/{len(eff)} CIs cover it")
    with open(os.path.join(a.out, "validate.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
