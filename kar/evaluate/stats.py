"""Paired statistics for A/B timing.

Reps come in (even, odd) pairs that ran in opposite orders; the geometric mean of a pair cancels position effects.
Copies of one variant are averaged per pair. Pair k ran on placement instance k % instances. Effects are log-ratios;
a shape's effect is the trimmed mean of paired log-ratios, the score effect the weight-averaged log effect (a weighted
geomean speedup). CIs come from a two-level bootstrap: resample instances, then pairs within each drawn instance, so the
spread between memory placements counts as uncertainty instead of hiding as a fixed bias.
"""
import math

import numpy as np

KEEP, SLOWER, NO_GAIN, MIXED, NOISY = "KEEP", "SLOWER", "NO-GAIN", "MIXED", "NOISY"


def pair_logs(times):
    t = np.log(np.asarray(times, dtype=float))
    n = len(t) // 2 * 2
    return (t[0:n:2] + t[1:n:2]) / 2


def group_logs(times_by_label, labels):
    return np.mean([pair_logs(times_by_label[lb]) for lb in labels], axis=0)


TRIM = 0.1


def center(x, axis=-1):
    """10%-trimmed mean. Small-kernel timings cluster in launch-timing modes (an empty graph read exactly 5.12 us
    in 724 of 2000 replays), so a median jumps between modes; a trimmed mean moves smoothly and still drops outliers."""
    x = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = x.shape[axis]
    k = int(n * TRIM)
    return np.mean(np.take(x, np.arange(k, n - k), axis=axis), axis=axis)


def _ci(boot):
    return [math.expm1(float(np.percentile(boot, 2.5))), math.expm1(float(np.percentile(boot, 97.5)))]


def boot_center(d, instances, n_boot, rng):
    """Bootstrap distribution of center(d), clustered by instance (pair k belongs to instance k % instances)."""
    inst = max(1, min(instances, len(d)))
    m = len(d) // inst
    if inst == 1 or m < 1:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        return center(d[idx], axis=1)
    groups = np.stack([d[c::inst][:m] for c in range(inst)])        # (inst, m)
    pick = rng.integers(0, inst, size=(n_boot, inst))                 # which instances
    within = rng.integers(0, m, size=(n_boot, inst, m))               # which pairs inside each
    sample = groups[pick[:, :, None], within].reshape(n_boot, inst * m)
    return center(sample, axis=1)


def _boot_index(n, instances, n_boot, rng):
    """Resampled pair indices, shape (n_boot, n'), clustered by instance exactly as boot_center: pair k belongs to
    instance k % instances; draw instances with replacement, then pairs within each drawn instance."""
    inst = max(1, min(instances, n))
    m = n // inst
    if inst == 1 or m < 1:
        return rng.integers(0, n, size=(n_boot, n))
    pick = rng.integers(0, inst, size=(n_boot, inst))
    within = rng.integers(0, m, size=(n_boot, inst, m))
    return (pick[:, :, None] + inst * within).reshape(n_boot, inst * m)


def compare(bench, a, b, n_boot=4000, seed=0, score_fn=None):
    """Speed of label group `a` relative to group `b`; effect > 0 means `a` is faster.

    The score effect is the weighted geomean of per-shape speedups, or, when the task defines its own score
    (`score_fn(times) -> higher is better`, times in us per shape), score(a) / score(b) - 1, with the same clustered
    bootstrap applied jointly to every shape's paired times."""
    rng = np.random.default_rng(seed)
    inst = int(bench.get("instances", 1))
    per, boots, weights, logs = {}, [], [], {}
    for name, s in bench["shapes"].items():
        la, lb = group_logs(s["times"], a), group_logs(s["times"], b)
        d = lb - la
        boot = boot_center(d, inst, n_boot, rng)
        per[name] = {"effect": math.expm1(float(center(d))), "ci": _ci(boot), "weight": s["weight"],
                     "a_us": float(np.exp(center(la))), "b_us": float(np.exp(center(lb))), "pairs": len(d)}
        boots.append(boot)
        weights.append(s["weight"])
        logs[name] = (la, lb)
    if score_fn is None:
        w = np.asarray(weights) / sum(weights)
        est = sum(wi * math.log1p(p["effect"]) for wi, p in zip(w, per.values()))
        boot = sum(wi * bt for wi, bt in zip(w, boots))
        return {"effect": math.expm1(est), "ci": _ci(boot), "shapes": per, "instances": inst}
    est = math.log(score_fn({k: v["a_us"] for k, v in per.items()}) / score_fn({k: v["b_us"] for k, v in per.items()}))
    samples = {}
    for name, (la, lb) in logs.items():
        idx = _boot_index(len(la), inst, n_boot, rng)
        samples[name] = (np.exp(center(la[idx], axis=1)), np.exp(center(lb[idx], axis=1)))
    boot = np.array([math.log(score_fn({k: float(v[0][i]) for k, v in samples.items()}) /
                              score_fn({k: float(v[1][i]) for k, v in samples.items()})) for i in range(n_boot)])
    return {"effect": math.expm1(est), "ci": _ci(boot), "shapes": per, "instances": inst, "score": "task"}


def decide(cmp, noise=None, band_floor=0.005, placement_tol=0.005):
    """Keep rule.

    NOISY   identical controls differ by more than placement_tol with confidence: the job measured nothing.
    KEEP    score-effect CI above zero and no shape clearly slower than its noise band.
    MIXED   score-effect CI above zero but some shape clearly slower.
    SLOWER  score-effect CI below zero.
    NO-GAIN anything else (no size limit on gains, but they must be resolved).
    """
    if noise is not None:
        lo, hi = noise["ci"]
        if lo > placement_tol or hi < -placement_tol:
            return NOISY, f"controls differ {noise['effect']:+.2%} [{lo:+.2%}, {hi:+.2%}]"
    slower = []
    for name, p in cmp["shapes"].items():
        band = band_floor
        if noise is not None and name in noise["shapes"]:
            lo, hi = noise["shapes"][name]["ci"]
            band = max(band_floor, (hi - lo) / 2)
        if p["ci"][1] < 0 and p["effect"] < -band:
            slower.append(name)
    lo, hi = cmp["ci"]
    if lo > 0 and not slower:
        return KEEP, ""
    if lo > 0:
        return MIXED, "slower on " + ", ".join(slower)
    if hi < 0:
        return SLOWER, ""
    return NO_GAIN, "effect not resolved from zero"
