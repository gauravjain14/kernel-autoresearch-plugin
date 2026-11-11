"""Calibration of the A/B statistics on synthetic timings with a known truth."""
import numpy as np

from kar.evaluate import stats


def synth(rng, effect=0.0, placement_sd=0.01, rep_sd=0.01, pairs=24, inst=4, shapes=4, copies=2):
    """Bench dict: labels a#c and b#c; a is `effect` faster. Each (variant copy, instance) has its own fixed
    placement offset, which is exactly the bias that a per-pair bootstrap alone would miss."""
    out = {"instances": inst, "shapes": {}}
    for s in range(shapes):
        base = rng.uniform(5, 500)
        times = {}
        for v, scale in (("a", 1 / (1 + effect)), ("b", 1.0)):
            for c in range(copies):
                place = rng.normal(0, placement_sd, size=inst)
                t = []
                for r in range(2 * pairs):
                    i = (r // 2) % inst
                    t.append(base * scale * np.exp(place[i] + rng.normal(0, rep_sd)))
                times[f"{v}#{c}"] = t
        out["shapes"][f"s{s}"] = {"weight": 1.0, "times": times}
    return out


A, B = ["a#0", "a#1"], ["b#0", "b#1"]


def test_null_coverage():
    rng = np.random.default_rng(1)
    covered = 0
    n = 200
    for k in range(n):
        cmp = stats.compare(synth(rng), A, B, n_boot=1000, seed=k)
        covered += cmp["ci"][0] <= 0 <= cmp["ci"][1]
    assert covered / n >= 0.88, covered / n


def test_null_without_clustering_is_overconfident():
    """The reason for the instance-clustered bootstrap: ignoring placement under-covers badly."""
    rng = np.random.default_rng(2)
    covered = 0
    n = 200
    for k in range(n):
        b = synth(rng)
        b["instances"] = 1
        cmp = stats.compare(b, A, B, n_boot=1000, seed=k)
        covered += cmp["ci"][0] <= 0 <= cmp["ci"][1]
    assert covered / n < 0.8, covered / n


def test_detects_real_gain():
    rng = np.random.default_rng(3)
    keep = 0
    for k in range(100):
        cmp = stats.compare(synth(rng, effect=0.03, placement_sd=0.005), A, B, n_boot=1000, seed=k)
        keep += stats.decide(cmp)[0] == stats.KEEP
    assert keep >= 90, keep


def test_slower_and_noisy():
    rng = np.random.default_rng(4)
    cmp = stats.compare(synth(rng, effect=-0.05, placement_sd=0.002), A, B, n_boot=1000)
    assert stats.decide(cmp)[0] == stats.SLOWER
    noise = {"effect": 0.02, "ci": [0.015, 0.025], "shapes": {}}
    assert stats.decide(cmp, noise)[0] == stats.NOISY


def test_effect_sign_and_size():
    rng = np.random.default_rng(5)
    cmp = stats.compare(synth(rng, effect=0.10, placement_sd=0.0, rep_sd=0.001), A, B, n_boot=500)
    assert abs(cmp["effect"] - 0.10) < 0.01
