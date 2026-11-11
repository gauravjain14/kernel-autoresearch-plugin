"""Timing.

Per shape, every variant (each copy of each candidate, each control copy) captures `instances` CUDA graphs, each on
its own freshly allocated copy of the inputs, so each instance sits at a different memory placement. Captures of all
(variant, instance) pairs are interleaved in a shuffled order. Identical code
at a fixed placement can read 1-2% apart from itself (measured on the first run of this bench), so one placement per
variant would turn placement into a fake effect. Rep pairs rotate through the instances.

Every rep makes a fresh input set and copies it into the active instance of every variant, then times each replay
after a cold-L2 read flush. Variants run in a random order on even reps and in the reverse order on odd reps, so each
(even, odd) pair sees every variant in opposite positions. Outputs are checked on the first and last rep.

mode="eager" is the same design with direct calls instead of graph replays, for work that cannot be captured: each
(variant, instance) keeps its own inputs, and the timed span is one Python call of the kernel.
"""
import random
import time

import torch

from kar.evaluate import compare as C
from kar.evaluate.check import capture


def l2_bytes():
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(getattr(p, "L2_cache_size", 0) or 0) or (64 << 20)


def gc_free():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def run_bench(variants, R, shapes, reps, seed, tol, instances=4, flush_bytes=0, warmup=3, spin_s=0.5,
              guard=lambda: None, phase=lambda name: None, unchecked=(), mode="graph"):
    """variants: list of (label, fn). Returns {"shapes": {name: {"weight", "times": {label: [us per rep]}}}, ...};
    rep r used instance (r // 2) % instances of every variant. Variants whose base label (before '#') is in
    `unchecked` skip the output check: bound probes, which are incomplete on purpose and are never candidates."""
    if reps < 2 or reps % 2:
        raise ValueError("reps must be even and >= 2")
    inst = max(1, int(instances))
    fb = flush_bytes or max(4 * l2_bytes(), 128 << 20)
    flush = torch.empty(fb // 4, dtype=torch.float32, device="cuda").normal_()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    t_end = time.perf_counter() + spin_s  # bring clocks up from idle before anything is timed
    while time.perf_counter() < t_end:
        flush.sum()
        torch.cuda.synchronize()

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)

    def time_empty():
        flush.sum()
        e0.record()
        if mode == "eager":
            tmp.add_(1)
        else:
            ge.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3

    for _ in range(10):
        time_empty()
    empty = sorted(time_empty() for _ in range(31))[15]

    rng = random.Random(seed)
    labels = [label for label, _ in variants]
    out = {"empty_us": empty, "flush_bytes": fb, "reps": reps, "instances": inst,
           "order": "random on even reps, reversed on odd reps", "shapes": {}}
    def one_shape(s, s0, inst):
        base = R.make(seed=s0, **s["args"])
        # capture (variant, instance) pairs in a shuffled order so no variant owns a contiguous address range
        # (measured: the second-loaded module read 0.3-0.5% faster than the first when each owned one range)
        cap = {label: [None] * inst for label in labels}
        slots = [(vi, i) for vi in range(len(variants)) for i in range(inst)]
        rng.shuffle(slots)
        for vi, i in slots:
            label, fn = variants[vi]
            a = C.clone_args(base)
            if mode == "eager":
                for _ in range(warmup):
                    fn(*a)
                torch.cuda.synchronize()
                cap[label][i] = [a, None, None]
            else:
                g, o = capture(fn, a, warm=warmup)
                cap[label][i] = [a, g, o]
        del base
        evs = {lb: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for lb in labels}
        times = {lb: [] for lb in labels}
        order = labels[:]
        for r in range(reps):
            i = (r // 2) % inst
            fresh = R.make(seed=s0 + 1 + r, **s["args"])
            for lb in labels:
                C.refresh(cap[lb][i][0], fresh)
            if r % 2 == 0:
                order = labels[:]
                rng.shuffle(order)
            else:
                order = order[::-1]
            for lb in order:
                flush.sum()
                b, e = evs[lb]
                if mode == "eager":
                    fn = dict(variants)[lb]
                    b.record()
                    cap[lb][i][2] = C.tensors(fn(*cap[lb][i][0]))
                    e.record()
                else:
                    b.record()
                    cap[lb][i][1].replay()
                    e.record()
            torch.cuda.synchronize()
            for lb in labels:
                b, e = evs[lb]
                times[lb].append(b.elapsed_time(e) * 1e3)
            if r in (0, reps - 1):
                ref = C.tensors(R.ref(*fresh))
                for lb in labels:
                    if lb.split("#")[0] in unchecked:
                        continue
                    ok, why, _ = C.judge(cap[lb][i][2], ref, None, tol)
                    if not ok:
                        raise RuntimeError(f"variant {lb} gave a wrong result during the bench ({s['name']}, rep {r}): "
                                           f"{why}")
            guard()
        return times

    for si, s in enumerate(shapes):
        phase(f"bench {s['name']}")
        s0 = seed + 7919 * (si + 1)
        n = inst
        while True:
            # every captured copy keeps its temporaries alive; with large ones, variants x instances does not fit.
            # Fewer instances keeps the design (all variants still share each placement), so retry smaller. The
            # cleanup happens after the except block, when the failed attempt's frames (and graphs) are released.
            try:
                times = one_shape(s, s0, n)
                break
            except torch.cuda.OutOfMemoryError:
                pass
            gc_free()
            if n == 1:
                total = torch.cuda.mem_get_info()[1]
                raise RuntimeError(
                    f"out of GPU memory timing {s['name']} with one captured copy of each of {len(labels)} "
                    f"variants ({total >> 20} MiB card): a variant keeps too much memory alive per call "
                    f"(temporaries stay allocated inside a captured graph). Compute in smaller pieces, reuse "
                    f"buffers, or time fewer variants at once.")
            n = max(1, n // 2)
            phase(f"bench {s['name']} (out of memory; retrying with {n} instance{'s' if n > 1 else ''})")
        out["shapes"][s["name"]] = {"weight": s["weight"], "times": times, "instances": n}
        gc_free()
    return out
