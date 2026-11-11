"""Correctness suite, run on every scored and every nearby shape:

1. eager call on fresh inputs; the inputs must come back unchanged;
2. a second eager call with new data copied into the SAME buffers (catches caching by pointer and state that does
   not reset);
3. CUDA-graph capture, then two replays, each on new data copied into the captured buffers and with the captured
   output filled with NaN first (catches work done outside the graph, or only at capture time). Tasks timed in
   eager mode skip this step: capture is not part of their contract.

Each output is copied on the caller's stream right after the call, so work left running on another stream is caught.
"""
import torch

from kar.evaluate import compare as C


class Fail(Exception):
    pass


def capture(fn, args, warm=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    torch.cuda.synchronize()
    return g, C.tensors(out)


def _expect(R, snap):
    ref = C.tensors(R.ref(*snap))
    exact = C.tensors(R.exact(*snap)) if R.exact is not None else None
    return ref, exact


def _judge(label, args, snap, outs, R, tol):
    changed = C.changed_input(args, snap)
    if changed is not None:
        raise Fail(f"{label}: the call modified input {changed}")
    ref, exact = _expect(R, snap)
    ok, why, m = C.judge(outs, ref, exact, tol)
    if not ok:
        raise Fail(f"{label}: {why}")
    return m


def check_shape(fn, R, shape, seed, tol, guard, graph=True):
    def fresh(k):
        return R.make(seed=seed + k, **shape["args"])

    a = fresh(0)
    metrics = []
    for k, label in enumerate(("eager call 1", "eager call 2 (same buffers, new data)")):
        if k:
            C.refresh(a, fresh(k))
        snap = C.clone_args(a)
        outs = [o.clone() for o in C.tensors(fn(*a))]
        torch.cuda.synchronize()
        metrics += _judge(label, a, snap, outs, R, tol)
        guard()
    C.refresh(a, fresh(2))
    if not graph:                               # an eager task: capture is not part of its contract
        return _worst(metrics)
    g, gout = capture(fn, a)
    for k in range(2):
        C.refresh(a, fresh(3 + k))
        snap = C.clone_args(a)
        C.poison(gout)
        g.replay()
        outs = [o.clone() for o in gout]
        torch.cuda.synchronize()
        metrics += _judge(f"graph replay {k + 1} (new data, output poisoned)", a, snap, outs, R, tol)
        guard()
    return _worst(metrics)


def _worst(metrics):
    worst = {}
    for m in metrics:
        for key in ("max_abs", "mean_ratio", "max_ratio"):
            if key in m:
                worst[key] = max(worst.get(key, 0.0), m[key])
    return worst


def run_check(fn, R, shapes, seed, tol, guard, phase, graph=True):
    done = []
    for i, s in enumerate(shapes):
        phase(f"shape {s['name']}")
        where = f"{s['kind']} shape {s['name']}"
        try:
            worst = check_shape(fn, R, s, seed + 1000 * i, tol, guard, graph=graph)
        except Fail as e:
            return {"status": "reject", "reason": f"{where}: {e}", "shapes": done}
        except Exception as e:  # noqa: BLE001 - any exception from a candidate is a rejection
            return {"status": "reject", "reason": f"{where}: raised {type(e).__name__}: {e}", "shapes": done}
        done.append({"shape": s["name"], "kind": s["kind"], **worst})
    return {"status": "accept", "shapes": done}
