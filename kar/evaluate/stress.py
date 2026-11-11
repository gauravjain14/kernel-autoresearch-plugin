"""Stress suite for a kernel that already passed the correctness suite: the hazards that suite does not reach.

1. More seeds: several extra fresh inputs per shape (scored and nearby), eager.
2. Concurrent calls: the kernel called on several streams at once, each with its own inputs, outputs checked per
   stream. This is how a serving engine uses an op. It catches a static global workspace that two in-flight calls
   share, cross-block waits that need every block of one call resident (two calls compete for the SMs, so such a
   kernel deadlocks; the job's phase time limit turns the hang into a rejection), and work that assumes it owns the
   GPU.
3. Back-to-back calls with no synchronization between them on one stream, every output kept as returned and
   checked at the end (catches an output buffer shared between calls, and state the previous call left behind).

4. A call from a fresh Python thread that has never touched CUDA (catches a kernel that launches through the driver
   API without making the context current: it works on the importing thread and nowhere else).

Outputs are kept exactly as the kernel returned them, not copied: a caller may hold several results at once.

What this cannot test: reliance on the hardware dispatching blocks in index order. That is undocumented behaviour;
the auditor reads the code for it.
"""
import torch

from kar.evaluate import compare as C
from kar.evaluate.check import Fail, _judge


def _expect_ok(label, a, snap, outs, R, tol):
    return _judge(label, a, snap, outs, R, tol)


def stress_shape(fn, R, shape, seed, tol, guard, phase, seeds=4, streams=3, rounds=2, chain=4):
    name = shape["name"]
    metrics = []
    phase(f"shape {name} seeds")
    for k in range(seeds):
        a = R.make(seed=seed + 101 + k, **shape["args"])
        snap = C.clone_args(a)
        outs = [o.clone() for o in C.tensors(fn(*a))]
        torch.cuda.synchronize()
        metrics += _expect_ok(f"seed {k + 1} of {seeds}", a, snap, outs, R, tol)
        guard()
    phase(f"shape {name} concurrent streams")
    ss = [torch.cuda.Stream() for _ in range(streams)]
    for rnd in range(rounds):
        args = [R.make(seed=seed + 211 + 17 * rnd + i, **shape["args"]) for i in range(streams)]
        snaps = [C.clone_args(a) for a in args]
        torch.cuda.synchronize()
        outs = [None] * streams
        for i, s in enumerate(ss):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                outs[i] = list(C.tensors(fn(*args[i])))
        torch.cuda.synchronize()
        for i in range(streams):
            metrics += _expect_ok(f"concurrent call {i + 1} of {streams} on its own stream (round {rnd + 1})",
                                  args[i], snaps[i], outs[i], R, tol)
        guard()
    phase(f"shape {name} back-to-back")
    args = [R.make(seed=seed + 307 + i, **shape["args"]) for i in range(chain)]
    snaps = [C.clone_args(a) for a in args]
    torch.cuda.synchronize()
    outs = [list(C.tensors(fn(*a))) for a in args]
    torch.cuda.synchronize()
    for i in range(chain):
        metrics += _expect_ok(f"call {i + 1} of {chain} back to back, no sync between", args[i], snaps[i], outs[i],
                              R, tol)
    guard()
    phase(f"shape {name} fresh thread")
    import threading
    a = R.make(seed=seed + 401, **shape["args"])
    snap = C.clone_args(a)
    torch.cuda.synchronize()
    box = {}

    def call():
        try:
            box["out"] = list(C.tensors(fn(*a)))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - reported as the stress failure below
            box["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join()
    if "err" in box:
        raise Fail(f"a call from a fresh thread raised {box['err']}")
    metrics += _expect_ok("a call from a fresh thread", a, snap, box["out"], R, tol)
    guard()
    worst = {}
    for m in metrics:
        for key in ("max_abs", "mean_ratio", "max_ratio"):
            if key in m:
                worst[key] = max(worst.get(key, 0.0), m[key])
    return worst


def run_stress(fn, R, shapes, seed, tol, guard, phase):
    done = []
    for i, s in enumerate(shapes):
        where = f"{s['kind']} shape {s['name']}"
        try:
            worst = stress_shape(fn, R, s, seed + 1000 * i, tol, guard, phase)
        except Fail as e:
            return {"status": "reject", "reason": f"stress, {where}: {e}", "shapes": done}
        except Exception as e:  # noqa: BLE001 - any exception from a candidate is a failure
            return {"status": "reject", "reason": f"stress, {where}: raised {type(e).__name__}: {e}", "shapes": done}
        done.append({"shape": s["name"], "kind": s["kind"], **worst})
    return {"status": "accept", "shapes": done}
