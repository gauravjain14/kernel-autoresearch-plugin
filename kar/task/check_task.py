"""The task checks `./kar-task validate` runs, as a worker job on whatever backend the task will run on (an H200-only
task is validated on an H200): the input builder is deterministic and uses its seed, inputs are CUDA tensors, the
reference agrees with the float64 `exact` within the tolerance, the baseline agrees with the reference, the op model
counts no more than the inputs and outputs hold, and there are enough nearby shapes and edge cases."""
import importlib


def check_task(t):
    import torch
    out = {"problems": [], "warnings": [], "shapes": []}
    R = t.attr(t.reference)
    X = t.attr(t.exact) if t.exact else None
    M = t.attr(t.inputs)
    t.activate()
    B = getattr(importlib.import_module(t.baseline[:-3]), t.entry)
    by = t.attr(t.opmodel["bytes"]) if t.opmodel.get("bytes") else None
    fl = t.attr(t.opmodel["flops"]) if t.opmodel.get("flops") else None
    S = t.attr(t.score) if t.score else None
    sc, nb = t.shape_list("scored"), t.shape_list("nearby")
    if len(sc) < 2:
        out["warnings"].append("fewer than 2 scored shapes")
    if len(nb) < 3:
        out["problems"].append("fewer than 3 nearby shapes: kernels right only on the scored shapes would pass")
    if not t.exact:
        out["problems"].append("no exact (float64) reference: the precision test would be off")
    if not by:
        out["problems"].append("no op model: no speed-of-light floor")
    tol = t.tolerance
    base_times = {}
    for s in sc + nb:
        row = {"shape": s["name"], "kind": s["kind"]}
        try:
            a, a2, a3 = M(seed=1, **s["args"]), M(seed=1, **s["args"]), M(seed=2, **s["args"])
            ts = [x for x in a if torch.is_tensor(x)]
            if not ts or not all(x.is_cuda for x in ts):
                out["problems"].append(f"{s['name']}: inputs must be CUDA tensors")
                continue
            if not all(torch.equal(x, y) for x, y in zip(a, a2) if torch.is_tensor(x)):
                out["problems"].append(f"{s['name']}: make() is not deterministic for a fixed seed")
            if all(torch.equal(x, y) for x, y in zip(a, a3) if torch.is_tensor(x)):
                out["problems"].append(f"{s['name']}: make() ignores the seed")
            r, ex, b = R(*a), (X(*a) if X else None), B(*a)
            rs = [r] if torch.is_tensor(r) else list(r)
            bs = [b] if torch.is_tensor(b) else list(b)
            for i, (ri, bi) in enumerate(zip(rs, bs)):
                if not torch.isfinite(ri.float()).all() and "edge" not in s["args"]:
                    out["problems"].append(f"{s['name']}: reference output {i} has NaN/inf")
                err = (bi.double() - ri.double()).abs()
                lim = tol["atol"] + tol["rtol"] * ri.double().abs()
                fin = torch.isfinite(ri.double())
                if (err[fin] > lim[fin]).any() or not torch.equal(torch.isnan(bi.double()), torch.isnan(ri.double())):
                    out["problems"].append(f"{s['name']}: baseline output {i} differs from the reference beyond tolerance")
            if ex is not None:
                es = [ex] if torch.is_tensor(ex) else list(ex)
                for i, (ri, ei) in enumerate(zip(rs, es)):
                    fin = torch.isfinite(ei.double())
                    err = (ri.double() - ei.double()).abs()[fin]
                    lim = (tol["atol"] + tol["rtol"] * ei.double().abs())[fin]
                    row["ref_vs_exact_max"] = float(err.max()) if err.numel() else 0.0
                    if (err > lim).any():
                        out["problems"].append(f"{s['name']}: reference output {i} is outside tolerance of exact; "
                                               f"the reference or the tolerance is wrong")
                    row["out_rms"] = float(ei.double()[fin].pow(2).mean().sqrt()) if fin.any() else 0.0
            if s["kind"] == "scored" and by:
                b_ = by(**s["args"])
                row["bytes"] = b_ if isinstance(b_, dict) else {"read": b_}
                nin = sum(x.numel() * x.element_size() for x in ts)
                nout = sum(x.numel() * x.element_size() for x in rs)
                rd, wr = row["bytes"].get("read", 0), row["bytes"].get("write", 0)
                if rd > 1.01 * nin or wr > 1.01 * nout:
                    out["warnings"].append(f"{s['name']}: op model counts more bytes ({rd:.0f} read, {wr:.0f} write) "
                                           f"than the inputs and outputs hold ({nin}, {nout}); a floor must count "
                                           f"only compulsory traffic")
                if rd < 0.5 * nin:
                    out["warnings"].append(f"{s['name']}: op model reads {rd:.0f} bytes, under half the input bytes "
                                           f"({nin}); is every input really optional?")
                if fl:
                    row["flops"] = fl(**s["args"])
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                B(*a)
                torch.cuda.synchronize()
                e0.record()
                B(*a)
                e1.record()
                torch.cuda.synchronize()
                row["baseline_ms"] = e0.elapsed_time(e1)
                base_times[s["name"]] = row["baseline_ms"] * 1e3
        except Exception as e:  # noqa: BLE001 - every problem becomes a line of the report
            import traceback
            out["problems"].append(f"{s['name']}: {type(e).__name__}: {e}")
            out["traceback"] = traceback.format_exc()[-1500:]
        out["shapes"].append(row)
    if S is not None and base_times and len(base_times) == len(sc):
        try:
            v = float(S(base_times))
            faster = float(S({k: x / 2 for k, x in base_times.items()}))
            out["score_check"] = {"baseline": v, "twice_as_fast": faster}
            if not faster > v:
                out["problems"].append("the score function does not grow when every shape gets faster: it must be "
                                       "higher-is-better over times in microseconds")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"score function raised {type(e).__name__}: {e}")
    if not any(s.get("args", {}).get("edge") for s in nb):
        out["warnings"].append("no nearby shape switches on edge values (args 'edge': masked entries, zeros, extremes, "
                               "the dtype's limits): random inputs alone miss what real callers pass")
    return out
