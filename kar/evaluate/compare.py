"""Output comparison.

Errors are measured against the task's exact (fp64) reference when it has one. Two tests apply: every element within
atol + rtol * |exact|, and the candidate's mean (and max) error at most a set multiple of the error of the task's own
reference, which runs at the precision the rules prescribe. The second test is what catches a kernel that computes at
lower precision than allowed but still lands inside a loose tolerance.
"""
import torch


def tensors(out):
    if torch.is_tensor(out):
        return [out]
    if isinstance(out, (tuple, list)) and out and all(torch.is_tensor(o) for o in out):
        return list(out)
    raise TypeError(f"output must be a tensor or a tuple of tensors, got {type(out).__name__}")


def clone_args(args):
    return tuple(a.clone() if torch.is_tensor(a) else a for a in args)


def refresh(args, fresh):
    """Copy a new input set into existing input buffers (same pointers, new contents)."""
    if len(args) != len(fresh):
        raise RuntimeError("input generator returned a different number of inputs")
    for a, f in zip(args, fresh):
        if torch.is_tensor(a):
            if a.shape != f.shape or a.dtype != f.dtype:
                raise RuntimeError(f"input changed shape or dtype between seeds: {tuple(a.shape)} -> {tuple(f.shape)}")
            a.copy_(f)
        elif a != f:
            raise RuntimeError("a non-tensor input changed between seeds")


def changed_input(args, snap):
    """Index of the first input the call modified, or None."""
    for i, (a, s) in enumerate(zip(args, snap)):
        if torch.is_tensor(a) and not torch.equal(a, s):
            return i
    return None


def poison(outs):
    for o in outs:
        o.fill_(float("nan") if o.is_floating_point() else -1)


def judge(outs, ref_outs, exact_outs, tol):
    """(ok, reason, metrics) for candidate outputs against the references."""
    if len(outs) != len(ref_outs):
        return False, f"{len(outs)} outputs, expected {len(ref_outs)}", []
    metrics = []
    for i, (o, r) in enumerate(zip(outs, ref_outs)):
        if tuple(o.shape) != tuple(r.shape):
            return False, f"output {i} has shape {tuple(o.shape)}, expected {tuple(r.shape)}", metrics
        if o.dtype != r.dtype:
            return False, f"output {i} has dtype {o.dtype}, expected {r.dtype}", metrics
        e = (exact_outs[i] if exact_outs is not None else r).double()
        od = o.double()
        m = {"numel": o.numel()}
        metrics.append(m)
        if not o.numel():
            continue
        if not bool(torch.isfinite(od).all()):
            return False, f"output {i} has non-finite values", metrics
        d = (od - e).abs()
        bad = int((d > tol["atol"] + tol["rtol"] * e.abs()).sum())
        m.update(max_abs=float(d.max()), mean_abs=float(d.mean()), n_bad=bad)
        if exact_outs is not None:
            dr = (r.double() - e).abs()
            m.update(ref_max_abs=float(dr.max()), ref_mean_abs=float(dr.mean()))
            if m["ref_mean_abs"] > 0:
                m["mean_ratio"] = m["mean_abs"] / m["ref_mean_abs"]
                m["max_ratio"] = m["max_abs"] / max(m["ref_max_abs"], 1e-300)
        if bad:
            return False, (f"output {i}: {bad} of {o.numel()} elements outside atol {tol['atol']} + rtol "
                           f"{tol['rtol']} (max error {m['max_abs']:.3g})"), metrics
        if m.get("mean_ratio", 0) > tol["mean_err_ratio"]:
            return False, (f"output {i}: mean error {m['mean_ratio']:.2f}x the reference's (limit "
                           f"{tol['mean_err_ratio']}): lower precision than the task allows"), metrics
        if m.get("max_ratio", 0) > tol["max_err_ratio"]:
            return False, (f"output {i}: max error {m['max_ratio']:.2f}x the reference's (limit "
                           f"{tol['max_err_ratio']}): lower precision than the task allows"), metrics
    return True, "", metrics
