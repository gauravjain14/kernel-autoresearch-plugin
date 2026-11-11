"""Agent-written tests: the team's own correctness probes, run on the GPU like a kernel.

A test file defines

    def test(kernel_fn, reference, make, shapes, close):
        ...
        return [(case_name, ok, detail), ...]

`shapes` is every scored and nearby shape of the task; `make(seed, **args)` builds inputs; `close(out, expected)`
compares an output with the reference's under the task tolerance, with NaN and inf required in the same places,
and returns (ok, why). A test edits inputs the task's own checks never use (masked entries, zeros, extremes, odd
sizes) and compares with the reference. The file is linted with the same rules as a kernel.

A test that the task's reference passes can be registered for the run; from then on every submit must pass it.
"""
import torch

from kar.evaluate import compare as C


def make_close(tol):
    def close(out, expected):
        outs, exps = C.tensors(out), C.tensors(expected)
        if len(outs) != len(exps):
            return False, f"{len(outs)} outputs, expected {len(exps)}"
        for i, (o, e) in enumerate(zip(outs, exps)):
            if tuple(o.shape) != tuple(e.shape):
                return False, f"output {i}: shape {tuple(o.shape)}, expected {tuple(e.shape)}"
            o, e = o.double(), e.double()
            on, en = torch.isnan(o), torch.isnan(e)
            if not torch.equal(on, en):
                n = int((on & ~en).sum())
                return False, f"output {i}: NaN in {n} places where the reference has none" if n else \
                    f"output {i}: the reference has NaN where this output does not"
            oi, ei = torch.isinf(o), torch.isinf(e)
            if not torch.equal(oi, ei) or not torch.equal(o[ei], e[ei]):
                return False, f"output {i}: infinities differ from the reference's"
            fin = ~(en | ei)
            err = (o[fin] - e[fin]).abs()
            lim = tol["atol"] + tol["rtol"] * e[fin].abs()
            bad = int((err > lim).sum())
            if bad:
                return False, f"output {i}: {bad} values outside atol {tol['atol']} + rtol {tol['rtol']} (max error " \
                              f"{float(err.max()):.3g})"
        return True, ""
    return close


def run_tests(fn, tests, R, shapes, tol, guard, phase):
    """tests: [(name, test_fn)]. -> {"status": accept|reject, "reason", "cases": [...]}"""
    close = make_close(tol)
    cases = []
    for name, tf in tests:
        phase(f"shape test {name}")
        try:
            got = tf(fn, R.ref, R.make, shapes, close)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - a test that raises on the kernel is a failed case
            cases.append({"test": name, "case": "(the test raised)", "ok": False,
                          "detail": f"{type(e).__name__}: {e}"[:500]})
            continue
        if not isinstance(got, (list, tuple)):
            cases.append({"test": name, "case": "(bad test)", "ok": False,
                          "detail": "test() must return a list of (case, ok, detail)"})
            continue
        for item in got:
            try:
                case, ok, detail = item
            except (TypeError, ValueError):
                case, ok, detail = str(item)[:80], False, "malformed case: want (case, ok, detail)"
            cases.append({"test": name, "case": str(case)[:120], "ok": bool(ok), "detail": str(detail)[:500]})
        guard()
    bad = [c for c in cases if not c["ok"]]
    if bad:
        b = bad[0]
        return {"status": "reject", "reason": f"test {b['test']}, case {b['case']}: {b['detail']}", "cases": cases}
    return {"status": "accept", "cases": cases}
