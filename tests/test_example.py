"""The CUDA example handed to agents must pass our own lint and, on a GPU, compile and run (an agent found that an
earlier version passed -O3, which NVRTC rejects)."""
import os
import sys

import pytest

from kar.evaluate import lint
from kar.run.session import sol_of
from kar.task import spec

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kar", "agents", "prompts")


def test_example_passes_lint():
    r = lint.lint(os.path.join(EX, "nvrtc_example.py"), spec.DEFAULT_RULES)
    assert r["ok"], r


def test_example_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    sys.path.insert(0, EX)
    try:
        import nvrtc_example as e
    finally:
        sys.path.remove(EX)
    x = torch.randn(100003, device="cuda").to(torch.bfloat16)
    assert torch.equal(e.kernel_fn(x), (2 * x.float()).to(torch.bfloat16))


def test_example_runs_from_a_fresh_thread_and_under_graph_capture():
    """A caller may use the op from a thread that never touched CUDA, or capture it in a graph."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    import threading
    sys.path.insert(0, EX)
    try:
        import nvrtc_example as e
    finally:
        sys.path.remove(EX)
    x = torch.randn(4099, device="cuda").to(torch.bfloat16)
    out = {}

    def call():
        try:
            out["y"] = e.kernel_fn(x)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            out["err"] = ex

    t = threading.Thread(target=call)
    t.start()
    t.join()
    assert "err" not in out, out.get("err")
    assert torch.equal(out["y"], (2 * x.float()).to(torch.bfloat16))
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        e.kernel_fn(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = e.kernel_fn(x)
    x.copy_(torch.randn(4099, device="cuda").to(torch.bfloat16))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, (2 * x.float()).to(torch.bfloat16))


def test_sol_caps_shapes_below_their_floor():
    rows = [{"weight": 1, "floor_us": 100, "time_us": 96}, {"weight": 1, "floor_us": 100, "time_us": 200}]
    assert abs(sol_of(rows) - 0.5 ** 0.5) < 1e-12
