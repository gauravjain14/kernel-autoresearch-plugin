"""A team run on the GPU without agents: the commands an agent would use, end to end, on the RMSNorm toy task.

Covers: the scheduler handing out a plan lane and two build lanes; a lane's own test run on the GPU; a test the
reference fails being refused; a registered test gating every later submit (a kernel without eps passes the task's
checks on random data and is turned away by the team's all-zero-rows test); a known-good kernel kept; the board.

Takes a few minutes: KAR_GPU_TESTS=1 python3 -m pytest tests/test_gpu_team.py
"""
import json
import os
import shutil

import pytest

from kar.run import commands
from kar.run.session import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(ROOT, "tasks", "toy", "rmsnorm")

ZERO_ROWS = '''import torch


def test(kernel_fn, reference, make, shapes, close):
    cases = []
    for s in shapes:
        x, w = make(seed=99, **s["args"])
        x = x.clone()
        x[0] = 0
        ok, why = close(kernel_fn(x, w), reference(x, w))
        cases.append((f"{s['name']}: row 0 all zeros", ok, why))
    return cases
'''
WRONG = '''def test(kernel_fn, reference, make, shapes, close):
    x, w = make(seed=1, **shapes[0]["args"])
    return [("output equals input", close(kernel_fn(x, w), x)[0], "")]
'''
NO_EPS = '''import torch


def kernel_fn(x, w):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True))
    return (xf * rstd * w.float()).to(x.dtype)
'''


def _gpu():
    if os.environ.get("KAR_GPU_TESTS") != "1":
        pytest.skip("set KAR_GPU_TESTS=1 to run (minutes on a GPU)")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")


def _do(run, lane, *argv):
    return commands.execute(run.root, list(argv), cwd=run.w(), lane=lane)


def test_team_commands_on_the_gpu(tmp_path):
    _gpu()
    run = Run.create(TASK, str(tmp_path / "run"), lanes=2, survey=0, target_sol=0.99, audit="off",
                     log=lambda *a: None)
    plan, _ = run.assign("t")
    assert plan["role"] == "plan"
    with open(run.w("lanes", plan["id"], "plan.json"), "w") as f:
        json.dump({"hypotheses": [
            {"title": "no eps", "target": "all", "mechanism": "m", "evidence": "e", "predicted_gain_pct": 400,
             "arithmetic": "a", "bound_test": "b", "structural": True},
            {"title": "the known-good kernel", "target": "all", "mechanism": "m", "evidence": "e",
             "predicted_gain_pct": 300, "arithmetic": "a", "bound_test": "b"}]}, f)
    out, code = _do(run, plan["id"], "plan", f"lanes/{plan['id']}/plan.json")
    assert code == 0, out
    a, _ = run.assign("t")
    b, _ = run.assign("t")
    la, lb = a["id"], b["id"]
    for name, text in (("zero_rows.py", ZERO_ROWS), ("wrong.py", WRONG)):
        with open(run.w("lanes", la, name), "w") as f:
            f.write(text)

    out, code = _do(run, la, "test", f"lanes/{la}/zero_rows.py", "best")
    assert code == 0 and "all 10 cases pass" in out, out
    out, code = _do(run, la, "register", f"lanes/{la}/wrong.py", "output equals input")
    assert code == 1 and "reference fails it" in out, out
    out, code = _do(run, la, "register", f"lanes/{la}/zero_rows.py", "all-zero rows stay finite")
    assert code == 0 and "registered as t001" in out, out
    assert os.path.exists(run.w("tests", "t001.py"))

    with open(run.w("lanes", la, "cand.py"), "w") as f:
        f.write(NO_EPS)
    out, code = _do(run, la, "check", f"lanes/{la}/cand.py")
    assert code == 0, out                                   # random inputs: the task's own checks pass it
    out, code = _do(run, la, "submit", f"lanes/{la}/cand.py")
    assert code == 1 and "registered tests" in out and "NaN" in out, out

    shutil.copy(os.path.join(TASK, "good_nvrtc.py"), run.w("lanes", lb, "cand.py"))
    out, code = _do(run, lb, "submit", f"lanes/{lb}/cand.py")
    assert code == 0 and "KEPT" in out, out
    assert "registered test t001" in out          # lane B heard about A's test with its next command
    out, _ = _do(run, la, "board")
    assert "Messages for you" in out and "new best" in out
    st = run.state
    assert st["best"] != "c0000" and st["best_sol"] > 0.5
    assert run.ledger.verify() and all(v == "ok" for v in run.ledger.verify().values())


def test_a_slower_correct_fix_replaces_a_faster_wrong_best(tmp_path):
    """The auditor proves the best wrong with a registered test; the fix is slower; correct comes before fast."""
    _gpu()
    good = open(os.path.join(TASK, "good_nvrtc.py")).read()
    assert " + 1e-6f" in good
    fast_wrong = good.replace(" + 1e-6f", "")               # NaN on an all-zero row; random inputs never show it
    slow_right = good + ("\n\n_fast = kernel_fn\n\n\ndef kernel_fn(x, w):\n    y = _fast(x, w)\n"
                         "    for _ in range(6):\n        x.float().sum()\n    return y\n")
    run = Run.create(TASK, str(tmp_path / "run"), lanes=2, survey=0, audit="every", log=lambda *a: None)
    plan, _ = run.assign("t")
    with open(run.w("lanes", plan["id"], "plan.json"), "w") as f:
        json.dump({"hypotheses": [
            {"title": "fused kernel", "target": "all", "mechanism": "m", "evidence": "e", "predicted_gain_pct": 300,
             "arithmetic": "a", "bound_test": "b", "structural": True}]}, f)
    out, code = _do(run, plan["id"], "plan", f"lanes/{plan['id']}/plan.json")
    assert code == 0, out
    a, _ = run.assign("t")
    with open(run.w("lanes", a["id"], "cand.py"), "w") as f:
        f.write(fast_wrong)
    out, code = _do(run, a["id"], "submit", f"lanes/{a['id']}/cand.py")
    assert code == 0 and "KEPT" in out, out
    wrong_best = run.state["best"]

    au, _ = run.assign("t")
    assert au["role"] == "audit", au
    with open(run.w("lanes", au["id"], "zero_rows.py"), "w") as f:
        f.write(ZERO_ROWS)
    out, code = _do(run, au["id"], "register", f"lanes/{au['id']}/zero_rows.py", "all-zero rows stay finite")
    assert code == 0 and "FAILS it" in out, out
    out, code = _do(run, au["id"], "audit", "hazard", "no eps under the rsqrt: an all-zero row gives 0 * inf = NaN; "
                    "the reference adds eps and returns zeros. Add the eps.")
    assert code == 0, out
    assert "KNOWN HAZARD in the best" in run.status()[0]

    fx, text = run.assign("t")
    assert fx["role"] == "build" and run._hyp(run.state, fx["hyp"])["kind"] == "fix"
    assert "Correct comes before fast" in run.lane_next(fx["id"])
    with open(run.w("lanes", fx["id"], "cand.py"), "w") as f:
        f.write(slow_right)
    out, code = _do(run, fx["id"], "submit", f"lanes/{fx['id']}/cand.py")
    assert code == 0 and "KEPT" in out and "it is kept" in out, out
    st = run.state
    assert st["best"] != wrong_best and st["faster_but_wrong"][0]["cid"] == wrong_best
    assert st["faster_but_wrong"][0]["faster_pct"] > 0
    text = run.status()[0]
    assert "KNOWN HAZARD" not in text and f"{wrong_best} is" in text and "wrong on some inputs" in text
