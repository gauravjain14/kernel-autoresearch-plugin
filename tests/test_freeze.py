"""A job refuses to run when the evaluator or the task changed after the run was frozen."""
import json
import os

from conftest import TOY
from kar.evaluate import worker
from kar.evaluate.manifest import evaluator_sha
from kar.task import spec


def run_job(tmp_path, **kw):
    job = {"mode": "smoke", "label": "x", "task_root": os.path.join(TOY, "rmsnorm"), "seed": 1, "path": "/dev/null",
           **kw}
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    worker.main(str(jp))
    return json.loads((tmp_path / "result.json").read_text())


def test_evaluator_changed(tmp_path):
    assert run_job(tmp_path, evaluator_sha="0" * 64)["status"] == "evaluator_changed"


def test_task_changed(tmp_path):
    res = run_job(tmp_path, evaluator_sha=evaluator_sha(), task_sha="0" * 64)
    assert res["status"] == "task_changed"


def test_sha_is_stable():
    assert evaluator_sha() == evaluator_sha()
    t = spec.load(os.path.join(TOY, "gemv"))
    assert t.sha() == t.sha()
