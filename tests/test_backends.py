"""The remote backends' moving parts, without a GPU or a network: a job is packed into a bundle, unpacked on a
"remote" folder, run by the real supervisor (path substitution, per-phase time limits, the result), packed back and
read as a result, exactly as the Modal and ssh backends do. Also the backend spec parser and the slot gate."""
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

import pytest

from kar import backends
from kar.backends import gate, remote

FAKE_OK = ("import json, os, sys, time; job = json.load(open(sys.argv[-1])); d = os.path.dirname(sys.argv[-1]); "
           "open(os.path.join(d, 'progress'), 'a').write(f'{time.time()}\\tshape x\\n'); "
           "assert job['task_root'].endswith('/task') and os.path.isdir(job['task_root']), job['task_root']; "
           "assert os.path.isfile(job['path']), job['path']; "
           "json.dump({'status': 'accept', 'seen': open(job['path']).read()}, open(os.path.join(d, 'result.json'), 'w'))")
FAKE_HANG = ("import os, sys, time; d = os.path.dirname(sys.argv[-1]); "
             "open(os.path.join(d, 'progress'), 'a').write(f'{time.time()}\\tshape slow\\n'); time.sleep(60)")


def _remote_roundtrip(tmp_path, code, limits):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("name = 'x'\n")
    cand = tmp_path / "cand.py"
    cand.write_text("kernel = 1\n")
    job = {"mode": "check", "task_root": str(task), "path": str(cand), "seed": 1}
    bundle, rewritten = remote.pack(job, limits=limits, total_s=30)
    assert rewritten["path"].startswith("@R@/files/") and rewritten["task_root"] == "@R@/task"
    root = tmp_path / "remote"
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        tf.extractall(root)
    assert (root / "kar" / "evaluate" / "worker.py").exists() and (root / "task" / "task.toml").exists()
    spec = json.loads((root / "spec.json").read_text())
    spec["root"] = str(root)
    spec["cmd"] = [sys.executable, "-c", code, "@R@/job/job.json"]           # a fake worker instead of the real one
    (root / "spec.json").write_text(json.dumps(spec))
    subprocess.run([sys.executable, str(root / "kar" / "backends" / "supervise.py"), str(root / "spec.json")],
                   check=True, timeout=60)
    local = tmp_path / "job_local"
    remote.unpack(remote.pack_result(str(root)), str(local))
    return remote.result(str(local))


def test_a_remote_job_round_trip(tmp_path):
    res = _remote_roundtrip(tmp_path, FAKE_OK, {"shape": 20})
    assert res["status"] == "accept" and res["seen"] == "kernel = 1\n", res
    assert os.path.exists(os.path.join(res["job_dir"], "supervisor.json"))


def test_the_supervisor_kills_a_phase_over_its_limit(tmp_path):
    t0 = time.time()
    res = _remote_roundtrip(tmp_path, FAKE_HANG, {"shape": 2})
    assert res["status"] == "timeout" and "shape slow" in res["reason"] and time.time() - t0 < 30, res


def test_unpack_never_writes_outside_the_job_folder(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"x"
        for name in ("../escape.txt", "ok.txt"):
            ti = tarfile.TarInfo(name)
            ti.size = 1
            tf.addfile(ti, io.BytesIO(data))
    remote.unpack(buf.getvalue(), str(tmp_path / "job"))
    assert (tmp_path / "job" / "ok.txt").exists() and not (tmp_path / "escape.txt").exists()


def test_backend_specs():
    assert backends.parse_spec("local") == {"kind": "local", "gpu": 0}
    assert backends.parse_spec("modal:H200:3", ["image=lmsysorg/sglang:dev", "pip=a,b", "env.X=1"]) == {
        "kind": "modal", "gpu": "H200", "max_boxes": 3, "image": "lmsysorg/sglang:dev", "pip": ["a", "b"],
        "env": {"X": "1"}}
    assert backends.parse_spec("ssh:me@box:0,1") == {"kind": "ssh", "host": "me@box", "gpus": [0, 1]}
    with pytest.raises(ValueError):
        backends.parse_spec("cloud:x")


def test_any_slot_caps_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "LOCKS", str(tmp_path))
    held, peak, lock = [], [0], threading.Lock()

    def work():
        with gate.any_slot("t", 2):
            with lock:
                held.append(1)
                peak[0] = max(peak[0], len(held))
            time.sleep(0.3)
            with lock:
                held.pop()

    ts = [threading.Thread(target=work) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 2
