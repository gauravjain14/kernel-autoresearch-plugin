"""What every remote backend shares: packing one job into a self-contained bundle and turning the returned bundle back
into the same result a local job gives.

A bundle is a gzipped tar with
    kar/          the harness package (the worker, the evaluator, the profilers; the frozen evaluator hash is checked
                  by the worker against these very files, so a remote job measures with exactly the frozen code)
    task/         the task folder
    files/NN_*    every candidate, control and test file the job names
    job/job.json  the job, with every path rewritten to "@R@/..." (the supervisor fills in the remote root)
    spec.json     the supervisor's command, limits and options
The returned bundle holds the job folder after the run: result.json, progress, log.txt, supervisor.json, profiler
reports and their exports, saved cubins.
"""
import io
import json
import os
import tarfile
import time

HARNESS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOOT = "import sys; sys.path.insert(0, sys.argv[1]); from kar.evaluate.worker import main; sys.exit(main(sys.argv[2]))"
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}


class BackendError(Exception):
    pass


def _add_dir(tf, src, arc):
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(fn):
            if f.endswith((".pyc", ".ncu-rep", ".nsys-rep", ".sqlite")):
                continue
            full = os.path.join(dp, f)
            tf.add(full, arcname=os.path.join(arc, os.path.relpath(full, src)))


def pack(job, prefix_fn=None, gpu=None, limits=None, total_s=3600, env=None, python="python3"):
    """-> (bundle bytes, the rewritten job). Paths in the job are replaced by @R@ paths inside the bundle."""
    files = {}

    def mapped(path):
        path = os.path.abspath(path)
        if path not in files:
            files[path] = f"files/{len(files):02d}_{os.path.basename(path)}"
        return "@R@/" + files[path]

    j = json.loads(json.dumps(job))
    task_root = os.path.abspath(j["task_root"])
    j["task_root"] = "@R@/task"
    for key in ("path",):
        if j.get(key):
            j[key] = mapped(j[key])
    for v in j.get("variants") or []:
        v["path"] = mapped(v["path"])
    if j.get("tests"):
        j["tests"] = [mapped(t) for t in j["tests"]]
    for path in files:
        if not os.path.isfile(path):
            raise BackendError(f"{path}: the job names a file that does not exist")
    cmd = list(prefix_fn("@R@/job") if prefix_fn else []) + [python, "-E", "-P", "-c", BOOT, "@R@", "@R@/job/job.json"]
    spec = {"cmd": cmd, "limits": limits or {}, "total_s": total_s, "gpu": gpu, "env": env or {}, "post": True}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_dir(tf, os.path.join(HARNESS, "kar"), "kar")
        _add_dir(tf, task_root, "task")
        for path, arc in files.items():
            tf.add(path, arcname=arc)
        for name, obj in (("job/job.json", j), ("spec.json", spec)):
            data = json.dumps(obj, indent=1).encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue(), j


def unpack(blob, job_dir):
    """Extract the returned job folder into the local job folder (only plain files, never outside it)."""
    os.makedirs(job_dir, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = os.path.normpath(m.name)
            if name.startswith("..") or os.path.isabs(name) or not (m.isfile() or m.isdir()):
                continue
            dest = os.path.join(job_dir, name)
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as out:
                out.write(src.read())


def pack_result(root):
    """On the remote side: the job folder as a bundle (used by the Modal function and the ssh runner)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(os.path.join(root, "job"), arcname=".")
    return buf.getvalue()


def result(job_dir, transport_error=None):
    """The same dict local.run returns, from a returned job folder."""
    res = {}
    rp = os.path.join(job_dir, "result.json")
    sp = os.path.join(job_dir, "supervisor.json")
    sup = {}
    if os.path.exists(sp):
        with open(sp) as f:
            sup = json.load(f)
    if os.path.exists(rp) and not sup.get("killed"):
        with open(rp) as f:
            res = json.load(f)
    if transport_error:
        res.update(status="error", reason=f"backend: {transport_error}")
    elif sup.get("killed"):
        res.update(status="timeout", reason=sup["killed"])
    elif not res:
        res.update(status="crashed", reason=f"worker exited {sup.get('returncode')} without a result")
    if res.get("status") in ("timeout", "crashed", "error"):
        lp = os.path.join(job_dir, "log.txt")
        if os.path.exists(lp):
            with open(lp, "rb") as f:
                res["log_tail"] = f.read()[-3000:].decode(errors="replace")
    res.update(wall_s=sup.get("wall_s", 0), smi_before=sup.get("smi_before"), smi_after=sup.get("smi_after"),
               job_dir=job_dir, remote=sup.get("info"))
    return res
