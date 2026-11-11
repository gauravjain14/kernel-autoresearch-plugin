"""Local GPU backend: one job = one `python -E -P` process on one GPU, inside that GPU's slot (-E: ignore PYTHON*
variables; -P: never put the working directory on sys.path; user site-packages stay visible).

A watchdog enforces a time limit per phase (from the worker's progress markers) and a total limit, and kills the job's
own process group by its exact id, never by pattern. nvidia-smi is recorded before and after.
"""
import json
import os
import signal
import subprocess
import sys
import time

from kar.backends import gate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOOT = "import sys; sys.path.insert(0, sys.argv[1]); from kar.evaluate.worker import main; sys.exit(main(sys.argv[2]))"
SMI_FIELDS = "name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active"


def smi(gpu):
    try:
        q = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        apps = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        vals = [v.strip() for v in q.stdout.strip().split(",")]
        out = dict(zip(SMI_FIELDS.split(","), vals))
        out["compute_apps"] = [a.strip() for a in apps.stdout.strip().splitlines() if a.strip()]
        return out
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}


def gpu_arch(gpu):
    try:
        q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        name, cc = [v.strip() for v in q.stdout.strip().split(",")]
        return name, "sm_" + cc.replace(".", "")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None


def _last_phase(path, t0):
    try:
        with open(path) as f:
            lines = f.read().strip().splitlines()
    except FileNotFoundError:
        return "start", t0
    if not lines:
        return "start", t0
    t, _, name = lines[-1].partition("\t")
    return name, float(t)


def run(job, job_dir, gpu=0, limits=None, total_s=3600, prefix=None):
    """Run one job; returns the worker's result dict plus status, timing and nvidia-smi snapshots. `prefix` wraps
    the worker command (a profiler such as nsys or ncu)."""
    limits = limits or {}
    os.makedirs(job_dir, exist_ok=True)
    jp = os.path.join(job_dir, "job.json")
    with open(jp + ".tmp", "w") as f:
        json.dump(job, f, indent=1)
    os.replace(jp + ".tmp", jp)
    for stale in ("result.json", "progress"):
        if os.path.exists(os.path.join(job_dir, stale)):
            os.remove(os.path.join(job_dir, stale))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = list(prefix or []) + [sys.executable, "-E", "-P", "-c", BOOT, ROOT, jp]
    killed = None
    with gate.slot(f"local-gpu{gpu}"):
        before = smi(gpu)
        t0 = time.time()
        with open(os.path.join(job_dir, "log.txt"), "wb") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=job_dir,
                                 start_new_session=True)
            while p.poll() is None:
                name, since = _last_phase(os.path.join(job_dir, "progress"), t0)
                key = name.split(" ")[0]
                lim = limits.get(key, limits.get("default", 300))
                now = time.time()
                if now - since > lim:
                    killed = f"phase '{name}' ran over its {lim} s limit"
                elif now - t0 > total_s:
                    killed = f"job ran over its {total_s} s total limit"
                if killed:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    break
                time.sleep(0.1)
        wall = time.time() - t0
        after = smi(gpu)
    res = {}
    rp = os.path.join(job_dir, "result.json")
    if os.path.exists(rp) and not killed:
        with open(rp) as f:
            res = json.load(f)
    if killed:
        res.update(status="timeout", reason=killed)
    elif not res:
        res.update(status="crashed", reason=f"worker exited {p.returncode} without a result")
    if res.get("status") in ("timeout", "crashed", "error"):
        with open(os.path.join(job_dir, "log.txt"), "rb") as f:
            res["log_tail"] = f.read()[-3000:].decode(errors="replace")
    res.update(wall_s=round(wall, 2), smi_before=before, smi_after=after, job_dir=job_dir)
    return res


class LocalBackend:
    """This machine's GPU `gpu`."""
    kind = "local"

    def __init__(self, gpu=0):
        self.gpu = int(gpu)
        self._info = None

    def run(self, job, job_dir, limits=None, total_s=3600, prefix_fn=None):
        return run(job, job_dir, gpu=self.gpu, limits=limits, total_s=total_s,
                   prefix=prefix_fn(job_dir) if prefix_fn else None)

    def info(self):
        if self._info is None:
            name, arch = gpu_arch(self.gpu)
            try:
                drv = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=10).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                drv = "unknown"
            self._info = {"name": name, "arch": arch, "driver": drv or "unknown"}
        return self._info

    def describe(self):
        return f"local GPU {self.gpu}"
