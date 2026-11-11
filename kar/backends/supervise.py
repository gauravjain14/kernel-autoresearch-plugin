"""Supervisor for one GPU job on a machine the harness does not run on (a Modal container, an ssh host).

Standard library only: it runs under whatever Python the remote machine has. Given a spec file, it

  1. substitutes the job's remote root for "@R@" in job.json and in the command (paths are only known there),
  2. records nvidia-smi, starts the command in its own process group, and enforces the same per-phase and total
     time limits as the local backend (the worker writes phase markers to `progress`), killing the group by its id,
  3. records nvidia-smi again and, when the job produced profiler reports, exports them where the tools that wrote
     them are installed (nsys -> sqlite, ncu -> raw CSV and the details page), so the harness never needs matching
     profiler versions locally,
  4. writes supervisor.json next to result.json.

    python3 supervise.py SPEC.json
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

SMI_FIELDS = "name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active"


def smi():
    try:
        q = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
        vals = [v.strip() for v in q.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(SMI_FIELDS.split(","), vals))
    except (OSError, subprocess.SubprocessError, IndexError) as e:
        return {"error": str(e)}


def info():
    """{name, arch, driver, sms?} of the GPU this process sees."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        name, cc, drv = [v.strip() for v in q.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "arch": "sm_" + cc.replace(".", ""), "driver": drv}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return {"error": str(e)}


def _last_phase(path, t0):
    try:
        with open(path) as f:
            lines = f.read().strip().splitlines()
    except FileNotFoundError:
        return "start", t0
    if not lines:
        return "start", t0
    t, _, name = lines[-1].partition("\t")
    try:
        return name, float(t)
    except ValueError:
        return name, t0


def _tool(name):
    import glob
    return shutil.which(name) or next(iter(sorted(glob.glob(f"/usr/local/cuda*/bin/{name}"), reverse=True)), None) or \
        next(iter(sorted(glob.glob(f"/opt/nvidia/nsight-*/*/{name}") + glob.glob(f"/opt/nvidia/nsight-*/*/bin/{name}"),
                         reverse=True)), None)


def post(job_dir):
    """Export profiler reports next to them, with the tools that wrote them."""
    out = {}
    rep = os.path.join(job_dir, "nsys.nsys-rep")
    if os.path.exists(rep):
        nsys = _tool("nsys")
        if nsys:
            r = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o",
                                os.path.join(job_dir, "nsys.sqlite"), rep], capture_output=True, text=True, timeout=900)
            out["nsys_export"] = r.returncode
    rep = os.path.join(job_dir, "ncu.ncu-rep")
    if os.path.exists(rep):
        ncu = _tool("ncu")
        if ncu:
            with open(os.path.join(job_dir, "ncu_raw.csv"), "w") as f:
                out["ncu_raw"] = subprocess.run([ncu, "--import", rep, "--page", "raw", "--csv"], stdout=f,
                                                stderr=subprocess.DEVNULL, timeout=900).returncode
            with open(os.path.join(job_dir, "ncu_details.txt"), "w") as f:
                out["ncu_details"] = subprocess.run([ncu, "--import", rep, "--page", "details", "--print-details",
                                                     "all"], stdout=f, stderr=subprocess.DEVNULL,
                                                    timeout=900).returncode
    return out


def main(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)
    root = spec["root"]
    job_dir = os.path.join(root, "job")
    jp = os.path.join(job_dir, "job.json")
    with open(jp) as f:
        text = f.read()
    with open(jp, "w") as f:
        f.write(text.replace("@R@", root))
    cmd = [c.replace("@R@", root) for c in spec["cmd"]]
    limits = spec.get("limits") or {}
    total_s = float(spec.get("total_s", 3600))
    env = dict(os.environ)
    if spec.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    for k, v in (spec.get("env") or {}).items():
        env[k] = str(v)
    before = smi()
    t0 = time.time()
    killed = None
    with open(os.path.join(job_dir, "log.txt"), "wb") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=job_dir, start_new_session=True)
        while p.poll() is None:
            name, since = _last_phase(os.path.join(job_dir, "progress"), t0)
            lim = float(limits.get(name.split(" ")[0], limits.get("default", 300)))
            now = time.time()
            if now - since > lim:
                killed = f"phase '{name}' ran over its {lim:.0f} s limit"
            elif now - t0 > total_s:
                killed = f"job ran over its {total_s:.0f} s total limit"
            if killed:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                break
            time.sleep(0.2)
    wall = time.time() - t0
    after = smi()
    sup = {"killed": killed, "wall_s": round(wall, 2), "returncode": p.returncode, "smi_before": before,
           "smi_after": after, "info": info()}
    if not killed and spec.get("post", True):
        try:
            sup["post"] = post(job_dir)
        except (OSError, subprocess.SubprocessError) as e:
            sup["post"] = {"error": str(e)}
    with open(os.path.join(job_dir, "supervisor.json"), "w") as f:
        json.dump(sup, f, indent=1)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--info"]:
        print(json.dumps(info()))
        sys.exit(0)
    sys.exit(main(sys.argv[1]))
