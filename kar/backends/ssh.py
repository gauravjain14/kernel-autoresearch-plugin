"""ssh backend: GPU jobs on a machine you can reach with `ssh HOST` (key-based login, no password prompt).

Config: host ("user@box" or an ~/.ssh/config alias), gpus (the GPU indices jobs may use; one job per GPU at a time),
python (the remote interpreter with torch and cuda-python, default "python3"), workdir (default /tmp/kar-jobs),
ssh_opts (extra ssh options), env.

Each job: copy its bundle over, unpack it in a fresh folder, run the supervisor on one GPU, copy the job folder back,
delete the remote folder. The remote machine needs only Python, torch, cuda-python and, for profiling, nsys/ncu.
"""
import os
import shlex
import subprocess
import uuid

from kar.backends import gate, remote


class SshBackend:
    kind = "ssh"

    def __init__(self, cfg):
        self.host = cfg["host"]
        self.gpus = [int(g) for g in cfg.get("gpus", [0])]
        self.python = cfg.get("python", "python3")
        self.workdir = cfg.get("workdir", "/tmp/kar-jobs")
        self.opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", *cfg.get("ssh_opts", [])]
        self.env = cfg.get("env") or {}
        self.name = f"ssh-{self.host.replace('@', '_').replace(':', '_')}"
        self._info = None

    def _ssh(self, command, timeout, stdin=None):
        return subprocess.run(["ssh", *self.opts, self.host, command], input=stdin, capture_output=True,
                              timeout=timeout)

    def run(self, job, job_dir, limits=None, total_s=3600, prefix_fn=None):
        os.makedirs(job_dir, exist_ok=True)
        with gate.any_slot(self.name, len(self.gpus), labels=[f"gpu{g}" for g in self.gpus]) as label:
            gpu = int(label[3:])
            bundle, _ = remote.pack(job, prefix_fn=prefix_fn, gpu=gpu, limits=limits, total_s=total_s,
                                    env=self.env, python=self.python)
            rdir = f"{self.workdir}/{uuid.uuid4().hex[:12]}"
            q = shlex.quote
            script = (f"set -e; mkdir -p {q(rdir)}; cd {q(rdir)}; tar xzf -; "
                      f"{q(self.python)} -c \"import json; p='spec.json'; s=json.load(open(p)); s['root']='{rdir}'; "
                      f"json.dump(s, open(p, 'w'))\"; "
                      f"{q(self.python)} kar/backends/supervise.py spec.json >/dev/null 2>&1 || true")
            try:
                r = self._ssh(script, timeout=total_s + 300, stdin=bundle)
                if r.returncode:
                    return remote.result(job_dir, transport_error=f"ssh exited {r.returncode}: "
                                                                  f"{r.stderr.decode(errors='replace')[-800:]}")
                back = self._ssh(f"cd {q(rdir)}/job && tar czf - . ; rm -rf {q(rdir)}", timeout=900)
                if back.returncode:
                    return remote.result(job_dir, transport_error=f"fetching results: "
                                                                  f"{back.stderr.decode(errors='replace')[-800:]}")
            except subprocess.TimeoutExpired:
                return remote.result(job_dir, transport_error="ssh timed out")
            remote.unpack(back.stdout, job_dir)
            return remote.result(job_dir)

    def info(self):
        if self._info is None:
            r = self._ssh(f"CUDA_VISIBLE_DEVICES={self.gpus[0]} nvidia-smi --query-gpu=name,compute_cap,driver_version "
                          f"--format=csv,noheader", timeout=60)
            if r.returncode:
                raise remote.BackendError(f"ssh {self.host}: {r.stderr.decode(errors='replace')[-400:]}")
            name, cc, drv = [v.strip() for v in r.stdout.decode().strip().splitlines()[0].split(",")]
            self._info = {"name": name, "arch": "sm_" + cc.replace(".", ""), "driver": drv}
        return self._info

    def describe(self):
        return f"ssh {self.host} (GPUs {', '.join(map(str, self.gpus))})"

