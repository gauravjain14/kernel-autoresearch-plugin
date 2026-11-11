"""Modal backend: each GPU job runs in its own Modal container.

Config (run.json "backend", or `kar start --backend modal:H200:3` plus `--backend-opt key=value`):
    gpu        Modal GPU type: "H100", "H200", "B200", "A100-80GB", "L40S", ...           (default H100)
    max_boxes  containers at once; also the number of local slot locks                       (default 3)
    image      a registry image to run in (for a challenge, its official image), or omit for a CUDA devel image
               with Python packages installed (`pip`)
    pip        Python packages to install into the image (default: torch built for the image's CUDA 12.8, numpy,
               cuda-python 12.x; with `image`, only cuda-python and numpy)
    add_python the Python version Modal adds to the image (default: this machine's, which the job function needs)
    apt        apt packages (for example profilers missing from the image)
    timeout_s  per-container time limit (default 3600)
    app        Modal app name (default "kar")
    env        environment variables for the job

Every job is one call: the job bundle (kar/backends/remote.py) goes in as bytes, the job folder comes back as bytes,
and the supervisor inside enforces the per-phase limits. Nothing is uploaded except the bundle: the harness package,
the task folder, and the files the job names. The Modal app is started once per harness process and stopped at exit.

Profilers: nsys is not part of the CUDA base images; use an image that has Nsight Systems (or apt-install it), or the
profile pages run without their nsys section. ncu needs GPU performance counters, which a container may not expose;
the pages then say so (ERR_NVGPUCTRPERM) and the A/B timing is unaffected.

Needs `pip install modal` and `modal token new` (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET) on the machine running the
harness. Modal bills the containers; the run's GPU budget (gpu_h) counts their wall time.
"""
import atexit
import os
import sys
import threading

from kar.backends import gate, remote

DEFAULT_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
# CUDA 12.8 image: cuda-python 13 would look for CUDA 13 libraries, and torch from PyPI may be built for another CUDA
DEFAULT_PIP = ["numpy", "cuda-python>=12.6,<13"]
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
LOCAL_PY = f"{sys.version_info.major}.{sys.version_info.minor}"

_lock = threading.Lock()
_sessions = {}


def _session(cfg):
    """(app context, the job function, the info function) for this config, started once per process."""
    key = repr(sorted((k, repr(v)) for k, v in cfg.items()))
    with _lock:
        if key in _sessions:
            return _sessions[key]
        try:
            import modal
        except ImportError:
            raise remote.BackendError("the modal backend needs `pip install modal` and `modal token new`") from None
        if cfg.get("image"):
            image = modal.Image.from_registry(cfg["image"], add_python=cfg.get("add_python"))
        else:
            image = modal.Image.from_registry(DEFAULT_IMAGE, add_python=cfg.get("add_python", LOCAL_PY))
        if cfg.get("apt"):
            image = image.apt_install(*cfg["apt"])
        # a custom image keeps its own torch; add only what the worker needs and it may lack
        if not cfg.get("image") and "pip" not in cfg:
            image = image.pip_install("torch", index_url=DEFAULT_TORCH_INDEX)
        pip = cfg.get("pip", DEFAULT_PIP)
        if pip:
            image = image.pip_install(*pip)
        image = image.entrypoint([])
        app = modal.App(cfg.get("app", "kar"))
        opts = dict(image=image, gpu=cfg.get("gpu", "H100"), timeout=int(cfg.get("timeout_s", 3600)),
                    max_containers=int(cfg.get("max_boxes", 3)), serialized=True)
        if cfg.get("cpu"):
            opts["cpu"] = cfg["cpu"]
        if cfg.get("memory_mb"):
            opts["memory"] = int(cfg["memory_mb"])

        @app.function(**opts)
        def run_job(bundle: bytes) -> bytes:
            import io
            import json
            import subprocess
            import tarfile
            import tempfile
            root = tempfile.mkdtemp(prefix="kar_")
            with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
                tf.extractall(root)
            spec = os.path.join(root, "spec.json")
            with open(spec) as f:
                s = json.load(f)
            s["root"] = root
            # the job's interpreter is the one this function runs in (the image's own `python3` may be another)
            s["cmd"] = [sys.executable if c == "python3" else c for c in s["cmd"]]
            with open(spec, "w") as f:
                json.dump(s, f)
            subprocess.run([sys.executable, os.path.join(root, "kar", "backends", "supervise.py"), spec], check=False)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(os.path.join(root, "job"), arcname=".")
            return buf.getvalue()

        @app.function(**{**opts, "timeout": 300})
        def gpu_info() -> str:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True)
            return r.stdout

        if sys.stderr.isatty() and os.environ.get("KAR_MODAL_QUIET") != "1":
            # on a terminal, show Modal's progress: the first run builds the image for several quiet minutes otherwise
            shown = modal.enable_output()
            shown.__enter__()
            atexit.register(lambda: shown.__exit__(None, None, None))
        ctx = app.run()
        ctx.__enter__()
        atexit.register(lambda: ctx.__exit__(None, None, None))
        _sessions[key] = (ctx, run_job, gpu_info)
        return _sessions[key]


class ModalBackend:
    kind = "modal"

    def __init__(self, cfg):
        self.cfg = {k: v for k, v in cfg.items() if k != "kind"}
        self.slots = int(self.cfg.get("max_boxes", 3))
        self.name = f"modal-{self.cfg.get('app', 'kar')}-{self.cfg.get('gpu', 'H100')}"
        self._info = None

    def run(self, job, job_dir, limits=None, total_s=3600, prefix_fn=None):
        bundle, _ = remote.pack(job, prefix_fn=prefix_fn, gpu=None, limits=limits, total_s=total_s,
                                env=self.cfg.get("env"), python=self.cfg.get("python", "python3"))
        os.makedirs(job_dir, exist_ok=True)
        with gate.any_slot(self.name, self.slots):
            try:
                _, run_job, _ = _session(self.cfg)
                out = run_job.remote(bundle)
            except remote.BackendError:
                raise
            except Exception as e:  # noqa: BLE001 - a transport failure is a failed job, not a crash
                return remote.result(job_dir, transport_error=f"{type(e).__name__}: {e}")
        remote.unpack(out, job_dir)
        return remote.result(job_dir)

    def info(self):
        """The GPU's name, arch and driver; cached on disk for a day so a new harness process does not start a
        container just to ask."""
        if self._info is None:
            import hashlib
            import json
            import time
            key = hashlib.sha256(repr(sorted((k, repr(v)) for k, v in self.cfg.items())).encode()).hexdigest()[:16]
            path = os.path.expanduser(f"~/.cache/kar/backends/modal_{key}.json")
            if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
                with open(path) as f:
                    self._info = json.load(f)
                return self._info
            _, _, gpu_info = _session(self.cfg)
            name, cc, drv = [v.strip() for v in gpu_info.remote().strip().splitlines()[0].split(",")]
            self._info = {"name": name, "arch": "sm_" + cc.replace(".", ""), "driver": drv}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._info, f)
        return self._info

    def describe(self):
        return f"Modal {self.cfg.get('gpu', 'H100')} (up to {self.slots} containers)"
