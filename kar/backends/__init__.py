"""GPU backends. Every GPU job goes through one, and every one has the same two calls:

    run(job, job_dir, limits, total_s, prefix_fn) -> result dict (status, reason, wall_s, smi_before/after, ...)
    info() -> {"name", "arch", "driver"} of the GPU it runs on

local   this machine's GPU `gpu`                                   (one slot per GPU)
modal   Modal containers (H100, H200, B200, ...), from a registry image or a CUDA base image
ssh     a machine you can ssh into, one slot per listed GPU

A run's backend is in run.json ("backend": {"kind": ..., ...}); `kar start --backend` sets it. Concurrency is capped
by slot locks (kar/backends/gate.py): at most `max_boxes` Modal containers, one job per ssh GPU, one per local GPU.
"""


def make(cfg=None, gpu=0):
    cfg = dict(cfg or {"kind": "local", "gpu": gpu})
    kind = cfg.get("kind", "local")
    if kind == "local":
        from kar.backends.local import LocalBackend
        return LocalBackend(int(cfg.get("gpu", gpu)))
    if kind == "modal":
        from kar.backends.modal_backend import ModalBackend
        return ModalBackend(cfg)
    if kind == "ssh":
        from kar.backends.ssh import SshBackend
        return SshBackend(cfg)
    raise ValueError(f"unknown backend kind {kind!r}: local, modal or ssh")


def parse_spec(text, opts=()):
    """'local', 'local:1', 'modal:H200', 'modal:H200:3' (max boxes), 'ssh:user@host:0,1' (GPUs), plus key=value
    options (`--backend-opt image=lmsysorg/sglang:dev`) -> config dict."""
    parts = (text or "local").split(":")
    kind = parts[0]
    if kind == "local":
        cfg = {"kind": "local", "gpu": int(parts[1]) if len(parts) > 1 and parts[1] else 0}
    elif kind == "modal":
        cfg = {"kind": "modal", "gpu": parts[1] if len(parts) > 1 and parts[1] else "H100"}
        if len(parts) > 2 and parts[2]:
            cfg["max_boxes"] = int(parts[2])
    elif kind == "ssh":
        if len(parts) < 2 or not parts[1]:
            raise ValueError("ssh backend: ssh:HOST[:GPU,GPU...]")
        cfg = {"kind": "ssh", "host": parts[1],
               "gpus": [int(g) for g in parts[2].split(",")] if len(parts) > 2 and parts[2] else [0]}
    else:
        raise ValueError(f"unknown backend {text!r}: local[:GPU], modal:GPU[:MAX_BOXES], ssh:HOST[:GPUS]")
    for kv in opts:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"--backend-opt {kv!r}: key=value")
        if k in ("pip", "apt", "ssh_opts"):
            cfg[k] = [x for x in v.split(",") if x]
        elif k in ("max_boxes", "timeout_s", "memory_mb", "cpu"):
            cfg[k] = int(v)
        elif k.startswith("env."):
            cfg.setdefault("env", {})[k[4:]] = v
        else:
            cfg[k] = v
    return cfg
