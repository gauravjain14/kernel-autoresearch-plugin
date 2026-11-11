"""Tamper check: a snapshot of every function the evaluator relies on, taken before any candidate is imported.

A candidate that patches timers, graph replay, tensor methods used for comparison, or the evaluator's own functions is
rejected. This covers Python-level patching; external timing (nsys, M1) covers the rest.
"""
import sys
import time

import numpy as np
import torch

TORCH = [
    ("torch.cuda.Event", lambda: torch.cuda.Event, ["record", "elapsed_time", "synchronize", "query", "wait"]),
    ("torch.cuda.CUDAGraph", lambda: torch.cuda.CUDAGraph, ["replay", "capture_begin", "capture_end"]),
    ("torch.cuda", lambda: torch.cuda, ["synchronize", "graph", "current_stream", "stream", "Stream", "Event",
                                        "CUDAGraph", "is_current_stream_capturing"]),
    ("torch.Tensor", lambda: torch.Tensor, ["copy_", "clone", "double", "float", "abs", "sub", "__sub__", "max", "mean",
                                            "sum", "fill_", "all", "item", "is_floating_point", "data_ptr", "__gt__",
                                            "__le__", "__mul__", "__add__", "to", "cpu", "numel", "normal_"]),
    ("torch", lambda: torch, ["equal", "isfinite", "abs", "empty", "empty_like", "is_tensor", "zeros"]),
    ("time", lambda: time, ["perf_counter", "time"]),
    ("numpy", lambda: np, ["median", "percentile", "log", "exp", "mean"]),
]
# Global numeric settings a candidate could flip to loosen the reference it is compared against.
SETTINGS = {
    "torch.backends.cuda.matmul.allow_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
    "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction":
        lambda: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction":
        lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    "torch.backends.cudnn.allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
    "torch.get_float32_matmul_precision": lambda: torch.get_float32_matmul_precision(),
    "torch.get_default_dtype": lambda: torch.get_default_dtype(),
}
KAR_MODULES = ["kar.evaluate.bench", "kar.evaluate.check", "kar.evaluate.compare", "kar.evaluate.loader",
               "kar.evaluate.worker", "kar.evaluate.integrity", "kar.task.spec"]


def _owners():
    out = [(label, get(), names) for label, get, names in TORCH]
    for m in KAR_MODULES:
        mod = sys.modules.get(m)
        if mod is not None:
            out.append((m, mod, [k for k, v in vars(mod).items() if callable(v) and not k.startswith("__")]))
    return out


def snapshot():
    snap = {"setting:" + k: (None, k, get()) for k, get in SETTINGS.items()}
    for label, owner, names in _owners():
        for n in names:
            if hasattr(owner, n):
                snap[f"{label}.{n}"] = (owner, n, getattr(owner, n))
    return snap


def verify(snap):
    """Names whose object changed since the snapshot (empty list = untouched)."""
    changed = []
    for key, (owner, n, obj) in snap.items():
        cur = SETTINGS[n]() if owner is None else getattr(owner, n, None)
        if cur is not obj and cur != obj:
            changed.append(key)
    return sorted(changed)
