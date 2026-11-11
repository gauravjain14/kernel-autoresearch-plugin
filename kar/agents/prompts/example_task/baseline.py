"""Baseline: PyTorch's BF16 matmul (cuBLAS), with reduced-precision BF16 reductions turned off for this call only.

PyTorch allows cuBLAS split-K reductions in BF16 by default; that measured 1.56x the reference's mean error on one
nearby shape, which breaks the task's FP32-accumulation rule. The flag is restored before returning, so the global
settings the evaluator checks are unchanged."""
import torch

_m = torch.backends.cuda.matmul


def kernel_fn(x, w):
    prev = _m.allow_bf16_reduced_precision_reduction
    _m.allow_bf16_reduced_precision_reduction = False
    try:
        return x @ w.t()
    finally:
        _m.allow_bf16_reduced_precision_reduction = prev
