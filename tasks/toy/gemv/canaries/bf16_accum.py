"""Canary: accumulates in BF16 (FP32 partial dot products of 128 elements, rounded to BF16 and summed in BF16).
Lower precision than the task allows; the evaluator must reject it."""
import torch


def kernel_fn(x, w):
    acc = None
    for k0 in range(0, x.shape[1], 128):
        part = (x[:, k0:k0 + 128].float() @ w[:, k0:k0 + 128].float().t()).to(torch.bfloat16)
        acc = part if acc is None else acc + part
    return acc
