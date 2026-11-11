"""y = x @ w.T. The rules: BF16 inputs and output, FP32 accumulation."""
import torch


def reference(x, w):
    return (x.float() @ w.float().t()).to(x.dtype)


def exact(x, w):
    return x.double() @ w.double().t()
