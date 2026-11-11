"""RMSNorm: y = x / sqrt(mean(x^2) + eps) * w over the last dimension. The rules: BF16 inputs and output, FP32 math."""
import torch

EPS = 1e-6


def reference(x, w):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * rstd * w.float()).to(x.dtype)


def exact(x, w):
    xd = x.double()
    return xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
