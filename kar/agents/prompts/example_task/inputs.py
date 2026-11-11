import math

import torch


def make(seed, b, n, k, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(b, k, generator=g, device=device).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device=device) / math.sqrt(k)).to(torch.bfloat16)
    return x, w
