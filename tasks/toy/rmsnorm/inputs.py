import torch


def make(seed, tokens, hidden, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g, device=device).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(hidden, generator=g, device=device)).to(torch.bfloat16)
    return x, w
