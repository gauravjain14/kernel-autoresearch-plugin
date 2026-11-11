"""Minimum DRAM traffic and math per call, for speed-of-light floors."""


def min_bytes(b, n, k):
    return {"read": 2 * (n * k + b * k), "write": 2 * b * n}


def flops(b, n, k):
    return {"fp32": 2 * b * n * k}
