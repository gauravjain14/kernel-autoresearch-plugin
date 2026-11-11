"""Minimum DRAM traffic and math per call, for speed-of-light floors."""


def min_bytes(tokens, hidden):
    return {"read": 2 * tokens * hidden + 2 * hidden, "write": 2 * tokens * hidden}  # x and w in, y out (BF16)


def flops(tokens, hidden):
    return {"fp32": 4 * tokens * hidden}
