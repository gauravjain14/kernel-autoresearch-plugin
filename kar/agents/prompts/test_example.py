"""A test of your own, run on the GPU by `test lanes/<id>/my_test.py [FILE]` (FILE: your kernel, `best`, or a
candidate id; default your lane's cand.py).

test() gets the kernel, the task's reference, the task's input builder and every scored and nearby shape, and returns
cases as (name, ok, detail). Build inputs the task's own checks never use, call the kernel and the reference on the
same inputs, and compare with close(), which applies the task tolerance and requires NaN and inf in the same places.

What real callers pass that random inputs never do: masked or -inf entries, whole rows masked, all zeros, very large
and very small magnitudes, the dtype's extremes, repeated values; and how they call: several results held at once,
the same buffers reused, calls from other streams. Read task/reference.py for what the inputs mean.

The file follows the kernel rules (no files, no network, no os). A test the reference passes can be registered:
`register lanes/<id>/my_test.py "what it checks"`; from then on every submit must pass it.
"""
import torch


def test(kernel_fn, reference, make, shapes, close):
    cases = []
    for s in shapes:
        args = list(make(seed=1234, **s["args"]))
        x = args[0]                          # adapt to the task: here, the first input
        if torch.is_tensor(x) and x.is_floating_point() and x.numel() > 1:
            x = x.clone()
            x.view(-1)[: max(1, x.numel() // 10)] = 0      # a block of zeros at the start
            args[0] = x
        ok, why = close(kernel_fn(*args), reference(*args))
        cases.append((f"{s['name']}: first 10% of the first input set to zero", ok, why))
    return cases
