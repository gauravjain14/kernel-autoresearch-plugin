# Step: audit $cid (lane $lane)

$status

## Your job

`$cid` passed every gate, including the stress suite, and was kept (its parent: $parent). Your job is to find what
the tests cannot reach, before the team builds more on it. Read `candidates/$cid/kernel.py` closely, and
`$tool diff $parent $cid` for what changed.

Look for:
- **dispatch-order reliance**: one block waiting on a flag or a counter that another block sets, where progress
  needs the hardware to start blocks in index order or to keep them all resident. That is undocumented behaviour; a
  kernel that needs ordering must take it from an atomic ticket, not from blockIdx;
- **races**: a shared-memory or global buffer reused before every reader is done, a missing barrier or fence, an
  async copy consumed before it lands, a slot released before its data is read;
- **state between calls**: anything static, cached or allocated once and reused, workspaces shared between calls;
- **shapes**: code paths only the scored shapes take, assumptions about divisibility or alignment that the task
  does not promise, integer overflow at large sizes;
- **precision**: anything below the task's precision rule, accumulation order that changes the error;
- **undefined behaviour**: out-of-bounds reads masked by luck, uninitialised shared memory.

**Prove what you suspect by testing it.** Write a test of your own (start from `help/test_example.py`): inputs
the task's checks never use, compared with the reference. Run it with `$tool test lanes/$lane/my_test.py $cid`.
If it fails, register it before you raise the hazard: `$tool register lanes/$lane/my_test.py "what it checks"`.
The harness registers a test only if the task's reference passes it; from then on every submit, the fix included,
must pass it. A hazard with a failing registered test is proven; one without is an argument. You have $tries_left
GPU commands (`test`, `register`, `stress`, `check`, `profile`, `ncu`, `sass`); spend them on tests.

## Hand back

- `$tool audit ok "what you checked and why it holds"`, or
- `$tool audit hazard "what can fail, where (function and line), why, the fix, and the registered test (tNNN)
  that shows it"`.

A hazard queues a fix ahead of every other hypothesis, and the run does not stop on its target until the fix is
in. Raise one only for a real failure mode, with the reasoning. Then end the session.

## Dead ends so far

$dead

## Measured facts

$facts
