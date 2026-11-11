# Write a kernel-optimization task

This folder will become a task for kar, a harness that optimizes GPU kernels against a frozen evaluator. Your job
is to turn the request in `DESCRIPTION.md` into the task files below. You define what "correct" and "fast" mean
here; you do not write an optimized kernel.

When you are done, `./kar-task validate` must pass (the same check as `kar validate <this folder>`). Run it after
every change. A person reviews `reference.py` before any optimization starts, because it defines correct.

A complete, working example task (a BF16 matrix-vector product: task.toml, reference.py, inputs.py, shapes.py,
opmodel.py, baseline.py) is in `$example`. Read those six files first; they show every convention this guide
describes, so you do not need to read the harness's own code.

## Files

| file | what it holds |
|---|---|
| `task.toml` | the name, the file references, the tolerance, the rules (template already here; fill in the TODOs) |
| `reference.py` | `reference(*args)`: the operation at the precision the task prescribes; `exact(*args)`: the same in float64 |
| `inputs.py` | `make(seed, **shape_args)`: fresh, deterministic inputs on the GPU (a `torch.Generator(device="cuda")` seeded with `seed`) |
| `shapes.py` | `SCORED`: the shapes that are timed, each `{"name", "weight", "args"}`; `NEARBY`: shapes that are only checked |
| `opmodel.py` | `min_bytes(**args)` -> `{"read": bytes, "write": bytes}` of compulsory DRAM traffic; `flops(**args)` -> `{pipe: flops}` |
| `baseline.py` | `kernel_fn`: what speedups are measured against. Default: the reference itself |
| `score.py` (optional) | `score(times)` -> a number, higher is better, from `{shape name: microseconds}`, when the score is not a weighted geomean of per-shape speedups (for example a geomean over weighted sums of shapes, a throughput, a total time) |
| `judge.py` (optional) | `parse(stdout, stderr, returncode)` -> `{"score": float, ...}` for an external judge's output |

## How to get each one right

- **reference:** written plainly in PyTorch, at the precision the request asks for (for example: BF16 inputs and
  output, FP32 accumulation). It is the ground truth for tolerance and for the error-ratio test, so it must not
  take shortcuts. `exact` computes the same thing in float64 from the same inputs.
- **inputs:** realistic value ranges (scale weights so outputs stay O(1); avoid inputs that make the result
  ill-conditioned unless the request is about that). Every tensor the kernel reads is created here; nothing is
  cached between calls.
- **SCORED shapes:** the sizes the request cares about, named so a person can read them (`t128_h4096`). Weights
  say how much each shape matters (equal unless the request says otherwise). Keep each call between a few
  microseconds and a few milliseconds on the target GPU.
- **NEARBY shapes:** 4-6 shapes that are not scored but must be correct: off-grid sizes (odd, not a multiple of
  8 or 16, one larger than a scored size, very small). They stop a kernel from being correct only on the scored
  shapes.
- **Edge values:** the values the op meets in real use but random inputs never produce, given as NEARBY shapes whose
  args switch them on (`make(seed, ..., edge="masked")`), each checked against the reference like any shape. Think
  through what callers really pass: masked or -inf entries (softmax after top-k/top-p filtering, attention masks),
  whole rows masked, zeros (norms of an all-zero row), very large or very small magnitudes, repeated values, the
  largest and smallest values of the dtype. A kernel that is right on random data and wrong on these is wrong; an
  auditor on a softmax run found exactly that (whole rows of NaN for masked logits) after every test had passed.
- **op model:** the bytes a perfect kernel must move to and from DRAM (each input read once, each output written
  once, at the stored dtype) and the flops it must do, by pipe: `fp32` for CUDA-core FP32 math, `tc_bf16`,
  `tc_fp16`, `tc_tf32`, `tc_fp8` for tensor-core math. This sets the speed-of-light floor, so count only what is
  compulsory.
- **tolerance:** the defaults (atol/rtol 0.02 for BF16 outputs, error ratio 1.5x mean / 4x max against the
  reference's own error) suit most BF16 tasks. Loosen only with a reason written in a comment.
- **rules:** write the precision rule in plain words in `[rules] precision`, so agents see it.

## When the request points at an existing project or challenge

If this folder has a `source/` folder, it is a copy of the project, challenge or benchmark the request is about. Your
task must measure what that source measures, so read it first, completely: its README, task statement and rules,
its reference implementation, its benchmark or evaluation script, its scoring formula, and any judge or leaderboard
instructions. Then mirror it, citing the source file and line for each decision under "Assumptions":

- **reference, inputs, shapes:** take them from the source (import from `source/` when its code is a clean
  reference; otherwise re-express it plainly). Use the source's exact shapes, input distributions and weights.
  Keep its edge cases, and add the ones real callers hit that it misses.
- **score:** if the source's score is not a weighted geomean of per-shape speedups, write `score.py` with its exact
  formula and set `score = "score.py:score"` in task.toml. `./kar-task validate` checks that it grows as times fall.
- **rules:** copy the source's rules into `[rules]`: the precision rule in words (`precision`), size caps
  (`max_source_bytes`), forbidden constructs as regular expressions (`forbidden_patterns`), allowed languages.
- **target GPU:** set `arch` to what the source targets (for example `"sm_90a"` for H100/H200). The harness runs on the
  backend the person configured; a task for a GPU this machine lacks is validated and run there.
- **timing mode:** if the work cannot be captured in a CUDA graph (host synchronization, data-dependent shapes or
  allocation, a whole model step), set `[bench] mode = "eager"`.
- **judge:** if the source has a command-line judge, leaderboard client or evaluation script, describe it in a
  `[judge]` table (`check` and `submit` command lines with `{file}`, `parse = "judge.py:parse"`, the quotas, the
  filename it expects) so results can be sent to it. Leave `policy.submit = "ask"`: scored submissions are a
  person's decision. Without a judge, leave the table out: the team works against the speed-of-light floor.
- **workload size:** `kernel_fn` can be anything the source times: one op, a fused chain, a whole layer or model step.
  Its op model must still count the compulsory bytes and flops of the whole thing, so the roofline is right.

## Ambiguity

If the request leaves something open (dtype, layout, which dimension, the shapes, epsilon), choose the most common
convention, and write each choice under "Assumptions" at the end of `DESCRIPTION.md`. A person will check them. Do
not stop to ask.

## Limits

Write only in this folder. Do not write optimized kernels, do not copy code from elsewhere, and do not edit
`DESCRIPTION.md` above the "Assumptions" heading.
