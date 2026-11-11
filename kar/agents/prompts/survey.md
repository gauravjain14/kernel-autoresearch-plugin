# Step: survey, lens "$lens" (lane $lane, round $cycle)

$status

## Your job

The planner for this round turns the team's findings into hypotheses. Your lens is **$lens**: $lens_text

Look at the current best ($best) through that lens only; other lanes cover the other lenses at the same time. Find
where time goes that the floor says need not, measure it, and say what would remove it and how much that is worth.

Start from:
- the profile of the current best: `$profile` (gap table, nsys per-kernel times, ncu summary, SASS audit);
- the kernel itself: `candidates/$best/kernel.py` (a copy is in `lanes/$lane/base.py`);
- the task: `task/README.md`, `task/reference.py`, `task/shapes.py`;
- `board`, the dead ends and the facts below, and `METHOD.md`.

Measure what the profile does not show (each counts against your $tries_left GPU commands):

| command | what it gives you |
|---|---|
| `$tool ncu best --shape NAME` | every ncu section on one scored shape, with ncu's rule notes, into your lane folder |
| `$tool sass best` | the full SASS of every kernel the best loads, into your lane folder |
| `$tool bound lanes/$lane/probe.py` | time a stripped probe (only the loads and stores, or the math removed) against the best: the most a design can gain |
| `$tool try lanes/$lane/v1.py lanes/$lane/v2.py ...` | quick A/Bs of variants against the best, all in one GPU job (correctness checked) |
| `$tool profile lanes/$lane/variant.py` | the full profile page of a variant |
| `$tool test lanes/$lane/my_test.py best` | your own test (from `help/test_example.py`) on the best: inputs the task's checks never use |
| `$tool docs find "TERM"` | the PTX ISA, CUDA guides, this GPU's tuning guide, the ncu/nsys guides: what an instruction or metric guarantees |

## Hand back

Write `lanes/$lane/FINDINGS.md` and run `$tool findings lanes/$lane/FINDINGS.md`. Start with a one-line summary.
Then, for each finding:

- **what**: the measured fact, with the number and where it comes from (a profile line, an ncu metric, a SASS
  count, a bound you timed);
- **so what**: how much of the score it costs, from the gap table ("t4096 is at 61% of floor and worth 40% of the
  score: closing half the gap there is +8%");
- **idea**: what would remove it, and the cheapest test that would kill the idea.

Numbers you did not measure or read are guesses; say so. A finding that says "nothing to gain here" with the
numbers that show it is as useful as an idea. Then end the session.

## Dead ends so far

$dead

## Measured facts

$facts
