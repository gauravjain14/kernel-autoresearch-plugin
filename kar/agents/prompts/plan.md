# Step: plan round $cycle (lane $lane)

$status

$plateau
## Your job

Turn what the team knows into 1 to $max_hyps hypotheses for making the current best ($best) faster, then hand them
back with `$tool plan lanes/$lane/plan.json`. Build lanes take them in order of predicted gain, $team at a time, so
the plan is a portfolio: independent ideas that different agents can build at once, the biggest first. Do not
propose what a lane on the board is already building.

Read first:
- the survey findings below (this round's, from the survey lanes);
- the profile of the current best: `$profile`, and the kernel: `candidates/$best/kernel.py`;
- the task: `task/README.md`, `task/reference.py`, `task/shapes.py`;
- what has been tried (below, `notes/`) and `METHOD.md`.

The gap table of the current best (the profile's roofline section says, per shape, which roof binds it, how far
below that roof it runs, and how much of its floor is the fixed cost of a call):

$gap

$strategy

## The plan file

`lanes/$lane/plan.json`:

```json
{"hypotheses": [
  {"title": "(an example from a different op) GEMV: split K over 4 warps per output, reduce in shared memory",
   "target": "b1_n4096_k4096 (memory-bound, 54% of floor, 31% of the score at floor)",
   "mechanism": "what changes in the kernel and why it removes time",
   "evidence": "the measured numbers that show the problem (cite the finding or the profile line)",
   "predicted_gain_pct": 12.0,
   "arithmetic": "how the predicted score gain follows from the gap table",
   "bound_test": "the cheapest measurement that would show it cannot pay",
   "structural": false}
]}
```

`predicted_gain_pct` is the gain in score (the weighted geometric-mean speedup over all scored shapes) against
the current best, not the gain on one shape. `structural` is true for a rewrite, a fusion or a new decomposition.

## Survey findings

$findings

## The team now

$board

## What has been kept so far

$kept

## Dead ends (do not repeat these without a new reason)

$dead

## Measured facts

$facts
