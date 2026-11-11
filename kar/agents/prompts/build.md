# Step: build hypothesis $hid (lane $lane, round $cycle)

$status

## The hypothesis

**$title**

- target: $target
- mechanism: $mechanism
- evidence: $evidence
- predicted score gain: $predicted% ($arithmetic)
- bound test: $bound_test

$fix$moved
## Your job

Implement this hypothesis as `lanes/$lane/cand.py`, starting from `lanes/$lane/base.py` (a copy of $base, the best
when this lane opened) or from scratch, and hand it back. Profile of the current best: `$profile`. Other lanes are
building other hypotheses at the same time: `$tool board`, and their `lanes/*/NOTES.md`.

Commands (each GPU command runs through the harness, queues with the team's, and counts against the budget):

| command | what it does |
|---|---|
| `$tool check lanes/$lane/cand.py` | rules lint, a 20 s smoke call, then the full correctness suite on every scored and nearby shape |
| `$tool try lanes/$lane/cand.py [lanes/$lane/v2.py ...]` | check, then a quick A/B against the current best ($try_reps reps); per-shape times and % of floor. Several variants (block sizes, tile shapes, unrolls) go in ONE job: use that instead of one try per turn |
| `$tool bound lanes/$lane/probe.py` | your bound test: time a probe that is incomplete on purpose against the best, output not checked; never submittable |
| `$tool profile lanes/$lane/cand.py` | the full profile page for your candidate (gap table, nsys, ncu, SASS) |
| `$tool ncu lanes/$lane/cand.py --shape NAME` / `sass ...` | every ncu section on one shape / the full SASS, into your lane folder |
| `$tool stress lanes/$lane/cand.py` | extra seeds, concurrent calls on several streams, back-to-back calls |
| `$tool test lanes/$lane/my_test.py` | your own test (from `help/test_example.py`) on your kernel: the inputs real callers pass that random ones never do |
| `$tool register lanes/$lane/my_test.py "what it checks"` | make your test part of the run (the reference must pass it); every later submit must pass it too |
| `$tool submit lanes/$lane/cand.py --report lanes/$lane/report.md` | hand back: the harness freezes the file and decides |
| `$tool abandon "why, with the numbers"` | drop the hypothesis when its bound test or your measurements say it cannot pay |
| `$tool docs find "TERM"` / `docs get NAME --section "TERM"` | the PTX ISA, CUDA guides, this GPU's tuning guide, ncu/nsys guides |
| `$tool note "fact with numbers"` / `post <lane\|all> "..."` | tell the team something |

You have $tries_left GPU commands (`try`, `bound`, `profile`, `ncu`, `sass`, `stress`, `test`, `register`) left in
this lane, and $fails_left failed submits. A `try` with several files is one command, so compare variants in one
`try`, and keep commands for `stress` and `test` on the kernel you hand in. Run the bound test first when it is cheap. Test your kernel the way a
user would call it before you submit; the team's registered tests are in `tests/`, and the judge runs them all.

The report (`lanes/$lane/report.md`, at most 40 lines) says what you changed, what you measured, and anything the
next agent should know.

## How the harness decides

On submit it re-runs every gate on the frozen file (lint, smoke, correctness, the stress suite, the team's
registered tests), then times it
against the current best inside one GPU job: $copies copies of each, $reps reps, on several memory placements, in
alternating order, with a fresh seed. It keeps the candidate if the confidence interval on the score effect is above
zero and no shape is slower than its noise band. Anything smaller than the noise reads as NO-GAIN. If the best moved
on while you worked and your kernel wins only against your base, it goes to a stacker, who ports it onto the new best.

## Dead ends so far

$dead

## Measured facts

$facts
