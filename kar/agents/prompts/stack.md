# Step: stack (lane $lane, round $cycle)

$status

## Your job

These kernels each won against the best they were built on, but the best has moved on since (it is now $best).
Port their changes onto the current best and hand back one kernel that has all of them:

$items

Start from `lanes/$lane/base.py` (a copy of $best). Read each change with the `diff` command above and the
candidate's report (`candidates/<id>/report.md`), and apply it to the new base. Where two changes touch the same
code, combine them by hand and measure; if one of them no longer pays on the new best, drop it and say so with the
numbers.

Commands: `$tool check|try|bound|profile|ncu|sass|stress FILE` as in a build lane ($tries_left GPU commands left),
then `$tool submit lanes/$lane/cand.py --report lanes/$lane/report.md`, or `$tool abandon "why, with numbers"` if
none of them stacks. You have $fails_left failed submits.

On submit the harness runs every gate and times your kernel against the current best: $copies copies, $reps reps,
in one GPU job. It is kept if the interval on the score effect is above zero and no shape is slower than its noise.
Then end the session.

## Dead ends so far

$dead

## Measured facts

$facts
