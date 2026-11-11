---
description: A kar run's external judge: results, quota, noise, and submissions waiting for your approval
argument-hint: <run folder>
---
Run `kar judge $ARGUMENTS status` (`kar` is this plugin's command, on the shell's PATH; else
`${CLAUDE_PLUGIN_ROOT}/bin/kar`) and summarize it for the user: the last judge results (score, which candidate, local
estimate), quota left, the judge's measured noise and the runs needed to see a 1-2% change, and every request waiting
for approval.

For each waiting request, tell the user what it is (check or scored submit, which candidate, why, the local gain since
the last judged build) and ask whether to approve it. Only on an explicit yes, run `kar judge RUN approve ID` and
report the result. Never approve a scored submit on your own.
