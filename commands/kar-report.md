---
description: Re-measure a kar run's best, check its integrity and write REPORT.md
argument-hint: <run folder>
allowed-tools: Bash(kar *), Read, Write, Edit, Glob, Grep
---
Run `kar report $ARGUMENTS` (`kar` is this plugin's command, on the shell's PATH; else
`${CLAUDE_PLUGIN_ROOT}/bin/kar`). It re-checks and re-measures the best on the GPU, which takes a few minutes. Then
summarize REPORT.md for the user: the final result with its interval, the integrity lines (any that are not ok or
none come first), the trajectory of the best, the team's cost per agent, and any hazards the auditors raised. Say
where the final kernel is: `$ARGUMENTS/candidates/<best id>/kernel.py`.
