---
description: Show where a kar run stands, its team and its board
argument-hint: <run folder>
allowed-tools: Bash(kar *), Read, Write, Edit, Glob, Grep
---
Show the state of the kar run at $ARGUMENTS.

Run `kar status $ARGUMENTS --board` (`kar` is this plugin's command, on the shell's PATH; else
`${CLAUDE_PLUGIN_ROOT}/bin/kar`). Read `$ARGUMENTS/ws/candidates/<id>/report.md` of the newest candidates only if the
user asks what changed. Summarize in a few lines: the best so far (% of speed of light and x baseline), what each
lane at work is doing, what was kept and what died recently, and the budget used. Don't paste the whole board.
