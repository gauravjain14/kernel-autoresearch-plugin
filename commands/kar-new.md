---
description: Write a GPU kernel-optimization task from a plain-language request
argument-hint: <dir> "<what to optimize, the shapes, the precision>"
allowed-tools: Bash(kar *), Read, Write, Edit, Glob, Grep
---
Create a kar task folder from this request: $ARGUMENTS

`kar` is this plugin's command, on the shell's PATH (else `${CLAUDE_PLUGIN_ROOT}/bin/kar`).

1. Run `kar new <dir> "<request>"` with the folder and request from the arguments. It writes DESCRIPTION.md,
   AGENTS.md (the task-writing instructions) and a skeleton.
2. Read `<dir>/AGENTS.md` and `<dir>/DESCRIPTION.md`, then write the task files yourself, following AGENTS.md.
3. Run `kar validate <dir>` after each change until it prints TASK OK. Record every choice the request left open
   under "Assumptions" in DESCRIPTION.md.
4. Tell the user to review `reference.py` (it defines correct) and the assumptions, and show the next command:
   `/kar:kar-run <dir> --lanes 4`.
