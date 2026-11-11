---
description: Start a team of agents on a kar task (or continue a run) in the background
argument-hint: <task-or-run> [--lanes N] [--agent claude,codex,...] [--agent-usd D] [--target-sol F]
allowed-tools: Bash(kar *), Read, Write, Edit, Glob, Grep
---
Start a kar team run in the background: $ARGUMENTS

`kar` is this plugin's command, on the shell's PATH (else `${CLAUDE_PLUGIN_ROOT}/bin/kar`).

1. If the user gave no `--agent-usd` or `--gpu-h`, ask what budget to use before starting (a team spends faster
   than one agent). Default to `--lanes 4` if no team size is given.
2. Start it in the background: `kar run $ARGUMENTS --detach`. It returns at once with a log file, and the team
   keeps running after this session ends.
3. A few minutes later, read the log: it names the run folder (`run folder ...`, for a new run) and shows the
   evaluator being proved on the task. Report the run folder and the log file to the user.
4. Tell the user how to follow it (`/kar:kar-status <run folder>`), how to stop it (`kar stop <run folder>`), and how to
   read the result (`/kar:kar-report <run folder>`).
