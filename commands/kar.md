---
description: One prompt to a team of agents making a GPU kernel or workload faster (task, checks, team, judge, report)
argument-hint: "<what to optimize, for which GPU, under which rules>" [--from FOLDER]
allowed-tools: Bash(kar *), Read, Write, Edit, Glob, Grep
---
The user wants a GPU kernel or workload made faster by a kar team: $ARGUMENTS

`kar` is this plugin's command and is on the shell's PATH: run it as plain `kar ...`, one command per call, with no
pipes, `&`, `nohup` or `$(...)` (if `kar` is not found, use `${CLAUDE_PLUGIN_ROOT}/bin/kar`). Do these steps in
order. Keep the user informed in a line or two per step; don't paste long outputs.

1. **Understand the request.** What is optimized (files, a folder, an op described in words), on which GPU, under
   which rules (precision, size limits), against which judge if any (a leaderboard CLI, an evaluation script), with
   what budget. Ask only for what is missing and matters: the agent budget (default $25 at list prices; under about
   $15 a team has room for little more than its first round) and, if the target GPU is not this machine's, the
   backend (`modal:H200:3`, `ssh:host:0,1`).
2. **Check the machine:** `kar doctor` (add `--backend SPEC` for a remote backend). Fix what you can; tell the user
   what you can't.
3. **Write the task.** `kar new TASK_DIR "REQUEST" [--from FOLDER]` (TASK_DIR: `kar-tasks/<short-name>`). Then write
   the task files yourself with the file tools, following `TASK_DIR/AGENTS.md`; if there is `TASK_DIR/source/`, read
   it completely first and mirror what it measures (reference, inputs, shapes, score formula, rules, judge). Run
   `kar validate TASK_DIR [--backend SPEC]` after each change until it prints TASK OK.
4. **Show the user what "correct" means:** `reference.py`, the scored shapes and the Assumptions in DESCRIPTION.md.
   Ask for a yes before going on (unless the user already gave it).
5. **The judge (only if the user has one):** write `TASK_DIR/judge.toml` with a `[judge]` table: the `check` and
   `submit` command lines (with `{file}`), how to read a score (`parse = "judge.py:parse"` with a small parser in the
   task, or a `score: N` line), the quotas, `policy = { check = "auto", submit = "ask" }`. Confirm the commands with
   the user. Scored submissions always wait for the user's approval.
6. **Start the team in the background:**
   `kar go --task TASK_DIR --yes --detach --lanes 4 --agent-usd BUDGET [--backend SPEC] [--judge TASK_DIR/judge.toml]`
   It returns at once with a log file; the team keeps running after this session ends. A few minutes later, read
   the log: it shows the evaluator being proved on the task and then names the run folder (`run folder ...`). If it
   says INIT FAILED, the reason is in the log: fix the task (step 3) and start again.
7. **Tell the user** the run folder and the log, how to follow it (`/kar:kar-status RUN`), how to approve judge
   submissions (`/kar:kar-judge RUN`), how to stop it (`kar stop RUN`), how to continue it with more budget
   (`/kar:kar-run RUN --agent-usd MORE`), and how to get the final re-measured result (`/kar:kar-report RUN`).
