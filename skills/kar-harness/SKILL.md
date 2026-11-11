---
name: kar-harness
description: Make a GPU kernel or workload (CUDA, Triton, PyTorch op, layer or model step) faster with a team of agents against a frozen evaluator, closing the gap to its roofline speed of light; optionally sending results to a judge or leaderboard. Use when someone wants to optimize, profile or speed up GPU code, find its gap to speed of light, replicate a kernel-competition workflow, or run, join or check a kar run.
---
# kar: a team of agents closing the gap to speed of light

A run is split into lanes, several at once:

- survey: read the code and the profile through one lens (structure, memory, instructions, latency), measure with
  ncu, nsys, SASS and bound probes, report findings with numbers;
- plan: turn findings into hypotheses with predicted gains, the arithmetic, and a bound test each;
- build: implement one hypothesis, test it like a user would, then submit (the harness judges) or abandon;
- stack: port wins judged on an older best onto the current one;
- audit: read a new best for hazards no test reaches, and prove them with tests that then gate every kernel.

The floor of every shape comes from a roofline: the task's compulsory bytes and flops at this GPU's measured
bandwidth and peaks, plus the fixed cost of a call. The judge runs lint, a 20 s smoke call, correctness on scored and
nearby shapes, a stress suite (concurrent streams, fresh threads, held results, more seeds), the team's registered
tests, then an A/B against the current best inside one GPU job. Agents read the PTX ISA, CUDA guides and tuning
guides through `docs`. GPU jobs run locally, on Modal (`modal:H200:3`) or over ssh. An external judge is optional.

| want | do |
|---|---|
| one prompt to a running team | `/kar:kar "<what, which GPU, rules>" [--from FOLDER]`, or the steps below |
| is this machine ready | `kar doctor [--backend SPEC]` |
| a task only | `/kar:kar-new <dir> "<request>"` |
| a team on an existing task or run | `/kar:kar-run <task|run> --lanes 4 --agent-usd 25` |
| join a run as one more agent | `/kar:kar-join <run folder>` |
| where a run stands | `/kar:kar-status <run folder>` (`kar status RUN --board`) |
| judge results and approvals | `/kar:kar-judge <run folder>` (`kar judge RUN status`) |
| stop a team | `kar stop RUN` |
| the final, re-measured result | `/kar:kar-report <run folder>` (`kar report RUN`) |

## One prompt to a running team, without slash commands (Codex, Grok, any agent)

1. `kar doctor` (add `--backend modal:H200:3` or `ssh:host` when the GPU is not this machine's); fix what it lists.
2. `kar new kar-tasks/<name> "<the user's request>" [--from FOLDER]`, then write the task files following
   `kar-tasks/<name>/AGENTS.md` (read `source/` first if it is there). `kar validate kar-tasks/<name>` until TASK OK.
3. Show the user `reference.py`, the scored shapes and the Assumptions in DESCRIPTION.md; get a yes. Ask the agent
   budget (default $25 at list prices).
4. `kar go --task kar-tasks/<name> --yes --detach --lanes 4 --agent-usd BUDGET [--agent codex] [--backend SPEC]
   [--judge kar-tasks/<name>/judge.toml]`. It returns with a log file; the team runs in the background. The log
   names the run folder after the evaluator is proved.
5. `kar status RUN` to follow, `kar stop RUN` to stop, `kar run RUN --agent-usd MORE --detach` to continue,
   `kar report RUN` for the final re-measured result. The best kernel is `RUN/candidates/<best id>/kernel.py`.

`kar` is the plugin's launcher, `bin/kar` in the plugin folder (two folders above this file); the engine ships
inside the plugin. In Claude Code it is on PATH as `kar` (also `${CLAUDE_PLUGIN_ROOT}/bin/kar`); in Codex, Grok or
another agent, run it by its path. It needs Python 3.11+ with numpy, and says what to install if not. Inside a run an agent's only GPU access is its lane's command, `./lanes/<id>/kar` (or the MCP
server `RUN/ws/kar-mcp`), and it writes only in its lane folder. The workspace's AGENTS.md has the rules, METHOD.md
the method, help/PITFALLS.md the measured traps.
