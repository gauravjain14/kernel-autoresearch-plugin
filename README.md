# kernel-autoresearch

A plugin for Claude Code and Codex that makes a GPU kernel or workload faster from one prompt.
By Gaurav Jain ([gauravjain14](https://github.com/gauravjain14)).

The prompt gives the kernel task and the GPU:

- **The task:** the operation to make fast. Describe it in words (what it computes, the data types, the sizes), or
  point at a folder with your code or a challenge's files (`--from FOLDER`).
- **The GPU:** this machine's, or a remote one (`--backend modal:H200:1`, `--backend ssh:HOST`).
- **Optional:** rules (precision, size limits), an agent budget, and a judge or leaderboard to send results to.

```
/kar:kar "Make a 4-bit quantized matrix-vector product fast on this machine's GPU: packed 4-bit weights with a
scale and zero-point per group of 128, BF16 input and output, sizes 4096x4096 to 4096x16384, batch 1 and 8.
Agent budget $25."
```

A team of agents then:

1. writes a reference for the operation and a correctness check, and proves the check rejects cheating kernels;
2. measures the GPU and works out how fast each input shape could run at best (its roofline floor);
3. profiles the current best kernel (nsys, ncu, SASS) and reads the CUDA and PTX documentation;
4. proposes changes, with an estimate of each one's gain before it is built;
5. builds each change and tests it on the GPU;
6. keeps the changes that measure faster and stay correct, and combines them;
7. audits the best kernel for bugs the checks miss, and fixes them;
8. repeats until the budget is spent or the kernel is close to its floor.

If you have a judge or a leaderboard, results can be sent to it.

The method was developed during the Muse Open Kernel Challenge (H200s, five problems, 7th on the kernels
leaderboard).

## Quick start

The repository is the plugin: installing the plugin brings the whole harness with it.

**Claude Code**
```
/plugin marketplace add gauravjain14/kernel-autoresearch-plugin
/plugin install kar@kernel-autoresearch
/kar:kar "make the attention kernel in ./attn fast on H100" --from ./attn
```
(also `/kar:kar-status`, `/kar:kar-judge`, `/kar:kar-report`, `/kar:kar-join`, `/kar:kar-new`, `/kar:kar-run`; the plugin puts
`kar` on the shell's PATH.)

**Codex**
```
codex plugin marketplace add gauravjain14/kernel-autoresearch-plugin
codex plugin add kar@kernel-autoresearch
```
then ask for it ("use kar to make ./attn fast on H100"); the kar-harness skill runs `kar go`. Grok and other agents
read the same skill folder, `skills/kar-harness/`.

**A shell, no plugin**
```
git clone https://github.com/gauravjain14/kernel-autoresearch-plugin && cd kernel-autoresearch-plugin
bin/kar doctor                          # is this machine ready (Python, numpy, torch, cuda-python, GPUs, nsys/ncu,
                                        # agent CLIs, Modal)? Each problem comes with its fix.
bin/kar go "Row-wise softmax over a BF16 [rows, vocab] matrix; vocab 32000 or 128256; rows 1, 8, 64; FP32 math"
bin/kar go "make the MoE layer in this challenge fast on H200" --from ./challenge --backend modal:H200:3 --judge judge.toml
```
(`pip install -e .` puts `kar` on PATH instead.)

What you need on your machine: Python 3.11+ with numpy, one agent CLI (Claude Code, Codex, Grok or Cursor, logged in),
and a GPU: this machine's (torch, cuda-python and the CUDA toolkit's profilers), a Modal account (`pip install modal`,
`modal token new`), or an ssh host.

`kar go` checks the machine, has an agent write the task (mirroring the `--from` folder when given: its reference,
shapes, score formula, rules, judge), validates it on the target GPU, shows you what "correct" means, proves the
evaluator, and starts a team of four. With no judge, the team keeps closing the gap to speed of light until its
budget or a plateau ends the run. `kar report RUN` re-checks and re-measures the result. Any MCP client can use
`RUN/ws/kar-mcp` (the lane commands as tools).

## One prompt, measured

One `/kar` prompt typed in an empty folder, through the installed plugin, on an RTX 4060 Ti with Claude Code agents
(2026-11-11): "Make a 4-bit quantized matrix-vector product fast on this machine's GPU ..." (packed 4-bit weights,
per-group scale and zero-point, BF16 in and out; 4096x4096, 8192x4096, 4096x16384; batch 1 and 8).

| | result |
|---|---|
| task written and validated by the session, evaluator proved (11 cheating kernels rejected) | about 5 minutes |
| first kept kernel (fused tensor-core kernel) | 49.6x the PyTorch baseline, 96.0% of speed of light, 30 minutes in |
| the team's auditor | proved that kernel wrong on extreme finite inputs, with a test the reference passes |
| final kernel (fixed, 2.6% slower; the faster one stays on record with what it fails) | **48.4x, 93.7% of speed of light**, passes the stress suite and the team's 2 registered tests |
| cost | $37 of agent use at list prices, 0.11 GPU-hours, 1.6 hours |

## What is generic, and how

| part | how it adapts |
|---|---|
| the work | `kernel_fn` is anything the task times: one op, a fused chain, a layer, a model step; CUDA C++ via NVRTC, Triton, PyTorch |
| the score | a weighted geomean of per-shape speedups, or the task's own formula (`score.py`: loads, throughput, total time); A/B confidence intervals bootstrap it jointly |
| the floor | a roofline from the task's op model (compulsory bytes and flops per pipe) at this GPU's measured bandwidth and peaks, plus a call's fixed cost; the profile page shows intensity against ridge points and % of each roof |
| timing | CUDA-graph replays, or eager calls (`[bench] mode = "eager"`) for work that cannot be captured |
| the GPU | this machine, Modal (`--backend modal:H200:3`, any GPU type, the challenge's own image), or ssh hosts (`ssh:box:0,1`) |
| the judge | optional: any `check` / `submit` command line, a parse hook, quotas; noise from repeats, judge-vs-local fit, auto checks of stacks past a bar, scored submits only with your approval |
| the agents | presets for claude, codex, grok, cursor; `--agent claude,codex` mixes a team; `--agent-cmd` for anything else; or by hand with `kar serve` |
| the docs | `docs find "TERM"`: the PTX ISA, CUDA guides, this GPU's tuning guide, Nsight guides, Triton, PyTorch CUDA notes, fetched by the harness from an allowlist |

## The evaluator

```
python3 -m kar init tasks/toy/rmsnorm                       # prove the evaluator on a task
python3 -m kar eval tasks/toy/rmsnorm my_kernel.py          # lint, smoke, correctness
python3 -m kar ab   tasks/toy/rmsnorm --cand my_kernel.py   # time against the baseline in one job
```

`kar init` freezes the task and the evaluator (sha256 of their files; every later job checks both), then requires:

- the task's reference, its baseline and its known-good kernels to be **accepted**;
- every canary (a known-cheating kernel) to be **rejected** at the expected stage:

  | canary | cheat | caught by |
  |---|---|---|
  | `cache_by_ptr` | returns a cached output for the same input pointers | 2nd call, same buffers, new data |
  | `scored_only` | correct only on the scored shapes | nearby shapes |
  | `low_precision` | computes from FP8-rounded inputs | tolerance |
  | `graph_skip` | launches nothing during graph capture | graph replay with a NaN-poisoned output |
  | `side_stream` | works on an unjoined side stream | output copied on the caller's stream |
  | `stateful` | workspace never reset | 2nd call |
  | `mutate_input` | scales an input in place | input comparison |
  | `timer_tamper` | patches `torch.cuda.Event.elapsed_time` | tamper snapshot |
  | `tf32_flip` | turns on TF32 globally (loosens the reference too) | tamper snapshot of numeric settings |
  | `blob` | ships an encoded, compressed payload | lint |
  | task-specific (`gemv`: `bf16_accum`) | BF16 accumulation, inside the tolerance | error ratio vs the reference (4.4x) |

- the baseline timed against itself to read zero within its confidence interval.

## How it measures

- **Correctness** runs on every scored and nearby shape:
  - two eager calls, the second on new data in the same buffers;
  - an input-mutation check;
  - CUDA-graph capture and two replays on new data, with the output poisoned before each replay.

  Errors are measured against an fp64 reference. Besides atol/rtol, the candidate's mean error may be at most 1.5x the
  error of the task's own reference at the prescribed precision.
- **Timing** uses CUDA-graph replays after a cold-L2 read flush:
  - Every variant, and every control, runs as two module copies × four memory placements.
  - All variants get fresh inputs every rep. Each pair of reps runs once in a random order and once reversed.
  - Effects come with a bootstrap CI that resamples placements as well as rep pairs. Identical code at a fixed
    placement measured 1.9% apart from itself on the first run, so ignoring placement produces fake wins.
- **Isolation:** each job is its own process, under a per-GPU lock, with a time limit per phase (a hung kernel costs
  one phase). nvidia-smi is recorded before and after.

## Speed of light and profiles

```
python3 -m kar calibrate                                     # measure this GPU (cached per GPU model + driver)
python3 -m kar gap tasks/toy/gemv --cand my_kernel.py        # floor per shape, ranked by score gain at the floor
python3 -m kar profile tasks/toy/gemv my_kernel.py           # one page: check, gap table, nsys, ncu, SASS
```

- **Calibration** measures, rather than reading the spec sheet:
  - DRAM read, write and copy bandwidth;
  - L2 read bandwidth;
  - FP32 FMA and tensor-core GEMM throughput per dtype;
  - the empty CUDA-graph replay cost;
  - how many bytes of writes L2 absorbs inside a timed call. Writes that fit finish as dirty lines and reach DRAM
    after the call ends; on the RTX 4060 Ti about 17.6 MB of the 32 MB L2.
- **Floors** come from the task's op model (bytes read, bytes written, flops per pipe):

  floor = max(memory time, math time) + graph overhead

  Only writes beyond the absorbed amount count toward memory time. The gap table ranks shapes by how much the score
  would gain if each reached its floor.
- **`kar profile`** writes `profile.md`:
  - nsys per-kernel times for each shape, with PyTorch's own replay overhead kernels reported separately;
  - ncu on the shapes with the most to gain: DRAM and SM throughput as % of peak, occupancy, registers, top warp stalls;
  - a SASS audit of every module the candidate loaded: registers, spills, GPU-scope fences, opcode mix.

## The loop, as a team, with any agent

An agent needs only shell commands and file edits, so a team can be Claude Code, Codex, Cursor, Grok, OpenCode,
Aider or a person, or a mix ([Status](#status) says which have been tested).

```
kar start <task> --lanes 4               # run folder: init (prove the evaluator), calibrate, profile the baseline
kar run <task|run> --lanes 4 --agent claude    # a team of 4 sessions at once; --agent claude,codex mixes them
kar serve <run>                          # or by hand: open agents in <run>/ws and tell each "run ./kar next"
kar status <run>  |  kar report <run>
```

Work is cut into **lanes**, many at once, each with one role. The scheduler hands them out:

| lane | does | hands back |
|---|---|---|
| survey | one lens on the current best (structure, memory, instructions, latency): `ncu --set full`, full SASS, bound probes, variants | findings with numbers and what they are worth |
| plan | turns the round's findings, the board, the wins and the dead ends into a portfolio of hypotheses | a plan; one that cannot reach the target without a structural change is refused |
| build | implements one hypothesis, tests it like a user would | a submit, or an abandon with the numbers |
| stack | ports a win judged against an older best onto the current best | a submit |
| audit | reads a new best for what tests cannot reach, proves suspicions with its own tests | ok, or a hazard (a fix jumps the queue) |

Agents coordinate through the harness: `board` (every lane, its base, its last measurement), `post` (messages
delivered with a lane's next command), lane notes with the verdict on line 1, shared facts and dead ends. Each agent
may use its own subagents inside its lane (Claude Code's are tested to stay inside the lane's permissions).

The team **tests extensively**: `test` runs an agent's own test file on the GPU (inputs the task's checks never use,
compared with the reference); `register` makes it part of the run once the reference passes it, and from then on
every submit must pass it too. The judge runs, in order: lint, a 20 s smoke call, correctness on scored and nearby
shapes (eager, same buffers, CUDA graphs, poisoned outputs), a stress suite (more seeds, concurrent calls on several
streams, back-to-back calls with every output kept as returned), the team's registered tests, then the A/B against
the current best inside one GPU job.

What keeps the result trustworthy whatever the agent:

- **The harness alone measures,** and its own state lives outside the agents' workspace (`RUN/ws/`), so "write only
  here" is a complete sandbox for any agent, and nothing an agent writes can change what the judge uses.
- **Frozen evaluator and candidates.** Every job checks the evaluator's and the task's hashes; candidates are frozen
  on hand-back and judged from the harness's own copy; kernels may not read files.
- **Hash-chained ledger,** safe under concurrent writers: every decision, cost, message, test and note.
- **The report re-measures.** `kar report` re-runs every gate and the stress suite on the best's frozen file and times
  it in a fresh job; auditors' hazards are listed.
- **Honest stops.** Target reached (confirmed by a longer measurement, after the audit), budget spent (GPU-hours, wall
  clock, submits, sessions, agent dollars or tokens), two plateaus, or nothing left to hand out.

Agent presets (`--agent`), each with the tightest permissions its CLI offers (from its docs and help, 11-10):
`claude` (tested: only the lane's command as a shell command, writes only in the lane folder, subagents on), `codex`
(its sandbox: writes only in the workspace, no network, no GPU; reports tokens, not dollars), `grok` (allowlist of
the lane's command, the user's own ~/.claude and ~/.cursor config switched off), `cursor`. Codex, Grok and Cursor are
not yet run end to end here, since that sends the task's code to another company. Anything else:
`--agent-cmd 'tool ... {prompt_file} ... {ws}'`.

### Results (RTX 4060 Ti, Claude Code with its default model, unattended)

Costs are what Claude Code reports per session at API list prices (`total_cost_usd`); on a subscription login they
are usage, not charges.

One agent at a time:

| task | start | result (re-measured by `kar report`) | kept / dropped | sessions | agent cost | wall | ended by |
|---|---|---|---|---|---|---|---|
| RMSNorm (toy), target 85% | 19.5% of SOL | **90.8%** of SOL, 4.66x the baseline | 1 / 0 | 2 | $1.32 | 6 min | target |
| RMSNorm (toy), target 95% | 19.5% | **92.0%**, 4.76x | 3 / 7 | 14 | $12.04 | 54 min | agent budget |
| Softmax, task written by `kar new` from one sentence ($0.47) | 23.7% | **76.4%**, 3.24x | 3 / 5 | 11 | $10.30 | 48 min | agent budget |

Teams of four, the same softmax task, target 90%:

| run | result | team's own tests passed (of 7) | sessions | agent cost | wall | notes |
|---|---|---|---|---|---|---|
| team 1 (4 survey lenses, first version) | **74.7%**, 3.16x | 0 | 18 | $25.69 | 40 min | 8 of 18 sessions were surveys ($13.7 in round 1); an auditor found whole rows of NaN for masked logits after the keep |
| team 2 (2 lenses, `test` and `register`) | **76.2%**, 3.23x | 7 | 14 | $25.52 | 54 min | agents registered 7 tests (masked rows, extremes, fresh threads, branch edges) that gated every later submit; a +47% win waited for a stacker when the budget ran out |
| the single agent's kernel, on team 2's tests | 76.4% | 3 | | | | crashes when called from a fresh thread (CUDA_ERROR_INVALID_CONTEXT): 4 tests fail |

- Same speed, different kernels: team 2's is the only one that is correct on masked rows, extreme values and calls
  from other threads. The single agent's kernel also waits across blocks on undocumented block-dispatch order.
- The fresh-thread failure came from the NVRTC example we hand to agents. An auditor proved it with a registered
  test; the example and the toy tasks' known-good kernels are fixed, and the stress suite now calls every kernel
  from a fresh thread.
- Neither team reached the target on $25. The team's cost goes into breadth (surveys, audits, tests) that one agent
  never does; whether that buys a higher ceiling with more budget is the next measurement.
- Agents found harness bugs on every run: an NVRTC flag and the thread-context bug in our example, a floor 4% too
  high, a missing tool (`bound`), `--help` hanging a request (an agent waited 23 minutes), and a fix starved by
  budget.

## Layout

```
kar/task/spec.py          task.toml loader and validation
kar/evaluate/             worker (GPU side), check, stress, agent tests, bench, stats, compare, lint, integrity, canaries
kar/backends/             local GPU runner with per-phase watchdog; GPU slot locks
kar/sol/                  calibration microbenchmarks, floors and the gap table
kar/profile/              nsys, ncu (summary and --set full) and SASS (audit and full dump)
kar/runner.py, cli.py     lint -> smoke -> check -> stress pipeline, same-job A/B, calibrate, gap, profile; `kar`
kar/checks.py             init proof and the profile page, shared by the CLI and runs
kar/run/                  session.py (state, scheduler, lanes, board), roles.py (each lane's commands, the judge),
                          commands.py, client.py (the lane command), server.py (request queue), store.py, records.py
kar/agents/               drive.py (the team driver and agent presets), prompts/ (plain files: AGENTS.md, METHOD.md,
                          one brief per role, the NVRTC and test examples)
kar/mcp.py                the lane commands as MCP tools
kar/newtask.py            `kar new`: task skeleton, instructions, ./kar-task validate
.claude-plugin/, .codex-plugin/   plugin manifests and marketplaces (Claude Code; Codex's is .agents/plugins/): the
                          repository is the plugin, so the engine ships with it
commands/, skills/, bin/  /kar and the other slash commands, the kar-harness skill, the `kar` launcher
tasks/toy/                rmsnorm and gemv (BF16), each with a known-good NVRTC kernel
tests/                    unit tests; KAR_GPU_TESTS=1 adds a team run on the GPU without agents
tools/validate_bench.py   GPU self-test of the A/B statistics (null runs, known-effect runs)
```

Requirements: Python 3.11+ and numpy for the harness; PyTorch with CUDA and cuda-python (NVRTC) where the GPU is (the
Modal backend installs both in its image).

## Status

| | tested |
|---|---|
| one prompt to a finished kernel, Claude Code agents, local GPU (RTX 4060 Ti, sm_89) | yes, the runs above |
| a task written from a folder of existing code or a challenge's files (`--from`) | not yet in a full run |
| Modal backend (H200, sm_90): image build, `kar doctor`, `kar eval`, `kar ab` | yes, single jobs (2026-11-11) |
| Modal backend: `kar init`, the profilers, a whole team run, several containers at once | not yet |
| ssh backend | not yet |
| Codex, Grok or Cursor as the team's agents (the plugin installs in Codex) | not yet |
| an external judge or leaderboard | not yet (unit-tested only) |

## Author and license

Written by Gaurav Jain ([gauravjain14](https://github.com/gauravjain14)). If you use kernel-autoresearch, or publish a kernel it
produced, please credit this repository: https://github.com/gauravjain14/kernel-autoresearch-plugin

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
