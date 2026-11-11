# Kernel optimization run

This folder is the workspace of one kar run: a team of agents making a GPU kernel faster against a frozen evaluator.
You are one of them. The harness keeps the score, the history and every decision; the team supplies findings,
hypotheses and kernels.

## How the team works

The work is split into lanes, and each agent works in one lane at a time:

| lane | what it does | hands back with |
|---|---|---|
| survey | reads the code and the profile through one lens (memory, instructions, latency, structure) and measures | `findings FILE` |
| plan | turns the findings, the dead ends and the board into hypotheses | `plan FILE` |
| build | implements one hypothesis | `submit FILE` or `abandon "why"` |
| stack | ports wins that were judged against an older best onto the current best | `submit FILE` or `abandon "why"` |
| audit | reads a new best for hazards the tests cannot reach and stress-tests it | `audit ok\|hazard "text"` |

Several lanes run at the same time. Your lane's command is `./lanes/<id>/kar` (your prompt names it; if you are
working by hand, `./kar next` gives you a lane). Your lane folder `lanes/<id>/` is the only place you write.

## The loop

1. Run `./lanes/<id>/kar next`. It prints your step, the numbers you need and the exact commands.
2. Do that step. `./lanes/<id>/kar board` shows what every other lane is doing.
3. Hand back your result with your role's command. The harness decides, and says so. Then end the session.

GPU commands can take several minutes, because the team's GPU jobs queue for one GPU so every timing is clean. If a
command stops early with "still running", run `./lanes/<id>/kar wait`.

## Working as a team

- Read before you start: `board`, the other lanes' `lanes/*/NOTES.md`, `notes/facts.md` and `notes/dead_ends.md`.
  Don't redo what another lane is doing or has measured.
- Keep `lanes/<id>/NOTES.md` current: what you tried, with numbers. Its first line is your verdict for the others
  (the harness writes it when you hand back).
- `note "fact, with numbers"` records something every later agent should know. `post <lane|all> "message"` tells
  a lane something now (it arrives with that lane's next command): a measurement that affects its idea, a conflict
  between two changes, a hazard you saw.
- If your agent can start subagents, use them inside your lane: to read the profile, the SASS and other lanes'
  notes in parallel, or to draft variants while you measure. Only `./lanes/<id>/kar` touches the GPU; keep every
  GPU command in your own lane.

## Rules

- Write only in your lane folder. Everything else here belongs to the harness or to other lanes: the task, the
  evidence, the candidates (read-only copies), the notes.
- Use the GPU only through your lane's command. Do not run Python, nvcc, profilers or GPU programs yourself.
- A kernel is one self-contained Python file that defines `kernel_fn` with the same arguments and result as the
  task's reference (`task/reference.py`). It may use torch, triton, and CUDA C++ compiled at import time by NVRTC
  through cuda-python (`help/nvrtc_example.py` shows how). It may not use precompiled binaries, encoded payloads,
  file reads, the network, or imports from this harness.
- Write the code yourself. Do not copy code from the internet, other projects or other people's kernels. If you
  believe outside code is needed, say so in your notes and do not use it.
- Follow the task's precision rules (`task/README.md`). The checker compares your error with the reference's own
  error, so a lower-precision shortcut fails.
- A kernel must be safe the way a library op is: correct on every input a caller can pass (masked entries, zeros,
  extremes, not only random data), when called from several streams at once, with no state carried between calls
  and no reliance on the order the hardware dispatches blocks in. The stress suite, the team's registered tests and
  the auditors check this.
- Test like a user. `test` runs a test of your own on the GPU (`help/test_example.py`); `register` makes it part of
  the run, so every later kernel must pass it. Finding a failure is a contribution, not a detour.
- Check what the hardware and the ISA guarantee before relying on it: `docs find "TERM"` searches the PTX ISA, the
  CUDA guides, this GPU's tuning guide and the Nsight guides. `help/PITFALLS.md` lists the traps that cost us most.
- The harness decides what is kept, from its own measurement inside one GPU job against the current best.
  `try` numbers guide you; only `submit` counts.
- `METHOD.md` is how we work. Read it once per session.
