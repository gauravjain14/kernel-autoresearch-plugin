# Method

Rules for making a GPU kernel fast with a team of agents. They come from the Muse Open Kernel Challenge (five
H200 kernels, 7th on the kernels leaderboard), and each rule replaced a mistake made there at least once. `help/PITFALLS.md` has the measured traps (compiler, hardware,
harness); read it before your first kernel.

## 0. The loop

1. Model the score: know what one microsecond on each shape is worth.
2. Measure floors, not just times: the gap to the floor, per phase, is the backlog.
3. Pick the cheapest instrument that answers the question (section 3).
4. Bound every idea before building it.
5. Keep every verified win, however small; stack them; kill directions only on a measured bound.
6. Judge only to confirm, and only stacks.

## 1. The score

- The gap table on the profile page (`evidence/<best>/profile.md`) ranks shapes by the score gained if each reached
  its floor. That ordering beats intuition: in our attention work a 3.8 us call was worth 1.4% of the score per
  microsecond and a 165 us call 0.08%.
- Ask what the target needs: plug plausible per-shape times into the score. It tells you which shapes must move and
  whether small changes can ever get there. If they can't, the plan needs a structural change.

## 2. Floors before fixes

For every shape class, measure (timing-only probes are fine; wrong output is allowed in a `bound` probe):
- **Data floor:** a probe that only moves the bytes, with the real access pattern and launch shape.
- **Skeleton:** the real kernel with the math removed (keep the data dependencies), then with each phase removed in
  turn. The deltas are the backlog, phase by phase.
- **Hardware facts** with microbenchmarks when a phase looks suspicious: bandwidth of a path, atomic throughput,
  in-flight limits, barrier latency, clock behaviour. `note` each one.
- **Phase traces** with `%globaltimer` or `clock64` stamps per block (start, first data landed, compute done,
  publish, out) explain where a latency-bound kernel spends its time. Instrumented builds run slower and their
  gains often vanish: traces explain, the A/B decides.

## 3. Which instrument

| question | instrument |
|---|---|
| Is it faster? | `try` / `submit`: the A/B against controls in one job. Nothing else decides. |
| Is a throughput-bound kernel at its ceiling? | `ncu`: DRAM %, issue active, eligible warps, pipe utilization, against a `bound` probe |
| Where does a latency-bound kernel spend its time? | phase stamps plus phase-removal probes (`bound`) |
| Did the compiler do what I wrote? | `sass`: spills and local memory in loops, serialization warnings, fences, register counts against occupancy |
| What does the hardware or the ISA actually guarantee? | `docs`: the PTX ISA, the CUDA programming guide, the tuning guide for this architecture |
| Will this idea pay? | a bound first (section 4); build only after |
| Nothing points anywhere | stop instrumenting; write the floor table; look for a structural change |

ncu's stall-by-line says where warps park, not what limits: a hot line in a memory-bound kernel is usually waiting on
memory, and removing it moves the stall elsewhere.

## 4. Choose and bound

- Rank by gap times score weight. Start with the biggest, even when the fix is a rewrite. Don't reject an idea
  because it is invasive; reject it because its bound says it can't pay.
- If the small, safe ideas can't reach the target even if every one of them pays, propose a structural change: a
  rewrite, a fusion, a different decomposition. Hours of 0.3% tweaks next to a hot path at 60% of its floor was the
  most expensive mistake we made.
- Give every idea the cheapest test that could kill it, and write the decision rule before the job ("build only if
  the bound saves at least X"): a timing-only build that deletes the phase the idea would shrink, a probe of the new
  primitive alone, or the arithmetic (instructions per element, bytes per SM, cycles per tile) against a measured
  rate. Many of our longest dead ends were multi-hour builds that a one-job bound would have killed.

## 5. Build

- Small shapes are bound by launch and latency: one kernel, no extra launches, no host work per call, no allocation
  you can avoid, enough parallelism to fill the SMs but not more passes than needed. A second graph node cost us
  about 0.75 us end to end.
- Large shapes are bound by bandwidth: wide coalesced loads (16 bytes per thread), enough bytes in flight per SM,
  each input byte read once (keep what you reuse in registers or shared memory), each output written once.
- Specialize to the scored shapes when it pays (compile-time constants, a variant per shape, chosen per call), and
  keep a correct general path for every other shape. Nearby shapes and the team's tests are checked too.
- Follow the precision rules exactly. Accumulate in FP32 where the task says so; the checker compares your error
  with the reference's own.
- Compile once at import, never per call; read every configuration at import. A kernel that compiled lazily on its
  first call once ran default code in every A/B variant.
- Async pipelines need a race audit: a buffer slot released before its data has landed passes most tests and fails
  some. Cross-block ordering must come from atomics, never from blockIdx and the hardware's dispatch order.
- Read the SASS after every structural change (`sass`): spills inside loops, full-GPU memory barriers from fences,
  divisions by non-power-of-two constants, guarded load loops turned into chains of dependent branches.

## 6. Test what the checks don't

- The task's checks use random inputs. Real callers pass masked entries, zeros, extremes, repeated values, call from
  other threads and streams, and hold several results at once. Write a test for what your change could break
  (`help/test_example.py`), run it with `test`, and `register` it when it finds something: then no later kernel can
  bring the bug back.
- A kernel that is fast and wrong on one real input is wrong. Our last-day robustness audit withdrew a run that
  would have ranked #4 (NaNs at one scale, a hang on short blocks); audit early and after every structural change.

## 7. Keep and stack every verified win

- Any change that passes every gate and is faster in the A/B is kept, with no minimum size. Five 0.1% wins are a
  real 0.5% that no single judge run can see.
- Don't kill a direction because its first build missed a bar; kill it on a measured bound.
- Pieces interact: re-measure stacked changes together (three "tie" pieces together were a real win once).

## 8. The judge (when the task has one)

- A judge reading is noisy (about 0.7-1.4% per run on our boards, 2-5% on small shapes). Never decide on one run;
  compare against the incumbent's mean over repeated runs, not its best draw.
- If shapes you didn't touch moved as much as the ones you did, the difference is the machine, not your change.
- Judge stacks, not single pieces: single-piece runs of +0.1-0.4% landed inside the noise on our final night.
- Scored submissions are the user's decision. The harness asks; agents never submit.

## 9. Read the documentation

- `docs find "TERM"` searches the PTX ISA, the CUDA programming and best-practices guides, the tuning guide for this
  GPU's architecture, the Nsight Compute and Systems guides, the driver API, NVRTC, Triton and PyTorch CUDA notes.
  `docs get NAME --section "TERM"` prints a section. Check what an instruction, a memory-ordering rule or a launch
  attribute guarantees before you rely on it, and cite the section in your notes.
- Documentation is for understanding; the code is still yours. Don't copy code from other projects.

## 10. Work as a team

- One lane, one idea. Keep your change to what your hypothesis needs, so it can be stacked with the other lanes'
  wins; a lane that also "tidies" unrelated code makes every later merge harder.
- When two lanes touch the same code, say so early (`post`), and agree who goes first.
- A measurement that kills or helps another lane's idea goes to that lane at once (`post`), not only to the notes.
- Use your agent's subagents for reading in parallel (the profile sections, the SASS, the docs, other lanes' notes
  and kernels) and for drafting variants; `try` several variants in one GPU job rather than one at a time.

## 11. Record

- `note` every measured fact that the next agent should not have to rediscover: "16-byte loads on t4096: +4.1% vs
  8-byte, same registers".
- Keep `lanes/<id>/NOTES.md` current while you work, with a "Resume here" line, so a lane that is taken over does
  not start from zero.
- When you abandon an idea, say why with the numbers, so it is not tried again.
- Re-read the task's rules before each new lever, and write down how you read any gray area.
