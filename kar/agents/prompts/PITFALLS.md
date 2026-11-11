# Pitfalls that cost us real time

Each one was measured in our competition work (H200, CUDA 12-13, NVRTC kernels driven from PyTorch) and cost hours or
a ranking. Hardware numbers are for the GPU named; re-measure on yours, and `note` what you find.

## Measuring

- **Placement noise.** Identical code at a fixed memory placement read 1.9% apart from itself; two module copies of
  one kernel differed about 0.3 us on small shapes. The harness times two copies on four placements per variant, in
  alternating order; don't trust any timing taken another way.
- **Instrumented builds lie.** Builds with timing stamps ran 0.5-0.7 us slower, and several gains they showed
  vanished in the real A/B. Stamps explain; the A/B decides.
- **The first module loaded in a process** sometimes behaved differently. The harness loads controls first.
- **Configuration applied only at import.** A kernel that compiled lazily on its first call ran default code in every
  A/B variant. Read configuration and compile at import.
- **Bandwidth from kernel lifetimes is wrong** when kernels wait on a programmatic launch dependency: the lifetime
  includes the wait. Compute bandwidth from ncu or from in-kernel clock stamps.
- **Clock and power move the numbers.** On H200 any tensor-core work capped the SM clock (about 1.75 vs 1.95 GHz) and
  a power-bound case ran at the 700 W cap with 1.4-1.75 GHz; instruction cuts buy clock there. Compare only inside
  one job, and report the clock with throughput claims.
- **A judge run is a draw.** Identical source scored 2.4347x and 2.5175x on two runs (+3.4%). Never decide on one.

## Integrating

- **Stack frozen copies, never live files.** We once shipped a regression by stacking a file that was still being
  edited. The harness freezes every submit.
- **Two versions of one module with the same global names** made the wrapper call itself. Give each version unique
  names when two can coexist in one file.
- **Patches that touch the same line** conflict silently in stacking: check the diff of every pair you combine.
- **Size caps are real:** a whole-layer entry ran at 999,965 bytes against a 1,000,000-byte cap; stacking needed a
  trim pass. Strip comments and dead code (the compiled code is unchanged).

## Compiler (ptxas / NVRTC)

- **Loads sunk below a copy burst.** Straight-line loads of lengths and scales were scheduled after a large
  `cp.async` burst and queued behind it for 4-6 us. Give the first copy's address an opaque dependency on those
  loads, and check the SASS.
- **Guarded load loops** (`for i < N: if (idx < G) { load; use }`) compiled to N dependent branches, N serial L2
  round trips. Clamp the index, weight past-the-end entries 0, issue every load first.
- **wgmma serialization** (ptxas warnings C7511-C7515): registers defined on two paths, accumulators touched while a
  group is pending, zero-init sunk below the first MMA. Read ptxas warnings from the compiler version the target
  uses; ours differed between machines.
- **Cluster fences lower to full-GPU barriers:** `barrier.cluster.arrive.release` and `fence.acq_rel.cluster` became
  `MEMBAR.ALL.GPU`; a `fence.release.sync_restrict::shared::cta.cluster` (PTX 8.6+) with a relaxed arrive was most of
  a 0.5 us win.
- **Division by a non-power-of-two constant** in a hot loop compiled to a division sequence; use running counters.
- **`.nc` loads moved past barriers** (`bar.sync`, mbarrier waits): stash values needed after a barrier in shared
  memory.
- **Register caps:** rewriting index math to "save instructions" under a 128-register cap made ptxas rematerialize
  more.
- **NVRTC options:** NVRTC rejects `-O3`. Compile for the exact target (`sm_90a` for Hopper features).
- **The CUDA context is per thread.** A kernel launched through the driver API from a thread that never touched CUDA
  fails with `CUDA_ERROR_INVALID_CONTEXT`; make the device's primary context current before every launch
  (`help/nvrtc_example.py` does).

## Hardware (H200; re-measure on yours)

- Cold instruction fetch after a cache flush costs about 0.15-1 us per KB of straight-line one-shot code on the
  critical path; code first executed after a large copy burst queues behind it. Keep start and end paths compact.
- `mbarrier.init` serializes per SM, about 25 ns each: 96 inits were 2.4 us on the start path.
- A warp keeps only about 16 `LDGSTS` in flight; bigger per-warp bursts stall issue.
- Every warp of every block doing a shared 16-byte `cp.async.cg` of the same small input made an L2 hot spot (18 us).
- Atomics with return on one address are contention-bound; `red.add` into many shards scales, until 16x more reds
  than a block-merged version saturated L2 atomics.
- Random 128-byte row gathers topped out at about 4.1-4.3 TB/s; gathering in physical-address order took one kernel
  from 3.98 to 4.27 TB/s.
- Cluster co-residency is limited (for example only 15 clusters of 8, 30 of 4) and clusters are placed unevenly.
- A `setmaxnreg` deadlock silently ate 50 minutes of a job: the harness's 20 s smoke call exists for that.

## Rules and robustness

- A late robustness audit found NaNs at one input scale and a hang on short blocks in our best run of one board; it
  was withdrawn on the last evening. Audit nearby shapes, held-out inputs, extremes, larger batches and graph
  capture as soon as a kernel matters, not at the end.
- Precision rules are read literally: a shortcut we thought allowed (FP16 where the reference used BF16) was not.
- Readable source only: no precompiled binaries, no encoded or compressed payloads.
