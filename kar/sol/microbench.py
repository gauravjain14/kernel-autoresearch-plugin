"""GPU calibration: what this GPU really delivers, measured, not the spec sheet.

DRAM read / write / copy bandwidth (streaming kernels over a buffer far larger than L2), L2 read bandwidth (a buffer
a quarter of L2, re-read), FP32 FMA throughput (CUDA cores), tensor-core GEMM throughput per dtype (large cuBLAS
GEMMs through PyTorch), and the cost of an empty CUDA-graph replay (the fixed overhead of every timed call). Each
number is the best of a few launch configurations, median over repeats. Runs inside a worker job (mode "calibrate").
"""
import time

import torch

from kar.runtime.nvrtc import Module, pack

SRC = r"""
typedef unsigned long long u64;
struct P { u64 src, dst; long long n, iters; };

extern "C" __global__ void rd(const P p) {
  const uint4* s = reinterpret_cast<const uint4*>(p.src);
  const long long n = p.n, stride = (long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (long long it = 0; it < p.iters; ++it) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
      uint4 a = __ldcg(s + i), b = __ldcg(s + i + stride), c = __ldcg(s + i + 2 * stride), d = __ldcg(s + i + 3 * stride);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) { uint4 a = __ldcg(s + i); acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x9e3779b9u) reinterpret_cast<unsigned*>(p.dst)[0] = acc;
}

extern "C" __global__ void wr(const P p) {
  uint4* d = reinterpret_cast<uint4*>(p.dst);
  const long long n = p.n, stride = (long long)gridDim.x * blockDim.x;
  const uint4 v = make_uint4(threadIdx.x, blockIdx.x, 1u, 2u);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride) __stcg(d + i, v);
}

extern "C" __global__ void cp(const P p) {
  const uint4* s = reinterpret_cast<const uint4*>(p.src);
  uint4* d = reinterpret_cast<uint4*>(p.dst);
  const long long n = p.n, stride = (long long)gridDim.x * blockDim.x;
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {  // four loads in flight per thread before the stores
    uint4 a = __ldcg(s + i), b = __ldcg(s + i + stride), c = __ldcg(s + i + 2 * stride), e = __ldcg(s + i + 3 * stride);
    __stcg(d + i, a); __stcg(d + i + stride, b); __stcg(d + i + 2 * stride, c); __stcg(d + i + 3 * stride, e);
  }
  for (; i < n; i += stride) __stcg(d + i, __ldcg(s + i));
}

extern "C" __global__ void ffma(const P p) {
  float a[8];
  const float b = 1.0f - 1e-7f * (float)(threadIdx.x & 1), c = 1e-7f * (float)(threadIdx.x & 3);
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] = (float)(threadIdx.x + j);
  for (long long k = 0; k < p.iters; ++k) {
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = fmaf(a[j], b, c);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += a[j];
  if (s == 1.2345f) reinterpret_cast<float*>(p.dst)[0] = s;
}
"""


def _time(fn, reps=7, warm=2):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e-3)
    return sorted(ts)[len(ts) // 2]


def device_info():
    from cuda.bindings import runtime as rt
    p = torch.cuda.get_device_properties(0)
    attr = rt.cudaDeviceAttr

    def get(a):
        err, v = rt.cudaDeviceGetAttribute(a, 0)
        return int(v) if int(err) == 0 else None

    mem_khz, bus = get(attr.cudaDevAttrMemoryClockRate), get(attr.cudaDevAttrGlobalMemoryBusWidth)
    return {"name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
            "l2_bytes": int(getattr(p, "L2_cache_size", 0) or 0), "mem_bytes": p.total_memory,
            "sm_clock_max_mhz": (get(attr.cudaDevAttrClockRate) or 0) / 1e3, "mem_clock_mhz": (mem_khz or 0) / 1e3,
            "bus_bits": bus, "nominal_dram_GBps": 2 * (mem_khz or 0) * 1e3 * (bus or 0) / 8 / 1e9}


def memory(info):
    m = Module(SRC)
    nbytes = int(min(1 << 30, info["mem_bytes"] // 8, max(16 * info["l2_bytes"], 256 << 20)))
    n = nbytes // 16
    a = torch.empty(n * 4, dtype=torch.int32, device="cuda").random_()
    b = torch.empty_like(a)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    sms = info["sms"]
    configs = [(sms * 2, 1024), (sms * 4, 512), (sms * 8, 256), (sms * 16, 256), (sms * 32, 128)]
    out = {}
    for name, kern, moved in (("dram_read_GBps", "rd", nbytes), ("dram_write_GBps", "wr", nbytes),
                              ("dram_copy_GBps", "cp", 2 * nbytes)):
        best = 0.0
        for grid, block in configs:
            src, dst = (a, sink) if kern == "rd" else (a, b)
            t = _time(lambda: m.launch(kern, grid, block, pack(src, dst, n, 1)))
            best = max(best, moved / t / 1e9)
        out[name] = round(best, 1)
    # L2: a buffer a quarter of L2, re-read many times (after the first pass it is L2-resident)
    l2n = max(info["l2_bytes"] // 4, 1 << 20) // 16
    iters = max(1, int(4e9 // (l2n * 16)))
    best = 0.0
    for grid, block in configs:
        t = _time(lambda: m.launch("rd", grid, block, pack(a, sink, l2n, iters)), reps=5)
        best = max(best, l2n * 16 * iters / t / 1e9)
    out["l2_read_GBps"] = round(best, 1)
    # L2 write absorption: writes that fit in L2 complete as dirty lines and reach DRAM after the kernel ends, so a
    # graph-replay timing does not see them. Right after a cold read flush, time a read-only kernel and a copy kernel
    # over the same bytes (0.5-1.5x L2); the copy's extra time pays for only part of its writes at the measured write
    # bandwidth, and the rest was absorbed. Measured on an RTX 4060 Ti: writes beside reads under 0.5x L2 added
    # ~3 us; ~16-18 MB of 32 MB L2 was absorbed at 1x L2. A floor must be a lower bound, so the largest estimate
    # is kept.
    flush = a[: min(a.numel(), 4 * info["l2_bytes"] // 4)]

    def after_flush(fn):
        flush.sum()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e-3

    est = []
    src = a[a.numel() // 2:]
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5):
        wn = int(mult * info["l2_bytes"]) // 16
        if wn * 16 > src.numel() * 4:
            continue
        tr = sorted(after_flush(lambda: m.launch("rd", sms * 8, 256, pack(src, sink, wn, 1))) for _ in range(9))[4]
        tc = sorted(after_flush(lambda: m.launch("cp", sms * 8, 256, pack(src, b, wn, 1))) for _ in range(9))[4]
        est.append((mult, max(0.0, wn * 16 - (tc - tr) * out["dram_write_GBps"] * 1e9)))
    out["l2_write_absorb_bytes"] = int(max(v for _, v in est)) if est else 0
    out["l2_write_absorb_by_size_MB"] = {f"{k:g}xL2": round(v / 1e6, 1) for k, v in est}

    # FP32 FMA (CUDA cores): 8 independent chains per thread
    iters = 1 << 14
    best = 0.0
    for grid, block in ((sms * 8, 256), (sms * 16, 256), (sms * 4, 1024)):
        t = _time(lambda: m.launch("ffma", grid, block, pack(sink, sink, 0, iters)), reps=5)
        best = max(best, 2 * 8 * iters * grid * block / t / 1e12)
    out["fp32_TFLOPs"] = round(best, 2)
    return out


def tensor_cores(info):
    out = {}
    n = 8192 if info["mem_bytes"] >= (8 << 30) else 4096
    flop = 2 * n ** 3
    mm = torch.backends.cuda.matmul
    for key, dtype in (("tc_bf16_TFLOPs", torch.bfloat16), ("tc_fp16_TFLOPs", torch.float16)):
        x, y = torch.randn(n, n, device="cuda", dtype=dtype), torch.randn(n, n, device="cuda", dtype=dtype)
        out[key] = round(flop / _time(lambda: torch.mm(x, y), reps=5) / 1e12, 1)
    x, y = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    prev = mm.allow_tf32
    try:
        mm.allow_tf32 = True
        out["tc_tf32_TFLOPs"] = round(flop / _time(lambda: torch.mm(x, y), reps=5) / 1e12, 1)
        mm.allow_tf32 = False
        out["fp32_gemm_TFLOPs"] = round(flop / _time(lambda: torch.mm(x, y), reps=3) / 1e12, 1)
    finally:
        mm.allow_tf32 = prev
    if torch.cuda.get_device_capability() >= (8, 9) and hasattr(torch, "_scaled_mm"):
        try:
            a8 = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            b8 = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()
            one = torch.ones((), device="cuda")
            out["tc_fp8_TFLOPs"] = round(
                flop / _time(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
                             reps=5) / 1e12, 1)
        except Exception as e:  # noqa: BLE001 - FP8 GEMM support varies by build
            out["tc_fp8_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def graph_overhead():
    x = torch.zeros(1, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1)
    flush = torch.empty(64 << 20, device="cuda")
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for i in range(240):
        flush.sum()
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        if i >= 40:
            ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    es = []
    for i in range(240):                        # an eager call: one tiny kernel launched from Python
        flush.sum()
        e0.record()
        x.add_(1)
        e1.record()
        torch.cuda.synchronize()
        if i >= 40:
            es.append(e0.elapsed_time(e1) * 1e3)
    es.sort()
    return {"graph_empty_us": round(ts[len(ts) // 2], 3), "graph_empty_p10_us": round(ts[len(ts) // 10], 3),
            "eager_empty_us": round(es[len(es) // 2], 3)}


def calibrate(spin_s=1.0):
    info = device_info()
    x = torch.empty(64 << 20, device="cuda")
    t_end = time.perf_counter() + spin_s
    while time.perf_counter() < t_end:
        x.sum()
        torch.cuda.synchronize()
    measured = {**memory(info), **tensor_cores(info), **graph_overhead()}
    measured["dram_GBps"] = max(measured["dram_read_GBps"], measured["dram_copy_GBps"])
    return {"device": info, "measured": measured, "torch": torch.__version__, "cuda": torch.version.cuda}
