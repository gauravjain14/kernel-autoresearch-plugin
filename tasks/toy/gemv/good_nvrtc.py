"""Known-good GEMV (the evaluator must accept it): one warp per output column, 16-byte loads of w and x, FP32
accumulation, up to 4 rows of x per launch (more rows are processed in chunks). Readable CUDA compiled by NVRTC."""
import ctypes
import struct

import torch
from cuda.bindings import driver, nvrtc

SRC = r"""
typedef unsigned short bf16;
typedef unsigned long long u64;
struct Params { u64 x, w, y; long long B, N, K, ldy; };

__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ bf16 f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16)(u >> 16);
}

extern "C" __global__ void __launch_bounds__(256) gemv(const Params p) {
  const int B = (int)p.B, N = (int)p.N, K = (int)p.K;
  const int n = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
  if (n >= N) return;
  const bf16* w = reinterpret_cast<const bf16*>(p.w) + (size_t)n * K;
  const bf16* x = reinterpret_cast<const bf16*>(p.x);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if ((K % 8) == 0) {
    for (int k = lane * 8; k < K; k += 256) {
      uint4 wv = *reinterpret_cast<const uint4*>(w + k);
      const unsigned* wu = reinterpret_cast<const unsigned*>(&wv);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        if (b < B) {
          uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)b * K + k);
          const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[b] = fmaf(lo(xu[j]), lo(wu[j]), acc[b]);
            acc[b] = fmaf(hi(xu[j]), hi(wu[j]), acc[b]);
          }
        }
      }
    }
  } else {
    for (int k = lane; k < K; k += 32) {
      float wf = __uint_as_float(((unsigned)w[k]) << 16);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        if (b < B) acc[b] = fmaf(__uint_as_float(((unsigned)x[(size_t)b * K + k]) << 16), wf, acc[b]);
    }
  }
  bf16* y = reinterpret_cast<bf16*>(p.y);
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    float v = acc[b];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0 && b < B) y[(size_t)b * p.ldy + n] = f2bf(v);
  }
}
"""


class Kernel:
    """NVRTC compile at import; launch on torch's current stream (so CUDA-graph capture records it)."""

    def __init__(self, src, name):
        torch.zeros(1, device="cuda")  # the primary context must exist and be current
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"k.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        image = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, image)
        err, dev = driver.cuDeviceGet(torch.cuda.current_device())
        err, self.ctx = driver.cuDevicePrimaryCtxRetain(dev)     # made current before every launch: callers may
        if int(err):                                            # use any thread
            raise RuntimeError(f"cuDevicePrimaryCtxRetain: {err}")
        err, self.mod = driver.cuModuleLoadData(image)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        err, self.fn = driver.cuModuleGetFunction(self.mod, name.encode())
        if int(err):
            raise RuntimeError(f"cuModuleGetFunction: {err}")

    def launch(self, grid, block, params):
        err, cur = driver.cuCtxGetCurrent()
        if int(cur) != int(self.ctx):
            (err,) = driver.cuCtxSetCurrent(self.ctx)
            if int(err):
                raise RuntimeError(f"cuCtxSetCurrent: {err}")
        buf = ctypes.create_string_buffer(params, len(params))
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = driver.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"cuLaunchKernel: {err}")


_k = Kernel(SRC, "gemv")


def kernel_fn(x, w):
    B, K = x.shape
    N = w.shape[0]
    y = torch.empty(B, N, dtype=x.dtype, device=x.device)
    grid = (N * 32 + 255) // 256
    for b0 in range(0, B, 4):
        rows = min(4, B - b0)
        _k.launch(grid, 256, struct.pack("QQQqqqq", x.data_ptr() + 2 * b0 * K, w.data_ptr(),
                                         y.data_ptr() + 2 * b0 * N, rows, N, K, N))
    return y
