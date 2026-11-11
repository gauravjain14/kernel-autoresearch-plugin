"""Known-good RMSNorm (the evaluator must accept it): one CTA per row, 16-byte loads, FP32 sum of squares,
round-to-nearest-even BF16 stores. Readable CUDA compiled by NVRTC at import."""
import ctypes
import struct

import torch
from cuda.bindings import driver, nvrtc

SRC = r"""
typedef unsigned short bf16;
typedef unsigned long long u64;
struct Params { u64 x, w, y; long long H; };

__device__ __forceinline__ float bf2f(bf16 v) { return __uint_as_float(((unsigned)v) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16)(u >> 16);
}

extern "C" __global__ void __launch_bounds__(256) rmsnorm(const Params p) {
  const int H = (int)p.H;
  const bf16* x = reinterpret_cast<const bf16*>(p.x) + (size_t)blockIdx.x * H;
  const bf16* w = reinterpret_cast<const bf16*>(p.w);
  bf16* y = reinterpret_cast<bf16*>(p.y) + (size_t)blockIdx.x * H;
  const bool vec = (H % 8) == 0;
  float ss = 0.f;
  if (vec) {
    for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8) {
      uint4 v = *reinterpret_cast<const uint4*>(x + i);
      const unsigned* u = reinterpret_cast<const unsigned*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float a = __uint_as_float(u[j] << 16), b = __uint_as_float(u[j] & 0xffff0000u);
        ss += a * a + b * b;
      }
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) { float a = bf2f(x[i]); ss += a * a; }
  }
  __shared__ float red[32];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) red[wid] = ss;
  __syncthreads();
  if (wid == 0) {
    ss = lane < (int)(blockDim.x >> 5) ? red[lane] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[0] = ss;
  }
  __syncthreads();
  const float rstd = rsqrtf(red[0] / (float)H + 1e-6f);
  if (vec) {
    for (int i = threadIdx.x * 8; i < H; i += blockDim.x * 8) {
      uint4 v = *reinterpret_cast<const uint4*>(x + i);
      uint4 g = *reinterpret_cast<const uint4*>(w + i);
      const unsigned* u = reinterpret_cast<const unsigned*>(&v);
      const unsigned* q = reinterpret_cast<const unsigned*>(&g);
      uint4 o;
      unsigned* r = reinterpret_cast<unsigned*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float a = __uint_as_float(u[j] << 16) * rstd * __uint_as_float(q[j] << 16);
        float b = __uint_as_float(u[j] & 0xffff0000u) * rstd * __uint_as_float(q[j] & 0xffff0000u);
        r[j] = (unsigned)f2bf(a) | ((unsigned)f2bf(b) << 16);
      }
      *reinterpret_cast<uint4*>(y + i) = o;
    }
  } else {
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[i] = f2bf(bf2f(x[i]) * rstd * bf2f(w[i]));
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


_k = Kernel(SRC, "rmsnorm")


def kernel_fn(x, w):
    T, H = x.shape
    y = torch.empty_like(x)
    _k.launch(T, 256, struct.pack("QQQq", x.data_ptr(), w.data_ptr(), y.data_ptr(), H))
    return y
