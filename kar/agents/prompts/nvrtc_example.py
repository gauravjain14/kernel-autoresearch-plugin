"""How a candidate compiles CUDA C++ at import with NVRTC and launches it on torch's current stream, so CUDA-graph
capture records the launch. The example scales a BF16 vector (y = 2x); it is only a template for the mechanics.

Rules that matter: compile once at import, not per call; launch on torch.cuda.current_stream(); allocate the output
with torch; pass arguments as one packed struct; make the device's primary context current before every launch,
because a caller may call from a thread that has never touched CUDA (an auditor found that a kernel built from an
earlier version of this file failed there with CUDA_ERROR_INVALID_CONTEXT).
"""
import ctypes
import struct

import torch
from cuda.bindings import driver, nvrtc

SRC = r"""
typedef unsigned short bf16;
typedef unsigned long long u64;
struct Params { u64 x, y; long long n; };   // one struct of 8-byte fields, packed on the host to match

__device__ __forceinline__ float bf2f(bf16 v) { return __uint_as_float(((unsigned)v) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {          // round to nearest even, NaN kept
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16)(u >> 16);
}

extern "C" __global__ void scale2(const Params p) {
  const bf16* x = reinterpret_cast<const bf16*>(p.x);
  bf16* y = reinterpret_cast<bf16*>(p.y);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < p.n; i += (long long)gridDim.x * blockDim.x)
    y[i] = f2bf(2.0f * bf2f(x[i]));
}
"""


def _check(err, what):
    if int(err):
        raise RuntimeError(f"{what}: {err}")


class _Kernel:
    def __init__(self, src, name):
        torch.zeros(1, device="cuda")                       # make sure the primary context exists and is current
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"k.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        image = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, image)
        err, dev = driver.cuDeviceGet(torch.cuda.current_device())
        _check(err, "cuDeviceGet")
        err, self.ctx = driver.cuDevicePrimaryCtxRetain(dev)  # the context torch uses; kept for every thread
        _check(err, "cuDevicePrimaryCtxRetain")
        err, self.mod = driver.cuModuleLoadData(image)
        _check(err, "cuModuleLoadData")
        err, self.fn = driver.cuModuleGetFunction(self.mod, name.encode())
        _check(err, "cuModuleGetFunction")

    def launch(self, grid, block, params, smem=0):
        err, cur = driver.cuCtxGetCurrent()                 # a fresh thread has no current context
        if int(cur) != int(self.ctx):
            (err,) = driver.cuCtxSetCurrent(self.ctx)
            _check(err, "cuCtxSetCurrent")
        buf = ctypes.create_string_buffer(params, max(len(params), 8))
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = driver.cuLaunchKernel(self.fn, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        _check(err, "cuLaunchKernel")


_K = _Kernel(SRC, "scale2")


def kernel_fn(x):
    y = torch.empty_like(x)
    n = x.numel()
    _K.launch(min(1024, (n + 255) // 256), 256, struct.pack("QQq", x.data_ptr(), y.data_ptr(), n))
    return y
