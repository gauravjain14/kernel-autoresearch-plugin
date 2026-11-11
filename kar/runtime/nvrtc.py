"""NVRTC compile + driver-API launch on torch's current stream, for the harness's own kernels (calibration
microbenchmarks). Candidates carry their own loader so each stays a single self-contained file."""
import ctypes
import struct

import torch
from cuda.bindings import driver, nvrtc


class Module:
    def __init__(self, src, opts=()):
        torch.zeros(1, device="cuda")  # the primary context must exist and be current
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"kar.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        arch = f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}"
        o = [arch.encode(), b"-std=c++17"] + [x.encode() for x in opts]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        self.image = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, self.image)
        err, self.mod = driver.cuModuleLoadData(self.image)
        _check(err, "cuModuleLoadData")
        self.fns = {}

    def fn(self, name):
        if name not in self.fns:
            err, f = driver.cuModuleGetFunction(self.mod, name.encode())
            _check(err, f"cuModuleGetFunction({name})")
            self.fns[name] = f
        return self.fns[name]

    def launch(self, name, grid, block, params, smem=0):
        """params: bytes of the kernel's single struct parameter (see pack())."""
        buf = ctypes.create_string_buffer(params, max(len(params), 8))
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        gx, gy, gz = grid if isinstance(grid, tuple) else (grid, 1, 1)
        (err,) = driver.cuLaunchKernel(self.fn(name), gx, gy, gz, block, 1, 1, smem, stream,
                                       ctypes.addressof(argv), 0)
        _check(err, f"launch {name}")


def pack(*vals):
    """Tensors and ints as 8-byte words, floats as doubles: kernels take one struct of u64 / i64 / double."""
    out = b""
    for v in vals:
        if torch.is_tensor(v):
            out += struct.pack("Q", v.data_ptr())
        elif isinstance(v, float):
            out += struct.pack("d", v)
        else:
            out += struct.pack("q", int(v))
    return out


def _check(err, what):
    if int(err):
        raise RuntimeError(f"{what} failed: {err}")
