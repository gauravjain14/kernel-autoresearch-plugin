"""SASS audit of the module images a candidate loaded (saved by the worker in profiling modes).

Per kernel: registers, stack, local and shared memory, instruction count, opcode mix, and flags for things that
cost time and are visible only in SASS: spills (local loads/stores), GPU-scope fences (MEMBAR.ALL.GPU, often from a
cluster or system fence written as a release), scalar global loads where 128-bit loads would do.
"""
import collections
import glob
import os
import re
import shutil
import subprocess

OPCODE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)", re.M)
RES = re.compile(r"Function (\S+):\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def _tool(name):
    return shutil.which(name) or next(iter(sorted(glob.glob(f"/usr/local/cuda*/bin/{name}"), reverse=True)), None)


def _run(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return r.stdout if r.returncode == 0 else ""


def audit_file(path):
    cuobjdump = _tool("cuobjdump")
    if not cuobjdump:
        return [{"file": os.path.basename(path), "error": "cuobjdump not found"}]
    res = {m.group(1): m.groups()[1:] for m in RES.finditer(_run([cuobjdump, "-res-usage", path]))}
    sass = _run([cuobjdump, "-sass", path])
    if not sass:
        nvdisasm = _tool("nvdisasm")
        sass = _run([nvdisasm, path]) if nvdisasm else ""
    if not sass:
        return [{"file": os.path.basename(path), "error": "could not disassemble (toolkit older than the cubin?)"}]
    out = []
    for block in re.split(r"\n\s*Function : ", sass)[1:]:
        name = block.split("\n", 1)[0].strip()
        ops = collections.Counter(OPCODE.findall(block))
        base = collections.Counter()
        for op, n in ops.items():
            base[op.split(".")[0]] += n
        reg, stack, shared, local = (int(x) for x in res.get(name, (0, 0, 0, 0)))
        flags = []
        spills = base["LDL"] + base["STL"]
        if spills:
            flags.append(f"{spills} local-memory loads/stores (register spills or stack arrays)")
        gpu_fence = sum(n for op, n in ops.items() if op.startswith("MEMBAR") and ("GPU" in op or "SYS" in op))
        if gpu_fence:
            flags.append(f"{gpu_fence} GPU/system-scope MEMBAR")
        ldg = {op: n for op, n in ops.items() if op.startswith("LDG")}
        wide = sum(n for op, n in ldg.items() if ".128" in op or ".64" in op)
        narrow = sum(ldg.values()) - wide
        if narrow > 4 and narrow > 2 * wide:
            flags.append(f"static count: {narrow} scalar vs {wide} 64/128-bit global loads (all code paths)")
        out.append({"file": os.path.basename(path), "kernel": name, "regs": reg, "stack": stack, "shared": shared,
                    "local": local, "n_inst": sum(ops.values()),
                    "top_ops": base.most_common(10), "flags": flags,
                    "mma": sum(n for op, n in base.items() if op in ("HMMA", "IMMA", "HGMMA", "QGMMA", "IGMMA",
                                                                    "UTCHMMA", "UTCQMMA"))})
    return out


def dump_dir(cubin_dir, limit=400_000):
    """Full SASS of every saved image, for reading: resource usage, then the instructions."""
    cuobjdump = _tool("cuobjdump")
    if not cuobjdump:
        return "cuobjdump not found"
    parts = []
    for p in sorted(glob.glob(os.path.join(cubin_dir, "*.cubin"))):
        parts.append(f"===== {os.path.basename(p)}\n" + _run([cuobjdump, "-res-usage", p]) + _run([cuobjdump, "-sass", p]))
    text = "\n".join(parts) if parts else "No module images were loaded (PyTorch or library kernels only)."
    return text[:limit]


def audit_dir(cubin_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(cubin_dir, "*.cubin"))):
        out += audit_file(p)
    return out


def markdown(audits):
    if not audits:
        return "No module images were loaded (PyTorch or library kernels only): no SASS audit.\n"
    lines = ["| kernel | regs | stack | smem | instructions | top opcodes | flags |", "|---|---|---|---|---|---|---|"]
    for a in audits:
        if "error" in a:
            lines.append(f"| {a['file']} | | | | | | {a['error']} |")
            continue
        ops = ", ".join(f"{op} {n}" for op, n in a["top_ops"][:6])
        lines.append(f"| `{a['kernel'][:40]}` | {a['regs']} | {a['stack']} | {a['shared']} | {a['n_inst']} | {ops} | "
                     f"{'; '.join(a['flags']) or '-'} |")
    lines.append("")
    return "\n".join(lines)
