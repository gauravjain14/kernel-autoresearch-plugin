"""ncu: is each kernel of one call at its hardware limit, and if not, what holds it back.

GPU side (`ncu_call`, run inside a worker under ncu): two warm-up calls, then one call inside the NVTX range
`kar_ncu`; ncu profiles only the kernels launched in that range, with caches flushed before each kernel and clocks
left alone (so durations are comparable with the bench).

Parent side: the ncu command line (unknown metrics are dropped and the run retried, since names vary by GPU
generation), the raw-page CSV, and `summarize` -> per kernel: duration, DRAM and SM throughput as % of peak,
occupancy, registers, cache hit rates, and the top warp-stall reasons.
"""
import csv
import io
import os
import re
import subprocess

RANGE = "kar_ncu"
METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread", "launch__block_size", "launch__grid_size",
    "launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic", "launch__waves_per_multiprocessor",
    "smsp__issue_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "smsp__cycles_active.avg", "sm__cycles_elapsed.avg.per_second",
    "regex:smsp__average_warps_issue_stalled_.*_per_issue_active\\.ratio",
]


def ncu_call(fn, R, shape, seed, clone_args):
    import torch
    a = clone_args(R.make(seed=seed, **shape["args"]))
    for _ in range(2):
        fn(*a)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(RANGE)
    fn(*a)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def command(rep, metrics):
    return ["ncu", "--nvtx", "--nvtx-include", f"{RANGE}/", "--cache-control", "all", "--clock-control", "none",
            "--metrics", ",".join(metrics), "-o", rep, "-f"]


def command_set(rep, section_set="full"):
    return ["ncu", "--nvtx", "--nvtx-include", f"{RANGE}/", "--cache-control", "all", "--clock-control", "none",
            "--set", section_set, "--import-source", "yes", "-o", rep, "-f"]


def details(rep, limit=200_000):
    """ncu's own per-kernel section report (every section with its rule notes), as text."""
    pre = os.path.join(os.path.dirname(rep), "ncu_details.txt")
    if os.path.exists(pre) and os.path.getsize(pre) > 0:
        with open(pre, errors="replace") as f:
            return f.read()[:limit]
    r = subprocess.run(["ncu", "--import", rep, "--page", "details", "--print-details", "all"], capture_output=True,
                       text=True, timeout=600)
    if r.returncode:
        r = subprocess.run(["ncu", "--import", rep, "--page", "details"], capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(f"ncu --import failed: {r.stderr[-1000:]}")
    return r.stdout[:limit]


def unknown_metrics(log):
    return set(re.findall(r'[Ff]ailed to find metric "?([A-Za-z0-9_.:]+)', log)) | \
        set(re.findall(r'[Uu]nknown metric[: ]+"?([A-Za-z0-9_.:]+)', log))


def raw_csv(rep):
    pre = os.path.join(os.path.dirname(rep), "ncu_raw.csv")      # exported where it was recorded (remote jobs)
    if os.path.exists(pre) and os.path.getsize(pre) > 0:
        with open(pre, errors="replace") as f:
            text = f.read()
    else:
        r = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True,
                           timeout=600)
        if r.returncode:
            raise RuntimeError(f"ncu --import failed: {r.stderr[-1000:]}")
        text = r.stdout
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return []
    header, units = rows[0], rows[1]
    out = []
    for row in rows[2:]:
        d = {}
        for h, u, v in zip(header, units, row):
            d[h] = (v, u)
        out.append(d)
    return out


def _num(d, key):
    v = d.get(key, ("", ""))[0].replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def _dur_us(d):
    v, u = d.get("gpu__time_duration.sum", ("", ""))
    try:
        x = float(v.replace(",", ""))
    except ValueError:
        return None
    return x / 1e3 if u in ("nsecond", "ns") else x * 1e3 if u in ("msecond", "ms") else x


def summarize(kernels):
    out = []
    for d in kernels:
        stalls = {}
        for k, (v, _) in d.items():
            m = re.match(r"smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio", k)
            if m and "_pipe_" not in m.group(1):  # *_pipe_* metrics split a parent reason; counting both doubles it
                try:
                    stalls[m.group(1)] = float(v.replace(",", ""))
                except ValueError:
                    pass
        tot = sum(stalls.values()) or 1.0
        top = sorted(stalls.items(), key=lambda kv: -kv[1])[:4]
        out.append({
            "kernel": d.get("Kernel Name", ("?", ""))[0],
            "duration_us": _dur_us(d),
            "dram_pct": _num(d, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "sm_pct": _num(d, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "dram_read_MB": (_num(d, "dram__bytes_read.sum") or 0) / 1e6,
            "dram_write_MB": (_num(d, "dram__bytes_write.sum") or 0) / 1e6,
            "l2_hit_pct": _num(d, "lts__t_sector_hit_rate.pct"),
            "l1_hit_pct": _num(d, "l1tex__t_sector_hit_rate.pct"),
            "occupancy_pct": _num(d, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            "occupancy_theory_pct": _num(d, "sm__maximum_warps_per_active_cycle_pct"),
            "issue_active_pct": _num(d, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "tensor_active_pct": _num(d, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
            "regs": _num(d, "launch__registers_per_thread"),
            "block": _num(d, "launch__block_size"), "grid": _num(d, "launch__grid_size"),
            "smem_bytes": (_num(d, "launch__shared_mem_per_block_static") or 0) +
                          (_num(d, "launch__shared_mem_per_block_dynamic") or 0),
            "waves": _num(d, "launch__waves_per_multiprocessor"),
            "sm_clock_mhz": (_num(d, "sm__cycles_elapsed.avg.per_second") or 0) / 1e6,
            "stalls": [(k, round(100 * v / tot)) for k, v in top],
        })
    return out


def markdown(summary, shape):
    lines = [f"**ncu, {shape}** (one call, caches flushed before each kernel, clocks unlocked)", "",
             "| kernel | us | DRAM % | SM % | issue % | occ % (theory) | regs | block x grid | L2 hit % | top stalls |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, fmt="{:.0f}"):
        return fmt.format(x) if x is not None else "-"

    for k in summary:
        st = ", ".join(f"{n} {p}%" for n, p in k["stalls"])
        lines.append(f"| `{k['kernel'][:50]}` | {f(k['duration_us'], '{:.2f}')} | {f(k['dram_pct'])} | {f(k['sm_pct'])} "
                     f"| {f(k['issue_active_pct'])} | {f(k['occupancy_pct'])} ({f(k['occupancy_theory_pct'])}) | "
                     f"{f(k['regs'])} | {f(k['block'])} x {f(k['grid'])} | {f(k['l2_hit_pct'])} | {st} |")
    lines.append("")
    return "\n".join(lines)
