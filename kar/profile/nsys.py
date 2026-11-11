"""nsys: where the time of one timed call goes, kernel by kernel.

GPU side (`replay_ranges`, run inside a worker under nsys): for each scored shape, capture the candidate's graph,
then replay it N times, each replay after a cold-L2 flush and inside its own NVTX range with a synchronize before
the range closes, so every kernel of that replay lies inside the range on nsys's timeline.

Parent side: the nsys command line, the sqlite export, and `parse` -> per shape: replay span, kernels per replay with
median durations, and the idle time between kernels (launch gaps), which kernel-only views hide.
"""
import os
import re
import sqlite3
import statistics
import subprocess

NSYS_ARGS = ["profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
             "--force-overwrite=true"]
RANGE_PREFIX = "kar:"


def replay_ranges(fn, R, shapes, reps, seed, capture, clone_args, phase=lambda n: None):
    import torch
    flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda").normal_()
    for si, s in enumerate(shapes):
        phase(f"bench {s['name']}")
        a = clone_args(R.make(seed=seed + si, **s["args"]))
        g = None
        if capture is not None:                 # graph mode; None = eager calls
            g, _ = capture(fn, a)
        else:
            for _ in range(3):
                fn(*a)
        for r in range(reps):
            flush.sum()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{RANGE_PREFIX}{s['name']}:{r}")
            if g is not None:
                g.replay()
            else:
                fn(*a)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        del g, a
        torch.cuda.empty_cache()


def command(base):
    return ["nsys", *NSYS_ARGS, "-o", base]


def export(rep, sqlite_path):
    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 0:   # exported where it was recorded
        return sqlite_path
    r = subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", sqlite_path, rep],
                       capture_output=True, text=True, timeout=600)
    if r.returncode or not os.path.exists(sqlite_path):
        raise RuntimeError(f"nsys export failed: {r.stdout[-500:]} {r.stderr[-1000:]}")
    return sqlite_path


OP_RX = re.compile(r"([A-Za-z_]\w*(?:Ops|Functor\w*|_kernel_cuda|_kernel_impl|OnSelf_\w+))")
DTYPES = {"float", "double", "c10::Half", "c10::BFloat16", "c10::Float8_e4m3fn", "c10::Float8_e5m2", "long", "bool"}
DTYPE_RX = re.compile(r"\[lambda\(([\w:]+)\) \(instance \d+\)\]")


def op_hint(full):
    """What a generic PyTorch kernel computes, from its template arguments ("MulFunctor", "MeanOps", ...)."""
    ops = [m for m in OP_RX.findall(full) if not m.startswith(("gpu_kernel", "launch_"))]
    if not ops:
        return ""
    op = ops[-1]
    m = re.search(re.escape(op) + r"<([\w:]+)", full)
    dt = [m.group(1)] if m else [d for d in DTYPE_RX.findall(full) if d in DTYPES]
    return op + (f"({dt[-1].replace('c10::', '')})" if dt else "")


def _cols(con, table):
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def parse(sqlite_path, skip=3):
    """{shape: {"span_us", "kernel_sum_us", "idle_us", "kernels": [{"name", "per_replay", "med_us", "share"}]}}"""
    con = sqlite3.connect(sqlite_path)
    ncols = _cols(con, "NVTX_EVENTS")
    if "textId" in ncols:
        q = ("SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
             "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL")
    else:
        q = "SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ranges = [(a, b, t) for a, b, t in con.execute(q) if t and t.startswith(RANGE_PREFIX)]
    kcols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    full_col = "demangledName" if "demangledName" in kcols else "shortName"
    names = dict(con.execute("SELECT id, value FROM StringIds").fetchall())
    node_col = "graphNodeId" if "graphNodeId" in kcols else "NULL"
    kernels = [(a, b, names.get(sn, "?"), names.get(fn, "?"), node) for a, b, sn, fn, node in con.execute(
        f"SELECT start, end, shortName, {full_col}, {node_col} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    by_shape = {}
    ki = 0
    for start, end, text in sorted(ranges):
        _, shape, rep = text.split(":")
        if int(rep) < skip:
            continue
        while ki < len(kernels) and kernels[ki][0] < start:
            ki += 1
        inside = []
        j = ki
        while j < len(kernels) and kernels[j][0] < end:
            if kernels[j][1] <= end:
                inside.append(kernels[j])
            j += 1
        by_shape.setdefault(shape, []).append(inside)
    out = {}
    for shape, reps in by_shape.items():
        # kernels that are not graph nodes ran outside the candidate's graph: PyTorch's replay() first fills two
        # int64 RNG-state tensors on every replay. They are inside every timed call, but they are not the candidate.
        extra = [[k for k in r if k[4] is None] for r in reps]
        reps = [[k for k in r if k[4] is not None] for r in reps]
        keep = [i for i, r in enumerate(reps) if r]
        reps, extra = [reps[i] for i in keep], [extra[i] for i in keep]
        if not reps:
            continue
        spans = [(max(k[1] for k in r) - r[0][0]) / 1e3 for r in reps]
        sums = [sum(k[1] - k[0] for k in r) / 1e3 for r in reps]
        idles = [sp - su for sp, su in zip(spans, sums)]
        per = {}
        for r in reps:
            for ks, ke, short, full, _ in r:
                per.setdefault((short, full), []).append((ke - ks) / 1e3)
        overhead = {}
        for r in extra:
            for ks, ke, short, full, _ in r:
                overhead.setdefault(short, []).append((ke - ks) / 1e3)
        span = statistics.median(spans)
        rows = []
        for (key, full), durs in per.items():
            med = statistics.median(durs)
            n = len(durs) / len(reps)
            rows.append({"name": key, "op": op_hint(full), "full_name": full[:300], "per_replay": round(n, 2),
                         "med_us": med, "share": n * med / span if span else 0.0})
        rows.sort(key=lambda x: -x["per_replay"] * x["med_us"])
        out[shape] = {"replays": len(reps), "span_us": span, "kernel_sum_us": statistics.median(sums),
                      "idle_us": statistics.median(idles), "kernels": rows,
                      "outside_graph": {k: {"per_replay": round(len(v) / len(reps), 2),
                                            "med_us": statistics.median(v)} for k, v in overhead.items()}}
    return out


def markdown(parsed, times=None, order=None, detail=3, top=8):
    """Kernel tables for the first `detail` shapes of `order` (the gap-table order), one line for the others."""
    order = [o for o in (order or []) if o in parsed] + [o for o in parsed if o not in (order or [])]
    lines = []
    for i, shape in enumerate(order):
        p = parsed[shape]
        t = f"; timed call {times[shape]:.2f} us" if times and shape in times else ""
        idle = (f"idle between its kernels {p['idle_us']:.2f} us" if p["idle_us"] >= 0 else
                f"its kernels overlap ({-p['idle_us']:.2f} us of concurrent kernel time)")
        lines.append(f"**{shape}**: the candidate's graph spans {p['span_us']:.2f} us on the GPU: "
                     f"{sum(k['per_replay'] for k in p['kernels']):g} kernels, kernel time {p['kernel_sum_us']:.2f} us, "
                     f"{idle}{t}.")
        if i >= detail:
            top_k = p["kernels"][0]
            lines += [f"Largest: `{top_k['name'][:50]}` {top_k['op']} {top_k['med_us']:.2f} us "
                      f"({100 * top_k['share']:.0f}%).", ""]
            continue
        if p.get("outside_graph"):
            ov = ", ".join(f"`{k[:40]}` x{v['per_replay']:g} ({v['med_us']:.2f} us)" for k, v in p["outside_graph"].items())
            lines.append(f"Also in every timed call, outside the graph (PyTorch replay overhead, same for every "
                         f"variant): {ov}.")
        lines += ["", "| kernel | op | per call | median us | share of span |", "|---|---|---|---|---|"]
        for k in p["kernels"][:top]:
            lines.append(f"| `{k['name'][:50]}` | {k['op'] or '-'} | {k['per_replay']:g} | {k['med_us']:.2f} | "
                         f"{100 * k['share']:.0f}% |")
        if len(p["kernels"]) > top:
            lines.append(f"| ({len(p['kernels']) - top} more) | | | | |")
        lines.append("")
    return "\n".join(lines)
