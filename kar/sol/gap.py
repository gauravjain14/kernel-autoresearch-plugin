"""Speed-of-light floors and the gap table.

A shape's floor is the slowest of its compulsory resources, at the rates this GPU was measured to deliver:
  memory  (min bytes read + bytes written beyond what L2 absorbs) / measured DRAM bandwidth.
          Writes that fit in L2 finish as dirty lines and reach DRAM after the timed call ends (measured: an RMSNorm
          that reads and writes 8 MB took the time of reading 8 MB alone), so they do not count against the call.
  math    flops on each pipe / measured peak of that pipe
plus the fixed cost of one CUDA-graph replay, which every timed call pays. The gap table ranks shapes by how much the
score would gain if that shape reached its floor, i.e. by (time above floor) x (score value of a microsecond there).
"""
import math

PIPES = {"fp32": "fp32_TFLOPs", "tc_bf16": "tc_bf16_TFLOPs", "tc_fp16": "tc_fp16_TFLOPs",
         "tc_tf32": "tc_tf32_TFLOPs", "tc_fp8": "tc_fp8_TFLOPs"}


class OpModelError(Exception):
    pass


def floors(task, hw, shapes):
    """{shape: {bytes, flops, t_mem_us, t_math_us, floor_us, launch_us, bound}} from the task's op model."""
    if not task.opmodel.get("bytes"):
        raise OpModelError("task.toml has no [opmodel] bytes = 'file.py:fn'")
    bytes_fn = task.attr(task.opmodel["bytes"])
    flops_fn = task.attr(task.opmodel["flops"]) if task.opmodel.get("flops") else (lambda **_: {})
    m = hw["measured"]
    absorb = float(m.get("l2_write_absorb_bytes", 0))
    launch = m["graph_empty_us"] if (getattr(task, "bench", None) or {}).get("mode", "graph") == "graph" else \
        m.get("eager_empty_us", m["graph_empty_us"])
    out = {}
    for s in shapes:
        b = bytes_fn(**s["args"])
        rd, wr = (float(b.get("read", 0)), float(b.get("write", 0))) if isinstance(b, dict) else (float(b), 0.0)
        nbytes = rd + wr
        counted = rd + max(0.0, wr - absorb)
        fl = dict(flops_fn(**s["args"]))
        t_mem = counted / (m["dram_GBps"] * 1e3)                     # GB/s * 1e3 = bytes per us
        t_math = {}
        for pipe, n in fl.items():
            key = PIPES.get(pipe)
            if key is None or key not in m:
                raise OpModelError(f"op model uses pipe {pipe!r}; known pipes: {', '.join(PIPES)}")
            t_math[pipe] = n / (m[key] * 1e6)                           # TFLOP/s * 1e6 = flop per us
        bound, t_bound = max([("memory", t_mem)] + list(t_math.items()), key=lambda kv: kv[1])
        out[s["name"]] = {"bytes": nbytes, "read_bytes": rd, "write_bytes": wr, "counted_bytes": counted,
                          "flops": fl, "t_mem_us": t_mem, "t_math_us": t_math,
                          "floor_us": t_bound + launch, "work_floor_us": t_bound,
                          "launch_us": launch, "bound": bound}
    return out


def _value(score_fn, times, name, t_new):
    """score gain (as a fraction) when shape `name` takes t_new instead of its time."""
    return score_fn({**times, name: t_new}) / score_fn(times) - 1


def table(fl, times, weights, baseline=None, score_fn=None):
    """Rows sorted by score gain at the floor. times/baseline: {shape: us}. With the default score (weighted geomean
    speedup) one microsecond on shape s is worth w_s / (W * t_s) of the score and reaching the floor is worth
    (w_s / W) * ln(t_s / floor_s) in log-score; with a task-defined score both come from the task's own formula."""
    W = sum(weights.values())
    rows = []
    for name, f in fl.items():
        t = times[name]
        w = weights[name] / W
        if score_fn is not None:
            d = min(1.0, 0.01 * t)
            per_us = 100.0 * _value(score_fn, times, name, t - d) / d
            at_floor = 100.0 * _value(score_fn, times, name, min(t, f["floor_us"]))
        else:
            per_us = 100.0 * w / t
            at_floor = 100.0 * math.expm1(w * math.log(t / f["floor_us"])) if t > f["floor_us"] else 0.0
        work_us = max(t - f["launch_us"], 1e-9)
        flops = sum(f["flops"].values())
        rows.append({
            "shape": name, "weight": weights[name], "baseline_us": baseline.get(name) if baseline else None,
            "time_us": t, "floor_us": f["floor_us"], "bound": f["bound"], "t_mem_us": f["t_mem_us"],
            "t_math_us": f["t_math_us"], "launch_us": f["launch_us"],
            "pct_of_floor": 100.0 * f["floor_us"] / t, "gap_us": t - f["floor_us"],
            "score_pct_per_us": per_us, "score_pct_at_floor": at_floor,
            "achieved_GBps": f["counted_bytes"] / work_us / 1e3,
            "achieved_TFLOPs": flops / work_us / 1e6,
            "intensity": flops / f["counted_bytes"] if f["counted_bytes"] else float("inf"),
            "flops_by_pipe": dict(f["flops"]), "work_floor_us": f["work_floor_us"],
        })
    rows.sort(key=lambda r: -r["score_pct_at_floor"])
    if score_fn is not None:
        floors_t = {r["shape"]: min(r["time_us"], r["floor_us"]) for r in rows}
        total = 100.0 * (score_fn(floors_t) / score_fn(times) - 1)
    else:
        total = 100.0 * math.expm1(sum(math.log1p(r["score_pct_at_floor"] / 100.0) for r in rows))
    return rows, total


def sol(rows, score_fn=None):
    """Fraction of speed of light: 1.0 means every shape at its floor. A shape measured below its floor counts as
    at it (floors are approximate where L2 absorbs writes). Default: weighted geomean of floor/time; with a task
    score, score(times) / score(floors)."""
    if score_fn is None:
        W = sum(r["weight"] for r in rows)
        return math.exp(sum(r["weight"] * min(0.0, math.log(r["floor_us"] / r["time_us"])) for r in rows) / W)
    t = {r["shape"]: max(r["time_us"], r["floor_us"]) for r in rows}
    f = {r["shape"]: r["floor_us"] for r in rows}
    return score_fn(t) / score_fn(f)


def roofline_markdown(rows, hw):
    """The roofline view: each shape's arithmetic intensity against this GPU's ridge points, how much of the memory
    and compute roofs it reaches, and how much of its floor is the fixed cost of a call."""
    m = hw["measured"]
    bw = m["dram_GBps"]
    ridges = [(p, m[k]) for p, k in PIPES.items() if k in m]
    rid = ", ".join(f"{p} {v * 1e3 / bw:.0f} flop/B" for p, v in ridges)
    lines = ["### Roofline", "",
             f"Ridge points (peak flop/s over measured DRAM bandwidth {bw:.0f} GB/s): {rid}. A shape whose intensity is "
             f"below the ridge of the pipe it uses is memory-bound there; above it, compute-bound. A shape whose floor "
             f"is mostly the fixed cost of a call is latency-bound: fewer launches and less serial work pay there, "
             f"not bandwidth.", "",
             "| shape | intensity flop/B | achieved GB/s | % of DRAM roof | achieved TFLOP/s | % of compute roof | "
             "fixed cost share of floor | limited by |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pipes = {p: n for p, n in r["flops_by_pipe"].items() if n}
        peak = sum(m[PIPES[p]] * n for p, n in pipes.items()) / max(1, sum(pipes.values())) if pipes else 0
        pct_c = 100 * r["achieved_TFLOPs"] / peak if peak else 0
        pct_m = 100 * r["achieved_GBps"] / bw
        fixed = 100 * (r["floor_us"] - r["work_floor_us"]) / r["floor_us"]
        lim = "latency (fixed cost)" if fixed > 50 else r["bound"]
        inten = f"{r['intensity']:.2f}" if r["intensity"] != float("inf") else "-"
        lines.append(f"| {r['shape']} | {inten} | {r['achieved_GBps']:.0f} | {pct_m:.0f}% | {r['achieved_TFLOPs']:.2f} | "
                     f"{pct_c:.0f}% | {fixed:.0f}% | {lim} |")
    return "\n".join(lines) + "\n"


def markdown(rows, total, hw, title="", score_fn=None):
    m, d = hw["measured"], hw["device"]
    lines = []
    if title:
        lines += [f"## {title}", ""]
    lines += [f"GPU: {d['name']} ({d['sms']} SMs, L2 {d['l2_bytes'] >> 20} MB). Measured: DRAM {m['dram_GBps']:.0f} GB/s "
              f"(nominal {d['nominal_dram_GBps']:.0f}), L2 read {m['l2_read_GBps']:.0f} GB/s, FP32 {m['fp32_TFLOPs']:.1f} "
              f"TFLOP/s, BF16 tensor {m.get('tc_bf16_TFLOPs', float('nan')):.0f} TFLOP/s, empty graph replay "
              f"{m['graph_empty_us']:.2f} us. L2 absorbs about {m.get('l2_write_absorb_bytes', 0) / 1e6:.0f} MB of writes "
              f"within a timed call; only writes beyond that count toward the memory floor.", "",
              "| shape | w | baseline us | ours us | floor us | bound | % of floor | gap us | score %/us | score % at floor "
              "| achieved GB/s (counted bytes) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = f"{r['baseline_us']:.2f}" if r["baseline_us"] is not None else "-"
        note = (" (faster than the floor: the floor is approximate here, likely L2 holding more of the writes "
                "than calibrated; counted as 100%)") if r["pct_of_floor"] > 101 else ""
        lines.append(f"| {r['shape']} | {r['weight']:g} | {base} | {r['time_us']:.2f} | {r['floor_us']:.2f} | "
                     f"{r['bound']} | {r['pct_of_floor']:.0f}%{note} | {r['gap_us']:.2f} | {r['score_pct_per_us']:.3f} | "
                     f"{r['score_pct_at_floor']:.2f} | {r['achieved_GBps']:.0f} |")
    lines += ["", f"Score gain if every shape reached its floor: {total:+.1f}%"
              + (" (from the task's own score formula)" if score_fn is not None else "")
              + f". The floor includes the fixed cost of one call ({rows[0]['launch_us']:.2f} us here), which no kernel "
              f"change can remove.", "", roofline_markdown(rows, hw)]
    return "\n".join(lines)
