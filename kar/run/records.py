"""What a run leaves behind for people: the task README the agent reads, milestone READMEs, and REPORT.md.

A milestone is written when the score has gained milestone_pct since the last one, or when a structural change is
kept: the diagnosis that motivated it, the prediction, the change as a diff, the measured result, and how to
reproduce it.
"""
import ast
import difflib
import os
import re
import shutil

from kar.run.store import now


def _doc(path):
    try:
        with open(path) as f:
            return ast.get_docstring(ast.parse(f.read())) or ""
    except (OSError, SyntaxError):
        return ""


def task_readme(t):
    with open(t.path("task.toml")) as f:
        head = []
        for line in f:
            if not line.startswith("#"):
                break
            head.append(line.lstrip("# ").rstrip())
    ref_file, _, ref_fn = t.reference.partition(":")
    inp_file, _, inp_fn = t.inputs.partition(":")
    lines = [f"# Task: {t.name}", ""]
    if head:
        lines += [" ".join(head), ""]
    doc = _doc(t.path(ref_file))
    if doc:
        lines += [doc, ""]
    lines += ["## What to write", "",
              f"A file that defines `{t.entry}(...)` taking the same arguments as `{ref_fn}` in `{ref_file}` and "
              f"returning the same result. Inputs come from `{inp_fn}` in `{inp_file}`. Output tensors must be "
              "allocated by the call (or be views the call owns); inputs must not be modified.", "",
              "## Shapes", "", "Scored (the score is the weighted geometric-mean speedup over the baseline):", ""]
    for s in t.shape_list("scored"):
        lines.append(f"- `{s['name']}` weight {s['weight']:g}: {s['args']}")
    near = t.shape_list("nearby")
    if near:
        lines += ["", "Nearby (checked for correctness, not timed; your kernel must handle them):", ""]
        for s in near:
            lines.append(f"- `{s['name']}`: {s['args']}")
    tol = t.tolerance
    r = t.rules
    lines += ["", "## Correctness", "",
              f"- Every scored and nearby shape, two eager calls (the second on new data in the same buffers), then "
              f"CUDA-graph capture and two replays with the output poisoned in between.",
              f"- Error against an fp64 reference within atol {tol['atol']} + rtol {tol['rtol']}, and your error "
              f"may be at most {tol['mean_err_ratio']}x (mean) and {tol['max_err_ratio']}x (max) the error of the "
              f"task's reference itself. Lower precision than the reference fails here.",
              "", "## Rules", "",
              f"- Source only, at most {r['max_source_bytes']:,} bytes; no precompiled binaries, encoded payloads or "
              f"file reads.",
              f"- Imports not allowed: {', '.join(r['forbidden_imports'])}."]
    if r.get("precision"):
        lines.append(f"- Precision: {r['precision']}")
    if t.opmodel:
        lines += ["", "## Speed of light", "",
                  f"Floors come from the op model in `{t.opmodel.get('bytes', '').partition(':')[0]}` (compulsory "
                  f"bytes and flops per shape) and this GPU's measured rates; see the gap table in each profile."]
    return "\n".join(lines) + "\n"


def _slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n] or "change"


def _gap_section(page):
    """The gap-table section of a profile page."""
    m = re.search(r"(## Gap to speed of light.*?)(?=\n## |\Z)", page, re.S)
    return m.group(1).strip() if m else ""


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _rows_table(rows):
    out = ["| shape | w | time us | floor us | % of floor | bound |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["shape"]):
        out.append(f"| {r['shape']} | {r['weight']:g} | {r['time_us']:.2f} | {r['floor_us']:.2f} | "
                   f"{r['pct_of_floor']:.0f}% | {r['bound']} |")
    return "\n".join(out)


def task_ref(run):
    rel = os.path.relpath(run.task.root, run.cfg["harness"])
    return rel if not rel.startswith("..") else run.task.root


def milestone(run, n, h, base_id, cid, prof, gain):
    d = run.p("milestones", f"{n:02d}_{_slug(h['title'])}")
    os.makedirs(os.path.join(d, "evidence"), exist_ok=True)
    shutil.copy(run.p("candidates", cid, "kernel.py"), os.path.join(d, "kernel.py"))
    os.chmod(os.path.join(d, "kernel.py"), 0o644)
    diff = run.diff(base_id, cid)
    with open(os.path.join(d, "change.diff"), "w") as f:
        f.write(diff)
    shutil.copy(run.p("evidence", cid, "profile.md"), os.path.join(d, "evidence", "profile.md"))
    base_page = _read(run.p("evidence", base_id, "profile.md"))
    report = _read(run.p("candidates", cid, "report.md")).strip()
    plus = sum(1 for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++"))
    minus = sum(1 for x in diff.splitlines() if x.startswith("-") and not x.startswith("---"))
    cand = next((c for c in run.ledger.records("candidates") if c["id"] == cid), {})
    lines = [f"# {n:02d}. {h['title']}", "",
             f"Run `{os.path.basename(run.root)}`, {now()[:16]}. Candidate `{cid}` (hypothesis `{h['id']}`), "
             f"measured against `{base_id}`, the previous milestone." + (" A structural change." if h.get("structural")
                                                                          else ""), "",
             "## Result", "",
             f"- **{gain * 100:+.1f}%** in score since `{base_id}`; now **{prof['score']:.3f}x** the baseline "
             f"[{prof['score_ci'][0]:.3f}, {prof['score_ci'][1]:.3f}] and **{prof['sol'] * 100:.1f}%** of speed "
             f"of light.",
             f"- The keep decision (`{cid}` against its parent `{cand.get('parent')}`, one GPU job): "
             f"{cand.get('effect', 0) * 100:+.2f}% [{cand.get('ci', [0, 0])[0] * 100:+.2f}, "
             f"{cand.get('ci', [0, 0])[1] * 100:+.2f}], {cand.get('decision')}.", "",
             _rows_table(prof["rows"]), "",
             "## Diagnosis", "",
             f"- Evidence: {h['evidence']}", f"- Target: {h['target']}", f"- Mechanism: {h['mechanism']}", "",
             f"The gap table it started from (`{base_id}`):", "", _gap_section(base_page) or "(not available)", "",
             "## Prediction", "",
             f"- Predicted: {h['predicted_gain_pct']:g}% ({h['arithmetic']}). Measured over its parent: "
             f"{cand.get('effect', 0) * 100:+.2f}%.",
             f"- Bound test: {h['bound_test']}", "",
             "## Change", "",
             f"`change.diff`: {plus} lines added, {minus} removed against `{base_id}`. The full kernel is "
             f"`kernel.py`.", ""]
    if report:
        lines += ["The agent's report:", "", "> " + report.replace("\n", "\n> "), ""]
    t = task_ref(run)
    lines += ["## Reproduce", "", "```",
              f"python3 -m kar ab {t} --cand kernel.py          # against the task baseline",
              f"python3 -m kar profile {t} kernel.py           # the evidence page",
              "```", "", "`evidence/profile.md` is the profile of this kernel at the time it was kept.", ""]
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return os.path.relpath(d, run.root)


def report_md(run, fin):
    """REPORT.md from the ledger and the final re-measurement `fin`."""
    st, c = run.state, run.costs()
    cands = run.ledger.records("candidates")
    hyps = {}
    for h in run.ledger.records("hypotheses"):
        hyps.setdefault(h["id"], {}).update(h)
    agents = sorted({x.get("agent") for x in run.ledger.records("cost") if x["kind"] == "agent" and x.get("agent")})
    lines = [f"# Report: {run.task.name}", "",
             f"Run `{os.path.basename(run.root)}` on {run.cfg['gpu_name']} ({run.cfg['arch']}), started "
             f"{st['started'][:16]}, report written {now()[:16]}. Ended: {st['done_reason'] or 'still running'}.",
             f"Agents: {', '.join(agents) if agents else 'none recorded by the harness (driven from outside)'}.", "",
             "## Final result (re-measured for this report)", ""]
    if fin.get("error"):
        lines += [f"The re-measurement failed: {fin['error']}", ""]
    else:
        lines += [f"- Best: `{fin['best']}`, **{fin['score']:.3f}x** the baseline [{fin['score_ci'][0]:.3f}, "
                  f"{fin['score_ci'][1]:.3f}] (weighted geometric mean over scored shapes, 95% CI), "
                  f"**{fin['sol'] * 100:.1f}%** of speed of light ({run.target_text()}).",
                  f"- Measured in a fresh job with seed {fin['seed']}, {fin['reps']} reps x {fin['copies']} copies, "
                  f"after re-running every correctness gate on the frozen file.", "",
                  _rows_table(fin["rows"]), ""]
    if fin.get("hazards"):
        lines += ["## Read this before using the kernel", ""] + [f"- {h}" for h in fin["hazards"]] + [""]
    lines += ["## Integrity", ""]
    for k, v in fin["integrity"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Trajectory of the best", "", "| best | % of SOL | x baseline | when |", "|---|---|---|---|"]
    for b in st.get("best_history", []):
        lines.append(f"| {b['id']} | {b['sol'] * 100:.1f}% | {b['score']:.3f} | {b['t'][:16]} |")
    lines += ["", "## The team", ""]
    lanes = list((st.get("lanes") or {}).values())
    roles = {}
    for x in lanes:
        roles[x["role"]] = roles.get(x["role"], 0) + 1
    lines.append(f"Team of {run.cfg.get('lanes', 1)} sessions at once. Lanes: "
                 + (", ".join(f"{k} {v}" for k, v in sorted(roles.items())) or "none") + ".")
    per = {}
    for x in run.ledger.records("cost"):
        if x["kind"] == "agent":
            a = per.setdefault(x.get("agent") or "?", {"sessions": 0, "usd": 0.0, "tokens": 0})
            a["sessions"] += 1
            a["usd"] += x.get("usd") or 0
            a["tokens"] += x.get("tokens") or 0
    for a, v in sorted(per.items()):
        lines.append(f"- {a}: {v['sessions']} sessions, ${v['usd']:.2f}, {v['tokens']:,} tokens")
    msgs = run.ledger.records("messages")
    lines.append(f"- messages between lanes: {sum(1 for m in msgs if m.get('from') != 'harness')} from agents, "
                 f"{sum(1 for m in msgs if m.get('from') == 'harness')} from the harness")
    aud = run.ledger.records("audits")
    lines += [f"- audit of {a['cid']} by {a['lane']}: **{a['verdict']}**: {a['text'][:300]}" for a in aud]
    lines += ["", "| lane | role | work | agent | outcome |", "|---|---|---|---|---|"]
    for x in lanes:
        work = x.get("hyp") or x.get("lens") or (", ".join(i["cid"] for i in x.get("items") or []) if x.get("items")
                                                  else x.get("base") if x["role"] == "audit" else f"round {x['cycle']}")
        lines.append(f"| {x['id']} | {x['role']} | {work} | {x.get('agent') or '-'} | "
                     f"{(x.get('outcome') or x['status'])[:140].replace('|', '/')} |")
    lines += ["", "## Every submit", "", "| cand | lane | hypothesis | decision | effect | against |",
              "|---|---|---|---|---|---|"]
    for x in cands:
        eff = (f"{x['effect'] * 100:+.2f}% [{x['ci'][0] * 100:+.2f}, {x['ci'][1] * 100:+.2f}]" if "effect" in x
               else x.get("stage", "") + ": " + x.get("reason", "")[:80].replace("|", "/"))
        lines.append(f"| {x['id']} | {x.get('lane', '-')} | {x.get('hyp') or '-'} {x.get('title', '')[:60]} | "
                     f"{x['decision']} | {eff} | {x.get('vs', x['parent'])} |")
    by = {}
    for h in hyps.values():
        by.setdefault(h.get("status", "open"), []).append(h)
    lines += ["", "## Hypotheses", "",
              ", ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())) or "none", ""]
    for h in sorted(hyps.values(), key=lambda h: h["id"]):
        lines.append(f"- `{h['id']}` [{h.get('status')}] {h.get('title')} (predicted {h.get('predicted_gain_pct')}%)"
                     + (f": {h['why']}" if h.get("why") else ""))
    from kar import judge as J
    if J.config(run):
        lines += ["", "## Judge", "", J.status_text(run)]
    ms = sorted(os.listdir(run.p("milestones")))
    lines += ["", "## Milestones", ""] + ([f"- `milestones/{m}/README.md`" for m in ms] or ["none"])
    lines += ["", "## Cost", "",
              f"- GPU: {c['gpu_s'] / 3600:.2f} GPU-hours over {sum(1 for x in run.ledger.records('cost') if x['kind'] == 'gpu')} jobs.",
              f"- Agents: {c['sessions']} sessions, {c['tokens']:,} tokens, ${c['agent_usd']:.2f} as reported by the "
              f"agent CLI (0 when the CLI does not report it).",
              f"- Wall clock: {c['wall_s'] / 3600:.1f} h.", "",
              "## Human interventions", ""]
    for i in run.ledger.records("interventions"):
        lines.append(f"- {i['t'][:16]} {i['by']}: {i['what']}")
    import kar
    lines += ["", "---", f"Made with kernel-autoresearch by {kar.__author__}: {kar.__url__}", ""]
    return "\n".join(lines)


def scan_sources(run):
    """Candidate sources that name a file the agent was not given (the task's known-good kernels and canaries)."""
    t = run.task
    hidden = [os.path.basename(x) for x in list(t.good) + list(t.canaries)] + [t.root]
    hits = []
    for cid in sorted(os.listdir(run.p("candidates"))):
        src = _read(run.p("candidates", cid, "kernel.py"))
        for h in hidden:
            if h and h in src:
                hits.append(f"{cid} mentions {h}")
    return hits


def unified(a_text, b_text, a_name, b_name):
    return "".join(difflib.unified_diff(a_text.splitlines(True), b_text.splitlines(True), a_name, b_name))
