"""From a plain-language request to a task folder.

`kar new DIR "request"` writes the request, the task-writing instructions (AGENTS.md) and a skeleton; any agent (or
a person) fills in the files; `./kar-task validate` checks them; `kar init` proves the evaluator on them. A person
reviews reference.py before a run starts, because it defines correct.
"""
import os
import shutil
import sys

from kar.task import spec

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
PROMPTS = os.path.join(HERE, "agents", "prompts")

SKELETON = {
    "task.toml": '''# TODO: one line saying what the operation is and its precision rule.
name = "TODO_name"
arch = "any"                  # or e.g. "sm_89" to refuse scoring on another GPU
entry = "kernel_fn"
reference = "reference.py:reference"
exact = "reference.py:exact"
baseline = "baseline.py"
inputs = "inputs.py:make"
shapes = "shapes.py:SCORED"
nearby = "shapes.py:NEARBY"

[tolerance]
atol = 0.02
rtol = 0.02
mean_err_ratio = 1.5
max_err_ratio = 4.0

[rules]
precision = "TODO: e.g. BF16 inputs and output, FP32 accumulation; no FP16"

[opmodel]
bytes = "opmodel.py:min_bytes"
flops = "opmodel.py:flops"

# Optional, see AGENTS.md:
# score = "score.py:score"          # put this line above the first [table] when the score is a custom formula
# [bench]
# mode = "eager"                    # when the work cannot be captured in a CUDA graph
# [judge]
# check = "mychallenge check {file}"
# submit = "mychallenge submit {file}"
# parse = "judge.py:parse"
# quota = { check = 20, submit = 10, window_h = 24 }
# policy = { check = "auto", submit = "ask" }
''',
    "reference.py": '''"""TODO: what the operation computes, and the precision rule."""
import torch


def reference(*args):
    raise NotImplementedError("TODO: the operation in plain PyTorch at the prescribed precision")


def exact(*args):
    raise NotImplementedError("TODO: the same operation in float64")
''',
    "inputs.py": '''import torch


def make(seed, device="cuda", **shape):
    g = torch.Generator(device=device).manual_seed(seed)
    raise NotImplementedError("TODO: build the inputs for one call from `shape`, using generator g")
''',
    "shapes.py": '''SCORED = [
    # {"name": "TODO", "weight": 1, "args": {...}},
]
NEARBY = [
    # off-grid shapes that are checked but not timed
]
''',
    "opmodel.py": '''"""Compulsory DRAM traffic and math per call, for speed-of-light floors."""


def min_bytes(**shape):
    raise NotImplementedError("TODO: {'read': bytes, 'write': bytes}")


def flops(**shape):
    raise NotImplementedError("TODO: {'fp32': n} or tensor-core pipes")
''',
    "baseline.py": '''"""Baseline: the reference itself (replace with what speedups should be measured against)."""
from reference import reference as kernel_fn  # noqa: F401
''',
}




SOURCE_SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
SOURCE_MAX_FILE = 50 << 20


def _copy_source(src, dst):
    """A copy of the project or challenge the task is about, for the task writer to read (and the task to import).
    Large files (weights, datasets, reports) are left out and listed, so the copy stays a readable source tree."""
    skipped = []
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if d not in SOURCE_SKIP)
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in sorted(fn):
            full = os.path.join(dp, f)
            if os.path.islink(full) or os.path.getsize(full) > SOURCE_MAX_FILE:
                skipped.append(os.path.relpath(full, src))
                continue
            shutil.copy2(full, os.path.join(dst, rel, f))
    if skipped:
        with open(os.path.join(dst, "NOT_COPIED.txt"), "w") as f:
            f.write("Files left out of this copy (over 50 MB, or links):\n" + "\n".join(skipped) + "\n")


def create(root, description, source=None):
    root = os.path.abspath(root)
    if os.path.exists(root) and os.listdir(root):
        raise SystemExit(f"{root} exists and is not empty")
    os.makedirs(root, exist_ok=True)
    if source:
        _copy_source(os.path.abspath(source), os.path.join(root, "source"))
        description += (f"\n\nThe task is about the project copied into `source/` (from {os.path.abspath(source)}). "
                        f"Read it first; mirror what it measures.")
    with open(os.path.join(root, "DESCRIPTION.md"), "w") as f:
        f.write("# Request\n\n" + description.strip() + "\n\n## Assumptions\n\n")
    with open(os.path.join(PROMPTS, "task_agents.md")) as f:
        guide = f.read()
    with open(os.path.join(root, "AGENTS.md"), "w") as f:      # where the worked example is on this machine
        f.write(guide.replace("$example", os.path.join(PROMPTS, "example_task")))
    with open(os.path.join(root, "CLAUDE.md"), "w") as f:
        f.write("@AGENTS.md\n")
    for name, text in SKELETON.items():
        with open(os.path.join(root, name), "w") as f:
            f.write(text)
    shim = os.path.join(root, "kar-task")
    py = sys.executable or shutil.which("python3") or "python3"
    with open(shim, "w") as f:
        f.write("#!/bin/sh\n# Checks this task folder: ./kar-task validate (the same check as `kar validate DIR`)\n"
                f"exec {py} -E -P -c 'import sys; sys.path.insert(0, sys.argv[1]); from kar.newtask import main; "
                f"sys.exit(main(sys.argv[2], sys.argv[3:]))' '{HARNESS}' \"$(cd \"$(dirname \"$0\")\" && pwd)\" \"$@\"\n")
    os.chmod(shim, 0o755)
    return root


def validate(root, timeout=900, backend=None):
    """-> (ok, text). Loads the task and runs reference, exact, baseline, the op model and the score on every shape,
    as a job on `backend` (default: the user's configured backend, else this machine's GPU)."""
    root = os.path.abspath(root)
    try:
        task = spec.load(root)
    except spec.TaskError as e:
        return False, f"task.toml or its files are not valid yet:\n  {e}"
    import tempfile
    from kar import config
    from kar.runner import Runner
    gpu = int(os.environ.get("KAR_GPU", "0"))
    r = Runner(task, tempfile.mkdtemp(prefix="kar-validate-"), gpu=gpu, backend=backend or config.backend())
    res = r.job("taskcheck", "task")
    out = res.get("task_check")
    if out is None:
        return False, f"the check crashed on {r.backend.describe()}: {res.get('reason')}\n{res.get('traceback') or res.get('log_tail', '')}"[-3500:]
    lines = []
    for s in out["shapes"]:
        extra = []
        if "baseline_ms" in s:
            extra.append(f"baseline {s['baseline_ms'] * 1e3:.1f} us")
        if "bytes" in s:
            extra.append(f"op model {s['bytes'].get('read', 0) / 1e6:.2f} MB read, "
                         f"{s['bytes'].get('write', 0) / 1e6:.2f} MB written, flops {s.get('flops')}")
        if "ref_vs_exact_max" in s:
            extra.append(f"reference vs exact max error {s['ref_vs_exact_max']:.3g} (output rms {s.get('out_rms', 0):.3g})")
        lines.append(f"  {s['kind']:6s} {s['shape']:24s} " + "; ".join(extra))
    ok = not out["problems"]
    text = [("TASK OK" if ok else "TASK NOT READY") + f" (checked on {r.backend.describe()})", ""] + lines
    if out.get("score_check"):
        text.append(f"  score: baseline {out['score_check']['baseline']:.4g}, every shape twice as fast "
                    f"{out['score_check']['twice_as_fast']:.4g}")
    if out["problems"]:
        text += ["", "problems:"] + [f"  - {p}" for p in out["problems"]]
    if out["warnings"]:
        text += ["", "warnings:"] + [f"  - {w}" for w in out["warnings"]]
    if out.get("traceback") and not ok:
        text += ["", out["traceback"]]
    if ok:
        text += ["", "Next: a person reviews reference.py and the assumptions in DESCRIPTION.md, then "
                     f"`kar init {root}` proves the evaluator on it and `kar run {root} --lanes 4` starts a team."]
    return ok, "\n".join(text)


def main(root, argv):
    if argv[:1] != ["validate"]:
        print("usage: ./kar-task validate [--backend local|modal:GPU[:N]|ssh:HOST[:GPUS]]")
        return 1
    backend = None
    if "--backend" in argv:
        from kar import backends
        i = argv.index("--backend")
        backend = backends.parse_spec(argv[i + 1] if i + 1 < len(argv) else "local")
    ok, text = validate(root, backend=backend)
    print(text)
    return 0 if ok else 1
