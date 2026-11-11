"""Task spec: a folder holding task.toml and the files it names.

A task names its reference ("file.py:function"), an optional exact fp64 reference, the baseline module, the input
generator, the scored and nearby shapes, tolerances, rules and bench settings. Task files are imported as top-level
modules from the task folder, so they may import each other.
"""
import hashlib
import importlib
import os
import sys
import tomllib
from dataclasses import dataclass, field

DEFAULT_TOLERANCE = {"atol": 1e-2, "rtol": 1e-2, "mean_err_ratio": 1.5, "max_err_ratio": 4.0}
DEFAULT_RULES = {
    "max_source_bytes": 1_000_000,
    "forbidden_imports": ["base64", "binascii", "zlib", "lzma", "bz2", "gzip", "zipfile", "tarfile", "marshal",
                          "pickle", "socket", "urllib", "http", "requests", "subprocess", "importlib",
                          "multiprocessing", "os", "io", "shutil", "pathlib", "glob", "tempfile", "mmap"],
    "forbidden_patterns": [],
    "third_party": "ask",
}
DEFAULT_BENCH = {"reps": 48, "copies": 2, "instances": 4, "warmup": 3, "flush_bytes": 0, "spin_s": 0.5,
                 "mode": "graph"}       # "graph": time CUDA-graph replays; "eager": time direct calls (for work that
#                                         cannot be captured: host syncs, dynamic shapes, whole model steps)
DEFAULT_LIMITS = {"start": 120, "import": 120, "smoke": 20, "shape": 180, "bench": 600, "default": 300}

REQUIRED = ("name", "entry", "reference", "baseline", "inputs", "shapes")
TABLES = ("tolerance", "rules", "bench", "limits", "opmodel", "judge")


class TaskError(Exception):
    pass


_ROOTS = set()      # every task folder this process has resolved files from
_ACTIVE = {}        # the folder plain-name task imports currently resolve to


@dataclass
class Task:
    root: str
    name: str
    entry: str
    reference: str
    baseline: str
    inputs: str
    shapes: str
    exact: str | None = None
    nearby: str | None = None
    score: str | None = None
    target: float | None = None
    arch: str = "any"
    smoke_shape: str | None = None
    good: list = field(default_factory=list)
    canaries: list = field(default_factory=list)
    tolerance: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    opmodel: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)

    def path(self, rel):
        return os.path.join(self.root, rel)

    def files(self):
        """Every file that defines the task (hashed to freeze it)."""
        out = []
        for dp, dn, fn in os.walk(self.root):
            dn[:] = sorted(d for d in dn if d != "__pycache__" and not d.startswith("."))
            for f in sorted(fn):
                if f.endswith((".py", ".toml", ".json", ".md", ".cu", ".cuh")):
                    out.append(os.path.relpath(os.path.join(dp, f), self.root))
        return out

    def sha(self):
        h = hashlib.sha256()
        for rel in self.files():
            h.update(rel.encode() + b"\0")
            with open(self.path(rel), "rb") as f:
                h.update(f.read())
            h.update(b"\0")
        return h.hexdigest()

    def activate(self):
        """Make this task's folder the one its plain-name imports resolve to. Task files import each other as
        top-level modules (`from reference import ...`), so a module cached from another task with the same name
        (every task has a shapes.py) is dropped, and other task folders leave sys.path."""
        if _ACTIVE.get("root") == self.root:
            return
        others = [r for r in _ROOTS if r != self.root]
        for key, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None) or ""
            if any(f.startswith(r + os.sep) for r in others):
                del sys.modules[key]
        sys.path[:] = [x for x in sys.path if x not in others]
        if self.root not in sys.path:
            sys.path.append(self.root)
        _ROOTS.add(self.root)
        _ACTIVE["root"] = self.root

    def attr(self, ref):
        """Resolve "file.py:name" by importing file.py as a top-level module from the task folder."""
        file, _, name = ref.partition(":")
        self.activate()
        mod = importlib.import_module(file[:-3].replace("/", "."))
        try:
            return getattr(mod, name)
        except AttributeError:
            raise TaskError(f"{file} has no {name!r}") from None

    def shape_list(self, which="scored"):
        ref = self.shapes if which == "scored" else self.nearby
        if not ref:
            return []
        out, names = [], set()
        for i, s in enumerate(self.attr(ref)):
            if not isinstance(s, dict) or not isinstance(s.get("args"), dict):
                raise TaskError(f"{ref}[{i}]: each shape is a dict with an 'args' dict")
            name = str(s.get("name", f"{which}{i}"))
            if name in names:
                raise TaskError(f"{ref}: duplicate shape name {name!r}")
            names.add(name)
            out.append({"name": name, "weight": float(s.get("weight", 1.0)), "args": dict(s["args"]), "kind": which})
        if which == "scored" and not out:
            raise TaskError(f"{ref}: no scored shapes")
        return out


def load(root):
    root = os.path.abspath(root)
    p = os.path.join(root, "task.toml")
    if not os.path.isfile(p):
        raise TaskError(f"{p} not found")
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise TaskError(f"task.toml lacks {', '.join(missing)}")
    known = set(REQUIRED) | set(TABLES) | {"exact", "nearby", "score", "target", "arch", "smoke_shape", "good",
                                           "canaries"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise TaskError(f"task.toml: unknown keys {', '.join(unknown)}")
    t = Task(root=root, **{k: raw[k] for k in raw if k not in TABLES})
    t.tolerance = {**DEFAULT_TOLERANCE, **raw.get("tolerance", {})}
    t.rules = {**DEFAULT_RULES, **raw.get("rules", {})}
    t.bench = {**DEFAULT_BENCH, **raw.get("bench", {})}
    t.limits = {**DEFAULT_LIMITS, **raw.get("limits", {})}
    t.opmodel = dict(raw.get("opmodel", {}))
    t.judge = dict(raw.get("judge", {}))
    if t.bench["mode"] not in ("graph", "eager"):
        raise TaskError(f"[bench] mode = {t.bench['mode']!r}: 'graph' or 'eager'")
    validate(t)
    return t


def validate(t):
    problems = []
    for key in ("reference", "exact", "inputs", "shapes", "nearby", "score"):
        v = getattr(t, key)
        if v is None:
            continue
        file, sep, name = v.partition(":")
        if not sep or not name or not file.endswith(".py"):
            problems.append(f"{key} = {v!r}: expected 'file.py:name'")
        elif not os.path.isfile(t.path(file)):
            problems.append(f"{key}: {file} not found")
    for v in t.opmodel.values():
        if not os.path.isfile(t.path(v.partition(":")[0])):
            problems.append(f"opmodel: {v} not found")
    for rel in [t.baseline, *t.good, *t.canaries]:
        if not os.path.isfile(t.path(rel)):
            problems.append(f"{rel} not found")
    for f in os.listdir(t.root):
        if f.endswith(".py") and f[:-3] in sys.stdlib_module_names:
            problems.append(f"{f} shadows the standard-library module {f[:-3]!r}; rename it")
    if problems:
        raise TaskError("\n".join(problems))
