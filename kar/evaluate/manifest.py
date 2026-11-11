"""Hash of the evaluator's own files. A run records it at init; every job checks it, so no candidate is ever
measured by an evaluator that changed mid-run."""
import hashlib
import os

KAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluator_files():
    out = []
    for sub in ("evaluate", "task"):
        for dp, dn, fn in os.walk(os.path.join(KAR, sub)):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith((".py", ".in")):
                    out.append(os.path.join(dp, f))
    return out


def evaluator_sha():
    h = hashlib.sha256()
    for p in evaluator_files():
        h.update(os.path.relpath(p, KAR).encode() + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()
