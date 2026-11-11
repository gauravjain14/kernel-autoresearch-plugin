"""Run storage: atomic JSON state, a run-wide lock, and an append-only ledger.

Every ledger record carries the hash of the record before it, so an edit or deletion anywhere in a ledger file breaks
the chain, and `kar report` says so. The ledger is the run's memory: agents are short-lived and read it back.
"""
import datetime
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager

GENESIS = "0" * 64


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


@contextmanager
def locked(path):
    """Exclusive lock (flock) while one command changes shared state. Not re-entrant: never take the same lock twice in
    one thread."""
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _digest(prev, rec):
    return hashlib.sha256((prev + json.dumps(rec, sort_keys=True, default=str)).encode()).hexdigest()


class Ledger:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.d, name + ".jsonl")

    def append(self, name, rec):
        """Many lanes (threads and processes) append at once: the ledger's own lock keeps each chain whole."""
        with locked(os.path.join(self.d, ".ledger.lock")):
            return self._append(name, rec)

    def _append(self, name, rec):
        rec = {"t": now(), **rec}
        rows = self.rows(name)
        prev = rows[-1]["sha"] if rows else GENESIS
        line = {"rec": rec, "prev": prev, "sha": _digest(prev, rec)}
        with open(self._path(name), "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def rows(self, name):
        try:
            with open(self._path(name)) as f:
                return [json.loads(x) for x in f if x.strip()]
        except FileNotFoundError:
            return []

    def records(self, name):
        return [r["rec"] for r in self.rows(name)]

    def verify(self):
        """{file: 'ok' | 'broken at record N'} for every ledger file."""
        out = {}
        for fn in sorted(os.listdir(self.d)):
            if not fn.endswith(".jsonl"):
                continue
            prev, status = GENESIS, "ok"
            for i, row in enumerate(self.rows(fn[:-6])):
                if row.get("prev") != prev or row.get("sha") != _digest(prev, row.get("rec")):
                    status = f"broken at record {i + 1}"
                    break
                prev = row["sha"]
            out[fn] = status
        return out

    def note(self, name, text):
        """Append a dated entry to a markdown log (facts.md, dead_ends.md). Also chained, through notes.jsonl."""
        with locked(os.path.join(self.d, ".ledger.lock")):
            self._append("notes", {"file": name, "text": text})
            with open(os.path.join(self.d, name), "a") as f:
                f.write(f"- {now()[:16]}  {text}\n")
