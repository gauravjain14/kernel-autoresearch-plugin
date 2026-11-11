"""GPU slots. A slot is a lock file; holding it means "one perf job on this GPU". Remote backends (M3) use N slot
files for an N-box cap. Locks live under ~/.cache/kar/locks and are released when the holding process exits."""
import fcntl
import os
import time
from contextlib import contextmanager

LOCKS = os.path.expanduser(os.environ.get("KAR_LOCK_DIR", "~/.cache/kar/locks"))


@contextmanager
def slot(name, wait_s=6 * 3600, poll_s=0.5):
    os.makedirs(LOCKS, exist_ok=True)
    f = open(os.path.join(LOCKS, f"{name}.lock"), "a+")
    deadline = time.time() + wait_s
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() > deadline:
                f.close()
                raise TimeoutError(f"slot {name} still busy after {wait_s} s")
            time.sleep(poll_s)
    try:
        f.seek(0)
        f.truncate()
        f.write(f"{os.getpid()} {time.time():.0f}\n")
        f.flush()
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


@contextmanager
def any_slot(prefix, n, labels=None, wait_s=6 * 3600, poll_s=0.5):
    """Hold the first free of n slots named prefix-<label>; yields the label (labels default to "0".."n-1")."""
    labels = list(labels or [str(i) for i in range(n)])
    os.makedirs(LOCKS, exist_ok=True)
    deadline = time.time() + wait_s
    while True:
        for lb in labels:
            f = open(os.path.join(LOCKS, f"{prefix}-{lb}.lock"), "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                f.seek(0)
                f.truncate()
                f.write(f"{os.getpid()} {time.time():.0f}\n")
                f.flush()
                yield lb
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        if time.time() > deadline:
            raise TimeoutError(f"no free slot among {prefix}-{{{','.join(labels)}}} after {wait_s} s")
        time.sleep(poll_s)
