"""`./kar` in a run's workspace. Light on purpose: no torch, no GPU, only files in the workspace.

If a harness process is serving this run (ws/queue/server.json is fresh), the command goes to it as a request file
and the answer comes back as a file. That is how agents in a sandbox without GPU access or writes outside their
folder (the usual Codex, Cursor or Grok setting) still work. Otherwise the command runs right here.
"""
import json
import os
import time
import uuid

FRESH_S = 15


def _serving(q):
    try:
        with open(os.path.join(q, "server.json")) as f:
            hb = json.load(f)
        return time.time() - hb["t"] < FRESH_S
    except (OSError, ValueError, KeyError):
        return False


def _await(q, rid, wait_s, tool):
    res = os.path.join(q, f"{rid}.res.json")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if os.path.exists(res):
            with open(res) as f:
                r = json.load(f)
            print(r["out"])
            return r["code"]
        if not _serving(q):
            print(f"the harness process stopped before answering request {rid}; run {tool} wait after it restarts")
            return 3
        time.sleep(0.5)
    print(f"still running (request {rid}; GPU jobs from the whole team queue for the GPU). Run {tool} wait to get "
          f"the result.")
    return 2


def main(ws, argv, lane=None):
    q = os.path.join(ws, "queue")
    tool = f"./lanes/{lane}/kar" if lane else "./kar"
    wait_s = float(os.environ.get("KAR_WAIT_S", "540"))
    last = os.path.join(q, f"last_{lane or 'ws'}")
    if argv[:1] == ["wait"]:
        try:
            with open(last) as f:
                rid = f.read().strip()
        except OSError:
            print("nothing to wait for")
            return 1
        return _await(q, rid, wait_s, tool)
    if _serving(q):
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        tmp = os.path.join(q, f".{rid}.tmp")
        with open(tmp, "w") as f:
            json.dump({"id": rid, "argv": argv, "cwd": os.getcwd(), "lane": lane, "t": time.time()}, f)
        os.replace(tmp, os.path.join(q, f"{rid}.req.json"))
        with open(last, "w") as f:
            f.write(rid)
        return _await(q, rid, wait_s, tool)
    from kar.run.commands import execute
    out, code = execute(os.path.dirname(os.path.abspath(ws)), argv, cwd=os.getcwd(), lane=lane)
    print(out)
    return code
