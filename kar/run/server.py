"""The harness side of the request queue: executes the agents' `./kar` requests outside their sandboxes, many at
once (each in its own thread; GPU jobs still queue for the GPU one at a time), and keeps a heartbeat so clients know
it is there."""
import glob
import json
import os
import threading
import time

from kar.run.commands import execute
from kar.run.store import write_json

MAX_THREADS = 64


def serve(root, stop=None, log=None):
    q = os.path.join(root, "ws", "queue")
    os.makedirs(q, exist_ok=True)
    stop = stop or threading.Event()
    hb = os.path.join(q, "server.json")
    ws = os.path.realpath(os.path.join(root, "ws"))
    live = []

    def beat():
        while not stop.is_set():
            write_json(hb, {"pid": os.getpid(), "t": time.time()})
            stop.wait(2.0)

    def handle(rp, req):
        cwd = req.get("cwd") or ws
        if not os.path.realpath(cwd).startswith(ws):
            cwd = ws
        lane = req.get("lane")
        lane = lane if isinstance(lane, str) and lane.isalnum() else None
        t0 = time.time()
        try:
            out, code = execute(root, req["argv"], cwd=cwd, lane=lane)
        except BaseException as e:  # noqa: BLE001 - the harness must answer every request, even a SystemExit
            out, code = f"harness error: {type(e).__name__}: {e}", 4
        write_json(os.path.join(q, f"{req['id']}.res.json"),
                   {"id": req["id"], "out": out, "code": code, "s": round(time.time() - t0, 1)})
        os.replace(rp, rp[:-len(".run.json")] + ".done.json")

    for rp in glob.glob(os.path.join(q, "*.run.json")):  # taken by a server that died: their clients are gone too
        os.replace(rp, rp[:-len(".run.json")] + ".stale.json")
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        while not stop.is_set():
            live[:] = [x for x in live if x.is_alive()]
            for rp in sorted(glob.glob(os.path.join(q, "*.req.json")), key=os.path.getmtime):
                if len(live) >= MAX_THREADS:
                    break
                try:
                    with open(rp) as f:
                        req = json.load(f)
                except (OSError, ValueError):
                    continue
                taken = rp[:-len(".req.json")] + ".run.json"
                try:
                    os.replace(rp, taken)
                except FileNotFoundError:            # another server took it first
                    continue
                if log:
                    log(f"[serve] {req.get('lane') or 'ws'} {req['id']}: ./kar {' '.join(req['argv'])}")
                th = threading.Thread(target=handle, args=(taken, req), daemon=True)
                th.start()
                live.append(th)
            stop.wait(0.3)
        for th in live:
            th.join(timeout=5)
    finally:
        stop.set()
        try:
            os.remove(hb)
        except OSError:
            pass
