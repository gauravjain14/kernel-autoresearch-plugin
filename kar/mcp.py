"""An MCP server (stdio, JSON-RPC 2.0) that gives an agent its lane's commands as tools.

For agents that cannot be limited to one shell command, or that work best through tools (IDE agents, Cursor, any MCP
client): the agent gets `kar_next`, `kar_try`, `kar_submit`, ... and nothing else needs a shell. Each tool call goes
through the same client as `./lanes/<id>/kar`, so it reaches the harness through the request queue when a harness
process is serving the run, and GPU work never runs inside the agent's sandbox.

    python3 -m kar.mcp --ws RUN/ws [--lane L03]

Without --lane, `kar_next` hands out a lane (as `./kar next` does) and the server remembers it. MCP clients time tool
calls out (Codex: 60 s by default), so a GPU command that is still running after `KAR_MCP_WAIT_S` (45 s) returns
"still running"; `kar_wait` collects the result.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

PROTOCOL = "2025-06-18"

FILE = {"type": "string", "description": "a file in your lane folder (lanes/<id>/...), or best, base, or a "
                                         "candidate id like c0003"}
TOOLS = {
    "kar_next": ("What to do now: your lane's step, with the numbers and commands. Start here.", {}, []),
    "kar_board": ("The team: every lane at work, the queue, recent decisions.", {}, []),
    "kar_wait": ("The result of a GPU command that was still running.", {}, []),
    "kar_check": ("Lint, smoke and correctness on every shape.", {"file": FILE}, ["file"]),
    "kar_try": ("Check, then a quick A/B against the current best; `files` times several variants in one job.",
                {"file": FILE, "files": {"type": "array", "items": {"type": "string"}}, "reps": {"type": "integer"}},
                []),
    "kar_bound": ("Time a probe that is incomplete on purpose against the best; output not checked.",
                  {"file": FILE, "reps": {"type": "integer"}}, ["file"]),
    "kar_profile": ("Full profile page: gap table, nsys, ncu summary, SASS audit.", {"file": FILE}, ["file"]),
    "kar_ncu": ("Every ncu section on one scored shape, as text.", {"file": FILE, "shape": {"type": "string"}},
                ["file"]),
    "kar_sass": ("The full SASS of every kernel the file loads.", {"file": FILE}, ["file"]),
    "kar_stress": ("Extra seeds, concurrent calls on several streams, back-to-back calls.", {"file": FILE}, ["file"]),
    "kar_test": ("Run your own test file (see help/test_example.py) on a kernel (default: your cand.py).",
                 {"testfile": FILE, "file": FILE}, ["testfile"]),
    "kar_register": ("Make your test part of the run: the reference must pass it; every later submit must too.",
                     {"testfile": FILE, "title": {"type": "string"}}, ["testfile", "title"]),
    "kar_copy": ("Copy a candidate (best, base, c0003) or a workspace file into your lane folder.",
                 {"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"]),
    "kar_diff": ("What changed between two candidates.", {"a": {"type": "string"}, "b": {"type": "string"}},
                 ["a", "b"]),
    "kar_docs": ("GPU documentation: action find (TERM), get (NAME or URL, optional section), or list.",
                 {"action": {"type": "string", "enum": ["find", "get", "list"]}, "term": {"type": "string"},
                  "name": {"type": "string"}, "section": {"type": "string"}, "in": {"type": "string"}}, ["action"]),
    "kar_note": ("Record a measured fact for every agent.", {"text": {"type": "string"}}, ["text"]),
    "kar_post": ("Message another lane, or all.", {"to": {"type": "string"}, "text": {"type": "string"}},
                 ["to", "text"]),
    "kar_findings": ("Survey lanes: hand back the findings file.", {"file": FILE}, ["file"]),
    "kar_plan": ("Plan lanes: hand back the plan (JSON file).", {"file": FILE}, ["file"]),
    "kar_submit": ("Build and stack lanes: hand the kernel back; the harness decides.",
                   {"file": FILE, "report": FILE}, ["file"]),
    "kar_abandon": ("Build and stack lanes: give the work up, with the numbers.", {"why": {"type": "string"}},
                    ["why"]),
    "kar_audit": ("Audit lanes: hand back the verdict.", {"verdict": {"type": "string", "enum": ["ok", "hazard"]},
                                                         "text": {"type": "string"}}, ["verdict", "text"]),
}


def _argv(name, a):
    cmd = name[len("kar_"):]
    out = [cmd]
    if cmd == "try":
        out += [str(x) for x in (a.get("files") or [a.get("file", "")])]
    elif cmd in ("check", "bound", "profile", "ncu", "sass", "stress", "findings", "plan", "submit"):
        out.append(str(a["file"]))
    if a.get("reps"):
        out += ["--reps", str(int(a["reps"]))]
    if cmd == "ncu" and a.get("shape"):
        out += ["--shape", str(a["shape"])]
    if cmd in ("test", "register"):
        out.append(str(a["testfile"]))
        if cmd == "test" and a.get("file"):
            out.append(str(a["file"]))
        if cmd == "register":
            out.append(str(a["title"]))
    if cmd == "submit" and a.get("report"):
        out += ["--report", str(a["report"])]
    if cmd == "docs":
        act = a.get("action", "list")
        out.append(act)
        if act == "find":
            out.append(str(a.get("term", "")))
            if a.get("in"):
                out += ["--in", str(a["in"])]
        elif act == "get":
            out.append(str(a.get("name", "")))
            if a.get("section"):
                out += ["--section", str(a["section"])]
    if cmd == "copy":
        out += [str(a["src"]), str(a["dst"])]
    if cmd == "diff":
        out += [str(a["a"]), str(a["b"])]
    if cmd in ("note",):
        out.append(str(a["text"]))
    if cmd == "post":
        out += [str(a["to"]), str(a["text"])]
    if cmd == "abandon":
        out.append(str(a["why"]))
    if cmd == "audit":
        out += [str(a["verdict"]), str(a["text"])]
    return out


class Server:
    def __init__(self, ws, lane=None):
        self.ws, self.lane = os.path.abspath(ws), lane

    def call(self, name, args):
        from kar.run import client
        os.environ.setdefault("KAR_WAIT_S", os.environ.get("KAR_MCP_WAIT_S", "45"))
        argv = _argv(name, args)
        buf = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self.ws)
            with contextlib.redirect_stdout(buf):
                code = client.main(self.ws, argv, lane=self.lane)
        finally:
            os.chdir(cwd)
        text = buf.getvalue()
        if self.lane is None and argv[0] == "next":
            m = re.search(r"You are lane (L\d+)", text)
            if m:
                self.lane = m.group(1)
                text += (f"\n\n(This MCP server now speaks for lane {self.lane}: use the kar_* tools, not "
                         f"./lanes/{self.lane}/kar.)")
        return text, code

    def handle(self, msg):
        mid, method = msg.get("id"), msg.get("method")
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "kar", "version": "0.3"},
                "instructions": "Tools for one lane of a kar GPU-kernel optimization run. Call kar_next first and "
                                "follow it. Read AGENTS.md in the workspace for the rules."}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            tools = [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p,
                                                                   "required": r}}
                     for n, (d, p, r) in TOOLS.items()]
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}}
        if method == "tools/call":
            p = msg.get("params", {})
            name = p.get("name")
            if name not in TOOLS:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
            try:
                text, code = self.call(name, p.get("arguments") or {})
            except Exception as e:  # noqa: BLE001 - every call gets an answer
                text, code = f"harness error: {type(e).__name__}: {e}", 4
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}],
                                                            "isError": code not in (0, 2)}}
        if mid is None:                     # a notification (initialized, cancelled, ...)
            return None
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m kar.mcp")
    ap.add_argument("--ws", required=True, help="the run's workspace (RUN/ws)")
    ap.add_argument("--lane", help="speak for this lane (default: kar_next hands one out)")
    a = ap.parse_args(argv)
    srv = Server(a.ws, a.lane)
    out = sys.stdout
    sys.stdout = sys.stderr                  # nothing but JSON-RPC may reach the real stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        reply = srv.handle(msg)
        if reply is not None:
            out.write(json.dumps(reply) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
