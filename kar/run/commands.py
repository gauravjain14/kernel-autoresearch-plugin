"""The agents' commands, parsed once for both paths: run directly (`./kar` with no harness process serving the run)
or executed by the harness process for a sandboxed agent (server.py). Every lane has its own `./lanes/<id>/kar`,
which names the lane; the workspace's `./kar` hands out lanes."""
import argparse
import os
import re
import time

from kar.run.session import BudgetSpent, Run, RunError

USAGE = """{tool} <command>

  next                               what to do now (start here, and come back after every step)
  board                              the team: every lane at work, the queue, recent decisions
  check FILE                         lint, smoke, correctness on every shape
  try FILE [FILE...] [--reps N]      check, then a quick A/B against the current best (several variants: one job)
  bound PROBE [--reps N]             time an incomplete-on-purpose probe against the best, output not checked
  profile FILE                       full profile page (gap table, nsys, ncu summary, SASS audit)
  ncu FILE [--shape S] [--set full]  every ncu section on one scored shape, as text
  sass FILE                          the full SASS of every kernel the file loads
  stress FILE                        extra seeds, concurrent calls on several streams, back-to-back calls
  test TESTFILE [FILE]               run your own test (help/test_example.py) on a kernel (default: your cand.py)
  register TESTFILE "what it checks" make your test part of the run: the reference must pass it; every later
                                     submit must too
  diff cA cB                         what changed between two candidates
  copy SRC DST                       copy a candidate (best, base, c0003) or a workspace file into your lane
  docs find "TERM" | docs get NAME [--section "TERM"] | docs list
                                     the PTX ISA, CUDA guides, this GPU's tuning guide, ncu/nsys guides, Triton...
  judge | judge request best "why"    the external judge's results, quota and noise; ask for a check of a stack
  note "fact, with numbers"          record a measured fact for every agent
  post <lane|all> "message"          message other lanes (delivered with their next command)

  survey lanes:  findings FILE                     hand back the survey
  plan lanes:    plan FILE                         hand back the hypotheses (JSON)
  build, stack:  submit FILE [--report FILE]       hand the kernel back; the harness decides
                 abandon "why, with numbers"       give the work up, with the reason
  audit lanes:   audit ok|hazard "text"            hand back the audit

FILE is a file in your lane folder, or `best`, `base` (your lane's starting point) or a candidate id like c0003.
GPU commands can take minutes; if one prints "still running", run `{tool} wait`.
"""

CID = re.compile(r"^c\d{4}$")


class _Parser(argparse.ArgumentParser):
    """Never exits: argparse's exit (on --help or an error) would end the server thread that runs the command, and
    that request would never be answered (an agent once waited 23 minutes on `test ... --help`)."""

    def __init__(self, *a, **kw):
        kw["add_help"] = False
        super().__init__(*a, **kw)

    def error(self, message):
        raise RunError(f"{message}\n\n{USAGE.format(tool='./kar')}")

    def exit(self, status=0, message=None):
        raise RunError(f"{message or ''}\n{USAGE.format(tool='./kar')}")


def _parser():
    ap = _Parser(prog="./kar")
    sub = ap.add_subparsers(dest="cmd", parser_class=_Parser)
    for name in ("next", "status", "board", "help"):
        sub.add_parser(name)
    for name in ("plan", "check", "profile", "findings", "sass", "stress"):
        sub.add_parser(name).add_argument("file")
    p = sub.add_parser("try")
    p.add_argument("file", nargs="+")
    p.add_argument("--reps", type=int)
    p = sub.add_parser("bound")
    p.add_argument("file")
    p.add_argument("--reps", type=int)
    p = sub.add_parser("test")
    p.add_argument("testfile")
    p.add_argument("file", nargs="?")
    p = sub.add_parser("register")
    p.add_argument("testfile")
    p.add_argument("title", nargs="+")
    p = sub.add_parser("ncu")
    p.add_argument("file")
    p.add_argument("--shape")
    p.add_argument("--set", dest="section_set", default="full")
    p = sub.add_parser("submit")
    p.add_argument("file")
    p.add_argument("--report")
    sub.add_parser("abandon").add_argument("why", nargs="+")
    p = sub.add_parser("audit")
    p.add_argument("verdict")
    p.add_argument("text", nargs="+")
    sub.add_parser("note").add_argument("text", nargs="+")
    p = sub.add_parser("post")
    p.add_argument("to")
    p.add_argument("text", nargs="+")
    p = sub.add_parser("copy")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    return ap


def _driven(st):
    d = st.get("driver")
    return bool(d and time.time() - d.get("t", 0) < 60)


def execute(root, argv, cwd=None, lane=None):
    """-> (text, exit code). Never raises for the agent's mistakes; they come back as text."""
    cwd = cwd or os.getcwd()
    tool = f"./lanes/{lane}/kar" if lane else "./kar"
    if argv[:1] == ["judge"]:                  # read the judge's state; agents may ask for a check, never submit
        from kar import judge as J
        run = Run(root)
        if argv[1:2] == ["request"] and len(argv) >= 4:
            if J.config(run) is None:
                return "no judge is configured for this run", 1
            cid = {"best": run.state["best"]}.get(argv[2], argv[2])
            rid = J.request(run, "check", cid, by=lane or "agent", why=" ".join(argv[3:]))
            return f"judge check of {cid} requested as {rid}; the harness or a person decides", 0
        return J.status_text(run), 0
    if argv[:1] == ["docs"]:                   # documentation: no lane, no GPU, fetched by the harness process
        from kar import docs
        try:
            arch = Run(root).cfg.get("arch")
        except RunError:
            arch = None
        return docs.main(argv[1:], arch=arch)
    if any(x in ("-h", "--help") for x in argv):
        return USAGE.format(tool=tool), 0
    try:
        a = _parser().parse_args(argv)
        if a.cmd in (None, "help"):
            return USAGE.format(tool=tool), 0
        run = Run(root)

        def path(x):
            if x in ("best", "base") or CID.match(x):
                return x
            return x if os.path.isabs(x) else os.path.normpath(os.path.join(cwd, x))

        msgs = run.touch(lane) if lane else ""             # every command from a lane renews it, brings its mail
        if a.cmd == "status":
            out, code = run.status()
            return msgs + out, code
        if a.cmd == "board":
            return msgs + run.board(lane), 0
        if a.cmd == "diff":
            return msgs + (run.diff(a.a, a.b) or "(no difference)"), 0
        if lane is None:
            if a.cmd == "note":
                return run.note(None, " ".join(a.text))
            if a.cmd == "post":
                return run.post(None, a.to, " ".join(a.text))
            if a.cmd != "next":
                raise RunError(f"`{a.cmd}` belongs to a lane: run ./kar next to get one, then use that lane's "
                               f"command (./lanes/<id>/kar {a.cmd} ...)")
            if _driven(run.state):
                return ("This run is driven by `kar run`, which gives each agent its lane and its command "
                        "(./lanes/<id>/kar). Use the command from your prompt."), 1
            got, text = run.assign(holder="manual")
            if got is None:
                return text, 0 if text.startswith("WAIT") else 1
            lid = got["id"]
            return (f"You are lane {lid} ({got['role']}). From now on use ./lanes/{lid}/kar for every command "
                    f"(./lanes/{lid}/kar next shows this step again).\n\n" + run.lane_next(lid)), 0
        out, code = _lane_cmd(run, lane, a, path)
        return msgs + out, code
    except (RunError, BudgetSpent) as e:
        return str(e), 1


def _lane_cmd(run, lid, a, path):
    c = a.cmd
    if c == "next":
        return run.lane_next(lid), 0
    if c == "note":
        return run.note(lid, " ".join(a.text))
    if c == "post":
        return run.post(lid, a.to, " ".join(a.text))
    if c == "check":
        return run.check(lid, path(a.file))
    if c == "try":
        return run.try_(lid, [path(x) for x in a.file], reps=a.reps)
    if c == "bound":
        return run.bound(lid, path(a.file), reps=a.reps)
    if c == "profile":
        return run.profile(lid, path(a.file))
    if c == "ncu":
        return run.ncu(lid, path(a.file), shape=a.shape, section_set=a.section_set)
    if c == "sass":
        return run.sass(lid, path(a.file))
    if c == "stress":
        return run.stress(lid, path(a.file))
    if c == "copy":
        return run.copy(lid, a.src, path(a.dst))
    if c == "test":
        return run.test(lid, path(a.testfile), path(a.file) if a.file else None)
    if c == "register":
        return run.register(lid, path(a.testfile), " ".join(a.title))
    if c == "findings":
        return run.findings(lid, path(a.file))
    if c == "plan":
        return run.plan(lid, path(a.file))
    if c == "submit":
        return run.submit(lid, path(a.file), report=path(a.report) if a.report else None)
    if c == "abandon":
        return run.abandon(lid, " ".join(a.why))
    if c == "audit":
        return run.audit(lid, a.verdict, " ".join(a.text))
    return USAGE.format(tool=run.tool(lid)), 1
