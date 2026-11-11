"""Run machinery that needs no GPU: the ledger's hash chain (also under concurrent writers), the team scheduler
(survey -> plan -> builds, stacking, audits, takeover), plan validation and the strategy rule, the plateau rule,
lane paths, messages, the command parser and the request queue."""
import json
import os
import threading
import time

import pytest

from kar.run import client, commands, server
from kar.run.session import DEFAULTS, Run, RunError
from kar.run.store import Ledger, write_json


def test_ledger_chain_detects_edits(tmp_path):
    lg = Ledger(str(tmp_path))
    for i in range(4):
        lg.append("candidates", {"id": f"c{i:04d}", "effect": 0.01 * i})
    assert lg.verify() == {"candidates.jsonl": "ok"}
    p = tmp_path / "candidates.jsonl"
    rows = p.read_text().splitlines()
    rows[1] = rows[1].replace("0.01", "0.5")
    p.write_text("\n".join(rows) + "\n")
    assert lg.verify()["candidates.jsonl"] == "broken at record 2"


def test_ledger_chain_survives_concurrent_writers(tmp_path):
    lg = Ledger(str(tmp_path))
    ts = [threading.Thread(target=lambda k=k: [lg.append("cost", {"k": k, "i": i}) for i in range(25)])
          for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert lg.verify() == {"cost.jsonl": "ok"} and len(lg.rows("cost")) == 200


def _fake_run(tmp_path, sol=0.5, **cfg_over):
    """A run folder with state but no GPU work, for the bookkeeping paths."""
    from conftest import TOY
    root = tmp_path / "run"
    for d in ("ledger", "candidates/c0000", "evidence", "milestones", "agents", "gpu"):
        (root / d).mkdir(parents=True)
    for d in ("task", "evidence", "candidates", "lanes", "queue", "notes", "survey"):
        (root / "ws" / d).mkdir(parents=True)
    (root / "candidates/c0000/kernel.py").write_text("from reference import reference as kernel_fn\n")
    cfg = {**DEFAULTS, "task_root": os.path.join(TOY, "rmsnorm"), "freeze": {}, "gpu_name": "test",
           "arch": "sm_00", "harness": "", "task_name": "rmsnorm_bf16", "layout": 2,
           "target_sol": 0.95 if sol >= 0.85 else 0.85, **cfg_over}
    write_json(str(root / "run.json"), cfg)
    st = {"v": 2, "phase": "running", "cycle": 1, "planned": False, "best": "c0000", "best_sol": sol,
          "best_score": 1.0, "best_history": [], "hyps": [], "lanes": {}, "inbox": {}, "next_lane": 1,
          "next_cand": 1, "next_hyp": 1, "since_keep": 0, "plateaus": 0, "plateau_note": "", "submits": 0,
          "milestones": 0, "milestone_base": "c0000", "survey": None, "stack": [], "audited": ["c0000"],
          "to_audit": [], "plan_failures": {}, "target_met": None, "done_reason": None,
          "started": "2026-11-10T00:00:00", "started_ts": time.time(), "driver": None, "baseline": "c0000"}
    write_json(str(root / "state.json"), st)
    for f in ("facts.md", "dead_ends.md"):
        (root / "ledger" / f).write_text("")
    return Run(str(root))


H = {"title": "t", "target": "all", "mechanism": "m", "evidence": "e", "predicted_gain_pct": 10, "arithmetic": "a",
     "bound_test": "b"}


def _plan(run, lid, hs):
    p = run.w("lanes", lid, "plan.json")
    with open(p, "w") as f:
        json.dump({"hypotheses": hs}, f)
    return run.plan(lid, p)


def _findings(run, lid):
    p = run.w("lanes", lid, "FINDINGS.md")
    with open(p, "w") as f:
        f.write("Summary: the t4096 store path is at 61% of floor.\n\n" + "measured numbers here. " * 20)
    return run.findings(lid, p)


def _get(run, holder="h"):
    lane, text = run.assign(holder)
    return lane, text


def test_team_round_survey_then_plan_then_builds(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=3, survey=2)       # 5.6% from the target: no structural change needed
    a, _ = _get(run)
    b, _ = _get(run)
    assert (a["role"], a["lens"]) == ("survey", "structure") and (b["role"], b["lens"]) == ("survey", "memory")
    c, text = _get(run)
    assert c is None and text.startswith("WAIT")                # the plan waits for the survey
    assert _findings(run, a["id"])[1] == 0
    assert _get(run)[0] is None                                 # still one survey lane at work
    _findings(run, b["id"])
    p, _ = _get(run)
    assert p["role"] == "plan"
    nxt = run.lane_next(p["id"])
    assert "t4096 store path" in nxt and "survey/r01/" in nxt   # the planner reads the findings
    out, code = _plan(run, p["id"], [{**H, "title": f"idea{i}", "predicted_gain_pct": g}
                                     for i, g in enumerate([2, 9, 5])])
    assert code == 0, out
    builds = [_get(run)[0] for _ in range(3)]
    assert [x["role"] for x in builds] == ["build"] * 3
    titles = [run._hyp(run.state, x["hyp"])["title"] for x in builds]
    assert titles == ["idea1", "idea2", "idea0"]                # in order of predicted gain
    idle, _ = _get(run)                                          # every build still in flight: nothing new to plan
    assert idle is None and run.state["cycle"] == 1
    out, code = commands.execute(run.root, ["abandon", "measured: the bound says it cannot pay"], run.w(),
                                 builds[2]["id"])
    assert code == 0, out
    nxt_lane, _ = _get(run)                                      # a verdict came in: the next round starts
    st = run.state
    assert st["cycle"] == 2 and nxt_lane["role"] == "plan"      # the best did not change: the survey is reused
    assert "has not changed since round 1" in run.lane_next(nxt_lane["id"])


def test_plan_needs_fields_and_strategy_rule(tmp_path):
    run = _fake_run(tmp_path, sol=0.5)          # target 0.85 needs +70%
    p, _ = _get(run)
    assert p["role"] == "plan"                  # one lane: no survey
    out, code = _plan(run, p["id"], [{k: v for k, v in H.items() if k != "bound_test"}])
    assert code == 1 and "bound_test" in out
    out, code = _plan(run, p["id"], [H, {**H, "title": "u", "predicted_gain_pct": 5}])
    assert code == 1 and "structural" in out
    out, code = _plan(run, p["id"], [H, {**H, "title": "rewrite", "predicted_gain_pct": 5, "structural": True}])
    assert code == 0, out
    assert run.state["planned"] and run.state["lanes"][p["id"]]["status"] == "closed"


def test_plateau_drops_queue_then_stops(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, plateau=2)
    p, _ = _get(run)
    _plan(run, p["id"], [{**H, "title": f"h{i}"} for i in range(4)])
    for _ in range(2):
        lane, _ = _get(run)
        out, code = run.abandon(lane["id"], "bound test says it cannot pay: 0.1% at most")
        assert code == 0
    st = run.state
    assert not st["planned"] and "Plateau" in st["plateau_note"] and st["plateaus"] == 1
    assert sum(h["status"] == "dropped" for h in st["hyps"]) == 2
    p, _ = _get(run)
    assert p["role"] == "plan" and "Plateau" in run.lane_next(p["id"])
    _plan(run, p["id"], [{**H, "title": f"k{i}"} for i in range(2)])
    for _ in range(2):
        lane, _ = _get(run)
        run.abandon(lane["id"], "bound test says it cannot pay: 0.1% at most")
    assert run.state["phase"] == "done" and "plateau" in run.state["done_reason"]
    assert run.lane_next(lane["id"]).startswith("STOP")
    assert "h0001" in open(run.p("ledger", "dead_ends.md")).read()
    assert "h0001" in open(run.w("notes", "dead_ends.md")).read()      # published to the workspace


def test_paths_must_be_in_the_lane(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run)
    b, _ = _get(run)
    (open(run.w("lanes", b["id"], "cand.py"), "w")).write("x = 1\n")
    with pytest.raises(RunError):
        run._inside(a["id"], run.w("lanes", b["id"], "cand.py"))          # another lane's file
    with pytest.raises(RunError):
        run._inside(a["id"], run.w("lanes", a["id"], "..", "..", "AGENTS.md"))
    with pytest.raises(RunError):
        run._inside(a["id"], run.p("run.json"))
    assert run._target(a["id"], "best")[0] == "c0000"                   # frozen candidates by name


def test_messages_arrive_with_the_next_command(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run)
    b, _ = _get(run)
    out, code = commands.execute(run.root, ["post", b["id"], "t4096 stores are 8-byte; 16-byte gave +4%"],
                                 lane=a["id"])
    assert code == 0 and b["id"] in out
    out, _ = commands.execute(run.root, ["board"], lane=b["id"])
    assert "Messages for you" in out and "16-byte gave +4%" in out and f"{b['id']} (you)" in out
    out, _ = commands.execute(run.root, ["board"], lane=b["id"])
    assert "Messages for you" not in out                                 # delivered once


def test_win_on_an_old_base_goes_to_a_stacker(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0, audit="off")
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run)
    with run.lock():                          # another lane's win moved the best on
        st = run.state
        os.makedirs(run.p("candidates", "c0001"))
        open(run.p("candidates", "c0001", "kernel.py"), "w").write("x = 2\n")
        st["best"] = "c0001"
        run.save(st)
    res = {"verdict": "KEEP", "effect": 0.05, "ci": [0.01, 0.09], "why": "", "shapes": {}}
    out, code = run._decide(a["id"], "c0002", {"id": "c0002", "hyp": a["hyp"], "lane": a["id"], "role": "build",
                                               "title": "t", "parent": "c0000", "base": "c0000"},
                            res, {"job_dir": run.root, "noise": None}, True, None, "c0000", "c0000", False, [])
    assert "WINS ON ITS BASE" in out
    st = run.state
    assert st["stack"] and st["stack"][0]["cid"] == "c0002" and st["since_keep"] == 0
    s, _ = _get(run)
    assert s["role"] == "stack" and s["items"][0]["cid"] == "c0002" and s["base"] == "c0001"
    assert "c0002" in run.lane_next(s["id"]) and "diff c0000 c0002" in run.lane_next(s["id"])


def test_audit_hazard_queues_a_fix_first(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run)
    _plan(run, p["id"], [{**H, "predicted_gain_pct": 30}])
    with run.lock():
        st = run.state
        st["to_audit"].append("c0000")
        st["audited"] = []
        run.save(st)
    au, _ = _get(run)
    assert au["role"] == "audit" and au["base"] == "c0000"
    out, code = run.audit(au["id"], "hazard", "block i spins on a flag block i-1 sets; this needs in-order "
                                              "dispatch. Take the order from an atomic ticket.")
    assert code == 0 and "fix is queued" in out
    with run.lock():                          # even while the next round is being surveyed
        st = run.state
        st["planned"] = False
        run.save(st)
    run.cfg["survey"] = 2
    nxt, _ = _get(run)
    h = run._hyp(run.state, nxt["hyp"])
    assert h["kind"] == "fix"                                            # ahead of the 30% idea and the survey
    assert "This is a **fix**" in run.lane_next(nxt["id"])


def test_a_lane_without_an_agent_is_taken_over_first(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run, holder="s1")
    run.end_session(a["id"], progressed=True)        # the session ended with the lane still open
    again, _ = _get(run, holder="s2")
    assert again["id"] == a["id"] and run.state["lanes"][a["id"]]["holder"] == "s2"
    run.end_session(a["id"], progressed=False)
    _get(run, holder="s3")
    run.end_session(a["id"], progressed=False)       # two sessions in a row without a step: closed as stalled
    st = run.state
    assert st["lanes"][a["id"]]["status"] == "closed" and "stalled" in st["lanes"][a["id"]]["outcome"]
    assert run._hyp(st, a["hyp"])["status"] == "dead"


def test_commands_parse_errors_and_manual_lanes(tmp_path):
    run = _fake_run(tmp_path, sol=0.9)
    out, code = commands.execute(run.root, ["try"])
    assert code == 1 and "./kar" in out
    out, code = commands.execute(run.root, ["help"])
    assert code == 0 and "submit" in out
    out, code = commands.execute(run.root, ["try", "x.py"])
    assert code == 1 and "belongs to a lane" in out
    out, code = commands.execute(run.root, ["next"])                     # by hand: a lane is handed out
    assert code == 0 and "You are lane L01 (plan)" in out and "./lanes/L01/kar" in out
    assert os.access(run.w("lanes", "L01", "kar"), os.X_OK)


def test_queue_round_trip_with_lanes(tmp_path, capsys):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    lane, _ = _get(run)
    stop = threading.Event()
    t = threading.Thread(target=server.serve, args=(run.root, stop), daemon=True)
    t.start()
    q = run.w("queue")
    for _ in range(50):
        if client._serving(q):
            break
        time.sleep(0.1)
    codes = []
    ths = [threading.Thread(target=lambda: codes.append(client.main(run.w(), ["board"], lane=lane["id"])))
           for _ in range(4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(30)
    code = client.main(run.w(), ["status"])
    stop.set()
    t.join(5)
    assert code == 0 and codes == [0, 0, 0, 0]
    assert "phase running" in capsys.readouterr().out
    assert sum(f.endswith(".done.json") for f in os.listdir(q)) == 5


def test_lanes_of_a_dead_driver_are_released(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run, holder=f"kar-run:{os.getpid()}:0")          # this process: alive, keeps its lane
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run, holder="kar-run:999999999:1")             # no such process
    b, _ = _get(run, holder=f"kar-run:{os.getpid()}:0")
    assert run.release_dead_holders() == [a["id"]]
    st = run.state
    assert st["lanes"][a["id"]]["holder"] is None and st["lanes"][b["id"]]["holder"] is not None
    again, _ = _get(run, holder="kar-run:1:0")
    assert again["id"] == a["id"]                               # resumed first


def test_build_lanes_start_with_cand_and_copy_stays_in_the_lane(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run)
    assert open(run.w("lanes", a["id"], "cand.py")).read() == open(run.p("candidates", "c0000", "kernel.py")).read()
    out, code = commands.execute(run.root, ["copy", "best", f"lanes/{a['id']}/v2.py"], cwd=run.w(), lane=a["id"])
    assert code == 0 and os.path.exists(run.w("lanes", a["id"], "v2.py"))
    out, code = commands.execute(run.root, ["copy", "best", "notes/x.py"], cwd=run.w(), lane=a["id"])
    assert code == 1 and "your lane folder" in out
    out, code = commands.execute(run.root, ["copy", run.p("run.json"), f"lanes/{a['id']}/"], cwd=run.w(),
                                 lane=a["id"])
    assert code == 1 and "workspace" in out                          # the harness's own files stay out of reach


def test_a_stalled_fix_goes_back_to_the_queue(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0)
    with run.lock():
        st = run.state
        st["hyps"].append({"id": "h0009", "status": "open", "kind": "fix", "title": "fix", "predicted_gain_pct": 0.01,
                           "target": "", "mechanism": "", "evidence": "", "arithmetic": "", "bound_test": ""})
        run.save(st)
    a, _ = _get(run)
    assert a["hyp"] == "h0009"
    run.end_session(a["id"], progressed=False)
    _get(run)
    run.end_session(a["id"], progressed=False)
    st = run.state
    assert st["lanes"][a["id"]]["status"] == "closed" and run._hyp(st, "h0009")["status"] == "open"


def test_no_new_survey_on_a_low_budget(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=2)
    run.cfg["budget"]["agent_usd"] = 10
    a, _ = _get(run)
    b, _ = _get(run)
    _findings(run, a["id"])
    _findings(run, b["id"])
    p, _ = _get(run)
    _plan(run, p["id"], [H])
    lane, _ = _get(run)
    with run.lock():                              # the best moves on: normally the next round surveys it again
        st = run.state
        st["best"] = "c0001"
        os.makedirs(run.p("candidates", "c0001"))
        open(run.p("candidates", "c0001", "kernel.py"), "w").write("x = 1\n")
        run.save(st)
    run.ledger.append("cost", {"kind": "agent", "usd": 7.5, "tokens": 1})
    nxt, _ = _get(run)
    assert nxt["role"] == "plan"                  # 75% spent: plan from the last findings, no new survey


def test_a_proven_hazard_is_fixed_by_any_later_keep(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0, audit="off")
    p, _ = _get(run)
    _plan(run, p["id"], [H, {**H, "title": "x"}])
    a, _ = _get(run)
    run.ledger.append("tests", {"event": "registered", "id": "t001", "lane": "L09", "file": "tests/s001.py",
                                "best_passes": False})
    with run.lock():
        st = run.state
        st["hyps"].append({"id": "h0009", "status": "open", "kind": "fix", "tests": ["t001"], "title": "fix",
                           "predicted_gain_pct": 0.01})
        run.save(st)
    res = {"verdict": "KEEP", "effect": 0.05, "ci": [0.01, 0.09], "why": "", "shapes": {}}
    with run.lock():
        st = run.state
        run._fixes_met(st, "c0005", a["hyp"])
        run.save(st)
    h = run._hyp(run.state, "h0009")
    assert h["status"] == "fixed" and "t001" in h["why"] and not run._fixes_pending(run.state)


def test_help_and_bad_arguments_never_exit(tmp_path):
    run = _fake_run(tmp_path, sol=0.9)
    lane, _ = _get(run)
    for argv in (["test", "x.py", "--help"], ["--help"], ["submit", "-h"], ["try", "a.py", "--bogus"], ["nope"]):
        out, code = commands.execute(run.root, argv, lane=lane["id"])      # returns text; never raises SystemExit
        assert isinstance(out, str) and "next" in out


def test_judge_policy_quota_noise_and_auto_checks(tmp_path):
    import sys as _sys
    from kar import judge as J
    run = _fake_run(tmp_path, sol=0.9)
    fake = tmp_path / "fake_judge.py"
    fake.write_text("import sys, hashlib; s = hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest();\n"
                    "print('score:', 2.0 + int(s[:2], 16) / 2550)\n")
    run.cfg["judge"] = {"check": f"{_sys.executable} {fake} {{file}}", "submit": f"{_sys.executable} {fake} {{file}}",
                        "quota": {"check": 3, "submit": 1}, "policy": {"check": "auto", "submit": "ask"}}
    with pytest.raises(J.JudgeError, match="approval"):
        J.run_judge(run, "submit", "c0000", by="agent")                  # a scored submit needs a person
    r1 = J.run_judge(run, "check", "c0000", by="t")
    r2 = J.run_judge(run, "check", "c0000", by="t")
    assert r1["score"] == r2["score"] and r1["sha"] == r2["sha"]
    sd, dof = J.noise(run)
    assert sd == 0 and dof == 1
    J.run_judge(run, "check", "c0000", by="t")
    with pytest.raises(J.JudgeError, match="quota"):
        J.run_judge(run, "check", "c0000", by="t")
    rid = J.request(run, "submit", "c0000", by="L03", why="stack of three wins")
    assert [p["id"] for p in J.pending(run)] == [rid]
    rec = J.approve(run, rid)
    assert rec["kind"] == "submit" and rec["approved_by"] == "user" and not J.pending(run)
    assert "submits left in the window: 0 of 1" in J.status_text(run)
    assert J.tick(run) is None                                           # the baseline is never auto-checked


def test_session_budget_never_gives_a_cap_too_small_to_finish():
    from kar.agents.drive import session_budget as sb
    go, cap, need = sb(15, 0, 4, [])                    # $15, four free lanes: $3.75 each covers the default need
    assert go and cap == 3.75 and need == 3.0
    go, cap, need = sb(8.95, 0, 4, [2.77, 3.28, 1.95])  # what a session costs here is now known: about $4.15
    assert go and abs(need - 4.155) < 1e-9 and cap == 4.475       # four do not fit; two at a time, with enough each
    go, cap, _ = sb(8.95, 4.475, 3, [2.77, 3.28, 1.95])  # the first one's cap is committed: one more starts
    assert go and cap == 4.475
    go, cap, _ = sb(8.95, 8.95, 2, [2.77, 3.28, 1.95])   # both committed: nobody else starts
    assert not go
    go, cap, _ = sb(20, 0, 4, [2.77, 3.28, 1.95])       # four do not fit at $4.15 each... but 4 x 4.15 = 16.6 does
    assert go and cap == 5.0
    go, cap, _ = sb(10, 0, 4, [2.77, 3.28, 1.95])       # two at a time
    assert go and cap == 5.0
    go, cap, _ = sb(5, 5, 3, [2.77, 3.28, 1.95])
    assert not go
    go, cap, _ = sb(2.0, 0, 4, [2.77, 3.28, 1.95])      # under most of a session's need: left unspent
    assert not go
    go, cap, _ = sb(3.0, 0, 4, [2.77, 3.28, 1.95])      # most of a need: one last session
    assert go and cap == 3.0
    go, cap, _ = sb(100, 0, 4, [2.0, 2.0], user_cap=2.5)
    assert go and cap == 2.5


def test_the_task_writers_example_is_the_toy_task_and_loads(tmp_path):
    import kar
    from kar import newtask
    from kar.task import spec
    here = os.path.dirname(os.path.dirname(os.path.abspath(kar.__file__)))
    ex = os.path.join(os.path.dirname(kar.__file__), "agents", "prompts", "example_task")
    for name in ("reference.py", "inputs.py", "shapes.py", "opmodel.py", "baseline.py"):
        assert open(os.path.join(ex, name)).read() == open(os.path.join(here, "tasks", "toy", "gemv", name)).read()
    assert spec.load(ex).name                                   # the example is itself a valid task
    root = newtask.create(str(tmp_path / "t"), "an op")
    guide = open(os.path.join(root, "AGENTS.md")).read()
    assert ex in guide and "$example" not in guide


def test_a_hazard_stays_on_the_books_when_its_fix_is_abandoned(tmp_path):
    run = _fake_run(tmp_path, sol=0.9, lanes=2, survey=0, audit="off")
    with run.lock():
        st = run.state
        st["planned"] = True
        st["hyps"].append({"id": "h0009", "cycle": 1, "status": "open", "kind": "fix", "tests": ["t001"],
                           "title": "fix", "target": "t", "mechanism": "overflows on large finite x",
                           "evidence": "e", "predicted_gain_pct": 0.01, "arithmetic": "a", "bound_test": "b",
                           "structural": False, "lane": None, "parent": st["best"]})
        run.save(st)
    for attempt in range(3):
        lane, _ = _get(run)
        assert lane["role"] == "build" and lane["hyp"] == "h0009"            # the fix comes first, again
        assert "Correct comes before fast" in run.lane_next(lane["id"])
        out, code = run.abandon(lane["id"], "the guard costs 2.4% and I could not find why")
        assert code == 0
        h = run._hyp(run.state, "h0009")
        assert h["status"] == ("open" if attempt < 2 else "unfixed") and len(h["notes"]) == attempt + 1
    text = run.status()[0]
    assert "KNOWN HAZARD in the best" in text and "t001" in text and "overflows on large finite x" in text
