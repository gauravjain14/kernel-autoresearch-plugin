"""What an agent in a lane can do. Every command names its lane; GPU work runs outside the run lock, so lanes work at
once and their GPU jobs queue for the GPU.

    any lane     next, board, note, post, check, diff
    survey       try, bound, profile, ncu, sass, stress, test, register, then findings FILE
    plan         plan FILE
    build/stack  try, bound, profile, ncu, sass, stress, then submit FILE or abandon WHY
    audit        profile, ncu, sass, stress, then audit ok|hazard TEXT

The judge (submit): lint, smoke, correctness, stress, the team's registered tests, then the A/B against the current
best in one GPU job. A win
over the current best becomes the best. A candidate built on an older best that wins against that base but not
against the current best goes to a stacker, who ports it onto the current best.
"""
import json
import math
import os

from kar import checks
from kar.run import records
from kar.run.store import now


def _ci(c):
    return f"[{c[0] * 100:+.2f}, {c[1] * 100:+.2f}]"


class Roles:
    FIELDS = ("title", "target", "mechanism", "evidence", "predicted_gain_pct", "arithmetic", "bound_test")

    # ---- shared helpers -----------------------------------------------------------------------------------

    def tool(self, lid):
        return f"./lanes/{lid}/kar"

    def _inside(self, lid, path):
        from kar.run.session import RunError
        full = os.path.realpath(path if os.path.isabs(path) else self.w(path))
        mine = os.path.realpath(self.w("lanes", lid))
        if not full.startswith(mine + os.sep):
            raise RunError(f"{path}: your files live under lanes/{lid}/")
        if not os.path.isfile(full):
            raise RunError(f"{path}: no such file")
        return full

    def _target(self, lid, arg):
        """A file in the lane, or `best`, `base` or a candidate id (its frozen file)."""
        from kar.run.session import RunError
        st = self.state
        lane = self._lane(st, lid)
        cid = {"best": st["best"], "base": lane["base"]}.get(arg, arg)
        if isinstance(cid, str) and len(cid) == 5 and cid[0] == "c" and cid[1:].isdigit():
            p = self.p("candidates", cid, "kernel.py")
            if not os.path.exists(p):
                raise RunError(f"no candidate {cid}")
            return cid, p
        full = self._inside(lid, arg)
        return os.path.splitext(os.path.basename(full))[0], full

    def _require(self, lane, roles, what):
        from kar.run.session import RunError
        if lane["status"] != "active":
            raise RunError(f"lane {lane['id']} is closed ({lane['outcome']}); run {self.tool(lane['id'])} next")
        if lane["role"] not in roles:
            raise RunError(f"`{what}` is not a {lane['role']} lane's command (see {self.tool(lane['id'])} next)")

    def _count_try(self, lid, what):
        from kar.run.session import RunError
        with self.lock():
            st = self.state
            if st["phase"] == "done":
                raise RunError(f"STOP: the run is over ({st['done_reason']}).")
            lane = self._lane(st, lid)
            self._require(lane, ("survey", "build", "stack", "audit"), what)
            cap = self.cfg["tries"].get(lane["role"], 0)
            if lane["tries"] >= cap:
                raise RunError(f"no GPU commands left in lane {lid} ({cap} used): hand back your result")
            lane["tries"] += 1
            self.save(st)
            return st, lane

    def _check_text(self, rec):
        if rec["verdict"] == "accept":
            shapes = rec.get("check", {}).get("shapes", [])
            worst = max(shapes, key=lambda s: s.get("mean_ratio", 0), default={})
            return (f"accepted: lint, smoke and correctness on {len(shapes)} shapes pass. Largest mean error vs the "
                    f"reference's own: {worst.get('mean_ratio', 0):.2f}x on {worst.get('shape', '-')} "
                    f"(limit {self.task.tolerance['mean_err_ratio']}x).")
        tb = ""
        for k in ("smoke", "check"):
            t = rec.get(k, {}).get("traceback")
            if t:
                tb = "\n" + t[-2500:]
        return f"REJECTED at {rec['stage']}: {rec['reason']}{tb}"

    # ---- next: the step for this lane ----------------------------------------------------------------------

    def lane_next(self, lid):
        from kar.run.session import LENSES, _template
        with self.lock():
            st = self.state
            if self._maybe_done(st):
                self.save(st)
            lane = self._lane(st, lid)
        if st["phase"] == "done":
            return f"STOP: the run is over ({st['done_reason']}). Nothing more to do; end the session."
        if lane["status"] != "active":
            return (f"Lane {lid} is finished: {lane['outcome']}\n\nEnd the session (if you are working by hand, run "
                    f"./kar next for new work).")
        tool = self.tool(lid)
        common = dict(lane=lid, tool=tool, status=self._status_line(st), cycle=lane["cycle"], best=st["best"],
                      base=lane["base"], profile=f"evidence/{st['best']}/profile.md", dead=self._dead_text(),
                      facts=self._facts_text(), team=self.lanes_n(),
                      tries_left=self.cfg["tries"].get(lane["role"], 0) - lane["tries"])
        role = lane["role"]
        if role == "survey":
            return _template("survey.md", lens=lane["lens"], lens_text=LENSES[lane["lens"]], **common)
        if role == "plan":
            sv = st.get("survey") or {}
            found = []
            for rel in sv.get("files", []):
                found.append(f"### {rel}\n\n" + records._read(self.w(rel))[:8000].strip())
            if sv.get("reused_from"):
                found.insert(0, f"(The best has not changed since round {sv['reused_from']}, so its survey still "
                                f"applies.)")
            return _template("plan.md", plateau=st.get("plateau_note") or "", strategy=self._strategy(st),
                             gap=records._gap_section(records._read(self.w("evidence", st["best"], "profile.md")))
                             or "(see the profile)", findings="\n\n".join(found) or "No survey this round: read the "
                             "profile yourself.", board=self.board(lid), kept=self._kept_text(),
                             max_hyps=self.max_hyps(), **common)
        b = self.task.bench
        if role == "build":
            h = self._hyp(st, lane["hyp"])
            moved = (f"The best has moved on since this lane opened: it is now {st['best']} "
                     f"(candidates/{st['best']}/kernel.py). If your change is independent of what changed, rebuild "
                     f"it on the new best; otherwise finish on your base and a stacker will port it.\n"
                     if st["best"] != lane["base"] else "")
            fix = ""
            if h.get("kind") == "fix" and h.get("tests"):
                fix = (f"This is a **fix**: the current best fails the team's registered test(s) "
                       f"{', '.join(h['tests'])}, which the reference passes, so the best is wrong on inputs real "
                       f"callers can pass. Correct comes before fast: your kernel is kept when it passes every gate "
                       f"and those tests, even if it is slower than the current best (the cost is recorded, and the "
                       f"team wins it back afterwards). Make the fix as cheap as you can, run `test` on it before "
                       f"`submit`, and do not abandon a working fix because it costs speed.\n")
            elif h.get("kind") == "fix":
                fix = ("This is a **fix**: it removes a hazard an auditor found by reading the code (no registered "
                       "test proves it). It is kept if it passes every gate and is not slower than the current best; "
                       "it does not need to be faster. If you can write a test that the reference passes and the "
                       "best fails, `register` it: the fix is then kept on correctness alone.\n")
            return _template("build.md", hid=h["id"], title=h["title"], target=h["target"], mechanism=h["mechanism"],
                             evidence=h["evidence"], predicted=h["predicted_gain_pct"], arithmetic=h["arithmetic"],
                             bound_test=h["bound_test"], moved=moved, fix=fix, try_reps=self.cfg["try_reps"],
                             fails_left=self.cfg["submit_failures"] - lane["fails"], copies=b["copies"],
                             reps=b["reps"], **common)
        if role == "stack":
            items = "\n".join(f"- {i['cid']} ({i.get('hyp')}: {i.get('title', '')}), {i['effect'] * 100:+.2f}% "
                              f"{_ci(i['ci'])} over {i['base']}. Its change: `{tool} diff {i['base']} {i['cid']}`"
                              for i in lane["items"])
            return _template("stack.md", items=items, try_reps=self.cfg["try_reps"],
                             fails_left=self.cfg["submit_failures"] - lane["fails"], copies=b["copies"],
                             reps=b["reps"], **common)
        if role == "audit":
            parent = next((c.get("parent") for c in self.ledger.records("candidates") if c["id"] == lane["base"]),
                          None)
            return _template("audit.md", cid=lane["base"], parent=parent or "-", **common)
        return f"unknown role {role}"

    # ---- any lane ------------------------------------------------------------------------------------------

    def note(self, lid, text):
        self.ledger.note("facts.md", f"{text.strip()} ({lid or 'agent'}, {now()[:16]})")
        self._publish_notes()
        return "noted in notes/facts.md (every agent reads it)", 0

    def post(self, lid, to, text):
        from kar.run.session import RunError
        with self.lock():
            st = self.state
            if to != "all" and to not in st["lanes"]:
                raise RunError(f"no lane {to}; lanes: {', '.join(st['lanes'])} or all")
            sent = self._post(st, to, lid or "human", text.strip())
            self.save(st)
        return f"delivered to {', '.join(sent) or 'nobody (no other lane is active)'} with their next command", 0

    def copy(self, lid, src, dst):
        """Copy a candidate (best, base, cNNNN) or any workspace file into the lane folder, to edit from there."""
        import shutil
        from kar.run.session import RunError
        st = self.state
        lane = self._lane(st, lid)
        cid = {"best": st["best"], "base": lane["base"]}.get(src, src)
        if isinstance(cid, str) and len(cid) == 5 and cid[0] == "c" and cid[1:].isdigit():
            sp = self.p("candidates", cid, "kernel.py")
        else:
            sp = os.path.realpath(src if os.path.isabs(src) else self.w(src))
            if not sp.startswith(os.path.realpath(self.w()) + os.sep):
                raise RunError(f"{src}: only files in the workspace can be copied")
        if not os.path.isfile(sp):
            raise RunError(f"{src}: no such file")
        mine = os.path.realpath(self.w("lanes", lid))
        dp = os.path.realpath(dst if os.path.isabs(dst) else self.w(dst))
        if os.path.isdir(dp):
            dp = os.path.join(dp, os.path.basename(sp))
        if not dp.startswith(mine + os.sep):
            raise RunError(f"{dst}: copies go into your lane folder, lanes/{lid}/")
        shutil.copy(sp, dp)
        os.chmod(dp, 0o644)
        return f"copied {src} to {os.path.relpath(dp, self.w())}", 0

    def check(self, lid, path):
        label, full = self._target(lid, path)
        rec = self.runner("check").evaluate(label, full)
        return self._check_text(rec), 0 if rec["verdict"] == "accept" else 1

    # ---- GPU commands that count as tries ------------------------------------------------------------------

    def try_(self, lid, path, reps=None):
        """One or more variants (a list keeps the GPU busy): each passes the gates, then all are timed against the
        current best in ONE job, so they are directly comparable. Counts as one GPU command, however many variants
        (measured: counted per variant, a builder who compared 11 variants had no command left to stress-test or
        test the one it handed in)."""
        paths = path if isinstance(path, (list, tuple)) else [path]
        if len(paths) > 8:
            from kar.run.session import RunError
            raise RunError("at most 8 variants per try")
        targets = [self._target(lid, x) for x in paths]
        st, lane = self._count_try(lid, "try")
        r = self.runner("try")
        good, text = [], []
        for label, full in targets:
            rec = r.evaluate(label, full)
            if rec["verdict"] != "accept":
                text.append(f"{label}: {self._check_text(rec)}")
            else:
                good.append((label, full))
        if not good:
            return "\n\n".join(text), 1
        best = (st["best"], self.p("candidates", st["best"], "kernel.py"))
        cands = list(good)
        if lane["base"] != st["best"]:
            cands.append((f"base_{lane['base']}", self.p("candidates", lane["base"], "kernel.py")))
        ab = r.ab(cands, best, reps=reps or self.cfg["try_reps"])
        text.append(checks.ab_text(ab, f"quick A/B against the current best {st['best']}"
                                       + (f", {len(good)} variants in one job" if len(good) > 1 else "")
                                       + " (guidance only; submit decides)"))
        if ab.get("status") == "ok":
            for label, full in good:
                res = ab["results"].get(label)
                if res is None:
                    continue
                self.ledger.append("tries", {"lane": lid, "hyp": lane["hyp"], "file": os.path.relpath(full, self.w()),
                                             "effect": res["effect"], "ci": res["ci"], "verdict": res["verdict"]})
                text.append(f"{label}:" + self._floor_text({k: v["a_us"] for k, v in res["shapes"].items()}, r)[1:])
        return "\n".join(text), 0

    def bound(self, lid, path, reps=None):
        """Time a probe that is incomplete on purpose (only the loads and stores, the math removed, ...) against the
        best, with its output check off. It answers "how fast could this design be"; it can never be submitted."""
        full = self._inside(lid, path)
        st, lane = self._count_try(lid, "bound")
        r = self.runner("bound")
        label = "probe_" + os.path.splitext(os.path.basename(path))[0]
        from kar.evaluate import lint as L
        lt = L.lint(full, self.task.rules)
        if not lt["ok"]:
            return "REJECTED at lint: " + "; ".join(lt["problems"]), 1
        sm = r.job("smoke", label, path=full)
        if sm.get("status") != "ok":
            return f"REJECTED at smoke: {sm.get('reason') or sm.get('status')}\n{sm.get('traceback', '')[-2000:]}", 1
        best = (st["best"], self.p("candidates", st["best"], "kernel.py"))
        ab = r.ab([(label, full)], best, reps=reps or self.cfg["try_reps"], unchecked=[label])
        text = checks.ab_text(ab, f"BOUND PROBE {label} against the best {st['best']}: output NOT checked, timing "
                                  f"only; this file can never be submitted")
        for v in ("KEEP", "SLOWER", "NO-GAIN", "MIXED"):
            text = text.replace(f" -> {v} ", " -> (probe, timing only) ")
        if ab.get("status") == "ok":
            res = next(iter(ab["results"].values()))
            self.ledger.append("bounds", {"lane": lid, "hyp": lane["hyp"], "file": os.path.relpath(full, self.w()),
                                          "effect": res["effect"], "ci": res["ci"],
                                          "us": {k: v["a_us"] for k, v in res["shapes"].items()}})
            text += "\n" + self._floor_text({k: v["a_us"] for k, v in res["shapes"].items()}, r)
        return text, 0

    def profile(self, lid, path):
        label, full = self._target(lid, path)
        self._count_try(lid, "profile")
        res = checks.profile_page(self.runner("profile"), self.task, full, ncu_n=self.cfg["ncu_shapes"], label=label)
        if res.get("rejected"):
            return self._check_text(res["check"]), 1
        out = self.w("lanes", lid, f"profile_{label}.md")
        with open(out, "w") as f:
            f.write(res["page"])
        return res["page"] + f"\n\n(written to lanes/{lid}/profile_{label}.md)", 0

    def ncu(self, lid, path, shape=None, section_set="full"):
        from kar.run.session import RunError
        label, full = self._target(lid, path)
        names = [s["name"] for s in self.task.shape_list("scored")]
        shape = shape or names[0]
        if shape not in names:
            raise RunError(f"unknown scored shape {shape}; scored shapes: {', '.join(names)}")
        self._count_try(lid, "ncu")
        res = self.runner("ncu").ncu_full(label, full, shape, section_set=section_set)
        if res.get("status") != "ok":
            return f"ncu failed: {res.get('reason')}\n{res.get('log_tail', '')}", 1
        out = self.w("lanes", lid, f"ncu_{label}_{shape}.txt")
        with open(out, "w") as f:
            f.write(res["text"])
        lines = res["text"].count("\n")
        return (f"ncu --set {section_set} on {shape}: {lines} lines written to lanes/{lid}/ncu_{label}_{shape}.txt "
                f"(every section, with ncu's own rule notes). Read it there; the first 60 lines:\n\n"
                + "\n".join(res["text"].splitlines()[:60])), 0

    def sass(self, lid, path):
        label, full = self._target(lid, path)
        self._count_try(lid, "sass")
        res = self.runner("sass").sass(label, full)
        if res.get("status") != "ok":
            return f"could not get the SASS: {res.get('reason')}\n{res.get('log_tail', '')}", 1
        out = self.w("lanes", lid, f"sass_{label}.txt")
        with open(out, "w") as f:
            f.write(res["text"])
        return f"{res['audit']}\nFull SASS written to lanes/{lid}/sass_{label}.txt ({len(res['text'])} bytes).", 0

    def stress(self, lid, path):
        label, full = self._target(lid, path)
        self._count_try(lid, "stress")
        r = self.runner("stress")
        rec = r.evaluate(label, full)
        if rec["verdict"] != "accept":
            return self._check_text(rec), 1
        sr = r.stress(label, full)
        self.ledger.append("stress", {"lane": lid, "target": label, "verdict": sr["verdict"],
                                      "reason": sr.get("reason")})
        if sr["verdict"] == "accept":
            return (f"stress suite passed on {label}: extra seeds, concurrent calls on several streams and "
                    f"back-to-back calls are all correct on every shape ({len(sr.get('shapes', []))} shapes)."), 0
        return f"STRESS FAILED on {label}: {sr.get('reason')}", 1

    # ---- the team's own tests ------------------------------------------------------------------------------

    def _registered(self):
        return [t for t in self.ledger.records("tests") if t.get("event") == "registered"]

    def _cases_text(self, cases, limit=40):
        lines = [f"  {'ok  ' if c['ok'] else 'FAIL'} {c['test']}: {c['case']}" + (f": {c['detail']}" if c['detail']
                                                                                   else "") for c in cases[:limit]]
        if len(cases) > limit:
            lines.append(f"  ... and {len(cases) - limit} more")
        return "\n".join(lines)

    def test(self, lid, test_path, target=None):
        """Run one of the team's own tests (a file defining test(...); see help/test_example.py) on a kernel."""
        from kar.evaluate import lint as L
        tfull = self._inside(lid, test_path)
        lt = L.lint(tfull, self.task.rules)
        if not lt["ok"]:
            return "test file REJECTED at lint (tests follow the kernel rules): " + "; ".join(lt["problems"]), 1
        if target is None:
            cand = self.w("lanes", lid, "cand.py")
            target = cand if os.path.exists(cand) else "best"
        label, full = self._target(lid, target)
        self._count_try(lid, "test")
        r = self.runner("test")
        res = r.tests(label, full, [tfull])
        if res.get("status") not in ("accept", "reject"):
            return f"the test job failed: {res.get('status')}: {res.get('reason')}\n{res.get('log_tail', '')}", 1
        self.ledger.append("test_runs", {"lane": lid, "test": os.path.relpath(tfull, self.w()), "target": label,
                                         "verdict": res["verdict"], "failed": sum(not c["ok"] for c in res["cases"])})
        head = (f"{label}: all {len(res['cases'])} cases pass" if res["verdict"] == "accept" else
                f"{label}: FAILS: {res.get('reason')}")
        return head + "\n" + self._cases_text(res["cases"]), 0 if res["verdict"] == "accept" else 1

    def register(self, lid, test_path, title):
        """Make a test part of the run: it must pass on the task's reference; then every later submit must pass it."""
        from kar.evaluate import canaries
        from kar.evaluate import lint as L
        tfull = self._inside(lid, test_path)
        lt = L.lint(tfull, self.task.rules)
        if not lt["ok"]:
            return "test file REJECTED at lint (tests follow the kernel rules): " + "; ".join(lt["problems"]), 1
        st, lane = self._count_try(lid, "register")
        import shutil
        with self.lock():                      # every attempt gets its own frozen file; ids go to registered tests
            os.makedirs(self.p("tests"), exist_ok=True)
            k = len([f for f in os.listdir(self.p("tests")) if f.endswith(".py")]) + 1
            frozen = self.p("tests", f"s{k:03d}_{lid}.py")
            shutil.copy(tfull, frozen)
            os.chmod(frozen, 0o444)
        r = self.runner(f"register_s{k:03d}")
        ref = canaries.reference_candidate(self.task, os.path.join(r.out, "ref"))
        on_ref = r.tests("reference", ref, [frozen])
        if on_ref["verdict"] != "accept":
            self.ledger.append("tests", {"event": "refused", "lane": lid, "title": title.strip(),
                                         "file": os.path.relpath(frozen, self.root), "reason": on_ref.get("reason")})
            return (f"not registered: the task's own reference fails it, so it tests something the task does not "
                    f"require.\n" + self._cases_text(on_ref["cases"])), 1
        on_best = r.tests(st["best"], self.p("candidates", st["best"], "kernel.py"), [frozen])
        with self.lock():                      # the id and its record together: two lanes may register at once
            tid = f"t{len(self._registered()) + 1:03d}"
            named = self.p("tests", f"{tid}.py")      # failures name the test by its id
            shutil.copy(frozen, named)
            os.chmod(named, 0o444)
            rec = {"event": "registered", "id": tid, "lane": lid, "title": title.strip(),
                   "file": os.path.relpath(named, self.root), "sha": self._sha_of(named),
                   "cases": len(on_ref["cases"]), "best_at_registration": st["best"],
                   "best_passes": on_best["verdict"] == "accept"}
            self.ledger.append("tests", rec)
            self._copy_ro(frozen, self.w("tests", f"{tid}.py"))
            verdict = "passes it" if rec["best_passes"] else f"FAILS it: {on_best.get('reason')}"
            st = self.state
            self._post(st, "all", lid, f"registered test {tid} ({title.strip()[:100]}; tests/{tid}.py): every submit "
                                       f"must pass it from now on. The current best {rec['best_at_registration']} "
                                       f"{verdict[:200]}")
            self.save(st)
        return (f"registered as {tid} ({len(on_ref['cases'])} cases; the reference passes all of them). The current "
                f"best {st['best']} {verdict}.\n" + self._cases_text(on_best["cases"])), 0

    # ---- survey --------------------------------------------------------------------------------------------

    def findings(self, lid, path):
        full = self._inside(lid, path)
        with open(full) as f:
            text = f.read()
        if len(text.strip()) < 200:
            return "findings are too short: give the measured numbers, where they come from, and what they imply", 1
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            self._require(lane, ("survey",), "findings")
            rel = os.path.join("survey", f"r{lane['cycle']:02d}", f"{lid}_{lane['lens']}.md")
            os.makedirs(self.p("evidence", os.path.dirname(rel)), exist_ok=True)
            with open(self.p("evidence", rel), "w") as f:
                f.write(text)
            self._copy_ro(self.p("evidence", rel), self.w(rel))
            sv = st.get("survey")
            if sv and sv["cycle"] == lane["cycle"]:
                sv["files"].append(rel)
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")[:200]
            self._close_lane(st, lane, f"findings ({lane['lens']}) in {rel}: {first}")
            self.save(st)
        return f"findings recorded in {rel}; the planner reads them. Your lane is done: end the session.", 0

    # ---- plan ----------------------------------------------------------------------------------------------

    def plan(self, lid, path):
        full = self._inside(lid, path)
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            self._require(lane, ("plan",), "plan")
            try:
                with open(full) as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                return f"cannot read the plan: {e}", 1
            hs = doc.get("hypotheses") if isinstance(doc, dict) else None
            cap = self.max_hyps()
            if not isinstance(hs, list) or not 1 <= len(hs) <= cap:
                return f"the plan needs 'hypotheses': a list of 1 to {cap} entries", 1
            problems = []
            for i, h in enumerate(hs):
                miss = [k for k in self.FIELDS if not isinstance(h, dict) or k not in h or h[k] in ("", None)]
                if miss:
                    problems.append(f"hypothesis {i + 1} lacks {', '.join(miss)}")
                    continue
                try:
                    g = float(h["predicted_gain_pct"])
                except (TypeError, ValueError):
                    problems.append(f"hypothesis {i + 1}: predicted_gain_pct must be a number")
                    continue
                if not 0 < g < 10000:
                    problems.append(f"hypothesis {i + 1}: predicted_gain_pct must be above 0")
            if problems:
                return "plan rejected:\n  " + "\n  ".join(problems), 1
            aim = self.aim(st)
            need = math.log(aim / st["best_sol"]) if st["best_sol"] < aim else 0
            live = [x for x in st["hyps"] if x["status"] == "building"]
            reach = sum(math.log1p(float(h["predicted_gain_pct"]) / 100) for h in hs) + \
                sum(math.log1p(x["predicted_gain_pct"] / 100) for x in live)
            if reach < need and not any(h.get("structural") for h in hs + live):
                return (f"plan rejected: together with the lanes still building, these predict "
                        f"{math.expm1(reach) * 100:.1f}% and this round needs {math.expm1(need) * 100:.1f}%. Add a "
                        f"structural change (a rewrite, fusion or new decomposition) whose bound can close the gap, "
                        f"and mark it \"structural\": true."), 1
            new = []
            for h in hs:
                hid = f"h{st['next_hyp']:04d}"
                st["next_hyp"] += 1
                rec = {"id": hid, "cycle": st["cycle"], "status": "open", "parent": st["best"], "kind": "idea",
                       **{k: h[k] for k in self.FIELDS}, "predicted_gain_pct": float(h["predicted_gain_pct"]),
                       "structural": bool(h.get("structural")), "lane": None, "planned_by": lid}
                new.append(rec)
                self.ledger.append("hypotheses", rec)
            new.sort(key=lambda h: -h["predicted_gain_pct"])
            for x in st["hyps"]:
                if x["status"] == "open":          # an older queue gives way to the new plan
                    x["status"] = "dropped"
                    self.ledger.append("hypotheses", {"id": x["id"], "status": "dropped", "why": f"replaced by "
                                                      f"round {st['cycle']}'s plan"})
            st["hyps"] += new
            st.update(planned=True, plateau_note="", plan_basis={
                "best": st["best"], "closed": sum(1 for x in st["hyps"] if x["status"] not in ("open", "building"))})
            self._close_lane(st, lane, f"round {st['cycle']} plan: " + "; ".join(
                f"{h['id']} {h['predicted_gain_pct']:.1f}% {h['title'][:60]}" for h in new))
            self.save(st)
        order = "\n".join(f"  {h['id']}  {h['predicted_gain_pct']:6.1f}%  {h['title']}" for h in new)
        return (f"plan accepted; build lanes take these in order of predicted gain, {self.lanes_n()} at a time:\n"
                f"{order}\n\nYour lane is done: end the session."), 0

    # ---- audit ---------------------------------------------------------------------------------------------

    def audit(self, lid, verdict, text):
        from kar.run.session import RunError
        if verdict not in ("ok", "hazard"):
            raise RunError("audit ok \"what you checked\"  or  audit hazard \"what, where, why it can fail, the fix\"")
        if len(text.strip()) < 40:
            return "say what you checked (or what the hazard is, where, and how to fix it), in a few sentences", 1
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            self._require(lane, ("audit",), "audit")
            cid = lane["base"]
            self.ledger.append("audits", {"lane": lid, "cid": cid, "verdict": verdict, "text": text.strip()})
            st["audited"].append(cid)
            msg = f"audit of {cid} recorded: {verdict}."
            if verdict == "hazard":
                hid = f"h{st['next_hyp']:04d}"
                st["next_hyp"] += 1
                proving = [t["id"] for t in self._registered() if t.get("lane") == lid and not t.get("best_passes")]
                h = {"id": hid, "cycle": st["cycle"], "status": "open", "parent": cid, "kind": "fix", "tests": proving,
                     "title": f"Fix the hazard in {cid} found by audit {lid}", "target": "correctness on every shape",
                     "mechanism": text.strip(), "evidence": f"audit {lid} of {cid}", "predicted_gain_pct": 0.01,
                     "arithmetic": ("a fix proven by registered tests: kept when it passes them, at any speed" if proving
                                    else "a fix: kept if it is not slower"),
                     "bound_test": "the stress suite, the registered tests and the A/B",
                     "structural": False, "lane": None, "planned_by": lid}
                st["hyps"].append(h)
                self.ledger.append("hypotheses", h)
                self._post(st, "all", lid, f"HAZARD in {cid}: {text.strip()[:300]} (fix queued as {hid})")
                msg += f" A fix is queued as {hid}, ahead of everything else; the run does not stop on the target " \
                       f"until it is done." + (f" It is done as soon as a kept kernel passes {', '.join(proving)} " \
                                               f"(every kept kernel must)." if proving else "")
            self._close_lane(st, lane, f"audit {verdict}: {text.strip()[:200]}")
            self.save(st)
        return msg + " Your lane is done: end the session.", 0

    # ---- hand-back: submit / abandon -----------------------------------------------------------------------

    def submit(self, lid, path, report=None):
        full = self._inside(lid, path)
        rep = self._inside(lid, report) if report else None
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            self._require(lane, ("build", "stack"), "submit")
            if self._maybe_done(st):
                self.save(st)
                return f"STOP: {st['done_reason']}", 1
            hid = lane["hyp"]
            h = self._hyp(st, hid) if hid else None
            fix = bool(h and h.get("kind") == "fix")
            title = h["title"] if h else "stack: " + ", ".join(i["cid"] for i in lane["items"] or [])
            best0, base = st["best"], lane["base"]
            cid = self._freeze_candidate(st, full, hyp=hid, report=rep, lane=lid)
            st["submits"] += 1
            self.save(st)
        frozen = self.p("candidates", cid, "kernel.py")
        rec_base = {"id": cid, "hyp": hid, "lane": lid, "role": lane["role"], "title": title, "parent": best0,
                    "base": base, "sha": self._sha_of(frozen), "file": os.path.relpath(full, self.w())}
        self.internal = True
        try:
            r = self.runner(f"submit_{cid}")
            rec = r.evaluate(cid, frozen)
            stage_fail = None
            if rec["verdict"] != "accept":
                stage_fail = (rec["stage"], rec["reason"], self._check_text(rec))
            else:
                sr = r.stress(cid, frozen)
                if sr["verdict"] != "accept":
                    stage_fail = ("stress", sr.get("reason") or "failed", f"REJECTED at stress: {sr.get('reason')}")
                else:
                    reg = [self.p(t["file"]) for t in self._registered()]
                    if reg:
                        tr = r.tests(cid, frozen, reg)
                        if tr["verdict"] != "accept":
                            stage_fail = ("tests", tr.get("reason") or "failed",
                                          f"REJECTED by the team's registered tests: {tr.get('reason')}")
            if stage_fail:
                return self._gate_failed(lid, cid, rec_base, *stage_fail)
            ab = r.ab([(cid, frozen)], (best0, self.p("candidates", best0, "kernel.py")))
            if ab.get("status") == "ok" and ab["results"][cid]["verdict"] == "NOISY":
                ab = r.ab([(cid, frozen)], (best0, self.p("candidates", best0, "kernel.py")))
            if ab.get("status") != "ok":
                self.ledger.append("candidates", {**rec_base, "decision": "ERROR", "reason": str(ab.get("reason"))})
                return f"{cid}: the A/B job failed ({ab.get('status')}: {ab.get('reason')}); submit again.", 1
            res = ab["results"][cid]
            text = [checks.ab_text(ab, f"{cid} against the current best {best0} (the decision)")]
            # a fix proven by registered tests (the best fails them, this kernel just passed them): correct comes
            # before fast, so it replaces the best at any speed
            proven = fix and bool(h.get("tests"))
            win = res["verdict"] == "KEEP" or (fix and res["verdict"] in ("NO-GAIN", "NOISY")) or proven
            if proven and res["effect"] < 0:
                text.append(f"\nThis fix is slower than {best0} ({res['effect'] * 100:+.2f}%), and it is kept: "
                            f"{best0} fails {', '.join(h['tests'])}, which the reference passes.")
            on_base = None
            if not win and base != best0 and not fix:
                ab2 = r.ab([(cid, frozen)], (base, self.p("candidates", base, "kernel.py")))
                if ab2.get("status") == "ok" and ab2["results"][cid]["verdict"] == "KEEP":
                    on_base = ab2["results"][cid]
                    text.append(checks.ab_text(ab2, f"{cid} against its base {base}"))
        finally:
            self.internal = False
        return self._decide(lid, cid, rec_base, res, ab, win, on_base, best0, base, fix, text)

    @staticmethod
    def _sha_of(path):
        from kar.run.session import _sha
        return _sha(path)

    def _gate_failed(self, lid, cid, rec_base, stage, reason, text):
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            lane["fails"] += 1
            self.ledger.append("candidates", {**rec_base, "decision": "REJECTED", "stage": stage,
                                              "reason": reason[:2000]})
            left = self.cfg["submit_failures"] - lane["fails"]
            msg = f"{cid} {text}"
            if left <= 0:
                self._finish(st, lane, "dead", f"{lane['fails']} submits failed the gates; last: {reason[:300]}")
                msg += f"\n\nLane {lid} is closed after {lane['fails']} failed submits."
            else:
                msg += f"\n\nFix it and submit again ({left} failed submits left in this lane)."
            self.save(st)
        return msg + f"\n\nNext: {self.tool(lid)} next", 1

    def _decide(self, lid, cid, rec_base, res, ab, win, on_base, best0, base, fix, text):
        shapes = {k: {"effect": v["effect"], "ci": v["ci"], "us": v["a_us"], "parent_us": v["b_us"]}
                  for k, v in res["shapes"].items()}
        crec = {**rec_base, "why": res["why"], "effect": res["effect"], "ci": res["ci"], "shapes": shapes,
                "noise": (ab.get("noise") or {}).get("ci"), "ab": os.path.relpath(ab["job_dir"], self.root),
                "vs": best0}
        new_best = False
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            if win and st["best"] == best0:
                r = self.runner(f"sol_{cid}")
                rows = self._sol_rows({k: v["a_us"] for k, v in res["shapes"].items()}, r)
                prev_score = st["best_score"]
                st.update(best=cid, best_sol=self._sol(rows), best_score=prev_score * (1 + res["effect"]),
                          since_keep=0)
                st["best_history"].append({"id": cid, "sol": st["best_sol"], "score": st["best_score"], "t": now(),
                                           "estimate": True})
                self.ledger.append("candidates", {**crec, "decision": "KEEP" + (" (fix)" if fix else ""),
                                                  "sol_estimate": st["best_sol"]})
                self._finish(st, lane, "kept", f"{res['effect'] * 100:+.2f}% {_ci(res['ci'])} over {best0} -> {cid}",
                             cand=cid, result=res["effect"])
                self._post(st, "all", "harness", f"new best {cid} ({lid}: {rec_base['title'][:80]}), "
                           f"{res['effect'] * 100:+.2f}% over {best0}. Read candidates/{cid}/kernel.py; its profile "
                           f"comes to evidence/{cid}/profile.md shortly.")
                text.append(f"\nKEPT: {cid} is the new best (about {st['best_sol'] * 100:.1f}% of speed of light; the "
                            f"profile that confirms it is being measured now: evidence/{cid}/profile.md).")
                new_best = True
                if fix and res["effect"] < 0:        # the faster kernel stays on record, with what it gets wrong
                    hh = self._hyp(st, rec_base["hyp"])
                    st.setdefault("faster_but_wrong", []).append(
                        {"cid": best0, "replaced_by": cid, "faster_pct": round(-res["effect"] * 100, 2),
                         "fails": hh.get("tests") or [], "hazard": (hh.get("mechanism") or "")[:400]})
                self._fixes_met(st, cid, rec_base["hyp"])
            elif win or on_base:
                eff = on_base or res
                vs = base if on_base else best0
                item = {"cid": cid, "base": vs, "hyp": rec_base["hyp"], "title": rec_base["title"],
                        "effect": eff["effect"], "ci": eff["ci"], "lane": lid,
                        "rounds": 1 + max((i.get("rounds", 1) for i in (lane.get("items") or [])), default=0)}
                st["since_keep"] = 0
                self.ledger.append("candidates", {**crec, "decision": "KEEP-ON-BASE", "effect": eff["effect"],
                                                  "ci": eff["ci"], "vs": vs})
                if item["rounds"] > 3:
                    self._finish(st, lane, "dead", f"won on {vs} but did not stack after two tries")
                    text.append(f"\n{cid} wins on {vs} but has failed to stack twice; recorded as a dead end.")
                else:
                    st["stack"].append(item)
                    self._finish(st, lane, "kept_on_base", f"{eff['effect'] * 100:+.2f}% {_ci(eff['ci'])} over {vs}; "
                                 f"the best moved on, so a stacker ports it onto {st['best']}", cand=cid,
                                 result=eff["effect"])
                    text.append(f"\nWINS ON ITS BASE ({vs}) but the best is now {st['best']}: queued for a stacker, "
                                f"who ports it onto the current best. Counted as a win.")
            else:
                self.ledger.append("candidates", {**crec, "decision": res["verdict"]})
                st["since_keep"] += 1
                worst = sorted(shapes.items(), key=lambda kv: kv[1]["effect"])
                nums = ", ".join(f"{k} {v['effect'] * 100:+.1f}%" for k, v in worst)
                self._finish(st, lane, "dead", f"{res['verdict']} {res['effect'] * 100:+.2f}% {_ci(res['ci'])} "
                                               f"({cid} vs {best0}; per shape: {nums}) {res['why']}",
                             cand=cid, result=res["effect"])
                text.append(f"\nNOT KEPT ({res['verdict']}). Recorded as a dead end with its numbers.")
            self._plateau(st)
            self.save(st)
        if new_best:
            text += self._after_keep(cid)
        return "\n".join(text) + f"\n\nYour lane is done: end the session ({self.tool(lid)} next shows the outcome).", 0

    def _fixes_met(self, st, cid, own):
        """A fix whose hazard is proven by registered tests is done once a kept kernel passes them: the judge ran
        every registered test on `cid` before keeping it."""
        reg = {t["id"] for t in self._registered()}
        for h in st["hyps"]:
            if h.get("kind") != "fix" or h["id"] == own or h["status"] not in ("open", "building"):
                continue
            if h.get("tests") and set(h["tests"]) <= reg:
                why = f"fixed in {cid}, which passes {', '.join(h['tests'])}"
                lane = st["lanes"].get(h.get("lane") or "")
                h.update(status="fixed", why=why, cand=cid)
                self.ledger.append("hypotheses", {"id": h["id"], "status": "fixed", "why": why})
                if lane and lane["status"] == "active":
                    self._close_lane(st, lane, f"fixed elsewhere: {why}")
                    st["inbox"].setdefault(lane["id"], []).append(
                        {"from": "harness", "t": now(), "text": f"your fix is no longer needed: {why}. End the session."})

    def _sol(self, rows):
        from kar.run.session import sol_of
        return sol_of(rows, self.score_fn())

    def _after_keep(self, cid):
        """Profile the new best (outside the lock), then milestone and the target check."""
        out = []
        self.internal = True
        try:
            prof = self._profile_best(cid)
        except Exception as e:  # noqa: BLE001 - a failed profile must not lose the decision
            self.internal = False
            return [f"(profiling {cid} failed: {e})"]
        conf = None
        if self.met(prof["sol"]):
            conf = self._confirm(cid)
        self.internal = False
        with self.lock():
            st = self.state
            for bh in st["best_history"]:
                if bh["id"] == cid:
                    bh.update(sol=prof["sol"], score=prof["score"], estimate=False)
            if st["best"] == cid:
                st.update(best_sol=prof["sol"], best_score=prof["score"])
            out.append(f"Profile of {cid}: {prof['sol'] * 100:.1f}% of speed of light, {prof['score']:.3f}x the "
                       f"baseline (evidence/{cid}/profile.md).")
            h = next((x for x in st["hyps"] if x.get("cand") == cid), {"title": "stack", "id": "-",
                                                                         "structural": False})
            ms = self._maybe_milestone(st, h, cid, prof)
            if ms:
                out.append(f"Milestone written: {ms}")
            mode = self.cfg["audit"]
            if st["best"] == cid and (mode == "every" or (mode == "milestones" and (ms or conf is not None))):
                st["to_audit"].append(cid)
            if conf is not None:
                out.append(f"At or above the target; a confirming measurement ({conf['reps']} reps) reads "
                           f"{conf['sol'] * 100:.1f}%.")
                if self.met(conf["sol"]) and st["best"] == cid:
                    st["target_met"] = cid
                    st["target_reason"] = (f"target reached ({prof['sol'] * 100:.1f}% of speed of light, confirmed at "
                                           f"{conf['sol'] * 100:.1f}%)")
                    if self._fixes_pending(st) or (self.cfg["audit"] != "off" and cid not in st["audited"]):
                        out.append("The run stops once the audit of this best (and any fix it asks for) is done.")
                    self._maybe_done(st)
                elif not self.met(conf["sol"]):
                    out.append("Not confirmed, so the run goes on.")
            self.save(st)
        return out

    def _finish(self, st, lane, status, why, cand=None, result=None):
        """Close a build or stack lane with its hypothesis (or its stack items)."""
        if lane["role"] == "build" and lane["hyp"]:
            h = self._hyp(st, lane["hyp"])
            h.update(status=status, why=why)
            if cand:
                h.update(cand=cand, result=result)
            self.ledger.append("hypotheses", {"id": h["id"], "status": status, "why": why})
            if status not in ("kept", "kept_on_base"):
                self.ledger.note("dead_ends.md", f"{h['id']} {h['title']} [{status}]: {why}")
        elif lane["role"] == "stack":
            for it in lane["items"] or []:
                if status == "kept":
                    self.ledger.note("facts.md", f"{it['cid']} ({it.get('title', '')}) stacked into {cand}: {why}")
                elif status != "kept_on_base":
                    self.ledger.note("dead_ends.md", f"{it['cid']} ({it.get('title', '')}) [did not stack]: {why}")
        self._publish_notes()
        self._close_lane(st, lane, f"{status}: {why}")

    def abandon(self, lid, why):
        if not why or len(why.strip()) < 15:
            return f"say why, with the numbers: {self.tool(lid)} abandon \"bound test: ... measured ...\"", 1
        with self.lock():
            st = self.state
            lane = self._lane(st, lid)
            self._require(lane, ("build", "stack"), "abandon")
            h = self._hyp(st, lane["hyp"]) if lane.get("hyp") else None
            self._finish(st, lane, "abandoned", why.strip())
            if h and h.get("kind") == "fix":          # the hazard is still in the best: it stays on the books
                h["attempts"] = h.get("attempts", 0) + 1
                h.setdefault("notes", []).append(f"{lid}: {why.strip()[:600]}")
                if h["attempts"] < 3:
                    h.update(status="open", lane=None)
                    self.ledger.append("hypotheses", {"id": h["id"], "status": "open", "why": f"{lid} gave up on the "
                                                      f"fix; the hazard is still in the best, so it is queued again"})
                else:
                    h.update(status="unfixed")
                    self.ledger.append("hypotheses", {"id": h["id"], "status": "unfixed", "why": "three lanes gave "
                                                      "up on the fix; the hazard stays listed with the result"})
            st["since_keep"] += 1
            self._plateau(st)
            self.save(st)
        return f"lane {lid} abandoned and recorded.\n\nYour lane is done: end the session.", 0

    def _plateau(self, st):
        if st["phase"] == "done" or st["since_keep"] < self.plateau_n():
            return
        st["plateaus"] += 1
        st["since_keep"] = 0
        if st["plateaus"] >= 2:
            st.update(phase="done", done_reason=f"plateau: {self.plateau_n()} hypotheses in a row did not pay, twice")
            return
        for x in st["hyps"]:
            if x["status"] == "open":
                x["status"] = "dropped"
                self.ledger.append("hypotheses", {"id": x["id"], "status": "dropped", "why": "plateau"})
        st.update(cycle=st["cycle"] + 1, planned=False,
                  plateau_note=(f"**Plateau.** The last {self.plateau_n()} hypotheses did not pay, so the queue was "
                                f"dropped and the team surveys the best again. Start from the profile and the new "
                                f"findings, not from the old queue: look for the biggest remaining gap, and propose "
                                f"only ideas that are not in the dead ends, each with a bound test. Another plateau "
                                f"ends the run.\n"))

    def _confirm(self, cid):
        """A second, longer measurement before stopping on the target: a single reading near the line is a draw."""
        from kar.run.session import sol_of
        reps = 2 * self.task.bench["reps"]
        g = self.runner(f"confirm_{cid}").gap(self.p("candidates", cid, "kernel.py"), reps=reps, label=cid)
        out = {"sol": sol_of(g["rows"], self.score_fn()), "reps": reps}
        self.ledger.append("confirmations", {"id": cid, **out})
        return out

    def _maybe_milestone(self, st, h, cid, prof):
        base = st.get("milestone_base")
        base_score = next((b["score"] for b in st.get("best_history", []) if b["id"] == base), 1.0)
        gain = prof["score"] / base_score - 1
        if gain * 100 < self.cfg["milestone_pct"] and not h.get("structural"):
            return None
        st["milestones"] += 1
        path = records.milestone(self, st["milestones"], h, base_id=base, cid=cid, prof=prof, gain=gain)
        st["milestone_base"] = cid
        return path

    # ---- the driver's view ---------------------------------------------------------------------------------

    def progress_key(self, lid):
        """Changes whenever the lane takes a step: used to tell a working session from an idle one."""
        st = self.state
        lane = st["lanes"].get(lid) or {}
        return (lane.get("status"), lane.get("tries"), lane.get("fails"), len(self.ledger.rows("candidates")))
