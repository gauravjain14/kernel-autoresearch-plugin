"""GPU-side job entry point. Runs in its own process (python -E -P), one job per process:

  smoke  import the candidate, call it once on the smoke shape, synchronize
  check  the correctness suite (check.py) on every scored and nearby shape
  stress more seeds, concurrent calls on several streams, back-to-back calls (stress.py)
  tests  the team's own registered or draft tests (agent_test.py)
  bench  timing of several variants in one process (bench.py)

Before anything else it verifies the evaluator and task hashes against the job, then takes the tamper snapshot. Phase
markers go to `progress` so the parent can enforce a time limit per phase (a hang costs one phase, not the job).
The result is written atomically to result.json beside job.json.
"""
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _keep_images(out_dir):
    """Profiling modes only: wrap cuModuleLoadData so every image a candidate loads (NVRTC cubins) is saved."""
    from cuda.bindings import driver
    os.makedirs(out_dir, exist_ok=True)
    orig = driver.cuModuleLoadData

    def keep(image, *a, **k):
        try:
            n = len(os.listdir(out_dir))
            with open(os.path.join(out_dir, f"{n:03d}.cubin"), "wb") as f:
                f.write(bytes(image))
        except (OSError, TypeError):
            pass
        return orig(image, *a, **k)

    driver.cuModuleLoadData = keep


def main(job_path):
    jd = os.path.dirname(os.path.abspath(job_path))
    with open(job_path) as f:
        job = json.load(f)
    res = {"mode": job["mode"], "label": job.get("label"), "status": "error"}
    prog = open(os.path.join(jd, "progress"), "a", buffering=1)

    def phase(name):
        prog.write(f"{time.time():.3f}\t{name}\n")

    phase("start")
    try:
        from kar.evaluate.manifest import evaluator_sha
        if job.get("evaluator_sha") and evaluator_sha() != job["evaluator_sha"]:
            res.update(status="evaluator_changed", reason="evaluator files changed since the run was frozen")
            return 0
        import torch
        from kar.evaluate import bench, check, integrity, loader
        from kar.task import spec
        task = spec.load(job["task_root"])
        if job.get("task_sha") and task.sha() != job["task_sha"]:
            res.update(status="task_changed", reason="task files changed since the run was frozen")
            return 0
        R = SimpleNamespace(ref=task.attr(task.reference), exact=task.attr(task.exact) if task.exact else None,
                            make=task.attr(task.inputs))
        shapes = task.shape_list("scored") + task.shape_list("nearby")
        dev = torch.cuda.get_device_properties(0)
        res["device"] = {"name": dev.name, "cc": f"{dev.major}.{dev.minor}", "sms": dev.multi_processor_count}
        snap = integrity.snapshot()
        verify = integrity.verify

        def guard():
            changed = verify(snap)
            if changed:
                raise check.Fail("tamper: " + ", ".join(changed[:6]))

        def load(path, name):
            phase(f"import {name}")
            t0 = time.perf_counter()
            mod = loader.load_file(path, name)
            res.setdefault("import_s", {})[name] = round(time.perf_counter() - t0, 3)
            fn = getattr(mod, task.entry, None)
            if fn is None:
                raise check.Fail(f"{os.path.basename(path)} does not define {task.entry}")
            guard()
            return fn

        mode = job["mode"]
        try:
            if mode == "smoke":
                fn = load(job["path"], "kar_cand")
                name = task.smoke_shape or shapes[0]["name"]
                s = next(x for x in shapes if x["name"] == name)
                args = R.make(seed=job["seed"], **s["args"])
                phase("smoke call")
                t0 = time.perf_counter()
                fn(*args)
                torch.cuda.synchronize()
                res.update(status="ok", shape=name, call_s=round(time.perf_counter() - t0, 3))
            elif mode == "check":
                fn = load(job["path"], "kar_cand")
                res.update(check.run_check(fn, R, shapes, job["seed"], task.tolerance, guard, phase,
                                           graph=task.bench["mode"] == "graph"))
            elif mode == "tests":
                from kar.evaluate import agent_test
                fn = load(job["path"], "kar_cand")
                tests = []
                for i, tp in enumerate(job["tests"]):
                    tm = loader.load_file(tp, f"kar_test{i}")
                    tests.append((os.path.splitext(os.path.basename(tp))[0], getattr(tm, "test")))
                res.update(agent_test.run_tests(fn, tests, R, shapes, task.tolerance, guard, phase))
            elif mode == "taskcheck":
                from kar.task.check_task import check_task
                phase("shape taskcheck")
                res.update(status="ok", task_check=check_task(task))
            elif mode == "stress":
                from kar.evaluate import stress
                fn = load(job["path"], "kar_cand")
                res.update(stress.run_stress(fn, R, shapes, job["seed"], task.tolerance, guard, phase))
            elif mode == "bench":
                variants = []
                for v in job["variants"]:
                    for c in range(v["copies"]):
                        lb = f"{v['label']}#{c}"
                        variants.append((lb, load(v["path"], f"kar_v{len(variants)}")))
                scored = [s for s in shapes if s["kind"] == "scored"]
                b = task.bench
                out = bench.run_bench(variants, R, scored, job["reps"], job["seed"], task.tolerance,
                                      instances=job.get("instances", b["instances"]), flush_bytes=b["flush_bytes"],
                                      warmup=b["warmup"], spin_s=b["spin_s"], guard=guard, phase=phase,
                                      unchecked=tuple(job.get("unchecked", ())), mode=b["mode"])
                res.update(status="ok", bench=out)
            elif mode == "calibrate":
                from kar.sol import microbench
                phase("bench calibrate")
                res.update(status="ok", hw=microbench.calibrate())
            elif mode in ("nsys", "ncu"):
                # profiling only: keep a copy of every module image the candidate loads, for the SASS audit
                _keep_images(os.path.join(jd, "cubins"))
                fn = load(job["path"], "kar_cand")
                from kar.evaluate.check import capture
                from kar.evaluate.compare import clone_args
                scored = [s for s in shapes if s["kind"] == "scored"]
                if mode == "nsys":
                    from kar.profile import nsys
                    nsys.replay_ranges(fn, R, scored, job["reps"], job["seed"],
                                       capture if task.bench["mode"] == "graph" else None, clone_args, phase)
                else:
                    from kar.profile import ncu
                    s = next(x for x in scored if x["name"] == job["shape"])
                    phase(f"shape {s['name']}")
                    ncu.ncu_call(fn, R, s, job["seed"], clone_args)
                res.update(status="ok")
            else:
                raise ValueError(f"unknown mode {mode}")
            guard()
        except check.Fail as e:
            res.update(status="reject", reason=str(e))
        except Exception as e:  # noqa: BLE001 - a candidate raising is a rejection, not a harness error
            res.update(status="reject", reason=f"raised {type(e).__name__}: {e}", traceback=traceback.format_exc())
        changed = verify(snap)
        if changed:
            res.update(status="reject", reason="tamper: " + ", ".join(changed[:6]))
    except Exception:  # noqa: BLE001
        res.update(status="error", traceback=traceback.format_exc())
    finally:
        phase("done")
        _write(os.path.join(jd, "result.json"), res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
