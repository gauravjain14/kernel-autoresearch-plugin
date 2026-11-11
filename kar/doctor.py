"""`kar doctor`: is this machine ready to run kar, and what to fix if not. Nothing here spends money or sends code
anywhere; `--backend` adds a check that the configured remote backend answers (one tiny job: nvidia-smi).

Each line is OK, WARN (works, with a limit) or FAIL (a run would fail), with the fix.
"""
import importlib
import os
import shutil
import subprocess
import sys


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def _tool(name):
    import glob
    return shutil.which(name) or next(iter(sorted(glob.glob(f"/usr/local/cuda*/bin/{name}"), reverse=True)), None)


def checks(backend=None):
    out = []

    def add(level, what, detail="", fix=""):
        out.append((level, what, detail, fix))

    v = sys.version_info
    add("OK" if v >= (3, 11) else "FAIL", "Python", f"{v.major}.{v.minor}.{v.micro}", "" if v >= (3, 11) else
        "Python 3.11 or newer")
    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda.is_available()
        add("OK" if cuda else "WARN", "PyTorch", f"{torch.__version__}, CUDA {torch.version.cuda}, GPU visible: {cuda}",
            "" if cuda else "no local GPU: run GPU jobs on a remote backend (--backend modal:H100 or ssh:HOST)")
    except ImportError:
        add("WARN", "PyTorch", "not installed here", "pip install torch (needed locally only for a local GPU; remote "
            "backends need it in their image)")
    try:
        importlib.import_module("cuda.bindings.nvrtc")
        add("OK", "cuda-python (NVRTC, driver API)", "", "")
    except ImportError:
        add("WARN", "cuda-python", "not installed", "pip install cuda-python (kernels compile CUDA C++ with NVRTC)")
    code, text = _run(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,memory.total",
                       "--format=csv,noheader"])
    if code == 0 and text:
        for ln in text.splitlines():
            add("OK", "GPU", ln.strip(), "")
    else:
        add("WARN", "GPU", "nvidia-smi found no GPU", "fine with a remote backend")
    for tool, why in (("nsys", "per-kernel timelines"), ("ncu", "hardware counters"),
                      ("cuobjdump", "SASS dumps and audits"), ("nvdisasm", "SASS of newer cubins")):
        p = _tool(tool)
        add("OK" if p else "WARN", tool, p or "not found", "" if p else f"install the CUDA toolkit's {tool} ({why}); "
            f"profiling pages lose that section without it")
    try:
        with open("/proc/driver/nvidia/params") as f:
            admin = "RmProfilingAdminOnly: 1" in f.read()
        add("WARN" if admin else "OK", "GPU performance counters for ncu",
            "admin only" if admin else "open to this user",
            "set NVreg_RestrictProfilingToAdminUsers=0 (see ncu's ERR_NVGPUCTRPERM page) or profile on a backend "
            "that allows it" if admin else "")
    except OSError:
        pass
    agents = {"claude": "Claude Code", "codex": "Codex CLI", "grok": "Grok Build", "cursor-agent": "Cursor CLI"}
    found = []
    for cmd, name in agents.items():
        p = shutil.which(cmd)
        if p:
            c, t = _run([cmd, "--version"])
            found.append(cmd)
            add("OK", f"agent: {name}", (t.splitlines() or [""])[0][:80], "")
    if not found:
        add("WARN", "agents", "no agent CLI found", "install one (claude, codex, grok, cursor-agent), or drive the run "
            "by hand: `kar serve RUN` and open any agent in RUN/ws")
    from kar import config
    cfg = config.load()
    b = config.backend()
    add("OK", "default backend", (b or {"kind": "local"}).get("kind") + (f" {b}" if b else " (this machine's GPU 0)"),
        "" if b else "set one in ~/.config/kar/config.toml (backend = \"modal:H200:3\") or pass --backend")
    if (b or {}).get("kind") == "modal" or (backend or {}).get("kind") == "modal":
        try:
            importlib.import_module("modal")
            tok = os.path.exists(os.path.expanduser("~/.modal.toml")) or os.environ.get("MODAL_TOKEN_ID")
            add("OK" if tok else "FAIL", "Modal", "client installed" + (", token present" if tok else ", no token"),
                "" if tok else "modal token new")
        except ImportError:
            add("FAIL", "Modal", "the modal package is not installed", "pip install modal && modal token new")
    if backend:
        try:
            from kar import backends
            inf = backends.make(backend).info()
            add("OK", "backend answers", f"{inf}", "")
        except Exception as e:  # noqa: BLE001
            add("FAIL", "backend answers", f"{type(e).__name__}: {e}", "fix the backend config or its credentials")
    if cfg.get("docs_allow"):
        add("OK", "extra docs sites", ", ".join(cfg["docs_allow"]), "")
    return out


def main(backend=None):
    rows = checks(backend)
    w = max(len(r[1]) for r in rows)
    for level, what, detail, fix in rows:
        print(f"{level:4s}  {what:{w}s}  {detail}" + (f"\n      {'':{w}s}  fix: {fix}" if fix and level != "OK" else ""))
    bad = sum(1 for r in rows if r[0] == "FAIL")
    print(f"\n{'ready' if not bad else f'{bad} problem(s) to fix before a run'}")
    return 1 if bad else 0
