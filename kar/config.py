"""The user's own defaults, from ~/.config/kar/config.toml (or $KAR_CONFIG):

    backend = "modal:H200:3"                       # where GPU jobs run when a command names no --backend
    backend_opts = ["image=nvcr.io/nvidia/pytorch:25.01-py3"]
    agent = "claude"                               # default agent preset for kar run / kar new
    docs_allow = ["docs.example.org"]              # extra documentation sites the docs command may fetch

Nothing here is required; without the file everything runs on this machine's GPU 0.
"""
import os
import tomllib

PATH = os.path.expanduser(os.environ.get("KAR_CONFIG", "~/.config/kar/config.toml"))


def load():
    try:
        with open(PATH, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def backend():
    """The configured default backend config dict, or None (this machine's GPU)."""
    c = load()
    if not c.get("backend"):
        return None
    from kar import backends
    return backends.parse_spec(c["backend"], c.get("backend_opts", ()))
