"""Load a candidate file as a fresh module. Loading one file twice gives two independent copies (separately compiled
kernels, separate placement), which is how the bench measures placement noise."""
import importlib.util
import sys


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
