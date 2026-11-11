"""Static rules check on a candidate file, before it ever reaches a GPU.

Readable source only: no binary blobs, no encoded or compressed payloads, no precompiled cubin/fatbin loading, no
forbidden imports (network, subprocess, dynamic import, file system), no file reads, size under the task's cap. NVRTC compiling readable CUDA
source at import is allowed.
"""
import ast
import re

BLOB_RX = re.compile(rb"[A-Za-z0-9+/=]{400,}")
HEX_RX = re.compile(rb"(?:0x[0-9a-fA-F]{2}\s*,\s*){256,}")
BYTES_ESC_RX = re.compile(rb"(?:\\x[0-9a-fA-F]{2}){128,}")
BINARY_MARKERS = [(re.compile(rb"\x7fELF"), "an ELF header"),
                  (re.compile(rb"cuModuleLoadFatBinary|cuLibraryLoad"), "precompiled-image loading"),
                  (re.compile(rb"cuModuleLoad\s*\("), "loading a module from a file"),
                  (re.compile(rb"\.(?:cubin|fatbin|ptx)[\'\"]"), "a .cubin/.fatbin/.ptx file name")]
DYNAMIC_CALLS = {"exec", "eval", "compile", "__import__"}
# A kernel is self-contained: it may not read files at run time (it could load an answer, e.g. a task's known-good
# kernel or a saved output). Calls by bare name, and module-level loaders by dotted name.
FILE_CALLS = {"open", "breakpoint", "input"}
FILE_ATTR_CALLS = {"torch.load", "torch.from_file", "numpy.load", "np.load", "numpy.fromfile", "np.fromfile",
                   "numpy.loadtxt", "np.loadtxt", "numpy.memmap", "np.memmap", "torch.utils.cpp_extension.load"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def lint(path, rules):
    problems = []
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > rules["max_source_bytes"]:
        problems.append(f"{len(data)} bytes, over the {rules['max_source_bytes']}-byte cap")
    if b"\x00" in data:
        problems.append("contains NUL bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "problems": problems + ["not UTF-8 text"]}
    for rx, what in BINARY_MARKERS:
        if rx.search(data):
            problems.append(f"contains {what}")
    if BLOB_RX.search(data):
        problems.append("contains a long base64-like run (an encoded payload?)")
    if HEX_RX.search(data) or BYTES_ESC_RX.search(data):
        problems.append("contains a long byte array (a binary payload?)")
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as e:
        return {"ok": False, "problems": problems + [f"syntax error: {e}"]}
    forbidden = set(rules["forbidden_imports"])
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.split(".")[0] in forbidden:
                problems.append(f"line {node.lineno}: imports {n}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_CALLS:
            problems.append(f"line {node.lineno}: calls {node.func.id}()")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FILE_CALLS:
            problems.append(f"line {node.lineno}: calls {node.func.id}() (kernels may not read files or stdin)")
        if isinstance(node, ast.Call) and _dotted(node.func) in FILE_ATTR_CALLS:
            problems.append(f"line {node.lineno}: calls {_dotted(node.func)}() (kernels may not read files)")
    for pat in rules.get("forbidden_patterns", []):
        m = re.search(pat, text)
        if m:
            line = text.count("\n", 0, m.start()) + 1
            problems.append(f"line {line}: matches the task's forbidden pattern {pat!r}")
    return {"ok": not problems, "problems": problems}
