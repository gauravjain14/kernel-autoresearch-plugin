import os

from conftest import TOY
from kar.evaluate import lint
from kar.task import spec

RULES = spec.DEFAULT_RULES


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_good_kernels_pass():
    for t in ("rmsnorm", "gemv"):
        r = lint.lint(os.path.join(TOY, t, "good_nvrtc.py"), RULES)
        assert r["ok"], r


def test_payloads_and_imports_fail(tmp_path):
    cases = {
        "b64.py": "X = '" + "QUJD" * 200 + "'\n",
        "hex.py": "X = [" + ", ".join(["0x7f"] * 300) + "]\n",
        "imp.py": "import subprocess\n",
        "imp2.py": "from urllib import request\n",
        "dyn.py": "eval('1')\n",
        "fat.py": "driver.cuModuleLoadFatBinary(x)\n",
        "file.py": "open('kernel.cubin')\n",
        "read.py": "x = open('answer.txt').read()\n",
        "tload.py": "import torch\nY = torch.load('y.pt')\n",
        "npload.py": "import numpy as np\nY = np.load('y.npy')\n",
        "os.py": "import os\n",
        "path.py": "from pathlib import Path\n",
    }
    for name, text in cases.items():
        r = lint.lint(write(tmp_path, name, text), RULES)
        assert not r["ok"], (name, r)


def test_size_cap(tmp_path):
    r = lint.lint(write(tmp_path, "big.py", "# x\n" * 10), {**RULES, "max_source_bytes": 10})
    assert not r["ok"]


def test_nvrtc_source_is_fine(tmp_path):
    text = "SRC = r'''extern \"C\" __global__ void k() {}'''\nimage = nvrtcGetCUBIN\nmod = cuModuleLoadData(image)\n"
    assert lint.lint(write(tmp_path, "ok.py", text), RULES)["ok"]


def test_triton_style_load_is_fine(tmp_path):
    r = lint.lint(write(tmp_path, "tl.py", "import triton.language as tl\ndef k(p):\n    return tl.load(p)\n"), RULES)
    assert r["ok"], r
