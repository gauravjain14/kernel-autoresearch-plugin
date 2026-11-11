import os

import pytest

from conftest import TOY
from kar.task import spec


def test_toy_tasks_load():
    for t in ("rmsnorm", "gemv"):
        task = spec.load(os.path.join(TOY, t))
        assert task.shape_list("scored") and task.shape_list("nearby")
        assert task.sha() == spec.load(os.path.join(TOY, t)).sha()


def test_missing_key(tmp_path):
    (tmp_path / "task.toml").write_text('name = "x"\n')
    with pytest.raises(spec.TaskError):
        spec.load(str(tmp_path))


def test_stdlib_shadow(tmp_path):
    (tmp_path / "task.toml").write_text(
        'name="x"\nentry="kernel_fn"\nreference="random.py:f"\nbaseline="random.py"\n'
        'inputs="random.py:f"\nshapes="random.py:S"\n')
    (tmp_path / "random.py").write_text("def f(): pass\nS=[]\n")
    with pytest.raises(spec.TaskError, match="shadows"):
        spec.load(str(tmp_path))
