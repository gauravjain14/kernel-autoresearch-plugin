"""`kar new`: the skeleton and its instructions, and validation refusing an incomplete task without touching a GPU."""
import os

from kar import newtask


def test_skeleton_layout(tmp_path):
    root = newtask.create(str(tmp_path / "t"), "Row-wise softmax over BF16 logits.")
    for f in ("DESCRIPTION.md", "AGENTS.md", "CLAUDE.md", "task.toml", "reference.py", "inputs.py", "shapes.py",
              "opmodel.py", "baseline.py", "kar-task"):
        assert os.path.exists(os.path.join(root, f)), f
    assert os.access(os.path.join(root, "kar-task"), os.X_OK)
    assert "softmax" in open(os.path.join(root, "DESCRIPTION.md")).read()


def test_incomplete_task_is_refused(tmp_path):
    root = newtask.create(str(tmp_path / "t"), "anything")
    os.remove(os.path.join(root, "shapes.py"))
    ok, text = newtask.validate(root)
    assert not ok and "shapes.py" in text
