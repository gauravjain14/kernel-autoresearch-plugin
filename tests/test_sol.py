"""Floors and the gap table against hand calculations."""
import math
import os

import pytest

from conftest import TOY
from kar.sol import gap
from kar.task import spec

HW = {"device": {"name": "test", "sms": 1, "l2_bytes": 32 << 20, "nominal_dram_GBps": 300},
      "measured": {"dram_GBps": 250.0, "fp32_TFLOPs": 20.0, "tc_bf16_TFLOPs": 50.0, "graph_empty_us": 5.0,
                   "l2_read_GBps": 1500.0, "l2_write_absorb_bytes": 16e6}}


def test_gemv_floor_by_hand():
    task = spec.load(os.path.join(TOY, "gemv"))
    s = [x for x in task.shape_list("scored") if x["name"] == "b1_n4096_k4096"]
    f = gap.floors(task, HW, s)["b1_n4096_k4096"]
    read = 2 * (4096 * 4096 + 4096)               # w and x; the 8 KB output fits in the absorbed 16 MB
    assert f["counted_bytes"] == read
    assert f["floor_us"] == pytest.approx(read / 250e3 + 5.0)
    assert f["bound"] == "memory"


def test_rmsnorm_writes_beyond_absorption_count():
    task = spec.load(os.path.join(TOY, "rmsnorm"))
    s = [x for x in task.shape_list("scored") if x["name"] == "t4096_h4096"]
    f = gap.floors(task, HW, s)["t4096_h4096"]
    rd, wr = 2 * 4096 * 4096 + 2 * 4096, 2 * 4096 * 4096
    assert f["counted_bytes"] == pytest.approx(rd + wr - 16e6)


def test_math_bound_and_table_ranking():
    class T:
        opmodel = {"bytes": "x", "flops": "y"}

        def attr(self, ref):
            return (lambda n: {"read": n, "write": 0}) if ref == "x" else (lambda n: {"tc_bf16": 1e9 * n})

    shapes = [{"name": "a", "args": {"n": 1000}}, {"name": "b", "args": {"n": 1}}]
    fl = gap.floors(T(), HW, shapes)
    assert fl["a"]["bound"] == "tc_bf16" and fl["a"]["floor_us"] == pytest.approx(1e12 / 50e6 + 5.0)
    rows, total = gap.table(fl, {"a": 2 * fl["a"]["floor_us"], "b": fl["b"]["floor_us"]}, {"a": 1, "b": 1})
    assert rows[0]["shape"] == "a"
    assert rows[0]["score_pct_at_floor"] == pytest.approx(100 * math.expm1(0.5 * math.log(2)))
    assert rows[1]["score_pct_at_floor"] == 0.0
    assert total == pytest.approx(rows[0]["score_pct_at_floor"])


def test_unknown_pipe():
    class T:
        opmodel = {"bytes": "x", "flops": "y"}

        def attr(self, ref):
            return (lambda: 1) if ref == "x" else (lambda: {"warp_magic": 1})

    with pytest.raises(gap.OpModelError):
        gap.floors(T(), HW, [{"name": "a", "args": {}}])
