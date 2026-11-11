"""Profile parsers on synthetic inputs with the real schemas."""
import sqlite3

from kar.profile import ncu, nsys

MUL = ("void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::"
       "BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, "
       "const T1 &)::[lambda(int) (instance 1)]>(int")
COPY = ("void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::"
        "[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::"
        "BFloat16) (instance 1)], at::detail::Array<cha")


def test_op_hint():
    assert nsys.op_hint(MUL) == "MulFunctor(float)"
    assert nsys.op_hint(COPY) == "direct_copy_kernel_cuda(BFloat16)"
    assert nsys.op_hint("gemv") == ""


def make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    con.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, "
                "demangledName INTEGER, graphNodeId INTEGER)")
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "fill"), (2, "k1"), (3, "k2")])
    t = 0
    for r in range(5):
        con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (t, t + 100_000, f"kar:s0:{r}"))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1, 1, NULL)", (t + 1000, t + 2000))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 2, 2, 7)", (t + 10_000, t + 30_000))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 3, 3, 8)", (t + 32_000, t + 42_000))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 2, 2, NULL)", (t + 150_000, t + 160_000))
        t += 200_000
    con.commit()


def test_nsys_parse(tmp_path):
    db = str(tmp_path / "x.sqlite")
    make_db(db)
    p = nsys.parse(db, skip=1)["s0"]
    assert p["replays"] == 4
    assert abs(p["span_us"] - 32.0) < 1e-9 and abs(p["kernel_sum_us"] - 30.0) < 1e-9 and abs(p["idle_us"] - 2) < 1e-9
    assert [k["name"] for k in p["kernels"]] == ["k1", "k2"]
    assert p["outside_graph"]["fill"]["per_replay"] == 1


def test_ncu_summarize_drops_pipe_substalls():
    row = {"Kernel Name": ("k", ""), "gpu__time_duration.sum": ("2,500", "nsecond"),
           "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": ("3", ""),
           "smsp__average_warps_issue_stalled_long_scoreboard_pipe_l1tex_per_issue_active.ratio": ("3", ""),
           "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": ("1", "")}
    s = ncu.summarize([row])[0]
    assert s["duration_us"] == 2.5
    assert s["stalls"] == [("long_scoreboard", 75), ("wait", 25)]
