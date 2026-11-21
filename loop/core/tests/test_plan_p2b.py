"""P2B plan builder: cell tables against P2's, the F split in the real router, R draws, block structure."""
import json

import pytest

import plan_p2 as P
import plan_p2b as B
from test_plan_p2 import route


def build(tmp_path, seed=7, mid=None, n_blocks=12, name="p2b"):
    root = tmp_path / name
    argv = ["--root", str(root), "--seed", str(seed), "--n-blocks", str(n_blocks)]
    assert B.main(argv + (["--mid-planner", mid] if mid else [])) == 0
    return root, json.loads((root / "plan.json").read_text())


def table_of(root, plan, cell):
    loop = next(l for l in plan["loops"] if l["cell"] == cell)
    return json.loads(Path_read(json.loads(Path_read(loop["config"]))["routing"]))


def Path_read(p):
    return open(p).read()


def test_p2_cells_identical_and_h_is_ss_with_harm(tmp_path):
    root, plan = build(tmp_path)
    for cell in ("SS", "ST", "QT"):
        assert table_of(root, plan, cell) == P.routing_table(cell)
    h = [l for l in plan["loops"] if l["cell"] == "H"]
    assert len(h) == 12
    for l in h:
        cfg = json.loads(Path_read(l["config"]))
        assert cfg["harm_rate"] == 0.5 and json.loads(Path_read(cfg["routing"])) == P.routing_table("SS")


def test_f_differs_from_qt_only_in_backlog_and_router_honors_it(tmp_path):
    root, plan = build(tmp_path)
    f, qt = table_of(root, plan, "F"), P.routing_table("QT")
    assert {k for k in f if f[k] != qt[k]} == {"planner_backlog"} and f["planner_backlog"] == "sonnet"
    assert route(f, "planner_backlog") == "sonnet"
    for r in ("planner_widen", "planner_refresh"):
        assert route(f, r) == "qwen35-4b"


def test_r_reproducible_varies_and_stored(tmp_path):
    a, pa = build(tmp_path, name="a")
    b, pb = build(tmp_path, name="b")
    ra = [l["r_draws"] for l in pa["loops"] if l["cell"] == "R"]
    assert ra == [l["r_draws"] for l in pb["loops"] if l["cell"] == "R"] and len(ra) == 12
    assert len({json.dumps(d, sort_keys=True) for d in ra}) > 1
    assert B.r_draws(7, 1) == B.r_draws(7, 1) != B.r_draws(8, 1)
    ss, qt = P.routing_table("SS"), P.routing_table("QT")
    for l in pa["loops"]:
        if l["cell"] != "R":
            continue
        cfg = json.loads(Path_read(l["config"]))
        assert cfg["r_draws"] == l["r_draws"] and set(cfg["r_draws"]) == set(B.R_KEYS)
        t = json.loads(Path_read(cfg["routing"]))
        for k in P.PLANNER_ROLES + ("repairer",):
            assert t[k] == (ss if l["r_draws"][k] == "SS" else qt)[k]
        for c in B.CLASSES:
            assert t["coder_by_class"][c] == ("sonnet" if l["r_draws"]["coder_by_class." + c] == "SS" else qt["coder_by_class"][c])
        assert (t["voter"], t["analyst"], t["critic"]) == ("gemma4-12b", "sonnet", "opus")


def test_blocks_have_each_cell_once_and_m_only_when_asked(tmp_path):
    _, plan = build(tmp_path)
    for ids in plan["blocks"]:
        assert sorted(i.split("_")[0] for i in ids) == ["F", "H", "QT", "R", "SS", "ST"]
    assert len(plan["loops"]) == 72 and plan["seed"] == 7
    root, plan = build(tmp_path, mid="qwen35-9b", name="m")
    for ids in plan["blocks"]:
        assert sorted(i.split("_")[0] for i in ids) == ["F", "H", "M", "QT", "R", "SS", "ST"]
    m = table_of(root, plan, "M")
    assert all(m[r] == "qwen35-9b" for r in P.PLANNER_ROLES) and m["coder_by_class"] == P.routing_table("QT")["coder_by_class"]


def test_refusals(tmp_path):
    root, _ = build(tmp_path)
    with pytest.raises(SystemExit):
        B.main(["--root", str(root), "--seed", "9"])
    with pytest.raises(SystemExit):
        B.main(["--root", str(tmp_path / "x"), "--seed", "2"])
    with pytest.raises(SystemExit):
        B.main(["--root", str(tmp_path / "y"), "--seed", "9", "--mid-planner", "nope"])
    assert P.FACTORIAL == ("SS", "ST", "QS", "QT")   # the patch is undone
