"""P2 plan builder: layout, configs, frozen routing tables and the factorial structure of the tables."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import controller
import plan_p2 as P
import run_loop
from arms import LoopEngine

ROLES = ("coder", "repairer", "planner_backlog", "planner_widen", "planner_refresh", "planner", "planner_cheap",
         "voter", "analyst", "critic", "base_step")
CLASSES = ("constant", "schedule", "structural", "mixed", None)


@pytest.fixture
def built(tmp_path):
    root = tmp_path / "p2"
    assert P.main(["--root", str(root), "--seed", "2"]) == 0
    return root, json.loads((root / "plan.json").read_text())


def route(table, role, cls=None):
    """The real LoopEngine._route on a minimal stand-in: it reads only self.arm and self.cfg."""
    eng = SimpleNamespace(arm="D", cfg={"routing": table, "voter_model": "gemma4-12b", "critic_model": "opus"})
    return LoopEngine._route(eng, role, cls)


def grid(table):
    return {(r, c): route(table, r, c) for r in ROLES for c in CLASSES}


def differing_roles(a, b):
    ga, gb = grid(a), grid(b)
    return {r for (r, c) in ga if ga[(r, c)] != gb[(r, c)]}


def table(cell):
    return P.routing_table(cell)


# ---------------------------------------------------------------- the factorial in the real routing function
def test_planner_factor_changes_only_planner_roles():
    for coder in "ST":
        assert differing_roles(table("S" + coder), table("Q" + coder)) == set(P.PLANNER_ROLES)
    g = grid(table("QS"))
    for r in ("planner_backlog", "planner_widen", "planner_refresh", "planner"):
        assert {g[(r, c)] for c in CLASSES} == {"qwen35-4b"}
    g = grid(table("SS"))
    for r in ("planner_backlog", "planner_widen", "planner_refresh", "planner"):
        assert {g[(r, c)] for c in CLASSES} == {"sonnet"}


def test_coder_factor_changes_only_coder_and_repairer():
    for planner in "SQ":
        assert differing_roles(table(planner + "S"), table(planner + "T")) == {"coder", "repairer"}
    s, t = grid(table("SS")), grid(table("ST"))
    assert {s[("coder", c)] for c in CLASSES} == {"sonnet"} and s[("repairer", None)] == "sonnet"
    assert (t[("coder", "constant")], t[("coder", "schedule")], t[("coder", "structural")]) == (
        "qwen35-4b", "gemma4-12b", "haiku")
    assert t[("repairer", None)] == "haiku"
    # a class the table does not list goes to the sonnet default, the same as in S, so it never confounds
    assert t[("coder", "mixed")] == t[("coder", None)] == "sonnet"


def test_everything_else_is_held_fixed_in_every_cell():
    for cell in P.FACTORIAL:
        g = grid(table(cell))
        for c in CLASSES:
            assert (g[("voter", c)], g[("analyst", c)], g[("critic", c)]) == ("gemma4-12b", "sonnet", "opus")
    ref = grid(table("SS"))
    for cell in P.FACTORIAL:
        g = grid(table(cell))
        fixed = [k for k in ref if k[0] in ("voter", "analyst", "critic", "base_step")]
        assert all(g[k] == ref[k] for k in fixed)


def test_structural_and_repairer_models_are_parameters():
    t = P.routing_table("QT", structural_model="sonnet", repairer_model="sonnet")
    assert route(t, "coder", "structural") == "sonnet" and route(t, "repairer") == "sonnet"
    assert route(t, "coder", "constant") == "qwen35-4b"


def test_no_string_value_that_is_not_a_model_name():
    known = P.usable_models()
    assert {"sonnet", "haiku", "opus", "qwen35-4b", "gemma4-12b"} <= known and "qwen7b-awq" not in known
    for cell in P.FACTORIAL:
        t = table(cell)
        strings = {v for v in list(t.values()) + list(t.get("coder_by_class", {}).values()) if isinstance(v, str)}
        assert strings <= known   # the engine's health check looks up every string value in the registry


def test_unknown_or_disabled_model_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        P.main(["--root", str(tmp_path / "x"), "--seed", "1", "--structural-model", "haikuu"])
    with pytest.raises(SystemExit):
        P.main(["--root", str(tmp_path / "y"), "--seed", "1", "--repairer-model", "qwen7b-awq"])
    assert not (tmp_path / "x" / "plan.json").exists()


# ---------------------------------------------------------------- layout
def test_plan_structure(built):
    root, plan = built
    loops = plan["loops"]
    assert len(loops) == 54 and len(plan["blocks"]) == 12
    cells = [l["cell"] for l in loops]
    assert {c: cells.count(c) for c in ("SS", "ST", "QS", "QT", "H")} == {"SS": 12, "ST": 12, "QS": 12, "QT": 12, "H": 6}
    assert all(l["arm"] == "D" for l in loops)
    by_id = {l["loop_id"]: l for l in loops}
    assert set(by_id) >= {"SS_01", "SS_12", "ST_01", "QS_12", "QT_12", "H_01", "H_06"} and len(by_id) == 54
    h_blocks = 0
    for b, ids in enumerate(plan["blocks"], start=1):
        got = sorted(by_id[i]["cell"] for i in ids)
        assert [c for c in got if c != "H"] == ["QS", "QT", "SS", "ST"] and got.count("H") <= 1
        h_blocks += "H" in got
        assert all(by_id[i]["block"] == b for i in ids)
    assert h_blocks == 6
    assert [l["loop_id"] for l in loops] == [i for ids in plan["blocks"] for i in ids]   # launch order
    assert [l["index"] for l in loops] == list(range(54))
    assert [l["loop_id"] for l in loops if l["cell"] == "SS"] == ["SS_%02d" % i for i in range(1, 13)]


def test_plan_top_level(built):
    root, plan = built
    assert plan["seed"] == 2 and plan["root"] == str(root) and plan["n_slots"] == 20
    assert plan["spend_ceiling_tokens"] == 60_000_000 and plan["sigma_used"] is None
    assert plan["anchor_file"] == str(root / "anchor.json")
    assert plan["sentinel_every"] == 45 and plan["confirm_seeds"] == [1001, 1002]
    assert set(plan["loops"][0]) == {"loop_id", "arm", "cell", "block", "index", "run_dir", "config", "stop_file"}


def test_configs(built):
    root, plan = built
    seeds = set()
    for l in plan["loops"]:
        c = json.loads(Path(l["config"]).read_text())
        seeds.add(c["rng_seed"])
        assert c["rng_seed"] == 2 * 1000 + l["index"]
        assert (c["loop_id"], c["arm"], c["cell"], c["block"]) == (l["loop_id"], "D", l["cell"], l["block"])
        assert c["run_dir"] == l["run_dir"] and c["stop_file"] == l["stop_file"] == c["run_dir"] + "/STOP_AT"
        assert c["anchor_file"] == str(root / "anchor.json")
        assert c["n_slots"] == 20 and c["job_timeout_s"] == 900 and c["timeout_s"] == 7200
        assert c["spend_ceiling_tokens"] == 60_000_000 and c["critic_model"] == "opus"
        assert (c["voter_model"], c["vote_k"]) == ("gemma4-12b", 3) and c["free_action_caps"] == {"slot": 2, "run": 40}
        assert c["harm_rate"] == (0.5 if l["cell"] == "H" else 0.0)
        table_name = "SS" if l["cell"] == "H" else l["cell"]
        assert c["routing"] == str(root / "routing" / (table_name + ".json"))
        assert Path(l["run_dir"]).is_dir()
    assert len(seeds) == 54


def test_tables_on_disk_match_the_cells(built):
    root, _ = built
    for cell in P.FACTORIAL:
        assert json.loads((root / "routing" / (cell + ".json")).read_text()) == P.routing_table(cell)
    assert sorted(p.name for p in (root / "routing").iterdir()) == ["QS.json", "QT.json", "SS.json", "ST.json"]


def test_deterministic_and_seed_dependent(tmp_path):
    a = P.make_plan(tmp_path / "a", 3)
    b = P.make_plan(tmp_path / "b", 3)
    strip = lambda p: [(l["loop_id"], l["block"], l["index"]) for l in p["loops"]]
    assert strip(a) == strip(b) and a["blocks"] == b["blocks"]
    layouts = {tuple(map(tuple, P.block_layout(s))) for s in range(6)}
    assert len(layouts) == 6
    pos = {c.index("H") for s in range(30) for c in P.block_layout(s) if "H" in c}
    assert len(pos) >= 4   # H lands at varied positions, not always last


def test_n_blocks_and_harm_blocks_options(tmp_path):
    p = P.make_plan(tmp_path / "s", 1, n_blocks=3, harm_blocks=1)
    assert len(p["loops"]) == 13 and len(p["blocks"]) == 3
    with pytest.raises(SystemExit):
        P.make_plan(tmp_path / "t", 1, n_blocks=3, harm_blocks=4)


def test_harm_blocks_12_gives_one_harm_loop_in_every_block(tmp_path):
    """The runbook's way to raise the power of the KILL gate: `--harm-blocks 12` is 12 H loops against 12 SS loops."""
    assert P.main(["--root", str(tmp_path / "p2"), "--seed", "2", "--harm-blocks", "12"]) == 0
    plan = json.loads((tmp_path / "p2" / "plan.json").read_text())
    assert len(plan["loops"]) == 60 and len(plan["blocks"]) == 12
    cells = [[l["cell"] for l in plan["loops"] if l["block"] == b] for b in range(1, 13)]
    assert all(sorted(c) == ["H", "QS", "QT", "SS", "ST"] for c in cells)
    h = [l for l in plan["loops"] if l["cell"] == "H"]
    assert len(h) == 12 and all(json.loads(Path(l["config"]).read_text())["harm_rate"] == 0.5 for l in h)


def test_refuses_to_overwrite(built):
    root, _ = built
    before = (root / "plan.json").read_text()
    with pytest.raises(SystemExit):
        P.main(["--root", str(root), "--seed", "9"])
    assert (root / "plan.json").read_text() == before


# ---------------------------------------------------------------- the plan works with the existing machinery
def test_configs_load_in_run_loop_and_the_controller_reads_the_plan(built):
    root, plan = built
    (root / "anchor.json").write_text(json.dumps({"anchor_mean": 1.114, "n": 16, "gpu_offsets": {"0": 0.0},
                                                  "sigma_used": 0.0047}))
    for l in plan["loops"]:
        cfg = run_loop.load_config(l["config"])
        assert cfg["routing"] == P.routing_table("SS" if l["cell"] == "H" else l["cell"])
        assert cfg["anchor"] == {"mean": 1.114, "n": 16} and cfg["sigma_used"] == 0.0047
    ctl = controller.Controller(root / "plan.json")
    assert len(ctl.order) == 54 and ctl.ceiling == 60_000_000 and ctl.plan["sentinel_every"] == 45
