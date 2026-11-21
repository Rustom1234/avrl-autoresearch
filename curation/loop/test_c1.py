"""The C1 plan and analysis, on made-up loops."""
import json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_c1 as A
import launch_c1 as L


def plan(**kw):
    a = dict(blocks=4, pool="v2", slots=12, sigma=0.007, strong="fable", cheap="gemma4-12b", middle="haiku", plan_seed=7)
    return L.make_plan(SimpleNamespace(**dict(a, **kw)))


def test_plan_has_every_cell_once_per_block_and_random_twice():
    p = plan()
    assert len(p["loops"]) == 48 and len({l["id"] for l in p["loops"]}) == 48
    for b in range(1, 5):
        cells = sorted(l["cell"] for l in p["loops"] if l["block"] == b)
        assert cells == sorted(c[0] for c in L.CELLS)
        seeds = [l["seed"] for l in p["loops"] if l["block"] == b and l["cell"] != "RR"]
        assert set(seeds) == {b}
        assert sorted(l["seed"] for l in p["loops"] if l["block"] == b and l["cell"] == "RR") == [b, 50 + b]
    hh = next(l for l in p["loops"] if l["cell"] == "HH")
    assert hh["policy"] == "fixed:haiku,haiku"
    assert [l["harm"] for l in p["loops"] if l["cell"] == "DH"] == [0.5] * 4
    assert p == plan() and [l["id"] for l in p["loops"]] != [l["id"] for l in plan(plan_seed=8)["loops"]]


def test_commands(tmp_path):
    p = plan()
    by = {l["id"]: l for l in p["loops"]}
    c = L.command(p, by["F1_02"], tmp_path, "willow")
    assert c[1].endswith("routed.py") and c[c.index("--policy") + 1] == "first" and c[c.index("--seed") + 1] == "2"
    assert c[c.index("--gpu-host") + 1] == "willow" and c[c.index("--cheap") + 1] == "gemma4-12b"
    n = L.command(p, by["N_03"], tmp_path, "local")
    assert n[1].endswith("menu.py") and "--gpu-host" not in n and "--policy" not in n
    (tmp_path / "F1_02").mkdir()
    (tmp_path / "F1_02" / "config.json").write_text("{}")
    assert "--resume" in L.command(p, by["F1_02"], tmp_path, "cherry") and "--gpu-host" not in L.command(p, by["F1_02"], tmp_path, "cherry")


def fake(cell, n, gain, usd, lost=False, accepts=1):
    return {"id": "%s_%02d" % (cell, n), "cell": cell, "sealed": 2.0 - gain, "usd": usd, "gain": gain, "n_accept": accepts,
            "infra_waits": 0, "resumes": 0, "lost_slot": lost, "curve": {1: gain / 2, 12: gain}}


def study(dh=0.05, f1=0.11):
    loops = []
    for n in range(1, 5):
        e = 0.002 * (n - 2.5)
        loops += [fake("SS", n, 0.12 + e, 7.5), fake("CC", n, 0.02 - e, 0.02), fake("F1", n, f1 + e, 0.65),
                  fake("SC", n, 0.07 - e, 2.5), fake("DH", n, dh + e, 7.5)]
    return loops


def test_advantage_and_verdict_when_the_damaged_loop_is_worse():
    r = A.analyse(study(), raw_sd=0.007)
    assert r["G2"]["pass"] and r["V"]["pass"] and r["verdict"] == "SCORED"
    f1 = r["cells"]["F1"]
    assert abs(f1["gap_recovered"] - 0.9) < 1e-9 and abs(f1["extra_cost"] - 0.63 / 7.48) < 1e-9
    assert f1["adv_ci"][0] < f1["advantage"] < f1["adv_ci"][1] and f1["p_adv_pos"] == 1.0
    assert r["H2"]["confirmed"] and r["H1"]["confirmed"]
    assert abs(r["cells"]["SC"]["advantage"] - (0.5 - 2.48 / 7.48)) < 1e-9
    assert r["cells"]["SS"]["curve"] == {1: 0.06, 12: 0.12}


def test_shallow_when_the_damaged_loop_does_as_well():
    r = A.analyse(study(dh=0.12), raw_sd=0.007)
    assert r["G2"]["pass"] and not r["V"]["pass"] and r["verdict"] == "SHALLOW"
    assert r["H2"]["ci_above_zero"] and not r["H2"]["confirmed"]


def test_no_gap_stops_the_scoring():
    loops = [fake(c, n, 0.05 + 0.002 * n, u) for n in range(1, 5) for c, u in (("SS", 7.5), ("CC", 0.02), ("F1", 0.6))]
    r = A.analyse(loops, raw_sd=0.007)
    assert not r["G2"]["pass"] and r["verdict"] == "NO GAP" and "advantage" not in r["cells"]["F1"]


def test_a_loop_that_lost_a_slot_is_left_out():
    loops = study() + [fake("F1", 5, -1.0, 0.65, lost=True)]
    r = A.analyse(loops, raw_sd=0.007)
    assert r["left_out"] == ["F1_05"] and r["cells"]["F1"]["n"] == 4


def test_lost_slot_and_curve_from_a_ledger():
    led = [{"slot": 1, "decision": "accept", "dev": 1.9, "inc": 2.0, "calls": [{"ok": True, "error_class": None}]},
           {"slot": 2, "decision": "reject", "dev": 1.95, "inc": 1.9, "calls": [{"ok": False, "error_class": "context"}]}]
    assert not A.lost_slot(led) and {k: round(v, 6) for k, v in A.curve(led).items()} == {1: 0.1, 2: 0.1}
    led[1]["calls"].append({"ok": False, "error_class": "infra"})
    assert A.lost_slot(led)
