"""The learned routing table, on a made-up world where one strong slot does all the work."""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routed as R
import train_router as TR
from test_engine import FakeModels, FakeRunner, GOOD


def world(n_loops=60, slots=12, seed=0):
    """Random routing. A slot with a strong coder makes the whole gain (0.1 bpb, 14 sds) if nothing
    has been gained yet; nothing else ever helps. Dollars: strong planner 0.2, strong coder 0.4."""
    rng, rows = random.Random(seed), []
    for _ in range(n_loops):
        gained, stall, loop = False, 0, []
        for _ in range(slots):
            a = rng.choice(TR.ACTIONS)
            k = TR.key({"gain_sd": 14.0 if gained else 0.0, "stall": stall})
            hit = (not gained) and a.endswith("strong")
            usd = 0.2 * a.startswith("strong") + 0.4 * a.endswith("strong")
            loop.append([k, a, 0.1 if hit else 0.0, usd, TR.END])
            gained, stall = gained or hit, 0 if hit else stall + 1
        for x, y in zip(loop, loop[1:]):
            x[4] = y[0]
        rows += loop
    return rows


def test_key():
    assert TR.key({"stall": 0}) == "00" and TR.key({"gain_sd": 5, "stall": 2}) == "11" and TR.key({"gain_sd": 30, "stall": 7}) == "22"


def test_table_pays_for_the_strong_coder_only_until_the_gain_is_made():
    t = TR.fit(world(), gain_gap=0.1, usd_gap=7.2, horizon=12)
    assert t["policy"]["1"]["00"] == "cheap,strong" and t["policy"]["1"]["02"] == "cheap,strong"
    assert t["policy"]["5"]["00"] == "cheap,cheap"      # earlier, waiting costs nothing in this world; ties go to cheap
    for left in ("1", "6", "12"):
        assert t["policy"][left]["20"] == "cheap,cheap" and t["policy"][left]["22"] == "cheap,cheap"
    assert TR.decide(t, {"left": 1, "stall": 0, "gain_sd": 0.0}) == {"planner": "cheap", "coder": "strong"}
    assert TR.decide(t, {"left": 40, "stall": 5, "gain_sd": 20.0}) == {"planner": "cheap", "coder": "cheap"}
    v = {n: TR.value_of(t, r) for n, r in (("table", lambda left, k: t["policy"][str(left)][k]),
                                           ("strong", lambda left, k: "strong,strong"), ("cheap", lambda left, k: "cheap,cheap"))}
    assert abs(v["table"] - (1 - 0.4 / 7.2)) < 1e-9 and v["table"] > v["strong"] and abs(v["cheap"]) < 1e-9


def test_a_choice_never_tried_is_not_offered():
    rows = [r for r in world() if r[1] != "strong,cheap"]
    t = TR.fit(rows, 0.1, 7.2, 12)
    assert all("strong,cheap" not in q for q in t["q"]["12"].values())


def test_routed_loop_follows_a_table_and_training_reads_its_records(tmp_path):
    states = ["%d%d" % (g, s) for g in range(3) for s in range(3)]
    t = {"horizon": 12, "policy": {str(n): {k: "cheap,strong" if k == "00" else "cheap,cheap" for k in states}
                                   for n in range(1, 13)}}
    (tmp_path / "router.json").write_text(json.dumps(t))
    cfg = {"loop_id": "T", "run_dir": str(tmp_path / "T_01"), "cell": "T", "pool": "v2", "slots": 5, "seed": 1, "sigma": 0.007,
           "gpu_host": None, "policy": "table:%s" % (tmp_path / "router.json"), "strong": "big", "cheap": "small",
           "harm": 0.0, "routing": None}
    (tmp_path / "T_01").mkdir()
    (tmp_path / "T_01" / "config.json").write_text(json.dumps(cfg))
    lp = R.RoutedLoop(cfg, models=FakeModels(), runner=FakeRunner(dev={1: 1.80}))
    lp.run()
    rows = [json.loads(l) for l in open(tmp_path / "T_01" / "routes.jsonl")]
    assert (rows[0]["planner"], rows[0]["coder"], rows[0]["decision"]) == ("small", "big", "accept")
    assert all((r["planner"], r["coder"]) == ("small", "small") for r in rows[1:])
    assert rows[0]["gain_sd"] == 0 and rows[1]["gain_sd"] > 9
    got = TR.loop_rows(tmp_path / "T_01")
    assert [g[0] for g in got] == [TR.key(r) for r in rows] and got[0][1] == "cheap,strong" and got[-1][4] == TR.END
    assert abs(got[0][2] - (rows[0]["inc"] - rows[0]["dev"])) < 1e-12 and got[1][2] == 0.0
