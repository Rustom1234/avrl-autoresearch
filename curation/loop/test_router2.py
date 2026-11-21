"""The level-model router trainer, on a made-up world where one slot with a strong planner does all the work."""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_router as TR
import train_router2 as T2

SIGMA, GAIN, USD_GAP = 0.01, 0.2, 7.2          # the whole gain is 20 sds; an all-strong loop costs 12 * 0.6 dollars
RULES = T2.rules(12)
RULES["RR"] = None


def loop(cell, seed, slots=12):
    """A slot with a strong planner makes the whole gain if nothing has been gained yet; nothing else ever helps.
    Dollars per slot: strong planner 0.2, strong coder 0.4."""
    rng, gained, stall, recs = random.Random(seed), False, 0, []
    for i in range(slots):
        k = TR.key({"gain_sd": GAIN / SIGMA if gained else 0.0, "stall": stall})
        a = rng.choice(T2.ACTIONS) if RULES[cell] is None else RULES[cell](slots - i, k)
        hit = (not gained) and a.startswith("strong")
        recs.append((k, a, 0.2 * a.startswith("strong") + 0.4 * a.endswith("strong"), 2 if hit else None))
        gained, stall = gained or hit, 0 if hit else stall + 1
    return {"cell": cell, "recs": recs, "gain": GAIN * gained, "end_level": 2 * gained, "usd": sum(r[2] for r in recs), "sigma": SIGMA}


def world(cells=("RR",) * 60):
    return {"%s_%02d" % (c, i): loop(c, i) for i, c in enumerate(cells)}


def test_levels_agree_with_the_state_key():
    for g in (-3, 0, 1.99, 2, 5, 8.99, 9, 40):
        assert str(T2.level(g)) == TR.key({"gain_sd": g, "stall": 0})[0]


def test_stall_part_of_the_state_moves_as_the_loop_moves_it():
    assert T2.step("01", 2) == [("20", 1.0)] and T2.step("21", 1) == [("20", 1.0)]      # an accept never lowers the level
    assert T2.step("10", None) == [("11", 1.0)] and T2.step("12", None) == [("12", 1.0)]
    assert T2.step("01", None) == [("01", 0.5), ("02", 0.5)]


def test_table_pays_for_the_strong_planner_only_until_the_gain_is_made():
    t = T2.fit(world(), GAIN, USD_GAP, horizon=12)
    policy, q, v = T2.plan(t)
    assert policy["1"]["00"] == "strong,cheap" and policy["1"]["02"] == "strong,cheap"
    assert policy["5"]["00"] == "cheap,cheap"           # earlier, waiting costs nothing in this world; ties go to cheap
    for left in ("1", "6", "12"):
        assert policy[left]["20"] == "cheap,cheap" and policy[left]["22"] == "cheap,cheap"
    assert abs(v - (1 - 0.2 / USD_GAP)) < 1e-9
    assert abs(T2.value(t, RULES["SS"])) < 1e-9 and abs(T2.value(t, RULES["CC"])) < 1e-9
    assert abs(T2.value(t, RULES["SC"]) - (1 - 12 * 0.2 / USD_GAP)) < 1e-9
    table = json.loads(json.dumps({"horizon": 12, "policy": policy}))                # as routed.py reads it
    assert TR.decide(table, {"left": 1, "stall": 0, "gain_sd": 0.0}) == {"planner": "strong", "coder": "cheap"}
    assert TR.decide(table, {"left": 40, "stall": 5, "gain_sd": 20.0}) == {"planner": "cheap", "coder": "cheap"}


def test_end_values_and_borrowing():
    t = T2.fit(world(), GAIN, USD_GAP, min_own=T2.LEVEL_ONLY)
    assert t["end_value"][0] == 0 and abs(t["end_value"][2] - 1) < 1e-9
    assert abs(t["end_value"][1] - 2 * SIGMA / GAIN) < 1e-9            # no loop ends in level 1: valued at its lower cut
    assert {m["source"] for (k, a), m in t["model"].items() if k[0] != "1"} == {"level"}
    assert {m["source"] for (k, a), m in t["model"].items() if k[0] == "1"} == {"all"}       # level 1 is never visited
    assert set(t["model"]["12", "strong,cheap"]["next"]) == {"20", "12"}      # a borrowed accept goes to the top, the rest stay
    own = T2.fit(world(), GAIN, USD_GAP, min_own=4)
    assert own["model"]["20", "cheap,cheap"]["source"] == "state"


def test_a_choice_never_tried_is_not_offered():
    loops = {i: dict(l, recs=[r for r in l["recs"] if r[1] != "strong,strong"]) for i, l in world().items()}
    t = T2.fit(loops, GAIN, USD_GAP)
    assert not any(a == "strong,strong" for _, a in t["model"])
    assert T2.value(t, RULES["SS"]) is None
    assert "strong,strong" not in {a for row in T2.plan(t)[0].values() for a in row.values()}


def test_held_out_cells_are_predicted_in_a_world_without_noise():
    res = T2.check(world(("RR",) * 60 + ("SS", "CC", "SC", "CS", "F1") * 4), GAIN, USD_GAP)
    assert abs(res["SC"][0] - (1 - 12 * 0.2 / USD_GAP)) < 1e-9 and abs(res["F1"][0] - (1 - 0.6 / USD_GAP)) < 1e-9
    for obs, with_it, without_it in res.values():
        assert abs(obs - with_it) < 1e-9 and abs(obs - without_it) < 1e-9
    err, complete = T2.errors(res)
    assert err < 1e-9 and complete


def test_read_turns_a_run_directory_into_records(tmp_path):
    run = tmp_path / "SC_01"
    run.mkdir()
    (run / "final.json").write_text(json.dumps({"loop_id": "SC_01", "cell": "SC"}))
    (run / "config.json").write_text(json.dumps({"strong": "fable", "cheap": "gemma", "sigma": 0.01}))
    base = {"planner": "fable", "coder": "gemma", "usd": 0.25}
    rows = [dict(base, slot=1, inc=2.0, dev=2.1, decision="reject", stall=0),
            dict(base, slot=2, inc=2.0, dev=1.95, decision="accept", stall=1),      # 5 sds: level 1
            dict(base, slot=3, inc=1.95, dev=None, decision="fail", stall=0),
            dict(base, slot=4, inc=1.95, dev=1.8, decision="accept", stall=1, coder="fable", usd=0.6)]     # 20 sds: level 2
    (run / "routes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    harmed = tmp_path / "DH_01"
    harmed.mkdir()
    (harmed / "final.json").write_text(json.dumps({"loop_id": "DH_01", "cell": "DH"}))
    (harmed / "config.json").write_text(json.dumps({"strong": "fable", "cheap": "gemma", "sigma": 0.01, "harm": 0.5}))
    (harmed / "routes.jsonl").write_text(json.dumps(rows[0]) + "\n")
    loops = T2.read(tmp_path)
    assert list(loops) == ["SC_01"]                      # damaged loops are not training data
    l = loops["SC_01"]
    assert l["recs"] == [("00", "strong,cheap", 0.25, None), ("01", "strong,cheap", 0.25, 1), ("10", "strong,cheap", 0.25, None),
                         ("11", "strong,strong", 0.6, 2)]
    assert abs(l["gain"] - 0.2) < 1e-9 and l["end_level"] == 2 and abs(l["usd"] - 1.35) < 1e-9


def test_four_levels_reach_the_live_decision():
    """With a third cut the table has four gain levels, carries its cuts, and train_router.decide reads them."""
    try:
        T2.use_cuts((2, 9, 14))
        assert T2.level(10) == 2 and T2.level(14) == 3 and len(T2.STATES) == 12
        table = {"horizon": 2, "cuts": [2, 9, 14],
                 "policy": {str(n): {k: "strong,cheap" if k[0] == "2" else "cheap,cheap" for k in T2.STATES} for n in (1, 2)}}
        assert TR.decide(table, {"left": 5, "stall": 1, "gain_sd": 10.0}) == {"planner": "strong", "coder": "cheap"}
        assert TR.decide(table, {"left": 1, "stall": 0, "gain_sd": 14.0}) == {"planner": "cheap", "coder": "cheap"}
        assert TR.key({"gain_sd": 10.0, "stall": 0}) == "20" and TR.key({"gain_sd": 15.0, "stall": 0}, (2, 9, 14)) == "30"
    finally:
        T2.use_cuts((2, 9))
    assert T2.CUTS == (0, 2, 9) and len(T2.STATES) == 9
