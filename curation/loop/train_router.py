#!/usr/bin/env python3
"""Learn a routing table from the slot records of finished loops; routed.py uses it with --policy table:FILE.

    python curation/loop/train_router.py --root curation/runs/C1 --out curation/runs/C1/router.json [--cells RR SS ...]

What is learned. Every slot of every loop gives one record: the state before the slot, the models used for planner
and coder (strong or cheap), the dev gain the slot made, and its model dollars. The reward of a slot is the study's
metric cut into slots:
    reward = dev gain / (dev gain of SS loops - dev gain of CC loops)  -  dollars / (dollars of SS - dollars of CC)
so the rewards of a loop add up to its gap recovered minus its extra cost (up to a constant shared by all policies).
A state is two numbers a loop always knows: how much it has gained so far (in noise sds: under 2, 2 to 9, 9 or more)
and how many slots have passed since the last accepted change (0, 1 or 2, 3 or more).
From the records the script estimates, for each state and each of the four model choices, the mean reward and where
the loop goes next; a choice seen fewer than MIN_N times in a state borrows from the same gain level, then from all
states. The table is then the best choice for every state and number of slots left, found by working backwards from
the last slot (finite-horizon dynamic programming on the estimated model). Ties go to the cheaper choice.
"""
import argparse, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ACTIONS = ("cheap,cheap", "cheap,strong", "strong,cheap", "strong,strong")     # planner,coder; cheapest first
MIN_N = 4
END = "end"


CUTS = (2, 9)           # where the gain levels start, in noise sds; a table can carry its own under "cuts"


def key(state, cuts=CUTS):
    g, stall = state.get("gain_sd") or 0.0, state["stall"]
    return "%d%d" % (sum(g >= c for c in cuts), 0 if stall == 0 else 1 if stall < 3 else 2)


def decide(table, state):
    """Models for the coming slot, as {"planner": "strong" or "cheap", "coder": ...}."""
    left = str(min(state["left"], table["horizon"]))
    p, c = table["policy"][left][key(state, table.get("cuts", CUTS))].split(",")
    return {"planner": p, "coder": c}


def loop_rows(run_dir):
    """Slot records of one loop: (state key, action, dev gain, dollars, next state key). None if the loop cannot be used."""
    cfg = json.loads((run_dir / "config.json").read_text())
    rf = run_dir / "routes.jsonl"
    if not rf.exists() or cfg.get("harm"):
        return None
    rows = [json.loads(l) for l in open(rf)]
    names = {cfg["strong"]: "strong", cfg["cheap"]: "cheap"}
    if any(r["planner"] not in names or r["coder"] not in names for r in rows):
        return None
    start = inc = next((r["inc"] for r in rows if r.get("inc") is not None), None)
    out = []
    for r in rows:
        s = dict(r, gain_sd=0.0 if start is None else (start - inc) / cfg["sigma"])
        gain = r["inc"] - r["dev"] if r["decision"] == "accept" else 0.0
        if r["decision"] == "accept":
            inc = r["dev"]
        out.append([key(s), names[r["planner"]] + "," + names[r["coder"]], gain, r["usd"], END])
    for a, b in zip(out, out[1:]):
        a[4] = b[0]
    return out


def collect(root, cells=None):
    loops = {}
    for f in sorted(Path(root).glob("*/final.json")):
        d = json.loads(f.read_text())
        rows = loop_rows(f.parent)
        if rows is not None and (cells is None or d["cell"] in cells):
            loops[d["loop_id"]] = {"cell": d["cell"], "rows": rows}
    return loops


def scales(loops, strong="SS", cheap="CC"):
    """Dev gain gap and dollar gap between the anchor cells (per loop)."""
    tot = lambda cell, i: st.mean(sum(r[i] for r in l["rows"]) for l in loops.values() if l["cell"] == cell)
    return tot(strong, 2) - tot(cheap, 2), tot(strong, 3) - tot(cheap, 3)


def fit(rows, gain_gap, usd_gap, horizon):
    """rows: [state, action, gain, usd, next state]. Returns the table (policy per slots left and state, plus the model)."""
    cell, level, glob = defaultdict(list), defaultdict(list), defaultdict(list)
    for k, a, gain, usd, nxt in rows:
        x = (gain / gain_gap - usd / usd_gap, nxt)
        cell[k, a].append(x)
        level[k[0], a].append(x)
        glob[a].append(x)
    states = ["%d%d" % (g, s) for g in range(3) for s in range(3)]
    model = {}
    for k in states:
        for a in ACTIONS:
            src = next((s for s in (cell[k, a], level[k[0], a], glob[a]) if len(s) >= MIN_N), None)
            if src is None:
                continue                              # this choice was never tried often enough anywhere: not offered
            nxt = defaultdict(float)
            for _, n in src:
                nxt[n] += 1 / len(src)
            model[k, a] = {"reward": st.mean(x[0] for x in src), "next": dict(nxt), "n_own": len(cell[k, a]), "n_used": len(src)}
    value, policy, q = {k: 0.0 for k in states}, {}, {}
    for left in range(1, horizon + 1):
        new, policy[str(left)], q[str(left)] = {}, {}, {}
        for k in states:
            qs = {a: model[k, a]["reward"] + sum(p * value.get(n, 0.0) for n, p in model[k, a]["next"].items() if n != END)
                  for a in ACTIONS if (k, a) in model}
            best = max(qs, key=lambda a: (round(qs[a], 9), -ACTIONS.index(a)))
            new[k], policy[str(left)][k], q[str(left)][k] = qs[best], best, qs
        value = new
    return {"horizon": horizon, "policy": policy, "q": q, "gain_gap": gain_gap, "usd_gap": usd_gap, "n_rows": len(rows),
            "model": {"%s|%s" % ka: v for ka, v in model.items()}}


def value_of(table, choose, start="00"):
    """Expected sum of rewards over the horizon, under the fitted model, of a rule choose(left, state key) -> action."""
    model = {tuple(k.split("|")): v for k, v in table["model"].items()}
    states = sorted({k for k, _ in model})
    value = {k: 0.0 for k in states}
    for left in range(1, table["horizon"] + 1):
        value = {k: (lambda m: m["reward"] + sum(p * value.get(n, 0.0) for n, p in m["next"].items() if n != END))(
            model[k, choose(left, k)]) for k in states if (k, choose(left, k)) in model}
    return value.get(start)


def show(table):
    print("%d slot records; anchor gaps: dev gain %.4f bpb, dollars %.2f" % (table["n_rows"], table["gain_gap"], table["usd_gap"]))
    states = sorted(table["policy"]["1"])
    print("state = gain so far (0: under 2 sds, 1: 2 to 9, 2: 9 or more) then slots since the last accept (0: none, 1: 1-2, 2: 3+)")
    print("slots left  " + "  ".join("%-13s" % s for s in states))
    for left in range(table["horizon"], 0, -1):
        print("%10d  %s" % (left, "  ".join("%-13s" % table["policy"][str(left)][s] for s in states)))
    h = table["horizon"]
    rules = {"learned table": lambda left, k: table["policy"][str(left)][k],
             "all strong": lambda left, k: "strong,strong", "all cheap": lambda left, k: "cheap,cheap",
             "strong planner": lambda left, k: "strong,cheap", "strong coder": lambda left, k: "cheap,strong",
             "strong in slot 1": lambda left, k: "strong,strong" if left == h else "cheap,cheap"}
    print("value under the fitted model (sum of slot rewards from the start; all cheap would be 0 with a perfect model):")
    for name, rule in rules.items():
        v = value_of(table, rule)
        print("  %-18s %s" % (name, "not estimable" if v is None else "%.3f" % v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=None, help="cells to learn from (default: every usable loop)")
    ap.add_argument("--horizon", type=int, default=12)
    a = ap.parse_args(argv)
    everything = collect(a.root)
    gain_gap, usd_gap = scales(everything)
    loops = everything if a.cells is None else {k: v for k, v in everything.items() if v["cell"] in a.cells}
    rows = [r for l in loops.values() for r in l["rows"]]
    table = fit(rows, gain_gap, usd_gap, a.horizon)
    table["loops"] = sorted(loops)
    Path(a.out).write_text(json.dumps(table, indent=1))
    show(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
