#!/usr/bin/env python3
"""Second router trainer (level model). Writes the same table format as train_router.py, so routed.py uses it
unchanged with --policy table:FILE.

    python curation/loop/train_router2.py --root curation/runs/C1 --out curation/runs/C1/router_lv.json --check

Why a second trainer. train_router.py adds up slot rewards (dev gain of the slot minus its dollars). On the C1 loops
that model fails a simple test: fitted without the loops of a cell, it gets the value of that cell's policy wrong
(it ranks all strong first and strong planner last; the loops say the opposite). The gain of one slot depends on
where the loop already is, and with a few hundred slots the per-slot means are too noisy to carry that.

What this one does. C1 loops end at one of a few levels of dev gain (--cuts, in noise sds; the default 2 and 9 gives
the three levels of train_router.key), and the final gain depends on the level, hardly on the path. So:
- the value of ending in a level is the mean final dev gain of the training loops that ended there, as a share of
  the gap between the anchor cells (SS minus CC); a level no training loop ends in is valued at its lower cut;
- a slot costs its dollars as a share of the anchor dollar gap;
- for each state and model choice the records give the chance that the slot is accepted and which level it reaches;
  a choice with fewer than --min-own records in a state borrows from all states of the same level, then from all
  states (a borrowed outcome never lowers the level);
- how the "slots since the last accept" part of the state moves is known, not estimated: it goes to 0 on an accept,
  from 0 to 1-2 otherwise, and from 1-2 to 3+ after two slots (taken as chance 0.5 per slot, since the state does
  not say which of the two it is).
The table is the best choice per state and slots left, found backwards from the last slot. Ties go to the cheaper
choice. Built in: once the top level is reached the model sees nothing more to gain, so it stops paying there.
That makes the top cut matter. With cuts 2 and 9 the table goes all cheap at 9 sds, but in C1 four of the seven loops
that first passed 9 sds below 14 later rose to 16 to 19, all with a strong planner. --cuts 2 9 14 lets the model see
that. The cuts are written into the table and train_router.decide uses them.

--check fits the model without each cell in turn and compares the value it gives that cell's policy with what the
cell's loops got (final dev gain share minus dollar share). It uses training loops only.
"""
import argparse, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import train_router as tr

ACTIONS = tr.ACTIONS
STATES = ["%d%d" % (g, s) for g in range(3) for s in range(3)]
MIN_BORROW = 4          # fewest records a level or the whole data set needs before it is borrowed from
LEVEL_ONLY = 10 ** 9    # --min-own value that makes every state use its level's records


CUTS = (0, 2, 9)        # lower ends of the gain levels, in noise sds (default: the same cuts as train_router.key)


def use_cuts(cuts):
    """Set the gain levels (lower ends in noise sds, without the 0) for everything below."""
    global CUTS, STATES
    CUTS = (0, *cuts)
    STATES = ["%d%d" % (g, s) for g in range(len(CUTS)) for s in range(3)]


def level(gain_sd):
    return sum(gain_sd >= c for c in CUTS[1:])


def read(root):
    """Usable loops (same filter as train_router): slot records (state, action, dollars, level reached if the slot
    was accepted else None), final dev gain in bpb, dollars."""
    loops = {}
    for f in sorted(Path(root).glob("*/final.json")):
        d, run = json.loads(f.read_text()), f.parent
        if tr.loop_rows(run) is None:
            continue
        cfg = json.loads((run / "config.json").read_text())
        rows = [json.loads(l) for l in open(run / "routes.jsonl")]
        names = {cfg["strong"]: "strong", cfg["cheap"]: "cheap"}
        start = inc = next(r["inc"] for r in rows if r.get("inc") is not None)
        recs = []
        for r in rows:
            k = tr.key(dict(r, gain_sd=(start - inc) / cfg["sigma"]), CUTS[1:])
            accepted = r["decision"] == "accept"
            if accepted:
                inc = r["dev"]
            recs.append((k, names[r["planner"]] + "," + names[r["coder"]], r["usd"],
                         level((start - inc) / cfg["sigma"]) if accepted else None))
        loops[d["loop_id"]] = {"cell": d["cell"], "recs": recs, "gain": start - inc, "end_level": level((start - inc) / cfg["sigma"]),
                               "usd": sum(r["usd"] for r in rows), "sigma": cfg["sigma"]}
    return loops


def anchors(loops, strong="SS", cheap="CC"):
    """Final dev gain gap and dollar gap between the anchor cells (per loop)."""
    m = lambda cell, f: st.mean(l[f] for l in loops.values() if l["cell"] == cell)
    return m(strong, "gain") - m(cheap, "gain"), m(strong, "usd") - m(cheap, "usd")


def step(k, out):
    """Next states and their chances from state k when the slot reached level out (None: not accepted)."""
    lvl, s = int(k[0]), int(k[1])
    if out is not None:
        return [("%d0" % max(lvl, out), 1.0)]
    if s == 0:
        return [("%d1" % lvl, 1.0)]
    if s == 1:
        return [("%d1" % lvl, 0.5), ("%d2" % lvl, 0.5)]
    return [("%d2" % lvl, 1.0)]


def fit(loops, gain_gap, usd_gap, horizon=12, min_own=4):
    own, lev, glob, ends = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for l in loops.values():
        ends[l["end_level"]].append(l["gain"] / gain_gap)
        for k, a, usd, out in l["recs"]:
            x = (usd / usd_gap, out)
            own[k, a].append(x)
            lev[k[0], a].append(x)
            glob[a].append(x)
    sigma = next(iter(loops.values()))["sigma"]
    floor = {L: cut * sigma / gain_gap for L, cut in enumerate(CUTS)}     # a level no training loop ends in: its lower cut
    end_value = {L: st.mean(ends[L]) if ends[L] else floor[L] for L in range(len(CUTS))}
    model = {}
    for k in STATES:
        for a in ACTIONS:
            cands = (("state", own[k, a], min_own), ("level", lev[k[0], a], MIN_BORROW), ("all", glob[a], MIN_BORROW))
            name, src = next(((n, s) for n, s, need in cands if len(s) >= need), (None, None))
            if src is None:
                continue                              # never tried often enough anywhere: not offered
            nxt = defaultdict(float)
            for _, out in src:
                for n, p in step(k, out):
                    nxt[n] += p / len(src)
            model[k, a] = {"cost": st.mean(x[0] for x in src), "next": dict(nxt), "n_own": len(own[k, a]), "n_used": len(src),
                           "source": name}
    return {"horizon": horizon, "end_value": end_value, "model": model}


def value(t, rule, start="00"):
    """Value under the fitted model of rule(slots left, state) -> action. None if the rule uses a choice not offered."""
    v = {k: t["end_value"][int(k[0])] for k in STATES}
    for left in range(1, t["horizon"] + 1):
        new = {}
        for k in STATES:
            m = t["model"].get((k, rule(left, k)))
            if m is None:
                return None
            new[k] = -m["cost"] + sum(p * v[n] for n, p in m["next"].items())
        v = new
    return v[start]


def plan(t, start="00"):
    """Best choice per slots left and state, and the value of following it from the start state."""
    v, policy, q = {k: t["end_value"][int(k[0])] for k in STATES}, {}, {}
    for left in range(1, t["horizon"] + 1):
        new, policy[str(left)], q[str(left)] = {}, {}, {}
        for k in STATES:
            qs = {a: -t["model"][k, a]["cost"] + sum(p * v[n] for n, p in t["model"][k, a]["next"].items())
                  for a in ACTIONS if (k, a) in t["model"]}
            best = max(qs, key=lambda a: (round(qs[a], 9), -ACTIONS.index(a)))
            new[k], policy[str(left)][k], q[str(left)][k] = qs[best], best, qs
        v = new
    return policy, q, v[start]


def rules(horizon):
    return {"SS": lambda left, k: "strong,strong", "CC": lambda left, k: "cheap,cheap", "SC": lambda left, k: "strong,cheap",
            "CS": lambda left, k: "cheap,strong", "F1": lambda left, k: "strong,strong" if left == horizon else "cheap,cheap"}


def check(loops, gain_gap, usd_gap, horizon=12, min_own=4):
    """Per fixed-policy cell: (observed value, value from a fit on all loops, value from a fit without the cell)."""
    full, out = fit(loops, gain_gap, usd_gap, horizon, min_own), {}
    for cell, rule in rules(horizon).items():
        mine = [l for l in loops.values() if l["cell"] == cell]
        if not mine:
            continue
        rest = fit({i: l for i, l in loops.items() if l["cell"] != cell}, gain_gap, usd_gap, horizon, min_own)
        out[cell] = (st.mean(l["gain"] / gain_gap - l["usd"] / usd_gap for l in mine), value(full, rule), value(rest, rule))
    return out


def errors(res):
    """Mean absolute error of the held-out values, and whether every held-out value could be computed."""
    done = [abs(o - h) for o, _, h in res.values() if h is not None]
    return (st.mean(done) if done else None), len(done) == len(res)


def show(table):
    print("anchor gaps: final dev gain %.4f bpb, dollars %.2f; value of ending in each gain level: %s" % (
        table["gain_gap"], table["usd_gap"], ", ".join("%.2f" % table["end_value"][str(L)] for L in range(len(CUTS)))))
    print("state = gain level (levels start at %s noise sds) then slots since the last accept (0: none, 1: 1-2, 2: 3+)" % (
        ", ".join("%g" % c for c in CUTS)))
    short = lambda a: "".join(x[0] for x in a.split(","))         # sc = strong planner, cheap coder
    print("slots left  " + "  ".join(STATES) + "     (planner then coder: s strong, c cheap)")
    for left in range(table["horizon"], 0, -1):
        print("%10d  %s" % (left, "  ".join(short(table["policy"][str(left)][k]) for k in STATES)))
    print("value of the table under its own model: %.2f" % table["value"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-own", type=int, default=4, help="records a state needs to use its own estimate (%d: level only)" % LEVEL_ONLY)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--cells", nargs="+", default=None, help="cells to learn from (default: every usable loop); the anchor "
                    "gaps always come from SS and CC")
    ap.add_argument("--check", action="store_true", help="held-out-cell check for several --min-own values")
    ap.add_argument("--cuts", nargs="+", type=float, default=list(tr.CUTS), help="where the gain levels start, in noise sds")
    a = ap.parse_args(argv)
    use_cuts(a.cuts)
    loops = read(a.root)
    gain_gap, usd_gap = anchors(loops)
    if a.cells:
        loops = {i: l for i, l in loops.items() if l["cell"] in a.cells}
    print("%d loops, %d slot records" % (len(loops), sum(len(l["recs"]) for l in loops.values())))
    if a.check:
        for mo in (4, 8, 16, LEVEL_ONLY):
            res = check(loops, gain_gap, usd_gap, a.horizon, mo)
            err, complete = errors(res)
            fmt = lambda v: " n/a" if v is None else "%.2f" % v
            print("min-own %s: mean abs error held out %s%s" % ("level only" if mo == LEVEL_ONLY else mo, fmt(err),
                                                                 "" if complete else " (some not computable)"))
            for cell, (obs, ins, held) in res.items():
                print("   %-3s observed %.2f  fitted with it %s  fitted without it %s" % (cell, obs, fmt(ins), fmt(held)))
    t = fit(loops, gain_gap, usd_gap, a.horizon, a.min_own)
    policy, q, v = plan(t)
    table = {"trainer": "train_router2", "horizon": a.horizon, "policy": policy, "q": q, "value": v, "min_own": a.min_own,
             "gain_gap": gain_gap, "usd_gap": usd_gap, "cuts": list(CUTS[1:]), "end_value": {str(L): x for L, x in t["end_value"].items()},
             "model": {"%s|%s" % ka: m for ka, m in t["model"].items()}, "loops": sorted(loops)}
    show(table)
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
