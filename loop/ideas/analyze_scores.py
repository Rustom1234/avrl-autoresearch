#!/usr/bin/env python3
"""Pre-test scores against outcomes, in detail.

    python3 loop/ideas/analyze_scores.py --root loop/runs/B1

Sets: retro (ideas the E1 continuation tested against the slot-40 incumbent) and bank (run B1). For each scorer:
  valid run      does the score separate ideas that produced a valid run from those that did not (coder abort,
                 static check, crash)?
  promising      z of the first run at least 0.75, among all ideas and among ideas with a valid run only
  lift           share of the promising ideas that sit in the top 10, 20, 30, 50 percent of scores inside their loop
All rankings are made inside a loop (a scorer never compares ideas of different loops). Intervals: bootstrap over
loops, 2,000 draws, 2.5 and 97.5 percentiles.
"""
import argparse, glob, json, math, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
a = ap.parse_args()
root = Path(a.root).resolve()


def bank_truth():
    out = {}
    for d in sorted(p.parent for p in root.glob("*/ideas.json")):
        o = {}
        for f in sorted(glob.glob(str(d / "s*" / "ledger.jsonl"))):
            for l in open(f):
                r = json.loads(l)
                iid = r.get("idea_id")
                if not iid or not iid.startswith("b"):
                    continue
                x = o.setdefault(iid, {"ran": False, "z1": None, "accepted": False, "first": None, "runs": 0, "crash": False})
                if x["first"] is None:
                    x["first"] = r["decision"]
                if r.get("job_id"):
                    x["runs"] += 1
                if r.get("z1") is not None and x["z1"] is None:
                    x["z1"] = r["z1"]
                x["crash"] |= r["decision"] == "crash"
                x["accepted"] |= r["decision"] == "accept"
        for iid, x in o.items():
            if x["first"] is None or (x["first"] == "replicate" and x["runs"] < 2):
                continue
            x["ran"] = x["z1"] is not None
            out[(d.name, iid)] = x
    return out


def retro_truth():
    out = {}
    for f in sorted(glob.glob(str(root / "retro" / "*.json"))):
        for i in json.loads(Path(f).read_text()):
            out[(Path(f).stem, i["id"])] = {"ran": i["z1"] is not None, "z1": i["z1"], "accepted": i["accepted"],
                                            "votes": i["alpha"] - 1 - int(i["accepted"]) if i["z1"] is not None else i["alpha"] - 1}
    return out


def load_scores(name):
    sc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(root / "scores" / ("%s_*.jsonl" % name)))):
        for r in map(json.loads, open(f)):
            for iid, p in r["scores"].items():
                sc[r["model"]][(r["loop"], iid)].append(p)
    return {m: {k: sum(v) / len(v) for k, v in d.items()} for m, d in sc.items()}


def auc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")


def within(val, keys):
    """percentile rank of each key inside its loop (ties share the mean rank)"""
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    pr = {}
    for lp, kk in by.items():
        vs = sorted(val[k] for k in kk)
        for k in kk:
            lo = sum(v < val[k] for v in vs)
            hi = sum(v <= val[k] for v in vs)
            pr[k] = (lo + hi) / 2 / len(kk)
    return pr, by


def stats(val, truth, keys):
    pr, by = within(val, keys)
    prom = lambda k: truth[k]["ran"] and truth[k]["z1"] >= 0.75
    res = {"n": len(keys), "n_prom": sum(prom(k) for k in keys), "n_ran": sum(truth[k]["ran"] for k in keys)}
    res["auc_ran"] = auc([pr[k] for k in keys if truth[k]["ran"]], [pr[k] for k in keys if not truth[k]["ran"]])
    res["auc_prom"] = auc([pr[k] for k in keys if prom(k)], [pr[k] for k in keys if not prom(k)])
    ran = [k for k in keys if truth[k]["ran"]]
    pr2, _ = within(val, ran)
    res["auc_prom_ran"] = auc([pr2[k] for k in ran if prom(k)], [pr2[k] for k in ran if not prom(k)])
    for f in (0.1, 0.2, 0.3, 0.5):
        res["top%d" % (f * 100)] = sum(1 for k in keys if prom(k) and pr[k] > 1 - f) / max(1, res["n_prom"])
    return res


def report(name, truth, extra=None):
    sc = load_scores(name)
    if extra:
        sc.update(extra)
    if not sc or not truth:
        return
    common = None
    if all(m in sc for m in ("haiku", "sonnet", "opus")):
        common = set(sc["haiku"]) & set(sc["sonnet"]) & set(sc["opus"])
        sc["mean of the three"] = {k: (sc["haiku"][k] + sc["sonnet"][k] + sc["opus"][k]) / 3 for k in common}
    if all(m in sc for m in ("haiku", "sonnet")):
        sc["mean of Haiku and Sonnet"] = {k: (sc["haiku"][k] + sc["sonnet"][k]) / 2 for k in set(sc["haiku"]) & set(sc["sonnet"])}
    table("%s set" % name, sc, truth)
    if common and len(common) < max(len(v) for v in sc.values()):
        table("%s set, only the ideas scored by all three" % name, {m: {k: v[k] for k in common if k in v} for m, v in sc.items()}, truth)


def table(title, sc, truth):
    print("\n### %s" % title)
    print("| scorer | ideas | valid runs | promising | AUC valid run | AUC promising | 95% interval | AUC promising among valid runs "
          "| promising in top 10% | top 20% | top 30% | top 50% |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, val in sc.items():
        keys = [k for k in val if k in truth]
        if len(keys) < 20:
            continue
        s = stats(val, truth, keys)
        loops = sorted({k[0] for k in keys})
        by = defaultdict(list)
        for k in keys:
            by[k[0]].append(k)
        rng, bs = random.Random(3), []
        for _ in range(2000):
            draw = [rng.choice(loops) for _ in loops]
            kk, v2, t2 = [], {}, {}
            for n, lp in enumerate(draw):
                for k in by[lp]:
                    k2 = ("%s#%d" % (lp, n), k[1])
                    kk.append(k2)
                    v2[k2], t2[k2] = val[k], truth[k]
            pr, _ = within(v2, kk)
            pos = [pr[k] for k in kk if t2[k]["ran"] and t2[k]["z1"] >= 0.75]
            neg = [pr[k] for k in kk if not (t2[k]["ran"] and t2[k]["z1"] >= 0.75)]
            if pos and neg:
                # rank-sum form of the AUC, fast enough for 2,000 draws
                allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
                r, i, tot = 0.0, 0, 0.0
                while i < len(allv):
                    j = i
                    while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
                        j += 1
                    avg = (i + j) / 2 + 1
                    tot += avg * sum(x[1] for x in allv[i:j + 1])
                    i = j + 1
                bs.append((tot - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        bs.sort()
        ci = "%.2f to %.2f" % (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]) if len(bs) > 100 else "-"
        f = lambda x: "%.2f" % x if x == x else "-"
        print("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            m, s["n"], s["n_ran"], s["n_prom"], f(s["auc_ran"]), f(s["auc_prom"]), ci, f(s["auc_prom_ran"]),
            f(s["top10"]), f(s["top20"]), f(s["top30"]), f(s["top50"])))


rt = retro_truth()
report("retro", rt, {"three Haiku votes (the engine's own)": {k: float(v["votes"]) for k, v in rt.items()}})
six = {"bbb_HH_1", "caco_HH_1", "clear_HH_1", "lipo_HH_1", "p26_HH_1", "p32_HH_1"}
report("bank", bank_truth())
by = defaultdict(lambda: [0, 0, 0])
for (lp, _), v in rt.items():
    c = lp.split("_")[1]
    by[c][0] += 1
    by[c][1] += int(v["ran"] and v["z1"] >= 0.75)
    by[c][2] += int(v["accepted"])
print("\nretro set by cell (ideas, promising, accepted):", dict(by))
