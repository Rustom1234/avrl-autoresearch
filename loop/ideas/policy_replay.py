#!/usr/bin/env python3
"""B2 analysis: is the strong planner worth calling where the cheap one fails, or where it succeeds?

    python3 loop/ideas/policy_replay.py --roots loop/runs/B1 loop/runs/B1F loop/runs/B2 [--skip lipo_HH_1]

Every planner's ideas were tested at every state against the same frozen incumbent, so the result of a routing policy
is read off the records (no estimate). Only continued-context ideas count. An idea that two planners proposed
(MiniLM cosine 0.90 or more, the engine's own duplicate rule) counts for both and is paid for once.
Questions Q1 to Q5 and the policies are fixed in loop/design/B2_PREREG.md.
"""
import argparse, glob, json, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", required=True)
ap.add_argument("--skip", nargs="*", default=["lipo_HH_1"])
ap.add_argument("--stage", default="s")
a = ap.parse_args()
PL = ("haiku", "sonnet", "opus", "fable")
PROM = 0.75


def outcomes(d):
    out = {}
    for f in sorted(glob.glob(str(d / (a.stage + "[0-9]*") / "ledger.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            iid = r.get("idea_id")
            if not iid or not iid.startswith("b"):
                continue
            o = out.setdefault(iid, {"ran": False, "z1": None, "accepted": False, "cost": 0.0})
            o["cost"] += sum(c.get("cost_usd_nominal") or 0 for c in r.get("model_calls") or [])
            o["ran"] = o["ran"] or bool(r.get("job_id"))
            if r.get("z1") is not None and o["z1"] is None:
                o["z1"] = r["z1"]
            o["accepted"] = o["accepted"] or r["decision"] == "accept"
    return out


# state -> list of ideas {key, planners:set, z1, prom, acc, cost, text}; state -> planner -> ask cost
S, ASK, UNDECIDED = defaultdict(list), defaultdict(dict), 0
for root in map(Path, a.roots):
    for d in sorted(p.parent for p in root.glob("*/ideas.json")):
        if d.name in a.skip:
            continue
        out = outcomes(d)
        for r in map(json.loads, open(d / "proposals.jsonl")):
            if r["ctx"] == "c" and r["ok"]:
                ASK[d.name][r["model"]] = r["cost_usd_nominal"]
        for i in json.loads((d / "ideas.json").read_text()):
            pl = {m for m, c, _ in i["sources"] if c == "c"}
            if not pl:
                continue
            o = out.get(i["id"])
            if o is None:
                UNDECIDED += 1
                continue
            S[d.name].append({"key": "%s/%s" % (root.name, i["id"]), "planners": pl, "z1": o["z1"], "ran": o["ran"],
                              "prom": o["z1"] is not None and o["z1"] >= PROM, "acc": o["accepted"],
                              "cost": o["cost"], "text": "%s. %s" % (i["title"], i["spec"])})

# ideas of one state that sit in different roots were pooled apart: merge near duplicates across roots here
multi = [s for s in S if len({i["key"].split("/")[0] for i in S[s]}) > 1]
if multi:
    from sentence_transformers import SentenceTransformer
    mini = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    for s in multi:
        E = mini.encode([i["text"] for i in S[s]], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        keep = []
        for k, i in enumerate(S[s]):
            j = next((n for n in keep if S[s][n]["key"].split("/")[0] != i["key"].split("/")[0]
                      and float(E[k] @ E[n]) >= 0.90), None)
            if j is None:
                keep.append(k)
            else:                       # same idea asked twice: one test is paid, either success counts
                h = S[s][j]
                h["planners"] |= i["planners"]
                h["prom"], h["acc"] = h["prom"] or i["prom"], h["acc"] or i["acc"]
                h["z1"] = h["z1"] if h["z1"] is not None else i["z1"]
                h["dup"] = True
        S[s] = [S[s][k] for k in keep]

states = sorted(s for s in S if all(m in ASK[s] for m in PL))
print("states with all four planners: %d (%s)" % (len(states), " ".join(states)))
print("ideas not decided yet (left out): %d" % UNDECIDED)


def arm(s, m):
    return [i for i in S[s] if m in i["planners"]]


def n_prom(s, ms):
    return sum(1 for i in S[s] if i["prom"] and i["planners"] & set(ms))


print("\nPer state: promising ideas (accepted) by planner, 10 ideas each")
print("| state | haiku | sonnet | opus | fable | mean z1 haiku | sonnet | opus | fable |\n|---|---|---|---|---|---|---|---|---|")
MZ = defaultdict(dict)
for s in states:
    c1, c2 = [], []
    for m in PL:
        A = arm(s, m)
        zs = [i["z1"] for i in A if i["z1"] is not None]
        MZ[s][m] = sum(zs) / len(zs) if zs else None
        c1.append("%d (%d) of %d" % (sum(i["prom"] for i in A), sum(i["acc"] for i in A), len(A)))
        c2.append("%.2f" % MZ[s][m] if zs else "-")
    print("| %s | %s | %s |" % (s, " | ".join(c1), " | ".join(c2)))
print("| all | %s | |" % " | ".join("%d (%d) of %d" % (sum(i["prom"] for s in states for i in arm(s, m)),
                                                         sum(i["acc"] for s in states for i in arm(s, m)),
                                                         sum(len(arm(s, m)) for s in states)) for m in PL))

print("\nQ1 and Q2: share of states where the strong planner finds at least one promising idea")
print("| strong planner | where haiku found none (Q1) | where haiku found one or more (Q2) | Q2 minus Q1 (95% interval) |\n|---|---|---|---|")
rng = random.Random(7)
for name, ms in (("sonnet", ["sonnet"]), ("opus", ["opus"]), ("fable", ["fable"]), ("opus or fable", ["opus", "fable"])):
    def shares(ss):
        fail = [s for s in ss if n_prom(s, ["haiku"]) == 0]
        succ = [s for s in ss if n_prom(s, ["haiku"]) > 0]
        q1 = (sum(n_prom(s, ms) > 0 for s in fail), len(fail))
        q2 = (sum(n_prom(s, ms) > 0 for s in succ), len(succ))
        return q1, q2
    q1, q2 = shares(states)
    diffs = []
    for _ in range(2000):
        b1, b2 = shares([rng.choice(states) for _ in states])
        if b1[1] and b2[1]:
            diffs.append(b2[0] / b2[1] - b1[0] / b1[1])
    diffs.sort()
    iv = "%.2f (%.2f to %.2f)" % (q2[0] / q2[1] - q1[0] / q1[1], diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]) if q1[1] and q2[1] and diffs else "-"
    print("| %s | %d of %d%s | %d of %d%s | %s |" % (name, q1[0], q1[1], " (%.2f)" % (q1[0] / q1[1]) if q1[1] else "",
                                                     q2[0], q2[1], " (%.2f)" % (q2[0] / q2[1]) if q2[1] else "", iv))


def policy(rule):
    """rule(s) -> list of planners called at state s. Returns ideas tested, promising, accepted, ask and test cost."""
    tot = {"calls": 0, "ideas": 0, "prom": 0, "acc": 0, "ask": 0.0, "test": 0.0}
    for s in states:
        ms = rule(s)
        I = [i for i in S[s] if i["planners"] & set(ms)]
        tot["calls"] += len(ms)
        tot["ideas"] += len(I)
        tot["prom"] += sum(i["prom"] for i in I)
        tot["acc"] += sum(i["acc"] for i in I)
        tot["ask"] += sum(ASK[s][m] for m in ms)
        tot["test"] += sum(i["cost"] for i in I)
    return tot


hp = {s: n_prom(s, ["haiku"]) for s in states}
POL = [("A always haiku", lambda s: ["haiku"]), ("B always sonnet", lambda s: ["sonnet"]),
       ("C always opus", lambda s: ["opus"]), ("D always fable", lambda s: ["fable"]),
       ("E cascade: fable where haiku found nothing", lambda s: ["haiku"] + (["fable"] if hp[s] == 0 else [])),
       ("F reverse: fable where haiku found something", lambda s: ["haiku"] + (["fable"] if hp[s] > 0 else [])),
       ("G reverse with sonnet", lambda s: ["haiku"] + (["sonnet"] if hp[s] > 0 else [])),
       ("E' cascade with opus", lambda s: ["haiku"] + (["opus"] if hp[s] == 0 else [])),
       ("F' reverse with opus", lambda s: ["haiku"] + (["opus"] if hp[s] > 0 else [])),
       ("all four everywhere", lambda s: list(PL))]
print("\nQ4: policies replayed on the %d states. USD per promising idea at the measured test price and at 0.05 and 1 USD per test" % len(states))
print("| policy | planner calls | ideas tested | promising | accepted | ask USD | test USD | USD per promising | at 0.05 | at 1.00 |\n|---|---|---|---|---|---|---|---|---|---|")
for name, rule in POL:
    t = policy(rule)
    per = lambda test: "%.2f" % ((t["ask"] + test) / t["prom"]) if t["prom"] else "none found"
    print("| %s | %d | %d | %d | %d | %.2f | %.2f | %s | %s | %s |" % (
        name, t["calls"], t["ideas"], t["prom"], t["acc"], t["ask"], t["test"], per(t["test"]),
        per(0.05 * t["ideas"]), per(1.0 * t["ideas"])))

print("\nQ5: mean z1 inside states, planner minus haiku (states where both have valid runs), sign-flip test")
for m in PL[1:]:
    d = [MZ[s][m] - MZ[s]["haiku"] for s in states if MZ[s][m] is not None and MZ[s]["haiku"] is not None]
    obs = sum(d) / len(d)
    hits = sum(abs(sum(x * rng.choice((-1, 1)) for x in d) / len(d)) >= abs(obs) - 1e-12 for _ in range(5000))
    print("  %s minus haiku: %+.2f over %d states (wins in %d), p = %.3f" % (m, obs, len(d), sum(x > 0 for x in d), hits / 5000))

print("\nSame idea from two planners (share of a planner's ideas that another planner also proposed, MiniLM 0.90)")
for m in PL:
    I = [i for s in states for i in arm(s, m)]
    print("  %s: %d of %d" % (m, sum(len(i["planners"]) > 1 for i in I), len(I)))
