"""L4a adversarial critic (CPU). Independent re-derivation from runs/L4a/sim_*.json with its own bootstrap seed, plus checks of the ledger tables and counts.
Usage: l4a_critic.py"""
import json, re, glob, os
import numpy as np
RD = "/homes/rdubash/avrl_live/FULL_R/avrl/runs/L4a"; LOG = "/homes/rdubash/avrl_live/FULL_R/avrl/RESEARCH_LOG.md"
SETS = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}; MS = [250, 500, 1000]; BUD = ["0.1", "0.25", "0.5"]
NCALS = {10, 15, 25, 50, 75, 100, 150, 200, 250}
rng = np.random.default_rng(777); NB = 20000
H = {}; META = {}
nfiles = 0
for lab, name in SETS.items():
    for m in MS:
        for s in range(16):
            f = f"{RD}/sim_{name}_m{m}_s{s}.json"; d = json.load(open(f)); nfiles += 1
            assert d["m"] == m and d["subset_seed"] == s and d["n_sim"] == m and d["n_policy_seeds"] == 8 and d["label"] == "judge_error", f
            META[(lab, m, s)] = d["hits"]
            for b in BUD:
                bud = int(float(b) * m)
                for p, v in d[f"budget_{b}"].items():
                    a = np.array(v["per_seed"]); assert a.shape == (8,)
                    assert a.max() <= bud, (f, b, p, a.max(), bud)
                    assert abs(v["mean"] - a.mean()) < 1e-9
                    H.setdefault((lab, m, b), {}).setdefault(p, np.full((16, 8), np.nan))[s] = a
print("files", nfiles)
for k, h in H.items():
    for p, a in h.items(): assert not np.isnan(a).any(), (k, p)
# oracle check: oracle hits = min(budget, errors) per seed
for (lab, m, b), h in H.items():
    bud = int(float(b) * m)
    for s in range(16):
        e = META[(lab, m, s)]; assert (h["oracle"][s] == min(bud, e)).all(), (lab, m, b, s, h["oracle"][s], bud, e)
print("oracle = min(B, errors) in all 288 (cell, subset) combos")
fam = lambda p: ("thr" if p == "thr_agreement" else "fixed" if re.fullmatch(r"(recal|seeded|split)_charged_(\d+)", p) and int(p.rsplit("_", 1)[1]) in NCALS
                 else "plain" if (p == "greedy_online" or re.fullmatch(r"greedy_(seeded_)?n\d+_R\d+", p)) else "offset" if p.startswith("offset_") else None)
def boot(L, F):
    n = len(L); idx = rng.integers(0, n, (NB, n)); rb = L[idx].mean(1) / F[idx].mean(1)
    return L.mean() / F.mean(), np.percentile(rb, 2.5), np.percentile(rb, 97.5)
R = {}
for (lab, m, b), h in sorted(H.items()):
    bud = int(float(b) * m)
    fx = [p for p in h if fam(p) in ("thr", "fixed")]; pl = [p for p in h if fam(p) == "plain"]; of = [p for p in h if fam(p) == "offset"]
    mean = lambda p, sl: h[p][sl].mean()
    A, Bf = slice(0, 8), slice(8, 16)
    best = lambda c, sl: max(sorted(c), key=lambda p: mean(p, sl))
    bf = best(fx, A); bp = best(pl, A); bo = best(of, A); bg = best(pl + of, A)
    Fh = h[bf][Bf].mean(1)
    out = {"B": bud, "fixed": bf, "plain": bp, "offset": bo, "pooled": bg, "n_var": len(pl + of)}
    for k, p in (("plain", bp), ("offset", bo), ("pooled", bg)): out[k + "_r"] = boot(h[p][Bf].mean(1), Fh)
    out["offset_vs_thr"] = boot(h[bo][Bf].mean(1), h["thr_agreement"][Bf].mean(1))
    out["pooled_vs_thr"] = boot(h[bg][Bf].mean(1), h["thr_agreement"][Bf].mean(1))
    # swapped
    bfB = best(fx, Bf); bgB = best(pl + of, Bf); out["swapped"] = boot(h[bgB][A].mean(1), h[bfB][A].mean(1))
    out["pooled16"] = boot(np.concatenate([h[bg][Bf].mean(1), h[bgB][A].mean(1)]), np.concatenate([Fh, h[bfB][A].mean(1)]))
    vr = {p: h[p][Bf].mean() / Fh.mean() for p in pl + of}; mv = max(vr, key=vr.get); out["maxvar"] = (mv, vr[mv], sum(v >= 1.0 for v in vr.values()), vr[max((p for p in pl), key=vr.get)], vr[max((p for p in of), key=vr.get)])
    out["means"] = {k: mean(p, Bf) for k, p in (("thr", "thr_agreement"), ("fixed", bf), ("plain", bp), ("offset", bo), ("pooled", bg), ("static", "static"), ("oracle", "oracle"), ("uniform", "uniform"))}
    # exact (unrounded) pooled ratio
    out["pooled_exact"] = mean(bg, Bf) / mean(bf, Bf)
    R[(lab, m, b)] = out
print()
print("cell | B | fixed | plain | offset | pooled | pooled/fixed [CI] | plain | offset | swapped | p16 | off/thr [CI] | maxvar")
for (lab, m, b), o in sorted(R.items(), key=lambda kv: (kv[1]["B"], kv[0])):
    print(f"{lab} m{m} {float(b):.0%} | {o['B']} | {o['fixed']} | {o['plain']} | {o['offset']} | {o['pooled']} | {o['pooled_r'][0]:.4f} [{o['pooled_r'][1]:.3f},{o['pooled_r'][2]:.3f}] | {o['plain_r'][0]:.3f} | {o['offset_r'][0]:.3f} | {o['swapped'][0]:.3f} [{o['swapped'][1]:.3f},{o['swapped'][2]:.3f}] | {o['pooled16'][0]:.3f} [{o['pooled16'][1]:.3f},{o['pooled16'][2]:.3f}] | {o['offset_vs_thr'][0]:.3f} [{o['offset_vs_thr'][1]:.3f},{o['offset_vs_thr'][2]:.3f}] | {o['maxvar'][0]} {o['maxvar'][1]:.4f} nge1={o['maxvar'][2]} plainmax {o['maxvar'][3]:.3f} offmax {o['maxvar'][4]:.3f}")
# verdict
B125 = [k for k, o in R.items() if o["B"] == 125]; SMALL = [k for k, o in R.items() if o["B"] <= 125]
print(); print("B125 cells", len(B125), "small cells", len(SMALL))
for fold, key in (("A", "pooled_r"), ("swapped", "swapped"), ("pooled16", "pooled16")):
    npass = sum(R[k][key][0] >= 1.02 and R[k][key][1] > 1.0 for k in B125); nfloor = sum(R[k][key][0] >= 0.97 for k in SMALL)
    print(f"fold {fold}: pass cells {npass}/4, floor cells {nfloor}/12")
print("offset < 0.97 in", sum(R[k]["offset_r"][0] < 0.97 for k in SMALL), "of 12;  swapped-fold offset not computed here")
print("max variant >= 1.00 in B125 cells:", [(k, round(R[k]["maxvar"][1], 4)) for k in B125])
print("pooled exact ratios B125:", [(k, round(R[k]["pooled_exact"], 5)) for k in B125])
# alternative readings of PASS: best family = whichever of plain/offset has the higher fold-A ratio
alt = sum(max(R[k]["plain_r"][0], R[k]["offset_r"][0]) >= 1.02 and (R[k]["plain_r"][1] > 1 if R[k]["plain_r"][0] >= R[k]["offset_r"][0] else R[k]["offset_r"][1] > 1) for k in B125)
print("alt reading (max of plain, offset ratio per cell) pass cells:", alt)
alt2 = sum(R[k]["maxvar"][1] >= 1.02 for k in B125); print("post hoc max-variant >= 1.02 cells (no CI):", alt2)
# B-only statements: share of selected pooled arm that is plain / offset; lam
print("selected pooled arm type:", {k: ("plain" if R[k]["pooled"].startswith("greedy") else "offset") for k in R}.values().__len__(), sum(R[k]["pooled"].startswith("greedy") for k in R), "plain of 18")
print("selected plain == greedy_n3_R1:", sum(R[k]["plain"] == "greedy_n3_R1" for k in R), "greedy_n5_R1:", sum(R[k]["plain"] == "greedy_n5_R1" for k in R), "others:", [(k, R[k]["plain"]) for k in R if R[k]["plain"] not in ("greedy_n3_R1", "greedy_n5_R1")])
print("selected fixed == thr_agreement:", sum(R[k]["fixed"] == "thr_agreement" for k in R), [(k, R[k]["fixed"]) for k in R if R[k]["fixed"] != "thr_agreement"])
# free-label gap
print("gap captured (pooled - thr)/(static - thr):")
for (lab, m, b), o in sorted(R.items(), key=lambda kv: (kv[1]["B"], kv[0])):
    me = o["means"]; print(f"  B={o['B']} {lab} m{m}: {(me['pooled'] - me['thr']) / (me['static'] - me['thr']):.3f}  ({me['pooled'] - me['thr']:+.2f} of {me['static'] - me['thr']:.2f}); static/thr = {me['static'] / me['thr']:.3f}")
# best random-start grid arm on all 16, R=1 count
cnt_best_grid = {}; r1 = 0; n3 = 0; corner = 0
for (lab, m, b), h in H.items():
    grid = [p for p in h if re.fullmatch(r"greedy_n\d+_R\d+", p)]; bgp = max(sorted(grid), key=lambda p: h[p].mean()); cnt_best_grid[bgp] = cnt_best_grid.get(bgp, 0) + 1
    r1 += bgp.endswith("_R1"); n3 += bgp == "greedy_n3_R1"
print("best random-start grid arm on all 16:", cnt_best_grid, "R=1 arms:", r1)
# per-subset errors / mean errors and hits check
for lab in SETS:
    for m in MS: print(lab, m, "mean errors per subset", np.mean([META[(lab, m, s)] for s in range(16)]))

# ---- ledger table cross-check
txt = open(LOG).read(); sec = txt[txt.index("## L4a 2026-10-18"):]
rows1 = [l for l in sec.split("\n") if re.match(r"\| (3B|7B-AWQ) \| \d+ \| \d+% \(B=\d+\)", l)]
rows2 = [l for l in sec.split("\n") if re.match(r"\| \d+ \| (3B|7B-AWQ) \| m=\d+, \d+% \|", l)]
print("\nledger rows table1", len(rows1), "table2", len(rows2))
lab = lambda s: "3B" if s == "3B" else "7B"
nbad = 0
def chk(cond, msg):
    global nbad
    if not cond: nbad += 1; print("MISMATCH", msg)
for l in rows1:
    c = [x.strip() for x in l.strip("|").split("|")]; j, m, bb = lab(c[0]), int(c[1]), c[2]; frac = re.match(r"(\d+)% \(B=(\d+)\)", bb); b = {"10": "0.1", "25": "0.25", "50": "0.5"}[frac.group(1)]; o = R[(j, m, b)]; me = o["means"]
    chk(int(frac.group(2)) == o["B"], f"B {l}")
    ex = lambda s: (s.rsplit(" ", 1)[0], float(s.rsplit(" ", 1)[1]))
    chk(abs(float(c[3]) - me["uniform"]) < 0.051, f"uniform {l}"); chk(abs(float(c[4]) - me["thr"]) < 0.051, f"thr {l}")
    for col, key, nm in ((5, "fixed", "fixed"), (6, "plain", "plain"), (7, "offset", "offset"), (8, "pooled", "pooled")):
        n, v = ex(c[col]); chk(n == o[nm] and abs(v - me[key]) < 0.051, f"{nm} {l} -> {o[nm]} {me[key]:.2f}")
    chk(abs(float(c[9]) - me["static"]) < 0.051 and abs(float(c[10]) - me["oracle"]) < 0.051, f"static/oracle {l}")
    mm = re.match(r"([\d.]+) \(([\d.]+) to ([\d.]+)\)", c[11]); r = o["pooled_r"]
    chk(abs(float(mm.group(1)) - r[0]) < 0.0006 and abs(float(mm.group(2)) - r[1]) < 0.003 and abs(float(mm.group(3)) - r[2]) < 0.003, f"pooled ratio/CI {l} mine {r}")
    chk(abs(float(c[12]) - o["swapped"][0]) < 0.0006 and abs(float(c[13]) - o["pooled16"][0]) < 0.0006, f"swapped/p16 {l} mine {o['swapped'][0]:.4f} {o['pooled16'][0]:.4f}")
for l in rows2:
    c = [x.strip() for x in l.strip("|").split("|")]; B_, j, mf = int(c[0]), lab(c[1]), re.match(r"m=(\d+), (\d+)%", c[2]); m = int(mf.group(1)); b = {"10": "0.1", "25": "0.25", "50": "0.5"}[mf.group(2)]; o = R[(j, m, b)]
    chk(B_ == o["B"], f"B2 {l}")
    chk(abs(float(c[4]) - o["plain_r"][0]) < 0.0006 and abs(float(c[5]) - o["offset_r"][0]) < 0.0006, f"plain/offset {l} mine {o['plain_r'][0]:.4f} {o['offset_r'][0]:.4f}")
    mm = re.match(r"([\d.]+) \(([\d.]+) to ([\d.]+)\)", c[6]); r = o["pooled_r"]
    chk(abs(float(mm.group(1)) - r[0]) < 0.0006 and abs(float(mm.group(2)) - r[1]) < 0.003 and abs(float(mm.group(3)) - r[2]) < 0.003, f"pooled {l} mine {r}")
    mm = re.match(r"([\d.]+) \(([\d.]+) to ([\d.]+)\)", c[7]); r = o["offset_vs_thr"]
    chk(abs(float(mm.group(1)) - r[0]) < 0.0006 and abs(float(mm.group(2)) - r[1]) < 0.003 and abs(float(mm.group(3)) - r[2]) < 0.003, f"off/thr {l} mine {r}")
    chk(abs(float(c[8]) - o["maxvar"][1]) < 0.0006, f"maxvar {l} mine {o['maxvar'][1]:.4f}")
print("ledger table mismatches:", nbad)
# exact offset ratio at 3B m500 50%
o = R[("3B", 500, "0.5")]; print("3B m500 50% offset/fixed exact", o["means"]["offset"] / o["means"]["fixed"])

# ---- swapped-fold offset ratio and kill condition (b) under swapped fold
print()
nlow = 0; vals = []
for (lab, m, b), o in R.items():
    if o["B"] > 125: continue
    h = H[(lab, m, b)]; fx = [p for p in h if fam(p) in ("thr", "fixed")]; of = [p for p in h if fam(p) == "offset"]
    bfB = max(sorted(fx), key=lambda p: h[p][8:16].mean()); boB = max(sorted(of), key=lambda p: h[p][8:16].mean())
    r = h[boB][0:8].mean() / h[bfB][0:8].mean(); vals.append(round(r, 3)); nlow += r < 0.97
print("swapped-fold selected offset / selected fixed, 12 small cells:", sorted(vals), "below 0.97:", nlow)
nv = [round(max(H[k][p][0:8].mean() for p in H[k] if fam(p) in ("plain", "offset")) / max((H[k][p][8:16].mean() for p in H[k] if fam(p) in ("thr", "fixed")), default=1) , 3) for k in []]

# ---- exact ties at the top of each selection (fold A selects on 0-7; swapped selects on 8-15)
print("\nTie check (exact equal top mean among candidates):")
for (lab, m, b), o in sorted(R.items(), key=lambda kv: (kv[1]["B"], kv[0])):
    h = H[(lab, m, b)]
    for famname, cands in (("fixed", [p for p in h if fam(p) in ("thr", "fixed")]), ("plain", [p for p in h if fam(p) == "plain"]), ("offset", [p for p in h if fam(p) == "offset"]), ("pooled", [p for p in h if fam(p) in ("plain", "offset")])):
        for fold, sl, rep in (("A", slice(0, 8), slice(8, 16)), ("swapped", slice(8, 16), slice(0, 8))):
            sc = {p: h[p][sl].mean() for p in cands}; top = max(sc.values()); tied = sorted(p for p in cands if sc[p] == top)
            if len(tied) > 1:
                reps = {p: round(h[p][rep].mean(), 3) for p in tied}
                if len(set(reps.values())) > 1: print(f"  TIE with different held-out means: {lab} m{m} B={o['B']} {famname} fold {fold}: {reps}")
