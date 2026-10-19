"""L4a independent audit: re-derive the selection protocol, ratios, CIs and the pre-registered tests straight from runs/L4a/sim_*.json.
Does not import l4a_summarize.py. CPU only. Usage: python exp/l4a_verify.py
"""
import json, os, re
import numpy as np

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "L4a")
JUDGES = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"]
MS = [250, 500, 1000]
BUD = ["0.1", "0.25", "0.5"]
NCALS = (10, 15, 25, 50, 75, 100, 150, 200, 250)
rng = np.random.default_rng(424242)  # differs from the summarizer's seed


def load(j, m, s):
    return json.load(open(f"{RUN}/sim_{j}_m{m}_s{s}.json"))


def is_fixed(n):
    if n == "thr_agreement":
        return True
    mm = re.match(r"(recal|seeded|split)_charged_(\d+)$", n)
    return mm is not None and int(mm.group(2)) in NCALS  # budget // 2 extras are sensitivity-only


def is_plain(n):
    return n == "greedy_online" or n.startswith("greedy_n") or n.startswith("greedy_seeded_")


def is_offset(n):
    return n.startswith("offset_")


def ratio_ci(l, f, B=10000):
    l = np.asarray(l); f = np.asarray(f)
    idx = rng.integers(0, len(l), size=(B, len(l)))
    r = l[idx].mean(1) / f[idx].mean(1)
    return l.mean() / f.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


cells = {}
for j in JUDGES:
    for m in MS:
        D = [load(j, m, s) for s in range(16)]
        assert all(d["m"] == m and d["subset_seed"] == s and d["n_sim"] == m for s, d in enumerate(D))
        assert all(d["n_policy_seeds"] == 8 and all(len(v["per_seed"]) == 8 for v in d["budget_0.5"].values()) for d in D)
        for b in BUD:
            names = list(D[0][f"budget_{b}"].keys())
            per = {n: np.array([np.mean(d[f"budget_{b}"][n]["per_seed"]) for d in D]) for n in names}
            B_abs = int(float(b) * m)
            fixed = [n for n in names if is_fixed(n)]
            plain = [n for n in names if is_plain(n)]
            offs = [n for n in names if is_offset(n)]
            pooled_fam = plain + offs
            res = {"B": B_abs, "n_fixed": len(fixed), "n_plain": len(plain), "n_off": len(offs)}
            for fold, (sel, rep) in {"A": (range(0, 8), range(8, 16)), "S": (range(8, 16), range(0, 8))}.items():
                sel = list(sel); rep = list(rep)
                fb = max(fixed, key=lambda n: per[n][sel].mean())
                pb = max(plain, key=lambda n: per[n][sel].mean())
                ob = max(offs, key=lambda n: per[n][sel].mean())
                lb = max(pooled_fam, key=lambda n: per[n][sel].mean())
                res[fold] = dict(fixed=fb, plain=pb, off=ob, pooled=lb,
                                 pooled_r=ratio_ci(per[lb][rep], per[fb][rep]),
                                 plain_r=ratio_ci(per[pb][rep], per[fb][rep]),
                                 off_r=ratio_ci(per[ob][rep], per[fb][rep]),
                                 off_vs_thr=ratio_ci(per[ob][rep], per["thr_agreement"][rep]),
                                 means=(per[lb][rep].mean(), per[fb][rep].mean(), per["thr_agreement"][rep].mean(),
                                        per["static"][rep].mean(), per["oracle"][rep].mean(), per["uniform"][rep].mean()),
                                 max_variant=max((per[n][rep].mean() / per[fb][rep].mean(), n) for n in pooled_fam))
            lA = per[res["A"]["pooled"]][8:16]; fA = per[res["A"]["fixed"]][8:16]
            lS = per[res["S"]["pooled"]][0:8]; fS = per[res["S"]["fixed"]][0:8]
            res["pooled16"] = ratio_ci(np.concatenate([lA, lS]), np.concatenate([fA, fS]))
            res["plain_all16_best"] = max((per[n].mean(), n) for n in plain)
            res["plain_all16_best_L2bgrid"] = max((per[n].mean(), n) for n in plain if n in
                                                 ("greedy_n10_R5", "greedy_n10_R10", "greedy_n15_R15", "greedy_n25_R10", "greedy_n25_R25", "greedy_online", "greedy_seeded_n25_R25"))
            cells[(j, m, b)] = res

print("judge | m | frac | B | foldA pooled ratio [CI] (sel pooled / sel fixed) | plain | offset | off/thr | swapped | pooled16 | max variant")
for (j, m, b), r in cells.items():
    A = r["A"]; S = r["S"]
    print(f"{j[:9]:9s} m={m:4d} {float(b)*100:3.0f}% B={r['B']:3d} | {A['pooled_r'][0]:.3f} [{A['pooled_r'][1]:.3f},{A['pooled_r'][2]:.3f}] ({A['pooled']} / {A['fixed']}) "
          f"| {A['plain_r'][0]:.3f} {A['plain']} | {A['off_r'][0]:.3f} {A['off']} | {A['off_vs_thr'][0]:.3f} [{A['off_vs_thr'][1]:.3f},{A['off_vs_thr'][2]:.3f}] "
          f"| {S['pooled_r'][0]:.3f} | {r['pooled16'][0]:.3f} [{r['pooled16'][1]:.3f},{r['pooled16'][2]:.3f}] | {A['max_variant'][0]:.3f} {A['max_variant'][1]}")
print()
print("held-out means (pooled sel, sel fixed, thr, static, oracle, uniform) at B<=125 and 250:")
for (j, m, b), r in cells.items():
    print(f"{j[:9]:9s} m={m:4d} {float(b)*100:3.0f}% B={r['B']:3d}", " ".join(f"{v:.1f}" for v in r["A"]["means"]))

# pre-registered tests
b125 = [(j, 250, "0.5") for j in JUDGES] + [(j, 500, "0.25") for j in JUDGES]
small = [(j, 250, b) for j in JUDGES for b in BUD] + [(j, 500, b) for j in JUDGES for b in ("0.1", "0.25")] + [(j, 1000, "0.1") for j in JUDGES]
assert len(small) == 12 and all(cells[k]["B"] <= 125 for k in small) and all(cells[k]["B"] == 125 for k in b125)
for fold, key in (("A", "A"), ("swapped", "S")):
    pass_n = sum(1 for k in b125 if cells[k][key]["pooled_r"][0] >= 1.02 and cells[k][key]["pooled_r"][1] > 1)
    floor_n = sum(1 for k in small if cells[k][key]["pooled_r"][0] >= 0.97)
    k1_n = sum(1 for k in b125 if cells[k][key]["max_variant"][0] >= 1.00)
    off_lt = sum(1 for k in small if cells[k][key]["off_r"][0] < 0.97)
    print(f"fold {fold}: B=125 pass cells {pass_n}/4; floor cells {floor_n}/12; B=125 cells with a variant >= 1.00: {k1_n}/4; offset < 0.97: {off_lt}/12")
pass16 = sum(1 for k in b125 if cells[k]["pooled16"][0] >= 1.02 and cells[k]["pooled16"][1] > 1)
print(f"pooled16: B=125 pass cells {pass16}/4; floor cells {sum(1 for k in small if cells[k]['pooled16'][0] >= 0.97)}/12")
print("min foldA pooled ratio over the 12 small-B cells:", min(cells[k]["A"]["pooled_r"][0] for k in small))
print("offset-vs-thr CI lower bound > 1 among 12 small-B cells (fold A):", [k for k in small if cells[k]["A"]["off_vs_thr"][1] > 1])
print("offset-vs-thr CI upper bound < 1 among 12 small-B cells (fold A):", [k for k in small if cells[k]["A"]["off_vs_thr"][2] < 1])
# interior check
corner = sum(1 for r in cells.values() if r["A"]["plain"] == "greedy_n3_R1")
print("cells where selected plain (fold A) is greedy_n3_R1:", corner, "of", len(cells))
print("cells where best random-start plain arm over all 16 subsets is greedy_n3_R1:", sum(1 for r in cells.values() if r["plain_all16_best"][1] == "greedy_n3_R1"),
      "; best is n3_R1 or n5_R1 or n10_R1 (R=1 column):", sum(1 for r in cells.values() if r["plain_all16_best"][1].endswith("_R1")))
print("cells where best L2b-grid arm over all 16 is greedy_n10_R5:", sum(1 for r in cells.values() if r["plain_all16_best_L2bgrid"][1] == "greedy_n10_R5"))
print("selected offset arm family counts (fold A): n0:", {n0: sum(1 for r in cells.values() if r["A"]["off"].startswith(f"offset_n{n0}_")) for n0 in (0, 5)},
      "lam:", {lam: sum(1 for r in cells.values() if r["A"]["off"].endswith(f"lam{lam}")) for lam in (1, 10, 100)})
# how often the pooled pick is plain vs offset, and which cells beat the fixed rule with CI lo > 1
print("pooled pick is plain (fold A):", [(k[0][:3], k[1], k[2]) for k, r in cells.items() if is_plain(r["A"]["pooled"])])
print("cells with fold-A pooled CI lower bound > 1:", [(k[0][:3], k[1], k[2], cells[k]["B"]) for k in cells if cells[k]["A"]["pooled_r"][1] > 1])


# ---- markdown tables for the ledger (written to runs/L4a/ledger_table.md); L2b reference recomputed from runs/L2b with the same fold-A protocol
L2B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "L2b")


def l2b_ratio(j, m, b):
    D = [json.load(open(f"{L2B}/sim_{j}_m{m}_s{s}.json")) for s in range(16)]
    names = list(D[0][f"budget_{b}"].keys())
    per = {n: np.array([np.mean(d[f"budget_{b}"][n]["per_seed"]) for d in D]) for n in names}
    fx = [n for n in names if n == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", n)]
    ln = [n for n in names if n == "greedy_online" or n.startswith("greedy_n") or n.startswith("greedy_seeded_")]
    fb = max(fx, key=lambda n: per[n][:8].mean()); lb = max(ln, key=lambda n: per[n][:8].mean())
    return per[lb][8:].mean() / per[fb][8:].mean()


lab = {"Qwen2.5-3B-Instruct": "3B", "Qwen2.5-7B-Instruct-AWQ": "7B-AWQ"}
rows = ["| judge | m | budget (B) | uniform | thr_agreement | selected fixed | selected plain | selected offset | selected pooled | static_out | oracle | pooled / fixed (95% CI) | swapped | pooled 16 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
curve = ["| B | judge | m, frac | L2b | L4a plain | L4a offset | L4a pooled (95% CI) | offset / thr_agreement (95% CI) | max variant (post hoc) |", "|---|---|---|---|---|---|---|---|---|"]
sortable = []
for (j, m, b), r in cells.items():
    D = [load(j, m, s) for s in range(16)]
    per = {n: np.array([np.mean(d[f"budget_{b}"][n]["per_seed"]) for d in D]) for n in D[0][f"budget_{b}"]}
    A = r["A"]; mp, mf, mt, ms_, mo, mu = A["means"]
    rows.append(f"| {lab[j]} | {m} | {float(b)*100:.0f}% (B={r['B']}) | {mu:.1f} | {mt:.1f} | {A['fixed']} {mf:.1f} | {A['plain']} {per[A['plain']][8:].mean():.1f} | {A['off']} {per[A['off']][8:].mean():.1f} | "
                f"{A['pooled']} {mp:.1f} | {ms_:.1f} | {mo:.1f} | {A['pooled_r'][0]:.3f} ({A['pooled_r'][1]:.3f} to {A['pooled_r'][2]:.3f}) | {r['S']['pooled_r'][0]:.3f} | {r['pooled16'][0]:.3f} |")
    sortable.append((r["B"], lab[j], m, b, l2b_ratio(j, m, b), A))
for B_, jl, m, b, l2, A in sorted(sortable, key=lambda t: (t[0], t[1])):
    curve.append(f"| {B_} | {jl} | m={m}, {float(b)*100:.0f}% | {l2:.3f} | {A['plain_r'][0]:.3f} | {A['off_r'][0]:.3f} | {A['pooled_r'][0]:.3f} ({A['pooled_r'][1]:.3f} to {A['pooled_r'][2]:.3f}) | "
                 f"{A['off_vs_thr'][0]:.3f} ({A['off_vs_thr'][1]:.3f} to {A['off_vs_thr'][2]:.3f}) | {A['max_variant'][0]:.3f} |")
open(os.path.join(RUN, "ledger_table.md"), "w").write("\n".join(rows) + "\n\n" + "\n".join(curve) + "\n")
print("wrote ledger_table.md")
