"""L2b independent audit: re-derive the selection protocol and ratios straight from runs/L2b/sim_*.json.
Does not import l2b_summarize.py. CPU only. Usage: python exp/l2b_verify.py
"""
import json, glob, re, os
import numpy as np

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "L2b")
JUDGES = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"]
MS = [250, 500, 1000]
BUD = ["0.1", "0.25", "0.5"]
rng = np.random.default_rng(12345)  # differs from the summarizer's seed


def load(j, m, s):
    return json.load(open(f"{RUN}/sim_{j}_m{m}_s{s}.json"))


def is_fixed(name):
    return name == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", name) is not None


def is_learned(name):
    return name == "greedy_online" or name.startswith("greedy_n") or name.startswith("greedy_seeded_")


def boot(vals, B=10000):
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    return np.percentile(vals[idx].mean(1), [2.5, 97.5])


def ratio_ci(l, f):
    l = np.asarray(l); f = np.asarray(f)
    idx = rng.integers(0, len(l), size=(10000, len(l)))
    r = l[idx].mean(1) / f[idx].mean(1)
    return l.mean() / f.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


out = {}
rows = []
for j in JUDGES:
    for m in MS:
        D = [load(j, m, s) for s in range(16)]
        assert all(d["m"] == m and d["subset_seed"] == s and d["n_sim"] == m for s, d in enumerate(D))
        assert all(d["n_policy_seeds"] == 8 for d in D)
        for b in BUD:
            names = list(D[0][f"budget_{b}"].keys())
            # per-subset mean over policy seeds
            per = {n: np.array([np.mean(d[f"budget_{b}"][n]["per_seed"]) for d in D]) for n in names}
            B_abs = int(round(float(b) * m))
            fixed = [n for n in names if is_fixed(n)]
            learned = [n for n in names if is_learned(n)]
            res = {}
            for fold, (sel, rep) in {"A": (range(0, 8), range(8, 16)), "B": (range(8, 16), range(0, 8))}.items():
                sel = list(sel); rep = list(rep)
                fb = max(fixed, key=lambda n: per[n][sel].mean())
                lb = max(learned, key=lambda n: per[n][sel].mean())
                r, lo, hi = ratio_ci(per[lb][rep], per[fb][rep])
                res[fold] = dict(fixed=fb, learned=lb, ratio=r, lo=lo, hi=hi,
                                 hits_l=per[lb][rep].mean(), hits_f=per[fb][rep].mean())
            # pooled: A reps (8..15) plus B reps (0..7)
            lA = per[res["A"]["learned"]][8:16]; fA = per[res["A"]["fixed"]][8:16]
            lB = per[res["B"]["learned"]][0:8]; fB = per[res["B"]["fixed"]][0:8]
            pooled = np.concatenate([lA, lB]).mean() / np.concatenate([fA, fB]).mean()
            # fixed-family curve over n on all 16 subsets
            curve = {n: per[n].mean() for n in fixed}
            # reference learned arms on held-out 8..15
            ref = {n: per[n][8:16].mean() for n in ["greedy_online", "greedy_n10_R5", "greedy_n25_R25"] if n in per}
            res.update(B_abs=B_abs, pooled=pooled, curve=curve, ref=ref, thr=per["thr_agreement"][8:16].mean(),
                       static=per["static"][8:16].mean(), oracle=per["oracle"][8:16].mean(),
                       uniform=per["uniform"][8:16].mean())
            out[(j, m, b)] = res
            rows.append((j, m, b, B_abs, res))

print("judge | m | budget | B_abs | foldA ratio (CI) | fixed / learned | foldB ratio | pooled | n_fixed_cands")
for j, m, b, B_abs, r in rows:
    A = r["A"]; Bf = r["B"]
    print(f"{j[:12]:12s} m={m:4d} {float(b)*100:3.0f}% B={B_abs:3d} | {A['ratio']:.3f} ({A['lo']:.3f} to {A['hi']:.3f}) | {A['fixed']} / {A['learned']} "
          f"| {Bf['ratio']:.3f} | {r['pooled']:.3f} | {len(r['curve'])}  hits {A['hits_l']:.1f} vs {A['hits_f']:.1f}")

# verdict re-derivation
print("\nPASS cells at m=500, 25% and 50%, ratio >= 1.03 and CI lo > 1 (fold A):")
npass = 0
for j in JUDGES:
    for b in ["0.25", "0.5"]:
        A = out[(j, 500, b)]["A"]
        ok = A["ratio"] >= 1.03 and A["lo"] > 1
        npass += ok
        print(" ", j, b, f"{A['ratio']:.3f} lo {A['lo']:.3f}", ok)
print("n pass:", npass)
print("\nKILL cells (ratio <= 1.00; m in 250,500; budgets 25,50):")
nk = 0
for j in JUDGES:
    for m in [250, 500]:
        for b in ["0.25", "0.5"]:
            A = out[(j, m, b)]["A"]
            ok = A["ratio"] <= 1.00
            nk += ok
            print(" ", j, m, b, f"{A['ratio']:.3f}", ok)
print("n kill cells:", nk, "of 8 (kill needs 8)")

# Ratio against absolute budget B
print("\nratio (fold A) by absolute budget B, sorted:")
tab = []
for j, m, b, B_abs, r in rows:
    tab.append((B_abs, j[:12], m, float(b), r["A"]["ratio"], r["pooled"]))
for t in sorted(tab):
    print(f"  B={t[0]:4d} {t[1]:12s} m={t[2]:4d} frac={t[3]:.2f} ratio={t[4]:.3f} pooled={t[5]:.3f}")

# Fixed family truncation check at selected cells
print("\nfixed curve over n (mean over all 16 subsets x 8 seeds) at cells where fold-A fixed pick is a charged rule:")
for (j, m, b), r in out.items():
    if r["A"]["fixed"] != "thr_agreement":
        c = r["curve"]
        print(f"  {j[:10]} m={m} {b}: pick {r['A']['fixed']}; thr={c['thr_agreement']:.1f}")
        for fam in ["recal", "seeded", "split"]:
            ns = sorted(int(n.split('_')[-1]) for n in c if n.startswith(fam))
            print("    ", fam, [(n, round(c[f'{fam}_charged_{n}'], 1)) for n in ns])

# Full-n budget//2 cap check
print("\nlargest fixed n available per (m, budget):")
for m in MS:
    for b in BUD:
        c = out[(JUDGES[0], m, b)]["curve"]
        print(f"  m={m} budget={b} B={int(round(float(b)*m))} cap={int(round(float(b)*m))//2}: max n={max(int(n.split('_')[-1]) for n in c if '_charged_' in n)}")

# (n0,R) ordering check on all 16 subsets
print("\n(n0,R) monotone check: does n10_R5 beat all other learned variants on all-16 mean in every cell?")
cnt = 0; tot = 0
for (j, m, b), r in out.items():
    pass
for j in JUDGES:
    for m in MS:
        for b in BUD:
            D = [load(j, m, s) for s in range(16)]
            per = {n: np.mean([np.mean(d[f'budget_{b}'][n]['per_seed']) for d in D]) for n in D[0][f'budget_{b}'] if is_learned(n)}
            best = max(per, key=per.get)
            tot += 1; cnt += best == "greedy_n10_R5"
            if best != "greedy_n10_R5":
                print("   not n10_R5:", j[:10], m, b, best, {k: round(v, 1) for k, v in per.items()})
print(f"  n10_R5 is best of all learned (all 16 subsets) in {cnt} of {tot} cells")
