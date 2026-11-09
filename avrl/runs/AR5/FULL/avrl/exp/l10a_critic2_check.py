"""L10a adversarial critic, second pass. Independent recomputation from the 24 raw xfit JSONs (plain loops, no finalizer code)."""
import json, math, os, sys, itertools
import numpy as np, pandas as pd
R = "/homes/rdubash/avrl_live/FULL/avrl/runs/L10a-xfit-fixed-comparator"
POOLS = [("L9c-weak-policy-tiebreak", "0.5B"), ("L8c-A-fresh-tiebreak", "1.5B")]; PROX = ["q3_4b", "rm"]; FR = ["0.1", "0.15", "0.2", "0.25", "0.3", "0.4"]
FIXED7 = ["agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff"]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
bad = 0
cells = {}
for pool, pol in POOLS:
    for px in PROX:
        G = np.zeros((6, 200)); K = np.zeros((6, 200)); Gp = np.zeros(6); Kp = np.zeros(6); A = np.zeros((6, 200)); Ap = np.zeros(6)
        Ip = np.zeros(6)
        for fi, f in enumerate(FR):
            d = json.load(open(f"{R}/{pool}/{px}/xfit_f{f}.json"))
            assert d["n_boot"] == 200 and d["seeds_per_resample"] == 4 and d["seeds_full_half"] == 8 and d["fixed7"] == FIXED7 and d["n_pool"] == 2000
            assert abs(d["budget_frac"] - float(f)) < 1e-12
            frozen_expect = "agree_acc_sdiff" if px == "q3_4b" else "agree_acc_soth"
            assert d["frozen"] == frozen_expect, (pool, px, f, d["frozen"])
            hs = d["halves"]
            for h in (0, 1):
                H = hs[h]; O = hs[1 - h]
                assert H["held_out_parity"] == h and H["selected_on_parity"] == 1 - h
                # selection table must equal the other half's full-half fixed-rule means (same pool, same budget, same seeds)
                sel = H["selection"]; om = O["full_half"]["held_mean_fixed7"]
                for p in FIXED7:
                    if abs(sel["mean_caught"][p] - om[p]) > 1e-9: bad += 1; P("SEL MISMATCH", pool, px, f, h, p, sel["mean_caught"][p], om[p])
                if sel["n_err"] != O["full_half"]["n_err"]: bad += 1; P("SEL n_err mismatch")
                if sel["budget"] != int(float(f) * 1000) or H["full_half"]["budget"] != int(float(f) * 1000): bad += 1; P("budget mismatch", pool, px, f, h, sel["budget"], H["full_half"]["budget"])
                ch = max(FIXED7, key=lambda p: om[p])  # first listed wins ties (python max returns first max)
                if ch != sel["ch"] or ch != H["full_half"]["ch"]: bad += 1; P("CH mismatch", pool, px, f, h, ch, sel["ch"])
                fh = H["full_half"]
                Gp[fi] += np.mean(fh["greedy_blend7"]); Kp[fi] += np.mean(fh[ch]); Ap[fi] += np.mean(fh["agree_acc"])
                hm = {p: np.mean(fh[p]) for p in FIXED7}; Ip[fi] += max(hm.values())
                assert len(H["resamples"]) == 200
                for i, r in enumerate(H["resamples"]):
                    assert r["boot"] == i and len(r["greedy_blend7"]) == 4 and r["budget"] == int(float(f) * 1000)
                    G[fi, i] += np.mean(r["greedy_blend7"]); K[fi, i] += np.mean(r[ch]); A[fi, i] += np.mean(r["agree_acc"])
                    # budget accounting: oracle <= budget, all arms same budget key
                    if max(r["oracle"]) > r["budget"] or max(r["greedy_blend7"]) > r["budget"] or max(r[ch]) > r["budget"]: bad += 1; P("caught > budget")
        cells[(pol, px)] = dict(G=G, K=K, Gp=Gp, Kp=Kp, A=A, Ap=Ap, Ip=Ip)
P("selection-table/other-half/CH/budget mismatches:", bad)
Cp = {c: float(np.mean(np.log(v["Gp"] / v["Kp"]))) for c, v in cells.items()}
Cb = {c: np.mean(np.log(v["G"] / v["K"]), axis=0) for c, v in cells.items()}
pooled_i = np.mean([Cb[c] for c in cells], axis=0); pooled_p = float(np.mean(list(Cp.values())))
lo, hi = np.percentile(pooled_i, [2.5, 97.5])
P("pooled plug-in %.4f boot mean %.4f CI [%.4f, %.4f] SD %.4f half-width %.4f" % (pooled_p, pooled_i.mean(), lo, hi, pooled_i.std(ddof=1), (hi - lo) / 2), "| SD ddof0 %.4f" % pooled_i.std(ddof=0))
for c in cells: P("cell", c, "plug %.4f" % Cp[c], "CI", np.round(np.percentile(Cb[c], [2.5, 97.5]), 4))
npos = sum(1 for v in Cp.values() if v > 0)
P("positive cells", npos)
PASS = pooled_p >= 0.02 and lo > 0 and npos >= 3; KILL = hi < 0.01
P("PASS", PASS, "KILL", KILL, "POWER-LIMITED", (hi - lo) / 2 > 0.03 and not PASS and not KILL, "plug in CI", lo <= pooled_p <= hi)
# sensitivity: mean-of-seed-then-log vs log-of-ratio ; unweighted vs sum-ratio pooling; median budget windows
alt = np.mean([np.log(v["Gp"].sum() / v["Kp"].sum()) for v in cells.values()]); P("alt: log of budget-summed ratio, cell-mean %.4f" % alt)
# CI with 2.5/97.5 using different percentile methods
for m in ("linear", "lower", "higher", "weibull", "hazen"):
    a, b = np.percentile(pooled_i, [2.5, 97.5], method=m); P("percentile method", m, "[%.4f, %.4f]" % (a, b))
# budget-by-budget pooled plug-in
P("by budget plug-in", [round(float(np.mean([np.log(v["Gp"][k] / v["Kp"][k]) for v in cells.values()])), 4) for k in range(6)])
# seed noise: how many resamples have G > K strictly
P("share of resamples with pooled C > 0: %.3f; > 0.02: %.3f" % ((pooled_i > 0).mean(), (pooled_i > 0.02).mean()))
# context contrasts
Ca = np.mean([np.log(v["Gp"] / v["Ap"]).mean() for v in cells.values()]); Ka = np.mean([np.log(v["Kp"] / v["Ap"]).mean() for v in cells.values()]); Ic = np.mean([np.log(v["Ip"] / v["Kp"]).mean() for v in cells.values()])
P("G/agree_acc %.4f  K/agree_acc %.4f  I/K %.4f" % (Ca, Ka, Ic))
# winner's curse
gaps = [(v["Ip"][k] - v["Kp"][k], (v["Ip"][k] - v["Kp"][k]) / v["Ip"][k]) for v in cells.values() for k in range(6)]
P("winner's curse mean gap caught %.4f, mean share %.4f (24 cell-budget pairs, halves summed)" % (np.mean([g[0] for g in gaps]), np.mean([g[1] for g in gaps])))
P("(entry says 1.30 per cell-budget summed over halves; per half = %.3f)" % (np.mean([g[0] for g in gaps]) / 2))
# idx sha identical across jobs
for pool, pol in POOLS:
    for h in (0, 1):
        sh = set(json.load(open(f"{R}/{pool}/{px}/xfit_f{f}.json"))["halves"][h]["idx_sha256"] for px in PROX for f in FR) if True else None
        P("idx sha unique count", pool, h, len(sh))
# hash across pools too
allsh = {h: set(json.load(open(f"{R}/{pool}/{px}/xfit_f{f}.json"))["halves"][h]["idx_sha256"] for pool, _ in POOLS for px in PROX for f in FR) for h in (0, 1)}
P("idx sha unique count across ALL 24 jobs per half", {h: len(s) for h, s in allsh.items()})
# error counts recompute from parquet
for pool, pol in POOLS:
    for px in PROX:
        df = pd.read_parquet(f"{R}/../{pool}/pool/{px}/feats_J.parquet")
        n0 = int(df[df.pid % 2 == 0].judge_error.sum()); n1 = int(df[df.pid % 2 == 1].judge_error.sum())
        d = json.load(open(f"{R}/{pool}/{px}/xfit_f0.25.json"))
        ok = d["halves"][0]["full_half"]["n_err"] == n0 and d["halves"][1]["full_half"]["n_err"] == n1
        P("parquet", pool, px, "rows", len(df), "pid==arange", bool((df.pid.values == np.arange(len(df))).all()), "errs half0/1", n0, n1, "match", ok, "cols", [c for c in df.columns][:40])
open(f"{R}/_critic2/check.txt", "w").write("\n".join(out) + "\n")

# ---- part 2: secondary claims of the entry and recorder note ----
out2 = []
def P2(*a):
    s = " ".join(str(x) for x in a); print(s); out2.append(s)
def load(pool, px, f): return json.load(open(f"{R}/{pool}/{px}/xfit_f{f}.json"))
def arm_name(j, h, w):
    hv = j["halves"][h]; return {"G": "greedy_blend7", "K": hv["selection"]["ch"], "A": "agree_acc", "F": j["frozen"], "I": hv["full_half"]["in_sample_best"], "B": "blend50_acc"}[w]
def plug(j, w): return sum(np.mean(j["halves"][h]["full_half"][arm_name(j, h, w)]) for h in (0, 1))
def bt(j, w): return np.array([sum(np.mean(j["halves"][h]["resamples"][i][arm_name(j, h, w)]) for h in (0, 1)) for i in range(200)])
J = {(pool, px, f): load(pool, px, f) for pool, _ in POOLS for px in PROX for f in FR}
def pooled(num, den):
    cp, cb = [], []
    for pool, _ in POOLS:
        for px in PROX:
            lp = [np.log(plug(J[pool, px, f], num) / plug(J[pool, px, f], den)) for f in FR]
            lb = [np.log(bt(J[pool, px, f], num) / bt(J[pool, px, f], den)) for f in FR]
            cp.append(np.mean(lp)); cb.append(np.mean(lb, axis=0))
    pb = np.mean(cb, axis=0); return float(np.mean(cp)), pb, cp
for nm, (n_, d_) in {"G/CH": ("G", "K"), "G/agree_acc": ("G", "A"), "G/frozen": ("G", "F"), "G/in-sample best": ("G", "I"), "CH/agree_acc": ("K", "A"), "I/CH": ("I", "K"), "blend50_acc/agree_acc": ("B", "A")}.items():
    pp, pb, cp = pooled(n_, d_); lo_, hi_ = np.percentile(pb, [2.5, 97.5])
    P2("%-22s plug %+.4f boot mean %+.4f bias %+.4f CI [%+.4f, %+.4f] SD %.4f pos cells %d  cells %s" % (nm, pp, pb.mean(), pb.mean() - pp, lo_, hi_, pb.std(ddof=1), sum(c > 0 for c in cp), np.round(cp, 3)))
pp, pb, cp = pooled("G", "K")
P2("plug-in percentile within its bootstrap distribution: %.1f%%" % (100 * (pb < pp).mean()))
lo_, hi_ = np.percentile(pb, [2.5, 97.5]); P2("basic CI [%+.4f, %+.4f]; bias-centred percentile [%+.4f, %+.4f]" % (2 * pp - hi_, 2 * pp - lo_, lo_ - (pb.mean() - pp), hi_ - (pb.mean() - pp)))
# budget windows
def window(ks):
    cp, cb = [], []
    for pool, _ in POOLS:
        for px in PROX:
            lp = [np.log(plug(J[pool, px, FR[k]], "G") / plug(J[pool, px, FR[k]], "K")) for k in ks]; lb = [np.log(bt(J[pool, px, FR[k]], "G") / bt(J[pool, px, FR[k]], "K")) for k in ks]
            cp.append(np.mean(lp)); cb.append(np.mean(lb, axis=0))
    pb = np.mean(cb, axis=0); return float(np.mean(cp)), pb
for nm, ks in (("15/20/25%", (1, 2, 3)), ("10/30/40%", (0, 4, 5))):
    pp_, pb_ = window(ks); l_, h_ = np.percentile(pb_, [2.5, 97.5]); P2("window", nm, "plug %+.4f boot mean %+.4f CI [%+.4f, %+.4f] SD %.4f" % (pp_, pb_.mean(), l_, h_, pb_.std(ddof=1)))
# ties at top of selection tables
nt = sum(1 for j in J.values() for h in (0, 1) if len(j["halves"][h]["selection"]["tied_with_ch"]) > 1); P2("selection tables with an exact tie at the top:", nt, "of 48 x ... counted over all 24 jobs x 2 halves =", len(J) * 2)
# resample SD of n_err; uniqueness fraction not stored
# error-set overlaps
dfs = {(pool, px): pd.read_parquet(f"{R}/../{pool}/pool/{px}/feats_J.parquet") for pool, _ in POOLS for px in PROX}
for pool, pol in POOLS:
    a, b = dfs[pool, "q3_4b"].judge_error.values.astype(bool), dfs[pool, "rm"].judge_error.values.astype(bool)
    P2("overlap within policy", pol, "inter/min = %.3f" % ((a & b).sum() / min(a.sum(), b.sum())))
for px in PROX:
    a, b = dfs[POOLS[0][0], px].judge_error.values.astype(bool), dfs[POOLS[1][0], px].judge_error.values.astype(bool)
    P2("overlap across policies, proxy", px, "inter/min = %.3f" % ((a & b).sum() / min(a.sum(), b.sum())))
for (pa, xa), (pb_, xb) in [(("L9c-weak-policy-tiebreak", "q3_4b"), ("L8c-A-fresh-tiebreak", "rm")), (("L9c-weak-policy-tiebreak", "rm"), ("L8c-A-fresh-tiebreak", "q3_4b"))]:
    a, b = dfs[pa, xa].judge_error.values.astype(bool), dfs[pb_, xb].judge_error.values.astype(bool); P2("overlap across policy+proxy", pa[:4], xa, pb_[:4], xb, "%.3f" % ((a & b).sum() / min(a.sum(), b.sum())))
# do the two policies share prompts?
P2("same questions across L9c / L8c-A:", bool((dfs[POOLS[0][0], "rm"].question.values == dfs[POOLS[1][0], "rm"].question.values).all()), "orig_pid equal:", bool((dfs[POOLS[0][0], "rm"].orig_pid.values == dfs[POOLS[1][0], "rm"].orig_pid.values).all()))
# frozen-arm provenance: are the pools disjoint from pids 0..1999 (where the tie-breaks / frozen arms were chosen)?
old = pd.read_parquet(f"{R}/../L5c-proxies/rm/feats_J.parquet")
for (pool, px), d in dfs.items(): P2("question overlap with pids 0..1999 pool (L5c/L7a):", pool[:4], px, len(set(d.question) & set(old.question)))
# learner determinism across seeds (does seed-averaging buy anything for G?)
j = J["L9c-weak-policy-tiebreak", "q3_4b", "0.25"]; P2("G across 8 seeds, L9c q3_4b f0.25 half0:", j["halves"][0]["full_half"]["greedy_blend7"])
open(f"{R}/_critic2/check.txt", "a").write("\n--- part 2 ---\n" + "\n".join(out2) + "\n")
