"""L15a critic (CPU, run from avrl/). Independent recomputation from the raw sim_*.json files, not from result.json.
Own bootstrap (seed 7, 20000 resamples), scipy ttest_rel over the 4 split means, own population loop.
(A) pre-registered cell (flip25, 25% budget, n_sim 2250) per rung: warm_stale/cold, warm_stale/free_stale, applies PASS / KILL literally.
(B) invariants over all 128 files: 8 finite per_seed values per arm, 0 <= hits <= min(B, errors in sim), STALE header matches the file name, refits on.
(C) side checks, not pre-registered: noise-level curve, share of the clean warm edge kept, and per-seed sign counts."""
import json, glob, os, sys
import numpy as np
from scipy import stats

rd = "runs/L15a"; BUDS = ("0.1", "0.25", "0.5"); rng = np.random.default_rng(7)
jl = lambda p: json.load(open(p))


def grid(tag, st):
    ds = [jl(f"{rd}/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json") for sp in range(4) for sub in range(4)]
    return ds, {b: {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in ds[0][f"budget_{b}"]} for b in BUDS}


def cmp(a, b, nb=20000):
    idx = rng.integers(0, 16, (nb, 16)); r = a[idx].mean((1, 2)) / b[idx].mean((1, 2))
    sa, sb = a.reshape(4, 4, 8).mean((1, 2)), b.reshape(4, 4, 8).mean((1, 2)); t = stats.ttest_rel(sa, sb)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), t.statistic, t.pvalue, (sa / sb), (a.mean((1,)) > b.mean((1,))).mean()


bad = 0; out = {}
for tag in ("3B_q_corrupted", "7B_q_corrupted"):
    for st in ("none", "flip25", "flip50", "lowlen"):
        ds, H = grid(tag, st)
        for k, d in enumerate(ds):
            if d["stale"] != st or d["warm_refit"] is not True or d["warm_oneclass_draws"]: bad += 1; print("HEADER", tag, st, k)
            for b in BUDS:
                B = int(float(b) * d["n_sim"])
                for p, v in d[f"budget_{b}"].items():
                    ps = np.array(v["per_seed"], float)
                    if len(ps) != 8 or not np.isfinite(ps).all() or ps.min() < 0 or ps.max() > min(B, d["hits"]): bad += 1; print("INVARIANT", tag, st, k, b, p)
                for p in ("uniform", "static", "greedy_online", "recal_free_200", "warm_online_200", "warm_online_200_w25"):
                    if p not in d[f"budget_{b}"]: bad += 1; print("MISSING ARM", tag, st, k, b, p)
        out[(tag, st)] = H
print(f"(B) invariants over 128 files: {bad} problems")

print("\n(A) pre-registered cell, flip25 / 25% / n_sim 2250")
verdicts = {}
for tag, rg in (("3B_q_corrupted", "3B"), ("7B_q_corrupted", "7B-AWQ")):
    H = out[(tag, "flip25")]["0.25"]; w, f, g, w25 = H["warm_online_200"], H["recal_free_200"], H["greedy_online"], H["warm_online_200_w25"]
    r1 = cmp(w, g); r2 = cmp(w, f); r3 = cmp(w25, g)
    pas = r1[0] >= 1.0 and r2[0] >= 1.06 and r2[1] > 1.0; kill = r1[0] <= 0.97 and r1[2] < 1.0; verdicts[rg] = (pas, kill)
    print(f"{rg}: warm_stale/cold {r1[0]:.4f} [{r1[1]:.4f}, {r1[2]:.4f}] split t {r1[3]:.2f} p {r1[4]:.3f} per-split {np.round(r1[5], 3).tolist()} clusters>1 {r1[6]:.2f}")
    print(f"     warm_stale/free_stale {r2[0]:.4f} [{r2[1]:.4f}, {r2[2]:.4f}] split t {r2[3]:.2f} p {r2[4]:.3f} per-split {np.round(r2[5], 3).tolist()}")
    print(f"     w25/cold {r3[0]:.4f} [{r3[1]:.4f}, {r3[2]:.4f}] ; means warm {w.mean():.2f} cold {g.mean():.2f} free {f.mean():.2f} w25 {w25.mean():.2f}; PASS clause {pas} KILL clause {kill}")
    pc = (w - g).mean(1); print(f"     paired cluster diffs warm_stale - cold: {int((pc > 0).sum())} of 16 positive; seed-level (mean over clusters) positive in {int(((w - g).mean(0) > 0).sum())} of 8 seeds")
print("literal verdict:", "PASS" if all(p for p, _ in verdicts.values()) else ("KILL" if any(k for _, k in verdicts.values()) else "INCONCLUSIVE"))

print("\n(C) side checks (not pre-registered), 25% budget, share of the clean warm edge over cold kept by warm_stale")
for tag, rg in (("3B_q_corrupted", "3B"), ("7B_q_corrupted", "7B-AWQ")):
    c = out[(tag, "none")]["0.25"]["greedy_online"].mean()
    for st in ("none", "lowlen", "flip25", "flip50"):
        H = out[(tag, st)]["0.25"]; w = H["warm_online_200"].mean(); f = H["recal_free_200"].mean(); x = H["warm_online_200_w25"].mean()
        e0 = out[(tag, "none")]["0.25"]["warm_online_200"].mean() - c
        print(f"{rg} {st:6s}: warm-cold {w - c:+7.1f} errors (share of clean edge {((w - c) / e0):+.2f}), free_stale-cold {f - c:+7.1f}, w25-cold {x - c:+7.1f}")
