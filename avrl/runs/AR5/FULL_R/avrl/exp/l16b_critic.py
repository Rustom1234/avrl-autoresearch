"""L16b critic (CPU, separate code from l16b_summarize.py). Recomputes from the raw sim_*.json per_seed lists, with plain numpy and a different bootstrap generator (seed 12345, 20000 resamples,
resampling clusters with np.random.choice instead of the summarizer's integers table), the ratios the verdict uses, and compares them with result.json. Usage (from avrl/): l16b_critic.py
Also recomputes the descriptive rows (all variants, 25 percent budget) and the 10 / 50 percent jlflip50 rows, and re-derives the pass / kill flags from the recomputed numbers."""
import json, glob, re, sys
import numpy as np

R = json.load(open("runs/L16b/result.json")); rng = np.random.default_rng(12345)
STALES = ["jlflip50", "jlflip50_hi", "jlrev100", "fn80", "fp35", "flip25"]; TAGS = {"3B": "3B_q_corrupted", "7B-AWQ": "7B_q_corrupted"}


def arm(tag, st, bud, a):
    """(16, 8) array of per-seed hits, rows ordered by (split, subsample), read straight from the files."""
    rows = []
    for sp in range(4):
        for sub in range(4):
            d = json.load(open(f"runs/L16b/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json")); rows.append(d[f"budget_{bud}"][a]["per_seed"])
    x = np.array(rows, float); assert x.shape == (16, 8), x.shape
    return x


def ratio_ci(a, b, nb=20000):
    est = a.sum() / b.sum()  # equal sizes, so ratio of sums == ratio of means
    bs = np.empty(nb)
    for i in range(nb):
        k = rng.choice(16, 16, replace=True); bs[i] = a[k].sum() / b[k].sum()
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def split_ratios(a, b): return [a[4 * s:4 * s + 4].sum() / b[4 * s:4 * s + 4].sum() for s in range(4)]


bad = 0; out = {}
for rg, tag in TAGS.items():
    for st in STALES:
        for bud in ("0.1", "0.25", "0.5"):
            w, g, f, w25, w0 = (arm(tag, st, bud, k) for k in ("warm_online_200", "greedy_online", "recal_free_200", "warm_online_200_w25", "warm_online_200_w0"))
            for name, a, b in (("warm_over_cold", w, g), ("warm_over_free_stale", w, f), ("w25_over_cold", w25, g), ("w0_over_cold", w0, g)):
                est, lo, hi = ratio_ci(a, b); sr = split_ratios(a, b); c = R["cells"][f"{rg}|{st}"]["budgets"][bud][name]
                out[f"{rg}|{st}|{bud}|{name}"] = (est, lo, hi, sum(r > 1 for r in sr))
                if abs(est - c["ratio"]) > 1e-9: bad += 1; print("RATIO MISMATCH", rg, st, bud, name, est, c["ratio"])
                if abs(lo - c["ci_lo"]) > 0.004 or abs(hi - c["ci_hi"]) > 0.004: bad += 1; print("CI MISMATCH (> 0.004)", rg, st, bud, name, (lo, hi), (c["ci_lo"], c["ci_hi"]))
                if sum(r > 1 for r in sr) != c["n_splits_ratio_gt_1"]: bad += 1; print("SPLIT COUNT MISMATCH", rg, st, bud, name)
print("primary (jlflip50, 25%), recomputed from raw per_seed lists:")
pas = {}; kill = {}
for rg in TAGS:
    G = out[f"{rg}|jlflip50|0.25|warm_over_cold"]; F = out[f"{rg}|jlflip50|0.25|warm_over_free_stale"]
    X5 = out[f"{rg}|jlflip50|0.25|w25_over_cold"]; X0 = out[f"{rg}|jlflip50|0.25|w0_over_cold"]
    pas[rg] = G[0] >= 1.0 and F[0] >= 1.06 and F[1] > 1.0; kill[rg] = G[0] <= 0.97 and G[2] < 1.0
    print(f"  {rg}: warm/cold {G[0]:.4f} [{G[1]:.4f}, {G[2]:.4f}] splits>1 {G[3]}/4 | warm/free_stale {F[0]:.4f} [{F[1]:.4f}, {F[2]:.4f}] | w25/cold {X5[0]:.4f} [{X5[1]:.4f}, {X5[2]:.4f}] | w0/cold {X0[0]:.4f} [{X0[1]:.4f}, {X0[2]:.4f}] | pass {pas[rg]} kill {kill[rg]}")
outcome = "PASS" if all(pas.values()) else ("KILL" if any(kill.values()) else "INCONCLUSIVE")
print("recomputed outcome", outcome, "| summarizer outcome", R["verdict"]["outcome"], "| mismatches", bad)
print("\ndescriptive rows, 25% budget (ratio [CI] / splits above 1):")
for st in STALES:
    for rg in TAGS:
        print(f"  {st:12s} {rg:7s} " + " | ".join(f"{n} {out[f'{rg}|{st}|0.25|{n}'][0]:.3f} [{out[f'{rg}|{st}|0.25|{n}'][1]:.3f}, {out[f'{rg}|{st}|0.25|{n}'][2]:.3f}] {out[f'{rg}|{st}|0.25|{n}'][3]}/4" for n in ("warm_over_cold", "warm_over_free_stale", "w25_over_cold", "w0_over_cold")))
print("\njlflip50 at 10% and 50%:")
for bud in ("0.1", "0.5"):
    for rg in TAGS:
        print(f"  {bud:4s} {rg:7s} " + " | ".join(f"{n} {out[f'{rg}|jlflip50|{bud}|{n}'][0]:.3f} [{out[f'{rg}|jlflip50|{bud}|{n}'][1]:.3f}, {out[f'{rg}|jlflip50|{bud}|{n}'][2]:.3f}] {out[f'{rg}|jlflip50|{bud}|{n}'][3]}/4" for n in ("warm_over_cold", "warm_over_free_stale", "w25_over_cold", "w0_over_cold")))
sys.exit(0 if (bad == 0 and outcome == R["verdict"]["outcome"]) else 3)
