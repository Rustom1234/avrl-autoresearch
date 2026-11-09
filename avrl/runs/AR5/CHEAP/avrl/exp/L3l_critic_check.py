#!/usr/bin/env python3
"""Critic check for L3l (CPU only, no GPU, no ssh, no pip).

Independent of avrl/allocators.py: re-derives the threshold and greedy6 loops from the pool, then
compares them with runs/L3l/result.json. Also recomputes every number quoted in the two L3l ledger
entries, checks the gold-free feature construction against rollouts.parquet, and tests the
post-midpoint growth of the gap with a paired t statistic.
"""
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.dont_write_bytecode = True
AVRL = "/homes/rdubash/avrl_live/CHEAP/avrl"
POOL = f"{AVRL}/runs/p0/feats.parquet"
ROLL = f"{AVRL}/runs/p0/rollouts.parquet"
FEAT = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
STEP, BUDGET, SEEDS, RS = 16, 0.25, list(range(8)), [10, 20, 40]

R = json.load(open(f"{AVRL}/runs/L3l/result.json"))
E03 = json.load(open(f"{AVRL}/runs/E03/result.json"))
L3I = json.load(open(f"{AVRL}/runs/L3i/result.json"))

df = pd.read_parquet(POOL)
n = len(df)
X_all = df[FEAT].to_numpy(dtype=np.float64)
y_all = df["proxy_error"].to_numpy(dtype=np.int64)
agr = df["agreement"].to_numpy(dtype=np.float64)
nd = df["n_distinct"].to_numpy(dtype=np.int64)
print(f"pool n={n} proxy_errors={int(y_all.sum())} (result.json says {R['n_proxy_errors']})")
assert int(y_all.sum()) == R["n_proxy_errors"] and n == R["n_groups"]

# 1. Gold-free features: agreement and n_distinct must come from rollout answers only.
ro = pd.read_parquet(ROLL)
pid_col = df["pid"].to_numpy()
assert (pid_col == np.arange(n)).all()
grp = ro.groupby("pid")["answer"]
max_share = grp.agg(lambda s: s.value_counts().iloc[0] / len(s)).to_numpy()
n_uniq = grp.nunique().to_numpy()
print("agreement == max answer share from rollouts:", np.allclose(max_share, agr))
print("n_distinct == unique rollout answers:", (n_uniq == nd).all())
maj = df["majority"].astype(str).to_numpy()
ro_maj = ro["pid"].map(dict(zip(df["pid"], maj)))
print("rollout 'proxy' == (answer == majority):",
      bool(((ro["answer"].astype(str) == ro_maj).astype(int) == ro["proxy"].astype(int)).all()))
print("log_qlen == log(q_len):", np.allclose(df["log_qlen"].to_numpy(), np.log(df["q_len"].to_numpy())))
print("gold/majority_correct/pass_rate not in FEAT:", not set(FEAT) & {"gold", "majority_correct", "pass_rate"})
print("split value counts:", df["split"].value_counts().to_dict())


# 2. Independent loops. Budget per step by cumulative rounding (same function as the run, re-typed here).
def k_for(t, glen):
    q = BUDGET * glen
    return math.floor(q * (t + 1)) - math.floor(q * t)


def run(policy, r, seed):
    perm = np.random.default_rng(1000 + seed).permutation(n)
    Xtr, ytr = [], []
    model, last = None, 0
    caught, calls, cum, refits = 0, 0, [], []
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = k_for(t, len(chunk))
        if policy == "threshold" or model is None:
            order = sorted(range(len(chunk)), key=lambda j: (agr[chunk[j]], -nd[chunk[j]]))
        else:
            p = model.predict_proba(X_all[chunk])[:, 1]
            order = list(np.argsort(-p, kind="stable"))
        sel = order[:k]
        assert len(set(sel)) == k
        caught += int(y_all[chunk[sel]].sum())  # gold used only for scoring
        calls += k
        cum.append(caught)
        for j in sel:  # gold label enters only for verified groups
            Xtr.append(X_all[chunk[j]])
            ytr.append(int(y_all[chunk[j]]))
        if policy == "greedy":
            m = len(ytr)
            if m >= r and m // r > last // r and len(set(ytr)) > 1:
                model = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(ytr))
                last = m
                refits.append(t)
    assert calls == round(BUDGET * n) == 750
    return caught, cum, refits


thr = [run("threshold", None, s) for s in SEEDS]
thr_caught = [c for c, _, _ in thr]
thr_cum = np.array([cu for _, cu, _ in thr], dtype=float)
print("\nthreshold caught per seed (independent):", thr_caught, "mean", np.mean(thr_caught))
print("threshold == E03 per seed:", thr_caught == E03["per_budget"]["0.25"]["arms"]["threshold"]["caught_per_seed"])
print("threshold == result.json:", thr_caught == R["threshold"]["caught_per_seed"],
      " cum curves equal:", np.array_equal(thr_cum, np.array(R["threshold"]["cum_caught"], dtype=float)))

gaps, firsts, ok_all = {}, {}, True
for r in RS:
    arm = [run("greedy", r, s) for s in SEEDS]
    g_caught = [c for c, _, _ in arm]
    g_cum = np.array([cu for _, cu, _ in arm], dtype=float)
    stored = R["greedy6"][str(r)]
    same_cum = np.array_equal(g_cum, np.array(stored["cum_caught"], dtype=float))
    refit_list = [rf for _, _, rf in arm]
    same_refit = all(x == stored["refit_steps"] for x in refit_list)
    ok_all &= same_cum and same_refit and g_caught == stored["caught_per_seed"]
    print(f"\nr={r}: caught {g_caught} mean {np.mean(g_caught):.3f} (stored mean {stored['caught_mean']:.3f})")
    print(f"  independent cum curves == result.json: {same_cum}; refit steps == result.json: {same_refit} "
          f"(n refits {len(refit_list[0])}, first {refit_list[0][0]})")
    gap = g_cum - thr_cum  # seeds x steps
    gaps[r] = gap
    gm = gap.mean(axis=0)
    print("  gap curve vs stored gap_mean_curve max abs diff:",
          float(np.max(np.abs(gm - np.array(stored["gap_mean_curve"])))))
    first_refit = refit_list[0][0]
    first_nz = [int(np.nonzero(gap[i])[0][0]) for i in range(len(SEEDS))]
    first_below = [int(np.nonzero(gap[i] < 0)[0][0]) for i in range(len(SEEDS))]
    firsts[r] = first_refit
    print("  first_nonzero per seed:", first_nz, "== stored:", first_nz == stored["first_nonzero_step_per_seed"])
    print("  first_below per seed:", first_below, "== stored:", first_below == stored["first_below_threshold_step_per_seed"])
    print("  zero through first refit step on every seed:", bool((gap[:, :first_refit + 1] == 0).all()))
    print(f"  gap mean at steps 4,20,50,93,150,187: " +
          ", ".join(f"{t}:{gm[t]:.3f}" for t in [4, 20, 50, 93, 150, 187]))
    frac = gm[93] / gm[187]
    lin = (93 - first_refit) / (187 - first_refit)
    print(f"  fraction of final gap at 93: {frac:.4f} (stored {stored['fraction_of_final_gap_at_mid']:.4f}); "
          f"linear-from-first-refit reference {lin:.3f}")
    post = gap[:, 187] - gap[:, 93]
    pre = gap[:, 93] - gap[:, first_refit]
    for name, d in [("post-mid growth (187 minus 93)", post), ("pre-mid growth (93 minus first refit)", pre)]:
        se = d.std(ddof=1) / math.sqrt(len(d))
        print(f"  {name}: mean {d.mean():.3f}, paired se {se:.3f}, t {d.mean() / se:.2f}")
    slopes = [(a, b, (gm[b] - gm[a]) / (b - a)) for a, b in [(5, 50), (50, 100), (100, 150), (150, 187)]]
    print("  slopes per step: " + ", ".join(f"{a}-{b}: {s:.3f}" for a, b, s in slopes))
    incr = int(sum(1 for t in range(first_refit + 1, 188) if gm[t] > gm[t - 1] + 1e-9))
    print(f"  steps after first refit where the mean gap moves up (bounce): {incr}")
    exc = []
    for i in range(len(SEEDS)):
        pos = np.nonzero(gap[i] > 0)[0]
        if len(pos):
            runs_, start = [], pos[0]
            for a, b in zip(pos[:-1], pos[1:]):
                if b != a + 1:
                    runs_.append((int(start), int(a)))
                    start = b
            runs_.append((int(start), int(pos[-1])))
            exc.append((i, runs_, int(gap[i].max())))
    print("  steps with gap > 0 (seed, runs, peak):", exc)

print("\nAll stored arms reproduced by the independent loop:", ok_all)


def paired(a, b):
    d = a - b
    se = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), se, d.mean() / se


g20, g10, g40 = (np.array(R["greedy6"][k]["caught_per_seed"], float) for k in ["20", "10", "40"])
thrc = np.array(thr_caught, float)
for name, (a, b) in {"r40 minus r20": (g40, g20), "r10 minus r20": (g10, g20),
                     "r40 minus threshold": (g40, thrc), "r20 minus threshold": (g20, thrc),
                     "r10 minus threshold": (g10, thrc)}.items():
    m, se, t = paired(a, b)
    print(f"{name}: mean {m:.3f} se {se:.3f} t {t:.2f}")
print("fraction of 750 calls per arm: all equal 750 (asserted)")
print("\nDone.")
