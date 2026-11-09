#!/usr/bin/env python3
"""Critic check for L3q (CPU only, no GPU, no ssh, no pip).

Independent of avrl/allocators.py: re-derives the threshold and greedy6 mixing loops from the pool and
compares them with runs/L3q/result.json for e in {0, 0.25, 0.5}, all seeds 0-7. Recomputes the first three
bootstrap resamples (seeds 0-3) from scratch and checks them against the stored bootstrap_diffs. Recomputes
the percentile CIs and the verdict from the stored diffs. Reads only runs/p0 and runs/E03, runs/L3i, runs/L3l
and runs/L3q, all inside CHEAP.
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
FEAT = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
STEP, BUDGET, TOTAL, R_CADENCE = 16, 0.25, 750, 20

R = json.load(open(f"{AVRL}/runs/L3q/result.json"))
E03 = json.load(open(f"{AVRL}/runs/E03/result.json"))
L3I = json.load(open(f"{AVRL}/runs/L3i/result.json"))
L3L = json.load(open(f"{AVRL}/runs/L3l/result.json"))

df = pd.read_parquet(POOL)
N = len(df)
X = df[FEAT].to_numpy(dtype=np.float64)
AGR = df["agreement"].to_numpy(dtype=np.float64)
ND = df["n_distinct"].to_numpy(dtype=np.int64)
Y = df["proxy_error"].to_numpy(dtype=np.int64)
print(f"pool n={N} proxy_errors={int(Y.sum())} (result.json {R['n_proxy_errors']})")
assert int(Y.sum()) == R["n_proxy_errors"] and N == R["n_groups"]
print("feature columns exclude gold-derived columns:",
      not set(FEAT) & {"gold", "majority_correct", "pass_rate", "proxy_error"})


def run(mode, e, seed, ids=None):
    """mode 'threshold' or 'greedy'. Returns (caught, caught_uniform, calls, uniform_calls)."""
    if ids is None:
        X_, A_, D_, Y_ = X, AGR, ND, Y
    else:
        X_, A_, D_, Y_ = X[ids], AGR[ids], ND[ids], Y[ids]
    # every per-group array below is indexed by positions of this (possibly resampled) pool
    n = len(Y_)
    U = int(round(e * TOTAL))
    perm = np.random.default_rng(1000 + seed).permutation(n)
    rng = np.random.default_rng(2000 + seed)
    Xf, yf = [], []
    model, last_fit = None, 0
    cum_k, cum_u = 0, 0
    caught = caught_u = calls = uni_calls = 0
    for t in range(math.ceil(n / STEP)):
        gi = perm[t * STEP:(t + 1) * STEP]
        m = len(gi)
        q = BUDGET * m
        k = math.floor(q * (t + 1)) - math.floor(q * t)
        cum_k += k
        target = (U * cum_k) // TOTAL
        u = target - cum_u
        cum_u = target
        assert 0 <= u <= k
        if u > 0:
            up = [int(p) for p in rng.choice(m, size=u, replace=False)]
            ups = set(up)
            rest = [p for p in range(m) if p not in ups]
        else:
            up, rest = [], list(range(m))
        if model is None or mode == "threshold":
            order = sorted(rest, key=lambda p: (A_[gi[p]], -D_[gi[p]]))
        else:
            pr = model.predict_proba(X_[gi[rest]])[:, 1]
            order = [rest[i] for i in np.argsort(-pr, kind="stable")]
        chosen = up + order[:k - u]
        assert len(chosen) == k and len(set(chosen)) == k
        for p in chosen:
            g = gi[p]
            Xf.append(X_[g])
            yf.append(int(Y_[g]))
            caught += int(Y_[g])
        for p in up:
            caught_u += int(Y_[gi[p]])
        calls += k
        uni_calls += u
        n_v = len(yf)
        if mode == "greedy" and n_v >= R_CADENCE and n_v // R_CADENCE > last_fit // R_CADENCE and len(set(yf)) > 1:
            model = LogisticRegression(max_iter=1000).fit(np.array(Xf), np.array(yf))
            last_fit = n_v
    assert calls == TOTAL and uni_calls == U
    return caught, caught_u, calls, uni_calls


# 1. Main runs: every stored per-seed caught count, both arms, all three shares.
ok_main = True
for e_key in ["0.0", "0.25", "0.5"]:
    e = float(e_key)
    if e_key == "0.0":
        stored = {"mixed_threshold": R["controls_e0"]["mixed_threshold"]["caught_per_seed"],
                  "greedy6_mix": R["controls_e0"]["greedy6_mix"]["caught_per_seed"]}
    else:
        stored = {arm: R["arms"][e_key][arm]["caught_per_seed"] for arm in ["mixed_threshold", "greedy6_mix"]}
    for arm, mode in [("mixed_threshold", "threshold"), ("greedy6_mix", "greedy")]:
        got = [run(mode, e, s) for s in range(8)]
        c = [g[0] for g in got]
        cu = [g[1] for g in got]
        match = c == stored[arm]
        ok_main &= match
        print(f"e={e_key} {arm:15s} independent caught {c} mean {np.mean(c):.2f} "
              f"| stored {stored[arm]} match={match} | uniform caught {cu} "
              f"uniform calls {got[0][3]}")

# 2. Controls against the external references (E03 threshold, L3i and L3l greedy6 r=20).
e03_thr = E03["per_budget"]["0.25"]["arms"]["threshold"]["caught_per_seed"]
l3i_g = L3I["main_25"]["greedy6"]["caught_per_seed"]
l3l_g = L3L["greedy6"]["20"]["caught_per_seed"]
ctl_thr = R["controls_e0"]["mixed_threshold"]["caught_per_seed"]
ctl_g = R["controls_e0"]["greedy6_mix"]["caught_per_seed"]
print("e=0 threshold equals E03:", ctl_thr == e03_thr, e03_thr)
print("e=0 greedy6_mix equals L3l r=20:", ctl_g == l3l_g, "and L3i:", ctl_g == l3i_g)

# 3. Bootstrap: recompute the first three resamples from scratch (seeds 0-3), compare with stored diffs.
ok_boot = True
for b in range(3):
    ids = np.random.default_rng(10000 + b).choice(N, size=N, replace=True)
    for e_key in ["0.25", "0.5"]:
        e = float(e_key)
        g = [run("greedy", e, s, ids)[0] for s in range(4)]
        t = [run("threshold", e, s, ids)[0] for s in range(4)]
        d = float(np.mean([x - y for x, y in zip(g, t)]))
        stored = R["bootstrap_diffs"][e_key][b]
        same = abs(d - stored) < 1e-12
        ok_boot &= same
        print(f"resample b={b} e={e_key}: independent {d:+.4f} stored {stored:+.4f} match={same}")

# 4. Percentile CIs and verdict recomputed from the stored 100 resamples.
verdict_pass = []
for e_key in ["0.25", "0.5"]:
    x = np.asarray(R["bootstrap_diffs"][e_key], dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    stored_ci = R["bootstrap"][e_key]["ci95"]
    print(f"e={e_key}: B={len(x)} mean {x.mean():+.3f} CI [{lo:+.3f}, {hi:+.3f}] "
          f"stored CI {stored_ci} match={np.allclose([lo, hi], stored_ci)} upper<0={hi < 0}")
    if lo > 0:
        verdict_pass.append(e_key)
verdict = "PASS_CANDIDATE" if verdict_pass else "KILL"
print("recomputed verdict:", verdict, "stored verdict:", R["verdict"], "match:", verdict == R["verdict"])

# 5. Descriptive rates per call (from the stored means).
for e_key in ["0.25", "0.5"]:
    U = R["uniform_total_per_e"][e_key]
    tm = R["arms"][e_key]["mixed_threshold"]
    gm = R["arms"][e_key]["greedy6_mix"]
    rest_calls = R["verifier_calls_per_run"] - U
    print(f"e={e_key}: rest picks per call threshold {(tm['caught_mean'] - tm['caught_uniform_mean']) / rest_calls:.4f} "
          f"greedy6_mix {(gm['caught_mean'] - gm['caught_uniform_mean']) / rest_calls:.4f}; "
          f"uniform per call {tm['caught_uniform_mean'] / U:.4f} (chance {436 / 3000:.4f})")

print("\nSUMMARY main_match:", ok_main, "bootstrap_match:", ok_boot, "verdict_match:", verdict == R["verdict"])
