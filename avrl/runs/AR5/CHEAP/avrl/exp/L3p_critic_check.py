#!/usr/bin/env python3
"""L3p critic check: independent re-derivation of the three arms and the bootstrap, without importing allocators.py.

Re-implements, from the spec text only: the threshold order (agreement ascending, n_distinct descending, stable ties in
chunk order), the per-step count (budget x chunk length, an integer here: 4 per full step, 2 on the last 8-group step),
the cold start, the refit cadence (refit when the verified count crosses a multiple of 20, both classes present), the
unscaled and StandardScaler logistic models (max_iter=1000), and the stable descending rank by predicted error.
Recomputes greedy6s, greedy6u and threshold caught counts per seed (seeds 0-7), the paired admitted differences, and the
paired bootstrap (seeds 0-3, B=100, default_rng(10000+b), per-resample denominator and fixed 436). Compares against
runs/L3p/result.json and runs/L3i/result.json. Prints per-feature scale. Writes runs/L3p/critic_check.txt.
Gold is used only for scoring and as the label of verified groups. CPU only, read-only on the pool.
"""
import math
import sys
import warnings

sys.dont_write_bytecode = True
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

POOL = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/p0/feats.parquet"
RES = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3p/result.json"
L3I = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3i/result.json"
OUT = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3p/critic_check.txt"
FEATS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
STEP = 16
BUDGET = 0.25
REFIT = 20

lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def run(X, agree, nd, y, arm, seed):
    """arm: 'threshold', 'greedy_unscaled' or 'greedy_scaled'. Returns (caught, calls)."""
    n = len(y)
    perm = np.random.default_rng(1000 + seed).permutation(n)
    model = None
    last_fit = 0
    Xv, yv = [], []
    caught = 0
    calls = 0
    for t in range(math.ceil(n / STEP)):
        ch = perm[t * STEP:(t + 1) * STEP]
        q = BUDGET * len(ch)
        assert abs(q - round(q)) < 1e-12, q
        k = int(round(q))
        if arm == "threshold" or model is None:
            order = sorted(range(len(ch)), key=lambda j: (agree[ch[j]], -nd[ch[j]]))
        else:
            p = model.predict_proba(X[ch])[:, 1]
            order = [int(j) for j in np.argsort(-p, kind="stable")]
        for j in order[:k]:
            g = ch[j]
            Xv.append(X[g])
            yv.append(int(y[g]))
            caught += int(y[g])
        calls += k
        n_v = len(yv)
        due = n_v >= REFIT and n_v // REFIT > last_fit // REFIT
        if arm != "threshold" and due and len(set(yv)) > 1:
            if arm == "greedy_scaled":
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
                    np.array(Xv), np.array(yv))
            else:
                model = LogisticRegression(max_iter=1000).fit(np.array(Xv), np.array(yv))
            last_fit = n_v
    assert calls == round(BUDGET * n), calls
    return caught, calls


def main():
    df = pd.read_parquet(POOL)
    assert len(df) == 3000
    X = df[FEATS].to_numpy(dtype=np.float64)
    agree = df["agreement"].to_numpy(dtype=np.float64)
    nd = df["n_distinct"].to_numpy(dtype=np.int64)
    y = df["proxy_error"].to_numpy(dtype=np.int64)
    n_err = int(y.sum())
    res = json.load(open(RES))
    l3i = json.load(open(L3I))

    say("L3p critic check (independent re-derivation; no import of allocators.py)")
    say(f"pool groups={len(y)} proxy_errors={n_err}")
    say("feature scale on the full pool (mean, std):")
    for c in FEATS:
        say(f"  {c:14s} mean {df[c].mean():12.4f}  std {df[c].std(ddof=0):12.4f}")

    seeds = list(range(8))
    thr = [run(X, agree, nd, y, "threshold", s) for s in seeds]
    g6u = [run(X, agree, nd, y, "greedy_unscaled", s) for s in seeds]
    g6s = [run(X, agree, nd, y, "greedy_scaled", s) for s in seeds]
    say("")
    say("per-seed caught, seeds 0-7 (recomputed):")
    say(f"  threshold  {[c for c, _ in thr]}")
    say(f"  greedy6u   {[c for c, _ in g6u]}")
    say(f"  greedy6s   {[c for c, _ in g6s]}")
    say(f"  means: threshold {np.mean([c for c, _ in thr]):.2f}, greedy6u {np.mean([c for c, _ in g6u]):.2f}, "
        f"greedy6s {np.mean([c for c, _ in g6s]):.2f}")

    stored = res["main_25"]
    ok_thr = [c for c, _ in thr] == stored["threshold"]["caught_per_seed"]
    ok_g6u = [c for c, _ in g6u] == l3i["main_25"]["greedy6"]["caught_per_seed"]
    ok_g6s = [c for c, _ in g6s] == stored["greedy6s"]["caught_per_seed"]
    say(f"match stored result.json: threshold {ok_thr}, greedy6s {ok_g6s}; match L3i greedy6 per seed: greedy6u {ok_g6u}")

    adm_t = [(n_err - c) / n_err for c, _ in thr]
    adm_s = [(n_err - c) / n_err for c, _ in g6s]
    paired = [s_ - t_ for s_, t_ in zip(adm_s, adm_t)]
    say(f"paired admitted (greedy6s - threshold), seeds 0-7: {[round(x, 6) for x in paired]}")
    say(f"  mean seeds 0-7 {np.mean(paired):+.6f}; all 8 negative: {all(x < 0 for x in paired)}")
    say(f"  seed SE {np.std(paired, ddof=1) / math.sqrt(len(paired)):.6f}, paired t "
        f"{np.mean(paired) / (np.std(paired, ddof=1) / math.sqrt(len(paired))):+.2f}")

    n = len(y)
    diffs_chk, fixed_chk = [], []
    for b in range(100):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        Xb, ab, nb, yb = X[ids], agree[ids], nd[ids], y[ids]
        eb = int(yb.sum())
        dd, ff = [], []
        for s in range(4):
            ct, _ = run(Xb, ab, nb, yb, "threshold", s)
            cg, _ = run(Xb, ab, nb, yb, "greedy_scaled", s)
            dd.append((eb - cg) / eb - (eb - ct) / eb)
            ff.append((ct - cg) / n_err)
        diffs_chk.append(float(np.mean(dd)))
        fixed_chk.append(float(np.mean(ff)))
    stored_diffs = res["metric"]["greedy6s_minus_threshold_admitted_per_resample"]
    max_dev = max(abs(a - b) for a, b in zip(diffs_chk, stored_diffs))
    lo, hi = np.percentile(diffs_chk, [2.5, 97.5])
    lo_f, hi_f = np.percentile(fixed_chk, [2.5, 97.5])
    say("")
    say(f"bootstrap recomputed, B=100, seeds 0-3: max |diff - stored| per resample = {max_dev:.2e}")
    say(f"  primary CI95 (per-resample denominator): [{lo:.6f}, {hi:.6f}], resamples below 0: "
        f"{sum(x < 0 for x in diffs_chk)}/100")
    say(f"  stored CI95: {[round(x, 6) for x in res['metric']['bootstrap']['ci95']]}")
    say(f"  fixed-436 CI95: [{lo_f:.6f}, {hi_f:.6f}]; stored {[round(x, 6) for x in res['metric']['bootstrap_fixed_denominator_436']['ci95']]}")
    say(f"  verdict by primary rule (upper bound < 0): {'PASS' if hi < 0 else 'KILL'}; stored verdict {res['verdict']}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run(X, agree, nd, y, "greedy_scaled", 0)
        conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    say(f"ConvergenceWarnings in one greedy6s seed-0 run (recomputed): {conv}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
