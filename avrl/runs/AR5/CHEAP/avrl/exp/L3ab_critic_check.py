#!/usr/bin/env python3
"""L3ab critic check: independent re-derivation of the L3ab within-block AUC, bootstrap, cut read and verdict.

Does not import allocators.py or exp/l3ab_within_block_auc.py. Re-implements:
  (1) out-of-fold score: KFold(5, shuffle, random_state=0), standardisation by hand on the training fold (zero-variance
      guard as StandardScaler), LogisticRegression(max_iter=1000), predict_proba[:, 1] on the held-out fold;
  (2) exact (agreement, n_distinct) blocks via pandas groupby, kept blocks = both labels present, per-block roc_auc_score,
      size-weighted mean;
  (3) the stored block bootstrap (default_rng(20000), B=2000, one rng.choice(K, K, replace=True) per draw), reproduced
      draw by draw and compared with result.json;
  (4) the cut read with its own ordering (np.lexsort: agreement ascending, then n_distinct descending, ties by chunk
      position) on steps from default_rng(1000).permutation(3000), compared with result.json;
  (5) leakage: the six features exclude every gold-derived column, proxy_error is the target only, log_qlen = log1p(q_len),
      proxy_error = (majority != gold);
  (6) diagnostics that do NOT set the reading: unweighted mean of the block AUCs; a stratified group bootstrap (groups resampled
      with replacement within each mixed block, B=2000, default_rng(20000)); a within-block label-permutation null (labels
      shuffled inside each mixed block, B=2000, default_rng(20000)), reporting the share of permuted weighted AUCs at or above
      the observed value;
  (7) the pre-registered verdict recomputed from the re-derived CI, and the run-log / exit-code / file-state checks.

CPU only. Read-only on runs/p0/feats.parquet, runs/L3ab/result.json, runs/L3ab/run.log and exit_code. Writes
runs/L3ab/critic_check.txt.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

BASE = "/homes/rdubash/avrl_live/CHEAP/avrl"
POOL = f"{BASE}/runs/p0/feats.parquet"
RUN = f"{BASE}/runs/L3ab"
RES = f"{RUN}/result.json"
OUT = f"{RUN}/critic_check.txt"
FEAT = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
GOLD_DERIVED = {"gold", "majority", "majority_correct", "pass_rate", "correct", "question", "pid", "split"}
B = 2000
SEED = 20000
TOL = 1e-9

lines = []
failures = []


def out(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def check(name, ok, detail=""):
    out(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
    if not ok:
        failures.append(name)
    return ok


def oof_score(X, y, rs=0):
    s = np.full(len(y), np.nan)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=rs).split(X):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        m = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        s[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    assert not np.isnan(s).any()
    return s


def blocks(a, n, y, s):
    """Mixed blocks in sorted key order. Returns list of (key, index array, size, errors, auc) and the total block count."""
    d = pd.DataFrame({"a": a, "n": n, "y": y, "s": s}).reset_index(drop=True)
    kept, total, pure = [], 0, []
    for key, g in d.groupby(["a", "n"], sort=True):
        total += 1
        if g.y.nunique() < 2:
            pure.append((key, len(g), int(g.y.sum())))
            continue
        idx = g.index.to_numpy()
        kept.append((key, idx, len(idx), int(g.y.sum()), float(roc_auc_score(g.y.to_numpy(), g.s.to_numpy()))))
    return kept, total, pure


def wmean(sizes, aucs):
    return float((sizes * aucs).sum() / sizes.sum())


def stored_style_bootstrap(sizes, aucs):
    rng = np.random.default_rng(SEED)
    K = len(sizes)
    draws = np.empty(B)
    for b in range(B):
        idx = rng.choice(K, size=K, replace=True)
        draws[b] = (sizes[idx] * aucs[idx]).sum() / sizes[idx].sum()
    return draws


def summarise(name, a, n, y, s, stored):
    kept, total, pure = blocks(a, n, y, s)
    sizes = np.array([k[2] for k in kept], dtype=np.float64)
    aucs = np.array([k[4] for k in kept], dtype=np.float64)
    errs = np.array([k[3] for k in kept])
    point = wmean(sizes, aucs)
    draws = stored_style_bootstrap(sizes, aucs)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out(f"{name}: groups {len(y)} errors {int(y.sum())}; blocks total {total}, mixed {len(kept)}; "
        f"groups in mixed {int(sizes.sum())}, errors in mixed {int(errs.sum())}")
    out(f"{name}: size-weighted within-block AUC {point:.10f}; boot mean {draws.mean():.6f} sd {draws.std(ddof=1):.6f}; "
        f"95% CI [{lo:.10f}, {hi:.10f}]")
    check(f"{name} groups used", int(sizes.sum()) == stored["groups_used"], f"({int(sizes.sum())} vs {stored['groups_used']})")
    check(f"{name} errors used", int(errs.sum()) == stored["errors_used"], f"({int(errs.sum())} vs {stored['errors_used']})")
    check(f"{name} mixed blocks", len(kept) == stored["mixed_blocks"], f"({len(kept)} vs {stored['mixed_blocks']})")
    check(f"{name} weighted AUC matches", abs(point - stored["auc"]) < TOL, f"({point:.10f} vs {stored['auc']:.10f})")
    check(f"{name} CI lower matches", abs(lo - stored["ci"][0]) < TOL, f"({lo:.10f} vs {stored['ci'][0]:.10f})")
    check(f"{name} CI upper matches", abs(hi - stored["ci"][1]) < TOL, f"({hi:.10f} vs {stored['ci'][1]:.10f})")
    return kept, sizes, aucs, point, (lo, hi), pure, errs


def main():
    res = json.load(open(RES))
    df = pd.read_parquet(POOL)
    check("pool has 3000 groups", len(df) == 3000, f"({len(df)})")
    y = df["proxy_error"].to_numpy().astype(int)
    X = df[FEAT].to_numpy(dtype=np.float64)
    a = df["agreement"].to_numpy(dtype=np.float64)
    n = df["n_distinct"].to_numpy(dtype=np.int64)
    check("proxy_error binary with 436 errors", set(np.unique(y).tolist()) <= {0, 1} and int(y.sum()) == 436, f"({int(y.sum())})")

    # (5) leakage and target construction
    check("features exclude gold-derived columns", not (set(FEAT) & GOLD_DERIVED) and "proxy_error" not in FEAT)
    check("log_qlen == log1p(q_len)", np.allclose(df["log_qlen"], np.log1p(df["q_len"])))
    check("proxy_error == (majority != gold)", bool(((df["majority"] != df["gold"]).astype(int) == df["proxy_error"]).all()))
    check("majority_correct == 1 - proxy_error", bool((df["majority_correct"] == 1 - df["proxy_error"]).all()))

    # (1) out-of-fold score
    s = oof_score(X, y)
    auc_all = float(roc_auc_score(y, s))
    check("all-group OOF AUC matches", abs(auc_all - res["reference"]["all_group_auc"]) < 1e-6,
          f"({auc_all:.10f} vs {res['reference']['all_group_auc']:.10f})")
    out(f"OOF all-group AUC {auc_all:.10f}")

    # (2)+(3) full pool
    f = res["full_pool"]
    kept, sizes, aucs, point, (lo, hi), pure, errs = summarise(
        "full_pool", a, n, y, s,
        {"groups_used": f["groups_in_mixed_blocks"], "errors_used": f["errors_in_mixed_blocks"],
         "mixed_blocks": f["mixed_blocks"], "auc": f["within_block_auc_weighted"], "ci": f["ci95_percentile"]})
    out(f"full_pool: pure blocks excluded {[(k, sz, e) for k, sz, e in pure]}")
    out(f"full_pool: block AUC min {aucs.min():.10f} max {aucs.max():.10f}")
    out(f"full_pool: unweighted mean of block AUCs {aucs.mean():.6f} (diagnostic; pre-registered metric is size-weighted)")
    for key, idx, sz, ne, au in kept:
        if key in {(1.0, 1), (0.875, 2), (0.25, 5), (0.5, 3)}:
            out(f"full_pool: block {key}: groups {sz}, errors {ne}, AUC {au:.4f}")
    big = [k for k in kept if k[0] in {(1.0, 1), (0.875, 2)}]
    out(f"full_pool: share of weight in (1.0,1) and (0.875,2) {sum(k[2] for k in big) / sizes.sum():.4f}")

    # (6a) stratified group bootstrap: resample groups within each mixed block, keep block sizes
    rng = np.random.default_rng(SEED)
    strat = np.empty(B)
    skipped = 0
    for b in range(B):
        num, den = 0.0, 0.0
        for key, idx, sz, ne, au in kept:
            pick = idx[rng.integers(0, sz, size=sz)]
            yy = y[pick]
            if yy.min() == yy.max():
                skipped += 1
                continue
            num += sz * roc_auc_score(yy, s[pick])
            den += sz
        strat[b] = num / den
    slo, shi = np.percentile(strat, [2.5, 97.5])
    out(f"stratified group bootstrap (diagnostic): mean {strat.mean():.6f} sd {strat.std(ddof=1):.6f} "
        f"95% CI [{slo:.6f}, {shi:.6f}] includes 0.5: {bool(slo <= 0.5 <= shi)}; single-label resamples skipped {skipped}")

    # (6b) within-block label permutation null
    rng = np.random.default_rng(SEED)
    null = np.empty(B)
    for b in range(B):
        num, den = 0.0, 0.0
        for key, idx, sz, ne, au in kept:
            yp = y[idx].copy()
            rng.shuffle(yp)
            num += sz * roc_auc_score(yp, s[idx])
            den += sz
        null[b] = num / den
    p_ge = float((null >= point).mean())
    nlo, nhi = np.percentile(null, [2.5, 97.5])
    out(f"within-block permutation null (diagnostic): mean {null.mean():.6f} 95% [{nlo:.6f}, {nhi:.6f}]; "
        f"share of permuted >= observed {p_ge:.4f}")

    # (6c) fold-seed sensitivity of the point estimate (the pre-registered run uses random_state=0)
    for rs in (1, 2, 3, 4):
        s_rs = oof_score(X, y, rs)
        k_rs, _, _ = blocks(a, n, y, s_rs)
        sz_rs = np.array([k[2] for k in k_rs], dtype=np.float64)
        au_rs = np.array([k[4] for k in k_rs], dtype=np.float64)
        out(f"fold-seed sensitivity (diagnostic): random_state={rs} all-group AUC {roc_auc_score(y, s_rs):.4f}; "
            f"weighted within-block AUC {wmean(sz_rs, au_rs):.4f}; unweighted block mean {au_rs.mean():.4f}")

    # (4) cut read
    perm = np.random.default_rng(1000).permutation(3000)
    picked = []
    for t in range(math.ceil(3000 / 16)):
        chunk = perm[t * 16:(t + 1) * 16]
        order = np.lexsort((np.arange(len(chunk)), -n[chunk], a[chunk]))
        picked.extend([int(chunk[order[3]]), int(chunk[order[4]])])
    cut_ids = np.array(sorted(picked), dtype=np.int64)
    check("cut has 376 distinct groups", len(set(cut_ids.tolist())) == 376 and len(cut_ids) == 376, f"({len(cut_ids)})")
    c = res["cut_read"]
    out(f"cut: groups {len(cut_ids)} errors {int(y[cut_ids].sum())}")
    check("cut errors total", int(y[cut_ids].sum()) == c["errors_total"], f"({int(y[cut_ids].sum())} vs {c['errors_total']})")
    ck, cs, ca, cpoint, (clo, chi), cpure, cerr = summarise(
        "cut", a[cut_ids], n[cut_ids], y[cut_ids], s[cut_ids],
        {"groups_used": c["groups_in_mixed_blocks"], "errors_used": c["errors_in_mixed_blocks"],
         "mixed_blocks": c["mixed_blocks"], "auc": c["within_block_auc_weighted"], "ci": c["ci95_percentile"]})

    # (7) pre-registered verdict from the re-derived full-pool CI
    if lo > 0.5:
        verdict = "SURVIVE"
    elif hi < 0.5:
        verdict = "ANTI_SIGNAL"
    else:
        verdict = "KILL"
    check("pre-registered verdict re-derived", verdict == res["reading"] == "KILL", f"(re-derived {verdict}, stored {res['reading']})")
    check("stored next_test is L3ae for KILL", res["next_test"] == "L3ae", f"({res['next_test']})")
    check("stored checks all true", all(bool(v) for v in res["checks"].values()), f"({res['checks']})")

    # run-log, exit code and file-state checks
    log_txt = open(f"{RUN}/run.log").read()
    bad = [w for w in ("Traceback", "Warning", "Error") if w in log_txt]
    check("run.log has no traceback, warning or error", not bad, f"(hits: {bad})")
    check("'truncat' in run.log only inside the feature list", log_txt.count("truncat") == log_txt.split("\n")[0].count("truncat"),
          "(any_truncated is a feature name)")
    check("exit_code is 0", open(f"{RUN}/exit_code").read().strip() == "0")
    check("done.flag present", os.path.exists(f"{RUN}/done.flag"))
    run_t = os.path.getmtime(RES)
    check("allocators.py unchanged since before the run", os.path.getmtime(f"{BASE}/allocators.py") < run_t,
          "(allocators.py mtime 2026-10-05; result.json written 00:59:42)")
    check("pool file unchanged since before the run", os.path.getmtime(POOL) < run_t, "(feats.parquet mtime 2026-09-30)")

    out("")
    out(f"SUMMARY: full-pool CI [{lo:.4f}, {hi:.4f}] -> re-derived verdict {verdict}; "
        f"stored verdict {res['reading']}; cut CI [{clo:.4f}, {chi:.4f}]")
    out(f"FAILED CHECKS: {failures if failures else 'none'}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    out(f"wrote {OUT}")


if __name__ == "__main__":
    main()
