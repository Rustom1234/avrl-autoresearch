"""L9a: cross-family prompt-difficulty check (CPU only, no generation).

Does E01's group proxy error (Qwen2.5 1.5B) carry proxy-error signal for E10 (OLMo-2 1B) beyond E10's
cheap features? Pre-registered in avrl/RESEARCH_LOG.md ("L9a pre-registration 2026-10-18").
Diagnostic upper bound: E01 needs another policy's rollouts, so this is not a label-free single-policy feature.

Usage:
  python l9a_crossfamily_difficulty.py --e01 <p0/feats.parquet> --e10 <E10/feats.parquet> \
      --boot 200 --boot_seed 123 --out <run_dir>
Writes <run_dir>/result.json (metric values, bootstrap draws, verdict) and prints a summary to stdout.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
EMB = [f"pc{i}" for i in range(16)]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def pooled_oof_auc(X, y):
    """Pooled out-of-fold ROC AUC over all rows (5-fold, unshuffled, stratified)."""
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=False).split(X, y):
        s[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, s)), s


def cluster_boot_gain(y, s_a, s_b, buckets, n_boot, seed):
    """Cluster bootstrap over cell100 buckets. OOF scores fixed; a bucket drawn twice contributes its rows twice."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(buckets)
    rows_by = {c: np.flatnonzero(buckets == c) for c in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by[c] for c in pick])
        draws.append(roc_auc_score(y[idx], s_b[idx]) - roc_auc_score(y[idx], s_a[idx]))
    draws = np.array(draws)
    return draws, np.percentile(draws, [2.5, 97.5])


def verdict(gain, lo):
    if gain < 0.02:
        return "KILL"
    if gain >= 0.05 and lo > 0:
        return "PASS"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e01", required=True)
    ap.add_argument("--e10", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot_seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    e01 = pd.read_parquet(a.e01)
    e10 = pd.read_parquet(a.e10)
    assert len(e01) == 3000 and len(e10) == 3000, (len(e01), len(e10))
    assert set(e01.pid) == set(e10.pid) and e10.pid.is_unique and e01.pid.is_unique
    # E10 columns keep their names; overlapping E01 columns get the _e01 suffix.
    m = e10.merge(e01, on="pid", suffixes=("", "_e01"), validate="one_to_one")
    assert len(m) == 3000, len(m)
    assert (m.question == m.question_e01).all(), "question text differs between E01 and E10"
    assert (m.gold == m.gold_e01).all(), "gold differs between E01 and E10"
    assert (m.cell100 == m.cell100_e01).all(), "cell100 labels differ between E01 and E10"

    y = m.proxy_error.to_numpy(int)          # target: E10 proxy_error
    yr = m.proxy_error_e01.to_numpy(int)     # reverse target: E01 proxy_error
    buckets = m.cell100.to_numpy()
    n_buckets = int(len(np.unique(buckets)))

    XA = m[CHEAP].to_numpy(float)
    e01_pe = m.proxy_error_e01.to_numpy(float)
    XB = np.column_stack([XA, e01_pe])
    XC = m[CHEAP + EMB].to_numpy(float)
    XD = e01_pe.reshape(-1, 1)

    auc_A, sA = pooled_oof_auc(XA, y)
    auc_B, sB = pooled_oof_auc(XB, y)
    auc_C, _ = pooled_oof_auc(XC, y)
    auc_D, _ = pooled_oof_auc(XD, y)
    gain = auc_B - auc_A
    draws, ci = cluster_boot_gain(y, sA, sB, buckets, a.boot, a.boot_seed)

    XRA = m[[c + "_e01" for c in CHEAP]].to_numpy(float)
    XRB = np.column_stack([XRA, m.proxy_error.to_numpy(float)])
    auc_rA, sRA = pooled_oof_auc(XRA, yr)
    auc_rB, sRB = pooled_oof_auc(XRB, yr)
    gain_r = auc_rB - auc_rA
    draws_r, ci_r = cluster_boot_gain(yr, sRA, sRB, buckets, a.boot, a.boot_seed)

    res = {
        "id": "L9a",
        "n_rows": int(len(m)), "n_cell100_buckets": n_buckets, "n_folds": 5,
        "boot": a.boot, "boot_seed": a.boot_seed,
        "positives_target_e10": int(y.sum()), "positives_e01": int(yr.sum()),
        "target": "E10 proxy_error",
        "auc_A_cheap_e10": auc_A,
        "auc_B_cheap_plus_e01_proxy_error": auc_B,
        "gain": gain,
        "gain_ci95": [float(ci[0]), float(ci[1])],
        "gain_boot_mean": float(draws.mean()),
        "gain_boot_draws": draws.tolist(),
        "context_auc_C_cheap_plus_pc0_15": auc_C,
        "context_gain_C_minus_A": auc_C - auc_A,
        "context_auc_D_e01_proxy_error_alone": auc_D,
        "reverse": {
            "target": "E01 proxy_error",
            "auc_A_cheap_e01": auc_rA,
            "auc_B_cheap_plus_e10_proxy_error": auc_rB,
            "gain": gain_r,
            "gain_ci95": [float(ci_r[0]), float(ci_r[1])],
            "gain_boot_mean": float(draws_r.mean()),
            "gain_boot_draws": draws_r.tolist(),
        },
        "verdict": verdict(gain, ci[0]),
        "verdict_rule": "PASS: gain>=0.05 and CI lower>0; KILL: gain<0.02; else INCONCLUSIVE",
    }
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)

    print(f"rows={res['n_rows']} buckets={n_buckets} positives_e10={res['positives_target_e10']} "
          f"positives_e01={res['positives_e01']}")
    print(f"AUC A (cheap, E10)              = {auc_A:.4f}")
    print(f"AUC B (A + E01 proxy_error)     = {auc_B:.4f}")
    print(f"gain = B - A                    = {gain:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  boot mean {draws.mean():.4f}")
    print(f"context AUC C (A + pc0..15)     = {auc_C:.4f}  (C - A = {auc_C - auc_A:.4f})")
    print(f"context AUC D (E01 pe alone)    = {auc_D:.4f}")
    print(f"reverse: AUC A_rev (E01 cheap)  = {auc_rA:.4f}")
    print(f"reverse: AUC B_rev (+E10 pe)    = {auc_rB:.4f}")
    print(f"reverse gain                    = {gain_r:.4f}  95% CI [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
    print(f"verdict (pre-registered rule)   = {res['verdict']}")


if __name__ == "__main__":
    main()
