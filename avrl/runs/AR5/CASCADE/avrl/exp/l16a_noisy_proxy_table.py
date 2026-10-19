"""L16a step 1 (CPU only): rebuild the proxy-derived columns of the L14b pool from K of the 8 rollouts per group.

Row order and the non-proxy columns (split, pid, gold, cell100, pc0 to pc15, log_qlen, q_len) come from
avrl/runs/L14b/pool_feats.parquet, so the pool order is identical. Rollouts come from avrl/runs/L13a/rollouts.parquet
(train part, pid >= 3000) and avrl/runs/L14a/rollouts.parquet (test part). For each group, K rollouts are kept without
replacement, drawn with np.random.default_rng(seed) visiting groups in pool order. On the kept rollouts only:
majority by the p0_rollouts.py rule (np.unique over answers, then argmax of counts, so a tie goes to the smallest answer),
proxy_error = (majority != gold), agreement = max count / K, n_distinct, mean_tokens, std_tokens (np.std, ddof 0),
any_truncated = max of the truncated column over the kept rollouts.

Built-in control: the same builder at K = 8 must reproduce avrl/runs/L14b/pool_feats.parquet exactly on majority,
proxy_error, agreement, n_distinct, mean_tokens, std_tokens and any_truncated for every row. If it fails, the first
mismatch is written to control.json, no table is written, and the script exits 1.

Usage: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l16a_noisy_proxy_table.py --keep 4 --seed 0 --out avrl/runs/L16a
Writes <out>/control.json, <out>/pool_feats_noisy<keep>.parquet and <out>/build.json.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L13A = os.path.join(ROOT, "avrl", "runs", "L13a")
L14A = os.path.join(ROOT, "avrl", "runs", "L14a")
L14B = os.path.join(ROOT, "avrl", "runs", "L14b")
G_SAMPLES, CUT_PID = 8, 3000
PROXY_COLS = ["majority", "proxy_error", "agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated"]
FLOAT_COLS = ["agreement", "mean_tokens", "std_tokens"]
COLS = ["pid", "k", "answer", "tokens", "truncated"]


def load_rollouts():
    """Rollouts as one row per (split, pid) group, sorted by (split, pid), with the 8 rollouts in k order."""
    r13 = pd.read_parquet(os.path.join(L13A, "rollouts.parquet"), columns=COLS)
    r13 = r13[r13.pid >= CUT_PID].assign(split="train")
    r14 = pd.read_parquet(os.path.join(L14A, "rollouts.parquet"), columns=COLS).assign(split="test")
    R = pd.concat([r13, r14], ignore_index=True).sort_values(["split", "pid", "k"]).reset_index(drop=True)
    sizes = R.groupby(["split", "pid"], sort=True).size()
    assert (sizes == G_SAMPLES).all(), "every group must hold exactly G rollouts"
    assert (R.k.values.reshape(-1, G_SAMPLES) == np.arange(G_SAMPLES)).all(), "rollout k must run 0..G-1 in each group"
    index = pd.MultiIndex.from_tuples(list(sizes.index), names=["split", "pid"])
    ans = np.asarray(R.answer.values, dtype=str).reshape(-1, G_SAMPLES)
    tok = R.tokens.values.astype(float).reshape(-1, G_SAMPLES)
    tru = R.truncated.values.astype(np.int64).reshape(-1, G_SAMPLES)
    return index, ans, tok, tru


def build(pool, blocks, keep, seed):
    """Proxy-derived columns for every pool row at K = keep, in pool row order."""
    index, ans_b, tok_b, tru_b = blocks
    pos = index.get_indexer(pd.MultiIndex.from_arrays([pool.split.values, pool.pid.values], names=["split", "pid"]))
    assert (pos >= 0).all(), "a pool row has no rollout group"
    rng = np.random.default_rng(seed)
    sel = np.array([rng.choice(G_SAMPLES, size=keep, replace=False) for _ in range(len(pool))])  # pool order
    ans = np.take_along_axis(ans_b[pos], sel, axis=1)
    tok = np.take_along_axis(tok_b[pos], sel, axis=1)
    tru = np.take_along_axis(tru_b[pos], sel, axis=1)
    maj, agree, nd, mean, std, anytr = [], [], [], [], [], []
    for i in range(len(pool)):
        vals, cnt = np.unique(ans[i], return_counts=True)
        maj.append(vals[np.argmax(cnt)])
        agree.append(cnt.max() / keep)
        nd.append(len(vals))
        mean.append(np.mean(tok[i]))
        std.append(np.std(tok[i]))
        anytr.append(int(tru[i].max()))
    maj = np.array(maj, dtype=str)
    gold = pool.gold.astype(str).values
    return {"majority": maj, "proxy_error": (maj != gold).astype(np.int64), "agreement": np.array(agree, float),
            "n_distinct": np.array(nd, np.int64), "mean_tokens": np.array(mean, float), "std_tokens": np.array(std, float),
            "any_truncated": np.array(anytr, np.int64)}


def compare(pool, rebuilt):
    """Counts of mismatching rows per proxy column, and the first mismatch in pool row order."""
    counts, first = {}, None
    for c in PROXY_COLS:
        stored, new = pool[c].values, rebuilt[c]
        if c == "majority":
            bad = stored.astype(str) != new.astype(str)
        elif c in FLOAT_COLS:
            bad = stored.astype(np.float64) != new.astype(np.float64)
        else:
            bad = stored.astype(np.int64) != new.astype(np.int64)
        counts[c] = int(bad.sum())
        if bad.any():
            i = int(np.argmax(bad))
            if first is None or i < first["row"]:
                first = {"column": c, "row": i, "split": str(pool.split.values[i]), "pid": int(pool.pid.values[i]),
                         "stored": str(stored[i]), "rebuilt": str(new[i])}
    return counts, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=4, help="rollouts kept per group (1 to 8)")
    ap.add_argument("--seed", type=int, default=0, help="np.random.default_rng seed for the subsample")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 1 <= a.keep <= G_SAMPLES:
        raise SystemExit(f"--keep must be between 1 and {G_SAMPLES}")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    pool = pd.read_parquet(os.path.join(L14B, "pool_feats.parquet"))
    blocks = load_rollouts()

    # Built-in control: K = 8 keeps every rollout of every group, so the rebuild must equal the stored L14b table.
    counts, first = compare(pool, build(pool, blocks, G_SAMPLES, seed=0))
    control = {"status": "PASS" if first is None else "FAILED", "rows": int(len(pool)), "K": G_SAMPLES,
               "mismatch_rows_per_column": counts, "first_mismatch": first}
    json.dump(control, open(os.path.join(out, "control.json"), "w"), indent=1)
    print(json.dumps(control, indent=1), flush=True)
    if first is not None:
        print("CONTROL FAILED: no K table written", flush=True)
        sys.exit(1)

    cols = build(pool, blocks, a.keep, a.seed)
    new = pool.copy()
    for c in PROXY_COLS:
        new[c] = cols[c]
    path = os.path.join(out, f"pool_feats_noisy{a.keep}.parquet")
    new.to_parquet(path, index=False)

    flips = int((new.proxy_error.values != pool.proxy_error.values).sum())
    build_info = {"keep": a.keep, "seed": a.seed, "rows": int(len(new)), "rows_by_split": {k: int(v) for k, v in new.split.value_counts().items()},
                  "proxy_error_rate": float(new.proxy_error.mean()), "proxy_error_rate_by_split": {s: float(new.proxy_error[new.split == s].mean()) for s in ["train", "test"]},
                  "total_proxy_errors": int(new.proxy_error.sum()), "rows_proxy_error_changed_vs_K8": flips,
                  "mean_agreement": float(new.agreement.mean()), "any_truncated_rate": float(new.any_truncated.mean()),
                  "table": path}
    json.dump(build_info, open(os.path.join(out, "build.json"), "w"), indent=1)
    print(json.dumps(build_info, indent=1), flush=True)


if __name__ == "__main__":
    main()
