"""L14b step 1 (CPU only): build the 5,792-group pool for the L14b claim 2 readout.

Train part: avrl/runs/L13a/feats.parquet rows with pid >= 3000 (4,473 groups), unchanged.
Test part: the 1,319 GSM8K test groups in avrl/runs/L14a (groups and rollouts). The six cheap columns are recomputed from the
rollouts with the L13a definitions. cell100 and pc0 to pc15 come from a refit on the 7,473 L13a train questions
(all-MiniLM-L6-v2 on CPU, KMeans(100, n_init=3, random_state=0), PCA(16, random_state=0), as avrl/exp/e06_analyze.py).

Gate (pre-registered in avrl/RESEARCH_LOG.md, "L14b pre-registration"): the refit must reproduce the L13a cell100 exactly and
pc0 to pc15 within 1e-4 for pid 0 to 7472, and the cheap-column recomputation must reproduce the stored L13a and L14a columns.
If the gate fails, refit_gate.json records FAILED with the mismatch, no pool is written, and the script exits 1.

Usage: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/l14b_feature_refit.py --out avrl/runs/L14b
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L13A = os.path.join(ROOT, "avrl", "runs", "L13a")
L14A = os.path.join(ROOT, "avrl", "runs", "L14a")
TEST_SRC = os.path.join(ROOT, "avrl", "data", "gsm8k_test.parquet")
N_TRAIN_ALL, N_TRAIN_POOL, N_TEST, G_SAMPLES, MAX_TOK, CUT_PID = 7473, 4473, 1319, 8, 512, 3000
ATOL_PC, ATOL_FLOAT = 1e-4, 1e-9
PCS = [f"pc{i}" for i in range(16)]
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]


def cheap_from_rollouts(R):
    """Majority answer and cheap statistics per pid. Definitions as in avrl/exp/l14a_rollouts_test.py (copy of avrl/p0_rollouts.py):
    majority = most frequent normalized answer (ties to the first in sorted order), agreement = its count / G,
    n_distinct = number of distinct answers, mean and population std of rollout token counts, any_truncated = longest rollout hit the cap."""
    rows = []
    for pid, g in R.sort_values(["pid", "k"]).groupby("pid", sort=True):
        ans = np.asarray(g.answer.values, dtype=str)
        lens = g.tokens.values.astype(float)
        vals, cnt = np.unique(ans, return_counts=True)
        rows.append({"pid": int(pid), "majority": str(vals[np.argmax(cnt)]), "agreement": cnt.max() / G_SAMPLES,
                     "n_distinct": len(vals), "mean_tokens": float(np.mean(lens)), "std_tokens": float(np.std(lens)),
                     "any_truncated": int(lens.max() >= MAX_TOK)})
    return pd.DataFrame(rows)


def norm_gold(x):
    """Same normalization as norm() in avrl/exp/l14a_rollouts_test.py."""
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return x


def cheap_mismatch(stored, rec):
    """Compare stored and recomputed columns on the same pids. Returns {column: number of mismatching rows}."""
    s = stored.set_index("pid").sort_index()
    c = rec.set_index("pid").sort_index()
    assert list(s.index) == list(c.index), "pid sets differ"
    out = {"majority": int((s["majority"].astype(str).values != c["majority"].astype(str).values).sum())}
    for col in ("n_distinct", "any_truncated"):
        out[col] = int((s[col].values.astype(np.int64) != c[col].values.astype(np.int64)).sum())
    for col in ("agreement", "mean_tokens", "std_tokens"):
        out[col] = int((np.abs(s[col].values.astype(float) - c[col].values.astype(float)) > ATOL_FLOAT).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    f13 = pd.read_parquet(os.path.join(L13A, "feats.parquet")).sort_values("pid").reset_index(drop=True)
    g13 = pd.read_parquet(os.path.join(L13A, "groups.parquet")).sort_values("pid").reset_index(drop=True)
    r13 = pd.read_parquet(os.path.join(L13A, "rollouts.parquet"))
    g14 = pd.read_parquet(os.path.join(L14A, "groups.parquet")).sort_values("pid").reset_index(drop=True)
    r14 = pd.read_parquet(os.path.join(L14A, "rollouts.parquet"))
    tsrc = pd.read_parquet(TEST_SRC)

    checks, detail = {}, {}
    checks["L13a_pid_0_to_7472"] = bool((f13.pid.values == np.arange(N_TRAIN_ALL)).all() and (g13.pid.values == np.arange(N_TRAIN_ALL)).all())
    checks["L14a_pid_0_to_1318"] = bool((g14.pid.values == np.arange(N_TEST)).all())
    # L14a pid order is a seed-0 permutation of the test split (avrl/exp/l14a_rollouts_test.py: test.sample(n, random_state=seed)).
    t_perm = tsrc.sample(n=N_TEST, random_state=0).reset_index(drop=True)
    checks["L14a_pid_order_is_seed0_sample_of_test"] = bool((g14.question.values == t_perm.question.values).all())
    checks["L14a_questions_unique_and_set_equal_test"] = bool(g14.question.nunique() == N_TEST and set(g14.question) == set(tsrc.question))
    gold_exp = t_perm.answer.str.split("####").str[-1].str.strip().str.replace(",", "").map(norm_gold)
    checks["L14a_gold_recomputed_from_test_answer"] = bool((g14.gold.astype(str).values == gold_exp.values).all())
    checks["q_len_is_question_chars_L13a"] = bool((g13.q_len.values == g13.question.str.len().values).all())
    checks["q_len_is_question_chars_L14a"] = bool((g14.q_len.values == g14.question.str.len().values).all())

    # Cheap columns: recompute from rollouts, compare with the stored L13a feats (train) and L14a groups (test).
    rec13 = cheap_from_rollouts(r13)
    rec14 = cheap_from_rollouts(r14)
    mm13 = cheap_mismatch(f13.assign(majority=g13.majority.values), rec13)
    mm14 = cheap_mismatch(g14, rec14)
    detail["cheap_mismatch_L13a_pid0_7472"] = mm13
    detail["cheap_mismatch_L14a_test"] = mm14
    checks["cheap_recompute_matches_L13a"] = all(v == 0 for v in mm13.values())
    checks["cheap_recompute_matches_L14a"] = all(v == 0 for v in mm14.values())
    checks["log_qlen_matches_L13a"] = bool(np.allclose(f13.log_qlen.values, np.log1p(f13.q_len.values), atol=ATOL_FLOAT))
    checks["proxy_error_is_majority_wrong_L14a"] = bool((g14.proxy_error.values == (g14.majority.astype(str).values != g14.gold.astype(str).values).astype(int)).all())

    # Refit cell100 and PCs on the L13a train questions (pid order), as avrl/exp/e06_analyze.py does.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def embed(qs):
        return model.encode(list(qs), batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    E_tr = embed(g13.question.tolist())
    km = KMeans(n_clusters=100, n_init=3, random_state=0)
    cell_tr = km.fit_predict(E_tr)
    pca = PCA(n_components=16, random_state=0)
    pc_tr = pca.fit_transform(E_tr)

    cell_ref = f13.cell100.values.astype(int)
    cell_mismatch = np.where(cell_tr != cell_ref)[0]
    pc_diff = np.abs(pc_tr - f13[PCS].values.astype(float))
    max_pc_diff = float(pc_diff.max())
    checks["refit_cell100_exact_L13a"] = bool(len(cell_mismatch) == 0)
    checks["refit_pc0_pc15_within_1e-4_L13a"] = bool(max_pc_diff <= ATOL_PC)
    detail["refit"] = {"n_pid_checked": int(len(cell_tr)), "cell100_mismatches": int(len(cell_mismatch)),
                       "first_cell100_mismatch_pids": cell_mismatch[:10].tolist(),
                       "max_abs_pc_diff": max_pc_diff, "max_abs_pc_diff_per_pc": [float(x) for x in pc_diff.max(axis=0)]}

    gate_ok = all(checks.values())
    gate = {"status": "PASS" if gate_ok else "FAILED", "checks": checks, "detail": detail}
    if not gate_ok:
        json.dump(gate, open(os.path.join(out, "refit_gate.json"), "w"), indent=1)
        print(json.dumps(gate, indent=1), flush=True)
        print("GATE FAILED: no pool written, no L14b readout run.", flush=True)
        sys.exit(1)

    # Test part: predict cell100 and transform PCs with the same fitted objects.
    E_te = embed(g14.question.tolist())
    cell_te = km.predict(E_te)
    pc_te = pca.transform(E_te)

    tr = f13[f13.pid >= CUT_PID].copy()
    tr["split"] = "train"
    te = pd.DataFrame({
        "split": "test", "pid": g14.pid.values, "gold": g14.gold.values, "majority": rec14.majority.values,
        "proxy_error": g14.proxy_error.values, "agreement": rec14.agreement.values, "n_distinct": rec14.n_distinct.values,
        "mean_tokens": rec14.mean_tokens.values, "std_tokens": rec14.std_tokens.values,
        "any_truncated": rec14.any_truncated.values, "q_len": g14.q_len.values,
        "log_qlen": np.log1p(g14.q_len.values.astype(float)), "cell100": cell_te.astype(int),
    })
    for i, c in enumerate(PCS):
        te[c] = pc_te[:, i]
    cols = ["split", "pid", "gold", "majority", "proxy_error"] + CHEAP + ["q_len", "cell100"] + PCS
    tr["majority"] = tr["majority"].astype(str)
    pool = pd.concat([tr[cols], te[cols]], ignore_index=True)

    model_cols = CHEAP + ["cell100"] + PCS + ["proxy_error"]
    pool_checks = {
        "rows": int(len(pool)) == N_TRAIN_POOL + N_TEST,
        "train_rows": int((pool.split == "train").sum()) == N_TRAIN_POOL,
        "test_rows": int((pool.split == "test").sum()) == N_TEST,
        "unique_split_pid": not bool(pool.duplicated(["split", "pid"]).any()),
        "no_nan_model_columns": not bool(pool[model_cols].isna().any().any()),
        "cell100_in_0_99": bool(pool.cell100.between(0, 99).all()),
        "no_question_overlap_train_test": len(set(tr.question) & set(g14.question)) == 0,
    }
    if not all(pool_checks.values()):
        print("POOL CHECK FAILED", json.dumps(pool_checks), flush=True)
        sys.exit(1)
    pool.to_parquet(os.path.join(out, "pool_feats.parquet"), index=False)

    gate["pool"] = {"rows": int(len(pool)), "train_rows": N_TRAIN_POOL, "test_rows": N_TEST, "checks": pool_checks,
                    "proxy_error_rate_train": float(pool.loc[pool.split == "train", "proxy_error"].mean()),
                    "proxy_error_rate_test": float(pool.loc[pool.split == "test", "proxy_error"].mean()),
                    "cell100_buckets_pool": int(pool.cell100.nunique()),
                    "cell100_buckets_test": int(te.cell100.nunique()), "cell100_buckets_train": int(tr.cell100.nunique())}
    json.dump(gate, open(os.path.join(out, "refit_gate.json"), "w"), indent=1)
    print(json.dumps(gate, indent=1), flush=True)
    print(f"wrote {os.path.join(out, 'pool_feats.parquet')} ({len(pool)} rows)", flush=True)


if __name__ == "__main__":
    main()
