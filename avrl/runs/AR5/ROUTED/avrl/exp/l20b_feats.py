"""L20b step 3: feature columns for the fresh fit pool and the fresh test pool. CPU only, no GPU.

For each run, reads <run>/groups.parquet and writes <run>/feats.parquet. log_qlen and split are computed exactly as in
avrl/exp/l2b_feats.py: log_qlen = log1p(q_len), split = (agreement < 1). The cell columns (cell30, cell100, cell300) are
attached by question text from the KMeans partitions already registered for these questions:
  fit  (the 3,000 p0 questions):           runs/p0/feats.parquet
  test (the 1,319 GSM8K test questions):   runs/L19c/feats.parquet (pids 0 to 1318 of the L4b pool)
KMeans is not run again. A re-fit on the new rollouts would change the cell100 partition, and the registered split
must stay the same. Only the rollout columns change between the pools.
Asserts: every question matches exactly one row of its source, no cell is missing, pids are 0 to n-1 in order.
Usage: python -I l20b_feats.py
"""
import os
import numpy as np, pandas as pd

ROOT = "/homes/rdubash/avrl_live/ROUTED"
RUNS = os.path.join(ROOT, "avrl", "runs")
CELLS = ["cell30", "cell100", "cell300"]
JOBS = [("fit", os.path.join(RUNS, "p0", "feats.parquet")),
        ("test", os.path.join(RUNS, "L19c", "feats.parquet"))]

for name, src_path in JOBS:
    run_dir = os.path.join(RUNS, "L20b", name)
    G = pd.read_parquet(os.path.join(run_dir, "groups.parquet"))
    assert list(G.pid) == list(range(len(G))), f"{name}: pids are not 0 to n-1 in order"
    src = pd.read_parquet(src_path)
    assert src.question.is_unique, f"{src_path}: questions are not unique"
    out = G.join(src.set_index("question")[CELLS], on="question", how="left")
    assert len(out) == len(G), f"{name}: row count changed"
    missing = int(out[CELLS].isna().any(axis=1).sum())
    assert missing == 0, f"{name}: {missing} rows have no cell (question text not found in {src_path})"
    out["log_qlen"] = np.log1p(out.q_len); out["split"] = (out.agreement < 1).astype(int)
    out.to_parquet(os.path.join(run_dir, "feats.parquet"), index=False)
    cnt = out.cell100.value_counts()
    print(f"{name}: wrote {os.path.join(run_dir, 'feats.parquet')}: {len(out)} groups, proxy_error rate {out.proxy_error.mean():.4f}, "
          f"cell100 buckets {out.cell100.nunique()}, buckets with >= 3 docs {int((cnt >= 3).sum())}, questions matched {len(out)}",
          flush=True)
