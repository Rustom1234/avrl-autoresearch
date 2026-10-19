#!/usr/bin/env python3
"""L3v step 3: build the 20 label-independent null pools.

For k = 0..19: perm = np.random.default_rng(20000+k).permutation(3000). The six feature columns are reordered together by
perm. proxy_error, cell100 and every other column stay in place, so the labels are unchanged and no feature carries label
information. Reads avrl/runs/p0/feats.parquet (read only); writes avrl/runs/L3v/pools/null_k.parquet.
CPU only. No GPU, no pip, no ssh.
"""
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/p0/feats.parquet"
OUT = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/L3v/pools"
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
N_NULL = 20


def corr(x, y):
    return float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])


def main():
    df = pd.read_parquet(SRC)
    assert len(df) == 3000, len(df)
    os.makedirs(OUT, exist_ok=True)
    y = df["proxy_error"].to_numpy()
    print(f"real pool: groups={len(df)} proxy_errors={int(y.sum())}")
    print("real corr(feature, proxy_error): "
          + ", ".join(f"{c}={corr(df[c].to_numpy(), y):+.3f}" for c in FEAT_COLS))
    for k in range(N_NULL):
        perm = np.random.default_rng(20000 + k).permutation(len(df))
        null = df.copy()
        for c in FEAT_COLS:
            null[c] = df[c].to_numpy()[perm]
        assert (null["proxy_error"].to_numpy() == y).all()
        assert (null["cell100"].to_numpy() == df["cell100"].to_numpy()).all()
        for c in FEAT_COLS:
            assert np.array_equal(np.sort(null[c].to_numpy()), np.sort(df[c].to_numpy())), c
        path = os.path.join(OUT, f"null_{k}.parquet")
        null.to_parquet(path, index=False)
        print(f"null_{k}: corr(agreement, proxy_error)={corr(null['agreement'].to_numpy(), y):+.3f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
