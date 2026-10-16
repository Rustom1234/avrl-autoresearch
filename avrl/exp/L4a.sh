#!/usr/bin/env bash
# L4a: L3a's own follow-up 1. Rerun L3a's exact bootstrap script
# (e03b_bootstrap_matched.py, static_matched vs ts_online) at the two
# untested tail budgets (10% and 50%) that bracket L3a's already-run
# midpoint (25%). No new code: this just calls the existing,
# byte-identical script twice with a different --budget_frac and --out.
# CPU-only (sklearn/pandas), reuses avrl/runs/p0/feats.parquet, no GPU.
set -euo pipefail
cd /homes/rdubash/active-audit

PY=/homes/rdubash/active-audit/.venv/bin/python
SCRIPT=/homes/rdubash/active-audit/avrl/exp/e03b_bootstrap_matched.py
FEATS=/homes/rdubash/active-audit/avrl/runs/p0/feats.parquet
OUTDIR=/homes/rdubash/active-audit/avrl/runs/L4a

mkdir -p "$OUTDIR"

"$PY" "$SCRIPT" --feats "$FEATS" --label proxy_error --group cell100 \
  --budget_frac 0.10 --n_boot 500 --seed 0 --out "$OUTDIR/b10/result.json"

"$PY" "$SCRIPT" --feats "$FEATS" --label proxy_error --group cell100 \
  --budget_frac 0.50 --n_boot 500 --seed 0 --out "$OUTDIR/b50/result.json"

"$PY" /homes/rdubash/active-audit/avrl/exp/L4a_combine.py \
  --b10 "$OUTDIR/b10/result.json" \
  --b25 /homes/rdubash/active-audit/avrl/runs/L3a/result.json \
  --b50 "$OUTDIR/b50/result.json" \
  --out "$OUTDIR/result.json"

touch "$OUTDIR/done.flag"
