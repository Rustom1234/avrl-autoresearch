#!/usr/bin/env bash
# L16a (CPU only, no GPU, no rollouts): claim 2 readout on one noisier proxy regime. The L14a gate-passing pool keeps its
# row order and non-proxy columns (avrl/runs/L14b/pool_feats.parquet); the majority proxy and cheap columns are rebuilt from
# K = 4 of the 8 rollouts per group. Step 1 is avrl/exp/l16a_noisy_proxy_table.py (its K = 8 control must pass first).
# Step 2 is the unchanged L14b readout, avrl/exp/l14b_claim2_readout.py, with the L14b pre-registered verdict rule.
# Primary result: subsample seed 0 (avrl/runs/L16a/result.json). Seeds 1 and 2 are descriptive repeats in avrl/runs/L16a/seed1 and seed2.
# Run directly in the shell (CPU analysis, about 4 minutes):
#   bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L16a.sh
CASCADE=/homes/rdubash/avrl_live/CASCADE
PY=/homes/rdubash/active-audit/.venv/bin/python
RUN=$CASCADE/avrl/runs/L16a
mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
cd "$CASCADE" || exit 1

# Primary: subsample seed 0.
$PY avrl/exp/l16a_noisy_proxy_table.py --keep 4 --seed 0 --out "$RUN" > "$RUN/build.log" 2>&1 && \
$PY avrl/exp/l14b_claim2_readout.py --feats "$RUN/pool_feats_noisy4.parquet" --cell cell100 --features cheap+emb \
  --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out "$RUN" > "$RUN/readout.log" 2>&1
echo $? > "$RUN/exit_code"

# Descriptive only: subsample seeds 1 and 2 (they do not change the verdict).
for s in 1 2; do
  d="$RUN/seed$s"
  mkdir -p "$d"
  $PY avrl/exp/l16a_noisy_proxy_table.py --keep 4 --seed "$s" --out "$d" > "$d/build.log" 2>&1 && \
  $PY avrl/exp/l14b_claim2_readout.py --feats "$d/pool_feats_noisy4.parquet" --cell cell100 --features cheap+emb \
    --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out "$d" > "$d/readout.log" 2>&1
done

touch "$RUN/done.flag"
