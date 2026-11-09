#!/usr/bin/env bash
# L16b (CPU only, no GPU, no rollouts): bucket oracle vs. the label-free agreement threshold on the L14b gate-passing pool.
# Pre-registered in avrl/RESEARCH_LOG.md ("L16b pre-registration"). Run directly in the shell (about 5 minutes):
#   bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L16b.sh
# Order: readout (copy of the L14b readout with one arm added), then the control (its result must be read before the
# bucket_oracle row), then the independent pandas check. exit_code is the readout's exit code.
CASCADE=/homes/rdubash/avrl_live/CASCADE
PY=/homes/rdubash/active-audit/.venv/bin/python
RUN=$CASCADE/avrl/runs/L16b
mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
cd "$CASCADE" || exit 1

$PY avrl/exp/l16b_bucket_oracle.py --feats avrl/runs/L14b/pool_feats.parquet --cell cell100 --features cheap+emb \
  --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out "$RUN" > "$RUN/readout.log" 2>&1
echo $? > "$RUN/exit_code"
$PY avrl/exp/L16b_control.py > "$RUN/control.log" 2>&1
$PY avrl/exp/L16b_check.py > "$RUN/check.log" 2>&1
touch "$RUN/done.flag"
