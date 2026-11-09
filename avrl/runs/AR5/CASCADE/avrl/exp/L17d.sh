#!/usr/bin/env bash
# L17d (CPU only, no GPU, no rollouts): labeled static ablation by feature family on the L14b sim pool (K = 8 majority proxy).
# Pre-registered in avrl/RESEARCH_LOG.md ("L17d pre-registration"). Run directly in the shell (about 3 minutes):
#   bash /homes/rdubash/avrl_live/CASCADE/avrl/exp/L17d.sh
# Order: readout (copy of the L14b readout with the static arms added), then the control (compares the shared L14b arms and
# gates on 938.0 / 0.519 / -0.077), then the independent pandas check. exit_code is the readout's exit code; the control and
# check exit codes are in control.log and check.log.
CASCADE=/homes/rdubash/avrl_live/CASCADE
PY=/homes/rdubash/active-audit/.venv/bin/python
RUN=$CASCADE/avrl/runs/L17d
mkdir -p "$RUN"
rm -f "$RUN/done.flag" "$RUN/exit_code"
cd "$CASCADE" || exit 1

$PY avrl/exp/l17d_static_ablation.py --feats avrl/runs/L14b/pool_feats.parquet --cell cell100 --features cheap+emb \
  --budgets 0.10,0.25,0.50 --seeds 8 --boot 200 --boot_budget 0.25 --workers 8 --out "$RUN" > "$RUN/readout.log" 2>&1
echo $? > "$RUN/exit_code"
$PY avrl/exp/L17d_control.py > "$RUN/control.log" 2>&1; echo "control exit $?" >> "$RUN/control.log"
$PY avrl/exp/L17d_check.py > "$RUN/check.log" 2>&1; echo "check exit $?" >> "$RUN/check.log"
touch "$RUN/done.flag"
