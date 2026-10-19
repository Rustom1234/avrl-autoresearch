#!/bin/bash
# L15b runbook. CPU only, runs directly in the shell. No GPU job: nothing goes through bin/gpu_run.sh, no ssh, no pip.
# Occupancy-weighted remaining-row cut (avrl/exp/l15b_occupancy_weighted.py), frozen 200-label table minus plain fixed at 10%.
# Stages, in order:
#   ref                 unweighted reference cuts (pid > 1318 and pid <= 1318), gated against runs/L13a before any weighted number
#   cuts LO HI          occupancy-weighted cuts for cut seeds LO..HI (batches of 8 keep each shell call under 10 minutes)
#   sens                UNREGISTERED sensitivity: drop strata with fewer than 5 test or remaining docs, cut seeds 100..147
#   combine             writes runs/L15b/result.json from the 48 cut results, then done.flag
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L15b
PY=/homes/rdubash/active-audit/.venv/bin/python
S=$R/avrl/exp/l15b_occupancy_weighted.py
COMMON="--fit_feats $R/avrl/runs/p0/feats.parquet --eval_feats $R/avrl/runs/L4b/feats.parquet --budget 0.10 --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8"
mkdir -p "$RUN"
case "$1" in
  ref)
    "$PY" -I "$S" $COMMON --eval_pid_min 1319 --out "$RUN/ref_rem" > "$RUN/ref_rem.log" 2>&1; echo "ref_rem exit $?"
    "$PY" -I "$S" $COMMON --eval_pid_max 1318 --out "$RUN/ref_test" > "$RUN/ref_test.log" 2>&1; echo "ref_test exit $?"
    tail -1 "$RUN/ref_rem.log"; tail -1 "$RUN/ref_test.log" ;;
  cuts)
    "$PY" -I "$S" $COMMON --weighted --eval_pid_min 1319 --cut_seeds "$2-$3" --out "$RUN" > "$RUN/cuts_$2_$3.log" 2>&1
    echo "cuts $2-$3 exit $?"; tail -2 "$RUN/cuts_$2_$3.log" ;;
  sens)
    "$PY" -I "$S" $COMMON --weighted --min_stratum 5 --eval_pid_min 1319 --cut_seeds 100-147 --out "$RUN/sens_min5" > "$RUN/sens_min5.log" 2>&1
    echo "sens exit $?"
    "$PY" -I "$S" $COMMON --combine --min_stratum 5 --cut_seeds 100-147 --ref_dir "$RUN" --out "$RUN/sens_min5" > "$RUN/sens_min5_combine.log" 2>&1
    echo "sens combine exit $?"; cat "$RUN/sens_min5_combine.log" ;;
  combine)
    "$PY" -I "$S" $COMMON --combine --cut_seeds 100-147 --ref_dir "$RUN" --out "$RUN" > "$RUN/combine.log" 2>&1
    echo "combine exit $?"; cat "$RUN/combine.log"
    touch "$RUN/done.flag" ;;
  *) echo "usage: $0 ref|cuts LO HI|sens|combine" >&2; exit 2 ;;
esac
