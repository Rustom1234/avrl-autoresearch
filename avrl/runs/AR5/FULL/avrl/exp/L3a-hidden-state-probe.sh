#!/bin/bash
# L3a-hidden-state-probe driver (the steps as actually run). Scripts: exp/l3a_hidden_probe.py (GPU), exp/l3a_stage1.py (CPU, calls exp/sim_e21.py), exp/l3a_finalize.py (CPU).
# Stage 2 (exp/sim_warm_boot_hs.py) is only written and run if stage 1 clears (primary delta AUC >= 0.05 with bootstrap CI lower end > 0); it did not.
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; E=$A/runs/E21-lowT-confident-errors; R=$A/runs/L3a-hidden-state-probe; cd $A
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R "/homes/rdubash/active-audit/.venv-judge/bin/python exp/l3a_hidden_probe.py --model $F/models/Qwen2.5-1.5B-Instruct --rollouts $E/T0.25/rollouts.parquet --groups $E/T0.25/groups.parquet --layers 14,20,28 --out $R"
until [ -f $R/done.flag ]; do ls $R > /dev/null; sleep 30; done; cat $R/exit_code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /homes/rdubash/active-audit/.venv/bin/python exp/l3a_stage1.py --run_dir $R --e21 $E > $R/stage1.log 2>&1
/homes/rdubash/active-audit/.venv/bin/python exp/l3a_finalize.py $R && touch $R/done.flag
