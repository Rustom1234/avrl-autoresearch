#!/bin/bash
# L3b-judge-proxy driver (the steps as run). Scripts: exp/l3b_judge_score.py (GPU, Qwen3-4B Yes/No judge scores), exp/l3b_analyze.py (CPU, claim 3), exp/sim_warm_boot_j.py (CPU replay bootstrap, copy of sim_warm_boot.py),
# exp/l3b_finalize.py (CPU, replay table and result.json). Analysis scripts were written and smoke-tested on synthetic scores before any real score existed.
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; E=$A/runs/E21-lowT-confident-errors; R=$A/runs/L3b-judge-proxy; PY=/homes/rdubash/active-audit/.venv/bin/python; cd $A
FEATS="s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct"
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R "/homes/rdubash/active-audit/.venv-judge/bin/python exp/l3b_judge_score.py --model $F/models/Qwen3-4B --rollouts $E/T0.25/rollouts.parquet --groups $E/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out $R"
until [ -f $R/done.flag ]; do ls $R > /dev/null; sleep 30; done; cat $R/exit_code
OMP_NUM_THREADS=8 $PY exp/l3b_analyze.py --run_dir $R --e21 $E --out $R > $R/analyze.log 2>&1
BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 OMP_NUM_THREADS=2 $PY exp/sim_warm_boot_j.py $R/feats_J.parquet judge_error cell100 $FEATS $R/boot_J_f0.25.json > $R/boot_J_f0.25.log 2>&1 &
BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.1 OMP_NUM_THREADS=2 $PY exp/sim_warm_boot_j.py $R/feats_J.parquet judge_error cell100 $FEATS $R/boot_J_f0.1.json > $R/boot_J_f0.1.log 2>&1 &
POOL=held BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 OMP_NUM_THREADS=2 $PY exp/sim_warm_boot_j.py $R/feats_J.parquet judge_error cell100 $FEATS $R/boot_J_held_f0.25.json > $R/boot_J_held_f0.25.log 2>&1 &
wait
$PY exp/l3b_finalize.py --run_dir $R > $R/finalize.log 2>&1 && touch $R/done.flag
