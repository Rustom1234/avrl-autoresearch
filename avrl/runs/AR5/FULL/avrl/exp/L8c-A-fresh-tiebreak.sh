#!/bin/bash
# L8c-A-fresh-tiebreak driver (the steps as run, interactively, in this order). Registered bins and frozen rules: docstring of exp/l8c_analyze.py.
# Scripts: exp/l8c_rollouts_fresh.py (GPU, copy of e21_lowT_rollouts.py on fresh prompts, orig pids 2000..3999), exp/l5c_proxy_score.py (GPU, rm, unchanged), exp/l3b_judge_score.py (GPU, Qwen3-4B, unchanged),
# exp/l8c_boot_forced.py (CPU, copy of sim_warm_boot_j4.py with env FORCE_BEST7), exp/l8c_analyze.py (CPU, build + final).
# Outputs under runs/L8c-A-fresh-tiebreak/: rollouts/ rm/ q3_4b/ cells/ pool/<proxy>/feats_J.parquet gate_i/<proxy>/ boot/<proxy>/boot_tb_f{0.25,0.1}.json result.json table.txt done.flag (last).
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; R=$A/runs/L8c-A-fresh-tiebreak; PY=/homes/rdubash/active-audit/.venv/bin/python; PYJ=/homes/rdubash/active-audit/.venv-judge/bin/python; cd $A || exit 9
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct
poll() { until [ -f $1/done.flag ]; do ls $1 > /dev/null; sleep 30; done; echo "$1 exit_code $(cat $1/exit_code)"; }

# Stage 1, GPU job A (about 12 min): 2,000 fresh prompts, T=0.25, G=8, Qwen2.5-1.5B-Instruct
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R/rollouts "$PYJ exp/l8c_rollouts_fresh.py --model $F/models/Qwen2.5-1.5B-Instruct --temps 0.25 --pid_start 2000 --n_prompts 2000 --G 8 --max_tokens 512 --seed 0 --out $R/rollouts"; poll $R/rollouts

# Stage 2, GPU jobs B and C, one at a time
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/rm "$PYJ exp/l5c_proxy_score.py --kinds rm --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out $R/rm"; poll $R/rm
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/q3_4b "$PYJ exp/l3b_judge_score.py --model $F/models/Qwen3-4B --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --out $R/q3_4b"; poll $R/q3_4b

# Stage 3, CPU. Cells: frozen KMeans(100, n_init=3, random_state=0) refit on the stored pids 0..1999 MiniLM embeddings (reproduces cell100 exactly, checked), fresh prompts assigned by predict (cells/cell100_fresh.npy, done once in the shell).
# Gate (i), run while GPU job A was generating (no fresh label existed yet): forced-BEST7 replay on the pids 0..1999 feats_J of both proxies, compared with runs/L7a-tiebreak by l8c_analyze.py --stage final.
mkdir -p $R/gate_i/rm $R/gate_i/q3_4b
FORCE_BEST7=agree_acc_soth OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/l8c_boot_forced.py $A/runs/L5c-proxies/rm/feats_J.parquet judge_error cell100 $FEATS $R/gate_i/rm/boot_f0.25.json > $R/gate_i/rm/boot.log 2>&1 &
FORCE_BEST7=agree_acc_sdiff OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 $PY exp/l8c_boot_forced.py $A/runs/L3b-judge-proxy/feats_J.parquet judge_error cell100 $FEATS $R/gate_i/q3_4b/boot_f0.25.json > $R/gate_i/q3_4b/boot.log 2>&1 &
wait
# NOTE as run: the first fresh replay (without MIN_CELL=1) used 1998 groups because the j4 min-3 cell filter dropped the 2 groups of frozen-centroid cell 94; the pool-size check in l8c_analyze.py caught it, the outputs were moved to boot_cellfilter1998_first_run/ (sensitivity only)
# and the replay was rerun once with MIN_CELL=1 (all 2,000 groups, 500 audits). Gate (i) was rerun with the final script (MIN_CELL unset, default 3, identical to j4); the first copy is kept in gate_i_script_v1/.
# Fresh feats_J (frozen accept rules), then the forced-BEST7 replay on the fresh pools at 25% (primary) and 10% (secondary), then gates + verdict
$PY exp/l8c_analyze.py --run_dir $R --stage build > $R/build.log 2>&1
mkdir -p $R/boot/rm $R/boot/q3_4b; JOBS=$R/boot_jobs.txt; : > $JOBS
for F_ in 0.25 0.1; do
  echo "MIN_CELL=1 FORCE_BEST7=agree_acc_soth OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F_ $PY exp/l8c_boot_forced.py $R/pool/rm/feats_J.parquet judge_error cell100 $FEATS $R/boot/rm/boot_tb_f$F_.json > $R/boot/rm/boot_f$F_.log 2>&1" >> $JOBS
  echo "MIN_CELL=1 FORCE_BEST7=agree_acc_sdiff OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F_ $PY exp/l8c_boot_forced.py $R/pool/q3_4b/feats_J.parquet judge_error cell100 $FEATS $R/boot/q3_4b/boot_tb_f$F_.json > $R/boot/q3_4b/boot_f$F_.log 2>&1" >> $JOBS
done
xargs -P 4 -I{} bash -c '{}' < $JOBS; grep -l "GATE FAILED\|Traceback" $R/boot/*/*.log
$PY exp/l8c_analyze.py --run_dir $R --stage final > $R/final.log 2>&1; echo $? > $R/exit_code; touch $R/done.flag
