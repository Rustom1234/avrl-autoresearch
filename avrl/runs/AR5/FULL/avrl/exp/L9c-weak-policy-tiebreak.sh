#!/bin/bash
# L9c-weak-policy-tiebreak driver (the steps as run, interactively, in this order). Registered rules and bins: docstring of exp/l9c_analyze.py.
# GPU scripts reused UNCHANGED (only --model changes): exp/l8c_rollouts_fresh.py, exp/l5c_proxy_score.py, exp/l3b_judge_score.py. CPU: exp/l8c_boot_forced.py (byte-identical to L8c-A), exp/l9c_analyze.py (copy of l8c_analyze.py with the operative-cell design), exp/l6a_cell_rate.py (unchanged).
# Outputs under runs/L9c-weak-policy-tiebreak/: rollouts/ rm/ q3_4b/ cells/ pool/<proxy>/feats_J.parquet gate_i/<proxy>/ boot/<proxy>/boot_tb_f{0.25,0.1}.json l6a/q3_4b/ result.json table.txt done.flag (last).
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; R=$A/runs/L9c-weak-policy-tiebreak; PY=/homes/rdubash/active-audit/.venv/bin/python; PYJ=/homes/rdubash/active-audit/.venv-judge/bin/python; cd $A || exit 9
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct
poll() { until [ -f $1/done.flag ]; do ls $1 > /dev/null; sleep 30; done; echo "$1 exit_code $(cat $1/exit_code)"; }

# Stage 1, GPU job A (about 9 min): 2,000 fresh prompts (orig pids 2000..3999, the L8c-A prompts), T=0.25, G=8, Qwen2.5-0.5B-Instruct
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R/rollouts "$PYJ exp/l8c_rollouts_fresh.py --model $F/models/Qwen2.5-0.5B-Instruct --temps 0.25 --pid_start 2000 --n_prompts 2000 --G 8 --max_tokens 512 --seed 0 --out $R/rollouts"
# While job A ran (CPU): gate (i) forced-BEST7 replay on the pids 0..1999 feats_J of both proxies (gate_i/run.sh, same commands as L8c-A), sha256 of exp/l8c_boot_forced.py in gate_i/l8c_boot_forced.sha256
poll $R/rollouts

# Stage 2, GPU jobs B and C, one at a time
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/rm "$PYJ exp/l5c_proxy_score.py --kinds rm --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out $R/rm"; poll $R/rm
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/q3_4b "$PYJ exp/l3b_judge_score.py --model $F/models/Qwen3-4B --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --out $R/q3_4b"; poll $R/q3_4b

# Stage 3, CPU. Build (frozen cells: same prompts as L8c-A; cell300 of the fresh prompts by frozen-centroid predict), then forced-BEST7 replays at 25% (verdict) and 10% (secondary), then l6a (descriptive), then gates + verdict
$PY exp/l9c_analyze.py --run_dir $R --stage build > $R/build.log 2>&1
mkdir -p $R/boot/rm $R/boot/q3_4b; JOBS=$R/boot_jobs.txt; : > $JOBS
for F_ in 0.25 0.1; do
  echo "MIN_CELL=1 FORCE_BEST7=agree_acc_soth OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F_ $PY exp/l8c_boot_forced.py $R/pool/rm/feats_J.parquet judge_error cell100 $FEATS $R/boot/rm/boot_tb_f$F_.json > $R/boot/rm/boot_f$F_.log 2>&1" >> $JOBS
  echo "MIN_CELL=1 FORCE_BEST7=agree_acc_sdiff OMP_NUM_THREADS=2 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=$F_ $PY exp/l8c_boot_forced.py $R/pool/q3_4b/feats_J.parquet judge_error cell100 $FEATS $R/boot/q3_4b/boot_tb_f$F_.json > $R/boot/q3_4b/boot_f$F_.log 2>&1" >> $JOBS
done
xargs -P 4 -I{} bash -c '{}' < $JOBS; grep -l "GATE FAILED\|Traceback" $R/boot/*/*.log
# claim 3 descriptive read: l6a_cell_rate.py UNCHANGED; --e21 points to a missing path so its pids 0..1999 E21 reproduction gate is skipped; --consensus = the two pools of this experiment (post hoc, descriptive)
mkdir -p $R/l6a/q3_4b; OMP_NUM_THREADS=4 $PY exp/l6a_cell_rate.py --pool $R/pool/q3_4b --out $R/l6a/q3_4b --e21 $R/none --consensus $R/pool/q3_4b $R/pool/rm > $R/l6a/q3_4b/run.log 2>&1
$PY exp/l9c_analyze.py --run_dir $R --stage final > $R/final.log 2>&1; echo $? > $R/exit_code; touch $R/done.flag
