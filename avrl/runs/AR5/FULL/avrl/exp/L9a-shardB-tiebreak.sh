#!/bin/bash
# L9a-shardB-tiebreak driver (the steps as run, interactively, in this order). Registered bins and frozen rules: docstring of exp/l9a_analyze.py (written and frozen before the shard-B rollout job was launched).
# Scripts: exp/l9a_rollouts_fresh.py (GPU, unchanged copy of l8c_rollouts_fresh.py, pid_start 4000), exp/l5c_proxy_score.py (GPU, rm, unchanged), exp/l3b_judge_score.py (GPU, Qwen3-4B, unchanged),
# exp/l9a_boot_forced.py (CPU, copy of l8c_boot_forced.py plus the optional CH_RULE arm), exp/l9a_analyze.py (CPU: cells, build, power, chA, final).
# Outputs under runs/L9a-shardB-tiebreak/: ref_groups_0_3999.parquet rollouts/ rm/ q3_4b/ cells/ pool/ pool_A/ pool_B/<proxy>/feats_J.parquet gate_i/ power_check/ boot_A0/ chA.json boot/ boot_B/ result.json table.txt done.flag (last).
# Library mode: L9A_LIB_ONLY=1 source this file defines the functions only (used for the synthetic dry run of the CPU half, which never touches shard-B labels).
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; R=${R:-$A/runs/L9a-shardB-tiebreak}; PY=/homes/rdubash/active-audit/.venv/bin/python; PYJ=/homes/rdubash/active-audit/.venv-judge/bin/python; cd $A || exit 9
FEATS=s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct; BN=${BN:-200}; CH=${CH:-4}; FR="0.1 0.15 0.2 0.25 0.3 0.4"
poll() { until [ -f $1/done.flag ]; do ls $1 > /dev/null; sleep 30; done; echo "$1 exit_code $(cat $1/exit_code)"; }
armof() { [ "$1" = rm ] && echo agree_acc_soth || echo agree_acc_sdiff; }
chrule() { $PY -c "import json,sys; print(json.load(open('$R/chA.json'))['$1']['$2']['rule'])"; }
# mkjobs <kind AB|B|A0> <proxy>... : one replay command per (proxy, budget, chunk), appended to $R/jobs_<kind>.txt, longest budgets first. AB = pooled A + B (verdict), B = shard B alone with the CH_A arm, A0 = shard A alone, pooled accept flags, BOOT_B=0 (selects CH_A).
# The 200 resamples are run as 4 chunks of 50 (BOOT_START / BOOT_END of the same default_rng(12345) stream, identical to an unchunked run, checked on gate (i)); chunks are joined by l9a_analyze.py --stage merge.
mkjobs() { local kind=$1; shift; local J=$R/jobs_$kind.txt; : > $J
  for f in 0.4 0.3 0.25 0.2 0.15 0.1; do for k in "$@"; do local arm=$(armof $k) pool out B ch=""
      case $kind in AB) pool=$R/pool/$k; out=$R/boot/$k; B=$BN;; B) pool=$R/pool_B/$k; out=$R/boot_B/$k; B=$BN; ch="CH_RULE=$(chrule $k $f)";; A0) pool=$R/pool_A/$k; out=$R/boot_A0/$k; B=0;; esac
      mkdir -p $out
      if [ $B -eq 0 ]; then echo "MIN_CELL=1 FORCE_BEST7=$arm $ch OMP_NUM_THREADS=1 BOOT_B=0 BOOT_SEEDS=4 BOOT_FRAC=$f $PY exp/l9a_boot_forced.py $pool/feats_J.parquet judge_error cell100 $FEATS $out/boot_tb_f$f.json > $out/boot_f$f.log 2>&1; echo \$? > $out/exit_f$f" >> $J
      else for c in $(seq 0 $((CH-1))); do local s0=$((c*B/CH)) s1=$(((c+1)*B/CH))
        echo "MIN_CELL=1 FORCE_BEST7=$arm $ch OMP_NUM_THREADS=1 BOOT_B=$B BOOT_START=$s0 BOOT_END=$s1 BOOT_SEEDS=4 BOOT_FRAC=$f $PY exp/l9a_boot_forced.py $pool/feats_J.parquet judge_error cell100 $FEATS $out/boot_tb_f$f.c$c.json > $out/boot_f$f.c$c.log 2>&1; echo \$? > $out/exit_f$f.c$c" >> $J
      done; fi
    done; done; }
[ "${L9A_LIB_ONLY:-0}" = 1 ] && { return 0 2>/dev/null || exit 0; }

# Stage 0 (CPU): reference table of orig pids 0..3999 (p0 pids 0..1999 followed by the L8c-A groups with pid = orig_pid), done once in the shell: runs/L9a-shardB-tiebreak/ref_groups_0_3999.parquet.
# Checked: gsm8k_train.sample(n=6000, random_state=0) is a prefix-extension of sample(n=4000, random_state=0) (equal on the first 4000 questions), 6000 distinct questions, so pids 4000..5999 share no question with pids 0..3999.
# Design-only power check (shard A alone, before any GPU job): 8 replays of the L8c-A pool at 15 / 20 / 30 / 40% (10 / 25% reused from L8c-A) plus gate (i) (forced-BEST7 replay of the pids 0..1999 pools at 25%, 2 jobs, run together, 10 jobs in parallel, 426 s):
#   see runs/L9a-shardB-tiebreak/power_jobs.txt; then: $PY exp/l9a_analyze.py --run_dir $R --stage power   (writes power_check/power.json). Cells for shard B (label-free): $PY exp/l9a_analyze.py --run_dir $R --stage cells

# Stage 1, GPU job A (about 12 min): 2,000 fresh prompts, T=0.25, G=8, Qwen2.5-1.5B-Instruct, orig pids 4000..5999
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R/rollouts "$PYJ exp/l9a_rollouts_fresh.py --model $F/models/Qwen2.5-1.5B-Instruct --temps 0.25 --pid_start 4000 --n_prompts 2000 --G 8 --max_tokens 512 --seed 0 --ref_groups $R/ref_groups_0_3999.parquet --out $R/rollouts"; poll $R/rollouts

# Stage 2, GPU jobs B and C, one at a time
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/rm "$PYJ exp/l5c_proxy_score.py --kinds rm --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out $R/rm"; poll $R/rm
$F/bin/gpu_status.sh
$F/bin/gpu_run.sh $R/q3_4b "$PYJ exp/l3b_judge_score.py --model $F/models/Qwen3-4B --rollouts $R/rollouts/T0.25/rollouts.parquet --groups $R/rollouts/T0.25/groups.parquet --per_group 2 --out $R/q3_4b"; poll $R/q3_4b

# Stage 3, CPU. build (pooled, A-only and B-only feats_J per proxy), chA (CH_A per budget from shard-A files only), then the replays: pooled AB (verdict), B alone with CH_RULE (sign check + L11b secondary).
$PY exp/l9a_analyze.py --run_dir $R --stage build > $R/build.log 2>&1
mkjobs A0 rm q3_4b; xargs -P 12 -I{} bash -c '{}' < $R/jobs_A0.txt; $PY exp/l9a_analyze.py --run_dir $R --stage chA > $R/chA.log 2>&1
mkjobs AB rm q3_4b; mkjobs B rm q3_4b; cat $R/jobs_AB.txt $R/jobs_B.txt > $R/jobs_all.txt
xargs -P ${PAR:-28} -I{} bash -c '{}' < $R/jobs_all.txt; grep -l "GATE FAILED\|Traceback" $R/boot/*/*.log $R/boot_B/*/*.log; cat $R/boot/*/exit_f* $R/boot_B/*/exit_f* | sort | uniq -c
$PY exp/l9a_analyze.py --run_dir $R --stage merge > $R/merge.log 2>&1
$PY exp/l9a_analyze.py --run_dir $R --stage final > $R/final.log 2>&1; echo $? > $R/exit_code; touch $R/done.flag
