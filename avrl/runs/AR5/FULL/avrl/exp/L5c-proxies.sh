#!/bin/bash
# L5c-proxies driver (the steps as run). Scripts: exp/l5c_proxy_score.py (GPU, Skywork-Reward-V2-Qwen3-0.6B reward model and Qwen2.5-3B-Instruct Yes/No judge scores on the same 4,000 rows as L3b),
# exp/l5c_analyze.py (CPU, claim 3, one call per proxy), exp/sim_warm_boot_j2.py (CPU, claim 2 replay with agree_acc and blend50_acc in the fixed set, learner warm-started from the best fixed rule),
# exp/l5c_finalize.py (CPU, verdict and replay table, result.json). Analysis scripts were smoke-tested on synthetic scores (exp/l5c_smoke_synth.py, outputs kept in runs/L5c-proxies/_smoke_synth,
# variants null / planted / gatefail / failed) before any real score existed. The bins are registered in the docstrings of l5c_proxy_score.py and l5c_analyze.py.
F=/homes/rdubash/avrl_live/FULL; A=$F/avrl; E=$A/runs/E21-lowT-confident-errors; R=$A/runs/L5c-proxies; PY=/homes/rdubash/active-audit/.venv/bin/python; cd $A
FEATS="s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct"
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R "/homes/rdubash/active-audit/.venv-judge/bin/python exp/l5c_proxy_score.py --kinds rm,judge3b --rm_model /homes/rdubash/active-audit/models/Skywork-Reward-V2-Qwen3-0.6B --judge_model /homes/rdubash/active-audit/models/Qwen2.5-3B-Instruct --rollouts $E/T0.25/rollouts.parquet --groups $E/T0.25/groups.parquet --per_group 2 --probe_rows 200 --out $R"
until [ -f $R/done.flag ]; do ls $R > /dev/null; sleep 30; done; cat $R/exit_code
for K in rm judge3b; do OMP_NUM_THREADS=8 $PY exp/l5c_analyze.py --kind $K --run_dir $R/$K --e21 $E --out $R/$K > $R/$K/analyze.log 2>&1 & done; wait
mkdir -p $R/qwen3_4b_ref
for K in rm judge3b; do BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 OMP_NUM_THREADS=2 $PY exp/sim_warm_boot_j2.py $R/$K/feats_J.parquet judge_error cell100 $FEATS $R/$K/boot_f0.25.json > $R/$K/boot_f0.25.log 2>&1 & done
BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 OMP_NUM_THREADS=2 $PY exp/sim_warm_boot_j2.py $A/runs/L3b-judge-proxy/feats_J.parquet judge_error cell100 $FEATS $R/qwen3_4b_ref/boot_f0.25.json > $R/qwen3_4b_ref/boot_f0.25.log 2>&1 &
wait
$PY exp/l5c_finalize.py --run_dir $R > $R/finalize.log 2>&1 && touch $R/done.flag
