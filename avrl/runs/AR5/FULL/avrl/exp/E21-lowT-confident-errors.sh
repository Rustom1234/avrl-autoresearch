#!/bin/bash
# E21-lowT-confident-errors driver (the steps as actually run). Scripts: exp/e21_lowT_rollouts.py, exp/e21_lowT_analyze.py, exp/sim_e21.py.
F=/homes/rdubash/avrl_live/FULL; R=$F/avrl/runs/E21-lowT-confident-errors
$F/bin/gpu_status.sh   # must show < 2 GB used
$F/bin/gpu_run.sh $R "/homes/rdubash/active-audit/.venv-judge/bin/python exp/e21_lowT_rollouts.py --model $F/models/Qwen2.5-1.5B-Instruct --temps 0.25,0.5 --n_prompts 2000 --G 8 --max_tokens 512 --seed 0 --ref_groups $F/avrl/runs/p0/groups.parquet --out $R"
until [ -f $R/done.flag ]; do ls $R > /dev/null; sleep 30; done; cat $R/exit_code
# CPU analysis after exit_code 0 (writes result.json, table.md):
HF_HUB_OFFLINE=1 /homes/rdubash/active-audit/.venv/bin/python $F/avrl/exp/e21_lowT_analyze.py --run_dir $R --ref_run $F/avrl/runs/p0 --n_prompts 2000
