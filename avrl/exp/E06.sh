#!/bin/bash
# E06: model-size axis, majority-vote proxy, Qwen2.5-0.5B-Instruct (vs E01's Qwen2.5-1.5B-Instruct).
# Reuses p0_rollouts.py unmodified (only --model/--out differ from E01); analysis is e06_analyze_rundir.py
# (p0_analyze.py copy with --run_dir added, otherwise byte-identical).
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
mkdir -p avrl/runs/E06
.venv-judge/bin/python avrl/p0_rollouts.py --model Qwen/Qwen2.5-0.5B-Instruct --n_prompts 3000 --G 8 --seed 0 --out avrl/runs/E06 > avrl/runs/E06/rollouts.log 2>&1
echo "rollouts exit $?" >> avrl/runs/E06/done.flag
.venv/bin/python avrl/exp/e06_analyze_rundir.py --run_dir avrl/runs/E06 > avrl/runs/E06/analyze.log 2>&1
echo "analyze exit $?" >> avrl/runs/E06/done.flag
