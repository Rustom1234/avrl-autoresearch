#!/bin/bash
# E10 (OLMo-2-0425-1B code RL, MBPP/HumanEval) on goldberg: gated smoke test of the learned
# arm (no training launched until the GPU is idle and no p1_judge_grpo run is in progress),
# then the two reference-bar evaluations (OLMo base, OLMo Instruct), no training involved.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
F=avrl/runs/e10_goldberg.flag
M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B
MI=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo' >/dev/null; do sleep 60; done

echo "=== smoke start $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python avrl/p1_code_grpo.py --arm learned --smoke --no_vllm --model $M --out avrl/runs/e10_smoke > avrl/runs/e10_smoke.log 2>&1
echo "smoke exit $? $(date -u)" >> $F

echo "=== base eval start $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python avrl/p1_code_grpo.py --eval_only $M --eval_only_n 200 --out avrl/runs/e10_eval_base > avrl/runs/e10_eval_base.log 2>&1
echo "base eval exit $? $(date -u)" >> $F

echo "=== instruct eval start $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python avrl/p1_code_grpo.py --eval_only $MI --eval_only_n 200 --out avrl/runs/e10_eval_instruct > avrl/runs/e10_eval_instruct.log 2>&1
echo "instruct eval exit $? $(date -u)" >> $F

echo "e10_goldberg done $(date -u)" >> $F
