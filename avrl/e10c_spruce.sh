#!/bin/bash
# E10c on spruce (A4000): proxy-only arm for 600 steps (twice E10b's horizon), same lr, to test whether the
# visible-test grader eventually gets gamed (false-positive group rate drifting up) when training runs longer.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10c.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
echo "=== proxy 600 steps s1 start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 9h .venv-rl/bin/python avrl/p1_code_grpo.py --arm proxy --no_vllm --budget 0.25 --steps 600 --max_hours 8.5 --lr 1e-5 --seed 1 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10c_proxy600_s1 > avrl/runs/e10c_proxy600_s1.log 2>&1
echo "proxy600 s1 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10c_proxy600_s1.log | tail -1)" >> $F
echo "e10c done $(date -u)" >> $F
