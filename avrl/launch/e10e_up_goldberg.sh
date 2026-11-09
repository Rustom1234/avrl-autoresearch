#!/bin/bash
# E10e-up: burn-in plus auto budget with realizable candidates and the upper-bound pick (PRODUCT-4), same seed as E10e.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10b_goldberg.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo|reeval_adapters' >/dev/null; do sleep 120; done
echo "=== e10e-up auto s1 start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_code_grpo.py --arm threshold --no_vllm --budget auto --burnin_steps 25 --target_corruption rel:0.25 --budget_pick upper --eval_seed 0 --steps 300 --lr 1e-5 --seed 1 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10e_up_auto_s1 > avrl/runs/e10e_up_auto_s1.log 2>&1
echo "e10e-up auto s1 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10e_up_auto_s1.log | tail -1)" >> $F
echo "e10e-up done $(date -u)" >> $F
