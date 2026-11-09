#!/bin/bash
# E10e-rel: burn-in plus auto budget with the new relative default (rel:0.25 of the measured base error).
# Same setting and seed as E10e so the two runs differ only in the target rule. Runs after M13 on goldberg.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10b_goldberg.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
until grep -q "m13 done" avrl/runs/m13.flag 2>/dev/null; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
echo "=== e10e-rel auto s1 start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_code_grpo.py --arm threshold --no_vllm --budget auto --burnin_steps 25 --target_corruption rel:0.25 --steps 300 --lr 1e-5 --seed 1 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10e_rel_auto_s1 > avrl/runs/e10e_rel_auto_s1.log 2>&1
echo "e10e-rel auto s1 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10e_rel_auto_s1.log | tail -1)" >> $F
echo "e10e-rel done $(date -u)" >> $F
