#!/bin/bash
# goldberg queue part 3: E10b seed 2 (threshold, random) then E10c (proxy-only arm, 600 steps, drift test).
# The A4000 boxes OOM at 8x8x512 with HF generation for OLMo 1B, so these run on the 5090 after part 2.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=avrl/runs/e10b_goldberg.flag; M=/homes/rdubash/active-audit/models/OLMo-2-0425-1B-Instruct
CUR=/homes/rdubash/active-audit/avrl/runs/E10p0/instruct_solvable_task_ids.json
until grep -q "part2 done" $F 2>/dev/null; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo' >/dev/null; do sleep 120; done
for arm in threshold random; do
  echo "=== $arm s2 start $(hostname) $(date -u)" >> $F
  CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_code_grpo.py --arm $arm --no_vllm --budget 0.25 --steps 300 --lr 1e-5 --seed 2 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10b_${arm}_s2 > avrl/runs/e10b_${arm}_s2.log 2>&1
  echo "$arm s2 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10b_${arm}_s2.log | tail -1)" >> $F
done
echo "=== e10c proxy600 s1 start $(hostname) $(date -u)" >> $F
CUDA_VISIBLE_DEVICES=0 timeout 6h .venv-rl/bin/python avrl/p1_code_grpo.py --arm proxy --no_vllm --budget 0.25 --steps 600 --max_hours 5.5 --lr 1e-5 --seed 1 --proxy visible --curriculum $CUR --model $M --out avrl/runs/e10c_proxy600_s1 > avrl/runs/e10c_proxy600_s1.log 2>&1
echo "proxy600 s1 exit $? $(date -u) $(grep -E 'trained .* steps' avrl/runs/e10c_proxy600_s1.log | tail -1)" >> $F
echo "e10b goldberg part3 done $(date -u)" >> $F
