#!/bin/bash
# Re-evaluate the realizable-budget auto arm's final adapter on the full MBPP test set (goldberg, after the hannah set).
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/homes/rdubash/active-audit/avrl/runs
until [ -f $R/reeval/code_hannah.flag ]; do sleep 60; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo|reeval_adapters' >/dev/null; do sleep 60; done
.venv-rl/bin/python avrl/exp/reeval_adapters.py --task code --eval_seed 0 --out $R/reeval/code_e10e_up --runs $R/e10e_up_auto_s1 > $R/reeval/code_e10e_up.log 2>&1
echo done > $R/reeval/code_e10e_up.flag
