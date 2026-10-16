#!/bin/bash
# Re-evaluate hannah's E10d seed 1 adapters (Qwen2.5-1.5B-Instruct) on the full MBPP test set and HumanEval,
# on goldberg, once the adapters have been copied locally (flag) and the GPU is free (after M15).
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/homes/rdubash/active-audit/avrl/runs
until [ -f $R/hannah_e10d_s1/copied.flag ]; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo|reeval_adapters' >/dev/null; do sleep 60; done
.venv-rl/bin/python avrl/exp/reeval_adapters.py --task code --eval_seed 0 --out $R/reeval/code_hannah_e10d_s1 --runs $R/hannah_e10d_s1/e10d_full $R/hannah_e10d_s1/e10d_proxy $R/hannah_e10d_s1/e10d_random $R/hannah_e10d_s1/e10d_threshold > $R/reeval/code_hannah_e10d_s1.log 2>&1
echo done > $R/reeval/code_hannah_e10d_s1.flag
