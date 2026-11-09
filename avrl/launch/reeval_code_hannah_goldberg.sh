#!/bin/bash
# Re-evaluate hannah's seed-0 code adapters (E10b OLMo, E10d Qwen) on the full MBPP test set and HumanEval
# with the common script, on goldberg after E10e-up. The E10d rows need the Qwen base reference row too.
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/homes/rdubash/active-audit/avrl/runs
until grep -q "e10e-up done" $R/e10b_goldberg.flag 2>/dev/null; do sleep 60; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] || pgrep -f 'p1_judge_grpo|p1_grpo|p1_code_grpo|reeval_adapters' >/dev/null; do sleep 60; done
.venv-rl/bin/python avrl/exp/reeval_adapters.py --task code --eval_seed 0 --out $R/reeval/code_hannah_e10b --runs $R/hannah_e10b/e10b_full $R/hannah_e10b/e10b_proxy $R/hannah_e10b/e10b_random $R/hannah_e10b/e10b_threshold > $R/reeval/code_hannah_e10b.log 2>&1
.venv-rl/bin/python avrl/exp/reeval_adapters.py --task code --eval_seed 0 --out $R/reeval/code_hannah_e10d --runs $R/hannah_e10d/e10d_full $R/hannah_e10d/e10d_proxy $R/hannah_e10d/e10d_random $R/hannah_e10d/e10d_threshold > $R/reeval/code_hannah_e10d.log 2>&1
echo done > $R/reeval/code_hannah.flag
