#!/bin/bash
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-judge/bin/python avrl/exp/E10p0_rollouts.py \
  --model /homes/rdubash/active-audit/models/OLMo-2-0425-1B \
  --out avrl/runs/E10p0/base \
  > avrl/runs/E10p0/base.log 2>&1
echo "exit $?" > avrl/runs/E10p0/base/done.flag
