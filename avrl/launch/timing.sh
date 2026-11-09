#!/bin/bash
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12 FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 30; done
for mode in vllm hf; do flag=""; [ $mode = hf ] && flag="--no_vllm"
  CUDA_VISIBLE_DEVICES=0 timeout 40m .venv-rl/bin/python avrl/p1_grpo.py --arm threshold --steps 3 --eval_every 1000 --eval_n 20 --ckpt_every 1000 $flag --out avrl/runs/timing_$mode --model models/Qwen2.5-1.5B-Instruct > avrl/runs/timing_$mode.log 2>&1
  echo "$mode exit $? $(grep -E 'trained .* steps' avrl/runs/timing_$mode.log | tail -1) $(grep -E 'peak_gpu' avrl/runs/timing_$mode.log | tail -1)" >> avrl/runs/timing.flag
done
