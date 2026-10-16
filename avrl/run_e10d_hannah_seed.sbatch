#!/bin/bash
# E10d: second policy family for the visible-test result. Qwen2.5-1.5B-Instruct on MBPP train (all 464
# problems, no curriculum: it solves 65 percent at pass@8), cheap grader = visible first assert, expensive =
# full assert set; four arms (full / proxy / random / threshold at 25 percent), 300 steps, then the untrained
# reference evaluation. Same gate structure as run_e10b_hannah.sbatch.
#SBATCH --job-name=e10d_seed
#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 32
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/nfs/home/rustom/.cache/huggingface
cd ~/avrl; mkdir -p runs; PY=~/avrl/.venv/bin/python
F=runs/e10d_s${SEED}_status.flag
M=~/avrl/models/Qwen2.5-1.5B-Instruct
COMMON="--no_vllm --budget 0.25 --lr 1e-5 --seed ${SEED:?} --eval_seed 0 --proxy visible --model $M"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== smoke test (threshold arm, visible proxy, Qwen 1.5B) $(date -u)"
CUDA_VISIBLE_DEVICES=0 timeout 25m $PY p1_code_grpo.py --arm threshold --smoke $COMMON --out runs/e10d_smoke_s${SEED} > runs/e10d_smoke_s${SEED}.log 2>&1
rc=$?; echo "smoke exit $rc"; tail -5 runs/e10d_smoke_s${SEED}.log
if [ $rc -ne 0 ] || ! ls runs/e10d_smoke_s${SEED}/*.jsonl >/dev/null 2>&1; then echo "SMOKE FAILED, not launching arms" > $F; exit 1; fi
echo "=== launching four arms $(date -u)"
for spec in "0 full" "1 proxy" "2 random" "3 threshold"; do set -- $spec
  rm -f runs/e10d_${2}_s${SEED}/log.jsonl runs/e10d_${2}_s${SEED}/groups.jsonl
  CUDA_VISIBLE_DEVICES=$1 $PY p1_code_grpo.py --arm $2 --steps 300 $COMMON --out runs/e10d_${2}_s${SEED} > runs/e10d_${2}_s${SEED}.log 2>&1 &
done
wait
echo "=== arms finished $(date -u)"
for a in full proxy random threshold; do echo "$a exit: $(grep -E 'trained .* steps' runs/e10d_$a.log | tail -1)"; done
echo "arms done $(date -u)" > $F
