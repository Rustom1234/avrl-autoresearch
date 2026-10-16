#!/bin/bash
# Hannah launcher for E10b: OLMo 2 1B Instruct, visible-test cheap grader vs full-test verifier,
# curriculum of the 108 Instruct-solvable MBPP problems. Smoke gate on GPU0 (fail closed), then
# four arms in parallel (full / proxy / random / threshold), then nothing else (reference evals
# for base and Instruct already exist from E10). Requires p1_code_grpo.py with --proxy and
# --curriculum (synced from spruce) and runs/E10p0/instruct_solvable_task_ids.json.
#SBATCH --job-name=e10b_code_grpo
#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 32
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/nfs/home/rustom/.cache/huggingface
cd ~/avrl; mkdir -p runs; PY=~/avrl/.venv/bin/python
F=runs/e10b_status.flag
M=~/avrl/models/OLMo-2-0425-1B-Instruct
CUR=~/avrl/runs/E10p0/instruct_solvable_task_ids.json
COMMON="--no_vllm --budget 0.25 --lr 1e-5 --seed 0 --proxy visible --curriculum $CUR --model $M"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== smoke test (threshold arm, visible proxy) $(date -u)"
CUDA_VISIBLE_DEVICES=0 timeout 25m $PY p1_code_grpo.py --arm threshold --smoke $COMMON --out runs/e10b_smoke > runs/e10b_smoke.log 2>&1
rc=$?; echo "smoke exit $rc"; tail -5 runs/e10b_smoke.log
if [ $rc -ne 0 ] || ! ls runs/e10b_smoke/*.jsonl >/dev/null 2>&1; then echo "SMOKE FAILED, not launching arms" > $F; exit 1; fi
echo "=== launching four arms $(date -u)"
for spec in "0 full" "1 proxy" "2 random" "3 threshold"; do set -- $spec
  CUDA_VISIBLE_DEVICES=$1 $PY p1_code_grpo.py --arm $2 --steps 300 $COMMON --out runs/e10b_$2 > runs/e10b_$2.log 2>&1 &
done
wait
echo "=== arms finished $(date -u)"
for a in full proxy random threshold; do echo "$a exit: $(grep -E 'trained .* steps' runs/e10b_$a.log | tail -1)"; done
echo "arms done $(date -u)" > $F
