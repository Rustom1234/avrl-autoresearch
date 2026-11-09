#!/bin/bash
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
.venv-judge/bin/python avrl/exp/e05_selfverify.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
  --model models/Qwen2.5-1.5B-Instruct --out avrl/runs/E05/ > avrl/runs/E05/selfverify.log 2>&1
echo "selfverify exit $?" >> avrl/runs/E05/done.flag
.venv/bin/python avrl/exp/e05_structure.py --groups avrl/runs/p0/groups.parquet --selfverify avrl/runs/E05/selfverify.parquet \
  --feats avrl/runs/p0/feats.parquet --out avrl/runs/E05/result.json > avrl/runs/E05/structure.log 2>&1
echo "structure exit $?" >> avrl/runs/E05/done.flag
