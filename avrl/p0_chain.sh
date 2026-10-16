#!/bin/bash
cd /homes/rdubash/active-audit; export CPATH=$HOME/.local/include/python3.12
source .venv-judge/bin/activate; python avrl/p0_rollouts.py > avrl/runs/p0/run.log 2>&1; echo "rollouts exit $?" >> avrl/runs/p0/done.flag; deactivate
source .venv/bin/activate; python avrl/p0_analyze.py > avrl/runs/p0/analyze.log 2>&1; echo "analyze exit $?" >> avrl/runs/p0/done.flag
