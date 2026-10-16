#!/bin/bash
cd /homes/rdubash/active-audit
while [ "$(ls data/github-code/data/*.parquet 2>/dev/null | wc -l)" -lt 10 ]; do sleep 30; done
sleep 20
source .venv/bin/activate
python runs/code/secrets_by_repo.py > runs/code/run10.log 2>&1; echo "secrets10 exit $?" >> runs/code/chain.flag
python runs/code/sim_code.py any_secret > runs/code/sim_any.log 2>&1; echo "sim_any exit $?" >> runs/code/chain.flag
python runs/code/sim_code.py high > runs/code/sim_high.log 2>&1; echo "sim_high exit $?" >> runs/code/chain.flag
