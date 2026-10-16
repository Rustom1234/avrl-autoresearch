#!/bin/bash
cd /homes/rdubash/active-audit; source .venv/bin/activate; rm -f runs/code/chain.flag
python runs/code/sim_code.py any_secret > runs/code/sim_any.log 2>&1; echo "sim_any exit $?" >> runs/code/chain.flag
python runs/code/sim_code.py high > runs/code/sim_high.log 2>&1; echo "sim_high exit $?" >> runs/code/chain.flag
