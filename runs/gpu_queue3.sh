#!/bin/bash
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12
while [ ! -f runs/judge/done_hyp.flag ]; do sleep 30; done
source .venv/bin/activate
python runs/hyp/unified.py > runs/hyp/unified.log 2>&1; echo "unified exit $?" >> runs/hyp/unified.flag
