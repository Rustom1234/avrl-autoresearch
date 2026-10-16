#!/bin/bash
# influence test rerun after the unified analysis (scores.parquet domain column repaired)
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12
while [ ! -f runs/hyp/unified.flag ]; do sleep 30; done
source .venv/bin/activate
rm -f runs/infl/done.flag
python runs/infl/influence_by_domain.py > runs/infl/run.log 2>&1; echo "infl exit $?" >> runs/infl/done.flag
python runs/infl/analyze_infl.py > runs/infl/analyze.log 2>&1; echo "analyze exit $?" >> runs/infl/done.flag
