#!/bin/bash
# waits for the judge (which waits for hyp scoring), then runs the influence test
cd /homes/rdubash/active-audit
while [ ! -f runs/judge/done.flag ] || [ ! -f models/Qwen3-0.6B-Base/config.json ]; do sleep 30; done
source .venv/bin/activate
python runs/infl/influence_by_domain.py > runs/infl/run.log 2>&1
echo "infl exit $?" >> runs/infl/done.flag
python runs/infl/analyze_infl.py > runs/infl/analyze.log 2>&1
echo "analyze exit $?" >> runs/infl/done.flag
