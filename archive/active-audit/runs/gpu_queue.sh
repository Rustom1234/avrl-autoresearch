#!/bin/bash
# sequential GPU queue: judge (already running) -> hyp scoring+analysis -> influence test. CPATH gives Triton the Python headers.
cd /homes/rdubash/active-audit
export CPATH=$HOME/.local/include/python3.12
while [ ! -f runs/judge/done.flag ]; do sleep 30; done
source .venv/bin/activate
python runs/hyp/score_sample.py > runs/hyp/score.log 2>&1; echo "score exit $?" >> runs/hyp/chain.log
python runs/hyp/analyze_hyp.py > runs/hyp/analyze.log 2>&1; echo "analyze exit $?" >> runs/hyp/chain.log
python runs/infl/influence_by_domain.py > runs/infl/run.log 2>&1; echo "infl exit $?" >> runs/infl/done.flag
python runs/infl/analyze_infl.py > runs/infl/analyze.log 2>&1; echo "analyze exit $?" >> runs/infl/done.flag
