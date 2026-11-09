#!/bin/bash
cd /homes/rdubash/active-audit
source .venv/bin/activate
python runs/hyp/score_sample.py > runs/hyp/score.log 2>&1
echo "score exit $?" >> runs/hyp/chain.log
python runs/hyp/analyze_hyp.py > runs/hyp/analyze.log 2>&1
echo "analyze exit $?" >> runs/hyp/chain.log
