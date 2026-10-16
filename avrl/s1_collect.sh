#!/bin/bash
# Copy the S1 arm logs (and final adapters when present) from hannah into avrl/runs/hannah_s1/ and run the
# pre-registered analysis. Usage: bash avrl/s1_collect.sh [--adapters]
cd /homes/rdubash/active-audit; D=avrl/runs/hannah_s1; mkdir -p $D
for n in proxy_s0 thrcons_s0 full_s0 proxy_lr3_s0 thrcons_s1 full_s1; do
  mkdir -p $D/s1_$n
  scp -q rustom@hannah:avrl/runs/s1_$n/log.jsonl $D/s1_$n/log.jsonl 2>/dev/null && echo "copied s1_$n log ($(grep -c train_step $D/s1_$n/log.jsonl) steps)"
  scp -q rustom@hannah:avrl/runs/s1_$n/groups.jsonl $D/s1_$n/groups.jsonl 2>/dev/null
  if [ "$1" = "--adapters" ]; then mkdir -p $D/s1_$n/final; scp -q -r "rustom@hannah:avrl/runs/s1_$n/final/." $D/s1_$n/final/ 2>/dev/null && sed -i 's|/nfs/home/rustom/avrl/models/|/homes/rdubash/active-audit/models/|; s|/home/rustom/avrl/models/|/homes/rdubash/active-audit/models/|' $D/s1_$n/final/adapter_config.json 2>/dev/null && echo "copied s1_$n adapter"; fi
done
.venv/bin/python avrl/exp/S1_analyze.py --runs_dir $D --out avrl/runs/s1_analysis
