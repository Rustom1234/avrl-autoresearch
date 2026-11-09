#!/bin/bash
cd /homes/rdubash/active-audit; source .venv/bin/activate
C="token_count,language_score,email,phone,ipv4,street,any_pii,log_tc"; E=$(python -c "print(','.join(f'pc{i}' for i in range(32)))")
for g in domain cell300; do for name in cheap cheap+emb; do
  cols=$C; [ "$name" = "cheap+emb" ] && cols="$C,$E"
  echo "=== ablation sim judge by $g, static features = $name"
  python runs/sim_generic.py runs/hyp/unified_feats.parquet judge $g $cols runs/hyp/sima_judge_${g}_${name}.json 2>&1 | grep -E '^sim docs|^\| (1|3|10)%'
done; done > runs/hyp/sims_hybrid.log 2>&1; echo done >> runs/hyp/sims_hybrid.flag
