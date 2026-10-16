#!/bin/bash
# Follow-on: score every response set with OLMo-2 13B (4-bit) once the first pool run and the download finish.
cd "$HOME/active-audit" || exit 1
X=avrl/runs/s1_xjudge
HOSTS=(willow walnut pine cherry)
SETS=(base s1_proxy_s0 s1_thrcons_s0 _goldberg_bf16_s1_proxy_s0)
until grep -q "ALL DONE" $X/pool.log && grep -q DONE $X/download_olmo13b.log; do sleep 60; done
echo "=== olmo13b stage $(date -u)"
for i in "${!SETS[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${HOSTS[$i]}" "bash ~/active-audit/avrl/s1_xjudge.sh judge ${SETS[$i]} olmo13b allenai/OLMo-2-1124-13B-Instruct" 2>&1 &
done
wait
echo "=== ALL DONE $(date -u)"
