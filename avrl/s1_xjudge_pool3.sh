#!/bin/bash
# Cross-family judges on the triple-learning-rate arm: peak (step 1400) and final (step 4000). Same units as s1_xjudge_pool.sh.
cd "$HOME/active-audit" || exit 1
X=avrl/runs/s1_xjudge
HOSTS=(willow walnut pine cherry)
GEN_ARMS=(s1_proxy_lr3_s0_peak1400 s1_proxy_lr3_s0)
JUDGES=("phi4 microsoft/phi-4" "olmo7b allenai/OLMo-2-1124-7B-Instruct" "olmo13b allenai/OLMo-2-1124-13B-Instruct")
run() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "bash ~/active-audit/avrl/s1_xjudge.sh ${@:2}" 2>&1; }
echo "=== stage 1 (gen) $(date -u)"
i=0; for a in "${GEN_ARMS[@]}"; do run "${HOSTS[$i]}" gen "$a" & i=$((i+1)); done; wait
echo "=== stage 2 (judge) $(date -u)"
units=(); for j in "${JUDGES[@]}"; do for s in "${GEN_ARMS[@]}"; do units+=("$s $j"); done; done
for h in "${!HOSTS[@]}"; do
  ( for k in "${!units[@]}"; do [ $((k % ${#HOSTS[@]})) -eq $h ] && run "${HOSTS[$h]}" judge ${units[$k]}; done ) &
done; wait
echo "=== ALL DONE $(date -u)"
