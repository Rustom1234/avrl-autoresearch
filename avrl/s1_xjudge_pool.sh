#!/bin/bash
# Dispatches the cross-family judge check over the lab pool (plain ssh, user level).
# Stage 1: one generation per host in parallel. Stage 2: judge units round-robin over hosts.
cd "$HOME/active-audit" || exit 1
X=avrl/runs/s1_xjudge
HOSTS=(willow walnut pine cherry)
GEN_ARMS=(base s1_proxy_s0 s1_thrcons_s0)
SETS=(base s1_proxy_s0 s1_thrcons_s0 _goldberg_bf16_s1_proxy_s0)
JUDGES=("phi4 microsoft/phi-4" "olmo7b allenai/OLMo-2-1124-7B-Instruct")
run() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "bash ~/active-audit/avrl/s1_xjudge.sh ${@:2}" 2>&1; }

echo "=== stage 1 (gen) $(date -u)"
i=0
for a in "${GEN_ARMS[@]}"; do run "${HOSTS[$i]}" gen "$a" & i=$((i+1)); done
wait
until grep -q DONE $X/download_phi4.log && grep -q DONE $X/download_olmo7b.log; do sleep 30; done

echo "=== stage 2 (judge) $(date -u)"
units=()
for j in "${JUDGES[@]}"; do for s in "${SETS[@]}"; do units+=("$s $j"); done; done
for h in "${!HOSTS[@]}"; do
  ( for k in "${!units[@]}"; do [ $((k % ${#HOSTS[@]})) -eq $h ] && run "${HOSTS[$h]}" judge ${units[$k]}; done ) &
done
wait
echo "=== ALL DONE $(date -u)"
