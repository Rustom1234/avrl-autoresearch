#!/bin/bash
# Orchestrates the cross-family judge check end to end on one 32 GB GPU.
cd "$HOME/active-audit" || exit 1
X=avrl/runs/s1_xjudge
bash avrl/s1_xjudge.sh gen
until grep -q DONE $X/download_phi4.log; do sleep 30; done
bash avrl/s1_xjudge.sh judge phi4 microsoft/phi-4
until grep -q DONE $X/download_olmo7b.log; do sleep 30; done
bash avrl/s1_xjudge.sh judge olmo7b allenai/OLMo-2-1124-7B-Instruct
echo "=== ALL DONE $(date -u)"
