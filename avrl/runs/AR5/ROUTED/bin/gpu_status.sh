#!/bin/bash
# Usage: bin/gpu_status.sh
# Prints GPU memory used and compute apps on this run's GPU host.
set -uo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(head -1 "$BASE/HOST")
on_host() { if [ "$HOST" = "$(hostname -s)" ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi; }
on_host "echo host \$(hostname -s); nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; echo compute apps:; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
