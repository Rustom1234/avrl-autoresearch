#!/usr/bin/env bash
# AR5 snapshotter: one process for all runs. Usage: ar5_snapshot.sh BASE [BASE ...]
#
# Every 15 s, per BASE:
#   sha256 of avrl/RESEARCH_LOG.md (the ledger) and lit/avrl/EXPERIMENT_BACKLOG.md.
#   Ledger changed:  copy to snap/ledger_<epoch>_<sha8>.md, append "epoch sha8 n_lines" to
#                    snap/ledger_index.txt.
#   Backlog changed: copy to snap/backlog_<epoch>_<sha8>.md, append the same line format to
#                    snap/backlog_index.txt (the design hashes the backlog but names only the ledger copy).
#   The sha8 names the copied bytes (hashed after the copy), so a file rewritten mid-poll is still exact.
# Every 60 s, per BASE:
#   `ls -l --time-style=+%s avrl/runs` appended to snap/runs_<epoch>.txt only when it changed.
#   GPU memory used and compute-app count of the run's HOST (BASE/HOST) appended to snap/gpu.csv as
#   "epoch,mem_used_mib,n_procs" (no header; NA,NA when nvidia-smi or ssh fails). ssh is used only for
#   nvidia-smi, and only when HOST differs from `hostname -s`.
# A BASE is dropped when BASE/STOP_SNAPSHOT exists; the process exits when every BASE is dropped.
# Log: BASE/events/snapshot.log ("epoch message").
set -u
[ $# -ge 1 ] || { echo "usage: $0 BASE [BASE ...]" >&2; exit 2; }

ME=$(hostname -s)
declare -A LAST_LEDGER LAST_BACKLOG LAST_RUNS DONE
BASES=("$@")
for B in "${BASES[@]}"; do
  [ -d "$B" ] || { echo "no such BASE: $B" >&2; exit 2; }
  mkdir -p "$B/snap" "$B/events"
  LAST_LEDGER[$B]=""; LAST_BACKLOG[$B]=""; LAST_RUNS[$B]=""; DONE[$B]=0
done

log() { echo "$(date +%s) $2" >> "$1/events/snapshot.log"; }

# snap_file BASE SRC PREFIX LASTVAR_NAME: copy SRC to snap/PREFIX_<epoch>_<sha8>.md if its sha changed
snap_file() {
  local B=$1 src=$2 prefix=$3 lastname=$4 sha last tmp sha8 ep n
  [ -f "$src" ] || return 0
  sha=$(sha256sum "$src" | cut -c1-64)
  eval "last=\${$lastname[\$B]}"
  [ "$sha" = "$last" ] && return 0
  tmp="$B/snap/.tmp_$prefix"
  cp "$src" "$tmp" || { log "$B" "ERROR copy $src"; return 0; }
  sha=$(sha256sum "$tmp" | cut -c1-64)
  sha8=${sha:0:8}; ep=$(date +%s); n=$(wc -l < "$tmp")
  mv "$tmp" "$B/snap/${prefix}_${ep}_${sha8}.md"
  echo "$ep $sha8 $n" >> "$B/snap/${prefix}_index.txt"
  eval "$lastname[\$B]=\$sha"
  log "$B" "$prefix changed sha8=$sha8 n_lines=$n"
}

gpu_sample() {  # BASE; runs in a subshell, appends one gpu.csv row
  local B=$1 host out mem n ep
  host=""; [ -f "$B/HOST" ] && host=$(tr -d '[:space:]' < "$B/HOST")
  ep=$(date +%s)
  if [ -z "$host" ]; then echo "$ep,NA,NA" >> "$B/snap/gpu.csv"; log "$B" "ERROR no HOST file"; return; fi
  case "$host" in
    goldberg*|hannah*) echo "$ep,NA,NA" >> "$B/snap/gpu.csv"; log "$B" "ERROR forbidden host $host, not sampled"; return;;
  esac
  [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "$ep,NA,NA" >> "$B/snap/gpu.csv"; log "$B" "ERROR bad HOST content"; return; }
  local q='nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; echo ---; nvidia-smi --query-compute-apps=pid --format=csv,noheader'
  if [ "$host" = "$ME" ]; then
    out=$(timeout 20 bash -c "$q" 2>/dev/null)
  else
    out=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 -- "$host" "$q" 2>/dev/null)
  fi
  mem=$(echo "$out" | sed -n '1p' | tr -d '[:space:]')
  case "$mem" in ''|*[!0-9]*) echo "$ep,NA,NA" >> "$B/snap/gpu.csv"; log "$B" "ERROR gpu sample failed host=$host"; return;; esac
  n=$(echo "$out" | sed '1,/^---$/d' | grep -c '[0-9]')
  echo "$ep,$mem,$n" >> "$B/snap/gpu.csv"
}

LAST_SLOW=0
SLOW_S=${AR5_SNAP_SLOW_S:-60}   # test hook; the design value is 60
trap 'for B in "${BASES[@]}"; do log "$B" "snapshot terminated by signal"; done; exit 0' TERM INT
for B in "${BASES[@]}"; do log "$B" "snapshot start host=$ME pid=$$"; done

while :; do
  NOW=$(date +%s)
  SLOW=0
  [ $((NOW - LAST_SLOW)) -ge "$SLOW_S" ] && { SLOW=1; LAST_SLOW=$NOW; }
  ALIVE=0
  for B in "${BASES[@]}"; do
    [ "${DONE[$B]}" = 1 ] && continue
    if [ -e "$B/STOP_SNAPSHOT" ]; then
      # one last copy so the final ledger/backlog state is kept
      snap_file "$B" "$B/avrl/RESEARCH_LOG.md" ledger LAST_LEDGER
      snap_file "$B" "$B/lit/avrl/EXPERIMENT_BACKLOG.md" backlog LAST_BACKLOG
      DONE[$B]=1; log "$B" "STOP_SNAPSHOT seen, stopping"; continue; fi
    ALIVE=1
    snap_file "$B" "$B/avrl/RESEARCH_LOG.md" ledger LAST_LEDGER
    snap_file "$B" "$B/lit/avrl/EXPERIMENT_BACKLOG.md" backlog LAST_BACKLOG
    if [ $SLOW = 1 ]; then
      R=$(cd "$B" && ls -l --time-style=+%s avrl/runs 2>&1)
      if [ "$R" != "${LAST_RUNS[$B]}" ]; then
        printf '%s\n' "$R" > "$B/snap/runs_${NOW}.txt"
        LAST_RUNS[$B]=$R
      fi
      gpu_sample "$B" &
    fi
  done
  wait
  [ $ALIVE = 0 ] && break
  # next tick 15 s after this one started
  SLEEP=$((NOW + 15 - $(date +%s)))
  [ $SLEEP -gt 0 ] && { sleep $SLEEP & wait $!; }  # wait so a TERM trap fires at once
done
exit 0
