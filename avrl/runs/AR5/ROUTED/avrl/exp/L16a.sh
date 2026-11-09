#!/bin/bash
# L16a runbook. CPU only, runs directly in the local shell: no GPU job (nothing goes through bin/gpu_run.sh), no ssh, no pip.
# Source-split null for the registered test-row gain (avrl/exp/l16a_source_split_null.py, imports the L13a helpers unchanged).
#   gate              reproduces the L13a references (+1.222287 test rows, +4.723665 L2b) and the spend and overlap gates
#   cuts A B          random 1,319-prompt cuts of runs/L2b, cut indices A to B-1 (48 cuts; batches of 8 below)
#   boot A B          200 bootstrap draws of the registered test rows, draw indices A to B-1
#   replay            independent replay (avrl/exp/l16a_critic_indep.py), imports no experiment module
#   combine           verdict, result.json and done.flag (after replay, so the replay result is included)
# Each call stays under the 10 minute shell limit. Usage: L16a.sh gate | cuts A B | boot A B | replay | combine
R=/homes/rdubash/avrl_live/ROUTED
RUN=$R/avrl/runs/L16a
PY=/homes/rdubash/active-audit/.venv/bin/python
SCRIPT=$R/avrl/exp/l16a_source_split_null.py
REPLAY=$R/avrl/exp/l16a_critic_indep.py
mkdir -p "$RUN/logs"

case "$1" in
  gate)
    "$PY" -I "$SCRIPT" --mode gate --out "$RUN" > "$RUN/logs/gate.log" 2>&1
    echo $? > "$RUN/logs/gate.exit"; cat "$RUN/logs/gate.log"; cat "$RUN/logs/gate.exit" ;;
  cuts)
    "$PY" -I "$SCRIPT" --mode cuts --start "$2" --stop "$3" --out "$RUN" > "$RUN/logs/cuts_$2_$3.log" 2>&1
    echo "cuts $2-$3 exit $?"; cat "$RUN/logs/cuts_$2_$3.log" ;;
  boot)
    "$PY" -I "$SCRIPT" --mode boot --start "$2" --stop "$3" --out "$RUN" > "$RUN/logs/boot_$2_$3.log" 2>&1
    echo "boot $2-$3 exit $?"; cat "$RUN/logs/boot_$2_$3.log" ;;
  replay)
    mkdir -p "$RUN/critic"
    "$PY" -I "$REPLAY" --out "$RUN" > "$RUN/logs/replay.log" 2>&1
    echo "replay exit $?"; cat "$RUN/logs/replay.log" ;;
  combine)
    "$PY" -I "$SCRIPT" --mode combine --out "$RUN" > "$RUN/logs/combine.log" 2>&1
    echo "combine exit $?"; cat "$RUN/logs/combine.log" ;;
  *) echo "usage: $0 gate | cuts A B | boot A B | replay | combine" >&2; exit 2 ;;
esac
