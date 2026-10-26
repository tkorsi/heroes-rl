#!/bin/bash
# Repeated resets under perf stat: one engine per run, one reused environment per seat (the probe default --seats 0,1
# alternates seats), with CPU, system calls, context switches and memory.
# Usage: soak.sh ROOT OUT BACKEND EPISODES [CONTROLLER] [extra probe arguments...]
# Environment: GAME_DATA, PERF (perf binary, default perf).
set -u
ROOT=${1:?root}; OUT=${2:?out}; BACKEND=${3:?backend}; N=${4:?episodes}; CTL=${5:-end_turn}
shift $(( $# < 5 ? $# : 5 ))
: ${GAME_DATA:?}
mkdir -p "$(dirname "$OUT")"  # paths are made absolute before cd
D=$(cd "$(dirname "$OUT")" && pwd) || exit 1; OUT=$D/$(basename "$OUT"); GAME_DATA=$(cd "$GAME_DATA" && pwd) || exit 1
case ${PERF:-perf} in */*) D=$(cd "$(dirname "$PERF")" && pwd) || exit 1; PERF=$D/$(basename "$PERF");; esac
cd "$ROOT"
${PERF:-perf} stat -x, -o "$OUT.perfstat" -e raw_syscalls:sys_enter,context-switches,cpu-migrations,task-clock,page-faults -- \
  .venv/bin/python -m validation.probes.validate_embedded_engine run --binary vcmi/build/bin/vcmiclient \
  --replay-binary vcmi/build/bin/vcmireplay --data "$GAME_DATA" --output "$OUT" --backend "$BACKEND" --controller "$CTL" \
  --map experiments/explorer/maps/hero-management-v1.vmap --episodes "$N" --replay none --audit none "$@" \
  > "$OUT.out" 2> "$OUT.err"
echo "exit $? $(grep -o '"verdict": "[a-z]*"' "$OUT.out" | head -1)"
