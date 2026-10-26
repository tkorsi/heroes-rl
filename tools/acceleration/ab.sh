#!/bin/bash
# Interleaved A/B of real games: each round runs every workload once per configuration, base then accelerated,
# 4 games per run (seats red, blue, red, blue), every ended game replayed and every policy game audited.
# Usage: ab.sh OUT ROUNDS [WORKLOADS]
# Environment: BASE_ROOT and ACCEL_ROOT (worktrees with a prepared vcmi/build), GAME_DATA, RECORDED_V7 and
# RECORDED_V8 (base-code policy runs whose delivered actions the recorded rows resubmit), BACKEND (embedded or cold).
set -u
OUT=${1:?out}; N=${2:-3}; WORKLOADS=${3:-"v7-recorded v7-policy v8-recorded v8-policy"}
: ${BASE_ROOT:?} ${ACCEL_ROOT:?} ${GAME_DATA:?} ${RECORDED_V7:?} ${RECORDED_V8:?}
BACKEND=${BACKEND:-embedded}
mkdir -p "$OUT"  # each run changes directory, so the paths are made absolute
OUT=$(cd "$OUT" && pwd) || exit 1; GAME_DATA=$(cd "$GAME_DATA" && pwd) || exit 1
RECORDED_V7=$(cd "$RECORDED_V7" && pwd) || exit 1; RECORDED_V8=$(cd "$RECORDED_V8" && pwd) || exit 1
for i in $(seq 1 "$N"); do
  for w in $WORKLOADS; do
    case $w in
      v7-*) iface=adventure-dialogs-v1; rec=$RECORDED_V7;;
      v8-*) iface=adventure-capabilities-v1; rec=$RECORDED_V8;;
    esac
    case $w in
      *-recorded) ctl="--controller recorded --recorded $rec";;
      *-policy) ctl="--controller policy --random-policy-seed 21";;
    esac
    for name in base accel; do
      root=$BASE_ROOT; [ $name = accel ] && root=$ACCEL_ROOT
      dir=$OUT/$w-$name-$i
      echo "== $w $name round $i $(date -u +%T)"
      (cd "$root" && .venv/bin/python -m validation.probes.validate_embedded_engine run \
        --binary vcmi/build/bin/vcmiclient --replay-binary vcmi/build/bin/vcmireplay --data "$GAME_DATA" \
        --output "$dir" --backend "$BACKEND" $ctl --interface $iface \
        --map experiments/explorer/maps/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365 \
        > "$dir.out" 2> "$dir.err")
      echo "exit $? $(grep -o '"verdict": "[a-z]*"' "$dir.out" | head -1)"
    done
  done
done
