#!/bin/bash
# The host B measurements behind validation/linux-acceleration-2026-10-26.json, sequentially on an otherwise idle
# machine, in the order they ran there: cold A/B, soak, scaling, v7 controller replay, embedded A/B, then (in later
# sessions on the same host) the two-pass, audit, v8 replay and review-fix measurements. Writes under OUT the host B
# directories and line files that record-notes-2026-10-26.json names. record.py then builds the record; OUT must
# also hold host A's ab-final/ and scale-host-a/results.txt and the capture_counts.py summary at
# accel-evidence-final/capture-verification.json (see docs/embedded-acceleration.md#reproducing):
#   python3 tools/acceleration/record.py tools/acceleration/record-notes-2026-10-26.json OUT RECORD.json
# Usage: measure.sh OUT
# Environment: BASE_ROOT and ACCEL_ROOT (worktrees with a prepared vcmi/build and .venv), GAME_DATA, POLICY_V7 and
# POLICY_V8 (the base code's 4-game policy-embedded runs from revalidate.sh: the recorded workloads resubmit their
# actions and the engine-free replays read their episodes), PERF (perf binary, default perf).
set -u
OUT=${1:?out}
: ${BASE_ROOT:?} ${ACCEL_ROOT:?} ${GAME_DATA:?} ${POLICY_V7:?} ${POLICY_V8:?}
mkdir -p "$OUT"  # the scripts below change directory, so every path is made absolute
OUT=$(cd "$OUT" && pwd); GAME_DATA=$(cd "$GAME_DATA" && pwd)
POLICY_V7=$(cd "$POLICY_V7" && pwd); POLICY_V8=$(cd "$POLICY_V8" && pwd)
BASE_ROOT=$(cd "$BASE_ROOT" && pwd); ACCEL_ROOT=$(cd "$ACCEL_ROOT" && pwd)
T=$(cd "$(dirname "$0")" && pwd)
export GAME_DATA RECORDED_V7=$POLICY_V7 RECORDED_V8=$POLICY_V8 BASE_ROOT ACCEL_ROOT
mkdir -p "$OUT/soak-final" "$OUT/scale-final"
prune() { find "$1" -mindepth 2 -maxdepth 2 -type d -name 'episode-*' -exec rm -rf -- {} + 2>/dev/null; true; }
root() { if [ "$1" = accel ]; then echo "$ACCEL_ROOT"; else echo "$BASE_ROOT"; fi; }

BACKEND=cold "$T/ab.sh" "$OUT/ab-cold-final" 3; prune "$OUT/ab-cold-final"

for i in 1 2 3; do for n in base accel; do
  "$T/soak.sh" "$(root $n)" "$OUT/soak-final/$n-embedded-$i" embedded 24 end_turn
done; done
for n in base accel; do
  "$T/soak.sh" "$(root $n)" "$OUT/soak-final/$n-cold" cold 8 end_turn
  "$T/soak.sh" "$(root $n)" "$OUT/soak-final/$n-embedded-policy" embedded 12 policy --random-policy-seed 21 \
    --choices 1024 --max-days 365
done
prune "$OUT/soak-final"

for k in 1 2 4; do for n in accel base; do
  echo "$n k=$k $("$T/scale.sh" "$(root $n)" "$OUT/scale-final/$n-$k" $k 4 | tail -1)" >> "$OUT/scale-final/results.txt"
  rm -rf "$OUT/scale-final/$n-$k"/w*/episode-*
done; done

E7=$(ls -d "$POLICY_V7"/episode-*); E8=$(ls -d "$POLICY_V8"/episode-*); V8="--interface adventure-capabilities-v1"
line() {  # line FILE CONFIG SCRIPT ARGS...: one interleaved repeat of a timing script in one checkout
  local file=$1 n=$2 script=$3; shift 3
  local typed=""; [ $n = accel ] && [ "$script" = controller_replay.py ] && typed=--typed
  echo "$n $(HEROES_ROOT=$(root $n) "$(root $n)/.venv/bin/python" "$T/$script" $typed "$@" 2>/dev/null | tail -1)" >> "$file"
}
for i in 1 2 3; do for n in base accel; do line "$OUT/scale-final/controller.txt" $n controller_replay.py $E7; done; done
BACKEND=embedded "$T/ab.sh" "$OUT/ab-final-b" 3; prune "$OUT/ab-final-b"
for i in 1 2 3; do for n in base accel; do
  line "$OUT/scale-final/controller-passes.txt" $n controller_replay.py --repeat 2 $E7
done; done
for i in 1 2 3; do for n in base accel; do
  line "$OUT/scale-final/audit-passes.txt" $n audit_passes.py $E7
  line "$OUT/scale-final/controller-v8.txt" $n controller_replay.py $V8 $E8
  line "$OUT/scale-final/audit-passes-v8.txt" $n audit_passes.py $V8 $E8
done; done
HEROES_ROOT=$ACCEL_ROOT "$ACCEL_ROOT/.venv/bin/python" "$T/review_fix_costs.py" $E7 2>/dev/null | tail -1 \
  > "$OUT/scale-final/review-fix-costs.json"
