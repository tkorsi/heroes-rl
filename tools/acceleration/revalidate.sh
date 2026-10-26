#!/bin/bash
# Real-game revalidation (v7 and v8, both backends, both seats), sequential, one scenario per probe process,
# then matched comparisons. Set HEROES_VERIFY_NATIVE_CAPTURE=1 to build and compare every skipped capture.
# Usage: revalidate.sh OUTPUT_ROOT     Environment: ROOT (worktree with a prepared vcmi/build), GAME_DATA.
set -u
: ${ROOT:?} ${GAME_DATA:?}
cd "$ROOT"
R=${1:?output root}
mkdir -p $R
B=vcmi/build/bin; D=$GAME_DATA; M=experiments/explorer/maps
P=".venv/bin/python -m validation.probes.validate_embedded_engine"
COMMON="--binary $B/vcmiclient --replay-binary $B/vcmireplay --data $D"
V8="--interface adventure-capabilities-v1"
run() { name=$1; shift; echo "== $name $(date -u +%T)"; $P run $COMMON --output $R/$name "$@" > $R/$name.out 2> $R/$name.err; echo "exit $? $(date -u +%T)"; }
for backend in embedded cold; do
  run endturn-$backend --backend $backend --controller end_turn --map $M/hero-management-v1.vmap --episodes 6
  run policy-$backend --backend $backend --controller policy --random-policy-seed 21 --map $M/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365
  run chest-red-$backend --backend $backend --controller visit --visit-kind 101 --seats 0 --map $M/dialog-chest-red-v1.vmap --episodes 2
  run chest-blue-$backend --backend $backend --controller visit --visit-kind 101 --seats 1 --map $M/dialog-chest-blue-v1.vmap --episodes 2
  run mine-red-$backend --backend $backend --controller visit --visit-kind 53 --seats 0 --map $M/dialog-guarded-mine-red-v1.vmap --episodes 2
  run poison-$backend --backend $backend --controller stall --bridge-timeout 5 --stall-seconds 8 --map $M/hero-management-v1.vmap --episodes 2
  run v8-endturn-$backend --backend $backend $V8 --controller end_turn --map $M/hero-management-v1.vmap --episodes 4
  run v8-policy-$backend --backend $backend $V8 --controller policy --random-policy-seed 21 --map $M/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365
  run v8-levelup-red-$backend --backend $backend $V8 --controller visit --visit-kind 101 --seats 0 --map $M/capability-level-up-red-v1.vmap --episodes 2
  run v8-levelup-blue-$backend --backend $backend $V8 --controller visit --visit-kind 101 --seats 1 --map $M/capability-level-up-blue-v1.vmap --episodes 2
done
run after-poison-embedded --backend embedded --controller end_turn --map $M/hero-management-v1.vmap --episodes 1
run recorded-embedded --backend embedded --controller recorded --recorded $R/policy-embedded --map $M/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365
run v8-recorded-embedded --backend embedded $V8 --controller recorded --recorded $R/v8-policy-embedded --map $M/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365
run endturn-embedded-fresh --backend embedded --controller end_turn --map $M/hero-management-v1.vmap --episodes 6 --env-per-row
for s in endturn policy chest-red chest-blue mine-red v8-endturn v8-policy v8-levelup-red v8-levelup-blue; do
  $P compare --left $R/$s-cold --right $R/$s-embedded --output $R/compare-$s.json > $R/compare-$s.out 2>&1; echo "compare $s exit $?"
done
$P compare --left $R/policy-embedded --right $R/recorded-embedded --output $R/compare-recorded.json > $R/compare-recorded.out 2>&1; echo "compare recorded exit $?"
$P compare --left $R/v8-policy-embedded --right $R/v8-recorded-embedded --output $R/compare-v8-recorded.json > $R/compare-v8-recorded.out 2>&1; echo "compare v8-recorded exit $?"
$P compare --left $R/endturn-embedded --right $R/endturn-embedded-fresh --output $R/compare-reuse.json > $R/compare-reuse.out 2>&1; echo "compare reuse exit $?"
$P codec --binary $B/vcmiclient --episode $R/policy-embedded/episode-000001 --output $R/codec.json > $R/codec.out 2>&1; echo "codec exit $?"
echo "DONE $(date -u +%T)"
