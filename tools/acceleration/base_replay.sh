#!/bin/bash
# The accelerated engine resubmits the delivered actions of the base code's policy runs, both backends, v7 and v8,
# and compares learner histories with the base runs.
# Usage: base_replay.sh OUTPUT_ROOT BASE_REVALIDATION_ROOT     Environment: ROOT, GAME_DATA.
set -u
: ${ROOT:?} ${GAME_DATA:?}
cd "$ROOT"
R=${1:?output root}; BASE=${2:?base revalidation root with policy-embedded and v8-policy-embedded}
mkdir -p $R
B=vcmi/build/bin; D=$GAME_DATA; M=experiments/explorer/maps
P=".venv/bin/python -m validation.probes.validate_embedded_engine"
COMMON="--binary $B/vcmiclient --replay-binary $B/vcmireplay --data $D"
for v in v7 v8; do
  if [ $v = v8 ]; then IFACE="--interface adventure-capabilities-v1"; src=$BASE/v8-policy-embedded; else IFACE=""; src=$BASE/policy-embedded; fi
  for backend in embedded cold; do
    name=$v-recorded-$backend; echo "== $name $(date -u +%T)"
    $P run $COMMON --output $R/$name --backend $backend $IFACE --controller recorded --recorded $src --map $M/hero-management-v1.vmap --episodes 4 --choices 1024 --max-days 365 > $R/$name.out 2> $R/$name.err; echo "exit $?"
    $P compare --left $src --right $R/$name --output $R/compare-$v-$backend.json > $R/compare-$v-$backend.out 2>&1; echo "compare $v-$backend exit $?"
  done
done
echo "DONE $(date -u +%T)"
