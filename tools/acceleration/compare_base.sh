#!/bin/bash
# Compare the final code's revalidation runs with the base code's runs of the same scenario, seeds and backend.
# Usage: compare_base.sh BASE_RUNS FINAL_RUNS OUT     Environment: ROOT (a worktree with the probe).
set -u
BASE=${1:?base revalidation root}; FINAL=${2:?final revalidation root}; OUT=${3:?out}
: ${ROOT:?}
mkdir -p "$OUT"; cd "$ROOT"
P=".venv/bin/python -m validation.probes.validate_embedded_engine"
names=""
for s in endturn policy chest-red chest-blue mine-red poison v8-endturn v8-policy v8-levelup-red v8-levelup-blue; do
  names="$names $s-embedded $s-cold"
done
names="$names after-poison-embedded endturn-embedded-fresh recorded-embedded v8-recorded-embedded"
for n in $names; do
  $P compare --left "$BASE/$n" --right "$FINAL/$n" --output "$OUT/compare-base-$n.json" > "$OUT/compare-base-$n.out" 2>&1
  echo "$n exit $?"
done
