#!/bin/bash
# K concurrent probe processes (one embedded engine each) with identical seeds, full rows: collection, replay, audit.
# Prints one JSON line: workers, rows, passed, natural games, decisions, wall seconds and rates.
# Usage: scale.sh ROOT OUT K EPISODES [INTERFACE]     Environment: GAME_DATA.
set -u
ROOT=${1:?root}; OUT=${2:?out}; K=${3:?workers}; N=${4:?episodes}; IFACE=${5:-adventure-dialogs-v1}
: ${GAME_DATA:?}
mkdir -p "$OUT"; cd "$ROOT"
start=$(date +%s.%N)
for k in $(seq 1 "$K"); do
  .venv/bin/python -m validation.probes.validate_embedded_engine run --binary vcmi/build/bin/vcmiclient \
    --replay-binary vcmi/build/bin/vcmireplay --data "$GAME_DATA" --output "$OUT/w$k" --backend embedded \
    --controller policy --random-policy-seed 21 --interface "$IFACE" --map experiments/explorer/maps/hero-management-v1.vmap \
    --episodes "$N" --choices 1024 --max-days 365 --engine-seed 121902 --policy-seed 121903 \
    > "$OUT/w$k.out" 2> "$OUT/w$k.err" &
done
wait
end=$(date +%s.%N)
python3 - "$OUT" "$K" "$start" "$end" <<'PY'
import json, sys
from pathlib import Path
out, k, start, end = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
rows = games = decisions = passed = 0
for w in range(1, k + 1):
    s = json.loads((out / f"w{w}" / "report.json").read_text())["summary"]
    rows += s["rows"]; passed += s["passed"]; games += s["natural_completed"]; decisions += s["submitted_decisions"]
wall = end - start
print(json.dumps({"workers": k, "rows": rows, "passed": passed, "natural_games": games, "decisions": decisions,
                  "wall_seconds": wall, "games_per_second": games / wall, "decisions_per_second": decisions / wall}))
PY
