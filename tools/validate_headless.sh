#!/usr/bin/env bash
# Real-game validation of the headless simulation, replay and video work (issue #4).
#
# Runs on a machine with locally supplied Heroes III data. Each stage writes
# under $OUT and is skipped when its result already exists, so an interrupted
# validation resumes where it stopped. Nothing here uploads data or commits.
#
#   DATA="$HOME/Library/Application Support/vcmi" tools/validate_headless.sh [stage ...]
#
# Stages: build tests smoke profile series replays videos benchmark evidence (default: all, in order).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="${DATA:?Set DATA to the directory that contains the game Data folder}"
OUT="${OUT:-runs/validation-$(date -u +%Y-%m-%d)}"
WORKERS="${WORKERS:-8}"
ATTEMPTS="${ATTEMPTS:-1000}"
REPLAYS="${REPLAYS:-20}"
VIDEOS="${VIDEOS:-3}"
PYTHON="${PYTHON:-python3}"
BINARY="${BINARY:-$ROOT/vcmi/build/bin/vcmiclient}"
REPLAY_BINARY="${REPLAY_BINARY:-$ROOT/vcmi/build/bin/vcmireplay}"
MAP="${MAP:-$ROOT/experiments/explorer/maps/random-baseline-v1.vmap}"
ENGINE=(--binary "$BINARY" --map "$MAP" --data "$DATA" --skip-check)
VERIFIER=(--binary "$REPLAY_BINARY" --data "$DATA")
if [ $# -eq 0 ]; then
  STAGES=(build tests smoke profile series replays videos benchmark evidence)
else
  STAGES=("$@")
fi
mkdir -p "$OUT"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

for stage in "${STAGES[@]}"; do
  case "$stage" in
    build)
      log "Prepare and build the engine"
      "$PYTHON" tools/prepare_vcmi.py --check || "$PYTHON" tools/prepare_vcmi.py
      "$PYTHON" tools/prepare_vcmi.py --check
      cmake --build vcmi/build --target vcmiclient vcmiserver ExplorerRL BattleAI vcmireplay 2>&1 | tail -5
      ;;
    tests)
      log "Unit and contract tests"
      "$PYTHON" -m unittest discover -s tests 2>&1 | tail -5
      ;;
    smoke)
      [ -f "$OUT/smoke/series.json" ] && { log "smoke: done"; continue; }
      log "Smoke: 4 attempts on 2 workers with replay evidence"
      "$PYTHON" experiments/headless/run_series.py --output "$OUT/smoke" --attempts 4 --workers 2 \
        "${ENGINE[@]}" --profile || true
      "$PYTHON" experiments/headless/replay.py series "$OUT/smoke" "${VERIFIER[@]}" --tamper 1 --trace 1 || true
      ;;
    profile)
      [ -f "$OUT/costs.json" ] && { log "profile: done"; continue; }
      log "Profiled single-worker series for the cost breakdown"
      "$PYTHON" experiments/headless/run_series.py --output "$OUT/profile" --attempts 16 --workers 1 \
        "${ENGINE[@]}" --profile
      "$PYTHON" experiments/headless/cost_breakdown.py "$OUT/profile/worker-0" --output "$OUT/costs.json" > /dev/null
      ;;
    series)
      [ -f "$OUT/attempts/series.json" ] && { log "series: done"; continue; }
      log "Predetermined series: $ATTEMPTS attempts on $WORKERS workers"
      "$PYTHON" experiments/headless/run_series.py --output "$OUT/attempts" --attempts "$ATTEMPTS" \
        --workers "$WORKERS" "${ENGINE[@]}"
      ;;
    replays)
      [ -f "$OUT/attempts/replay-series.json" ] && { log "replays: done"; continue; }
      log "Replay verification of $REPLAYS attempts, natural endings first, with tamper tests"
      "$PYTHON" experiments/headless/replay.py series "$OUT/attempts" "${VERIFIER[@]}" --count "$REPLAYS" \
        --prefer-natural --tamper 2 --trace "$VIDEOS" || true
      ;;
    videos)
      log "Offline videos of the traced, verified episodes"
      mkdir -p "$OUT/videos"
      for trace in "$OUT"/attempts/worker-*/episodes/episode-*/verification/trace.jsonl; do
        [ -f "$trace" ] || continue
        episode="$(dirname "$(dirname "$trace")")"
        name="$(basename "$episode")"
        [ -f "$OUT/videos/$name.mp4" ] && continue
        "$PYTHON" experiments/headless/render_replay.py "$episode" --output "$OUT/videos/$name.mp4" > /dev/null || true
      done
      ls -l "$OUT/videos"
      ;;
    benchmark)
      [ -f "$OUT/benchmark/benchmark.json" ] && { log "benchmark: done"; continue; }
      log "Benchmark: 1, 2, 4 and 8 workers, 3 repetitions of 180 s"
      "$PYTHON" experiments/headless/benchmark.py --output "$OUT/benchmark" "${ENGINE[@]}"
      ;;
    evidence)
      log "Compact evidence"
      args=(--output "$OUT/evidence.json")
      [ -f "$OUT/attempts/series.json" ] && args+=(--series "$OUT/attempts/series.json")
      [ -f "$OUT/attempts/replay-series.json" ] && args+=(--replays "$OUT/attempts/replay-series.json")
      [ -f "$OUT/benchmark/benchmark.json" ] && args+=(--benchmark "$OUT/benchmark/benchmark.json")
      [ -f "$OUT/costs.json" ] && args+=(--costs "$OUT/costs.json")
      videos=("$OUT"/videos/*.json)
      [ -f "${videos[0]}" ] && args+=(--videos "${videos[@]}")
      "$PYTHON" experiments/headless/compact_evidence.py "${args[@]}"
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      exit 2
      ;;
  esac
done
