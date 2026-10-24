#!/usr/bin/env bash
# Real-game validation of the headless simulation, replay and video work (issue #4).
#
# Runs on a machine with locally supplied Heroes III data. Each stage writes
# under $OUT and is skipped once its completion marker exists. An unfinished
# stage output is moved aside to <name>.interrupted-<UTC time> and the stage
# reruns from scratch, so an interrupted validation resumes where it stopped
# and keeps every earlier file. Without OUT, the first run picks
# runs/validation-<UTC date> and pins it in runs/validation-current, so a
# resume on a later date continues the same validation. A failed smoke stage
# stops before the long stages. Nothing here uploads data or commits.
#
# The build stage prepares vcmi/ from this recipe and configures vcmi/build
# when it has no CMake cache. A vcmi/ prepared from an older recipe is never
# changed: the stage stops and prints how to keep it aside and prepare again.
# Build, test and configure output is kept in $OUT/*.log.
#
#   DATA="$HOME/Library/Application Support/vcmi" tools/validate_headless.sh [stage ...]
#
# Stages: build tests smoke profile series replays videos benchmark evidence (default: all, in order).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="${DATA:?Set DATA to the directory that contains the game Data folder}"
PIN=runs/validation-current
if [ -z "${OUT:-}" ]; then
  OUT="$(cat "$PIN" 2>/dev/null || true)"
  if [ -z "$OUT" ]; then
    OUT="runs/validation-$(date -u +%Y-%m-%d)"
    mkdir -p "$(dirname "$PIN")"
    echo "$OUT" > "$PIN"
  fi
  RESUME="rerun the same command; $PIN pins this output until you remove it"
else
  RESUME="rerun the same command with OUT=\"$OUT\""
fi
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

# Moves a stage output that lacks its completion marker aside; nothing is deleted.
set_aside() {
  [ -e "$1" ] || return 0
  local base moved n=1
  base="$1.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
  moved="$base"
  while [ -e "$moved" ]; do  # never move into an output set aside in the same second
    n=$((n + 1))
    moved="$base-$n"
  done
  mv "$1" "$moved"
  log "Unfinished $1 moved to $moved"
}

# All 4 smoke attempts played without a failure, every one verified and every tamper case rejected.
smoke_passed() {
  "$PYTHON" - "$OUT/smoke" <<'EOF'
import json, sys
from pathlib import Path

def load(name):
    try:
        return json.loads((Path(sys.argv[1]) / name).read_text())
    except (OSError, ValueError):
        return None

series, replays, problems = load("series.json"), load("replay-series.json"), []
if not isinstance(series, dict):
    problems.append("run_series.py wrote no series.json")
elif series.get("attempts_played") != 4 or series.get("attempts_missing"):
    problems.append(f"{series.get('attempts_played')} of 4 attempts played, missing {series.get('attempts_missing')}")
elif series.get("failures") != 0:
    problems.append(f"{series.get('failures')} of 4 attempts failed: {series.get('failure_details')}")
if not isinstance(replays, dict):
    problems.append("replay.py series wrote no replay-series.json")
else:
    rows = replays.get("episodes") or []
    verified = sum(1 for row in rows if row.get("verified") is True)
    if not rows or verified != len(rows):
        problems.append(f"{verified} of {len(rows)} selected attempts verified")
    if (replays.get("aggregate") or {}).get("tamper_all_detected") is not True:
        problems.append("no tamper test ran, or a tampered recording was accepted")
for problem in problems:
    print(f"smoke: {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
EOF
}

# benchmark.py marks its report complete only after the last configuration.
benchmark_complete() {
  "$PYTHON" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("complete") is not True)' \
    "$OUT/benchmark/benchmark.json" 2> /dev/null
}

log "Output: $OUT (to resume after an interruption, $RESUME)"
for stage in "${STAGES[@]}"; do
  case "$stage" in
    build)
      log "Prepare and build the engine"
      if ! "$PYTHON" tools/prepare_vcmi.py --check; then
        if [ -e vcmi ]; then
          # prepare_vcmi.py never resets a checkout; an older recipe's vcmi/ is kept aside instead.
          kept="vcmi.before-$(date -u +%Y%m%dT%H%M%SZ)"
          echo "vcmi/ does not match this recipe (an older recipe or local edits), and prepare_vcmi.py will not change it." >&2
          echo "Keep it aside, prepare again and reuse its build directory, then rerun this command:" >&2
          echo "  mv vcmi $kept && $PYTHON tools/prepare_vcmi.py && mv $kept/build vcmi/build" >&2
          exit 1
        fi
        "$PYTHON" tools/prepare_vcmi.py
      fi
      "$PYTHON" tools/prepare_vcmi.py --check
      if [ ! -f vcmi/build/CMakeCache.txt ]; then
        log "Configure vcmi/build"
        cmake -S vcmi -B vcmi/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/homebrew}" -DENABLE_GOLDMASTER=ON \
          -DENABLE_STRICT_COMPILATION=OFF -DENABLE_EXPLORER_RL=ON > "$OUT/configure.log" 2>&1 \
          || { tail -30 "$OUT/configure.log" >&2; echo "Configure failed; full log: $OUT/configure.log" >&2; exit 1; }
      fi
      cmake --build vcmi/build --target vcmiclient vcmiserver ExplorerRL BattleAI vcmireplay > "$OUT/build.log" 2>&1 \
        || { grep -m 20 -E 'error:|FAILED:' "$OUT/build.log" >&2 || true; tail -30 "$OUT/build.log" >&2
             echo "Build failed; full log: $OUT/build.log" >&2; exit 1; }
      tail -3 "$OUT/build.log"
      ;;
    tests)
      log "Unit and contract tests"
      "$PYTHON" -m unittest discover -s tests > "$OUT/tests.log" 2>&1 \
        || { grep -E '^(ERROR|FAIL):' "$OUT/tests.log" >&2 || true; tail -30 "$OUT/tests.log" >&2
             echo "Tests failed; full log: $OUT/tests.log" >&2; exit 1; }
      tail -3 "$OUT/tests.log"
      ;;
    smoke)
      [ -f "$OUT/smoke/ok" ] && { log "smoke: done"; continue; }
      set_aside "$OUT/smoke"
      log "Smoke: 4 attempts on 2 workers with replay evidence"
      # Both exit codes are recorded in the files that smoke_passed checks.
      "$PYTHON" experiments/headless/run_series.py --output "$OUT/smoke" --attempts 4 --workers 2 \
        "${ENGINE[@]}" --profile || true
      "$PYTHON" experiments/headless/replay.py series "$OUT/smoke" "${VERIFIER[@]}" --tamper 1 --trace 1 || true
      smoke_passed || { echo "Smoke failed; fix it before the long stages (evidence in $OUT/smoke)" >&2; exit 1; }
      date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/smoke/ok"
      ;;
    profile)
      [ -f "$OUT/costs.json" ] && { log "profile: done"; continue; }
      if [ ! -f "$OUT/profile/series.json" ]; then
        set_aside "$OUT/profile"
        log "Profiled single-worker series for the cost breakdown"
        # A missing attempt is recorded in series.json; the breakdown covers the attempts played.
        "$PYTHON" experiments/headless/run_series.py --output "$OUT/profile" --attempts 16 --workers 1 \
          "${ENGINE[@]}" --profile || log "profile: run_series.py exited $?; see $OUT/profile/series.json"
        [ -f "$OUT/profile/series.json" ] || { echo "profile: run_series.py wrote no series.json" >&2; exit 1; }
      fi
      "$PYTHON" experiments/headless/cost_breakdown.py "$OUT/profile/worker-0" --output "$OUT/costs.json" > /dev/null \
        || { echo "profile: cost_breakdown.py failed" >&2; set_aside "$OUT/profile"; exit 1; }
      ;;
    series)
      [ -f "$OUT/attempts/series.json" ] && { log "series: done"; continue; }
      set_aside "$OUT/attempts"
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
        "$PYTHON" experiments/headless/render_replay.py "$episode" --output "$OUT/videos/$name.mp4" \
          > "$OUT/videos/$name.log" 2>&1 || log "videos: $name not rendered: $(tail -1 "$OUT/videos/$name.log")"
      done
      ls -l "$OUT/videos"
      # The selection needs enough videos, one of them a natural ending with a battle. A
      # shortfall is recorded in the evidence instead of stopping the benchmark.
      "$PYTHON" - "$OUT/videos" "$VIDEOS" <<'PY' || true
import json, sys
from pathlib import Path
videos = [json.loads(path.read_text()) for path in sorted(Path(sys.argv[1]).glob("*.json"))]
problems = []
if len(videos) < int(sys.argv[2]):
    problems.append(f"{len(videos)} of {sys.argv[2]} videos rendered")
if not any(video.get("status") == "completed" and video.get("battle_frames_drawn") for video in videos):
    problems.append("no rendered video shows a natural ending with a battle")
shortfall = Path(sys.argv[1]) / "shortfall.txt"
if problems:
    shortfall.write_text("; ".join(problems) + "\n")
    print("videos: " + "; ".join(problems), file=sys.stderr)
elif shortfall.exists():
    shortfall.unlink()
PY
      ;;
    benchmark)
      benchmark_complete && { log "benchmark: done"; continue; }
      set_aside "$OUT/benchmark"
      log "Benchmark: 1, 2, 4 and 8 workers, 3 repetitions of 180 s"
      "$PYTHON" experiments/headless/benchmark.py --output "$OUT/benchmark" "${ENGINE[@]}"
      ;;
    evidence)
      log "Compact evidence"
      args=(--output "$OUT/evidence.json")
      [ -f "$OUT/attempts/series.json" ] && args+=(--series "$OUT/attempts/series.json")
      [ -f "$OUT/attempts/replay-series.json" ] && args+=(--replays "$OUT/attempts/replay-series.json")
      # compact_evidence.py records an incomplete benchmark as such.
      [ -f "$OUT/benchmark/benchmark.json" ] && args+=(--benchmark "$OUT/benchmark/benchmark.json")
      [ -f "$OUT/costs.json" ] && args+=(--costs "$OUT/costs.json")
      [ -f "$OUT/build.log" ] && args+=(--build-log "$OUT/build.log")
      [ -f "$OUT/tests.log" ] && args+=(--tests-log "$OUT/tests.log")
      videos=("$OUT"/videos/*.json)
      [ -f "${videos[0]}" ] && args+=(--videos "${videos[@]}")
      [ -f "$OUT/videos/shortfall.txt" ] && args+=(--note "Video selection: $(cat "$OUT/videos/shortfall.txt")")
      "$PYTHON" experiments/headless/compact_evidence.py "${args[@]}"
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      exit 2
      ;;
  esac
done
