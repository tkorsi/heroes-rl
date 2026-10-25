#!/usr/bin/env bash
# Explicit CPU bootstrap stages; never starts a game or a training campaign.
# Package resolution is host-specific. This is not a validated Linux lock.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: tools/setup_cloud_cpu.sh [--dry-run] STAGE [STAGE ...]

No arguments (or --help) prints this message without changing anything.
Run only on a dedicated Ubuntu 24.04+ or Debian 12+ Linux host. Python 3.11+
is required; CPU PyTorch is pinned to 2.9.1. The Debian 13 amd64 build is tested
through setup_cloud_locked.sh; other hosts and real gameplay need separate checks.

Explicit stages, normally in this order:
  apt       Install system build/test prerequisites (root or passwordless sudo).
  python    Create/reuse .venv and install CPU PyTorch from its official index.
  prepare   Recreate pinned vcmi/ from canonical sources, then verify agreement.
  build     Configure Release and build client, server, AIs and native replay.
  tests     Run repository unit/contract tests; no game assets are required.
  info      Print OS, tool, package and Python resolution for a future Linux lock.

Examples:
  tools/setup_cloud_cpu.sh --dry-run apt python prepare build tests info
  tools/setup_cloud_cpu.sh apt
  tools/setup_cloud_cpu.sh python prepare build
  tools/setup_cloud_cpu.sh tests info

Environment overrides (relative paths are relative to the repository root):
  HEROES_CLOUD_PYTHON       Python used to create the venv (default: python3)
  HEROES_CLOUD_VENV         Venv directory (default: .venv, ignored by Git)
  HEROES_CLOUD_BUILD_JOBS   Concurrent compiler jobs (default: 2)
  HEROES_CLOUD_SDL2_LIBRARY Explicit SDL2 shared library for a relocated prefix

The build directory is fixed at vcmi/build because native tests use that path.

--dry-run prints shell-quoted commands on any OS, without running them.
This bootstrap neither installs/provisions a cloud service nor starts gameplay,
downloads game assets, uploads files, or runs an experiment/training runner.
The optional tests include synthetic PPO optimizer tests, not gameplay training.
An actual Linux build, tests and bounded native checks remain separate gates.
The existing macOS learner.lock is not reused as a verified Linux CPU lock;
transitive Python and apt dependencies are resolved on the selected Linux host.
Capture the info output before proposing a platform-specific hashed lock.
HELP
}

die() { printf 'setup_cloud_cpu: %s\n' "$*" >&2; exit 1; }

DRY_RUN=0
STAGES=()
for argument in "$@"; do
  case "$argument" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    apt|python|prepare|build|tests|info) STAGES+=("$argument") ;;
    *) die "Unknown argument '$argument'. Use --help." ;;
  esac
done
if [ "${#STAGES[@]}" -eq 0 ]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CLOUD_PYTHON="${HEROES_CLOUD_PYTHON:-python3}"
CLOUD_VENV="${HEROES_CLOUD_VENV:-$ROOT/.venv}"
CLOUD_BUILD="$ROOT/vcmi/build"
CLOUD_JOBS="${HEROES_CLOUD_BUILD_JOBS:-2}"
CLOUD_CMAKE_ARGS=()
if [ -n "${HEROES_CLOUD_SDL2_LIBRARY:-}" ]; then
  CLOUD_CMAKE_ARGS+=("-DSDL2_LIBRARY=$HEROES_CLOUD_SDL2_LIBRARY")
fi
[[ "$CLOUD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "HEROES_CLOUD_BUILD_JOBS must be a positive integer."

if [ "$DRY_RUN" -eq 0 ]; then
  [ "$(uname -s)" = Linux ] || die "Real stages require Linux; use --dry-run to inspect commands on this host."
  [ -r /etc/os-release ] || die "Missing /etc/os-release; expected Ubuntu or Debian."
  # shellcheck source=/dev/null
  . /etc/os-release
  case "${ID:-}" in
    ubuntu) minimum_version=24.04 ;;
    debian) minimum_version=12 ;;
    *) die "Unsupported distribution '${ID:-unknown}'; expected Ubuntu or Debian." ;;
  esac
  dpkg --compare-versions "${VERSION_ID:-0}" ge "$minimum_version" \
    || die "Use $ID $minimum_version or newer with Python 3.11+."
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

check_python() {
  run "$1" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
}

check_torch() {
  run "$CLOUD_VENV/bin/python" -c 'import platform, sys, torch; assert torch.__version__.split("+")[0] == "2.9.1", torch.__version__; assert torch.version.cuda is None and torch.version.hip is None, "CPU PyTorch required"; print("Python:", sys.version); print("Platform:", platform.platform(), platform.machine()); print("PyTorch:", torch.__version__); print(torch.__config__.show())'
}

# Subset of the pinned upstream docs/developers/Building_Linux.md for the
# selected targets. Qt, LuaJIT and ONNX are omitted because their consumers are
# disabled. FFmpeg CLI is included for repository video integration tests.
APT_PACKAGES=(
  ca-certificates git build-essential cmake ninja-build pkg-config ccache
  python3 python3-venv python3-pip
  libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev
  zlib1g-dev libboost-dev libboost-filesystem-dev libboost-system-dev
  libboost-thread-dev libboost-program-options-dev libboost-locale-dev
  libboost-date-time-dev libboost-iostreams-dev libtbb-dev liblzma-dev
  libminizip-dev libsquish-dev libfmt-dev ffmpeg
)

for stage in "${STAGES[@]}"; do
  printf '\nStage: %s\n' "$stage"
  case "$stage" in
    apt)
      PRIVILEGE=()
      if [ "$(id -u)" -ne 0 ]; then
        PRIVILEGE=(sudo -n)
      fi
      run "${PRIVILEGE[@]}" apt-get update
      run "${PRIVILEGE[@]}" env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
      ;;
    python)
      check_python "$CLOUD_PYTHON"
      run "$CLOUD_PYTHON" -m venv "$CLOUD_VENV"
      check_python "$CLOUD_VENV/bin/python"
      run "$CLOUD_VENV/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --requirement dependencies/learner.in
      run "$CLOUD_VENV/bin/python" -m pip check
      check_torch
      run "$CLOUD_VENV/bin/python" -m pip freeze --all
      ;;
    prepare)
      check_python "$CLOUD_PYTHON"
      run "$CLOUD_PYTHON" tools/prepare_vcmi.py
      run "$CLOUD_PYTHON" tools/prepare_vcmi.py --check
      ;;
    build)
      check_python "$CLOUD_PYTHON"
      run "$CLOUD_PYTHON" tools/prepare_vcmi.py --check
      run cmake -S vcmi -B "$CLOUD_BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_GOLDMASTER=ON \
        -DENABLE_STRICT_COMPILATION=OFF -DENABLE_CCACHE=ON \
        -DENABLE_STATIC_LIBS=OFF -DENABLE_MINIMAL_LIB=OFF \
        -DENABLE_CLIENT=ON -DENABLE_SERVER=ON -DENABLE_EXPLORER_RL=ON \
        -DENABLE_BATTLE_AI=ON -DENABLE_NULLKILLER_AI=ON \
        -DENABLE_NULLKILLER2_AI=OFF -DENABLE_STUPID_AI=OFF \
        -DENABLE_LAUNCHER=OFF -DENABLE_EDITOR=OFF -DENABLE_MMAI=OFF \
        -DENABLE_DISCORD=OFF -DENABLE_TRANSLATIONS=OFF -DENABLE_VIDEO=OFF \
        -DENABLE_LUA=OFF -DENABLE_ERM=OFF -DENABLE_LOBBY=OFF -DENABLE_TEST=OFF \
        "${CLOUD_CMAKE_ARGS[@]}"
      run cmake --build "$CLOUD_BUILD" --parallel "$CLOUD_JOBS" \
        --target vcmiclient vcmiserver ExplorerRL BattleAI Nullkiller vcmireplay
      ;;
    tests)
      check_python "$CLOUD_VENV/bin/python"
      check_torch
      run "$CLOUD_VENV/bin/python" -m unittest discover -s tests -v
      ;;
    info)
      printf 'Environment inventory only; not Linux validation or a hashed lock.\n'
      run date -u +%Y-%m-%dT%H:%M:%SZ
      run uname -a
      run cat /etc/os-release
      run git rev-parse HEAD
      run git status --short
      run git -C vcmi rev-parse HEAD
      run cat dependencies/vcmi.json
      run dpkg-query -W '-f=${binary:Package}\t${Version}\n' "${APT_PACKAGES[@]}"
      run cmake --version
      run ninja --version
      run c++ --version
      run "$CLOUD_VENV/bin/python" -m pip --version
      check_torch
      run "$CLOUD_VENV/bin/python" -m pip freeze --all
      ;;
  esac
done
