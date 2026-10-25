#!/usr/bin/env bash
# Debian 13 amd64 CPU setup. No game or optimizer is launched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CLOUD_PREFIX="${HEROES_CLOUD_PREFIX:-/workspace/cloud-setup}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use CPython 3.12 for the tested Linux wheel lock"'
python3 tools/setup_cloud_user.py --prefix "$CLOUD_PREFIX"
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r dependencies/learner-linux-x86_64-py312.lock
.venv/bin/python -m pip check
source "$CLOUD_PREFIX/activate.sh"
python -c 'import torch; assert torch.__version__ == "2.9.1+cpu" and torch.version.cuda is None and torch.version.hip is None'
bash tools/setup_cloud_cpu.sh prepare build
