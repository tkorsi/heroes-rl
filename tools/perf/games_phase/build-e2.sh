#!/bin/sh
cd /Users/orsi/Projects/Heroes-claude
echo "== $(date) build"
cmake --build vcmi/build --parallel 6 --target vcmiclient vcmiserver ExplorerRL Nullkiller BattleAI vcmireplay 2>&1 | tail -n 3
echo "== $(date) build exit $?"
python3 tools/freeze_runtime.py $HOME/Projects/Heroes-artifacts/runtime/claude-g10-e2 2>&1 | tail -n 1
echo "== $(date) frozen"
