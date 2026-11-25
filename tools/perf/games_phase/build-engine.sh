#!/bin/sh
# Build the engine from the checkout as it stands and freeze it as runtime NAME.   build-engine.sh NAME
cd /Users/orsi/Projects/Heroes-claude
echo "== $(date) build for $1"
cmake --build vcmi/build --parallel 4 --target vcmiclient vcmiserver ExplorerRL Nullkiller BattleAI vcmireplay 2>&1 | tail -n 3
STATUS=$?
echo "== $(date) build exit $STATUS"
test -d vcmi/build/bin/config || ln -s ../../config vcmi/build/bin/config
test -d vcmi/build/bin/Mods || ln -s ../../Mods vcmi/build/bin/Mods
python3 tools/freeze_runtime.py $HOME/Projects/Heroes-artifacts/runtime/$1 2>&1 | tail -n 2
echo "== $(date) frozen $1"
