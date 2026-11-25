#!/bin/sh
# Snapshot the working tree of Heroes-claude (committed + uncommitted) into claude-g10/src-NAME.   snap.sh NAME
G=$HOME/Projects/Heroes-artifacts/claude-g10; D=$G/src-$1
[ -e $D ] && { echo "$D exists"; exit 1; }
mkdir -p $D && cd /Users/orsi/Projects/Heroes-claude && git archive HEAD | tar -x -C $D && git diff HEAD | (cd $D && patch -p1 -s) && echo "snapshot $D"
