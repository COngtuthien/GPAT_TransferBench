#!/bin/bash
# usage: e04_s7_clone.sh <EXECUTION_COMMIT> <BUNDLE>   fresh isolated execution clone (the training checkout is read only)
set -eu
C=$1; B=$2
W=/home/student20261/workdir
REPO=$W/m8_learned_banks/clones/s7e04_$C
test ! -e $REPO || { echo "CLONE EXISTS $REPO"; exit 2; }
git bundle verify $B >/dev/null
git clone -q --no-checkout $B $REPO
git -C $REPO checkout -q -B m6-baselines $C
mkdir -p $REPO/third_party/source_cache
cp -a $W/GPAT_TransferBench/third_party/source_cache/stdn $REPO/third_party/source_cache/stdn
test "$(git -C $REPO rev-parse HEAD)" = "$C"
test -z "$(git -C $REPO status --porcelain)" || { echo "clone not clean"; git -C $REPO status --porcelain | head; exit 2; }
echo "clone $REPO at $(git -C $REPO rev-parse HEAD) clean; training checkout HEAD $(git -C $W/GPAT_TransferBench rev-parse HEAD)"
