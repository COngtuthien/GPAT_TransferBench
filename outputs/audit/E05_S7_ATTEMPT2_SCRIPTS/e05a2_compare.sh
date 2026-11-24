#!/bin/bash
# usage: e05a2_compare.sh <SEED>   (after run B exited 0)
set -eu
SEED=$1
C=7ff4d36f74cd64d6c405944d3b9fa4aee1fae521
W=/home/student20261/workdir
REPO=$W/m8_learned_banks/clones/s7e05a2_$C
SROOT=$W/m8_learned_banks/s7/$C/E05/seed_$SEED
T=m8s7_E05_a2_s$SEED
test "$(cat $W/${T}_B.exit)" = 0
test ! -e $W/${T}_compare.json
cd $REPO
set +e
env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/student20261/miniconda3/envs/gpat-m6-e05/bin/python -B tools/learned_bank.py s7-compare --method E05 --out-root $SROOT --seed $SEED > $W/${T}_compare.json 2> $W/${T}_compare.err
echo $? > $W/${T}_compare.rc
set -e
chmod a-w $W/${T}_compare.json $W/${T}_compare.rc
echo "compare rc=$(cat $W/${T}_compare.rc)"; head -c 400 $W/${T}_compare.json; echo
