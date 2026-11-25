#!/bin/bash
# usage (GPU host): m8prod_post.sh <METHOD> <SEED>  -- after the process exited: stat-only post atime ledger (read-only)
set -eu
M=$1; S=$2
W=/home/student20261/workdir
RT=$W/GPAT_TransferBench_runtime
T=m8prod_${M}_s$S
test -s $W/$T.end_time || { echo "NOT ENDED"; exit 2; }
test "$(tmux list-panes -t $T -F '#{pane_dead}')" = 1 || { echo "PANE NOT DEAD"; exit 2; }
test ! -e $W/${T}_atime_post.tsv || { echo "POST LEDGER EXISTS"; exit 2; }
date -u +%FT%TZ > $W/${T}_atime_post.time
( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $W/${T}_atime_post.tsv
chmod a-w $W/${T}_atime_post.tsv $W/${T}_atime_post.time
echo "post ledger $(cat $W/${T}_atime_post.time) files=$(wc -l < $W/${T}_atime_post.tsv) exit=$(cat $W/$T.exit)"
