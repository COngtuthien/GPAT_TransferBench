#!/bin/bash
# usage (GPU host): m8prod_wait.sh <METHOD> <SEED> <MAX_SECONDS>  -- waits for the end marker; never touches the process
M=$1; S=$2; MAX=${3:-3000}
W=/home/student20261/workdir
T=m8prod_${M}_s$S
n=0
until test -s $W/$T.end_time; do sleep 10; n=$((n+10)); test $n -ge $MAX && { echo "STILL RUNNING after $MAX s"; exit 9; }; done
echo "$T launch=$(cat $W/$T.launch_time) end=$(cat $W/$T.end_time) exit=$(cat $W/$T.exit) pane=$(tmux list-panes -t $T -F '#{pane_dead} #{pane_dead_status}')"
tail -c 600 $W/$T.pane.log; echo
