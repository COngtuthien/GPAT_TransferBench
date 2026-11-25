#!/bin/bash
# usage: e04_s7_launch.sh <EXECUTION_COMMIT> <SEED> <A|B>   one OS process for (seed, run) in tmux; returns at once
SELF=$(dirname "$0"); . $SELF/e04_s7_common.sh "$1"
SEED=$2; L=$3
SROOT=$OUT/seed_$SEED
T=m8s7_E04_s${SEED}_${L}
TS=m8s7_E04_s${SEED}
test -s $W/${T}_preflight.json || { echo "NO PREFLIGHT"; exit 2; }
for f in command launch_time exit end_time pane.log gpu_samples.tsv; do test ! -e $W/$T.$f || { echo "EXISTS $W/$T.$f"; exit 2; }; done
tmux has-session -t $T 2>/dev/null && { echo "SESSION EXISTS $T"; exit 2; }
if ps -eo args | grep -q "[p]ython.*tools/learned_bank.py\|[p]ython.*tools/run_e0"; then echo "LEARNED-BANK/RUN PROCESS ACTIVE"; exit 2; fi
if [ "$L" = A ]; then
  test ! -e $SROOT || { echo "SEED ROOT EXISTS"; exit 2; }
  test ! -e $W/${TS}_atime_pre.tsv || { echo "ATIME PRE EXISTS"; exit 2; }
  date -u +%FT%TZ > $W/${TS}_atime_pre.time
  ( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $W/${TS}_atime_pre.tsv
  chmod a-w $W/${TS}_atime_pre.tsv $W/${TS}_atime_pre.time
  mkdir -p $S7
  test -d $OUT || mkdir $OUT
  mkdir $SROOT
else
  test "$(cat $W/${TS}_A.exit)" = 0 || { echo "RUN A NOT EXITED 0"; exit 2; }
  test "$(tmux list-panes -t ${TS}_A -F '#{pane_dead}')" = 1 || { echo "RUN A PANE NOT DEAD"; exit 2; }
  test ! -e $SROOT/run_B || { echo "RUN B ROOT EXISTS"; exit 2; }
fi
LINE=$(tool_line $SEED $L)
case "$LINE" in "cd $REPO && env -i "*"tools/learned_bank.py s7 --method E04 --label $L --out-root $SROOT --runtime-root $RT --seed $SEED") ;; *) echo "UNEXPECTED LINE $LINE"; exit 2;; esac
cat > $W/$T.command <<CMD
#!/bin/bash
# M8 S7 E04 Physics-STD toy acceptance: seed $SEED run $L, one OS process -- execution authority $C
date -u +%FT%TZ > $W/$T.launch_time
( while [ ! -s $W/$T.end_time ]; do nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | sed "s/^/\$(date -u +%FT%TZ), /" >> $W/$T.gpu_samples.tsv; sleep 2; done ) &
$LINE
echo \$? > $W/$T.exit
date -u +%FT%TZ > $W/$T.end_time
wait
CMD
chmod 0555 $W/$T.command
tmux new-session -d -s $T -x 200 -y 50 "sleep 2; bash $W/$T.command"
tmux set-option -t $T remain-on-exit on >/dev/null
tmux pipe-pane -t $T -o "cat >> $W/$T.pane.log"
echo "launched $T"
