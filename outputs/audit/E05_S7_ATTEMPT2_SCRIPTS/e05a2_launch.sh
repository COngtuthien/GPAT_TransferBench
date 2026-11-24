#!/bin/bash
# E05 S7 attempt 2 (owner ruling B2 P2): one OS process per (seed, run).  usage: e05a2_launch.sh <SEED> <A|B>
set -eu
SEED=$1; L=$2
C=7ff4d36f74cd64d6c405944d3b9fa4aee1fae521
W=/home/student20261/workdir
REPO=$W/m8_learned_banks/clones/s7e05a2_$C
OUT=$W/m8_learned_banks/s7/$C/E05
SROOT=$OUT/seed_$SEED
RT=$W/GPAT_TransferBench_runtime
PY=/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python
ENVS="PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1"
T=m8s7_E05_a2_s${SEED}_${L}
test -s $W/${T}_preflight.json || { echo "NO PREFLIGHT"; exit 2; }
for f in command launch_time exit end_time pane.log; do test ! -e $W/$T.$f || { echo "EXISTS $W/$T.$f"; exit 2; }; done
tmux has-session -t $T 2>/dev/null && { echo "SESSION EXISTS $T"; exit 2; }
if [ "$L" = A ]; then
  test ! -e $SROOT || { echo "SEED ROOT EXISTS"; exit 2; }
  if [ "$SEED" = 42 ]; then
    test ! -e $W/m8s7_E05_a2_atime_pre.tsv || { echo "ATIME PRE EXISTS"; exit 2; }
    date -u +%FT%TZ > $W/m8s7_E05_a2_atime_pre.time
    ( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $W/m8s7_E05_a2_atime_pre.tsv
    chmod a-w $W/m8s7_E05_a2_atime_pre.tsv $W/m8s7_E05_a2_atime_pre.time
    test ! -e $OUT || { echo "E05 OUT ROOT EXISTS"; exit 2; }
    mkdir $OUT
  fi
  mkdir $SROOT
else
  test "$(cat $W/m8s7_E05_a2_s${SEED}_A.exit)" = 0 || { echo "RUN A NOT EXITED 0"; exit 2; }
  test "$(tmux list-panes -t m8s7_E05_a2_s${SEED}_A -F '#{pane_dead}')" = 1 || { echo "RUN A PANE NOT DEAD"; exit 2; }
  test ! -e $SROOT/run_B || { echo "RUN B ROOT EXISTS"; exit 2; }
fi
cat > $W/$T.command <<CMD
#!/bin/bash
# M8 S7 E05 attempt 2 (owner rulings O2 + B1/B2 P2): seed $SEED run $L, one OS process -- execution authority $C
date -u +%FT%TZ > $W/$T.launch_time
cd $REPO || { echo 97 > $W/$T.exit; exit 97; }
env -i HOME=/home/student20261 $ENVS nice -n 10 $PY -B tools/learned_bank.py s7 --method E05 --label $L --out-root $SROOT --runtime-root $RT --seed $SEED
echo \$? > $W/$T.exit
date -u +%FT%TZ > $W/$T.end_time
CMD
chmod 0555 $W/$T.command
tmux new-session -d -s $T -x 200 -y 50 "sleep 2; bash $W/$T.command"
tmux set-option -t $T remain-on-exit on >/dev/null
tmux pipe-pane -t $T -o "cat >> $W/$T.pane.log"
until test -s $W/$T.end_time; do sleep 3; done
echo "$T launch=$(cat $W/$T.launch_time) end=$(cat $W/$T.end_time) exit=$(cat $W/$T.exit)"
tail -c 700 $W/$T.pane.log
