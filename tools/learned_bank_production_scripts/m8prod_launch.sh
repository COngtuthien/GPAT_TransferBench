#!/bin/bash
# M8 learned-bank PRODUCTION: one bank = one OS process.  usage (GPU host): m8prod_launch.sh <METHOD> <SEED> <EXEC_COMMIT>
# Refuses unless the host preflight file exists, every marker/ledger/session is fresh, the registered bank directory is
# absent and the execution clone is clean at EXEC_COMMIT. Takes the stat-only pre atime ledger, writes the read-only
# command file and starts it in tmux (remain-on-exit, pipe-pane log). Returns after the process has started.
set -eu
M=$1; S=$2; C=$3
W=/home/student20261/workdir
REPO=$W/m8_learned_banks/clones/prod_${M}_$C
RT=$W/GPAT_TransferBench_runtime
BANK=/media/ext_data/student20261_gpat_m8_banks/$M/seed_$S
T=m8prod_${M}_s$S
case $M in
  E03) PY=/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python
       ENVS="PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0" ;;
  E05) PY=/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python
       ENVS="PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1" ;;
  E06b|E06c) PY=/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python
       ENVS="PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1" ;;
  *) echo "UNKNOWN METHOD"; exit 2 ;;
esac
case $S in 42|1337|2026) ;; *) echo "UNKNOWN SEED"; exit 2 ;; esac
test -s $W/${T}_preflight.json || { echo "NO PREFLIGHT"; exit 2; }
for f in command launch_time exit end_time pane.log; do test ! -e $W/$T.$f || { echo "EXISTS $W/$T.$f"; exit 2; }; done
for f in atime_pre.tsv atime_pre.time atime_post.tsv atime_post.time; do test ! -e $W/${T}_$f || { echo "EXISTS ${T}_$f"; exit 2; }; done
tmux has-session -t $T 2>/dev/null && { echo "SESSION EXISTS $T"; exit 2; }
test ! -e $BANK || { echo "BANK DIR EXISTS $BANK"; exit 2; }
test "$(git -C $REPO rev-parse HEAD)" = "$C" || { echo "CLONE HEAD"; exit 2; }
test -z "$(git -C $REPO status --porcelain --untracked-files=all)" || { echo "CLONE DIRTY"; exit 2; }
if pgrep -f 'tools/learned_bank.py|tools/run_e04.py' >/dev/null; then echo "OTHER LEARNED/E04 PROCESS"; exit 2; fi
date -u +%FT%TZ > $W/${T}_atime_pre.time
( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $W/${T}_atime_pre.tsv
chmod a-w $W/${T}_atime_pre.tsv $W/${T}_atime_pre.time
cat > $W/$T.command <<CMD
#!/bin/bash
# M8 learned-bank production: method $M seed $S, one OS process -- authority configs/amendments/learned_bank_production_$M.yaml at $C
date -u +%FT%TZ > $W/$T.launch_time
cd $REPO || { echo 97 > $W/$T.exit; exit 97; }
env -i HOME=/home/student20261 $ENVS nice -n 10 $PY -B tools/learned_bank.py production --method $M --seed $S --runtime-root $RT
echo \$? > $W/$T.exit
date -u +%FT%TZ > $W/$T.end_time
CMD
chmod 0555 $W/$T.command
tmux new-session -d -s $T -x 200 -y 50 "sleep 2; bash $W/$T.command"
tmux set-option -t $T remain-on-exit on >/dev/null
tmux pipe-pane -t $T -o "cat >> $W/$T.pane.log"
sleep 5
echo "$T started launch=$(cat $W/$T.launch_time 2>/dev/null) command_sha256=$(sha256sum $W/$T.command | cut -c1-16)"
