#!/bin/bash
# E04 bank adapter CPU synthetic-fixture qualification on the GPU host (NOT S7, NOT production).
# Usage: e04_fixture_qualification.sh <commit>   (the commit's bundle must be at ~/workdir/m8_learned_banks/bundles/)
# Fresh clean clone from the bundle (never the training checkout ~/workdir/GPAT_TransferBench at 18ba00e); the pinned
# STDN predecessor source cache is copied read-only from that checkout (mkdir -p first). CUDA hidden, idle priority.
set -euo pipefail
C="$1"
W=/home/student20261/workdir
B="$W/m8_learned_banks"
CLONE="$B/clones/e04qual_$C"
OUT="$B/e04_bank_prep/fixture_qualification_$C"
test ! -e "$CLONE"
test ! -e "$OUT"
git clone -q -b m6-baselines "$B/bundles/e04qual_$C.bundle" "$CLONE"
test "$(git -C "$CLONE" rev-parse HEAD)" = "$C"
mkdir -p "$CLONE/third_party/source_cache"
cp -a "$W/GPAT_TransferBench/third_party/source_cache/stdn" "$CLONE/third_party/source_cache/stdn"
mkdir -p "$OUT"
cd "$CLONE"
rc=0
env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 \
  TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 \
  TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 \
  TF_XLA_FLAGS=--tf_xla_auto_jit=0 TF_CPP_MIN_LOG_LEVEL=2 \
  nice -n 19 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B tools/e04_bank_fixture_qualification.py \
  --work "$OUT/work" --out "$OUT/evidence.json" --runtime-root "$W/GPAT_TransferBench_runtime" \
  > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" || rc=$?
echo "$rc" > "$OUT/exit"
git -C "$CLONE" status --porcelain --untracked-files=no | wc -l > "$OUT/clone_dirty_tracked_files"
