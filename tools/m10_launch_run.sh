#!/bin/bash
# Laptop-side launch of ONE M10 scientific run (no chaining). Usage: tools/m10_launch_run.sh <group> <seed> <host>
# 1. host exec clone ~/workdir/m10_launch/science_exec: fetch + ff-only to the pushed laptop HEAD; clean worktree
# 2. preflight-only with the exact authority launch command (+ --preflight-only); must pass
# 3. tmux m10_<G>_s<S> (remain-on-exit, pipe-pane log) running the exact authority launch command; VRAM probe
set -euo pipefail
G="$1"; S="$2"; H="$3"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HEAD="$(git -C "$REPO" rev-parse HEAD)"
[ "$(git -C "$REPO" rev-parse origin/m6-baselines)" = "$HEAD" ] || { echo "laptop HEAD not pushed"; exit 3; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ] || { echo "laptop worktree dirty"; exit 3; }
ssh "$H" "set -euo pipefail; G=$G; S=$S; W=\$HOME/workdir; T=\$W/m10_launch/science_exec
  test ! -e \$W/m10_\${G}_s\${S}.exit; test ! -e \$W/m10_\${G}_s\${S}.command
  test ! -e /media/ext_data/student20261_gpat_m10_detectors/\$G/seed_\$S
  ! tmux has-session -t m10_\${G}_s\${S} 2>/dev/null
  if ps -eo args | grep -q '^[^ ]*/gpat-m8-downstream/bin/python -B tools/[r]un_m10_detector.py'; then echo 'detector already running'; exit 4; fi
  cd \$T; test -z \"\$(git status --porcelain --untracked-files=normal)\"
  git fetch -q origin m6-baselines; git merge -q --ff-only origin/m6-baselines
  test \"\$(git rev-parse HEAD)\" = $HEAD; test -z \"\$(git status --porcelain --untracked-files=normal)\"
  echo exec_head=\$(git rev-parse HEAD)
  env -i HOME=\$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py --group \$G --seed \$S --runtime-root ~/workdir/GPAT_TransferBench_runtime --preflight-only > \$W/m10_\${G}_s\${S}_preflight.json || { echo PREFLIGHT_FAILED; python3 -c \"import json;print(json.load(open('\$W/m10_\${G}_s\${S}_preflight.json'))['failed'])\"; exit 5; }
  python3 -c \"import json;d=json.load(open('\$W/m10_\${G}_s\${S}_preflight.json'));g=d['gates'];print('preflight',d['preflight_ok'],d['failed'],'gpu_free',g['gpu']['observed']['free_mib'],'bank',g['bank_identity'].get('manifest_sha256'))\"
  cat > \$W/m10_\${G}_s\${S}.command <<EOF
#!/bin/bash
date -u +%FT%TZ > ~/workdir/m10_\${G}_s\${S}.launch_time
cd ~/workdir/m10_launch/science_exec && env -i HOME=\\\$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py --group \$G --seed \$S --runtime-root ~/workdir/GPAT_TransferBench_runtime; echo \\\$? > ~/workdir/m10_\${G}_s\${S}.exit
date -u +%FT%TZ > ~/workdir/m10_\${G}_s\${S}.end_time
EOF
  sha256sum \$W/m10_\${G}_s\${S}.command
  N=m10_\${G}_s\${S}
  tmux new-session -d -s \$N -x 200 -y 50
  tmux set-option -t \$N remain-on-exit on
  tmux pipe-pane -t \$N -o \"cat >> \$W/\$N.pane.log\"
  tmux send-keys -t \$N \"bash \$W/\$N.command; exit\" Enter
  for i in 1 2 3 4 5 6 7 8 9 10; do P=\$(pgrep -f \"^\$HOME/workdir/envs/gpat-m8-downstream/bin/python -B tools/run_m10_detector.py --group \$G --seed \$S \" || true); [ -n \"\$P\" ] && break; sleep 2; done
  [ -n \"\$P\" ] || { echo 'process not found'; exit 6; }
  echo pid=\$P
  nohup bash -c \"while kill -0 \$P 2>/dev/null; do echo \\\"\\\$(date -u +%FT%TZ)\\t\\\$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=\$P '\\\$1==p{print \\\$2}')\\t\\\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)\\\"; sleep 15; done\" > \$W/\$N.vram.tsv 2>&1 < /dev/null &
  disown || true
  sleep 3; cat \$W/\$N.launch_time"
