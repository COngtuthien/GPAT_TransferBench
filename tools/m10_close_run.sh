#!/bin/bash
# Laptop-side per-run close for one finished M10 detector run (one group/seed). Read-only on the run root.
#   tools/m10_close_run.sh <group> <seed> [host]
# 1. host: exit marker present, no detector process, run_status COMPLETED
# 2. host: independent audit with the exact authority audit command (exec clone), out to ~/workdir/m10_audits/
# 3. host: tools/m10_run_diagnostics.py (stdin, python3 -I -B) into ~/workdir/m10_evidence/<G>_seed<S>/
# 4. laptop: copy the audit to outputs/audit/M10_RUNS/M10_<G>_SEED<S>_AUDIT.json and the evidence beside it
# Committing / pushing and the exec-clone fast-forward are separate, explicit steps.
set -euo pipefail
G="$1"; S="$2"; H="${3:-student20261@10.31.131.201}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/outputs/audit/M10_RUNS"
REC="M10_${G}_SEED${S}_AUDIT.json"
[ ! -e "$DEST/$REC" ] || { echo "refusing: $DEST/$REC exists"; exit 3; }
[ ! -e "$DEST/${G}_seed${S}" ] || { echo "refusing: $DEST/${G}_seed${S} exists"; exit 3; }
ssh "$H" "set -e; test -s ~/workdir/m10_${G}_s${S}.exit; echo exit_marker=\$(cat ~/workdir/m10_${G}_s${S}.exit)
  if ps -eo args | grep -q '[r]un_m10_detector.py'; then echo 'detector process still running'; exit 4; fi
  python3 -c \"import json;print('run_status', json.load(open('/media/ext_data/student20261_gpat_m10_detectors/${G}/seed_${S}/run_status.json'))['status'])\"
  mkdir -p ~/workdir/m10_audits ~/workdir/m10_evidence
  cd ~/workdir/m10_launch/science_exec; git rev-parse HEAD; test -z \"\$(git status --porcelain --untracked-files=normal)\"
  env -i HOME=\$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/m10_run_auditor.py --repo . --run-root /media/ext_data/student20261_gpat_m10_detectors/${G}/seed_${S} --group ${G} --seed ${S} --runtime-root ~/workdir/GPAT_TransferBench_runtime --exit-marker ~/workdir/m10_${G}_s${S}.exit --out ~/workdir/m10_audits/${REC} || echo AUDIT_RC=\$?
  test -z \"\$(git status --porcelain --untracked-files=normal)\""
ssh "$H" "cd /tmp && python3 -I -B - --group ${G} --seed ${S} --out-dir ~/workdir/m10_evidence/${G}_seed${S}" < "$REPO/tools/m10_run_diagnostics.py"
mkdir -p "$DEST"
scp -q "$H:workdir/m10_audits/${REC}" "$DEST/${REC}"
scp -q -r "$H:workdir/m10_evidence/${G}_seed${S}" "$DEST/${G}_seed${S}"
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['status'], f\"{d['gates_passed']}/{d['gates_total']}\", d['failed_gates'])" "$DEST/${REC}"
