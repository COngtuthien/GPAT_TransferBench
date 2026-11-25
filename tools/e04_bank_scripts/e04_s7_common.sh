# E04 S7 host helpers: shared settings (sourced). usage in callers: . e04_s7_common.sh <EXECUTION_COMMIT>
set -eu
C=$1
case "$C" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;; *) echo "bad commit $C"; exit 2;; esac
test ${#C} -eq 40 || { echo "full 40-hex execution commit required"; exit 2; }
W=/home/student20261/workdir
REPO=$W/m8_learned_banks/clones/s7e04_$C
S7=$W/m8_learned_banks/s7/$C
OUT=$S7/E04
RT=$W/GPAT_TransferBench_runtime
PYE04=/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python
PYAUD=/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python
AUDIT_ENV="env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1"
test "$(git -C $REPO rev-parse HEAD)" = "$C" || { echo "clone HEAD is not $C"; exit 2; }
test -z "$(git -C $REPO status --porcelain)" || { echo "clone not clean"; exit 2; }
# the exact launch / compare lines come from the auditor (single source of truth with its gates)
tool_line() { $AUDIT_ENV $PYAUD -I -B -c "
import importlib.util, sys
s = importlib.util.spec_from_file_location('a', '$REPO/tools/learned_bank_s7_audit.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
sys.stdout.write(m.e04_command_line('$REPO', '$S7', $1, '$2') if '$2' in 'AB' and '$2' else m.e04_compare_line('$REPO', '$S7', $1))"; }
