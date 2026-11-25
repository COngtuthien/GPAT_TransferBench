#!/bin/bash
# E04 S7 reproduction into a NEW immutable root (not run by the S7 acceptance session; needs an owner instruction).
# usage: e04_s7_reproduce.sh <EXECUTION_COMMIT> <SEED> <NEW_ROOT> [--dry-run]
#   Re-executes the recorded run-A command of LEARNED_BANK_S7_E04.json for one seed in a fresh OS process, with the
#   identical interpreter, precision environment, checkpoint and toy population, but --out-root <NEW_ROOT>/seed_<SEED>
#   (created here; an existing NEW_ROOT is refused). Then compares every PNG and the manifest byte for byte with the
#   accepted run A bank (tools/learned_bank.py s7-compare semantics via learned_bank.compare_runs) and writes
#   <NEW_ROOT>/reproduction_seed_<SEED>.json. A stat-only faces atime ledger is taken before and after.
SELF=$(dirname "$0"); . $SELF/e04_s7_common.sh "$1"
SEED=$2; NEW=$3; DRY=${4:-}
case "$SEED" in 42|1337|2026) ;; *) echo "seed must be 42, 1337 or 2026"; exit 2;; esac
case "$NEW" in /*) ;; *) echo "NEW_ROOT must be absolute"; exit 2;; esac
case "$NEW" in $RT*|/media/*|$S7*) echo "NEW_ROOT must lie outside the runtime root, the HDD and the accepted S7 root"; exit 2;; esac
ACCEPTED=$OUT/seed_$SEED/run_A/E04/seed_$SEED
LINE=$(tool_line $SEED A | sed "s#--out-root $OUT/seed_$SEED #--out-root $NEW/seed_$SEED #")
case "$LINE" in *"--out-root $NEW/seed_$SEED --runtime-root $RT --seed $SEED") ;; *) echo "UNEXPECTED LINE $LINE"; exit 2;; esac
if [ "$DRY" = --dry-run ]; then echo "$LINE"; echo "compare against $ACCEPTED"; exit 0; fi
test -d $ACCEPTED || { echo "accepted bank missing: $ACCEPTED"; exit 2; }
test ! -e $NEW || { echo "NEW_ROOT EXISTS $NEW"; exit 2; }
if ps -eo args | grep -q "[p]ython.*tools/learned_bank.py\|[p]ython.*tools/run_e0"; then echo "LEARNED-BANK/RUN PROCESS ACTIVE"; exit 2; fi
mkdir -p $NEW/seed_$SEED
( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $NEW/atime_pre.tsv
set +e
bash -c "$LINE" > $NEW/run.log 2>&1
RC=$?
set -e
( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $NEW/atime_post.tsv
cd $REPO
$AUDIT_ENV $PYAUD -I -B -c "
import json, sys
sys.path.insert(0, '$REPO')
from methods.common import learned_bank as lb
a, b = '$ACCEPTED', '$NEW/seed_$SEED/run_A/E04/seed_$SEED'
out = {'seed': $SEED, 'exit_code': $RC, 'accepted': a, 'reproduction': b, 'atime_ledger_identical': open('$NEW/atime_pre.tsv').read() == open('$NEW/atime_post.tsv').read()}
if $RC == 0:
    out['compare'] = lb.compare_runs(a, b)
    out['reader'] = {k: v for k, v in lb.read_bank(b, expect_method='E04', expect_seed=$SEED, expect_mode='S7_TOY', expect_rows=186).items() if k != 'rows'}
json.dump(out, open('$NEW/reproduction_seed_$SEED.json', 'w'), indent=1, sort_keys=True)
print(json.dumps({k: out.get(k) for k in ('exit_code', 'atime_ledger_identical', 'compare')}))
"
