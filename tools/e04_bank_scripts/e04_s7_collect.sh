#!/bin/bash
# usage: e04_s7_collect.sh <EXECUTION_COMMIT> <OUT_JSON>   read-only independent collection (CPU, gpat-m6-e05)
SELF=$(dirname "$0"); . $SELF/e04_s7_common.sh "$1"
O=$2
test ! -e $O || { echo "EXISTS $O"; exit 2; }
cd $REPO
$AUDIT_ENV $PYAUD -I -B - collect --method E04 --repo $REPO --s7-root $S7 < tools/learned_bank_s7_audit.py > $O.tmp
mv $O.tmp $O && chmod a-w $O
ls -l $O; sha256sum $O
