#!/bin/bash
# usage: e04_s7_post.sh <EXECUTION_COMMIT> <SEED>   after run B ended: per-seed atime ledger post + CLI s7-compare
SELF=$(dirname "$0"); . $SELF/e04_s7_common.sh "$1"
SEED=$2
TS=m8s7_E04_s${SEED}
test -s $W/${TS}_B.end_time || { echo "RUN B NOT ENDED"; exit 2; }
test ! -e $W/${TS}_atime_post.tsv || { echo "ATIME POST EXISTS"; exit 2; }
date -u +%FT%TZ > $W/${TS}_atime_post.time
( cd $RT/data/processed && find faces_256 -type f -printf "%A@\t%T@\t%p\n" ) > $W/${TS}_atime_post.tsv
chmod a-w $W/${TS}_atime_post.tsv $W/${TS}_atime_post.time
if [ "$(cat $W/${TS}_A.exit)" = 0 ] && [ "$(cat $W/${TS}_B.exit)" = 0 ]; then
  test ! -e $W/${TS}_compare.json
  LINE=$(tool_line $SEED compare)
  printf '#!/bin/bash\n%s\n' "$LINE" > $W/${TS}_compare.command
  chmod 0555 $W/${TS}_compare.command
  set +e
  bash $W/${TS}_compare.command > $W/${TS}_compare.json 2> $W/${TS}_compare.err
  echo $? > $W/${TS}_compare.rc
  set -e
  chmod a-w $W/${TS}_compare.json $W/${TS}_compare.rc
  echo "compare rc=$(cat $W/${TS}_compare.rc)"; tail -c 600 $W/${TS}_compare.json; echo
else
  echo "a run did not exit 0: no comparison (evidence preserved)"
fi
wc -l < $W/${TS}_atime_post.tsv
