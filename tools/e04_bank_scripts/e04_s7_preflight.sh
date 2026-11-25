#!/bin/bash
# usage: e04_s7_preflight.sh <EXECUTION_COMMIT> <SEED> <A|B>   read-only preflight immediately before a run
SELF=$(dirname "$0"); . $SELF/e04_s7_common.sh "$1"
SEED=$2; L=$3
T=m8s7_E04_s${SEED}_${L}
P=$W/${T}_preflight.json
test ! -e $P || { echo "PREFLIGHT EXISTS $P"; exit 2; }
cd $REPO
$AUDIT_ENV $PYAUD -I -B - preflight --method E04 --repo $REPO --s7-root $S7 --seed $SEED --label $L < tools/learned_bank_s7_audit.py > $P.tmp
mv $P.tmp $P && chmod a-w $P
$AUDIT_ENV $PYAUD -I -B -c "
import json; d = json.load(open('$P'))
pr = d['environment']['probe'] or {}
print('processes', d['processes'], 'panes', d['m8s7_panes'])
print('gpu', d['gpu']['uuid'], d['gpu']['driver'], 'free_mib', d['gpu']['free_mib'], 'apps', [(a['pid'], a['used_mib']) for a in d['gpu']['compute_apps']])
print('disk_gib', d['disk']['free_bytes'] >> 30, 'mem_gib', d['memory']['mem_available_bytes'] >> 30, 'per_seed', d['per_seed'])
print('ckpt', {k: v.get('ok') for k, v in d['checkpoints'].items()}, 'authority', d['authority'].get('ok'), d['authority'].get('error'))
print('inputs', d['inputs'], 'hdd', d['hdd_bank_root_entries'])
print('probe rc', d['environment']['probe_rc'], {k: pr.get(k) for k in ('python', 'tensorflow', 'precision_errors', 'bn_only_difference', 'contract_sha256', 'checkpoint_counters')})
print('envlock', (pr.get('environment_lock') or {}).get('ok'), (pr.get('environment_lock') or {}).get('error'), 'frozen', 'error' not in (pr.get('frozen_sources') or {'error': 1}), 's7auth', 'error' not in (pr.get('s7_authority') or {'error': 1}))
"
