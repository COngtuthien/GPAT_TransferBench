#!/usr/bin/env python3
"""Verify the M6D4b Phase A STOP record. This is not a training runner.

Only explicit evidence paths are opened. Historical index metadata is carried
forward without opening targets. --rebuild-index is the sole write operation.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '370d36bcbd670861aa5ebf120fa1ee0bd13d9d37'
BASE = 'outputs/audit/M6D4B_E05_TRAINING_RUNNER_QUALIFICATION'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LOCK_SHA = '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50'
PREFIX_SHA = '5402e9d9207222be01104675ba95003ed9bbbdb7f85a8336778cd91ed4e1448d'
PROCESSES = tuple(f'outputs/audit/M6D4B_E05_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
ARTIFACTS = (BASE+'.md', BASE+'.json', *PROCESSES,
             'tools/m6d4b_e05_training_preflight.py', 'tests/test_m6d4b_e05_stop.py')
FINAL_STATUS = ['E05_ARCHITECTURE_RUNTIME_QUALIFIED',
                'E05_TRAINING_RUNNER_NOT_YET_QUALIFIED',
                'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']


def require(ok, reason):
    if not ok:
        raise ValueError('M6D4b: '+reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(bool(p.parts) and p.parts[0] in {'configs', 'docs', 'methods', 'outputs',
                'tools', 'tests', 'environments', 'frozen_config_snapshot'}, 'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs', 'cache'} & set(p.parts),
            'data firewall')
    require(p.suffix.lower() in {'.json', '.jsonl', '.md', '.yaml', '.py', '.txt', '.csv', '.docx'},
            'evidence extension')
    require(not (ROOT/p).is_symlink(), 'evidence symlink')
    return (ROOT/p).read_bytes()


def validate_stop(audit):
    require(audit['decision'] == 'STOP_AND_REPORT', 'Phase A STOP decision')
    require(audit['final_status'] == FINAL_STATUS, 'qualification status')
    phase = audit['phase_a']
    require(phase['review_complete'] and not phase['all_required_semantics_resolved'],
            'review completed without inventing resolution')
    table = phase['evidence_table']
    require({r['item'] for r in table} == set('ABCDEFGHIJ') and len(table) == 10,
            'complete A-J evidence coverage')
    for row in table:
        require(all(row[k] for k in ('topic', 'status', 'evidence', 'resolution', 'remaining_gap')),
                'complete evidence row')
    require({'DISCRIMINATOR_OBJECTIVE', 'UPDATE_SCHEDULE'} <= set(phase['blockers']),
            'scientific blockers retained')
    execution = audit['execution']
    require(execution['synthetic_processes'] == execution['optimizer_applications'] == 0,
            'no optimizer execution after Phase A STOP')
    require(not execution['runner_implemented'] and not execution['models_constructed'],
            'no runner implementation after STOP')
    for key in ('initial_five_losses', 'external_total', 'discriminator_losses',
                'gradient_connectivity', 'optimizer_parameter_inventory', 'adam_states',
                'post_step_parameters', 'iteration_transition', 'repeatability'):
        require(execution[key] is None, 'unexecuted evidence must be null: '+key)
    require(all(v is False for v in audit['safety'].values()), 'strict end state')


def validate_process(record, number):
    require(record['process_number'] == number and record['status'] == 'NOT_RUN_PHASE_A_STOP',
            'process is an explicit non-execution record')
    require(record['launched'] is False and record['measurements'] is None,
            'no fabricated process measurements')
    require(record['optimizer_applications'] == record['checkpoints'] == 0,
            'no process optimizer/checkpoint')


def validate_sync(sync):
    require(sync['status'] == 'PASS' and sync['authority'] == AUTHORITY, 'GPU reconciliation')
    require(sync['clean_before_fetch'] and sync['clean_after'], 'GPU clean gates')
    require(sync['after_head'] == sync['after_origin'] == AUTHORITY and
            sync['divergence'] == [0, 0], 'GPU synchronized authority')
    require(len(sync['tracked']) == 2 and len(sync['untracked']) == 14 and
            set(sync['tracked']+sync['untracked']) == set(sync['files']), 'exact GPU files')
    for path, hashes in sync['files'].items():
        require(len(hashes) == 4 and set(hashes.values()) == {sha(git('show', AUTHORITY+':'+path))},
                'GPU archive/worktree/commit/post-sync identity: '+path)
    require(sync['archive'].startswith('/home/student20261/workdir/GPAT_TransferBench_runtime/'
                                      'recovery/M6D4a_pre_M6D4b_sync_'), 'external recovery archive')


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY+':'+INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and
            len(rows) == len(baseline) == 544, 'baseline index identity')
    for path in ARTIFACTS:
        require(path not in rows, 'additive index artifact')
        raw = read(path)
        rows[path] = {'path': path, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


def verify(check_index=True):
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref+' authority')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').strip()
            == b'0\t0', 'laptop divergence')
    audit = json.loads(read(BASE+'.json'))
    validate_stop(audit)
    validate_sync(audit['gpu_reconciliation'])
    for path, digest in audit['preserved_inputs_sha256'].items():
        raw = read(path)
        require(raw == git('show', AUTHORITY+':'+path) and sha(raw) == digest,
                'preserved input: '+path)
    for path, digest in audit['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'artifact identity: '+path)
    for number, path in enumerate(PROCESSES, 1):
        validate_process(json.loads(read(path)), number)
    require(sha(read('environments/e05.lock.json')) == LOCK_SHA == audit['environment_lock_sha256'],
            'environment lock')
    env = audit['environment_read_only_verification']
    require(env['status'] == 'PASS' and env['environment_lock_sha256'] == LOCK_SHA and
            env['environment_name'] == 'gpat-m6-e05' and not env['environment_mutated'] and
            not env['build_invoked'] and not env['cuda_reexecuted'], 'read-only environment scope')
    lock = json.loads(read('environments/e05.lock.json'))
    require(env['packages'] == lock['identity']['packages'] and
            env['extensions'] == {k: v['extension'] for k, v in lock['custom_ops'].items()},
            'environment packages/extensions')
    prefix = git('show', AUTHORITY+':'+LEDGER)
    current = read(LEDGER)
    require(sha(prefix) == PREFIX_SHA and len(prefix.splitlines()) == 106, 'ledger prefix authority')
    require(current.startswith(prefix) and len(current.splitlines()) == 107, 'exactly one ledger append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == 'M6D4B_E05_TRAINING_RUNNER_QUALIFICATION' and
            row['decision'] == 'STOP_AND_REPORT', 'ledger STOP classification')
    require(set(row['artifacts_sha256']) == set(ARTIFACTS), 'ledger artifact coverage')
    for path, digest in row['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'ledger artifact: '+path)
    expected, count = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    return {'status': 'PASS', 'scope': 'STOP_AUDIT_INTEGRITY_ONLY',
            'training_runner_qualified': False, 'decision': 'STOP_AND_REPORT',
            'ledger_rows': 107, 'first_106_rows_byte_identical': True,
            'artifact_index_rows': count, 'synthetic_processes': 0, 'optimizer_applications': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    result = verify(check_index=not args.rebuild_index)
    if args.rebuild_index:
        (ROOT/INDEX).write_bytes(expected_index()[0])
        result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
