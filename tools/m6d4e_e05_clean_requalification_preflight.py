#!/usr/bin/env python3
"""Static M6D4e evidence validation: no Torch import or model execution.

Reuse the committed M6D4d static numerical validators without modifying history.
Read only named evidence and repository Git metadata. Index rebuilding carries
all 566 committed metadata rows without opening their targets and adds six paths.
Workflow declarations describe the observed session; they are not an OS I/O trace.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '32c8f979e14d1ea37ee2047cbd5489cb83c64b72'
BASE = 'outputs/audit/M6D4E_E05_CLEAN_REQUALIFICATION'
PROCESSES = tuple(f'outputs/audit/M6D4E_E05_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
LOG = 'outputs/audit/M6D4E_E05_SYNTHETIC_RUNTIME_LOG.txt'
SELF = 'tools/m6d4e_e05_clean_requalification_preflight.py'
ARTIFACTS = (BASE + '.md', BASE + '.json', *PROCESSES, LOG, SELF)
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
PREFIX_SHA = '6af0ac0b6e9145addd996b4415a3bca118d1d2a09ffe2cba3c36612bbcf05357'
INDEX_SHA = '7852ed797720a3b82e3f91731c1e85a356e4307dfc592f257bf0f17d579f9f66'
IMPLEMENTATION_SHA = {
    'methods/pcgan/training_losses.py': '08f159622d41becb8df2502be297e3d4ffe05e5e4f3ff3bc6a074c1fbaa67202',
    'methods/pcgan/training_runner.py': 'd249722dd7653e981ab7cd8dcf734ab8345bc89074ece5ba01e04489cefd92e5',
    'methods/pcgan/training_diagnostics.py': '68bed70699a7a6457ea480a9fd718ed5badf78c7078be61b3b76659fcb1619f4',
    'methods/pcgan/training_qualification.py': '617ac13b2373620028438ef0b2c370eb39895f46cf1a3b1a456863dfcda8a8b7',
}
OLD_PREFLIGHT = 'tools/m6d4d_e05_training_preflight.py'
OLD_PREFLIGHT_SHA = 'bbc4c5b4ff0033fe92f54204b7f980f303bfd1adeba6c8c7b3a01b2b9f3ddef8'
SCOPE_FLAGS = ('recursive_filesystem_search_performed', 'recursive_home_search',
               'recursive_media_search', 'recursive_runtime_search', 'generic_agents_discovery',
               'benchmark_filename_enumeration', 'benchmark_data_access')
FINAL_STATUS = ['E05_ARCHITECTURE_RUNTIME_QUALIFIED', 'E05_TRAINING_RUNNER_CONTRACT_RESOLVED',
                'E05_TRAINING_RUNNER_QUALIFIED', 'E05_CLEAN_REQUALIFICATION_PASS',
                'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
OPERATION_CLASSES = ['repository Git metadata operations', 'explicit known-path reads',
                     'explicit environment/build identity reads', 'explicit GPU repo commands',
                     'synthetic model execution']


def require(ok, message):
    if not ok:
        raise ValueError('M6D4e: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative known evidence path')
    require(not set(p.parts) & {'data', 'manifests', 'faces_256', 'runs', 'cache'}, 'data firewall')
    require(p.suffix in {'.py', '.json', '.jsonl', '.md', '.yaml', '.csv', '.txt'}, 'evidence type')
    require(not (ROOT / p).is_symlink(), 'no evidence symlinks')
    return (ROOT / p).read_bytes()


def historical(path, expected=None):
    raw = read(path)
    require(raw == git('show', AUTHORITY + ':' + path), 'historical bytes: ' + path)
    if expected is not None:
        require(sha(raw) == expected, 'historical SHA256: ' + path)
    return sha(raw)


def previous_validator():
    historical(OLD_PREFLIGHT, OLD_PREFLIGHT_SHA)
    spec = importlib.util.spec_from_file_location('_m6d4d_static_preflight', ROOT / OLD_PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_records(log, marker):
    return [json.loads(line[len(marker):]) for line in log.splitlines() if line.startswith(marker)]


def compare_historical(pre, runs):
    old = [json.loads(read(path)) for path in pre.PROCESSES]
    comparisons = {}
    for i, current in enumerate(runs, 1):
        for j, reference in enumerate(old, 1):
            result = pre.comparison(current, reference)
            require(result['all_losses_bitwise_equal'], 'measured losses differ from retained M6D4d')
            comparisons[f'M6D4e_{i}_vs_M6D4d_{j}'] = result
    return comparisons


def expected_index():
    raw = git('show', AUTHORITY + ':' + INDEX)
    require(sha(raw) == INDEX_SHA and raw.count(b'\n') == raw.count(b'\r\n'), 'committed CRLF index')
    reader = csv.DictReader(io.StringIO(raw.decode()))
    rows = {r['path']: r for r in reader}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == 566, '566 prior rows')
    for path in ARTIFACTS:
        require(path not in rows, 'six additive artifacts')
        content = read(path)
        rows[path] = dict(path=path, size_bytes=str(len(content)), sha256=sha(content))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    result = stream.getvalue().encode()
    # Removing only the six new rows must recover the exact old index bytes.
    new_paths = {p.encode() for p in ARTIFACTS}
    retained = b''.join(line for line in result.splitlines(keepends=True)
                        if line.split(b',', 1)[0] not in new_paths)
    require(retained == raw, 'all 566 historical index rows byte-identical')
    return result


def verify(check_index=True):
    require('torch' not in sys.modules, 'preflight must not import Torch')
    require(all(git('rev-parse', ref).decode().strip() == AUTHORITY
                for ref in ('HEAD', 'origin/m6-baselines')), 'authority HEAD/origin')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').split()
            == [b'0', b'0'], 'zero divergence')
    require(set(git('diff', '--name-only', AUTHORITY).decode().splitlines()) <= {LEDGER, INDEX},
            'all historical tracked files unchanged')
    pre = previous_validator()
    audit = json.loads(read(BASE + '.json'))
    require(audit['authority'] == AUTHORITY and audit['final_status'] == FINAL_STATUS, 'audit authority/status')
    require(audit['synthetic_numerical_status'] == audit['workflow_scope_status'] == 'PASS', 'both independent gates PASS')
    require(audit['scope_exception'] is False, 'no M6D4e scope exception')
    require(audit['historical_note'] == 'M6D4D_SCOPE_EXCEPTION_RETAINED_AS_HISTORICAL_EVIDENCE', 'historical exception retained')
    old_audit = json.loads(read(pre.BASE + '.json'))
    require(old_audit['scope_status'] == 'EXCEPTION_DISCLOSED' and len(old_audit['scope_exceptions']) == 1,
            'M6D4d exception remains historical truth')
    historical_paths = sorted(set(pre.ARTIFACTS) | set(old_audit['preserved_inputs_sha256'])
                              | {'tests/test_m6a7_e05_blur_contract.py'})
    observed = {path: historical(path) for path in historical_paths}
    require(audit['historical_inputs_sha256'] == observed, 'complete historical identity inventory')
    for path, digest in IMPLEMENTATION_SHA.items():
        require(observed[path] == digest, 'committed M6D4d implementation SHA')
    require(audit['implementation_sha256'] == IMPLEMENTATION_SHA, 'recorded implementation identity')
    require(historical(pre.OVERLAY) == audit['overlay_sha256'] == pre.OVERLAY_SHA, 'M6D4c overlay')
    require(historical('environments/e05.lock.json') == audit['environment_lock_sha256'] == pre.LOCK_SHA,
            'environment lock SHA')
    scope = audit['scope_operation_audit']
    require(all(scope[k] is False for k in SCOPE_FLAGS) and scope['recursive_search_count'] == 0,
            'workflow-scope declarations')
    require(scope['operation_classes'] == OPERATION_CLASSES, 'scope operation classes')
    require(scope['instruction_file_check'] == {'path': str(ROOT / 'AGENTS.md'), 'exists': False,
            'parent_search': False}, 'single permitted instruction check')
    require(all(v is False for v in audit['forbidden_actions'].values()), 'execution firewall declarations')
    runs = [json.loads(read(path)) for path in PROCESSES]
    log = read(LOG).decode()
    for path, digest in IMPLEMENTATION_SHA.items():
        require(digest + '  ' + path in log, 'pre-execution GPU implementation hash')
    transfers = log_records(log, 'M6D4E_GPU_PROCESS_SHA ')
    require(len(transfers) == 2, 'two transferred GPU records')
    for i, (r, path, transfer) in enumerate(zip(runs, PROCESSES, transfers), 1):
        pre.validate_process(r, i)
        require(r['diagnostic_seed'] == 60401, 'diagnostic seed')
        raw = read(path)
        require(transfer == dict(process=i, sha256=sha(raw), size_bytes=len(raw)), 'exact GPU evidence transfer')
        require(audit['process_firewalls'][str(i)]['observed'] == r['firewall'], 'all firewall counters retained')
        require(all(v == 0 for v in audit['process_firewalls'][str(i)]['benchmark_access_counts'].values()),
                'zero benchmark/sample/pair access')
        require(r['environment_before'] == json.loads(read(pre.PROCESSES[0]))['environment_before'],
                'unchanged retained environment and precision')
        for group, count in (('D', 80), ('G', 59)):
            inv = r[group + '_step_gradients'][group]
            require(inv['total'] == inv['non_none'] == inv['finite'] == inv['nonzero'] == count,
                    'complete finite nonzero gradient inventory')
        calls = r['forward_calls']
        d = [x for x in calls if x['phase'] == 'D_step']
        g = [x for x in calls if x['phase'] == 'G_step']
        require([x['component'] for x in d] == [x['component'] for x in g]
                == ['Encoder', 'Encoder', 'Generator', 'Generator'], 'fresh explicit pair paths')
        require([x['input_identity'] for x in d[:2]] == [x['input_identity'] for x in g[:2]], 'same source/target pair')
        require(all(x['D_applications'] == 1 and x['benchmark_iteration'] == 0 for x in g), 'post-D G recomputation')
    comparison = pre.comparison(*runs)
    require(comparison['all_losses_bitwise_equal'] and audit['repeatability'] == comparison, 'new two-process comparison')
    require(audit['comparison_to_M6D4d'] == compare_historical(pre, runs), 'retained M6D4d comparison')
    totals = {k: sum(r['optimizer_applications'][k] for r in runs) for k in ('D', 'G', 'total')}
    require(totals == audit['optimizer_applications_total'] == dict(D=2, G=2, total=4), 'exact total applications')
    require(audit['model_parameter_counts'] == pre.COUNTS and audit['events'] == runs[0]['events'], 'structural summary')
    for key in ('initial_G', 'probe_D', 'D_step_losses', 'G_step_losses', 'build_closure', 'checkpoint_policy'):
        require(audit[key] == runs[0][key], 'measured summary ' + key)
    tests = log_records(log, 'M6D4E_FOCUSED_TEST_RESULT ')
    require(len(tests) == 2 and {t['host'] for t in tests} == {'gpu', 'laptop'}, 'both focused test records')
    for test in tests:
        require(test == audit['tests'][test['host']], 'retained test result')
        require(test['total'] == 111 and test['failed'] == 0 and not test['firewall']['denied']
                and test['real_optimizer_applications'] == 0, 'focused tests and firewall')
        require((test['passed'], test['skipped']) == ((111, 0) if test['host'] == 'gpu' else (95, 16)), 'test totals')
    final = log_records(log, 'M6D4E_GPU_FINAL ')
    require(final == [audit['gpu_final']] == [dict(head=AUTHORITY, origin=AUTHORITY, divergence=[0, 0], clean=True)],
            'final GPU clean authority')
    for path, digest in audit['evidence_sha256'].items():
        require(path in (*PROCESSES, LOG, SELF) and sha(read(path)) == digest, 'audit evidence hash')
    require(set(audit['evidence_sha256']) == {*PROCESSES, LOG, SELF}, 'complete evidence hashes')
    prefix, current = git('show', AUTHORITY + ':' + LEDGER), read(LEDGER)
    require(len(prefix.splitlines()) == 109 and sha(prefix) == PREFIX_SHA, '109-row committed ledger prefix')
    require(current.startswith(prefix) and len(current.splitlines()) == 110, 'single ledger append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == 'M6D4E_E05_CLEAN_REQUALIFICATION', 'ledger classification')
    require(row['synthetic_numerical_status'] == row['workflow_scope_status'] == 'PASS'
            and row['scope_exception'] is False, 'ledger independent gates')
    require(row['optimizer_applications'] == totals and not row['benchmark_training']
            and not row['benchmark_data_access'], 'ledger budget/firewall')
    require(row['committed_prefix_sha256'] == PREFIX_SHA and row['committed_prefix_bytes'] == len(prefix)
            and row['committed_prefix_rows'] == 109 and row['final_rows'] == 110, 'ledger prefix metadata')
    require(row['artifacts_sha256'] == {p: sha(read(p)) for p in ARTIFACTS}, 'six ledger artifact hashes')
    index = expected_index()
    if check_index:
        require(read(INDEX) == index, 'exact 572-row CRLF artifact index')
    require('torch' not in sys.modules, 'no Torch imported')
    return dict(synthetic_numerical_status='PASS', workflow_scope_status='PASS', scope_exception=False,
                optimizer_applications=totals, ledger_rows=110, first_109_rows_byte_identical=True,
                artifact_index_rows=572, previous_566_index_rows_byte_identical=True,
                historical_M6D4d_byte_identical=True, model_execution_in_preflight=False,
                torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    result = verify(check_index=not args.rebuild_index)
    if args.rebuild_index:
        (ROOT / INDEX).write_bytes(expected_index())
        result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
