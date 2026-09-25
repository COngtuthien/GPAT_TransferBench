#!/usr/bin/env python3
"""Verify retained M6D5b E06c STOP_OOM evidence; rebuild the index from committed metadata.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest payload read and no image I/O. Validates the exact
retained STOP state: one clean-resource B=240 process that hit CUDA OOM, zero
optimizer applications, zero backward passes and no fabricated process 2.
Historical index rows are carried from the authoritative commit unopened.
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
AUTHORITY = '6e3dd3384add2ef723c4fc00e30512e58cd761c5'
PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
BASE = 'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION'
PROCESS_1 = 'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_1.json'
PROCESS_2 = 'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_2.json'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 111
LEDGER_PREFIX_SHA = '27b2c27563394b2540f82eff06f3f8ed71079e481c18e24f450958254e01e299'
INDEX_BASELINE_ROWS = 584
CLASSIFICATION = 'M6D5B_E06C_TRAINING_GRAPH_STOPPED_OOM'
ARTIFACTS = ('methods/dsdg/training_graph.py', 'methods/dsdg/training_qualification.py',
             'tests/test_m6d5b_e06c_training_graph.py', 'tools/m6d5b_e06c_training_preflight.py',
             PROCESS_1, 'outputs/audit/M6D5B_E06C_RUNTIME_LOG.txt', BASE + '.md', BASE + '.json')
PRESERVED = ('configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml',
             'methods/dsdg/adapter.py', 'methods/dsdg/source.py', 'methods/dsdg/runtime.py', 'methods/dsdg/__init__.py',
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py',
             'environments/e06c.lock.json', 'environments/e06c.runtime.json',
             'environments/e06c.conda-explicit.txt', 'environments/e06c.pip-freeze.txt',
             'outputs/audit/M6C2A_IMPLEMENTATION.md', 'outputs/audit/M6C2A_IMPLEMENTATION.json',
             'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json', 'third_party/source_pins.json',
             'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.md',
             'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.json',
             'outputs/audit/M6D5A_E06C_RUNTIME_LOG.txt', 'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_1.json',
             'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_2.json',
             'tools/m6d5a_e06c_runtime_preflight.py', 'tests/test_m6d5a_e06c_runtime.py')
EXPECTED_PARAMETERS = {'netE_nir': (13_790_048, 17), 'netE_vis': (11_692_640, 17),
                       'netG': (19_887_494, 21), 'netCls': (129, 2), 'netIP': (10_475_872, 60)}
FINAL_STATUS = ['E06c_RUNTIME_ENVIRONMENT_QUALIFIED', 'E06c_ARCHITECTURE_RUNTIME_QUALIFIED',
                'E06c_TRAINING_GRAPH_PHYSICAL_BATCH_240_STOPPED_OOM', 'E06c_PHYSICAL_BATCH_240_NOT_QUALIFIED',
                'E06c_FULL_TRAINING_NOT_EXECUTED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
SCOPE = {'physical_batch_size': 240, 'gradient_accumulation_steps': 1, 'replica_factor': 1,
         'optimizer_constructed': True, 'optimizer_applications': 0, 'backward_passes': 0,
         'benchmark_training': False, 'benchmark_data_access': False, 'TEST_access': False,
         'checkpoint_created': False, 'synthetic_bank': False}
COUNTERS = {'optimizer_constructions': 1, 'optimizer_step_entries': 0, 'optimizer_applications': 0,
            'backward_passes': 0, 'autograd_backward_calls': 0, 'autograd_grad_calls': 0,
            'checkpoint_saves': 0, 'activation_checkpoint_calls': 0}


def require(ok, message):
    if not ok:
        raise ValueError('M6D5b: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'third_party'},
            'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs'} & set(p.parts), 'data firewall')
    return (ROOT / p).read_bytes()


def source_cache(lock):
    root = ROOT / 'third_party/source_cache/facexzoo'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    status = git('status', '--porcelain', root=root).decode().strip()
    require(status == 'D addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl', 'source-cache status')
    for rel, h in lock['source']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    return status


def check_process(r, lock):
    """The exact retained STOP state of the single B=240 process."""
    require(r['status'] == 'STOP_OOM' and r['label'] == 'SYNTHETIC_PHYSICAL_BATCH_240', 'STOP_OOM label')
    require(r['diagnostic_seed'] == 60502 and r['diagnostic_seed'] not in (42, 1337, 2026), 'qualification seed')
    require(r['resource_clean'] and r['gpu_before']['compute_processes'] == [] and
            r['gpu_before']['used_mib'] <= 1024, 'clean GPU resource state')
    for k in ('source_before', 'source_after'):
        s = r[k]
        require((s['commit'], s['tree']) == (PIN, TREE) and s['files_sha256'] == lock['source']['files_sha256'],
                'source identity ' + k)
    require(r['asset_before'] == r['asset_after'] and r['asset_before']['sha256'] == LIGHTCNN_SHA and
            r['asset_before']['bytes'] == 123844849, 'LightCNN identity')
    require(r['environment_lock_sha256'] == LOCK_SHA and r['environment_before'] == r['environment_after'], 'env')
    require(all(r['environment_before'][k] == v for k, v in lock['identity'].items() if k != 'launch_environment'),
            'runtime identity equals lock')
    require(r['pinned_statement_mismatches'] == [] and r['pinned_statements_verified'] == 46, 'pinned statements')
    l = r['lightcnn']
    require((l['checkpoint_tensor_count'], l['expected_model_tensor_count'], l['matched']) == (61, 60, 60) and
            not l['missing'] and not l['shape_mismatches'], 'LightCNN 60/60')
    for n, (count, tensors) in EXPECTED_PARAMETERS.items():
        p = r['parameters_initial'][n]
        require((p['parameters'], p['parameter_tensors']) == (count, tensors), 'parameter count ' + n)
    o = r['optimizer']
    require((o['parameters'], o['parameter_tensors'], o['duplicates'], o['overlap_netCls_netIP']) ==
            (45_370_182, 55, 0, 0) and o['resolved_group_defaults']['lr'] == 2e-4 and
            o['resolved_group_defaults']['weight_decay'] == 0, 'optimizer ownership')
    i = r['inputs']
    require(i['x_spoof']['shape'] == i['x_live']['shape'] == [240, 3, 256, 256] and i['distinct_rows'] == 480,
            'B=240 distinct synthetic inputs')
    oom = r['oom']
    require(oom['error_type'] == 'OutOfMemoryError' and 'CUDA out of memory' in oom['error'], 'OOM error')
    require(oom['phase'] == 'forward_classifier_generator', 'OOM phase')
    require(not oom['optimizer_step_entered'] and not oom['optimizer_application_occurred'] and
            not oom['any_parameter_changed'] and not any(oom['parameters_changed'].values()), 'no update')
    require(oom['allocator']['num_ooms'] == 1, 'one OOM event')
    require([m['phase'] for m in r['memory_by_phase']] == ['after_model_construction', 'after_input_allocation'],
            'memory phases reached')
    require(r['counters'] == COUNTERS, 'counters')
    require(all(r[k] == v for k, v in SCOPE.items()), 'process scope flags')
    require(not r['physical_batch_240_feasible'] and not r['training_graph_executed'], 'not qualified')
    require(r['firewall']['denied'] == [] and r['compatibility_patch'] == 'NONE' and
            r['fidelity'] == 'CONTROLLED_ADAPTATION', 'firewall/patch/fidelity')


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and
            len(rows) == len(baseline) == INDEX_BASELINE_ROWS, 'baseline index')
    for p in ARTIFACTS:
        require(p not in rows, 'additive artifact ' + p)
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


def verify(check_index=True):
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    lock_raw = read('environments/e06c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock identity')
    lock = json.loads(lock_raw)
    status = source_cache(lock)
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS and audit['decision'] == 'STOP_AND_REPORT' and
            audit['classification'] == CLASSIFICATION, 'audit status')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    process = json.loads(read(PROCESS_1))
    check_process(process, lock)
    require(not (ROOT / PROCESS_2).exists(), 'process 2 must not be fabricated after a clean OOM')
    require(all(audit[k] == v for k, v in SCOPE.items()), 'audit scope flags')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix) and len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == CLASSIFICATION and row['final_status'] == FINAL_STATUS and
            row['decision'] == 'STOP_AND_REPORT', 'ledger row')
    require(all(row[k] == v for k, v in SCOPE.items()), 'ledger scope flags')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(set(row['artifacts_sha256']) == set(ARTIFACTS), 'ledger artifact set')
    expected, count = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'validated_state': 'M6D5B_STOP_OOM', 'ledger_rows': LEDGER_PREFIX_ROWS + 1,
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'gpu_processes': 1, 'process_2_present': False, 'source_cache_status': status,
            'optimizer_applications': 0, 'backward_passes': 0, 'torch_imported': False,
            'model_execution_in_preflight': False, 'benchmark_data_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    result = verify(check_index=not args.rebuild_index)
    if args.rebuild_index:
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
