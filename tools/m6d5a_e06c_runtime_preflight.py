#!/usr/bin/env python3
"""Verify retained M6D5a E06c evidence; rebuild the index from committed metadata.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest payload read and no image I/O. Only the explicit new
artifact allowlist is hashed; historical index rows are carried from the
authoritative commit without opening their targets.
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
AUTHORITY = '13fc86c15ecb9303e4ec678ab3f0f55ff61a51d8'
PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
BASE = 'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 110
LEDGER_PREFIX_SHA = 'ee11822a9d0d45c0952ffc4af180772b897ce0c47825ffbb2ab6df42633dd434'
INDEX_BASELINE_ROWS = 572
CLASSIFICATION = 'M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION'
PROCESSES = tuple(f'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
ENVIRONMENTS = tuple('environments/e06c.' + s for s in
                     ('runtime.json', 'lock.json', 'conda-explicit.txt', 'pip-freeze.txt'))
ARTIFACTS = ('methods/dsdg/runtime.py', 'tests/test_m6d5a_e06c_runtime.py',
             'tools/m6d5a_e06c_runtime_preflight.py', *ENVIRONMENTS, *PROCESSES,
             'outputs/audit/M6D5A_E06C_RUNTIME_LOG.txt', BASE + '.md', BASE + '.json')
PRESERVED = ('configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml',
             'outputs/audit/M6C2A_IMPLEMENTATION.md', 'outputs/audit/M6C2A_IMPLEMENTATION.json',
             'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json', 'third_party/source_pins.json',
             'methods/dsdg/adapter.py', 'methods/dsdg/source.py', 'methods/dsdg/__init__.py',
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py')
EXPECTED_PARAMETERS = {'netE_nir': (13_790_048, 17), 'netE_vis': (11_692_640, 17),
                       'netG': (19_887_494, 21), 'netCls': (129, 2), 'netIP': (10_475_872, 60)}
FINAL_STATUS = ['E06c_RUNTIME_ENVIRONMENT_QUALIFIED', 'E06c_ARCHITECTURE_RUNTIME_QUALIFIED',
                'E06c_TRAINING_GRAPH_NOT_YET_QUALIFIED', 'E06c_PHYSICAL_BATCH_240_NOT_YET_QUALIFIED',
                'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
SCOPE = {'training_launched': False, 'optimizer_constructed': False, 'optimizer_applications': 0,
         'backward_passes': 0, 'physical_batch_240_qualified': False, 'benchmark_data_access': False,
         'TEST_access': False, 'checkpoint_created': False, 'synthetic_bank': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6D5a: ' + message)


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


def source_cache():
    """Git metadata and the exact pinned closure; the removed CDCN weight is not read."""
    root = ROOT / 'third_party/source_cache/facexzoo'
    status = git('status', '--porcelain', root=root).decode().strip()
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(status == 'D addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl', 'source-cache status')
    return root, status


def repeatability(p, q):
    fields = ('diagnostic_seed', 'contract', 'source_before', 'source_after', 'asset_before', 'lightcnn',
              'binding', 'parameters', 'optimizer_ownership', 'modes', 'shapes', 'forward',
              'reparameterization', 'one_logit_classifier', 'cuda_memory', 'environment_before', 'counters')
    identity = {k: p[k] == q[k] for k in fields}
    require(p['forward'].keys() == q['forward'].keys(), 'same diagnostic tensor inventory')
    bitwise = {n: p['forward'][n]['sha256'] == q['forward'][n]['sha256'] for n in p['forward']}
    params = {m: p['parameters'][m]['sha256'] == q['parameters'][m]['sha256'] for m in p['parameters']}
    return {'independent_processes': 2, 'field_identity': identity,
            'forward_tensors_compared': len(bitwise), 'forward_bitwise_equal': sum(bitwise.values()),
            'all_forward_bitwise_equal': all(bitwise.values()),
            'initial_parameters_bitwise_equal': all(params.values()),
            'process_json_bytes_identical': sha(json.dumps(p, sort_keys=True).encode()) ==
                                            sha(json.dumps(q, sort_keys=True).encode()),
            'stochastic_note': ('reparameterize draws eps from the global CUDA default generator; with the fixed '
                                'qualification seed both fresh processes drew identical eps. Equality is observed, '
                                'not forced: RNG semantics were not changed.')}


def check_process(r):
    require(r['status'] == 'PASS' and r['label'] == 'DIAGNOSTIC_ARCHITECTURE_BATCH_ONLY', 'GPU process pass')
    require(r['source_before'] == r['source_after'], 'source mutation')
    s = r['source_before']
    require((s['commit'], s['tree']) == (PIN, TREE), 'pinned source identity')
    require(r['asset_before'] == r['asset_after'] and r['asset_before']['sha256'] == LIGHTCNN_SHA and
            r['asset_before']['bytes'] == 123844849, 'LightCNN identity')
    require(r['environment_before'] == r['environment_after'], 'environment mutation')
    l = r['lightcnn']
    require((l['checkpoint_tensor_count'], l['expected_model_tensor_count'], l['matched']) == (61, 60, 60) and
            not l['missing'] and not l['shape_mismatches'] and
            l['filtered_extra_keys'] == {'module.fc2.weight': [80013, 256]}, 'LightCNN 60/60')
    for n, (count, tensors) in EXPECTED_PARAMETERS.items():
        require((r['parameters'][n]['parameters'], r['parameters'][n]['parameter_tensors']) == (count, tensors),
                'parameter count ' + n)
    require(r['parameters']['netIP']['requires_grad_parameters'] == 0 and r['modes']['netIP'] is False,
            'netIP frozen/eval')
    require((r['optimizer_ownership']['parameters'], r['optimizer_ownership']['parameter_tensors']) ==
            (45_370_182, 55) and r['optimizer_ownership']['netCls_excluded'], 'optimizer ownership')
    require(r['finite_all'] and r['one_logit_classifier']['cross_entropy'] == 0.0, 'finite/degeneracy')
    require(r['counters'] == {'optimizer_constructions': 0, 'optimizer_applications': 0,
                              'backward_passes': 0, 'checkpoint_saves': 0}, 'zero training calls')
    require(all(r[k] == v for k, v in SCOPE.items()), 'scope flags')
    require(not r['firewall']['denied'] and r['compatibility_patch'] == 'NONE', 'firewall/patch')
    require(r['cuda_memory']['label'] == 'DIAGNOSTIC_BATCH1_ONLY' and
            not r['cuda_memory']['establishes_physical_batch_240_feasibility'], 'memory label')


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
    return out.getvalue().encode(), len(rows), baseline


def verify(check_index=True):
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    root, status = source_cache()
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS, 'qualification status')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    p, q = [json.loads(read(n)) for n in PROCESSES]
    for r in (p, q):
        check_process(r)
    for rel, h in p['source_before']['files_sha256'].items():
        raw = (root / rel).read_bytes()
        require(sha(raw) == h, 'laptop source closure ' + rel)
    require(repeatability(p, q) == audit['repeatability'], 'repeatability evidence')
    lock = read('environments/e06c.lock.json')
    require(sha(lock) == audit['environment_lock_sha256'], 'environment lock digest')
    lockj = json.loads(lock)
    require(lockj['compatibility_patch'] == 'NONE' and
            lockj['runtime_json_sha256'] == sha(read('environments/e06c.runtime.json')), 'lock linkage')
    require(lockj['runtime_harness_sha256'] == sha(read('methods/dsdg/runtime.py')), 'harness identity')
    for name, h in lockj['exports_sha256'].items():
        require(sha(read('environments/e06c.' + name)) == h, 'export ' + name)
    runtime = json.loads(read('environments/e06c.runtime.json'))
    require(runtime['created_by_E06c'] and not runtime['reused_existing_environment'] and
            runtime['gpat_m5_unchanged'], 'environment ownership')
    require(runtime['identity'] == p['environment_before'], 'runtime identity')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix) and len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == CLASSIFICATION and row['final_status'] == FINAL_STATUS, 'ledger row')
    require(all(row[k] == v for k, v in SCOPE.items()), 'ledger scope flags')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count, _ = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'ledger_rows': LEDGER_PREFIX_ROWS + 1,
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'independent_gpu_processes': 2, 'source_cache_status': status,
            'optimizer_constructed': False, 'backward_passes': 0, 'torch_imported': False,
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
