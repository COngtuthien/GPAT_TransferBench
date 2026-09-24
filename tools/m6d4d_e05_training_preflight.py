#!/usr/bin/env python3
"""Validate retained synthetic evidence without executing Torch or optimizer steps.

Index rebuild is LAST, from committed metadata plus explicit new artifact paths.
Historical index targets, benchmark manifests and runtime data are never opened.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'c27a110258f58f9bc035875aa40caf1ae681f9cb'
OVERLAY = 'configs/amendments/e05_training_runner_resolution.yaml'
OVERLAY_SHA = '8eee350c5642f2de38c92af366b99e80f7b6dab4404433ff88ca71da87b92828'
LOCK_SHA = '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50'
PREFIX_SHA = 'a0d88e113c6e23f90b33d9841b718eac6125072aae204b59b1a91e61c9c99570'
BASE = 'outputs/audit/M6D4D_E05_TRAINING_RUNNER_QUALIFICATION'
PROCESSES = tuple(f'outputs/audit/M6D4D_E05_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2))
LOG = 'outputs/audit/M6D4D_E05_SYNTHETIC_RUNTIME_LOG.txt'
IMPLEMENTATION = tuple('methods/pcgan/' + n + '.py' for n in
                       ('training_losses', 'training_runner', 'training_diagnostics', 'training_qualification'))
ARTIFACTS = (*IMPLEMENTATION, 'tests/test_m6d4d_e05_training.py',
             'tools/m6d4d_e05_training_preflight.py', BASE + '.md', BASE + '.json', *PROCESSES, LOG)
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
COUNTS = dict(Encoder=929960, Generator=3699233, ImageD=28859521, PatchD=24522145)
G_TERMS = ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat')
D_TERMS = ('L_D_real', 'L_D_rec', 'L_D_mix', 'L_D_patch_real', 'L_D_patch_fake')
FINAL_STATUS = ['E05_ARCHITECTURE_RUNTIME_QUALIFIED', 'E05_TRAINING_RUNNER_CONTRACT_RESOLVED',
                'E05_TRAINING_RUNNER_QUALIFIED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']


def require(ok, message):
    if not ok:
        raise ValueError('M6D4d: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence')
    require(p.parts and p.parts[0] in {'methods', 'tests', 'tools', 'outputs', 'configs', 'docs', 'environments'},
            'evidence root')
    require(not set(p.parts) & {'data', 'manifests', 'faces_256', 'runs', 'cache'}, 'data firewall')
    require(p.suffix in {'.py', '.json', '.jsonl', '.md', '.yaml', '.csv', '.txt'}, 'evidence extension')
    require(not (ROOT / p).is_symlink(), 'no evidence symlink')
    return (ROOT / p).read_bytes()


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def validate_totals(record):
    v = record['scalars']
    if 'L_G_total' in v:
        total = v[G_TERMS[0]]
        for n in G_TERMS[1:]:
            total = f32(total + v[n])
        expected = dict(L_G_total=total)
    else:
        image = f32(f32(v['L_D_real'] + f32(.5 * v['L_D_rec'])) + f32(.5 * v['L_D_mix']))
        patch = f32(v['L_D_patch_real'] + v['L_D_patch_fake'])
        expected = dict(L_D_image=image, L_D_patch=patch, L_D_total=f32(image + patch))
    for name, value in expected.items():
        require(value == v[name] == record['external'][name]['value'], 'external exact scalar equality')
        bits = struct.pack('<f', value).hex()
        require(record['external'][name]['fp32_hex'] == record['external'][name]['runner_fp32_hex'] == bits,
                'external exact FP32 bits')
        require(record['external'][name]['exact'] is True, 'external equality measured')


def tensors(value, prefix=''):
    if isinstance(value, dict):
        if {'shape', 'dtype', 'finite', 'sha256', 'min', 'max', 'mean', 'l2'} <= value.keys():
            yield prefix, value
        else:
            for key, item in value.items():
                yield from tensors(item, prefix + '/' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from tensors(item, prefix + '/' + str(index))


def validate_process(r, number):
    require(r['status'] == 'PASS' and r['process_number'] == number and r['launched'], 'GPU process PASS')
    require(r['overlay_sha256'] == OVERLAY_SHA and r['environment_lock_sha256'] == LOCK_SHA, 'authority identity')
    require(r['environment_before'] == r['environment_after'], 'environment unchanged')
    require(r['source_before'] == r['source_after'], 'source unchanged')
    require(r['build_closure_unchanged'] and not r['build_closure']['jit_build_invoked'], 'read-only build')
    require(r['optimizer_applications'] == dict(D=1, G=1, total=2) and r['benchmark_iteration'] == 1,
            'one complete D->G iteration')
    require(r['D_fakes_detached'] and r['G_recomputation_verified'], 'detach and recomputation')
    require(not r['firewall']['denied'], 'process firewall')
    require(not any(r[k] for k in ('benchmark_data_access', 'benchmark_training', 'checkpoints',
                                   'pretrained_loads', 'synthetic_bank')), 'synthetic execution scope')
    require(r['fidelity'] == 'CONTROLLED_ADAPTATION', 'fidelity')
    require([x['stage'] for x in r['events']] == ['before_D', 'D_forward', 'D_backward', 'after_D',
            'before_G', 'G_forward', 'G_backward', 'after_G'], 'D->G event order')
    require([x['benchmark_iteration'] for x in r['events']] == [0]*7 + [1], 'counter boundary')
    for record in ('initial_G', 'probe_D', 'D_step_losses', 'G_step_losses'):
        validate_totals(r[record])
    identities = []
    for group, components in [('G', ('Encoder', 'Generator')), ('D', ('ImageD', 'PatchD'))]:
        inv = r['optimizer_membership'][group]
        entries = inv['entries']
        require(sum(v['numel'] for v in entries) == inv['parameter_count'] == sum(COUNTS[k] for k in components),
                'optimizer parameter counts')
        require(len(entries) == inv['tensor_count'] and len({v['name'] for v in entries}) == len(entries), 'names')
        require(inv['lr'] == 1e-6 and inv['betas'] == [.9, .999] and inv['weight_decay'] == 0, 'Adam settings')
        identities.extend(v['identity'] for v in entries)
        expected_names = {k + '.' + n for k in components for n in r['parameters_initial'][k]}
        require({v['name'] for v in entries} == expected_names, 'exact named membership')
        state = r[group + '_adam']
        require(state['count'] == len(entries) and state['ownership_verified'], 'Adam ownership')
        require(set(state['entries']) == expected_names, 'Adam state exact names')
        for name, item in state['entries'].items():
            require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'} and item['step']['mean'] == 1, 'Adam state structure')
            shape = next(v['shape'] for v in entries if v['name'] == name)
            require(item['exp_avg']['shape'] == item['exp_avg_sq']['shape'] == shape, 'Adam moments shape')
        grads = r[group + '_step_gradients'][group]
        require(grads['non_none'] == grads['finite'] == grads['total'] == len(entries), 'finite expected gradients')
        require(grads['nonzero'] > 0 and set(grads['entries']) == expected_names, 'active gradients')
        require(r[group + '_step_gradients']['G' if group == 'D' else 'D']['non_none'] == 0, 'gradient isolation')
    require(len(set(identities)) == len(identities) and not r['optimizer_overlap'], 'no duplicates or overlap')
    for name, count in COUNTS.items():
        require(r['models'][name]['parameter_count'] == count, 'live model counts')
        if name in ('Encoder', 'Generator'):
            require(r['parameters_initial'][name] == r['parameters_post_D'][name], 'D-step E/G isolation')
            require(r['G_changes']['counts'][name]['changed'] > 0, 'G update')
        else:
            require(r['parameters_post_D'][name] == r['parameters_post_G'][name], 'G-step D isolation')
            require(r['D_changes']['counts'][name]['changed'] > 0, 'D update')
    for generator, key, terms in [(True, 'G_connectivity', G_TERMS), (False, 'D_connectivity', D_TERMS)]:
        require(set(r[key]) == set(terms), 'complete per-loss probes')
        for term, groups in r[key].items():
            if generator:
                connected = {'Encoder', 'Generator', 'x_src', 'z_pat_src'}
                connected.update({'z_con_src'} if term in ('L_rec', 'L_advrec') else {'x_tgt', 'z_con_tgt'})
                if term in ('L_advrec', 'L_advmix'):
                    connected.add('ImageD')
                if term == 'L_pat':
                    connected.add('PatchD')
            else:
                connected = {'ImageD' if term in D_TERMS[:3] else 'PatchD'}
                if term == 'L_D_real':
                    connected.update(('x_src', 'x_tgt'))
                if term in D_TERMS[3:]:
                    connected.add('x_src')
            for name, inventory in groups.items():
                require(inventory['finite'] == inventory['non_none'], 'finite probe')
                require(inventory['nonzero'] > 0 if name in connected else inventory['non_none'] == 0,
                        'expected connectivity ' + term + '/' + name)
    for name, record in tensors(r):
        require(record['finite'] and record['dtype'] == 'float32', 'finite FP32 tensor ' + name)
    for record in r['noise_rng']:
        require(record['before'] != record['after'], 'fresh noise RNG')
    require(r['checkpoint_policy'] == json.loads(read(OVERLAY))['checkpoint'], 'static checkpoint policy')


def comparison(p, q):
    structural = {key: p[key] == q[key] for key in ('environment_before', 'source_before', 'models',
        'build_closure', 'overlay_sha256', 'environment_lock_sha256', 'parameters_initial',
        'inputs', 'noise_rng', 'optimizer_applications', 'benchmark_iteration', 'events')}
    def membership(r):
        return {key: {**v, 'entries': [{n: x for n, x in e.items() if n != 'identity'} for e in v['entries']]}
                for key, v in r['optimizer_membership'].items()}
    structural['optimizer_membership_by_name'] = membership(p) == membership(q)
    require(all(structural.values()), 'two-process structural identity')
    left, right = dict(tensors(p)), dict(tensors(q))
    require(left.keys() == right.keys(), 'same tensor inventory')
    differences = {}
    max_delta = dict(min=0., max=0., mean=0., l2=0.)
    for path, record in left.items():
        other = right[path]
        require(record['shape'] == other['shape'] and record['dtype'] == other['dtype'], 'tensor structure')
        if record['sha256'] != other['sha256']:
            delta = {k: abs(record[k] - other[k]) for k in max_delta}
            differences[path] = dict(summary_absolute_differences=delta,
                                     nonzero_element_count_difference=other['nonzero'] - record['nonzero'])
            for k in max_delta:
                max_delta[k] = max(max_delta[k], delta[k])
    losses = {key: {n: abs(p[key]['scalars'][n] - q[key]['scalars'][n]) for n in p[key]['scalars']}
              for key in ('initial_G', 'probe_D', 'D_step_losses', 'G_step_losses')}
    return dict(structural_identity=structural, tensor_records_compared=len(left),
                different_tensor_hashes=len(differences), differences=differences,
                maximum_summary_absolute_differences=max_delta, loss_absolute_differences=losses,
                all_losses_bitwise_equal=all(x == 0 for v in losses.values() for x in v.values()),
                measurement_limit='Summary-statistic differences are not elementwise error bounds. No tensors/checkpoints persisted.',
                likely_source='Native CUDA backward accumulation, including bilinear interpolation/grid_sample; inferred, not profiled.')


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    rows = {r['path']: r for r in reader}
    require(len(rows) == 555 and reader.fieldnames == ['path', 'size_bytes', 'sha256'], 'baseline index')
    for path in ARTIFACTS:
        require(path not in rows, 'additive artifact')
        raw = read(path)
        rows[path] = dict(path=path, size_bytes=str(len(raw)), sha256=sha(raw))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return stream.getvalue().encode(), len(rows)


def verify(check_index=True):
    require(all(git('rev-parse', ref).decode().strip() == AUTHORITY for ref in ('HEAD', 'origin/m6-baselines')),
            'laptop authority')
    require(git('rev-list', '--left-right', '--count', 'HEAD...origin/m6-baselines').split() == [b'0', b'0'], 'divergence')
    require(set(git('diff', '--name-only', AUTHORITY).decode().splitlines()) <= {LEDGER, INDEX},
            'all historical tracked files unchanged')
    require(sha(read(OVERLAY)) == OVERLAY_SHA and sha(read('environments/e05.lock.json')) == LOCK_SHA, 'immutable authority')
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS and audit['synthetic_numerical_status'] == 'PASS', 'numerical qualification')
    require(audit['scope_status'] == 'EXCEPTION_DISCLOSED' and len(audit['scope_exceptions']) == 1,
            'broad instruction-file search exception must remain disclosed')
    for path, digest in audit['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'artifact hash ' + path)
    for path, digest in audit['preserved_inputs_sha256'].items():
        raw = read(path)
        require(sha(raw) == digest and raw == git('show', AUTHORITY + ':' + path), 'historical identity ' + path)
    p, q = [json.loads(read(path)) for path in PROCESSES]
    for number, r in enumerate((p, q), 1):
        validate_process(r, number)
    require(comparison(p, q) == audit['repeatability'], 'retained repeatability')
    log = read(LOG).decode()
    for path in IMPLEMENTATION:
        require(sha(read(path)) + '  /tmp/gpat_m6d4d_qualification/' + path in log,
                'executed/staged implementation identity')
    prefix, current = git('show', AUTHORITY + ':' + LEDGER), read(LEDGER)
    require(len(prefix.splitlines()) == 108 and sha(prefix) == PREFIX_SHA, 'committed 108-row prefix')
    require(current.startswith(prefix) and len(current.splitlines()) == 109, 'single ledger append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == 'M6D4D_E05_TRAINING_RUNNER_QUALIFICATION', 'ledger classification')
    require(row['optimizer_applications'] == dict(D=2, G=2, total=4), 'total application budget')
    require(row['scope_status'] == audit['scope_status'], 'ledger scope disclosure')
    for path, digest in row['artifacts_sha256'].items():
        require(sha(read(path)) == digest, 'ledger artifact hash')
    raw, count = expected_index()
    if check_index:
        require(read(INDEX) == raw, 'CRLF index rebuilt last')
    return dict(synthetic_numerical_status='PASS', scope_status=audit['scope_status'], ledger_rows=109,
                first_108_rows_byte_identical=True, artifact_index_rows=count,
                optimizer_applications=dict(D=2, G=2, total=4), model_execution_in_preflight=False)


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
