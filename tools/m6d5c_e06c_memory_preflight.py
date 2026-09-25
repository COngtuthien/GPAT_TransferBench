#!/usr/bin/env python3
"""Verify retained M6D5c E06c microbatch execution-resolution evidence; rebuild the index.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest payload read and no image I/O. Validates the owner overlay,
the B=4 reference check, both fresh B=240 GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1
processes and their repeatability, the retained M6D5b physical-B=240 OOM history,
the single ledger append and the CRLF artifact index. Historical index rows are
carried from the authoritative commit unopened.
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
AUTHORITY = '0c62dbe0bc07b305b80eabadf050d8aedeb22859'
PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
BASE = 'outputs/audit/M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION'
OVERLAY = 'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml'
REFERENCE = 'outputs/audit/M6D5C_E06C_REFERENCE_CHECK.json'
PROCESSES = ('outputs/audit/M6D5C_E06C_PROCESS_1.json', 'outputs/audit/M6D5C_E06C_PROCESS_2.json')
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 112
LEDGER_PREFIX_SHA = 'ff9814f6e3af4806193e801fa0626eb34333f18d3759ed533668558a19a5754b'
INDEX_BASELINE_ROWS = 592
CLASSIFICATION = 'M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION'
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1'
ARTIFACTS = (OVERLAY,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Memory_Execution_Resolution_Addendum_M6D5c.md',
             'methods/dsdg/microbatch_execution.py', 'methods/dsdg/microbatch_qualification.py',
             'tests/test_m6d5c_e06c_microbatch_execution.py', 'tools/m6d5c_e06c_memory_preflight.py',
             REFERENCE, *PROCESSES, 'outputs/audit/M6D5C_E06C_RUNTIME_LOG.txt', BASE + '.md')
AUDIT_ARTIFACTS = ARTIFACTS                      # the audit JSON records every other new artifact
LEDGER_ARTIFACTS = ARTIFACTS + (BASE + '.json',)
PRESERVED = ('configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml',
             'methods/dsdg/adapter.py', 'methods/dsdg/source.py', 'methods/dsdg/runtime.py', 'methods/dsdg/__init__.py',
             'methods/dsdg/training_graph.py', 'methods/dsdg/training_qualification.py',
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py',
             'environments/e06c.lock.json', 'environments/e06c.runtime.json',
             'environments/e06c.conda-explicit.txt', 'environments/e06c.pip-freeze.txt',
             'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json', 'third_party/source_pins.json',
             'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.md',
             'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.json',
             'outputs/audit/M6D5A_E06C_RUNTIME_LOG.txt', 'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_1.json',
             'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_2.json',
             'tools/m6d5a_e06c_runtime_preflight.py', 'tests/test_m6d5a_e06c_runtime.py',
             'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.md',
             'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.json',
             'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_1.json', 'outputs/audit/M6D5B_E06C_RUNTIME_LOG.txt',
             'tools/m6d5b_e06c_training_preflight.py', 'tests/test_m6d5b_e06c_training_graph.py')
FINAL_STATUS = ['M6D5c PASS', 'E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED', 'E06c_GLOBAL_STATISTIC_MICROBATCH_QUALIFIED',
                'E06c_TRAINING_GRAPH_EXECUTION_QUALIFIED', 'E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED',
                'E06c_FULL_TRAINING_NOT_YET_EXECUTED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
SCOPE = {'physical_batch_240': 'OOM_RETAINED', 'global_batch_size': 240, 'microbatch_size': 20, 'microbatches': 12,
         'execution': EXECUTION_MODE, 'optimizer_applications': 2, 'backward_calls': 24,
         'benchmark_training': False, 'benchmark_data_access': False, 'TEST_access': False,
         'checkpoint_created': False, 'synthetic_bank': False}
EXPECTED_PARAMETERS = {'netE_nir': (13_790_048, 17), 'netE_vis': (11_692_640, 17),
                       'netG': (19_887_494, 21), 'netCls': (129, 2), 'netIP': (10_475_872, 60)}
B240_COUNTERS = {'optimizer_constructions': 1, 'optimizer_step_entries': 1, 'optimizer_applications': 1,
                 'backward_calls': 12, 'autograd_backward_calls': 12, 'autograd_grad_calls': 0, 'checkpoint_saves': 0,
                 'activation_checkpoint_calls': 0, 'zero_grad_calls': [{'backward_calls_before': 0, 'set_to_none': False}]}
REPEAT_KEYS = ('epsilon', 'inputs', 'losses', 'global_statistics', 'gradient_sha256', 'gradient_inventory',
               'parameters_after_sha256', 'memory_by_phase', 'overall_peak', 'counters', 'parameters_initial')


def require(ok, message):
    if not ok:
        raise ValueError('M6D5c: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'third_party', 'docs'},
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


def check_common(r, lock, overlay_sha):
    require(r['diagnostic_seed'] == 60503 and r['diagnostic_seed'] not in (42, 1337, 2026), 'qualification seed')
    require(r['execution_mode'] == EXECUTION_MODE and r['classification'] == 'CONTROLLED_EXECUTION_ADAPTATION' and
            r['draw_policy'] == 'STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION', 'execution mode')
    require(r['resolution_sha256'] == overlay_sha and r['adapter_generic_accumulation_refused'], 'overlay binding')
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
    p = r['environment_before']['precision']
    require((p['default_dtype'], p['matmul_tf32'], p['cudnn_tf32'], p['autocast_cuda'], p['matmul_precision']) ==
            ('torch.float32', False, False, False, 'highest'), 'FP32 / no TF32 / no AMP')
    require(r['environment_before']['launch_environment']['NVIDIA_TF32_OVERRIDE'] == '0', 'TF32 override off')
    require(r['pinned_statement_mismatches'] == [] and r['pinned_statements_verified'] == 46, 'pinned statements')
    l = r['lightcnn']
    require((l['checkpoint_tensor_count'], l['expected_model_tensor_count'], l['matched']) == (61, 60, 60) and
            not l['missing'] and not l['shape_mismatches'], 'LightCNN 60/60')
    for n, (count, tensors) in EXPECTED_PARAMETERS.items():
        q = r['parameters_initial'][n]
        require((q['parameters'], q['parameter_tensors']) == (count, tensors), 'parameter count ' + n)
    require(r['batch_coupling']['batchnorm_modules'] == 0 and r['batch_coupling']['instancenorm_modules'] == 36 and
            not r['batch_coupling']['instancenorm_track_running_stats'], 'per-sample architecture')
    e = r['epsilon']
    require(e['order'] == ['cls', 'nir', 'vis'] and e['dtype'] == 'torch.float32' and e['rng_state_after_equal'] and
            all(e['bitwise_equal_to_pinned_reparameterize_draws'].values()), 'epsilon draw proof')
    require(r['firewall']['denied'] == [] and r['compatibility_patch'] == 'NONE' and
            r['fidelity'] == 'CONTROLLED_ADAPTATION', 'firewall/patch/fidelity')
    require(not any(r[k] for k in ('checkpoint_created', 'benchmark_training', 'benchmark_data_access', 'TEST_access',
                                   'synthetic_bank')), 'process scope flags')
    c = r['counters']
    require(c['autograd_grad_calls'] == c['checkpoint_saves'] == c['activation_checkpoint_calls'] == 0, 'forbidden calls')


def check_reference(r, overlay):
    g = overlay['qualification']['reference_check']['gates']
    require(r['status'] == 'PASS' and r['mode'] == 'reference', 'reference PASS')
    require((r['reference_global_batch'], r['reference_microbatch']) == (4, 2), 'reference batch 4 / microbatch 2')
    require(r['gate_thresholds'] == g, 'reference gate thresholds frozen in overlay')
    c = r['comparison']
    values = list(r['path_a_full_batch']['losses'].values()) + list(r['path_b_microbatch']['losses'].values())
    require(all(v == v and abs(v) != float('inf') for v in values), 'finite reference losses')
    require(c['total_rel_error'] <= g['total_loss_relative_error_max'], 'reference total loss')
    require(c['gradient']['cosine'] >= g['aggregate_gradient_cosine_min'], 'reference gradient cosine')
    require(c['gradient']['relative_l2'] <= g['aggregate_gradient_relative_l2_max'], 'reference gradient rel-L2')
    require(c['update']['cosine'] >= g['post_step_parameter_delta_cosine_min'], 'reference update direction')
    require(r['path_a_full_batch']['owned_grad_non_none'] == r['path_b_microbatch']['owned_grad_non_none'] == 55,
            'reference coverage')
    require(all(r['gates'].values()), 'reference recorded gates')
    k = r['counters']
    require((k['optimizer_constructions'], k['optimizer_applications'], k['backward_calls']) == (2, 2, 3),
            'reference accounting')
    return {'total_rel_error': c['total_rel_error'], 'gradient_cosine': c['gradient']['cosine'],
            'gradient_relative_l2': c['gradient']['relative_l2'], 'update_cosine': c['update']['cosine']}


def check_b240(r):
    require(r['status'] == 'PASS' and r['mode'] == 'b240' and r['global_batch_240_executed'], 'B=240 PASS')
    require((r['global_batch_size'], r['microbatch_size'], r['microbatches_per_step']) == (240, 20, 12), '240=12x20')
    require(r['physical_batch_240'] == 'OOM_RETAINED', 'physical B=240 history retained')
    require(r['counters'] == B240_COUNTERS, 'B=240 counters')
    require((r['optimizer_constructions'], r['optimizer_applications'], r['backward_calls']) == (1, 1, 12), 'accounting')
    i = r['inputs']
    require(i['distinct_rows'] == 480 and all(i['equal_to_m6d5b_inputs'].values()) and
            i['x_spoof']['shape'] == [240, 3, 256, 256], 'B=240 synthetic inputs equal M6D5b')
    phases = [m['phase'] for m in r['memory_by_phase']]
    require(phases == ['after_model_construction', 'after_optimizer_and_input_allocation', 'after_epsilon_allocation',
                       'pass1_peak'] + ['pass2_chunk_%02d_peak' % n for n in range(12)] +
            ['before_optimizer_step', 'after_optimizer_step'], 'memory phases')
    require(r['overall_peak']['allocated_bytes'] == max(m['peak_allocated_bytes'] for m in r['memory_by_phase']),
            'overall peak')
    g = r['global_statistics']
    require(all(g['pass2_latents_bitwise_equal_pass1'].values()) and g['pass2_recomputed_sign_mismatches'] == 0 and
            g['pass2_recomputed_ort_sign_equal'], 'epsilon replay / sign consistency')
    for k, v in (('loss_mmd_surrogate', 'loss_mmd_global_fp32'), ('loss_ort_surrogate', 'loss_ort_global_fp32')):
        require(abs(g['surrogate_sums_pass2'][k] - g[v]) <= 1e-4 * max(1.0, abs(g[v])), 'surrogate sum ' + k)
    for k in ('mmd', 'ort'):
        a, b = g['loss_%s_global_fp32' % k], g['loss_%s_global_float64_recomputed' % k]
        require(abs(a - b) <= 1e-4 * max(1.0, abs(b)), 'float64 recomputation ' + k)
    L = r['losses']
    comp = L['global_components']
    require(all(v == v and abs(v) != float('inf') for v in comp.values()), 'finite global losses')
    require(comp['loss_cls'] == comp['loss_pair'] == 0.0, 'cls/pair zero')
    epoch1 = comp['loss_rec'] + 0.01 * comp['loss_kl'] + 0.01 * comp['loss_mmd'] + 0.01 * comp['loss_ip'] + \
        0.01 * comp['loss_pair'] + 0.01 * comp['loss_cls'] + 0.01 * comp['loss_ort']
    require(abs(epoch1 - L['epoch1_total_reconstructed']) <= 1e-9 * abs(epoch1), 'epoch-1 reconstruction')
    require(abs(L['sum_of_chunk_objectives'] - epoch1) <= 1e-4 * abs(epoch1), 'sum of chunk objectives')
    require(L['postwarmup_executed'] is False, 'post-warmup algebraic only')
    require(len(r['chunk_evidence']) == 12, '12 chunks')
    for h in r['chunk_evidence']:
        require(not h['nir_fc_requires_grad'] and not h['vis_fc_requires_grad'], 'LightCNN target no_grad')
        require(h['rec_nir_fc_requires_grad'] and h['rec_vis_fc_requires_grad'], 'reconstruction features keep graph')
        require(h['grads']['rec_nir']['nonzero'] > 0 and h['grads']['rec_vis']['nonzero'] > 0, 'LightCNN gradient path')
        require(h['grads']['pre_spoof']['nonzero'] == 0, 'one-logit CE degeneracy')
    inv = r['gradient_inventory']
    require(sum(inv[n]['non_none'] for n in ('netE_nir', 'netE_vis', 'netG')) == 55 and inv['netIP']['non_none'] == 0,
            '55/55 owned gradients; netIP none')
    require(r['zero_grad']['calls'] == B240_COUNTERS['zero_grad_calls'] and r['rng_no_redraw'], 'zero_grad / no redraw')
    s = r['optimizer_state']
    require(s['entries'] == 55 and s['only_owned'] and s['step_values'] == [1.0] and s['finite_moments'], 'Adam state')
    c = r['parameter_change']
    require(c['netCls']['changed_elements'] == c['netIP']['changed_elements'] == 0, 'netCls/netIP unchanged')
    require(all(c[n]['changed_tensors'] == c[n]['tensors'] for n in ('netE_nir', 'netE_vis', 'netG')), 'owned changed')
    return {'epoch1_total': L['epoch1_total_reconstructed'], 'peak_allocated_bytes': r['overall_peak']['allocated_bytes'],
            'peak_reserved_bytes': r['overall_peak']['reserved_bytes']}


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and
            len(rows) == len(baseline) == INDEX_BASELINE_ROWS, 'baseline index')
    for p in LEDGER_ARTIFACTS:
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
    require(not (ROOT / 'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_2.json').exists(), 'M6D5b process 2 still absent')
    lock_raw = read('environments/e06c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock identity')
    lock = json.loads(lock_raw)
    status = source_cache(lock)
    overlay_raw = read(OVERLAY)
    overlay = json.loads(overlay_raw)
    require((overlay['execution_mode'], overlay['global_batch_size'], overlay['microbatch_size'],
             overlay['microbatches_per_step'], overlay['optimizer_steps_per_global_batch'], overlay['precision'],
             overlay['amp'], overlay['tf32'], overlay['naive_microbatch_loss_averaging'], overlay['physical_batch_240'],
             overlay['fidelity_class']) ==
            (EXECUTION_MODE, 240, 20, 12, 1, 'FP32', False, False, 'forbidden', 'OOM_RETAINED', 'CONTROLLED_ADAPTATION'),
            'owner overlay')
    for path, h in overlay['immutable_inputs_sha256'].items():
        require(sha(read(path)) == h, 'overlay immutable input ' + path)
    m6d5b = json.loads(read('outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.json'))
    require(m6d5b['failure_classification'] == 'CUDA_OOM_PHYSICAL_BATCH_240' and
            m6d5b['physical_batch_240_qualified'] is False, 'M6D5b OOM history')
    overlay_sha = sha(overlay_raw)
    ref = json.loads(read(REFERENCE))
    check_common(ref, lock, overlay_sha)
    reference = check_reference(ref, overlay)
    procs = [json.loads(read(p)) for p in PROCESSES]
    b240 = []
    for r in procs:
        check_common(r, lock, overlay_sha)
        b240.append(check_b240(r))
    repeat = {k: procs[0][k] == procs[1][k] for k in REPEAT_KEYS}
    require(all(repeat.values()), 'repeatability ' + json.dumps(repeat))
    require(procs[0]['gpu_before']['own_pid'] != procs[1]['gpu_before']['own_pid'], 'two fresh processes')
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS and audit['decision'] == 'PASS' and
            audit['classification'] == CLASSIFICATION, 'audit status')
    require(set(audit['artifacts_sha256']) == set(AUDIT_ARTIFACTS), 'audit artifact set')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    require(all(audit[k] == v for k, v in SCOPE.items()), 'audit scope flags')
    report = read(BASE + '.md').decode()
    for word in ('CONTROLLED_ADAPTATION PRESERVED', 'E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED', EXECUTION_MODE):
        require(word in report, 'report wording ' + word)
    require('E06c_PHYSICAL_BATCH_240_QUALIFIED' not in report, 'no physical-B=240 qualification claim')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix) and len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == CLASSIFICATION and row['final_status'] == FINAL_STATUS and
            row['decision'] == 'PASS', 'ledger row')
    require(all(row[k] == v for k, v in SCOPE.items()), 'ledger scope flags')
    require(set(row['artifacts_sha256']) == set(LEDGER_ARTIFACTS), 'ledger artifact set')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'validated_state': 'M6D5C_PASS', 'execution_mode': EXECUTION_MODE,
            'ledger_rows': LEDGER_PREFIX_ROWS + 1, f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'reference_check': reference, 'b240_processes': b240, 'repeatability': repeat,
            'b240_optimizer_applications': 2, 'b240_backward_calls': 24, 'physical_batch_240': 'OOM_RETAINED',
            'source_cache_status': status, 'torch_imported': False, 'model_execution_in_preflight': False,
            'benchmark_data_access': False}


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
