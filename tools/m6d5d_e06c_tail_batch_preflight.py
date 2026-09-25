#!/usr/bin/env python3
"""Verify retained M6D5d E06c tail-batch (V2) execution-resolution evidence; rebuild the index.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no image data,
no manifest payload read, no VAL/TEST access. Validates the owner V2 overlay, the
pinned DataLoader (no drop_last), the retained 8838-row TRAIN provenance and epoch
arithmetic, the B=7 remainder reference, the V1/V2 B=240 compatibility check, both
fresh B=198 processes and their repeatability, V1/M6D5a-c immutability, the single
ledger append and the CRLF artifact index. Historical index rows are carried from the
authoritative commit unopened.
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
sys.path.insert(0, str(ROOT))
AUTHORITY = '666aba4803468c1d2888e347f8b09afa2ca47401'
PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
PAIRS_SHA = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
BASE = 'outputs/audit/M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION'
OVERLAY = 'configs/amendments/e06c_m6d5d_tail_batch_execution_resolution.yaml'
V1_OVERLAY = 'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml'
REFERENCE = 'outputs/audit/M6D5D_E06C_REFERENCE_CHECK.json'
V1COMPAT = 'outputs/audit/M6D5D_E06C_V1_COMPATIBILITY.json'
PROCESSES = ('outputs/audit/M6D5D_E06C_B198_PROCESS_1.json', 'outputs/audit/M6D5D_E06C_B198_PROCESS_2.json')
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 113
LEDGER_PREFIX_SHA = 'aa6e1ea56fb36c547e94bf750fbf3c86c698c61eac20c96bec05a4b740c2c292'
INDEX_BASELINE_ROWS = 604
CLASSIFICATION = 'M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION'
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'
TAIL_CHUNKS = [20] * 9 + [18]
ARTIFACTS = (OVERLAY,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Tail_Batch_Execution_Resolution_Addendum_M6D5d.md',
             'methods/dsdg/microbatch_execution_v2.py', 'methods/dsdg/microbatch_qualification_v2.py',
             'tests/test_m6d5d_e06c_tail_batch_execution.py', 'tools/m6d5d_e06c_tail_batch_preflight.py',
             REFERENCE, V1COMPAT, *PROCESSES, 'outputs/audit/M6D5D_E06C_RUNTIME_LOG.txt', BASE + '.md')
AUDIT_ARTIFACTS = ARTIFACTS
LEDGER_ARTIFACTS = ARTIFACTS + (BASE + '.json',)
PRESERVED = ('configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml', V1_OVERLAY,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Memory_Execution_Resolution_Addendum_M6D5c.md',
             'methods/dsdg/microbatch_execution.py', 'methods/dsdg/microbatch_qualification.py',
             'methods/dsdg/training_graph.py', 'methods/dsdg/training_qualification.py', 'methods/dsdg/runtime.py',
             'methods/dsdg/adapter.py', 'methods/dsdg/source.py', 'methods/dsdg/__init__.py',
             'methods/common/learned.py', 'methods/common/upstream.py', 'methods/common/config.py',
             'environments/e06c.lock.json', 'environments/e06c.runtime.json',
             'third_party/source_pins.json', 'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json',
             'outputs/audit/pairs_train_v1.sha256', 'outputs/audit/M4_COMMON_PAIR_AUDIT.json',
             'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.json', 'tests/test_m6d5a_e06c_runtime.py',
             'tools/m6d5a_e06c_runtime_preflight.py',
             'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.json', 'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_1.json',
             'tests/test_m6d5b_e06c_training_graph.py', 'tools/m6d5b_e06c_training_preflight.py',
             'outputs/audit/M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION.md',
             'outputs/audit/M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION.json',
             'outputs/audit/M6D5C_E06C_REFERENCE_CHECK.json', 'outputs/audit/M6D5C_E06C_PROCESS_1.json',
             'outputs/audit/M6D5C_E06C_PROCESS_2.json', 'outputs/audit/M6D5C_E06C_RUNTIME_LOG.txt',
             'tests/test_m6d5c_e06c_microbatch_execution.py', 'tools/m6d5c_e06c_memory_preflight.py')
FINAL_STATUS = ['M6D5d PASS', 'E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED', 'E06c_GLOBAL_STATISTIC_MICROBATCH_QUALIFIED',
                'E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED', 'E06c_EPOCH_BATCHING_QUALIFIED',
                'E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED', 'E06c_PRODUCTION_RUNNER_NOT_YET_QUALIFIED',
                'E06c_FULL_TRAINING_NOT_YET_EXECUTED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
SCOPE = {'execution': EXECUTION_MODE, 'nominal_global_batch': 240, 'final_global_batch': 198, 'max_microbatch': 20,
         'final_chunk': 18, 'drop_last': False, 'full_batches_per_epoch': 36, 'optimizer_steps_per_epoch': 37,
         'expected_train_rows': 8838, 'physical_batch_240': 'OOM_RETAINED',
         'benchmark_training': False, 'VAL_access': False, 'TEST_access': False, 'scientific_checkpoint': False,
         'checkpoint_created': False, 'synthetic_bank': False, 'benchmark_data_access': False}
QUALIFICATION_COUNTS = {'reference7': {'optimizer_constructions': 2, 'optimizer_applications': 2, 'backward_calls': 4},
                        'v1compat': {'optimizer_constructions': 2, 'optimizer_applications': 2, 'backward_calls': 24},
                        'b198_process_1': {'optimizer_constructions': 1, 'optimizer_applications': 1, 'backward_calls': 10},
                        'b198_process_2': {'optimizer_constructions': 1, 'optimizer_applications': 1, 'backward_calls': 10},
                        'total': {'optimizer_constructions': 6, 'optimizer_applications': 6, 'backward_calls': 48}}
B198_COUNTERS = {'optimizer_constructions': 1, 'optimizer_step_entries': 1, 'optimizer_applications': 1,
                 'backward_calls': 10, 'autograd_backward_calls': 10, 'autograd_grad_calls': 0, 'checkpoint_saves': 0,
                 'activation_checkpoint_calls': 0, 'zero_grad_calls': [{'backward_calls_before': 0, 'set_to_none': False}]}
REPEAT_KEYS = ('epsilon', 'inputs', 'losses', 'global_statistics', 'gradient_sha256', 'gradient_inventory',
               'parameters_after_sha256', 'memory_by_phase', 'overall_peak', 'counters', 'parameters_initial',
               'chunk_sizes', 'epoch_batch_plan', 'dataloader_evidence')


def require(ok, message):
    if not ok:
        raise ValueError('M6D5d: ' + message)


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
    require(p.suffix != '.parquet', 'no parquet reads')
    return (ROOT / p).read_bytes()


def source_cache(lock):
    root = ROOT / 'third_party/source_cache/facexzoo'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    status = git('status', '--porcelain', root=root).decode().strip()
    require(status == 'D addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl', 'source-cache status')
    for rel, h in lock['source']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    return (root / 'addition_module/DSDG/train_generator.py').read_text(), status


def dataloader_and_epoch(train_generator):
    """Static re-derivation, independent of the V2 module: lines 94-97, no drop_last, 8838 = 36*240 + 198."""
    lines = train_generator.splitlines()
    expected = {94: 'train_loader = torch.utils.data.DataLoader(',
                95: 'GenDataset_s(img_root=args.img_root, list_file=args.train_list, attack_type=args.attack_type),',
                96: 'batch_size=args.batch_size, shuffle=True,', 97: 'num_workers=args.workers, pin_memory=True)'}
    require(all(lines[n - 1].strip() == s for n, s in expected.items()), 'pinned DataLoader lines 94-97')
    require('drop_last' not in train_generator and train_generator.count('DataLoader(') == 1, 'no drop_last')
    sidecar = json.loads(read('outputs/audit/pairs_train_v1.sha256'))
    common = json.loads(read('outputs/audit/M4_COMMON_PAIR_AUDIT.json'))
    frozen = read('configs/frozen/dsdg_bin_idfree_v1.yaml').decode()
    require(sidecar['rows'] == 8838 and sidecar['sha256'] == PAIRS_SHA, 'M4 sidecar rows/sha')
    require(common['expected_rows']['TRAIN'] == 8838 and common['status'] == 'PASS', 'M4 common pair audit')
    require('  rows: 8838\n' in frozen and PAIRS_SHA in frozen, 'A1 frozen relation rows/sha')
    full, tail = divmod(8838, 240)
    require((full, tail, full + 1, 36 * 240 + 198) == (36, 198, 37, 8838), 'epoch arithmetic')
    return {'rows': 8838, 'full_batches': full, 'final_batch': tail, 'optimizer_steps_per_epoch': full + 1,
            'drop_last_in_source': False}


def check_common(r, lock, overlay_sha, v1_overlay_sha):
    require(r['diagnostic_seed'] == 60504 and r['diagnostic_seed'] not in (42, 1337, 2026), 'qualification seed')
    require(r['environment_before']['launch_environment']['PYTHONHASHSEED'] == '60504', 'PYTHONHASHSEED')
    require(r['execution_mode'] == EXECUTION_MODE and r['classification'] == 'CONTROLLED_EXECUTION_ADAPTATION', 'mode')
    require(r['v2_resolution_sha256'] == overlay_sha and r['v1_resolution_sha256'] == v1_overlay_sha and
            r['adapter_generic_accumulation_refused'], 'overlay binding')
    require(r['dataloader_evidence']['mismatches'] == [] and r['dataloader_evidence']['runtime_torch_default_drop_last']
            is False and r['dataloader_evidence']['drop_last_in_source'] is False, 'DataLoader evidence')
    require(r['epoch_batch_plan'] == {'batch_size': 240, 'drop_last': False, 'final_batch': 198, 'full_batches': 36,
                                      'optimizer_steps_per_epoch': 37, 'rows': 8838, 'rows_covered': 8838}, 'epoch plan')
    require(r['resource_clean'] and r['gpu_before']['compute_processes'] == [] and r['gpu_before']['used_mib'] <= 1024,
            'clean GPU')
    for k in ('source_before', 'source_after'):
        s = r[k]
        require((s['commit'], s['tree']) == (PIN, TREE) and s['files_sha256'] == lock['source']['files_sha256'], k)
    require(r['asset_before'] == r['asset_after'] and r['asset_before']['sha256'] == LIGHTCNN_SHA, 'LightCNN identity')
    require(r['environment_lock_sha256'] == LOCK_SHA and r['environment_before'] == r['environment_after'], 'env')
    require(all(r['environment_before'][k] == v for k, v in lock['identity'].items() if k != 'launch_environment'),
            'runtime identity equals lock')
    p = r['environment_before']['precision']
    require((p['default_dtype'], p['matmul_tf32'], p['cudnn_tf32'], p['autocast_cuda'], p['matmul_precision']) ==
            ('torch.float32', False, False, False, 'highest'), 'FP32 / no TF32 / no AMP')
    require(r['pinned_statement_mismatches'] == [] and r['pinned_statements_verified'] == 46, 'pinned statements')
    require(r['batch_coupling']['batchnorm_modules'] == 0 and r['batch_coupling']['instancenorm_modules'] == 36,
            'per-sample architecture')
    e = r['epsilon']
    require(e['order'] == ['cls', 'nir', 'vis'] and e['dtype'] == 'torch.float32' and e['rng_state_after_equal'] and
            all(e['bitwise_equal_to_pinned_reparameterize_draws'].values()), 'epsilon draw proof')
    require(r['firewall']['denied'] == [] and r['compatibility_patch'] == 'NONE' and
            r['fidelity'] == 'CONTROLLED_ADAPTATION', 'firewall/patch/fidelity')
    require(not any(r[k] for k in ('checkpoint_created', 'benchmark_training', 'benchmark_data_access', 'TEST_access',
                                   'synthetic_bank')), 'scope flags')
    c = r['counters']
    require(c['autograd_grad_calls'] == c['checkpoint_saves'] == c['activation_checkpoint_calls'] == 0, 'forbidden calls')


def check_reference(r, overlay):
    g = overlay['qualification']['reference_check']['gates']
    require(r['status'] == 'PASS' and r['mode'] == 'reference7' and r['gate_thresholds'] == g, 'reference7')
    require((r['reference_global_batch'], r['reference_max_microbatch'], r['inputs']['chunk_sizes']) == (7, 3, [3, 3, 1]),
            'B=7 chunks 3,3,1')
    c = r['comparison']
    vals = list(r['path_a_full_batch']['losses'].values()) + list(r['path_b_v2']['losses'].values())
    require(all(v == v and abs(v) != float('inf') for v in vals), 'finite reference losses')
    require(c['total_rel_error'] <= g['total_loss_relative_error_max'], 'reference total loss')
    require(c['gradient']['cosine'] >= g['aggregate_gradient_cosine_min'], 'reference gradient cosine')
    require(c['gradient']['relative_l2'] <= g['aggregate_gradient_relative_l2_max'], 'reference gradient rel-L2')
    require(all(r['gates'].values()), 'reference gates')
    k = r['counters']
    require((k['optimizer_constructions'], k['optimizer_applications'], k['backward_calls']) == (2, 2, 4), 'accounting')
    return {'total_rel_error': c['total_rel_error'], 'gradient_cosine': c['gradient']['cosine'],
            'gradient_relative_l2': c['gradient']['relative_l2']}


def check_v1compat(r, overlay):
    g = overlay['qualification']['v1_compatibility']['gates']
    require(r['status'] == 'PASS' and r['mode'] == 'v1compat' and r['gate_thresholds'] == g, 'v1compat')
    c = r['comparison']
    require(c['chunk_sizes'] == {'v1': [20] * 12, 'v2': [20] * 12} and c['backward_calls'] == {'v1': 12, 'v2': 12},
            'chunk inventory')
    require(max(c['global_loss_rel_diff'].values()) <= g['global_loss_relative_diff_max'], 'global losses')
    require(c['delta_max_abs_diff'] <= g['delta_max_abs_diff'] and c['mmd_sign_equal'] and
            c['ort_sign']['v1'] == c['ort_sign']['v2'] and
            abs(c['ort_mean']['v1'] - c['ort_mean']['v2']) <= g['ort_mean_abs_diff'], 'statistics and signs')
    require(c['gradient']['cosine'] >= g['gradient_cosine_min'] and
            c['gradient']['relative_l2'] <= g['gradient_relative_l2_max'], 'gradients')
    inv = c['gradient_inventory']
    require(all(inv['v1'][n]['non_none'] == inv['v2'][n]['non_none'] for n in inv['v1']) and
            sum(inv['v1'][n]['non_none'] for n in ('netE_nir', 'netE_vis', 'netG')) == 55, 'gradient inventory')
    require(all(r['gates'].values()), 'v1compat gates')
    k = r['counters']
    require((k['optimizer_constructions'], k['optimizer_applications'], k['backward_calls']) == (2, 2, 24), 'accounting')
    return {'delta_bitwise_equal': c['delta_bitwise_equal'], 'gradient_bitwise_equal': c['gradient_bitwise_equal'],
            'post_step_parameters_bitwise_equal': c['post_step_parameters_bitwise_equal_reported_not_gated']}


def check_b198(r):
    require(r['status'] == 'PASS' and r['mode'] == 'b198' and r['tail_batch_198_executed'], 'B=198 PASS')
    require((r['global_batch_size'], r['nominal_global_batch'], r['max_microbatch'], r['final_chunk'], r['drop_last'])
            == (198, 240, 20, 18, False) and r['chunk_sizes'] == TAIL_CHUNKS, 'B=198 chunk plan')
    require(r['counters'] == B198_COUNTERS and
            (r['optimizer_constructions'], r['optimizer_applications'], r['backward_calls']) == (1, 1, 10), 'counters')
    require(r['physical_batch_240'] == 'OOM_RETAINED' and r['VAL_access'] is False, 'history / VAL')
    i = r['inputs']
    require(i['distinct_rows'] == 396 and i['x_spoof']['shape'] == i['x_live']['shape'] == [198, 3, 256, 256], 'inputs')
    require(r['epsilon']['shape'] == [198, 128], 'epsilon [198,128]')
    phases = [m['phase'] for m in r['memory_by_phase']]
    require(phases == ['after_model_construction', 'after_optimizer_and_input_allocation', 'after_epsilon_allocation',
                       'pass1_peak'] + ['pass2_chunk_%02d_peak' % n for n in range(10)] +
            ['before_optimizer_step', 'after_optimizer_step'], 'memory phases')
    require(r['overall_peak']['allocated_bytes'] == max(m['peak_allocated_bytes'] for m in r['memory_by_phase']),
            'overall peak')
    g = r['global_statistics']
    require(all(g['pass2_latents_bitwise_equal_pass1'].values()) and g['pass2_recomputed_sign_mismatches'] == 0 and
            g['pass2_recomputed_ort_sign_equal'], 'epsilon replay / signs')
    for k, v in (('loss_mmd_surrogate', 'loss_mmd_global_fp32'), ('loss_ort_surrogate', 'loss_ort_global_fp32')):
        require(abs(g['surrogate_sums_pass2'][k] - g[v]) <= 1e-4 * max(1.0, abs(g[v])), 'surrogate ' + k)
    L = r['losses']
    comp = L['global_components']
    require(all(v == v and abs(v) != float('inf') for v in comp.values()), 'finite global losses')
    require(comp['loss_cls'] == comp['loss_pair'] == 0.0, 'cls/pair zero')
    epoch1 = comp['loss_rec'] + 0.01 * (comp['loss_kl'] + comp['loss_mmd'] + comp['loss_ip'] + comp['loss_pair'] +
                                        comp['loss_cls'] + comp['loss_ort'])
    require(abs(epoch1 - L['epoch1_total_reconstructed']) <= 1e-9 * abs(epoch1), 'epoch-1 reconstruction')
    require(abs(L['sum_of_chunk_objectives'] - epoch1) <= 1e-4 * abs(epoch1), 'sum of chunk objectives')
    require([round(w * 198) for w in L['local_weights']] == TAIL_CHUNKS and abs(sum(L['local_weights']) - 1) < 1e-12,
            'local weights m/B')
    require(L['postwarmup_executed'] is False, 'post-warmup algebraic only')
    require([h['size'] for h in r['chunk_evidence']] == TAIL_CHUNKS, 'chunk evidence')
    for h in r['chunk_evidence']:
        require(not h['nir_fc_requires_grad'] and not h['vis_fc_requires_grad'], 'LightCNN target no_grad')
        require(h['rec_nir_fc_requires_grad'] and h['rec_vis_fc_requires_grad'], 'reconstruction features keep graph')
        require(h['grads']['rec_nir']['nonzero'] > 0 and h['grads']['rec_vis']['nonzero'] > 0, 'LightCNN gradient path')
        require(h['grads']['pre_spoof']['nonzero'] == 0 and h['grads']['pre_spoof']['shape'] == [h['size'], 1],
                'one-logit CE degeneracy')
    inv = r['gradient_inventory']
    for n in ('netE_nir', 'netE_vis', 'netG'):
        require(inv[n]['non_none'] == inv[n]['tensors'] == inv[n]['finite'], 'finite gradients ' + n)
    require(sum(inv[n]['non_none'] for n in ('netE_nir', 'netE_vis', 'netG')) == 55 and inv['netIP']['non_none'] == 0,
            '55/55 owned gradients; netIP none')
    require(r['zero_grad']['calls'] == B198_COUNTERS['zero_grad_calls'] and r['rng_no_redraw'], 'zero_grad / no redraw')
    s = r['optimizer_state']
    require(s['entries'] == 55 and s['only_owned'] and s['step_values'] == [1.0] and s['finite_moments'], 'Adam state')
    ch = r['parameter_change']
    require(ch['netCls']['changed_elements'] == ch['netIP']['changed_elements'] == 0, 'netCls/netIP unchanged')
    require(all(ch[n]['changed_tensors'] == ch[n]['tensors'] for n in ('netE_nir', 'netE_vis', 'netG')), 'owned changed')
    require(r['parameters_after_sha256']['netCls'] == r['parameters_initial']['netCls']['sha256'] and
            r['parameters_after_sha256']['netIP'] == r['parameters_initial']['netIP']['sha256'], 'netCls/netIP bytes')
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
    lock_raw = read('environments/e06c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock identity')
    lock = json.loads(lock_raw)
    train_generator, status = source_cache(lock)
    plan = dataloader_and_epoch(train_generator)
    overlay_raw = read(OVERLAY)
    overlay = json.loads(overlay_raw)
    require((overlay['execution_mode'], overlay['classification'], overlay['fidelity_class'],
             overlay['nominal_global_batch'], overlay['max_microbatch'], overlay['drop_last'],
             overlay['partial_global_batch_allowed'], overlay['final_batch_policy'], overlay['expected_train_rows'],
             overlay['full_batches_per_epoch'], overlay['final_batch_size'], overlay['optimizer_steps_per_epoch'],
             overlay['final_batch_chunks'], overlay['precision'], overlay['amp'], overlay['tf32']) ==
            (EXECUTION_MODE, 'CONTROLLED_EXECUTION_ADAPTATION', 'CONTROLLED_ADAPTATION', 240, 20, False, True,
             'EXACT_REMAINDER_NO_PADDING_NO_DUPLICATION', 8838, 36, 198, 37, TAIL_CHUNKS, 'FP32', False, False),
            'owner V2 overlay')
    for path, h in overlay['immutable_inputs_sha256'].items():
        require(sha(read(path)) == h, 'overlay immutable input ' + path)
    overlay_sha, v1_sha = sha(overlay_raw), sha(read(V1_OVERLAY))
    ref, compat = json.loads(read(REFERENCE)), json.loads(read(V1COMPAT))
    procs = [json.loads(read(p)) for p in PROCESSES]
    for r in (ref, compat, *procs):
        check_common(r, lock, overlay_sha, v1_sha)
    reference = check_reference(ref, overlay)
    v1c = check_v1compat(compat, overlay)
    b198 = [check_b198(r) for r in procs]
    repeat = {k: procs[0][k] == procs[1][k] for k in REPEAT_KEYS}
    require(all(repeat.values()), 'repeatability ' + json.dumps(repeat))
    require(len({r['gpu_before']['own_pid'] for r in (ref, compat, *procs)}) == 4, 'four fresh processes')
    audit = json.loads(read(BASE + '.json'))
    require(audit['final_status'] == FINAL_STATUS and audit['decision'] == 'PASS' and
            audit['classification'] == CLASSIFICATION, 'audit status')
    require(set(audit['artifacts_sha256']) == set(AUDIT_ARTIFACTS), 'audit artifact set')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    require(all(audit[k] == v for k, v in SCOPE.items()), 'audit scope flags')
    require(audit['qualification_counts'] == QUALIFICATION_COUNTS, 'audit qualification counts')
    report = read(BASE + '.md').decode()
    for word in FINAL_STATUS[1:-2] + ['CONTROLLED_ADAPTATION PRESERVED', 'E06c REMAINS IMPLEMENTED_NOT_EXECUTED',
                                      EXECUTION_MODE, 'NO VAL ACCESS', 'NO TEST ACCESS']:
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
    require(row['qualification_counts'] == QUALIFICATION_COUNTS, 'ledger qualification counts')
    require(set(row['artifacts_sha256']) == set(LEDGER_ARTIFACTS), 'ledger artifact set')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'validated_state': 'M6D5D_PASS', 'execution_mode': EXECUTION_MODE,
            'epoch_batching': plan, 'ledger_rows': LEDGER_PREFIX_ROWS + 1,
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'reference7': reference, 'v1compat': v1c, 'b198_processes': b198, 'repeatability': repeat,
            'qualification_counts': QUALIFICATION_COUNTS, 'physical_batch_240': 'OOM_RETAINED',
            'source_cache_status': status, 'torch_imported': False, 'model_execution_in_preflight': False,
            'benchmark_data_access': False, 'VAL_access': False, 'TEST_access': False}


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
