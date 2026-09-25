#!/usr/bin/env python3
"""Verify retained M6D5e / M6D5e-r E06c production-runner qualification evidence; rebuild the artifact index.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no image decode, no
parquet payload read, no training, and no read of any manifest (split_v1.parquet included).

Two runs are distinguished:
  INITIAL  M6D5e seed 60505 (M6D5E_E06C_*): retained historical evidence, byte-frozen here. Its
           development process parsed split_v1.parquet metadata (TRAIN+VAL+TEST rows) on the laptop:
           M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION. It omitted the upstream visualization block.
  CLEAN    M6D5e-r seed 60506 (M6D5E_R_E06C_*): fully validated. Executed code bytes = retained files;
           TRAIN-only benchmark access audit (0 VAL/TEST metadata or image accesses in every process);
           the real-TRAIN epoch (8838 rows, 36 x 240 + 198, 37 optimizer applications, 442 backward
           calls, 55/55 finite gradients); the upstream visualization block with its CPU-RNG replay
           proof; the qualification-only checkpoint event and resume sidecar; bitwise fresh-process
           resume (re-derived here independently).
Also: M6D5a-d immutability, the single corrected ledger append and the CRLF artifact index.
Historical index rows are carried from the authoritative commit unopened.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'cd125dbdd1b45f3963fc501c5f7dc6ff9091d536'
PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
PAIRS_SHA = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
CONFIG_SHA = '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002'
SEED, INITIAL_SEED = 60506, 60505
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'
CONTRACT = 'configs/amendments/e06c_m6d5e_production_runner_contract.yaml'
BASE = 'outputs/audit/M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION'
CLEAN_BASE = 'outputs/audit/M6D5E_R_E06C_CLEAN_REQUALIFICATION'
EVIDENCE = {'loader': 'outputs/audit/M6D5E_R_E06C_LOADER_DETERMINISM.json',
            'train': 'outputs/audit/M6D5E_R_E06C_REAL_TRAIN_EPOCH1.json',
            'reference': 'outputs/audit/M6D5E_R_E06C_RESUME_REFERENCE.json',
            'resume': 'outputs/audit/M6D5E_R_E06C_RESUME_FRESH_PROCESS.json'}
RUNTIME_LOG = 'outputs/audit/M6D5E_R_E06C_RUNTIME_LOG.txt'
# retained initial M6D5e evidence: byte-frozen (historical; never rewritten)
INITIAL_EVIDENCE_SHA = {
    'outputs/audit/M6D5E_E06C_LOADER_DETERMINISM.json': '43c446d4ba9915e748b427e69d2ce7ccd3a4d1fcc4001e30cb5ef5c564d94804',
    'outputs/audit/M6D5E_E06C_REAL_TRAIN_EPOCH1.json': '1bc20aa92526804e9026dcee9303e9ac1254cea8ecc87847bf361d9b5b7e26fa',
    'outputs/audit/M6D5E_E06C_RESUME_REFERENCE.json': '471a47c8c222d35117eb9e904d1791db8e887653c24d58041311e1a09d4a02b6',
    'outputs/audit/M6D5E_E06C_RESUME_FRESH_PROCESS.json': 'e4759499bfae59d7e7ba2275e121f9d93257ab4ed904c2505e57e9de4eedbdf1',
    'outputs/audit/M6D5E_E06C_RUNTIME_LOG.txt': '62a3108892275e9226a7f337862826e5d664a0e1164e643d7954699d38d24ca5'}
INITIAL_CODE_SHA = {   # code bytes the initial GPU processes recorded (superseded by the M6D5e-r files)
    'configs/amendments/e06c_m6d5e_production_runner_contract.yaml': 'e4420abbb7b1c00f2b097b21940cf2655c48aa97e2c47898544c9a854e8b8934',
    'methods/dsdg/runner_io.py': '36c16e020be74828fb2848c5900dac581627f1d921ded864aadb212ab89659bc',
    'methods/dsdg/runner.py': '60cb3cf45d8b22c44e9ec5c9fd488d84732bcfbf16642bd6da49d43e7a282ffe',
    'methods/dsdg/runner_qualification.py': '10b923f930dc24d34eb8416ca70af39b91d5ae7b7bdb7e81f20f9dd609aa9852',
    'tools/run_e06c.py': '9246c3b9c98913a0986b159a6480f51dcb44651a89478d744b6c01c658869f51'}
INITIAL_RUN_DIR_SUFFIX = '/qualification/m6d5e/E06c/q60505-834bd365c0e4e88a'
CODE = ('configs/amendments/e06c_m6d5e_production_runner_contract.yaml', 'methods/dsdg/runner_io.py',
        'methods/dsdg/runner.py', 'methods/dsdg/runner_qualification.py', 'tools/run_e06c.py')
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 114
LEDGER_PREFIX_SHA = 'c14c60fa253373994fa8b536f148e2a801c38062b1c99ff51bed17e032f3de10'
INDEX_BASELINE_ROWS = 617
CLASSIFICATION = 'M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION'
ARTIFACTS = (CODE + ('docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Production_Runner_Addendum_M6D5e.md',
                     'tests/test_m6d5e_e06c_runner.py', 'tools/m6d5e_e06c_runner_preflight.py')
             + tuple(INITIAL_EVIDENCE_SHA) + tuple(EVIDENCE.values()) + (RUNTIME_LOG, BASE + '.md', CLEAN_BASE + '.md'))
AUDIT_ARTIFACTS = ARTIFACTS
LEDGER_ARTIFACTS = ARTIFACTS + (BASE + '.json', CLEAN_BASE + '.json')
PRESERVED = (
    'configs/methods/e06c_dsdg_bin_idfree.yaml', 'configs/frozen/dsdg_bin_idfree_v1.yaml', 'configs/run_logging_v1.yaml',
    'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml',
    'configs/amendments/e06c_m6d5d_tail_batch_execution_resolution.yaml', 'configs/execution/m5_gpu_3090.yaml',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Memory_Execution_Resolution_Addendum_M6D5c.md',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Tail_Batch_Execution_Resolution_Addendum_M6D5d.md',
    'methods/dsdg/adapter.py', 'methods/dsdg/runtime.py', 'methods/dsdg/training_graph.py',
    'methods/dsdg/training_qualification.py', 'methods/dsdg/microbatch_execution.py',
    'methods/dsdg/microbatch_qualification.py', 'methods/dsdg/microbatch_execution_v2.py',
    'methods/dsdg/microbatch_qualification_v2.py', 'methods/dsdg/source.py', 'methods/dsdg/__init__.py',
    'methods/common/learned.py', 'methods/common/learned_runlog.py', 'methods/common/runlog.py',
    'methods/common/upstream.py', 'methods/common/config.py', 'methods/common/seeding.py',
    'environments/e06c.lock.json', 'environments/e06c.runtime.json', 'third_party/source_pins.json',
    'outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json', 'outputs/audit/pairs_train_v1.sha256',
    'tests/test_m6d5a_e06c_runtime.py', 'tests/test_m6d5b_e06c_training_graph.py',
    'tests/test_m6d5c_e06c_microbatch_execution.py', 'tests/test_m6d5d_e06c_tail_batch_execution.py',
    'tools/m6d5a_e06c_runtime_preflight.py', 'tools/m6d5b_e06c_training_preflight.py',
    'tools/m6d5c_e06c_memory_preflight.py', 'tools/m6d5d_e06c_tail_batch_preflight.py',
    'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.json',
    'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.md', 'outputs/audit/M6D5A_E06C_RUNTIME_LOG.txt',
    'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_1.json', 'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_2.json',
    'outputs/audit/M6D5B_E06C_RUNTIME_LOG.txt', 'outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_1.json',
    'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.json',
    'outputs/audit/M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.md',
    'outputs/audit/M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION.json',
    'outputs/audit/M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION.md', 'outputs/audit/M6D5C_E06C_PROCESS_1.json',
    'outputs/audit/M6D5C_E06C_PROCESS_2.json', 'outputs/audit/M6D5C_E06C_REFERENCE_CHECK.json',
    'outputs/audit/M6D5C_E06C_RUNTIME_LOG.txt', 'outputs/audit/M6D5D_E06C_B198_PROCESS_1.json',
    'outputs/audit/M6D5D_E06C_B198_PROCESS_2.json', 'outputs/audit/M6D5D_E06C_REFERENCE_CHECK.json',
    'outputs/audit/M6D5D_E06C_RUNTIME_LOG.txt', 'outputs/audit/M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION.json',
    'outputs/audit/M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION.md', 'outputs/audit/M6D5D_E06C_V1_COMPATIBILITY.json')
LABELS = ['QUALIFICATION_ONLY', 'NOT_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_SYNTHETIC_BANK',
          'NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION', 'NOT_ELIGIBLE_FOR_REPORTING']
FINAL_STATUS = ['M6D5e PASS_AFTER_CLEAN_REQUALIFICATION', 'E06c_PRODUCTION_RUNNER_QUALIFIED',
                'E06c_REAL_TRAIN_DATA_PATH_QUALIFIED', 'E06c_DETERMINISTIC_TRAIN_LOADER_QUALIFIED',
                'E06c_CHECKPOINT_WRITER_QUALIFIED', 'E06c_RESUME_PATH_QUALIFIED', 'E06c_TRAIN_ONLY_FIREWALL_QUALIFIED',
                'E06c_UPSTREAM_VISUALIZATION_RNG_SEMANTICS_QUALIFIED', 'E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED',
                'E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED', 'E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED',
                'E06c_SCIENTIFIC_FULL_TRAINING_NOT_YET_EXECUTED', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
INITIAL_STATUS = 'M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION'
SCOPE = {'initial_run_status': INITIAL_STATUS,
         'initial_development_val_metadata_access': True, 'initial_development_test_metadata_access': True,
         'initial_development_val_image_access': False, 'initial_development_test_image_access': False,
         'initial_val_or_test_used_for_optimization': False, 'initial_val_or_test_used_for_checkpoint_selection': False,
         'initial_test_metrics_computed': False,
         'clean_requalification_val_access': False, 'clean_requalification_test_access': False,
         'real_train_data_access': True, 'qualification_only_training': True, 'scientific_training': False,
         'clean_qualification_seed': SEED, 'clean_qualification_epoch_count': 1,
         'clean_qualification_train_rows': 8838, 'clean_qualification_optimizer_steps': 37,
         'clean_qualification_backward_calls': 442, 'clean_qualification_resume_reference_steps': 1,
         'clean_qualification_resume_fresh_steps': 1,
         'upstream_visualization_rng_semantics': 'UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED',
         'scientific_checkpoint_created': False, 'qualification_checkpoint_created': True, 'synthetic_bank': False,
         'downstream_evaluation': False, 'execution': EXECUTION_MODE, 'physical_batch_240': 'OOM_RETAINED'}
ZERO_ACCESS = ('VAL_metadata_accesses', 'TEST_metadata_accesses', 'VAL_image_accesses', 'TEST_image_accesses',
               'split_manifest_accesses', 'non_train_face_accesses', 'denied_events')
BITWISE_FIELDS = (('batch_pair_sha256',), ('batch_size',), ('chunk_sizes',), ('epsilon_sha256',), ('losses',),
                  ('total_loss',), ('owned_gradients',), ('netCls_grad_nonzero',), ('probe', 'pass1'),
                  ('probe', 'gradient_sha256'), ('probe', 'gradient_nonzero_tensors'), ('probe', 'optimizer_state'),
                  ('probe', 'parameters_before_sha256'), ('probe', 'parameters_after_sha256'), ('probe', 'rng_before'),
                  ('probe', 'rng_before_step'), ('probe', 'rng_after'), ('probe', 'surrogate_sums'),
                  ('probe', 'sum_of_chunk_objectives'), ('global_step',), ('epoch',), ('learning_rate',))




def require(ok, message):
    if not ok:
        raise ValueError('M6D5e: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'third_party', 'docs'},
            'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs', 'qualification'} & set(p.parts), 'data firewall')
    require(p.suffix not in ('.parquet', '.png', '.pth', '.pt'), 'no data/weight reads')
    return (ROOT / p).read_bytes()


def check_common(r, mode, lock):
    require(r['status'] == 'PASS' and r['mode'] == mode, mode + ' status')
    require(r['qualification_seed'] == SEED and r['experiment_seed'] is None and r['labels'] == LABELS, 'seed/labels')
    require(r['runner_mode'] == 'QUALIFICATION' and r['execution_mode'] == EXECUTION_MODE, 'mode')
    le = r['launch_environment']
    require((le['PYTHONHASHSEED'], le['NVIDIA_TF32_OVERRIDE'], le['CUBLAS_WORKSPACE_CONFIG'], le['CUDA_VISIBLE_DEVICES'],
             le['TMPDIR']) == (str(SEED), '0', ':4096:8', '0', '/tmp/gpat-m6d5er'), 'launch environment')
    for path in CODE:
        require(r['m6d5e_file_sha256'][path] == sha(read(path)), 'executed code bytes = retained ' + path)
    ids = r['identities']
    require((ids['config_sha256'], ids['environment_lock_sha256'], ids['train_relation_sha256'], ids['lightcnn_sha256'],
             ids['source_commit'], ids['source_tree'], ids['m6d5e_contract_sha256']) ==
            (CONFIG_SHA, LOCK_SHA, PAIRS_SHA, LIGHTCNN_SHA, PIN, TREE, sha(read(CONTRACT))), 'identities')
    tr = r['train_relation']
    require((tr['sha256'], tr['rows'], tr['splits'], tr['subject_columns_read'], tr['sha256_checked_before_parse']) ==
            (PAIRS_SHA, 8838, ['TRAIN'], [], True) and
            tr['rows_by_dataset'] == {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118} and
            tr['subject_columns_present_in_file'] == ['source_subject', 'target_subject'], 'TRAIN relation + firewall')
    s = r['storage']
    require(s['faces_256_root'] == s['runtime_root'] + '/data/processed/faces_256' and
            s['exec_config'] == 'configs/execution/m5_gpu_3090.yaml' and
            s['exec_config_sha256'] == sha(read(s['exec_config'])), 'canonical faces root')
    for k in ('source_before', 'source_after'):
        require((r[k]['commit'], r[k]['tree']) == (PIN, TREE) and r[k]['files_sha256'] == lock['source']['files_sha256'], k)
    fw = r['firewall']
    require(fw['denied'] == [] and set(fw['manifest_opens']) == {'manifests/pairs_train_v1.parquet'}, 'firewall')
    a = r['benchmark_access_audit']
    require(all(a[k] == 0 for k in ZERO_ACCESS) and a['train_faces_opened_all_in_relation'] and
            all(v == 0 for k, v in a['counts_by_category'].items() if k not in ('TRAIN_RELATION', 'TRAIN_FACE')) and
            a['counts_by_category']['TRAIN_RELATION'] == 2 and
            a['events_logged'] == sum(a['counts_by_category'].values()) and
            re.fullmatch(r'M6D5E_R_E06C_BENCHMARK_ACCESS_(loader|train|resume)\.tsv', a['log']) and
            re.fullmatch(r'[0-9a-f]{64}', a['log_sha256']), 'TRAIN-only benchmark access audit ' + mode)
    require(not any(r[k] for k in ('VAL_access', 'TEST_access', 'scientific_training', 'scientific_checkpoint_created',
                                   'synthetic_bank', 'downstream_evaluation')), 'scope flags')
    require(r['fidelity'] == 'CONTROLLED_ADAPTATION' and r['compatibility_patch'] == 'NONE', 'fidelity')
    c = r['counters']
    require(c['autograd_grad_calls'] == c['activation_checkpoint_calls'] == 0, 'forbidden calls')
    require(r['overlays']['v2']['execution_mode'] == EXECUTION_MODE and r['overlays']['v1']['physical_batch_240'] ==
            'OOM_RETAINED' and r['overlays']['epoch_plan']['optimizer_steps_per_epoch'] == 37, 'overlay binding')


def check_gpu(r, lock):
    require(r['resource_clean'] and r['gpu_before']['compute_processes'] == [], 'clean GPU')
    require(r['asset_before'] == r['asset_after'] and r['asset_before']['sha256'] == LIGHTCNN_SHA, 'LightCNN')
    require(r['environment_before'] == r['environment_after'] and
            all(r['environment_before'][k] == v for k, v in lock['identity'].items() if k != 'launch_environment'),
            'runtime identity equals environment lock')
    p = r['precision']
    require((p['default_dtype'], p['matmul_tf32'], p['cudnn_tf32'], p['autocast_cuda'], p['matmul_precision'], p['amp'],
             p['cudnn_benchmark'], p['cudnn_deterministic']) ==
            ('torch.float32', False, False, False, 'highest', False, False, True), 'FP32 / no TF32 / no AMP')
    require(r['lightcnn']['matched'] == 60 and not r['lightcnn']['missing'], 'LightCNN 60/60')
    require(r['parameter_counts'] == {'netCls': {'parameter_tensors': 2, 'parameters': 129},
                                      'netE_nir': {'parameter_tensors': 17, 'parameters': 13790048},
                                      'netE_vis': {'parameter_tensors': 17, 'parameters': 11692640},
                                      'netG': {'parameter_tensors': 21, 'parameters': 19887494},
                                      'netIP': {'parameter_tensors': 60, 'parameters': 10475872}}, 'parameter counts')
    o = r['optimizer']
    require((o['parameter_tensors'], o['parameters'], o['overlap_netCls_netIP'], o['param_groups']) ==
            (55, 45370182, 0, 1) and o['resolved_group_defaults']['lr'] == 2e-4, 'Adam ownership')
    d = r['dataloader']
    require((d['batch_size'], d['shuffle'], d['num_workers'], d['pin_memory'], d['drop_last'], d['len_batches'], d['rows'])
            == (240, True, 8, True, False, 37, 8838), 'DataLoader')


def check_visualization(t):
    v = t['visualization']
    ev = v['evidence']
    require(v['status'] == 'UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED' and
            v['executed'] == 'FULL_BLOCK_WITH_IMAGE_GRIDS' and all(v['proof'].values()) and
            set(v['proof']) == {'replay_noise_sha256_equal', 'replay_noise_s_sha256_equal',
                                'replay_end_state_equals_global_cpu_state', 'cpu_rng_advanced', 'cuda_rng_unchanged',
                                'parameters_unchanged', 'adam_state_unchanged'} and
            v['resume_sidecar_torch_cpu_rng_equals_post_visualization'] is True, 'visualization RNG proof')
    require((ev['epoch'], ev['global_step'], ev['last_batch_global_step'], ev['last_batch_size'], ev['noise_shape'],
             ev['noise_draw_order'], ev['not_validation'], ev['selects_nothing']) ==
            (1, 37, 37, 198, [240, 128], ['noise', 'noise_s'], True, True) and
            ev['torch_cpu_rng_before_sha256'] != ev['torch_cpu_rng_after_sha256'] and
            ev['data'].startswith('TRAIN'), 'visualization evidence')
    names = [f['path'] for f in ev['files']]
    require(names == [f'diagnostics/visualization/Epoch_001_{n}.png' for n in
                      ('img_spoof', 'img_live', 'rec_spoof', 'rec_live', 'fake_spoof', 'fake_live')] and
            [f['images'] for f in ev['files']] == [198] * 4 + [240, 240], 'visualization grids')
    for f in ev['files']:
        require(t['run_files'][f['path']] == {'bytes': f['bytes'], 'sha256': f['sha256']}, 'grid file ' + f['path'])
        require(not f['path'].startswith('checkpoints/'), 'diagnostics separate from checkpoints')
    rs = t['checkpoint_event']['resume_state']
    require(rs['torch_cpu_rng_sha256'] == ev['torch_cpu_rng_after_sha256'], 'sidecar CPU RNG = post-visualization')


def check_train(t):
    e = t['epoch1']
    require(e['batch_sizes'] == [240] * 36 + [198] and (e['rows'], e['unique_pair_ids'], e['optimizer_steps'],
                                                        e['global_step_end']) == (8838, 8838, 37, 37), 'epoch plan')
    require((t['optimizer_applications_epoch1'], t['backward_calls_epoch1']) == (37, 442), 'epoch-1 accounting')
    c = t['counters']
    require((c['backward_calls'], c['autograd_backward_calls'], c['zero_grad_calls'], c['zero_grad_set_to_none_values'])
            == (454, 454, 38, [False]), 'process accounting: 37 + 1 reference steps')
    losses = t['epoch1_losses']
    require([x['global_step'] for x in losses] == list(range(1, 38)) and
            [x['batch_size'] for x in losses] == [240] * 36 + [198], 'per-step records')
    for x in losses:
        vals = [x[k] for k in ('loss_rec', 'loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort',
                               'total_loss')]
        require(all(isinstance(v, float) and v == v and abs(v) != float('inf') for v in vals), 'finite losses')
        require(x['loss_cls'] == 0.0 and x['loss_pair'] == 0.0, 'cls/pair zero')
        warm = x['loss_rec'] + 0.01 * sum(x[k] for k in ('loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls',
                                                        'loss_ort'))
        require(abs(warm - x['total_loss']) <= 1e-9 * abs(warm), 'epoch-1 warmup assembly')
    ck = t['checkpoint_event']
    names = sorted(o['path'] for o in ck['official'])
    require(names == ['checkpoints/netE_live_model_epoch_1_iter_0.pth', 'checkpoints/netE_spoof_model_epoch_1_iter_0.pth',
                      'checkpoints/netG_model_epoch_1_iter_0.pth'], 'epoch-1 official checkpoint files')
    for o in ck['official']:
        require(o['sha256'] == o['sha256_recomputed'] and o['state_dict_equal_in_memory'] and o['keys'] ==
                ['epoch', 'model'] and o['checkpoint_type'] == 'periodic' and o['selected_for_final'] is False and
                o['selection_reason'].startswith('QUALIFICATION_ONLY') and (o['epoch'], o['global_step']) == (1, 37) and
                o['file_size_bytes'] == t['run_files'][o['path']]['bytes'] and
                o['sha256'] == t['run_files'][o['path']]['sha256'], 'official checkpoint ' + o['path'])
    rs = ck['resume_state']
    require(rs['sha256'] == rs['sha256_recomputed'] == t['run_files'][rs['path']]['sha256'] and
            all(rs['equal_in_memory'].values()) and (rs['completed_epoch'], rs['global_step']) == (1, 37) and
            rs['weights_only_load'] and rs['models'] == ['netCls', 'netE_nir', 'netE_vis', 'netG'] and
            rs['rng'] == ['numpy', 'python', 'torch_cpu', 'torch_cuda'] and 'loader_generator' in rs['inventory'] and
            rs['optimizer_state_entries'] == 55 and not rs['scientific_checkpoint'] and not rs['selection_candidate'],
            'resume sidecar')
    require(t['parameters_initial_sha256']['netIP'] == t['parameters_end_epoch1_sha256']['netIP'] and
            t['parameters_initial_sha256']['netCls'] == t['parameters_end_epoch1_sha256']['netCls'],
            'netIP frozen and netCls (optimizer-excluded) unchanged over the epoch')
    require(all(t['parameters_initial_sha256'][n] != t['parameters_end_epoch1_sha256'][n]
                for n in ('netE_nir', 'netE_vis', 'netG')), 'owned models trained')
    require(re.fullmatch(r'/.+/qualification/m6d5e/E06c/q60506-[0-9a-f]{16}', t['run_dir']) and '/runs/' not in t['run_dir']
            and not t['run_dir'].endswith(INITIAL_RUN_DIR_SUFFIX), 'new clean qualification root')
    a = t['benchmark_access_audit']
    require(a['train_faces_opened_distinct'] == t['train_relation']['unique_sample_ids'] == 12668 and
            a['counts_by_category']['TRAIN_FACE'] >= 2 * 8838, 'every TRAIN face of the epoch read, TRAIN only')
    check_visualization(t)
    require(t['warnings'] == [], 'no warnings')


def compare(ref, fresh):
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    fields = {'.'.join(p): get(ref['step'], p) == get(fresh['step'], p) for p in BITWISE_FIELDS}
    diff = fresh['difference_vs_reference_probe']
    for k in ('gradients', 'parameters_after'):
        fields[k + '_tensors_bitwise'] = (diff[k]['bitwise_equal_tensors'] == diff[k]['tensors'] == 55 and
                                          diff[k]['max_abs_diff'] == 0.0)
    return fields


def check_resume(t, ref, fresh, loader):
    require(ref['role'] == 'UNINTERRUPTED_IN_MEMORY_REFERENCE_CONTINUATION' and ref['steps_executed'] == 1 and
            ref['logged_to_metrics'] is False and ref['optimizer_applications_total'] == 38, 'reference role')
    require(ref['probe_file']['sha256'] == fresh['run_files'][ref['probe_file']['path']]['sha256'], 'probe file identity')
    require(fresh['run_dir'] == t['run_dir'] == ref['run_dir'], 'same qualification run')
    rs = fresh['resume']
    require(rs['entry']['path'] == 'checkpoints/runner_state_epoch_1.pth' and
            rs['entry']['sha256'] == t['checkpoint_event']['resume_state']['sha256'] and
            rs['metrics_agreement'] == {'agreement': True, 'last_epoch': 1, 'last_global_step': 37,
                                        'trajectory_records': 37}, 'explicit resume + metrics agreement')
    require(fresh['parameters_restored_sha256'] == t['parameters_end_epoch1_sha256'] and
            fresh['optimizer_state_restored'] == t['optimizer_state_end_epoch1'] and
            fresh['rng_restored'] == t['rng_end_epoch1'] and
            fresh['loader_generator_restored_sha256'] == t['loader_generator_end_epoch1_sha256'], 'restored state')
    fields = compare(ref, fresh)
    require(all(fields.values()), 'bitwise resume equivalence: ' + json.dumps(fields))
    require(fresh['comparison_vs_reference']['all_bitwise_equal'] and
            fresh['comparison_vs_reference']['fields'] == {k: v for k, v in fields.items() if not k.endswith('_bitwise')}
            | {'gradient_tensors_bitwise': True, 'parameter_tensors_bitwise': True}, 'harness comparison agrees')
    s = fresh['step']
    require((s['epoch'], s['global_step'], s['batch_size'], s['chunk_sizes']) == (2, 38, 240, [20] * 12), 'epoch-2 step 1')
    require(s['batch_pair_sha256'] == loader['epoch2_first_batch_pair_sha256'], 'next batch = ID-only loader epoch 2')
    require(all(v['step'] == 38.0 for v in s['probe']['optimizer_state'].values()) and
            len(s['probe']['optimizer_state']) == 55, 'Adam step 38 on 55 owned tensors')
    c = fresh['counters']
    require((c['backward_calls'], c['zero_grad_calls']) == (12, 1), 'fresh accounting: one step, 12 backward')
    require(fresh['metrics_append_only'] and fresh['metrics_records_after'].count('epoch') == 38, 'append-only metrics')
    return fields


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


def check_initial(lock):
    """Initial M6D5e run: retained byte-frozen; its historical facts only (not re-qualified here)."""
    for path, h in INITIAL_EVIDENCE_SHA.items():
        require(sha(read(path)) == h, 'initial evidence byte-frozen ' + path)
    ev = {k: json.loads(read(p)) for k, p in zip(('loader', 'train', 'reference', 'resume'), INITIAL_EVIDENCE_SHA)}
    for k in ('loader', 'train', 'resume'):
        r = ev[k]
        require(r['status'] == 'PASS' and r['qualification_seed'] == INITIAL_SEED and
                r['m6d5e_file_sha256'] == INITIAL_CODE_SHA, 'initial run facts ' + k)
    require(ev['train']['run_dir'].endswith(INITIAL_RUN_DIR_SUFFIX) and 'visualization' not in ev['train'],
            'initial run root; visualization block omitted there')
    require(ev['resume']['comparison_vs_reference']['all_bitwise_equal'], 'initial resume bitwise')
    log = read('outputs/audit/M6D5E_E06C_RUNTIME_LOG.txt').decode()
    require('DISCLOSED METADATA READ: manifests/split_v1.parquet' in log, 'initial access disclosure retained')
    return {'run_dir': ev['train']['run_dir'], 'epoch1_pair_order_sha256': ev['train']['epoch1']['epoch_pair_order_sha256']}


def check_reports():
    for base in (BASE, CLEAN_BASE):
        audit = json.loads(read(base + '.json'))
        require(audit['final_status'] == FINAL_STATUS and audit['decision'] == 'PASS_AFTER_CLEAN_REQUALIFICATION' and
                audit['classification'] == CLASSIFICATION, 'audit status ' + base)
        require(set(audit['artifacts_sha256']) == set(AUDIT_ARTIFACTS), 'audit artifact set ' + base)
        for path, h in audit['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'artifact ' + path)
        for path, h in audit['preserved_inputs_sha256'].items():
            require(sha(read(path)) == h, 'preserved input ' + path)
        require(all(audit[k] == v for k, v in SCOPE.items()), 'audit scope flags ' + base)
        require('VAL_access' not in audit and 'TEST_access' not in audit, 'no unqualified VAL/TEST access flag')
    clean = read(CLEAN_BASE + '.md').decode()
    main = read(BASE + '.md').decode()
    for word in FINAL_STATUS[:-2] + ['CONTROLLED_ADAPTATION PRESERVED', 'E06c REMAINS IMPLEMENTED_NOT_EXECUTED', *LABELS,
                                     INITIAL_STATUS, 'UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED',
                                     'REAL TRAIN DATA ACCESSED FOR QUALIFICATION', 'QUALIFICATION OPTIMIZER STEPS EXECUTED',
                                     'NO VAL ACCESS DURING CLEAN REQUALIFICATION',
                                     'NO TEST ACCESS DURING CLEAN REQUALIFICATION', 'NO SCIENTIFIC FULL TRAINING',
                                     'NO SCIENTIFIC CHECKPOINT', 'NO SYNTHETIC BANK', 'NO DOWNSTREAM EVALUATION',
                                     'NO VAL/TEST IMAGE ACCESS', 'NO VAL/TEST TRAINING USE', 'NO VAL/TEST SELECTION USE',
                                     'NO TEST METRIC USE']:
        require(word in clean, 'corrective report wording ' + word)
    for word in ('M6D5e PASS_AFTER_CLEAN_REQUALIFICATION', INITIAL_STATUS, 'M6D5E_R_E06C_CLEAN_REQUALIFICATION.md',
                 'NO VAL/TEST IMAGE ACCESS', 'NO VAL/TEST TRAINING USE', 'NO VAL/TEST SELECTION USE', 'NO TEST METRIC USE',
                 'DISCLOSED LAPTOP METADATA READ'):
        require(word in main, 'main report wording ' + word)
    for text in (clean, main):
        require('E06c_PHYSICAL_BATCH_240_QUALIFIED' not in text, 'no physical-B=240 qualification claim')
        require(not re.search(r'\bNO (VAL|TEST) ACCESS\b(?! DURING CLEAN REQUALIFICATION)', text),
                'no unqualified NO VAL/TEST ACCESS claim')


def verify(check_index=True):
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    lock = json.loads(read('environments/e06c.lock.json'))
    require(sha(read('environments/e06c.lock.json')) == LOCK_SHA, 'environment lock')
    contract = json.loads(read(CONTRACT))
    for rel, h in contract['bound_inputs_sha256'].items():
        if not rel.endswith('.parquet'):              # the TRAIN relation hash is verified by the GPU processes
            require(sha(read(rel)) == h, 'contract bound input ' + rel)
    require(contract['bound_inputs_sha256']['manifests/pairs_train_v1.parquet'] == PAIRS_SHA and
            json.loads(read('outputs/audit/pairs_train_v1.sha256'))['sha256'] == PAIRS_SHA, 'pair manifest identity')
    require((contract['fidelity_class'], contract['execution_mode'], contract['modes']['QUALIFICATION']['qualification_seed'],
             contract['modes']['SCIENTIFIC']['experiment_seeds'], contract['modes']['SCIENTIFIC']['launched_in_m6d5e']) ==
            ('CONTROLLED_ADAPTATION', EXECUTION_MODE, SEED, [42, 1337, 2026], False), 'contract')
    ev = {k: json.loads(read(p)) for k, p in EVIDENCE.items()}
    loader, t, ref, fresh = ev['loader'], ev['train'], ev['reference'], ev['resume']
    initial = check_initial(lock)
    require(contract['corrective_requalification']['initial_run_status'] == INITIAL_STATUS and
            contract['corrective_requalification']['clean_run']['qualification_seed'] == SEED and
            contract['execution']['official_visualization_block']['executed_by_runner'] is True, 'contract M6D5e-r')
    for r in (loader, t, fresh):
        check_common(r, r['mode'], lock)
    for r in (t, fresh):
        check_gpu(r, lock)
    require(loader['image_bytes_read'] == 0 and all(loader['repeat_equal'].values()) and loader['epochs_differ'],
            'ID-only loader determinism')
    for rep in ('A', 'B'):
        require(loader['constructions'][rep]['epoch1']['batch_sizes'] == [240] * 36 + [198], 'loader plan ' + rep)
    require(loader['epoch1_pair_order_sha256'] == t['epoch1']['epoch_pair_order_sha256'], 'loader = real epoch order')
    check_train(t)
    fields = check_resume(t, ref, fresh, loader)
    require(len({r['pid'] for r in (loader, t, fresh)}) == 3, 'three fresh processes')
    check_reports()
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix) and len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1, 'exactly one append')
    row = json.loads(current[len(prefix):])
    require(row['classification'] == CLASSIFICATION and row['final_status'] == FINAL_STATUS and
            row['decision'] == 'PASS_AFTER_CLEAN_REQUALIFICATION', 'ledger row')
    require('VAL_access' not in row and 'TEST_access' not in row, 'no unqualified VAL/TEST access flag in the ledger')
    require(all(row[k] == v for k, v in SCOPE.items()), 'ledger scope flags')
    require(set(row['artifacts_sha256']) == set(LEDGER_ARTIFACTS), 'ledger artifact set')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if check_index:
        require(read(INDEX) == expected, 'CRLF artifact index')
    require('torch' not in sys.modules and 'pyarrow' not in sys.modules and 'cv2' not in sys.modules,
            'static preflight must not import torch/pyarrow/cv2')
    return {'status': 'PASS', 'validated_state': 'M6D5E_PASS_AFTER_CLEAN_REQUALIFICATION', 'execution_mode': EXECUTION_MODE,
            'initial_run': {'status': INITIAL_STATUS, 'run_dir': initial['run_dir'], 'evidence_byte_frozen': True},
            'clean_qualification_run_dir': t['run_dir'],
            'clean_epoch1_pair_order_sha256': t['epoch1']['epoch_pair_order_sha256'],
            'clean_benchmark_access': {m: {k: ev[m]['benchmark_access_audit'][k] for k in ZERO_ACCESS}
                                       for m in ('loader', 'train', 'resume')},
            'visualization': t['visualization']['status'],
            'resume_bitwise_fields': len(fields), 'ledger_rows': LEDGER_PREFIX_ROWS + 1,
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows': count,
            'torch_imported': False, 'parquet_parsed': False, 'images_decoded': False, 'manifests_read': False,
            'model_execution_in_preflight': False}


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
