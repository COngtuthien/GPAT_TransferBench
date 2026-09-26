#!/usr/bin/env python3
"""Verify M6D6e E07c auxiliary production-runner evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch, no YAML parser, no CUDA, no model, no checkpoint I/O, no Parquet decode and no image
read. The split manifest and class map are NOT opened: their identity is checked through Git blob
ids against the authority commit. The engine is checked by stdlib AST; the recorded process evidence
is re-derived (bitwise comparison, epoch accounting, source-native epoch loss) without re-running
anything. Historical index rows are carried from the authority commit without opening their targets
(tools/build_artifact_index.py is NOT used, because it hashes every file); only the explicit new M6D6e
artifacts are hashed.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '9bfb8dc0fc1e1a4be7ea7f1d4c788b6d2dbbd00f'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
SOURCE_FILES = {'models/pretrain_classifier.py': '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16',
                'models/custom_rn.py': 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'}
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
A7_OVERLAY_SHA = '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3'
DATA_INPUTS = ('manifests/split_v1.parquet', 'manifests/artifact_probe_classes_v1.json',
               'manifests/difffas_bin_idfree_train_v1.parquet')
QUALIFICATION_SEED = 60605
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 119
INDEX_BASELINE_ROWS = 685
CLASSIFICATION = 'M6D6E_E07C_AUX_PRODUCTION_RUNNER_QUALIFICATION'
BASE = 'outputs/audit/M6D6E_E07C_AUX_RUNNER_QUALIFICATION'
PROCESSES = {'b1': 'outputs/audit/M6D6E_E07C_AUX_B256_PROCESS_1.json',
             'b2': 'outputs/audit/M6D6E_E07C_AUX_B256_PROCESS_2.json',
             'epoch': 'outputs/audit/M6D6E_E07C_AUX_ONE_EPOCH.json'}
PROCESS_SHA = {'b1': 'a653055a241c6025e5955f2390e6cfde956580a9001e26a70ee24a62cfa166d5',
               'b2': '391406984821818fac08597d89ee634c49bd97a965dde6929385c93cf6093b9f',
               'epoch': 'fb7adaa5bebf2d62828dbb95a2af375dcf0de43cb1c649a02809f299e023d7a6'}
LOG = 'outputs/audit/M6D6E_E07C_AUX_RUNTIME_LOG.txt'
CONTRACT = 'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml'
ADDENDUM = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Aux_Production_Runner_Addendum_M6D6e.md'
IO_MOD, ENGINE, HARNESS = ('methods/difffas/aux_runner_io.py', 'methods/difffas/aux_runner.py',
                           'methods/difffas/aux_runner_qualification.py')
CLI, TESTS, PREFLIGHT = 'tools/run_e07c_aux.py', 'tests/test_m6d6e_e07c_aux_runner.py', \
    'tools/m6d6e_e07c_aux_runner_preflight.py'
EXECUTED_CODE = {IO_MOD: 'a8a28439c0ff88de5cd2cb2d6da32424dc723aac44908fe95cc2d87b9a342943',
                 ENGINE: '49635cb83bd3fb264f295d7444b6080f992110d4cbf16bc244b38e95496e0eb5',
                 HARNESS: 'a09749a11172ce8310700653605d5402efbe28346c0449280e52a61819d7d814',
                 CLI: 'd4aaf2785c2a48bf16c2fefaf7768be0d540f6ad9c600b76e11cf2c6c81bce61',
                 CONTRACT: '09c0af7e79aeb3acda3e188f7a2a9d94f6d0401485a061d46767ab1c71069ae6'}
ARTIFACTS = tuple(sorted((CONTRACT, ADDENDUM, IO_MOD, ENGINE, HARNESS, CLI, TESTS, PREFLIGHT, *PROCESSES.values(),
                          LOG, BASE + '.md', BASE + '.json')))
PRESERVED = ('configs/methods/e07c_difffas_bin_idfree.yaml', 'configs/frozen/difffas_bin_idfree_v1.yaml',
             'frozen_config_snapshot/configs/methods/e07c_difffas_bin_idfree.yaml',
             'frozen_config_snapshot/configs/frozen/difffas_bin_idfree_v1.yaml',
             'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
             'configs/amendments/e07c_a7_execution_policy.yaml',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md',
             'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
             'configs/run_logging_v1.yaml', 'configs/execution/m5_gpu_3090.yaml', 'configs/CONFIG_STATUS.md',
             'outputs/audit/STAGE_STATE.json', 'third_party/source_pins.json',
             'methods/difffas/__init__.py', 'methods/difffas/adapter.py', 'methods/difffas/contract.py',
             'methods/difffas/encoder.py', 'methods/difffas/source.py', 'methods/difffas/sampler.py',
             'methods/difffas/seed_adapter.py', 'methods/difffas/runtime_qualification.py',
             'methods/difffas/aux_training_qualification.py', 'methods/difffas/aux_checkpoint.py',
             'methods/difffas/aux_checkpoint_qualification.py', 'methods/difffas/execution_policy.py',
             'methods/difffas/execution_policy_qualification.py',
             'methods/common/learned.py', 'methods/common/learned_runlog.py', 'methods/common/runlog.py',
             'methods/common/upstream.py', 'methods/common/config.py', 'gpatbench/preprocess/m2b.py',
             'gpatbench/probe/data.py', 'environments/e07c.lock.json', 'environments/e07c.runtime.json',
             'environments/e07c.conda-explicit.txt', 'environments/e07c.pip-freeze.txt',
             'environments/e07c.pip-requirements.txt',
             'tests/test_m6d6d_e07c_execution_policy.py', 'tests/test_m6d6c_e07c_aux_checkpoint.py',
             'tests/test_m6d6b_e07c_aux_training_graph.py', 'tests/test_m6d6a_e07c_runtime.py',
             'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py', 'tests/test_m6c2b3_runtime.py',
             'tools/build_artifact_index.py')
QUALIFIED = ['E07c_AUX_REAL_TRAIN_PATH_QUALIFIED', 'E07c_AUX_B256_TRAINING_MEMORY_QUALIFIED',
             'E07c_AUX_PRODUCTION_RUNNER_QUALIFIED', 'E07c_AUX_EPOCH_CHECKPOINT_INTEGRATION_QUALIFIED',
             'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
RETAINED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
            'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
            'E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED', 'E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED',
            'E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED',
            'E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED', 'E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED',
            'E07c_PRODUCTION_PRECISION_POLICY_FROZEN', 'E07c_MAIN_ENCODER_LOADER_RNG_POLICY_FROZEN',
            'E07c_THROWAWAY_ENCODER_RNG_COMPATIBILITY_QUALIFIED', 'E07c_EXECUTION_POLICY_RUNTIME_QUALIFIED']
NOT_QUALIFIED = ['AUXILIARY_ENCODER_200_EPOCH_TRAINING', 'AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT',
                 'AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', 'AUX_RESUME', 'MAIN_DIFFFAS_TRAINING_GRAPH',
                 'MAIN_CHECKPOINT_RESUME', 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION', 'MAIN_PRODUCTION_RUNNER',
                 'SCIENTIFIC_TRAINING', 'M8_BANK']
CLASS_COUNTS = {'live': 5629, 'makeup': 759, 'mask_2d': 96, 'mask_3d': 1056, 'partial': 1911, 'print': 2838,
                'replay': 2178}
EXPECTED_PRECISION = {'default_dtype': 'torch.float32', 'matmul_tf32': False, 'cudnn_tf32': False,
                      'cudnn_benchmark': False, 'cudnn_deterministic': True, 'float32_matmul_precision': 'highest',
                      'autocast_cuda': False, 'autocast_cpu': False, 'deterministic_algorithms_forced': False}
BITWISE_FIELDS = (('rng_after_seeding',), ('initial_state',), ('rng_before_iterator',), ('first_batch', 'indices_sha256'),
                  ('first_batch', 'sample_ids_sha256'), ('first_batch', 'class_histogram'), ('first_batch', 'input'),
                  ('first_batch', 'target'), ('step', 'loss_hex'), ('step', 'batch_sample_sha256'),
                  ('step', 'gradient_aggregate_sha256'), ('step', 'parameters_after_aggregate_sha256'),
                  ('step', 'buffers_after_aggregate_sha256'), ('step', 'momentum_aggregate_sha256'),
                  ('rng_after_step',))
ZERO_COUNTERS = ('other_optimizer_step_calls', 'scheduler_constructions', 'grad_clipping_calls', 'autograd_grad_calls',
                 'activation_checkpoint_calls', 'autocast_entries', 'grad_scaler_constructions')
SESSION_ZERO = ('VAL_rows_exposed', 'TEST_rows_exposed', 'VAL_image_reads', 'TEST_image_reads',
                'raw_benchmark_image_reads', 'unauthorized_manifest_reads', 'benchmark_path_firewall_denials',
                'scientific_auxiliary_runs', 'scientific_main_runs', 'scientific_checkpoints_created',
                'scientific_run_root_writes', 'main_difffas_training_runs', 'synthetic_bank_outputs')
FORBIDDEN_WORDING = ('faithful difffas', 'native difffas', 'official reproduction of', 'scientifically trained auxiliary',
                     'auxiliary encoder is trained', '200_epoch_scientific_training qualified',
                     'aux_resume_qualified', 'main_production_runner_qualified', 'm8_ready',
                     'benchmark_data_access: false', 'zero benchmark access')
REQUIRED_WORDING = ('controlled_adaptation', 'implemented_not_executed', 'one_real_train_epoch_qualification',
                    'not 200_epoch_scientific_training', 'authorized real train access', '14467', '14336', '131',
                    'not** used as the denominator', 'no fallback', 'e07c_aux_b256_training_memory_qualified',
                    'not** been scientifically trained', 'superseded')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6e: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'docs',
                           'frozen_config_snapshot', 'third_party', 'gpatbench'}, 'evidence root ' + path)
    require(not {'data', 'manifests', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall')
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image read')
    return (ROOT / p).read_bytes()


def data_inputs_unchanged():
    """Git blob identity of the data inputs against the authority commit; the files themselves are not opened."""
    status = git('status', '--porcelain', '--', *DATA_INPUTS).decode().strip()
    require(status == '', 'data inputs have no worktree modification')
    for rel in DATA_INPUTS:
        head = git('rev-parse', f'HEAD:{rel}').decode().strip()
        auth = git('rev-parse', f'{AUTHORITY}:{rel}').decode().strip()
        require(head == auth, 'data input blob changed ' + rel)
    return True


def source_cache():
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip() == '', 'pinned source clean')
    for rel, digest in SOURCE_FILES.items():
        require(sha((root / rel).read_bytes()) == digest, rel + ' SHA256')
    tree = ast.parse((root / 'models/pretrain_classifier.py').read_bytes())
    loader = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'train_loader')
    require({k.arg: ast.literal_eval(k.value) for k in loader.value.keywords} ==
            {'batch_size': 256, 'shuffle': True, 'num_workers': 6, 'drop_last': True}, 'pinned loader')
    body = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    require('rus = running_loss / len(train_dataset)' in body, 'pinned epoch-loss denominator = len(train_dataset)')
    return True


def engine_ast():
    """Stdlib AST of the executed engine: source-verbatim components, step order, denominator, save seam."""
    tree = ast.parse(read(ENGINE))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    src = {k: ast.unparse(v) for k, v in fns.items()}
    loader = next(n for n in ast.walk(fns['build_loader']) if isinstance(n, ast.Call) and
                  ast.unparse(n.func) == 'torch.utils.data.DataLoader')
    require(sorted(k.arg for k in loader.keywords) == ['batch_size', 'drop_last', 'num_workers', 'shuffle'],
            'DataLoader keywords exactly the pinned four')
    sgd = next(n for n in ast.walk(fns['build_optimizer']) if isinstance(n, ast.Call) and
               ast.unparse(n.func) == 'torch.optim.SGD')
    require(sorted(k.arg for k in sgd.keywords) == ['lr', 'momentum', 'weight_decay'], 'SGD keywords')
    require('transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])'
            in src['build_transform'], 'verbatim transform')
    step = src['step']
    order = ['self.optimizer.zero_grad()', '_, _, _, outputs = self.model(inputs)', 'loss = self.criterion(outputs, labels)',
             'loss.backward()', 'self.optimizer.step()', 'value = loss.item()']
    pos = [step.index(s) for s in order]
    require(pos == sorted(pos), 'step order')
    require('rus = running_loss / len(self.dataset)' in src['run_epoch'], 'source denominator')
    require('save_whole_module(self.model, partial, config)' in src['checkpoint_event'], 'M6D6c whole-module seam')
    require("ep.apply_e07c_precision_policy(config)" in src['configure_precision'], 'A7 mandatory')
    text = read(ENGINE).decode() + read(IO_MOD).decode()
    for token in ('autocast(', 'GradScaler(', 'WeightedRandomSampler', 'RandomHorizontalFlip', 'lr_scheduler',
                  'load_state_dict', 'use_deterministic_algorithms('):
        require(token not in text, 'forbidden token ' + token)
    require(text.count("'manifests/difffas_bin_idfree_train_v1.parquet'") == 1, 'main relation named once, never read')
    return True


def compare(reference, fresh):
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    rows = {'.'.join(p): get(reference, p) == get(fresh, p) for p in BITWISE_FIELDS}
    d = fresh['difference_vs_process_1']
    rows['parameter_tensors_bitwise'] = d['parameters_after']['bitwise_equal_tensors'] == d['parameters_after']['tensors'] == 112
    rows['gradient_tensors_bitwise'] = d['gradients']['bitwise_equal_tensors'] == d['gradients']['tensors'] == 110
    return rows


def check_common(r, lock):
    require(r['status'] == 'PASS' and r['milestone'] == 'M6D6e' and r['label'] == 'QUALIFICATION_ONLY', 'process pass')
    require((r['qualification_seed'], r['experiment_seed'], r['auxiliary_encoder_training_seed'],
             r['scientific_seed_consumed']) == (QUALIFICATION_SEED, None, None, False), 'seed')
    s = r['seeding']
    require((s['seed'], s['role'], s['pythonhashseed']) == (60605, 'QUALIFICATION_SEED_NOT_AUXILIARY_SEED', '60605'),
            'qualification seeding')
    require(r['m6d6e_file_sha256'] == EXECUTED_CODE, 'executed code bytes = the files in this worktree')
    src = r['source_before']
    require(src == r['source_after'] and src['worktree_status'] == '' and (src['commit'], src['tree']) == (PIN, TREE) and
            all(src['files_sha256'][k] == v for k, v in SOURCE_FILES.items()), 'pinned source')
    env = r['environment_before']
    require(env == r['environment_after'] and
            {k: v for k, v in env.items() if k != 'launch_environment'} ==
            {k: v for k, v in lock['identity'].items() if k != 'launch_environment'}, 'environment = lock')
    le = r['launch_environment']
    require((le['NVIDIA_TF32_OVERRIDE'], le['CUBLAS_WORKSPACE_CONFIG'], le['PYTHONHASHSEED'], le['TMPDIR']) ==
            ('0', ':4096:8', '60605', '/tmp/gpat-m6d6e'), 'launch environment')
    p = {k: v for k, v in r['precision'].items() if k in EXPECTED_PRECISION}
    require(p == EXPECTED_PRECISION and r['precision_state_after'] == EXPECTED_PRECISION and
            r['precision']['grad_scaler'] is False and r['precision']['amp'] is False, 'A7 precision')
    require(r['a7']['overlay_sha256'] == A7_OVERLAY_SHA, 'A7 identity')
    pop = r['population']
    require((pop['train_rows_materialized'], pop['val_rows_materialized'], pop['test_rows_materialized'],
             pop['other_spoof_rows'], pop['class_counts'], pop['file_rows_total_from_footer'], pop['row_groups'],
             pop['main_relation_opened']) == (14467, 0, 0, 0, CLASS_COUNTS, 20615, 1, False), 'population')
    require(pop['columns_read'] == ['sample_id', 'dataset', 'attack_macro', 'split', 'm2_status'] and
            'subject_id_global' in pop['columns_never_read'], 'column allowlist')
    require([c['api'] for c in r['pyarrow_calls']] == ['pyarrow.parquet.read_metadata', 'pyarrow.parquet.read_schema',
                                                       'pyarrow.parquet.read_table'] and
            {c['path'] for c in r['pyarrow_calls']} == {'manifests/split_v1.parquet'}, 'pyarrow calls')
    u = r['upstream_training_semantics']
    require(u['sha256'] == SOURCE_FILES['models/pretrain_classifier.py'] and u['epochs'] == 200 and
            u['optimizer']['hyperparameters'] == {'lr': 0.002, 'momentum': 0.9, 'weight_decay': 5e-3}, 'upstream semantics')
    c = r['components']
    require(c['loader']['sampler'] == 'RandomSampler(replacement=False, generator=None)' and
            (c['loader']['batch_size'], c['loader']['num_workers'], c['loader']['drop_last'], c['loader']['len_batches'],
             c['loader']['generator'], c['loader']['worker_init_fn'], c['loader']['pin_memory']) ==
            (256, 6, True, 56, None, None, False), 'loader')
    require((c['optimizer']['lr'], c['optimizer']['momentum'], c['optimizer']['weight_decay'], c['optimizer']['scheduler']) ==
            (0.002, 0.9, 0.005, 'NONE') and c['criterion']['class'] == 'torch.nn.CrossEntropyLoss', 'SGD / CE')
    require((c['model']['class'], c['model']['fc'], c['model']['disconnected_parameters']) ==
            ('custom_rn.ResNet', [512, 7], ['norm.bias', 'norm.weight']), 'model')
    require(c['transform']['augmentation'] == 'NONE' and c['transform']['random_transforms'] == 0, 'transform')
    require(all(r['counters'][k] == 0 for k in ZERO_COUNTERS) and r['counters']['zero_grad_set_to_none_values'] == [True],
            'no forbidden call')
    fw = r['firewall']
    require(fw['denied'] == [] and all(a[0] in ('git', 'nvidia-smi') or a in (['/sbin/ldconfig', '-p'], ['uname', '-p'])
                                       for a in fw['subprocesses']), 'firewall / subprocesses')
    a = r['benchmark_access_audit']
    require(a['denied_events'] == 0 and a['train_faces_all_in_population'] and
            all(v == 0 for k, v in a['counts_by_category'].items() if k not in ('SPLIT_MANIFEST', 'CLASS_MAP', 'TRAIN_FACE')) and
            (a['split_manifest_python_opens'], a['class_map_python_opens']) == (2, 2) and
            a['VAL_image_reads'] == a['TEST_image_reads'] == 0, 'access audit')
    s = r['access_summary']
    require((s['TRAIN_rows_exposed_to_dataset'], s['VAL_rows_exposed_to_dataset'], s['TEST_rows_exposed_to_dataset'],
             s['VAL_image_reads'], s['TEST_image_reads'], s['raw_dataset_image_reads'], s['unauthorized_manifest_reads'],
             s['firewall_denials']) == (14467, 0, 0, 0, 0, 0, 0, 0), 'access summary')
    require((r['VAL_access'], r['TEST_access'], r['authorized_TRAIN_access'], r['scientific_training'],
             r['scientific_checkpoint_created'], r['scientific_run_root_written'], r['synthetic_bank'],
             r['main_difffas_training']) == (False, False, True, False, False, False, False, False), 'scope')
    return a


def check_processes(lock):
    raw = {k: read(p) for k, p in PROCESSES.items()}
    require({k: sha(v) for k, v in raw.items()} == PROCESS_SHA, 'process evidence digests')
    ev = {k: json.loads(v) for k, v in raw.items()}
    audits = {k: check_common(ev[k], lock) for k in ev}
    for k in ('b1', 'b2'):
        r = ev[k]
        m = r['memory']
        require(set(m) == {'after_model_construction', 'after_transfer', 'after_forward', 'after_backward',
                           'after_optimizer_step', 'peak_during_step'} and 'oom_stage' not in r, 'B256 stages, no OOM')
        require(m['peak_during_step']['max_reserved_bytes'] < r['cuda_mem_get_info_before']['total_bytes'], 'fits device')
        fb, st = r['first_batch'], r['step']
        require(fb['input']['shape'] == [256, 3, 256, 256] and fb['input']['dtype'] == 'torch.float32' and
                fb['input']['within_minus1_plus1'] and fb['target']['shape'] == [256] and
                fb['target']['dtype'] == 'torch.int64' and sum(fb['class_histogram']) == 256 and
                len(fb['sample_ids']) == len(set(fb['sample_ids'])) == 256, 'first batch')
        require(st['batch_size'] == 256 and st['gradients_all_finite'] and st['parameter_tensors_changed'] == 110 and
                st['parameter_tensors_unchanged'] == ['norm.bias', 'norm.weight'] and st['optimizer_applications'] == 1 and
                st['backward_calls'] == 1 and float.fromhex(st['loss_hex']) == st['loss'], 'B256 step')
        require(r['counters']['backward_calls'] == r['counters']['sgd_step_calls'] == r['counters']['zero_grad_calls'] == 1,
                'one step')
        require(r['transform_check']['resize_identity_bytes'] and r['transform_check']['loader_row_equals_transform'],
                'transform check')
        require(audits[k]['train_face_opens'] == 3329, 'B256 face reads (13 prefetched batches + 1 check)')
    rows = compare(ev['b1'], ev['b2'])
    require(all(rows.values()) and ev['b2']['comparison_vs_process_1']['all_bitwise_equal'] and
            ev['b2']['difference_vs_process_1']['parameters_after']['max_abs_diff'] == 0.0 and ev['b2']['probe_removed'],
            'fresh-process repeatability bitwise: ' + json.dumps({k: v for k, v in rows.items() if not v}))
    e = ev['epoch']
    s = e['epoch1']
    require((s['optimizer_steps'], s['batch_sizes_distinct'], s['consumed_examples'], s['unique_consumed'],
             s['dropped_examples'], s['global_step_end'], s['epoch_loss_denominator']) ==
            (56, [256], 14336, 14336, 131, 56, 14467), 'epoch accounting')
    require(sum(s['dropped_class_histogram']) == 131, 'dropped histogram')
    losses = e['epoch1_losses']
    require(len(losses) == 56 and all(float.fromhex(h) == v for h, v in zip(e['epoch1_losses_hex'], losses)), 'losses')
    running = 0
    for v in losses:
        running += v * 256
    require(running == s['running_loss'] and running / 14467 == s['source_epoch_loss'] and
            running / 14336 != s['source_epoch_loss'] and float(s['source_epoch_loss']).hex() == s['source_epoch_loss_hex'],
            'source-native epoch loss / 14467')
    c = e['counters']
    require((c['backward_calls'], c['autograd_backward_calls'], c['sgd_step_calls'], c['zero_grad_calls'],
             c['torch_save_calls'], c['torch_load_calls'], e['optimizer_applications'], e['backward_calls']) ==
            (56, 56, 56, 56, 1, 1, 56, 56), 'epoch counters')
    fs = e['first_step']
    require((fs['loss_hex'], fs['batch_sample_sha256'], fs['parameters_after_aggregate_sha256']) ==
            (ev['b1']['step']['loss_hex'], ev['b1']['step']['batch_sample_sha256'],
             ev['b1']['step']['parameters_after_aggregate_sha256']), 'epoch first step = B256 step')
    ck = e['checkpoint_event']
    require((ck['path'], ck['epoch'], ck['global_step'], ck['checkpoint_type'], ck['selected_for_final'],
             ck['serialization_api'], ck['identity']['fc']) ==
            ('checkpoints/encoder_final.pkl', 1, 56, 'periodic', False, 'torch.save(model, path)',
             {'in_features': 512, 'out_features': 7, 'bias': True}), 'epoch checkpoint')
    require(e['checkpoint_file_before_cleanup'] == {'bytes': ck['file_size_bytes'], 'sha256': ck['sha256']} and
            e['checkpoint_reload']['state_dict_equal_in_memory'] and e['checkpoint_reload']['sha256_verified_first'] and
            e['checkpoint_outside_runs'] and e['partial_absent_after_replace'] and
            '/qualification/m6d6e/E07c_aux/q60605-' in e['checkpoint_path'], 'checkpoint reload / location')
    require(e['checkpoint_cleanup'] == {'removed': True, 'checkpoints_dir_listing': [], 'weight_files_in_run_dir': []},
            'qualification checkpoint cleanup')
    require(e['metrics_records'].count('epoch') == 56 and e['logged_epoch_summary_equal'], 'run_logging_v1 trajectory')
    require(e['run_files']['generation_log.jsonl']['bytes'] == 0 and
            {'resolved_config.yaml', 'run_manifest.json', 'metrics.jsonl', 'checkpoint_index.json', 'run_summary.json',
             'stdout.log', 'stderr.log'} <= set(e['run_files']), 'run files')
    a = audits['epoch']
    require((a['train_face_opens'], a['train_faces_distinct']) == (14336, 14336) and
            a['train_faces_opened_equal_consumed_set'], 'epoch face reads = consumed set')
    return ev


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


def worktree(ledger_and_index_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in ARTIFACTS]
    if ledger_and_index_modified:
        expected += [' M ' + INDEX, ' M ' + LEDGER]
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6e changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    return status


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(git('branch', '--show-current').decode().strip() == 'm6-baselines', 'branch')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    data_inputs_unchanged()
    source_cache()
    for path, digest in EXECUTED_CODE.items():
        require(sha(read(path)) == digest, 'executed code unchanged since the GPU runs: ' + path)
    contract = json.loads(read(CONTRACT))
    require((contract['authority_commit'], contract['method_id'], contract['fidelity_class'], contract['deviation'],
             contract['new_deviation'], contract['owner_decision_made'], contract['new_scientific_amendment']) ==
            (AUTHORITY, 'E07c', 'CONTROLLED_ADAPTATION', 'DEV-021', False, False, False), 'contract identity')
    for rel, digest in contract['bound_inputs_sha256'].items():
        if rel.startswith('manifests/'):
            continue                                 # data files: identity via Git blob above, never opened here
        require(sha(read(rel)) == digest, 'contract bound input ' + rel)
    require(contract['bound_inputs_sha256']['environments/e07c.lock.json'] == LOCK_SHA and
            contract['bound_inputs_sha256']['configs/amendments/e07c_a7_execution_policy.yaml'] == A7_OVERLAY_SHA,
            'contract binds lock / A7')
    require((contract['train_population']['rows'], contract['train_population']['class_counts'],
             contract['loader']['batch_size'], contract['loader']['drop_last'], contract['epoch_loss']['denominator'],
             contract['resume']['status']) == (14467, CLASS_COUNTS, 256, True, 14467, 'AUX_RESUME_NOT_QUALIFIED'),
            'contract values')
    engine_ast()
    lock_raw = read('environments/e07c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock digest')
    check_processes(json.loads(lock_raw))
    log = read(LOG).decode()
    after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
    require(len(after) == 3 and set(after) == {'scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT '
                                               'encoder_final=ABSENT'}, 'no scientific root in any authoritative process')
    for token in ('RUNS_ABSENT', 'PROTECTED_UNCHANGED', 'ANCESTOR_OK', 'SUPERSEDED', 'ABORTED BEFORE ANY STEP',
                  "['uname', '-p']", 'never combined', *PROCESS_SHA.values()):
        require(token in log, 'runtime log: ' + token)
    report = read(BASE + '.md').decode().lower()
    for phrase in FORBIDDEN_WORDING:
        require(phrase not in report, 'forbidden wording: ' + phrase)
    for phrase in REQUIRED_WORDING:
        require(phrase in report, 'required wording: ' + phrase)
    audit = json.loads(read(BASE + '.json'))
    require((audit['classification'], audit['milestone'], audit['method_id'], audit['final_status'],
             audit['method_status'], audit['fidelity_class'], audit['deviation'], audit['new_deviation'],
             audit['qualification_scope']) ==
            (CLASSIFICATION, 'M6D6e', 'E07c', 'PASS', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION', 'DEV-021',
             False, 'ONE_REAL_TRAIN_EPOCH_QUALIFICATION'), 'audit identity')
    require(audit['qualified_statuses'] == QUALIFIED and audit['retained_statuses'] == RETAINED and
            audit['not_qualified'] == NOT_QUALIFIED, 'status fields')
    sa = audit['session_access_audit']['counts']
    require(all(sa[k] == 0 for k in SESSION_ZERO) and sa['TRAIN_rows_exposed_per_process'] == 14467 and
            sa['authorized_TRAIN_access'] is True, 'session access audit')
    require(audit['environment_lock_sha256'] == LOCK_SHA and
            not audit['environment_reverification']['environment_rebuilt_or_mutated'], 'environment re-verification')
    require(sorted(audit['artifacts_sha256']) == sorted(set(ARTIFACTS) - {BASE + '.json'}), 'artifact list')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    final_tests = [t for t in audit['tests'] if t['final']]
    require(len(final_tests) == 16 and all(t['failures'] == t['errors'] == 0 and t['audit_denied'] == 0
                                           for t in final_tests) and
            all(t['skipped'] == 0 for t in final_tests if t['host'] == 'gpu'), 'final test runs')
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        row = json.loads(current[len(prefix):])
        require((row['classification'], row['milestone'], row['method_id'], row['final_status']) ==
                (CLASSIFICATION, 'M6D6e', 'E07c', 'PASS'), 'ledger row identity')
        require(row['qualified_statuses'] == QUALIFIED and row['not_qualified'] == NOT_QUALIFIED and
                row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and row['fidelity_class'] == 'CONTROLLED_ADAPTATION',
                'ledger statuses')
        require((row['authorized_TRAIN_access'], row['VAL_access'], row['TEST_access'], row['scientific_training'],
                 row['qualification_seed'], row['scientific_auxiliary_seed_used'],
                 row['scientific_auxiliary_run_completed'], row['benchmark_data_access']) ==
                (True, False, False, False, QUALIFICATION_SEED, False, False, 'AUTHORIZED_TRAIN_ONLY'), 'ledger access fields')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
        require(sorted(row['artifacts_sha256']) == sorted(ARTIFACTS), 'ledger artifact list')
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules and 'yaml' not in sys.modules and 'pyarrow' not in sys.modules,
            'static preflight: no torch, yaml or pyarrow')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'process_sha256': PROCESS_SHA, 'b256_repeatability': 'BITWISE',
            'epoch': {'steps': 56, 'consumed': 14336, 'dropped': 131, 'loss_denominator': 14467},
            'torch_imported': False, 'benchmark_files_opened': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true', help='evidence-only check before the ledger append')
    parser.add_argument('--rebuild-index', action='store_true', help='after the ledger append: write the index LAST')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
