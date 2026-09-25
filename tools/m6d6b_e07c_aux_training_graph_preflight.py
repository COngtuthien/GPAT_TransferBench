#!/usr/bin/env python3
"""Verify M6D6b E07c auxiliary-encoder training-graph evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch import, no CUDA, no model construction, no optimizer, no data
traversal, no manifest or image I/O. Only explicit allowlisted paths are read.
Historical index rows are carried from the authoritative commit without opening
their targets (tools/build_artifact_index.py is NOT used: it hashes every file);
only the explicit new M6D6b artifacts are hashed.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'ca206cdc15de830e467484baace84208678fb5ce'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
PRETRAIN_SHA = '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'
A6_OVERLAY_SHA = 'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
M6D6A_HELPERS_SHA = '1eeb7c037dbb25876eb6257a4eb0dee40bc3a998d5d20c03a2b4b937abe12541'
QUALIFICATION_SEED = 60602
HYPER = {'lr': 0.002, 'momentum': 0.9, 'weight_decay': 0.005}
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 116
LEDGER_PREFIX_SHA = '0d663a7e02cbd758ec2cf96bd74c7f09f3800c292278aad3d7c4f97006158615'
INDEX_BASELINE_ROWS = 656
CLASSIFICATION = 'M6D6B_E07C_AUX_TRAINING_GRAPH_QUALIFICATION'
BASE = 'outputs/audit/M6D6B_E07C_AUX_TRAINING_GRAPH_QUALIFICATION'
PROCESSES = tuple(f'outputs/audit/M6D6B_E07C_AUX_TRAINING_PROCESS_{i}.json' for i in (1, 2))
LOG = 'outputs/audit/M6D6B_E07C_AUX_TRAINING_RUNTIME_LOG.txt'
HARNESS = 'methods/difffas/aux_training_qualification.py'
TESTS = 'tests/test_m6d6b_e07c_aux_training_graph.py'
PREFLIGHT = 'tools/m6d6b_e07c_aux_training_graph_preflight.py'
ARTIFACTS = tuple(sorted((*PROCESSES, LOG, HARNESS, TESTS, PREFLIGHT, BASE + '.md', BASE + '.json')))
PRESERVED = ('configs/methods/e07c_difffas_bin_idfree.yaml', 'configs/frozen/difffas_bin_idfree_v1.yaml',
             'frozen_config_snapshot/configs/methods/e07c_difffas_bin_idfree.yaml',
             'frozen_config_snapshot/configs/frozen/difffas_bin_idfree_v1.yaml',
             'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md',
             'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
             'configs/run_logging_v1.yaml', 'configs/CONFIG_STATUS.md', 'outputs/audit/STAGE_STATE.json',
             'third_party/source_pins.json', 'methods/difffas/__init__.py', 'methods/difffas/adapter.py',
             'methods/difffas/contract.py', 'methods/difffas/encoder.py', 'methods/difffas/source.py',
             'methods/difffas/sampler.py', 'methods/difffas/seed_adapter.py',
             'methods/difffas/runtime_qualification.py', 'methods/common/learned.py', 'methods/common/upstream.py',
             'methods/common/config.py', 'environments/e07c.lock.json', 'environments/e07c.runtime.json',
             'environments/e07c.conda-explicit.txt', 'environments/e07c.pip-freeze.txt',
             'environments/e07c.pip-requirements.txt',
             'outputs/audit/M6C2B3_IMPLEMENTATION.md', 'outputs/audit/M6C2B3_IMPLEMENTATION.json',
             'outputs/audit/M6D6A_E07C_RUNTIME_QUALIFICATION.md',
             'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_1.json', 'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_2.json',
             'outputs/audit/M6D6A_R_E07C_CLEAN_REQUALIFICATION.md',
             'outputs/audit/M6D6A_R_E07C_CLEAN_REQUALIFICATION.json',
             'outputs/audit/M6D6A_R_E07C_SYNTHETIC_PROCESS_1.json',
             'outputs/audit/M6D6A_R_E07C_SYNTHETIC_PROCESS_2.json',
             'tests/test_m6d6a_e07c_runtime.py', 'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py',
             'tests/test_m6c2b3_runtime.py', 'tools/build_artifact_index.py')
QUALIFIED = ['E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED', 'E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED',
             'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
RETAINED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
            'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED']
NOT_QUALIFIED = ['REAL_TRAIN_PATH', 'AUXILIARY_ENCODER_TRAINING', 'AUX_B256_TRAINING_MEMORY',
                 'AUX_PRODUCTION_RUNNER', 'CHECKPOINT_WRITER', 'CHECKPOINT_LOADER', 'RESUME',
                 'MAIN_DIFFFAS_TRAINING_GRAPH', 'SCIENTIFIC_TRAINING', 'M8_BANK']
SCOPE = {'backward_calls': 1, 'optimizer_step_calls': 1, 'epochs': 0, 'checkpoint_created': False,
         'checkpoint_loaded': False, 'auxiliary_encoder_trained': False, 'scientific_training': False,
         'main_difffas_model_constructed': False, 'main_difffas_training': False, 'benchmark_data_access': False,
         'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False, 'synthetic_bank': False,
         'QUALIFICATION_ONLY': True, 'NOT_ELIGIBLE_FOR_BANK': True, 'NOT_ELIGIBLE_FOR_DOWNSTREAM': True,
         'NOT_ELIGIBLE_FOR_REPORTING': True, 'scientific_seed_consumed': False}
COUNTERS = {'optimizer_constructions': 1, 'sgd_constructions': 1, 'optimizer_step_calls': 1, 'backward_calls': 1,
            'zero_grad_calls': 1, 'autograd_grad_calls': 0, 'scheduler_constructions': 0, 'grad_clipping_calls': 0,
            'checkpoint_saves': 0, 'checkpoint_loads': 0}
SESSION = {'benchmark_manifest_reads': 0, 'benchmark_image_reads': 0, 'TRAIN_reads': 0, 'VAL_reads': 0,
           'TEST_reads': 0, 'firewall_denials': 0, 'backward_calls': 2, 'optimizer_step_calls': 2,
           'checkpoint_saves': 0, 'checkpoint_loads': 0, 'scientific_checkpoints_created': 0,
           'auxiliary_encoder_training_runs': 0, 'scientific_seed_runs_completed': 0}
DISCONNECTED = ['norm.bias', 'norm.weight']
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')
FORBIDDEN_WORDING = ('faithful difffas', 'native difffas', 'official reproduction of', 'auxiliary_encoder_trained:',
                     'auxiliary encoder trained', 'training_memory_qualified', 'checkpoint_writer_qualified',
                     'resume_qualified', 'scientifically_trained', 'm8_ready', 'production_runner_qualified',
                     'real_train_path_qualified')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6b: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(p.parts[0] in {'methods', 'tools', 'tests', 'configs', 'environments', 'outputs', 'docs',
                           'frozen_config_snapshot', 'third_party'}, 'evidence root')
    require(not {'data', 'manifests', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall')
    return (ROOT / p).read_bytes()


def source_cache():
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip() == '',
            'tracked pinned source unmodified')
    require(sha((root / 'models/custom_rn.py').read_bytes()) == CUSTOM_RN_SHA, 'custom_rn SHA256')
    require(sha((root / 'models/pretrain_classifier.py').read_bytes()) == PRETRAIN_SHA, 'pretrain_classifier SHA256')
    return root


def upstream_ast(root):
    """Stdlib re-derivation of the pinned SGD call and loop order (independent of the harness)."""
    mod = ast.parse((root / 'models/pretrain_classifier.py').read_bytes())
    opt = next(n for n in mod.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'optimizer')
    inner = next(n for n in next(n for n in mod.body if isinstance(n, ast.For)).body if isinstance(n, ast.For))
    body = {ast.unparse(n): n.lineno for n in inner.body}
    calls = [ast.unparse(n.func) for n in ast.walk(mod) if isinstance(n, ast.Call)]
    return {'optimizer_line': opt.lineno, 'params': [ast.unparse(a) for a in opt.value.args],
            'hyper': {k.arg: ast.literal_eval(k.value) for k in opt.value.keywords},
            'order': [body.get(s) for s in ('optimizer.zero_grad()', '_, _, _, outputs = resnet18(inputs)',
                                            'loss_asym = criteria(outputs, labels)', 'loss.backward()',
                                            'optimizer.step()')],
            'mode_calls': [c for c in calls if c.endswith(('.train', '.eval'))]}


def check_process(r, lock):
    require(r['status'] == 'PASS' and r['label'] == 'AUX_ENCODER_SYNTHETIC_TRAINING_GRAPH_ONLY', 'GPU process pass')
    require(r['qualification_seed'] == QUALIFICATION_SEED and r['experiment_seed'] is None and
            r['auxiliary_encoder_training_seed'] is None and QUALIFICATION_SEED not in (42, 1337, 2026), 'seed')
    require((r['qualification_batch_size'], r['scientific_aux_batch_size']) == (4, 256) and
            r['batch_256_training_memory_not_qualified'], 'batch disclosure')
    s = r['source_before']
    require(s == r['source_after'] and s['worktree_status'] == '', 'source mutation')
    require((s['commit'], s['tree'], s['a6_overlay_sha256']) == (PIN, TREE, A6_OVERLAY_SHA), 'pinned source / A6')
    require(s['files_sha256']['models/custom_rn.py'] == CUSTOM_RN_SHA and
            s['files_sha256']['models/pretrain_classifier.py'] == PRETRAIN_SHA, 'source digests')
    env = r['environment_before']
    require(env == r['environment_after'], 'environment mutation')
    require({k: v for k, v in env.items() if k != 'launch_environment'} ==
            {k: v for k, v in lock['identity'].items() if k != 'launch_environment'}, 'executed identity = lock')
    strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
    require(strip(env['launch_environment']) == strip(lock['identity']['launch_environment']), 'launch env = lock')
    e = r['encoder']
    require((e['module'], e['topology'], e['stage_widths'], e['structural_difference_vs_pinned_resnet18']) ==
            ('custom_rn', [3, 4, 6, 3], [64, 256, 512, 512], ['fc']), 'pinned custom_rn encoder')
    require(e['head'] == {'type': 'Linear', 'in_features': 512, 'out_features': 7, 'bias': True} and
            not e['torchvision_substitute'] and not e['projection_or_adapter_modules_added'] and
            not e['weights_loaded'], 'A3 head; no substitute/projection/weights')
    require(e['mode']['default_training_all_modules'] and e['mode']['model_train_called'] and
            not e['mode']['eval_called'], 'train mode per source default')
    st = e['state_initial']
    require(st['parameters'] == sum(math.prod(v) for v in st['shapes'].values()) == 46233707, 'parameter arithmetic')
    require(r['forward']['logits_shape'] == [4, 7] and r['forward']['loss_attached_to'] == 'outputs[3] only', 'logits')
    lo = r['loss']
    require(lo['finite'] and math.isfinite(lo['value']) and lo['criterion'] == 'torch.nn.CrossEntropyLoss()' and
            lo['label_smoothing'] == 0.0 and lo['weight'] is None and lo['reduction'] == 'mean', 'plain finite CE')
    o = r['optimizer']
    h = o['hyperparameters']
    require((h['lr'], h['momentum'], h['weight_decay'], h['dampening'], h['nesterov'], h['maximize']) ==
            (0.002, 0.9, 0.005, 0, False, False) and o['scheduler'] is None and o['grad_clipping'] is None and
            (o['optimized_tensors'], o['optimized_parameters']) == (st['parameter_tensors'], st['parameters']),
            'exact SGD over model.parameters(); no scheduler')
    g = r['gradients']
    require(g['all_finite'] and g['none_grad_tensors'] == DISCONNECTED and g['unexpected_disconnected_tensors'] == []
            and g['tensors_with_nonzero_grad'] == g['tensors_with_grad'] == o['optimized_tensors'] - 2 and
            g['fc_weight']['nonzero_elements'] > 0 and g['fc_bias']['nonzero_elements'] > 0 and
            g['stem_conv1']['nonzero_elements'] > 0, 'gradient gates')
    u = r['parameter_update']
    require(u['changed_tensors'] == g['tensors_with_grad'] and u['unchanged_tensors'] == DISCONNECTED and
            u['fc_changed'] and u['backbone_changed_tensors'] > 0 and u['all_finite_after'] and
            u['changed_parameters_outside_optimizer_scope'] == 0 and
            u['sgd_first_step_formula_max_abs_deviation'] <= u['sgd_formula_tolerance'] and
            u['buffers_changed_by_backward_or_step'] == [], 'parameter update gates')
    require(r['counters'] == COUNTERS and all(r[k] == v for k, v in SCOPE.items()), 'counters / scope flags')
    fw = r['firewall']
    require(not fw['denied'] and not any(fw['attempts'].values()), 'firewall zero denials')
    require(all(a[0] in ('git', 'nvidia-smi') or a == ['/sbin/ldconfig', '-p'] for a in fw['subprocesses']),
            'subprocess allowlist')
    require(r['compatibility_patch'] == 'NONE' and r['fidelity'] == 'CONTROLLED_ADAPTATION', 'patch/fidelity')
    m = r['cuda_memory']
    require(m['label'] == 'AUX_ENCODER_SYNTHETIC_TRAINING_STEP_MEMORY_OBSERVATION' and
            not m['qualifies_training_memory'] and not m['qualifies_scientific_batch_size'], 'memory label')
    p = r['precision']
    require(p['label'] == 'ENGINEERING_QUALIFICATION_CONTROLS' and not p['freezes_scientific_precision_policy'],
            'precision label')
    require(r['warnings'] == [], 'no runtime warnings')


def repeatability(p, q, raw_p, raw_q):
    grads = {n: p['gradients']['per_tensor'][n] == q['gradients']['per_tensor'][n] for n in p['gradients']['per_tensor']}
    after = p['parameter_update']['parameter_sha256_after']
    params = {n: after[n] == q['parameter_update']['parameter_sha256_after'][n] for n in after}
    return {'independent_processes': 2, 'process_json_bytes_identical': raw_p == raw_q, 'process_sha256': sha(raw_p),
            'initial_state_identical': p['encoder']['state_initial'] == q['encoder']['state_initial'],
            'synthetic_input_sha256_identical': p['synthetic_batch']['input']['sha256'] ==
            q['synthetic_batch']['input']['sha256'],
            'synthetic_target_sha256_identical': p['synthetic_batch']['target_sha256'] ==
            q['synthetic_batch']['target_sha256'],
            'pre_step_loss_bitwise_equal': p['loss']['value_float32_hex'] == q['loss']['value_float32_hex'],
            'gradient_tensors_compared': len(grads), 'gradient_tensors_bitwise_equal': sum(grads.values()),
            'post_step_parameter_tensors_compared': len(params),
            'post_step_parameter_tensors_bitwise_equal': sum(params.values()),
            'changed_tensor_counts_equal': p['parameter_update']['changed_tensors'] ==
            q['parameter_update']['changed_tensors'],
            'all_bitwise_equal': raw_p == raw_q and all(grads.values()) and all(params.values()),
            'tolerance_used': 'NONE (bitwise)',
            'determinism_note': ('qualification_seed 60602; frozen cudnn benchmark=False/deterministic=True; TF32 off; '
                                 'CUBLAS_WORKSPACE_CONFIG=:4096:8; torch deterministic algorithms NOT forced. '
                                 'Equality observed, not manufactured.')}


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
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6b changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    return status


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    require(sha(read('methods/difffas/runtime_qualification.py')) == M6D6A_HELPERS_SHA, 'M6D6a helpers unchanged')
    root = source_cache()
    lock_raw = read('environments/e07c.lock.json')
    require(sha(lock_raw) == LOCK_SHA, 'environment lock digest')
    lock = json.loads(lock_raw)
    raw_p, raw_q = [read(n) for n in PROCESSES]
    p, q = json.loads(raw_p), json.loads(raw_q)
    for r in (p, q):
        check_process(r, lock)
    for rel, h in p['source_before']['files_sha256'].items():
        require(sha((root / rel).read_bytes()) == h, 'laptop source closure ' + rel)
    up = upstream_ast(root)
    pu = p['upstream_training_semantics']
    require(up['params'] == ['resnet18.parameters()'] and up['hyper'] == HYPER == pu['optimizer']['hyperparameters']
            and up['optimizer_line'] == pu['optimizer']['line'] == 27 and up['order'] == sorted(up['order']) ==
            [36, 37, 38, 40, 41] and up['mode_calls'] == [] == pu['mode_calls'], 'upstream semantics (stdlib AST)')
    rep = repeatability(p, q, raw_p, raw_q)
    require(rep['all_bitwise_equal'], 'two-process bitwise repeatability')
    audit = json.loads(read(BASE + '.json'))
    require(audit['classification'] == CLASSIFICATION and audit['milestone'] == 'M6D6b' and
            audit['final_status'] == 'PASS' and audit['qualified_statuses'] == QUALIFIED and
            audit['retained_statuses'] == RETAINED and audit['not_qualified'] == NOT_QUALIFIED and
            audit['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'status fields')
    require(audit['repeatability'] == rep, 'repeatability evidence')
    require(audit['session_access_audit']['counts'] == SESSION and
            audit['session_access_audit']['scope'] == 'COMPLETE_M6D6B_SESSION_LAPTOP_AND_GPU', 'session access audit')
    require(audit['environment_lock_sha256'] == LOCK_SHA and not audit['environment_reverification'][
        'environment_rebuilt_or_mutated'], 'environment re-verification')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    require(sorted(audit['artifacts_sha256']) == sorted(set(ARTIFACTS) - {BASE + '.json'}), 'artifact list')
    for path, h in audit['preserved_inputs_sha256'].items():
        require(sha(read(path)) == h, 'preserved input ' + path)
    final_tests = [t for t in audit['tests'] if t['final']]
    require(len(final_tests) == 6 and all(t['failures'] == t['errors'] == 0 for t in final_tests), 'final test runs')
    require(all(t['audit_denied'] == t['audit_manifest'] == t['audit_image'] == t['audit_data_or_runs'] == 0
                for t in audit['tests']), 'test audit counts')
    log = read(LOG).decode()
    after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
    require(after == ['scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT encoder_final=ABSENT'] * 2,
            'no scientific run root or auxiliary checkpoint')
    require('RUNS_ABSENT' in log and 'status_lines 0' in log, 'GPU cleanup recorded')
    report = read(BASE + '.md').decode().lower()
    for phrase in FORBIDDEN_WORDING:
        require(phrase not in report, 'forbidden wording: ' + phrase)
    for phrase in ('controlled_adaptation', 'implemented_not_executed', 'e07c_aux_encoder_training_graph_qualified',
                   'b256 scientific training memory remains unqualified', 'qualification_only'):
        require(phrase in report, 'required wording: ' + phrase)
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = read(LEDGER)
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA, 'ledger authority')
    require(current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        row = json.loads(current[len(prefix):])
        require((row['classification'], row['milestone'], row['method_id']) == (CLASSIFICATION, 'M6D6b', 'E07c'),
                'ledger row identity')
        require(row['final_status'] == 'PASS' and row['qualified_statuses'] == QUALIFIED and
                row['not_qualified'] == NOT_QUALIFIED and row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and
                row['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'ledger status fields')
        require((row['qualification_seed'], row['backward_passes'], row['optimizer_step_calls']) ==
                (QUALIFICATION_SEED, 2, 2) and row['scientific_training'] is False and
                row['auxiliary_encoder_trained'] is False and row['checkpoint_created'] is False and
                row['benchmark_data_access'] is False and row['experiment_seeds_used'] == [] and
                row['session_access_counts'] == SESSION, 'ledger scope/access')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if stage == 'final':
        require(read(INDEX) == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules, 'static preflight must not import torch')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True,
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'qualification_processes': 2, 'all_bitwise_equal': rep['all_bitwise_equal'],
            'session_access_counts': SESSION, 'torch_imported': False, 'model_execution_in_preflight': False}


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
