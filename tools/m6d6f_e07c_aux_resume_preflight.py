#!/usr/bin/env python3
"""Verify M6D6f E07c auxiliary resume (Amendment A8) evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch, no YAML parser, no pyarrow, no CUDA, no model, no checkpoint/sidecar I/O, no Parquet decode
and no image read. Data inputs are checked through Git blob ids against the authority commit and never
opened. The A8 overlay is the JSON subset of YAML. The five recorded qualification processes are
re-derived (bitwise verdicts, reconciliation, step accounting, per-process access) without re-running
anything. Historical index rows are carried from the authority commit without opening their targets;
only the M6D6f artifacts (new files and the three modified runner files) are hashed.
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
AUTHORITY = 'a00aba9f998fb0d91122617012af0dec7794c007'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
WRITER_SHA = '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
QUALIFICATION_SEED = 60606
DATA_INPUTS = ('manifests/split_v1.parquet', 'manifests/artifact_probe_classes_v1.json',
               'manifests/difffas_bin_idfree_train_v1.parquet')
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 120
INDEX_BASELINE_ROWS = 699
CLASSIFICATION = 'M6D6F_E07C_AUX_RESUME_POLICY_QUALIFICATION'
A8_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A8_E07c_Aux_Resume_Policy.md'
A8_OVERLAY = 'configs/amendments/e07c_a8_aux_resume_policy.yaml'
A8 = {A8_DOC: 'deb11b5cd80fa873bde1ef9e87160fed3a8065903cd8c81c9463b087fbf73b2f',
      A8_OVERLAY: 'a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a'}
IO_MOD, ENGINE, RESUME, HARNESS = ('methods/difffas/aux_runner_io.py', 'methods/difffas/aux_runner.py',
                                   'methods/difffas/aux_resume.py', 'methods/difffas/aux_resume_qualification.py')
CLI, TESTS, PREFLIGHT = 'tools/run_e07c_aux.py', 'tests/test_m6d6f_e07c_aux_resume.py', \
    'tools/m6d6f_e07c_aux_resume_preflight.py'
EXECUTED_CODE = {IO_MOD: '297bc344b51a789a0edab209541a77f7515a53354fe7ca26b4fba7a5530d6351',
                 ENGINE: 'ab07c90aba0f25b8e49bdb02ab919f28512d369f7bb3c4c964fb52e9692f1f69',
                 RESUME: '55626c2596165dbf59845270c1d050f23ddb8e1873d4446083398a12ab8d5fcf',
                 HARNESS: '202021e3a597c530d65d6c6843f52c91091d09a97a05a7315cd01797585e4404',
                 CLI: 'a9a7bcb509d36f81e892ef31e74cac446fc93d08b3ab35f007046d9181af56d4', **A8}
MODIFIED = (IO_MOD, ENGINE, CLI)
BASE = 'outputs/audit/M6D6F_E07C_AUX_RESUME'
PROCESSES = {'reference': BASE + '_REFERENCE.json', 'interrupted': BASE + '_INTERRUPTED.json',
             'restored': BASE + '_RESTORED.json', 'epoch0_interrupted': BASE + '_EPOCH0_INTERRUPTED.json',
             'epoch0_restored': BASE + '_EPOCH0_RESTORED.json'}
PROCESS_SHA = {'reference': '18b763c3e67c7af835656bf2f8e409e94f67ccf8774a5eb0f764a96f6c58a565',
               'interrupted': '2cc770f16934e0e552a2b9d3e4a01170195b3b18d95a0e2707be7028cc088b7e',
               'restored': '957a4a35b3819580fe005156cec46a2070ada467dc988a1f4020b5f5cbc2e8e0',
               'epoch0_interrupted': '88c0964a7d81249aa616546a1e23b007b0f02abfde672159fa4cfb01463a5c92',
               'epoch0_restored': 'b4df84272f31f48efdd16eb22386ffc84307b2bb9f97f192b20e8a477c0b3212'}
LOG = BASE + '_RUNTIME_LOG.txt'
NEW = tuple(sorted((A8_DOC, A8_OVERLAY, RESUME, HARNESS, TESTS, PREFLIGHT, *PROCESSES.values(), LOG,
                    BASE + '_POLICY.md', BASE + '_POLICY.json')))
ARTIFACTS = tuple(sorted(NEW + MODIFIED))
PRESERVED = ('configs/methods/e07c_difffas_bin_idfree.yaml', 'configs/frozen/difffas_bin_idfree_v1.yaml',
             'frozen_config_snapshot/configs/methods/e07c_difffas_bin_idfree.yaml',
             'frozen_config_snapshot/configs/frozen/difffas_bin_idfree_v1.yaml', 'configs/run_logging_v1.yaml',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Aux_Production_Runner_Addendum_M6D6e.md',
             'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
             'configs/amendments/e07c_a7_execution_policy.yaml',
             'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml',
             'environments/e07c.lock.json', 'environments/e07c.runtime.json',
             'methods/difffas/aux_runner_qualification.py', 'methods/difffas/aux_checkpoint.py',
             'methods/difffas/execution_policy.py', 'methods/difffas/encoder.py', 'methods/difffas/seed_adapter.py',
             'methods/difffas/runtime_qualification.py', 'methods/difffas/aux_training_qualification.py',
             'methods/difffas/adapter.py', 'methods/common/runlog.py', 'methods/common/learned_runlog.py',
             'methods/common/learned.py', 'methods/common/config.py', 'methods/dsdg/runner.py',
             'methods/dsdg/runner_io.py', 'tests/test_m6d6e_e07c_aux_runner.py', 'tests/test_m6d6d_e07c_execution_policy.py',
             'tests/test_m6d6c_e07c_aux_checkpoint.py', 'tests/test_m6d6b_e07c_aux_training_graph.py',
             'tests/test_m6d6a_e07c_runtime.py', 'tests/test_m6c2b3_contract.py', 'tests/test_m6c2b3_encoder.py',
             'tests/test_m6c2b3_runtime.py', 'tools/m6d6e_e07c_aux_runner_preflight.py', 'tools/build_artifact_index.py',
             'outputs/audit/M6D6E_E07C_AUX_ONE_EPOCH.json', 'outputs/audit/M6D6E_E07C_AUX_RUNNER_QUALIFICATION.json')
NEW_STATUSES = ['E07c_AUX_FAILURE_RECOVERY_POLICY_FROZEN', 'E07c_AUX_EPOCH_BOUNDARY_RESUME_QUALIFIED',
                'E07c_AUX_APPEND_ONLY_RECONCILIATION_QUALIFIED']
QUALIFIED = NEW_STATUSES + ['IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION']
RETAINED = ['E07c_EXECUTION_ENVIRONMENT_QUALIFIED', 'E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED',
            'E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED', 'E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED',
            'E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED', 'E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED',
            'E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED',
            'E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED', 'E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED',
            'E07c_PRODUCTION_PRECISION_POLICY_FROZEN', 'E07c_MAIN_ENCODER_LOADER_RNG_POLICY_FROZEN',
            'E07c_THROWAWAY_ENCODER_RNG_COMPATIBILITY_QUALIFIED', 'E07c_EXECUTION_POLICY_RUNTIME_QUALIFIED',
            'E07c_AUX_REAL_TRAIN_PATH_QUALIFIED', 'E07c_AUX_B256_TRAINING_MEMORY_QUALIFIED',
            'E07c_AUX_PRODUCTION_RUNNER_QUALIFIED', 'E07c_AUX_EPOCH_CHECKPOINT_INTEGRATION_QUALIFIED']
NOT_QUALIFIED = ['AUXILIARY_ENCODER_200_EPOCH_TRAINING', 'AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT',
                 'AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', 'MAIN_DIFFFAS_TRAINING_GRAPH', 'MAIN_CHECKPOINT_RESUME',
                 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION', 'MAIN_PRODUCTION_RUNNER', 'SCIENTIFIC_TRAINING', 'M8_BANK']
STEP_FIELDS = ('sample_ids', 'sample_ids_sha256', 'indices_sha256', 'input_sha256', 'input_shape', 'input_dtype',
               'labels_sha256', 'labels', 'class_histogram', 'loss_hex', 'gradients_sha256', 'parameters_after_sha256',
               'buffers_after_sha256', 'momentum_after_sha256', 'without_grad', 'rng_after')
TENSOR_COUNTS = {'gradients': 110, 'parameters_after': 112, 'buffers_after': 111, 'momentum_after': 110}
FACE_OPENS = {'reference': 17664, 'interrupted': 17664, 'restored': 3328, 'epoch0_interrupted': 3328,
              'epoch0_restored': 3328}
PHYSICAL = {'reference': 57, 'interrupted': 57, 'restored': 1, 'epoch0_interrupted': 1, 'epoch0_restored': 1}
SAVES = {'reference': 4, 'interrupted': 4, 'restored': 1, 'epoch0_interrupted': 2, 'epoch0_restored': 1}
LOADS = {'reference': 0, 'interrupted': 0, 'restored': 3, 'epoch0_interrupted': 0, 'epoch0_restored': 3}
ZERO_COUNTERS = ('other_optimizer_step_calls', 'scheduler_constructions', 'grad_clipping_calls', 'autograd_grad_calls',
                 'activation_checkpoint_calls', 'autocast_entries', 'grad_scaler_constructions')
FORBIDDEN_WORDING = ('faithful difffas', 'native difffas', 'official reproduction of', 'scientifically trained auxiliary',
                     'auxiliary encoder is trained', 'scientific attempt 1', 'seed 42 was launched',
                     'main_production_runner_qualified', 'm8_ready', 'main_checkpoint_resume qualified')
REQUIRED_WORDING = ('controlled_adaptation', 'dev-021', 'implemented_not_executed', 'one_logical_run',
                    'deterministic_implementation_clarification', 'superseded_by_resume_rollback', 'weights_only=true',
                    'bitwise', '60606', 'not** launched', 'qualification_only', 'e07c_aux_epoch_boundary_resume_qualified',
                    'm6d6g', 'max |δ| = 0')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6f: ' + message)


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
    require(git('status', '--porcelain', '--', *DATA_INPUTS).decode().strip() == '', 'data inputs unmodified')
    for rel in DATA_INPUTS:
        require(git('rev-parse', f'HEAD:{rel}').decode().strip() == git('rev-parse', f'{AUTHORITY}:{rel}').decode().strip(),
                'data input blob changed ' + rel)


def source_cache():
    root = ROOT / 'third_party/source_cache/difffas'
    require(git('rev-parse', 'HEAD', root=root).decode().strip() == PIN, 'source commit')
    require(git('rev-parse', 'HEAD^{tree}', root=root).decode().strip() == TREE, 'source tree')
    require(git('status', '--porcelain', '--untracked-files=no', root=root).decode().strip() == '', 'pinned source clean')
    raw = (root / 'models/pretrain_classifier.py').read_bytes()
    require(sha(raw) == WRITER_SHA, 'pretrain_classifier.py SHA256')
    text = raw.decode()
    require(all(t not in text for t in ('load_state_dict', 'torch.load', 'resume', 'start_epoch', 'optimizer.state_dict')) and
            'for epoch in range(0,num_epochs)' in text, 'pinned source has NO resume mechanism')


def a8_policy():
    raw = read(A8_OVERLAY)
    require(sha(raw) == A8[A8_OVERLAY] and sha(read(A8_DOC)) == A8[A8_DOC], 'A8 identity')
    p = json.loads(raw)
    require(p['amendment_document'] == {'path': A8_DOC, 'sha256': A8[A8_DOC]}, 'A8 overlay binds its document')
    for rel, digest in p['bound_authority_sha256'].items():
        require(sha(read(rel)) == digest, 'A8-bound authority ' + rel)
    require((p['classification'], p['authority_commit'], p['milestone'], p['method_id'], p['fidelity']['fidelity_class'],
             p['fidelity']['deviation'], p['fidelity']['new_deviation'], p['fidelity']['new_fidelity_class'],
             p['logical_run']['identity'], p['logical_run']['scientific_auxiliary_runs'], p['boundary']['rule'],
             p['sidecar']['unsafe_load'], p['cli']['resume_latest'], p['cli']['automatic_discovery'],
             p['metrics']['truncate_or_rewrite'], p['step_accounting']['logical_steps_completed_run'],
             p['qualification']['qualification_seed'], p['source']['source_resume_capability']) ==
            ('DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', AUTHORITY, 'M6D6f', 'E07c', 'CONTROLLED_ADAPTATION', 'DEV-021',
             False, False, 'ONE_LOGICAL_RUN', 1, 'EXACT_EPOCH_BOUNDARY_ONLY', 'FORBIDDEN', 'NOT_SUPPORTED',
             'NOT_SUPPORTED', 'FORBIDDEN', 11200, QUALIFICATION_SEED, 'NONE'), 'A8 frozen semantics')
    return p


def code_ast():
    """Stdlib AST/text checks of the executed code (engine unchanged in substance, resume isolated, safe loads)."""
    engine = read(ENGINE).decode()
    for token in ('resume_state', 'load_state_dict', 'torch.load(', 'autocast(', 'GradScaler('):
        require(token not in engine, 'engine token ' + token)
    fns = {n.name: ast.unparse(n) for n in ast.walk(ast.parse(engine)) if isinstance(n, ast.FunctionDef)}
    require('save_whole_module(self.model, partial, config)' in fns['checkpoint_event'] and
            'state_dict' not in fns['checkpoint_event'], 'scientific whole-module checkpoint seam unchanged')
    require('rus = running_loss / len(self.dataset)' in fns['run_epoch'], 'source epoch-loss denominator')
    loader = fns['build_loader']
    require(all(t not in loader for t in ('generator=', 'worker_init_fn=', 'persistent_workers=', 'sampler=')),
            'DataLoader semantics unchanged (no generator added)')
    sci = fns['run_scientific']
    order = [sci.index(t) for t in ('seed_process(torch, mode, seed, config)', 'production_components(',
                                    'run.verify_before_open(resume_sidecar)', 'run.begin_resume()', 'run.begin_fresh()',
                                    'trainer.run_epoch(epoch, ctx, t0)', 'run.commit_boundary(epoch)')]
    require(order == sorted(order), 'scientific driver order (seed -> construct -> verify -> restore/epoch-0 -> loop)')
    resume = read(RESUME).decode()
    loads = [ast.unparse(n) for n in ast.walk(ast.parse(resume)) if isinstance(n, ast.Call) and
             ast.unparse(n.func).endswith('torch.load')]
    require(loads == ["torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)"], 'one safe sidecar load')
    for text in (resume, read(HARNESS).decode()):
        require('weights_only=False' not in text, 'no unsafe load')
    cli = read(CLI).decode()
    require("parser.add_argument('--resume-state'" in cli and 'ar.resolve_committed(run_dir, args.resume_state)' in cli,
            'CLI --resume-state is exact and statically resolved')
    require('import torch' not in cli.split('def main')[0], 'CLI gates import no Torch')
    io_mod = read(IO_MOD).decode()
    refused = next(n for n in ast.parse(io_mod).body if isinstance(n, ast.Assign) and
                   ast.unparse(n.targets[0]) == 'REFUSED_OPTIONS')
    flags = ast.literal_eval(refused.value)
    require({'--resume-latest', '--resume', '--auto-resume', '--resume-from'} <= set(flags) and
            '--resume-state' not in flags, 'CLI refuses every discovery-style resume flag')
    harness = read(HARNESS).decode()
    require('run_scientific' not in harness and 'SCIENTIFIC_SEED' not in harness, 'harness never launches scientific mode')


def compare_steps(a, b):
    return all(a[f] == b[f] for f in STEP_FIELDS)


def check_processes():
    ev = {}
    for k, path in PROCESSES.items():
        raw = read(path)
        require(sha(raw) == PROCESS_SHA[k], 'process evidence bytes ' + k)
        ev[k] = json.loads(raw)
    R, I, S, EI, ES = (ev[k] for k in PROCESSES)
    for k, e in ev.items():
        require(e['status'] == 'PASS' and e['label'] == 'QUALIFICATION_ONLY' and e['milestone'] == 'M6D6f', 'status ' + k)
        require((e['qualification_seed'], e['auxiliary_encoder_training_seed'], e['experiment_seed'],
                 e['scientific_seed_consumed'], e['scientific_attempt']) == (QUALIFICATION_SEED, None, None, False, False),
                'seed fields ' + k)
        require(e['launch_environment']['PYTHONHASHSEED'] == str(QUALIFICATION_SEED) and
                e['launch_environment']['NVIDIA_TF32_OVERRIDE'] == '0', 'launch env ' + k)
        require(e['code_sha256'] == {IO_MOD: EXECUTED_CODE[IO_MOD], ENGINE: EXECUTED_CODE[ENGINE],
                                     RESUME: EXECUTED_CODE[RESUME], CLI: EXECUTED_CODE[CLI],
                                     'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml':
                                         sha(read('configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml')),
                                     A8_OVERLAY: A8[A8_OVERLAY]} and e['harness_sha256'] == EXECUTED_CODE[HARNESS],
                'executed code = this worktree ' + k)
        require('/qualification/m6d6f/E07c_aux/q60606-' in e['run_dir'] and '/runs/' not in e['run_dir'], 'root ' + k)
        require(not any(e[f] for f in ('scientific_training', 'scientific_checkpoint_created', 'scientific_run_root_written',
                                       'main_difffas_training', 'synthetic_bank', 'VAL_access', 'TEST_access')),
                'no scientific / VAL / TEST ' + k)
        c = e['counters']
        require(all(c[z] == 0 for z in ZERO_COUNTERS) and c['sgd_step_calls'] == c['backward_calls'] ==
                c['zero_grad_calls'] == PHYSICAL[k] and c['torch_save_calls'] == SAVES[k] and
                c['torch_load_calls'] == LOADS[k], 'counters ' + k)
        require(all(x == {'map_location': "'cpu'", 'weights_only': True} for x in e['load_calls']) and
                len(e['load_calls']) == LOADS[k], 'every load weights_only=True ' + k)
        a = e['benchmark_access_audit']
        require((a['VAL_image_reads'], a['TEST_image_reads'], a['non_train_face_accesses'],
                 a['raw_or_other_benchmark_data_accesses'], a['unauthorized_manifest_accesses'],
                 a['faces_enumeration_or_write'], a['scientific_run_root_accesses'], a['foreign_weight_accesses'],
                 a['denied_events']) == (0,) * 9 and a['train_faces_all_in_population'] and
                a['train_face_opens'] == FACE_OPENS[k], 'per-process access ' + k)
        require((e['access_summary']['TRAIN_rows_exposed_to_dataset'], e['access_summary']['VAL_rows_exposed_to_dataset'],
                 e['access_summary']['TEST_rows_exposed_to_dataset']) == (14467, 0, 0) and
                not e['population']['main_relation_opened'], 'rows exposed ' + k)
        require(not e['firewall']['denied'] and e['environment_before'] == e['environment_after'] and
                e['source_before'] == e['source_after'], 'firewall / environment / source stable ' + k)
    require(len({e['run_id'] for e in ev.values()}) == 1 and len({e['pid'] for e in ev.values()}) == 5, 'five processes')
    require(S['run_dir'] == I['run_dir'] and ES['run_dir'] == EI['run_dir'], 'restores resume the interrupted roots')
    # uninterrupted determinism across the two fresh epoch-1 processes
    require(R['epoch1'] == I['epoch1'] and R['epoch1_losses_hex'] == I['epoch1_losses_hex'] and
            R['epoch1_checkpoint_event']['sha256'] == I['epoch1_checkpoint_event']['sha256'] and
            R['epoch1_sidecar']['model_sha256'] == I['epoch1_sidecar']['model_sha256'], 'epoch 1 identical')
    require((R['epoch1']['optimizer_steps'], R['epoch1']['consumed_examples'], R['epoch1']['dropped_examples'],
             R['epoch1']['epoch_loss_denominator']) == (56, 14336, 131, 14467), 'epoch accounting')
    # epoch-0 boundary
    for e in (R, I, EI):
        require((e['epoch0_sidecar']['completed_epoch'], e['epoch0_sidecar']['global_step']) == (0, 0) and
                e['rng_before_epoch0_sidecar'] == e['rng_after_epoch0_sidecar'], 'epoch-0 sidecar before the iterator')
    # restore: verified load, state, RNG, reconciliation
    for e, boundary, superseded, g in ((S, I['epoch1_sidecar'], I, 57), (ES, EI['epoch0_sidecar'], EI, 1)):
        v, rec = e['verified_sidecar'], e['reconciliation']
        require(v['sha256_verified_before_load'] and v['weights_only'] and v['entry']['sha256'] == boundary['sha256'] and
                v['torch_load_calls_so_far'] == [{'map_location': "'cpu'", 'weights_only': True}] and
                e['fresh_context_refuses_existing_root'], 'SHA256 before the weights_only load')
        require(rec['restore']['model_sha256'] == boundary['model_sha256'] and
                rec['restore']['optimizer_digest'] == boundary['optimizer_digest'] and
                rec['restore']['rng_digest'] == boundary['rng_digest'], 'restored state = committed state')
        require((rec['overwrite'], rec['truncated'], rec['superseded']['classification'], rec['superseded']['step_records'],
                 rec['superseded']['global_step_range']) == (False, False, 'SUPERSEDED_BY_RESUME_ROLLBACK', 1, [g, g]),
                'append-only reconciliation')
        require(rec['superseded']['loss_hex'] == [superseded['steps'][str(g)]['loss_hex']], 'superseded record identity')
        require(e['metrics_append_only']['prefix_sha256_unchanged'] and
                e['metrics_before_resume']['sha256'] == superseded['metrics_after']['sha256'], 'metrics never truncated')
        m = e['metrics_after']
        require((m['physical_step_records'], m['logical_step_records']) == (g + 1, g) and
                m['events'][-2:] == ['resume_reconciliation', 'e07c_aux_resume_reconciliation'], 'physical vs logical view')
        require([s['kind'] for s in e['index_after']['sessions']] == ['FRESH', 'RESUME'], 'one logical run, two sessions')
    require(S['rng_before_iterator'] == R['rng_at_epoch1_boundary'] and
            ES['rng_before_iterator'] == R['rng_after_epoch0_sidecar'], 'RNG restored to the committed boundary')
    # bitwise verdicts (re-derived from the recorded fields; tensor comparisons recorded by the restore processes)
    verdict = {}
    for e, g, others in ((S, 57, (('reference', R), ('interrupted', I))), (ES, 1, (('reference', R), ('e0interrupted', EI)))):
        for name, other in others:
            require(compare_steps(other['steps'][str(g)], e['steps'][str(g)]), f'bitwise fields vs {name} (step {g})')
            cmp = e['comparisons'][name]
            require(cmp['fields']['all_bitwise_equal'] and cmp['tensors']['all_bitwise_equal'] and cmp['global_step'] == g,
                    'recorded bitwise verdict')
            for key, n in TENSOR_COUNTS.items():
                t = cmp['tensors'][key]
                require((t['tensors'], t['bitwise_equal_tensors'], t['max_abs_diff']) == (n, n, 0.0), key + ' bitwise')
            require(cmp['tensors']['rng_after'] == {'torch_cpu': True, 'torch_cuda_device': True}, 'RNG bitwise')
            verdict[f'step_{g}_vs_{name}'] = 'BITWISE'
    require(S['step_accounting']['logical_authoritative_optimizer_steps'] == 57 and
            S['step_accounting']['physical_optimizer_steps_executed'] == 58 and
            S['step_accounting']['superseded_optimizer_steps'] == 1, 'step accounting')
    return ev, verdict


def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in MODIFIED:
        require(p in rows and rows[p]['sha256'] != EXECUTED_CODE[p], 'modified artifact row ' + p)
    for p in ARTIFACTS:
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


def worktree(ledger_and_index_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in NEW] + [' M ' + p for p in MODIFIED]
    if ledger_and_index_modified:
        expected += [' M ' + INDEX, ' M ' + LEDGER]
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6f changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def verify(stage):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(git('branch', '--show-current').decode().strip() == 'm6-baselines', 'branch')
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for path in PRESERVED:
        require(read(path) == git('show', AUTHORITY + ':' + path), 'immutable input ' + path)
    require(sha(read('environments/e07c.lock.json')) == LOCK_SHA, 'environment lock')
    data_inputs_unchanged()
    source_cache()
    for path, digest in EXECUTED_CODE.items():
        require(sha(read(path)) == digest, 'executed code unchanged since the GPU runs: ' + path)
    contract = json.loads(read('configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml'))
    require(contract['resume']['status'] == 'AUX_RESUME_NOT_QUALIFIED', 'M6D6e contract record untouched')
    a8_policy()
    code_ast()
    ev, verdict = check_processes()
    log = read(LOG).decode()
    after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
    require(len(after) == 5 and set(after) == {'scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT '
                                               'encoder_final=ABSENT'}, 'no scientific root in any process')
    for token in ('RUNS_ABSENT', 'PROTECTED_UNCHANGED', 'ANCESTOR_OK', 'SOURCE RESUME CAPABILITY: NONE',
                  'never combined', 'DELIBERATE QUALIFICATION INTERRUPTION', "['uname', '-p']", *PROCESS_SHA.values()):
        require(token in log, 'runtime log: ' + token)
    report = read(BASE + '_POLICY.md').decode().lower()
    for phrase in FORBIDDEN_WORDING:
        require(phrase not in report, 'forbidden wording: ' + phrase)
    for phrase in REQUIRED_WORDING:
        require(phrase in report, 'required wording: ' + phrase)
    audit = json.loads(read(BASE + '_POLICY.json'))
    require((audit['classification'], audit['milestone'], audit['method_id'], audit['final_status'], audit['method_status'],
             audit['fidelity_class'], audit['deviation'], audit['new_deviation'], audit['new_fidelity_class'],
             audit['authority_commit'], audit['qualification_seed'], audit['bitwise_resume']) ==
            (CLASSIFICATION, 'M6D6f', 'E07c', 'PASS', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION', 'DEV-021',
             False, False, AUTHORITY, QUALIFICATION_SEED, True), 'audit identity')
    require(audit['qualified_statuses'] == QUALIFIED and audit['retained_statuses'] == RETAINED and
            audit['not_qualified'] == NOT_QUALIFIED, 'status fields')
    require(audit['bitwise_verdicts'] == verdict, 'bitwise verdicts re-derived')
    sc = audit['scientific_counters']
    require((sc['scientific_auxiliary_runs'], sc['scientific_main_runs'], sc['scientific_seed_used'],
             sc['seed_42_launched']) == (0, 0, False, False), 'scientific counters')
    require(sorted(audit['artifacts_sha256']) == sorted(set(ARTIFACTS) - {BASE + '_POLICY.json'}), 'artifact list')
    for path, h in audit['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'artifact ' + path)
    final_tests = [t for t in audit['tests'] if t['final']]
    require(final_tests and all(t['failures'] == t['errors'] == 0 for t in final_tests) and
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
        require((row['classification'], row['milestone'], row['method_id'], row['final_status'], row['status']) ==
                (CLASSIFICATION, 'M6D6f', 'E07c', 'PASS', 'PASS'), 'ledger row identity')
        require(row['qualified_statuses'] == QUALIFIED and row['not_qualified'] == NOT_QUALIFIED and
                row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED' and row['fidelity_class'] == 'CONTROLLED_ADAPTATION',
                'ledger statuses')
        require((row['qualification_seed'], row['scientific_auxiliary_runs'], row['scientific_seed_used'],
                 row['authorized_TRAIN_access'], row['VAL_access'], row['TEST_access'], row['bitwise_resume'],
                 row['resume_reference_process'], row['interrupted_process'], row['restore_process'],
                 row['scientific_training'], row['commit'], row['push']) ==
                (QUALIFICATION_SEED, 0, False, True, False, False, True, 'qualification', 'qualification',
                 'qualification', False, False, False), 'ledger fields')
        require(sorted(row['artifacts_sha256']) == sorted(ARTIFACTS), 'ledger artifact list')
        for path, h in row['artifacts_sha256'].items():
            require(sha(read(path)) == h, 'ledger artifact ' + path)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF artifact index')
        worktree(True)
    require('torch' not in sys.modules and 'yaml' not in sys.modules and 'pyarrow' not in sys.modules,
            'static preflight: no torch, yaml or pyarrow')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'artifact_rows_updated': len(MODIFIED),
            'process_sha256': PROCESS_SHA, 'bitwise_verdicts': verdict, 'torch_imported': False,
            'benchmark_files_opened': 0, 'qualification_rerun': False}


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
