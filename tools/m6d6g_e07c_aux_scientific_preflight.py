#!/usr/bin/env python3
"""Verify M6D6g E07c auxiliary SCIENTIFIC run evidence; append-check the ledger; rebuild the index LAST.

STATIC: no Torch, no YAML parser, no pyarrow, no CUDA, no model, no Parquet decode, no image read, and never
opens encoder_final.pkl or a resume sidecar (their SHA256 values are validated as recorded evidence only).
Nothing is retrained. Scientific implementation files must be byte-identical to the authority commit.
Optional --runtime-root (GPU host only) re-hashes the small runtime JSON/log files read-only.
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
AUTHORITY = '7642b23e60a1d89fcd481b03d4c9cb361f4c6e20'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
SEED = 42
RUN_ID = '7014cdb366da52e4'
RUN_UUID = '7031fd2f-92d7-465a-94d1-ab99bf8a3960'
RUN_DIR = '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42'
FINAL_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
FINAL_BYTES = 185136819
PENDING = 'SHA256_RECORDED_PENDING_OWNER_FREEZE'
EPOCHS, STEPS, BATCH, CONSUMED, DROPPED, DENOM = 200, 56, 256, 14336, 131, 14467
IDENTITIES = {'config_sha256': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
              'a1_adaptation_sha256': 'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
              'a3_sha256': 'b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a',
              'a6_overlay_sha256': 'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b',
              'a7_overlay_sha256': '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3',
              'a8_overlay_sha256': 'a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a',
              'a8_document_sha256': 'deb11b5cd80fa873bde1ef9e87160fed3a8065903cd8c81c9463b087fbf73b2f',
              'm6d6e_contract_sha256': '09c0af7e79aeb3acda3e188f7a2a9d94f6d0401485a061d46767ab1c71069ae6',
              'logging_contract_sha256': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
              'environment_lock_sha256': '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450',
              'split_manifest_sha256': 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d',
              'class_map_sha256': 'c7d23e3e3e6a526ae412a50125d0d56379ce98bca99d4133ead3fda7139a3812',
              'source_commit': PIN, 'source_tree': TREE}
# Executed code/config: byte-identical to the authority commit (verified again here by SHA256).
EXECUTED = {'tools/run_e07c_aux.py': 'a9a7bcb509d36f81e892ef31e74cac446fc93d08b3ab35f007046d9181af56d4',
            'methods/difffas/aux_runner.py': 'ab07c90aba0f25b8e49bdb02ab919f28512d369f7bb3c4c964fb52e9692f1f69',
            'methods/difffas/aux_resume.py': '55626c2596165dbf59845270c1d050f23ddb8e1873d4446083398a12ab8d5fcf',
            'methods/difffas/aux_runner_io.py': '297bc344b51a789a0edab209541a77f7515a53354fe7ca26b4fba7a5530d6351',
            'configs/amendments/e07c_a8_aux_resume_policy.yaml': IDENTITIES['a8_overlay_sha256'],
            'configs/amendments/e07c_a7_execution_policy.yaml': IDENTITIES['a7_overlay_sha256'],
            'configs/amendments/e07c_a6_feature_interface_source_correction.yaml': IDENTITIES['a6_overlay_sha256'],
            'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml': IDENTITIES['m6d6e_contract_sha256'],
            'configs/methods/e07c_difffas_bin_idfree.yaml': IDENTITIES['config_sha256'],
            'configs/frozen/difffas_bin_idfree_v1.yaml': IDENTITIES['a1_adaptation_sha256'],
            'configs/run_logging_v1.yaml': IDENTITIES['logging_contract_sha256'],
            'environments/e07c.lock.json': IDENTITIES['environment_lock_sha256'],
            'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md':
                IDENTITIES['a3_sha256'],
            'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A8_E07c_Aux_Resume_Policy.md':
                IDENTITIES['a8_document_sha256']}
# Whole directories that M6D6g must not touch (compared against the authority commit through Git).
FROZEN_TREES = ('methods', 'configs', 'docs', 'environments', 'manifests', 'frozen_config_snapshot', 'gpatbench',
                'third_party', 'tools/run_e07c_aux.py')
DATA_INPUTS = ('manifests/split_v1.parquet', 'manifests/artifact_probe_classes_v1.json',
               'manifests/difffas_bin_idfree_train_v1.parquet')
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 121
INDEX_BASELINE_ROWS = 713
CLASSIFICATION = 'M6D6G_E07C_AUX_SCIENTIFIC_RUN'
BASE = 'outputs/audit/M6D6G_E07C_AUX_'
RUN_JSON, FINAL_JSON = BASE + 'SCIENTIFIC_RUN.json', BASE + 'FINAL_CHECKPOINT.json'
REPORT, LOG = BASE + 'SCIENTIFIC_RUN.md', BASE + 'SCIENTIFIC_RUNTIME_LOG.txt'
TESTS, PREFLIGHT = 'tests/test_m6d6g_e07c_aux_scientific_run.py', 'tools/m6d6g_e07c_aux_scientific_preflight.py'
NEW = tuple(sorted((RUN_JSON, FINAL_JSON, REPORT, LOG, TESTS, PREFLIGHT)))
RUNTIME_SMALL = ('resolved_config.yaml', 'run_manifest.json', 'metrics.jsonl', 'checkpoint_index.json',
                 'resume_state_index.json', 'run_summary.json', 'stdout.log', 'stderr.log')
NEW_STATUSES = ['E07c_AUXILIARY_ENCODER_200_EPOCH_TRAINING_COMPLETED', 'E07c_AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT_PRODUCED',
                'E07c_AUXILIARY_ENCODER_FINAL_SHA_RECORDED', 'FINAL_SHA_PENDING_OWNER_FREEZE']
NOT_QUALIFIED = ['AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', 'MAIN_DIFFFAS_TRAINING_GRAPH', 'MAIN_CHECKPOINT_RESUME',
                 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION', 'MAIN_PRODUCTION_RUNNER', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING',
                 'M8_BANK']
UNINSTRUMENTED = ('TRAIN_physical_face_reads', 'TRAIN_superseded_face_reads')
FORBIDDEN_CLAIMS = ('E07c_AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING_COMPLETED')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6g: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative evidence path')
    require(not {'data', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image read')
    return (ROOT / p).read_bytes()


def load(path):
    return json.loads(read(path))


# ----------------------------------------------------------------- authority / immutability
def authority():
    for ref in ('HEAD', 'origin/m6-baselines'):
        require(git('rev-parse', ref).decode().strip() == AUTHORITY, ref + ' authority')
    require(git('branch', '--show-current').decode().strip() == 'm6-baselines', 'branch')


def scientific_files_unchanged():
    """Every scientific implementation/config/amendment/environment/manifest path is byte-identical to AUTHORITY."""
    require(git('status', '--porcelain', '--untracked-files=all', '--', *FROZEN_TREES).decode().strip() == '',
            'scientific implementation/config/amendment/environment/manifest paths untouched')
    require(git('diff', '--name-only', AUTHORITY, '--', *FROZEN_TREES).decode().strip() == '', 'no diff vs authority')
    for rel, digest in EXECUTED.items():
        require(sha(read(rel)) == digest, 'executed file SHA256 ' + rel)
        require(read(rel) == git('show', f'{AUTHORITY}:{rel}'), 'byte-identical to authority ' + rel)
    require(sha(read('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')) == SPEC_SHA, 'spec SHA256')
    for rel in DATA_INPUTS:     # Git blob identity; the data files themselves are never opened
        require(git('rev-parse', f'HEAD:{rel}').decode().strip() == git('rev-parse', f'{AUTHORITY}:{rel}').decode().strip(),
                'data input blob ' + rel)
    src = ROOT / 'third_party/source_cache/difffas'
    if (src / '.git').exists():
        require(git('rev-parse', 'HEAD', root=src).decode().strip() == PIN, 'pinned source commit')
        require(git('rev-parse', 'HEAD^{tree}', root=src).decode().strip() == TREE, 'pinned source tree')


def checkpoint_semantics_unchanged():
    """Stdlib AST: the whole-module save seam and the source epoch-loss denominator of the executed engine."""
    engine = read('methods/difffas/aux_runner.py').decode()
    fns = {n.name: ast.unparse(n) for n in ast.walk(ast.parse(engine)) if isinstance(n, ast.FunctionDef)}
    require('save_whole_module(self.model, partial, config)' in fns['checkpoint_event'] and
            'state_dict' not in fns['checkpoint_event'], 'whole-module checkpoint seam')
    require('rus = running_loss / len(self.dataset)' in fns['run_epoch'], 'source epoch-loss denominator')
    require("'final_checkpoint_status': 'SHA256_RECORDED_PENDING_OWNER_FREEZE'" in fns['run_scientific'] and
            "'authoritative_for_main_difffas': False" in fns['run_scientific'], 'final status is pending owner freeze')
    ck = read('methods/difffas/aux_checkpoint.py').decode()
    require("FROZEN_FORMAT = 'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())'" in ck,
            'frozen whole-module format')


# ----------------------------------------------------------------- recorded evidence
def check_evidence():
    run, final = load(RUN_JSON), load(FINAL_JSON)
    require((run['milestone'], run['classification'], run['final_status'], run['method_id'], run['method_status'],
             run['fidelity_class'], run['deviation'], run['new_deviation'], run['new_fidelity_class'],
             run['authority_commit'], run['runtime_git_commit'], run['runtime_git_dirty']) ==
            ('M6D6g', CLASSIFICATION, 'PASS', 'E07c', 'IMPLEMENTED_NOT_EXECUTED', 'CONTROLLED_ADAPTATION', 'DEV-021',
             False, False, AUTHORITY, AUTHORITY, False), 'run evidence identity')
    require(run['identities'] == IDENTITIES, 'runtime identities (config/A1/A3/A6/A7/A8/lock/data/source)')
    ri = run['run_identity']
    require((ri['run_id'], ri['run_uuid'], ri['run_dir'], ri['runner_mode'], ri['scientific_run'],
             ri['auxiliary_encoder_training_seed'], ri['experiment_seed'], ri['completion_status']) ==
            (RUN_ID, RUN_UUID, RUN_DIR, 'SCIENTIFIC', True, SEED, None, 'completed'), 'logical run identity')
    require(RUN_DIR.endswith('/runs/m6/E07c/aux_encoder/seed_42'), 'seed_42 auxiliary root')
    la = run['launch']
    require(la['launches'] == 1 and la['exit_status'] == 0 and la['resume_state_argument'] is None and
            la['environment'] == {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0',
                                  'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1',
                                  'PYTHONNOUSERSITE': '1'} and
            la['command_line'].endswith('tools/run_e07c_aux.py --seed 42 --execution-config configs/execution/m5_gpu_3090.yaml'),
            'one launch with the frozen command')
    require(len(run['process_sessions']) == 1 and run['process_sessions'][0]['kind'] == 'FRESH' and
            run['interruptions'] == [] and run['resume_reconciliations'] == [], 'one process session, no resume')
    e0 = run['epoch0_boundary']
    require((e0['epoch'], e0['global_step']) == (0, 0), 'epoch-0 boundary')
    a = run['step_accounting']
    require((a['logical_authoritative_optimizer_steps'], a['physical_optimizer_steps_executed'],
             a['superseded_optimizer_steps'], a['process_sessions'], a['completed_epochs'],
             a['epochs_with_56_logical_steps'], a['optimizer_applications'], a['uncommitted_in_progress_steps']) ==
            (EPOCHS * STEPS, EPOCHS * STEPS, 0, 1, EPOCHS, EPOCHS, EPOCHS * STEPS, 0), 'step accounting')
    require(a['committed_boundary'] == {'completed_epoch': EPOCHS, 'global_step': EPOCHS * STEPS,
                                        'sidecar': 'checkpoints/resume/runner_state_epoch_200.pth'}, 'final boundary')
    mv = run['metrics_verification']
    require((mv['physical_step_records'], mv['logical_step_records'], mv['logical_global_steps_contiguous'],
             mv['logical_batch_sizes'], mv['logical_steps_per_epoch_distinct'], mv['iterations_ok'], mv['learning_rates'],
             mv['finite_losses'], mv['val_losses_all_null'], mv['batch_sample_hash_count'], mv['metrics_ends_with_newline']) ==
            (11200, 11200, True, [BATCH], [STEPS], True, [0.002], True, True, 11200, True), 'metrics verification')
    require(mv['event_counts'] == {'e07c_aux_checkpoint_event': EPOCHS, 'e07c_aux_epoch_complete': EPOCHS,
                                   'e07c_aux_resume_state_committed': EPOCHS + 1, 'e07c_aux_run_start': 1},
            'event counts (no reconciliation record)')
    eps = run['epochs']
    require([e['epoch'] for e in eps] == list(range(1, EPOCHS + 1)), '200 logical epochs')
    for e in eps:
        require((e['optimizer_steps'], e['global_step_end'], e['consumed'], e['dropped']) ==
                (STEPS, e['epoch'] * STEPS, CONSUMED, DROPPED), f"epoch {e['epoch']} accounting")
        require(float.fromhex(e['source_epoch_loss_hex']) == e['source_epoch_loss'] and
                0 < e['source_epoch_loss'] < float('inf'), f"epoch {e['epoch']} finite loss")
        require(e['checkpoint_type'] == ('selected' if e['epoch'] == EPOCHS else 'periodic'), 'checkpoint type per epoch')
    require(len({e['epoch_sample_order_sha256'] for e in eps}) == EPOCHS, 'distinct shuffled epoch orders')
    require(eps[-1]['checkpoint_sha256'] == FINAL_SHA, 'epoch-200 checkpoint event SHA256')
    require('14467' in run['epoch_loss_semantics'], 'source-native denominator')
    c = run['contract']
    require((c['epochs'], c['batch_size'], c['shuffle'], c['num_workers'], c['drop_last'], c['steps_per_epoch'],
             c['consumed_per_epoch'], c['dropped_per_epoch'], c['scheduler']) ==
            (EPOCHS, BATCH, True, 6, True, STEPS, CONSUMED, DROPPED, 'NONE'), 'frozen contract')
    p = run['precision']
    require((p['matmul_tf32'], p['cudnn_tf32'], p['cudnn_benchmark'], p['cudnn_deterministic'], p['amp'],
             p['grad_scaler'], p['autocast_cuda'], p['default_dtype']) ==
            (False, False, False, True, False, False, False, 'torch.float32'), 'A7 precision')
    require(run['seeding']['seed'] == SEED and run['seeding']['pythonhashseed'] == '42', 'seed 42')
    pop = run['population']
    require((pop['train_rows_materialized'], pop['val_rows_materialized'], pop['test_rows_materialized'],
             pop['main_relation_opened'], pop['sha256']) == (DENOM, 0, 0, False, IDENTITIES['split_manifest_sha256']),
            'TRAIN-only population')
    d = run['data_access']
    require((d['TRAIN_rows_exposed'], d['TRAIN_logical_consumption'], d['superseded_optimizer_steps'],
             d['VAL_rows_used'], d['VAL_image_reads'], d['TEST_rows_used'], d['TEST_image_reads'],
             d['raw_benchmark_image_reads'], d['main_8838_row_relation_used'], d['val_split_accessed'],
             d['test_split_accessed']) ==
            (DENOM, EPOCHS * STEPS * BATCH, 0, 0, 0, 0, 0, 0, 0, False, False), 'data access')
    # Face reads were not instrumented in the production process: explicit null with reason (run_logging_v1).
    for field in UNINSTRUMENTED:
        require(d[field] is None and 'not independently instrumented' in d['missing_field_reasons'][field],
                field + ' is null with a not-instrumented reason')
    require('NOT a filesystem/image-read count' in d['TRAIN_logical_consumption_basis'] and
            d['non_train_zero_basis'].startswith('STRUCTURAL, not audit-hook instrumented') and
            'not a face-image read' in d['manifest_read_disclosure'], 'logical vs structural vs manifest disclosures')
    ce = run['checkpoint_events']
    require((ce['count'], ce['periodic_not_selected'], ce['selected_final_epoch']) == (EPOCHS, EPOCHS - 1, EPOCHS),
            'only the epoch-200 bytes are selected (no VAL/TEST selection)')
    sc = run['scientific_counters']
    require(sc == {'scientific_auxiliary_logical_runs': 1, 'scientific_auxiliary_seed': SEED,
                   'scientific_auxiliary_process_sessions': 1, 'main_difffas_scientific_runs': 0,
                   'val_training_or_selection': 0, 'test_training_or_selection': 0, 'M8_bank': False}, 'scientific counters')
    require(run['qualified_statuses'] == NEW_STATUSES and run['not_qualified'] == NOT_QUALIFIED and
            run['not_claimed'] == list(FORBIDDEN_CLAIMS), 'statuses')
    fc = run['final_checkpoint']
    require((fc['sha256'], fc['file_size_bytes'], fc['epoch'], fc['global_step'], fc['checkpoint_type'],
             fc['selected_for_final'], fc['final_checkpoint_status'], fc['authoritative_for_main_difffas']) ==
            (FINAL_SHA, FINAL_BYTES, EPOCHS, EPOCHS * STEPS, 'selected', True, PENDING, False), 'run final checkpoint')
    # final checkpoint record
    require((final['sha256'], final['file_size_bytes'], final['epoch'], final['global_step'], final['checkpoint_type'],
             final['selected_for_final'], final['final_checkpoint_status'], final['authoritative_for_main_difffas'],
             final['owner_freeze_performed'], final['deserialized_for_evidence'], final['val_or_test_selection'],
             final['run_id'], final['run_uuid'], final['process_sessions'], final['logical_optimizer_steps'],
             final['physical_optimizer_steps'], final['superseded_optimizer_steps'], final['authority_commit']) ==
            (FINAL_SHA, FINAL_BYTES, EPOCHS, EPOCHS * STEPS, 'selected', True, PENDING, False, False, False, False,
             RUN_ID, RUN_UUID, 1, 11200, 11200, 0, AUTHORITY), 'final checkpoint record')
    require(final['absolute_path'] == RUN_DIR + '/checkpoints/encoder_final.pkl' and
            final['relative_path'] == 'checkpoints/encoder_final.pkl', 'final checkpoint path')
    require(set(final['sha256_sources_agree'].values()) == {FINAL_SHA} and len(final['sha256_sources_agree']) == 5,
            'independent SHA256 = run_summary = checkpoint_index = A8 index = epoch-200 event')
    require(final['selection_reason'].startswith('FINAL_STATE_AFTER_EPOCH_200') and
            final['format'].startswith('torch.save of the WHOLE nn.Module'), 'selection rule / whole-module format')
    require(final['final_source_native_epoch_loss'] == eps[-1]['source_epoch_loss'], 'final loss consistent')
    # runtime file hashes: JSON evidence == runtime log
    rf = run['runtime_files']
    require(set(RUNTIME_SMALL) <= set(rf) and rf['checkpoints/encoder_final.pkl'] ==
            {'size_bytes': FINAL_BYTES, 'sha256': FINAL_SHA}, 'runtime file inventory')
    require(rf['checkpoints/resume/runner_state_epoch_200.pth']['sha256'] == run['resume_sidecars']['committed_sha256'],
            'committed sidecar SHA256')
    log = read(LOG).decode()
    for name, v in rf.items():
        require(f"{v['sha256']}  {v['size_bytes']:>10}  {name}" in log, 'runtime log hash line ' + name)
    for token in ('ANCESTOR_OK', 'ROOT_ABSENT', 'NO_TMUX', 'NO_PROC', '"torch_imported": false', '"resume": null',
                  'EXIT=0', 'M6D6G_EXIT=0', '(no --resume-state)', FINAL_SHA, 'Not a training interruption'):
        require(token in log, 'runtime log token ' + token)
    report = read(REPORT).decode()
    for token in (FINAL_SHA, PENDING, 'CONTROLLED_ADAPTATION', 'DEV-021', 'IMPLEMENTED_NOT_EXECUTED', 'M6D6h',
                  '**Not claimed:**', '11 200', '| TRAIN physical face reads | `null`: not independently instrumented'):
        require(token in report, 'report token ' + token)
    require('derived: one session' not in report, 'no derived physical face-read count in the report')
    for claim in FORBIDDEN_CLAIMS:
        require(report.count(claim) == 1 and report.split('**Not claimed:**')[1].find(claim) >= 0,
                'forbidden claim appears only as not-claimed ' + claim)
    return run, final


def runtime_rehash(runtime_run_dir, run):
    """GPU host only: re-hash the small runtime files read-only (never the .pkl or the sidecar)."""
    base = Path(runtime_run_dir)
    out = {}
    for name in RUNTIME_SMALL:
        raw = (base / name).read_bytes()
        require({'size_bytes': len(raw), 'sha256': sha(raw)} == run['runtime_files'][name], 'runtime file ' + name)
        out[name] = sha(raw)
    summary = json.loads((base / 'run_summary.json').read_text())
    require(summary['final_or_selected_checkpoint_sha256'] == FINAL_SHA and summary['final_checkpoint_status'] == PENDING,
            'runtime run_summary')
    return out


# ----------------------------------------------------------------- ledger / index / worktree
def expected_index():
    reader = csv.DictReader(io.StringIO(git('show', AUTHORITY + ':' + INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
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
    expected = ['?? ' + p for p in NEW]
    if ledger_and_index_modified:
        expected += [' M ' + INDEX, ' M ' + LEDGER]
    require(status == sorted(expected), 'worktree holds exactly the intended M6D6g changes: ' + json.dumps(status))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def check_ledger_row(row):
    require((row['milestone'], row['classification'], row['method_id'], row['status'], row['final_status']) ==
            ('M6D6g', CLASSIFICATION, 'E07c', 'PASS', 'PASS'), 'ledger row identity')
    require((row['scientific_auxiliary_logical_runs'], row['scientific_auxiliary_seed'],
             row['auxiliary_scientific_training_completed'], row['completed_epochs'], row['logical_optimizer_steps'],
             row['physical_optimizer_steps'], row['superseded_optimizer_steps'],
             row['auxiliary_scientific_checkpoint_created'], row['final_checkpoint_sha256'],
             row['final_checkpoint_sha_status'], row['auxiliary_checkpoint_authoritative_for_main'],
             row['main_difffas_scientific_runs'], row['M8_bank'], row['VAL_access'], row['TEST_access'],
             row['commit'], row['push']) ==
            (1, SEED, True, EPOCHS, 11200, 11200, 0, True, FINAL_SHA, PENDING, False, 0, False, False, False,
             False, False), 'ledger fields')
    require(not any('FROZEN' in k.upper() and 'SHA' in k.upper() for k in row), 'no owner-freeze SHA field')
    require(row['TRAIN_logical_consumption'] == EPOCHS * STEPS * BATCH and
            all(row[f] is None and 'not independently instrumented' in row['missing_field_reasons'][f]
                for f in UNINSTRUMENTED), 'ledger TRAIN read accounting (logical numeric; face reads null + reason)')
    require(row['qualified_statuses'] == NEW_STATUSES and row['not_qualified'] == NOT_QUALIFIED and
            row['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'ledger statuses')
    require(sorted(row['artifacts_sha256']) == sorted(NEW), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)


def verify(stage, runtime_run_dir=None):
    """stage: 'before_ledger' (evidence only), 'before_index' (ledger appended), 'final'."""
    authority()
    scientific_files_unchanged()
    checkpoint_semantics_unchanged()
    run, _ = check_evidence()
    rehashed = runtime_rehash(runtime_run_dir, run) if runtime_run_dir else None
    prefix = git('show', AUTHORITY + ':' + LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]))
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF artifact index')
        worktree(True)
    require(not {'torch', 'yaml', 'pyarrow'} & set(sys.modules), 'static preflight: no torch, yaml or pyarrow')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'final_checkpoint_sha256': FINAL_SHA,
            'final_checkpoint_status': PENDING, 'runtime_rehash': rehashed, 'torch_imported': False,
            'checkpoint_bytes_opened': 0, 'training_rerun': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true', help='evidence-only check before the ledger append')
    parser.add_argument('--rebuild-index', action='store_true', help='after the ledger append: write the index LAST')
    parser.add_argument('--runtime-run-dir', help='GPU host only: re-hash the small runtime files read-only')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger', args.runtime_run_dir)
    elif args.rebuild_index:
        verify('before_index', args.runtime_run_dir)
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final', args.runtime_run_dir)
    else:
        result = verify('final', args.runtime_run_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
