#!/usr/bin/env python3
"""Verify the M6D6j E07c MAIN production-runner QUALIFICATION candidate; ledger/index LAST.

STATIC: stdlib only (no Torch, YAML, numpy, PIL or pyarrow); never opens a weight file, a manifest, an image or a runtime
path. Candidate-time checks (authority, branch, exact worktree, every tracked file vs the M6D6iR authority) run only
here, once. check_evidence(reader) takes a reader so tests validate the evidence at the M6D6j commit, not a moving HEAD.

  --before-ledger   evidence + candidate checks, ledger not yet appended
  --rebuild-index   after the one-row ledger append: rebuild ARTIFACT_INDEX.csv LAST (CRLF, sorted)
  (default)         final check
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
AUTHORITY = '39508719a57e5afde0a8c71bf7db0f9f09ec25e0'
BRANCH = 'm6-baselines'
MILESTONE = 'M6D6j'
CLASSIFICATION = 'M6D6J_E07C_MAIN_PRODUCTION_RUNNER_QUALIFICATION'
SEED = 60608
FROZEN_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 125
INDEX_BASELINE_ROWS = 741
CONTRACT = 'configs/amendments/e07c_m6d6j_main_production_runner_contract.yaml'
CODE = ('methods/difffas/main_runner_io.py', 'methods/difffas/main_checkpoint.py', 'methods/difffas/main_runner.py',
        'methods/difffas/main_runner_qualification.py', 'tools/run_e07c_main.py')
TESTS = 'tests/test_m6d6j_e07c_main_runner.py'
PREFLIGHT = 'tools/m6d6j_e07c_main_runner_preflight.py'
EV_JSON = 'outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.json'
EV_MD = 'outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.md'
LOG = 'outputs/audit/M6D6J_E07C_RUNTIME_LOG.txt'
NEW = tuple(sorted((CONTRACT, *CODE, TESTS, PREFLIGHT, EV_JSON, EV_MD, LOG)))
BOOKKEEPING = (INDEX, LEDGER)
PROTECTED = ('methods/difffas/main_graph.py', 'methods/difffas/main_graph_qualification.py',
             'methods/difffas/execution_policy.py', 'methods/difffas/aux_checkpoint.py', 'methods/difffas/aux_runner.py',
             'methods/difffas/aux_runner_io.py', 'methods/difffas/aux_resume.py', 'methods/difffas/encoder.py',
             'methods/difffas/source.py', 'methods/difffas/contract.py', 'methods/difffas/seed_adapter.py',
             'methods/difffas/runtime_qualification.py', 'methods/difffas/__init__.py', 'tools/run_e07c_aux.py',
             'methods/common/runlog.py', 'methods/common/learned_runlog.py', 'methods/common/learned.py',
             'methods/common/upstream.py', 'methods/common/config.py', 'configs/methods/e07c_difffas_bin_idfree.yaml',
             'configs/frozen/difffas_bin_idfree_v1.yaml', 'configs/run_logging_v1.yaml',
             'configs/amendments/e07c_a7_execution_policy.yaml', 'configs/amendments/e07c_a8_aux_resume_policy.yaml',
             'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml',
             'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml', 'configs/execution/m5_gpu_3090.yaml',
             'environments/e07c.lock.json', 'tests/test_m6d6i_e07c_main_graph.py',
             'tests/test_m6d6ir_e07c_checkpoint_retention.py', 'tools/m6d6ir_e07c_checkpoint_retention_preflight.py',
             'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx')
QUALIFIED = ['MAIN_PRODUCTION_RUNNER', 'A1_TRAIN_DATASET_INTEGRATION', 'CANONICAL_FACE_READER', 'GUIDE_MAPPING',
             'RUN_LOGGING_V1_MAIN_RUN', 'MAIN_CHECKPOINT_CADENCE', 'MAIN_TERMINAL_CHECKPOINT_CREATION',
             'MAIN_CHECKPOINT_RETENTION_RUNTIME_INTEGRATION']
QUALIFIED_ALIASES = ['E07c_' + s for s in QUALIFIED]
NOT_QUALIFIED = ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
COUNTERS = {'backward': 2, 'zero_grad': 2, 'optimizer_step': 2, 'scheduler_step': 2, 'torch_save': 3,
            'torch_load_seam': 1, 'torch_load_roundtrip': 1, 'autocast_entries': 0}
STAGES = ['seed', 'precision', 'transform', 'dataset', 'dataloader', 'objects', 'encoder_eval']
SCIENTIFIC_ZERO = {'scientific_main_runs': 0, 'scientific_optimizer_steps': 0, 'experiment_seed_runs': 0,
                   'scientific_checkpoint_writes': 0}
LOG_TOKENS = ('M6D6J_GPU_HEAD=' + AUTHORITY, 'M6D6J_FF_ONLY=', 'M6D6J_QUALIFICATION_EXIT=0',
              'M6D6J_CLI_REFUSALS_OK', 'M6D6J_GPU_EXIT=0', f'{FROZEN_SHA}  ')


def require(ok, message):
    if not ok:
        raise ValueError('M6D6j: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/manifest/image')
    return (ROOT / p).read_bytes()


def worktree_reader(rel):
    return read(rel)


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- evidence (reader-scoped; no HEAD lock)
def check_evidence(reader=worktree_reader):
    ev = json.loads(reader(EV_JSON))
    require((ev['status'], ev['milestone'], ev['qualification_seed'], ev['experiment_seed'], ev['experiment_seed_reason'],
             ev['method_status'], ev['fidelity_class'], ev['deviation'], ev['TRAIN_access'], ev['VAL_access'],
             ev['TEST_access'], ev['M8_outputs'], ev['MAIN_CHECKPOINT_RESUME'],
             ev['FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION']) ==
            ('PASS', MILESTONE, SEED, None, 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN', 'IMPLEMENTED_NOT_EXECUTED',
             'CONTROLLED_ADAPTATION', 'DEV-021', True, False, False, 0, 'UNQUALIFIED', True), 'evidence identity')
    require(ev['candidate_sha256'] == {rel: sha(reader(rel)) for rel in (CONTRACT, *CODE)}, 'evidence ran the candidate')
    require(ev['owner_decisions'] == {'D1_visualization': 'EXECUTE_EXACT_SOURCE',
                                      'D2_terminal_position': 'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION'}, 'D1/D2')
    require(ev['qualified_statuses'] == QUALIFIED and ev['not_qualified'] == NOT_QUALIFIED, 'statuses')
    require(ev['counters'] == COUNTERS and ev['scientific_counters'] == SCIENTIFIC_ZERO and ev['stages'] == STAGES,
            'counters / A7 stages')
    require(re.fullmatch(r'/.+/qualification/m6d6j/E07c/q60608-[0-9a-f]{16}', ev['run_root']) is not None and
            '/runs/' not in ev['run_root'], 'non-scientific qualification root')
    require(ev['frozen_encoder']['read_opens'] == 2 and [e['sha256'] for e in ev['frozen_encoder']['sha256_events']] ==
            [FROZEN_SHA], 'frozen encoder through the seam')
    b4, b2 = ev['b4_step'], ev['b2_tail_step']
    require(all(s == [4, 3, 256, 256] for s in b4['batch_shapes'].values()) and b4['global_step'] == 1 and
            all(s == [2, 3, 256, 256] for s in b2['batch_shapes'].values()) and b2['global_step'] == 2 and
            b4['encoder_unchanged'] and b2['encoder_unchanged'] and b4['ema_equals_main'] and b2['ema_equals_main'],
            'real B=4 and B=2 steps')
    v = ev['visualization']
    require((v['model_forwards'], v['encoder_calls'], v['cpu_rng_changed'], v['cuda_rng_changed'], v['rng_restored'],
             v['model_mode'], sorted(v['unchanged'])) == (500, 2, True, True, False, 'train',
             sorted(['model_parameters', 'ema_parameters', 'ema_buffers', 'optimizer', 'scheduler', 'encoder'])) and
            len(v['items']) == 4, 'visualization executed with source effects')
    c = ev['checkpoint_lifecycle']
    idx = c['index']
    require([e['global_step'] for e in idx] == [10000, 20000, 884000] and
            [e['checkpoint_type'] for e in idx] == ['periodic', 'periodic', 'terminal'] and
            [e['bytes_pruned'] for e in idx] == [True, True, False] and all(e['index_verified'] for e in idx) and
            idx[0]['successor_checkpoint_sha256'] == idx[1]['sha256'] and
            idx[1]['successor_checkpoint_sha256'] == idx[2]['sha256'] and idx[2]['selected_for_final'] is True and
            idx[2]['selection_reason'] == 'BASELINE_FINAL_STATE_V1' and
            idx[2]['logical_roles'] == ['terminal', 'selected', 'authoritative_final', 'officially_required'] and
            c['protected_refusal'] and c['roundtrip_first_file']['keys'] == ['model', 'ema', 'scheduler', 'optimizer',
                                                                            'conf'] and c['torch_save_calls'] == 3 and
            c['bytes_remaining_after_cleanup'] == 0, 'checkpoint lifecycle')
    r = ev['reads']
    require(r['physical_face_opens'] == {'b4_step': 12, 'b2_tail_step': 6, 'visualization': 12} and
            (r['val_ids'], r['test_ids'], r['prefetch_reads'], r['num_workers']) == (0, 0, 0, 0), 'read accounting')
    require(not ev['firewall']['denied'], 'firewall denials')
    lg = ev['run_logging']
    require(lg['step_records'] == 2 and lg['run_summary']['test_split_accessed'] is False and
            {'resolved_config.yaml', 'run_manifest.json', 'metrics.jsonl', 'checkpoint_index.json', 'run_summary.json',
             'stdout.log', 'stderr.log'} <= set(lg['files']) and
            any(f.startswith('diagnostics/visualization/') for f in lg['files']), 'run_logging_v1 outputs')
    report = reader(EV_MD).decode()
    for token in (*QUALIFIED, *NOT_QUALIFIED, 'QUALIFICATION ONLY', '60608', 'EXECUTE_EXACT_SOURCE',
                  'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION', 'FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION',
                  'test_24_cli_accepts_only_seed42', 'IMPLEMENTED_NOT_EXECUTED', 'DEV-021', 'M6D6k'):
        require(token in report, 'report token ' + token)
    log = reader(LOG).decode()
    for token in LOG_TOKENS:
        require(token in log, 'runtime log token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6iR authority')


def tracked_unchanged(expected_changed):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted(expected_changed), 'only ledger/index may differ from the authority: ' + json.dumps(changed))
    for rel in PROTECTED:
        require(read(rel) == at_authority(rel), 'protected byte-identical ' + rel)
    for rel in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').decode().splitlines():
        if Path(rel).name.startswith(tuple(f'M6D6{c}_' for c in 'ABCDEFGHI') + ('M6D6IR_',)):
            require(read(rel) == at_authority(rel), 'historical evidence byte-identical ' + rel)
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '',
            'no tracked weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in NEW] + [' M ' + p for p in bookkeeping]
    require(status == sorted(expected), 'worktree holds exactly the M6D6j candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no whitespace before line endings ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
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


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'method_id': 'E07c', 'status': 'PASS',
    'final_status': 'PASS', 'qualification_seed': SEED, 'experiment_seed': None, 'experiment_seed_runs': 0,
    'scientific_main_runs': 0, 'TRAIN_access': True, 'VAL_access': False, 'TEST_access': False,
    'D1_visualization_owner_decision': 'EXECUTE_EXACT_SOURCE',
    'D2_terminal_position': 'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION', 'visualization_source_cadence': 1000,
    'visualization_executed_in_qualification': True, 'checkpoint_retention_policy': 'M6D6iR',
    'MAIN_CHECKPOINT_RESUME': 'UNQUALIFIED', 'M8_bank': False, 'fidelity_class': 'CONTROLLED_ADAPTATION',
    'deviation': 'DEV-021', 'new_deviation': False, 'new_fidelity_class': False, 'commit': False, 'push': False,
    'amendment_created': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'qualified_statuses': QUALIFIED + QUALIFIED_ALIASES, 'not_qualified': NOT_QUALIFIED,
    'method_status': 'IMPLEMENTED_NOT_EXECUTED', 'FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION': True,
    'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(row['committed_prefix_sha256'] == sha(prefix), 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted(NEW), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)


def verify(stage):
    authority()
    changed = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(changed)
    whitespace()
    check_evidence()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
    worktree(changed)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'new_files': list(NEW),
            'modified_existing_files': changed, 'torch_imported': False, 'checkpoint_bytes_opened': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--rebuild-index', action='store_true')
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
