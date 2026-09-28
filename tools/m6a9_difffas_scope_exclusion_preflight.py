#!/usr/bin/env python3
"""Verify the M6A9 candidate: Amendment A9, DiffFAS-family (E07c, E07b) resource-constrained scope exclusion.

A9 is a standalone owner scope decision. It is NOT the M6 closure: M6 stays OPEN (M6_CLOSED = false), E06b DSDG-NATIVE
stays ACTIVE M6 work, and outputs/audit/method_status.csv is NOT created here.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no checkpoint, no model.
Candidate-time checks run only here; check_record / derive / check_evidence take a reader (no moving-HEAD lock).

  --write-evidence-json   candidate creation only
  --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '632e274bde083fd175dc6a2645542d281a767063'
ATTEMPT1_AUTHORITY = '8358d8b6fe468478ad86a715616becb81bb9b339'
BRANCH = 'm6-baselines'
MILESTONE = 'M6A9'
CLASSIFICATION = 'M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION'
AMENDMENT_ID = 'A9_DIFFFAS_FAMILY_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION'
EXCLUSION_CLASS = 'OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION'
REASON = 'OWNER_EXCLUDED_RESOURCE_CONSTRAINT'
RECORD_KIND = 'AMENDMENT / OWNER_SCOPE_DECISION'
STAGE_STATE = 'outputs/audit/STAGE_STATE.json'
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 128
LEDGER_PREFIX_SHA = '23dab8c384eeb9c385d83fdbe13cdd683ed3665e68e6077bfa41f6869cb98075'
INDEX_BASELINE_ROWS = 763
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
RECORD = 'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'
RECORD_SHA = '5ddd59f3560eea63471dd5bc1b4f65281e8f529a74a1e9d4d990ddbb427f3644'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A9_DiffFAS_Family_Resource_Constrained_Scope_Exclusion.md'
EV_JSON = 'outputs/audit/M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION.json'
EV_MD = 'outputs/audit/M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION.md'
TESTS = 'tests/test_m6a9_difffas_scope_exclusion.py'
PREFLIGHT = 'tools/m6a9_difffas_scope_exclusion_preflight.py'
NEW = tuple(sorted((RECORD, DOC, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
METHOD_STATUS = 'outputs/audit/method_status.csv'
M6E_CLOSURE_ARTIFACTS = (METHOD_STATUS, 'outputs/audit/M6E_BASELINE_CLOSURE.json', 'outputs/audit/M6E_BASELINE_CLOSURE.md',
                         'tools/m6e_baseline_closure_preflight.py', 'tests/test_m6e_baseline_closure.py')
A1 = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md'
FAIR_TRACK = 'configs/frozen/fair_track_v1.yaml'
DEVIATIONS = 'outputs/audit/deviation_report.md'
TRACK_A_MATRIX = 'outputs/audit/M4_TRACK_A_METHOD_MATRIX.md'
JR_RECORD = 'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml'
JF_RECORD = 'configs/amendments/e07c_m6d6jf_scientific_context_hotfix.yaml'
IR_RECORD = 'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml'
JF_EVIDENCE = 'outputs/audit/M6D6JF_E07C_SCIENTIFIC_CONTEXT_HOTFIX.json'
METHOD_CONFIGS = {'E01': 'configs/methods/e01_fas_aug.yaml', 'E02': 'configs/methods/e02_freqsub.yaml',
                  'E03': 'configs/methods/e03_stdn.yaml', 'E04': 'configs/methods/e04_physics_std.yaml',
                  'E05': 'configs/methods/e05_pcgan.yaml', 'E06c': 'configs/methods/e06c_dsdg_bin_idfree.yaml',
                  'E07c': 'configs/methods/e07c_difffas_bin_idfree.yaml'}
GPAT_B0 = 'configs/methods/gpat_b0.yaml'
GPAT_ABSENT = ('configs/methods/gpat_b1.yaml', 'configs/methods/gpat_b2.yaml', 'configs/methods/gpat_b3.yaml')
BOUND = {SPEC: SPEC_SHA,
         A1: '03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472',
         FAIR_TRACK: 'd9b02214d1e4701fe7d53972e94be0c2e68c6199a3378cacd69f2c1c87826d9d',
         METHOD_CONFIGS['E01']: '87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0',
         METHOD_CONFIGS['E02']: 'fdf74776cc100d5b5c1ed5f58b84ebb0dfd534782fbb7fc34382919d38d67bf0',
         METHOD_CONFIGS['E03']: '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0',
         METHOD_CONFIGS['E04']: '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a',
         METHOD_CONFIGS['E05']: '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de',
         METHOD_CONFIGS['E06c']: '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002',
         METHOD_CONFIGS['E07c']: 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
         GPAT_B0: '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
         'configs/run_logging_v1.yaml': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
         IR_RECORD: '94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6',
         JR_RECORD: '2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830',
         JF_RECORD: '946ca134ef514650002f368a8d87715f16a80855d10725587d9e6e855747af13',
         JF_EVIDENCE: '2180203404bcb83885b4f26de2f67e220751bae58351f4cd6be0d438dca1ca4e',
         DEVIATIONS: '1a5995dcaf9c764f37fa2646a4af2d520a94762e502338a0dc9370df22049ee0',
         STAGE_STATE: 'a8abf8288feaba561916b7ffe513014cbe2b5831be86d048702f2718059dd50d'}
CONFIG_STATUS_AUTHORITY_SHA = '0bfe711ad973124d068066895c34c7a62653da7401f98de6f0fc0ec431874f04'
PARTIAL_ATTEMPT = {
    'attempt': 2, 'authority': AUTHORITY, 'seed': 42, 'runner_mode': 'SCIENTIFIC', 'run_id': '394e17cccba0954d',
    'runtime_root': '/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/seed_42',
    'start_utc': '2026-09-28T06:33:06Z', 'end_utc': '2026-09-28T17:56:18Z', 'interruption': 'OWNER_MANUAL_INTERRUPT',
    'traceback_terminal': 'KeyboardInterrupt during train_iteration', 'oom': False, 'model_error': False,
    'last_completed_global_step': 54836, 'epoch': 25, 'iteration_in_epoch': 1795,
    'wall_clock_seconds_at_last_completed_step': 40987.14273164095, 'total_iterations_contract': 884000,
    'metrics_jsonl_lines': 54900, 'checkpoint_index_records': 5, 'bytes_present_checkpoints': 1,
    'visualizations': 54, 'run_root_footprint': '~2.5 GiB',
    'run_summary': {'completion_status': 'failed', 'failure_reason': 'KeyboardInterrupt: ',
                    'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
                    'seed_level_evaluation_metrics': None, 'test_split_accessed': False, 'runner_mode': 'SCIENTIFIC'},
    'historical_logging_inconsistency': ('run_summary.failure_reason contains KeyboardInterrupt while '
                                         'missing_field_reasons.failure_reason says the caller did not supply it; '
                                         'metadata only; not rewritten; no hotfix milestone'),
    'classification': 'PARTIAL_OWNER_INTERRUPTED_SCIENTIFIC_ATTEMPT', 'completed_seed': False,
    'benchmark_result': False, 'provenance': 'OWNER_REPORTED_GPU_HOST_OBSERVATION', 'reverified_by_m6e': False}
PARTIAL_CHECKPOINT = {
    'path': 'checkpoints/model_050000.pt', 'global_step': 50000, 'epoch': 22, 'file_size_bytes': 2475667461,
    'sha256': 'b4ae6325bd04fdf615db24ca0cfb883680401c494b9c9d39f4830dbc8d775cae', 'checkpoint_type': 'periodic',
    'selected_for_final': False, 'selection_reason': 'Official cadence; not final selection', 'bytes_present': True,
    'bytes_pruned': False, 'physical_footprint': '~2.4 GiB', 'classification': 'PARTIAL_FAILED_RUN',
    'science_eligibility': 'NOT_ELIGIBLE_FOR_SCIENCE', 'selected': False, 'terminal': False, 'final': False,
    'eligible_for_M8': False, 'eligible_for_downstream_evaluation': False, 'eligible_for_paper_numeric_reporting': False,
    'retention_decision': 'RETAIN', 'new_pruning_permission': False,
    'm6d6ir_successor_gate_applies': True, 'successor_exists': False}
FAILED_ATTEMPT = {'attempt': 1, 'authority': ATTEMPT1_AUTHORITY, 'seed': 42, 'failure_phase': 'RUN_CONTEXT_OPEN',
                  'error': "dict.update() got multiple values for keyword argument 'experiment_seed'",
                  'scientific_optimizer_steps': 0, 'scientific_checkpoint_writes': 0, 'documented_by': 'M6D6jF'}
SMOKE = {'smoke_test_required': True, 'smoke_test_completed': False, 'smoke_test_waived': False,
         'm6_closure_requirement': False, 'blocks_M7_generator_training': False, 'blocks_full_M8_plus_execution': True,
         'source': 'M6D6jR O3 STAGED_MILESTONE_PRECEDENCE_CLARIFICATION (unchanged)'}
ZERO = {'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'banks_generated': 0,
        'model_executions': 0, 'TRAIN_image_reads': 0, 'VAL_image_reads': 0, 'TEST_image_reads': 0,
        'GPU_contacted': False, 'GPU_host_runtime_files_modified': False}


E06B = {'method_name': 'DSDG-NATIVE', 'track': 'B_NATIVE_FULL_SECONDARY', 'in_a9_scope': False, 'owner_excluded': False,
        'resource_blocked': False, 'status': 'ACTIVE_M6_WORK',
        'basis': 'A1 section 3 deferred Track B to M6; no owner exclusion decision', 'implemented': False,
        'fidelity_assessed': False}
E07B = {'method_name': 'DiffFAS-NATIVE', 'track': 'B_NATIVE_FULL_SECONDARY', 'm6_gate_status': 'blocked',
        'blocker_or_exclusion_reason': REASON, 'exclusion_basis': 'DiffFAS family discontinued',
        'fidelity_class': 'NOT_ASSESSED_NOT_IMPLEMENTED', 'fidelity_assessed': False, 'implemented': False,
        'eligible_for_future_bank': False, 'deleted': False, 'a1_text_altered': False, 'native_manifest_created': False,
        'identity_fabricated': False, 'source_gap_claimed': False, 'downstream_training': False,
        'downstream_evaluation': False, 'numeric_result_reporting': False}
M6 = {'M6_CLOSED': False, 'M6_status': 'OPEN', 'method_status_csv_created': False,
      'method_status_csv_created_at': 'M6E closure, after E06b M6 work is resolved',
      'baseline_full_scientific_execution_complete': False, 'provisional_m6e_closure_artifacts_revived': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6A9: ' + message)


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


def spec_paragraphs(raw):
    xml = zipfile.ZipFile(io.BytesIO(raw)).read('word/document.xml').decode('utf-8')
    return [re.sub(r'<[^>]+>', '', p).replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
            for p in re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S)]


# ----------------------------------------------------------------- authority derivation (reader-scoped)
def derive(reader=worktree_reader):
    raw = reader(SPEC)
    require(sha(raw) == SPEC_SHA, 'frozen specification SHA256')
    paras = spec_paragraphs(raw)
    i = paras.index('M6 Baselines')
    require(paras[i + 1:i + 3] == ['method adapters + method_status.csv',
                                   'each row tagged faithful/adapted/blocked; no silent simplification'],
            'spec section 25 M6 gate row (adapters + method_status.csv)')
    require('Negative results and failed baselines remain in the evidence package; they are not deleted to make the '
            'proposed method look cleaner.' in paras, 'spec section 28 negative-result retention')
    fair = reader(FAIR_TRACK).decode()
    require('  status: DEFERRED_TO_M6_SECONDARY_TRACK\n' in fair and '  methods: [E06b, E07b, E09, E10, E11]\n' in fair,
            'fair_track_v1 Track B (E06b, E07b) deferred to M6')
    a1 = reader(A1).decode()
    require('`DSDG-NATIVE` (E06b) and `DIFFFAS-NATIVE` (E07b) keep their' in a1 and 'deferred to M6' in a1,
            'A1 keeps E06b/E07b definitions and defers them to M6')
    jf, jr = json.loads(reader(JF_RECORD)), json.loads(reader(JR_RECORD))
    require(jr['fidelity'] == {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
                               'new_fidelity_class': False} and jf['fidelity'] == jr['fidelity'],
            'E07c historical fidelity CONTROLLED_ADAPTATION / DEV-021')
    o3 = jr['O3_smoke_test']
    require((o3['smoke_test_required'], o3['smoke_test_completed'], o3['smoke_test_waived'],
             o3['blocks_full_M8_plus_execution_until_completed']) == (True, False, False, True), 'M6D6jR O3 unchanged')
    jfe = json.loads(reader(JF_EVIDENCE))['failed_attempt']
    require((jfe['authority'], jfe['seed'], jfe['failure_phase'], jfe['scientific_optimizer_steps'],
             jfe['scientific_checkpoint_writes']) == (ATTEMPT1_AUTHORITY, 42, 'RUN_CONTEXT_OPEN', 0, 0),
            'attempt 1 preserved by M6D6jF')
    rows = [json.loads(line) for line in reader(LEDGER).decode().splitlines()]
    require(not [r for r in rows if r.get('scientific_training_completed') is True], 'no main training ever completed')
    require(not [r for r in rows if r.get('method_id') in ('E06b', 'E07b')], 'no E06b/E07b ledger history to supersede')
    require('successor' in json.dumps(json.loads(reader(IR_RECORD))).lower(), 'M6D6iR successor-gated retention')
    return {'spec_m6_gate_adapters_plus_method_status': True, 'spec_negative_results_retained': True,
            'track_b_deferred_to_m6': ['E06b', 'E07b'], 'a1_track_b_definitions_kept': True,
            'e07c_fidelity_historical': 'CONTROLLED_ADAPTATION / DEV-021', 'm6d6jr_o3_unchanged': True,
            'attempt_1_preserved_by_m6d6jf': True, 'no_main_scientific_training_completed': True}


def check_record(reader=worktree_reader):
    raw = reader(RECORD)
    require(sha(raw) == RECORD_SHA, 'amendment record SHA256 == pinned constant')
    r = json.loads(raw)
    require(r['amendment_document'] == {'path': DOC, 'sha256': sha(reader(DOC))}, 'amendment document binding')
    require(r['audit_evidence'] == {'json': EV_JSON, 'md': EV_MD}, 'audit evidence paths')
    require(r['bound_authority_sha256'] == BOUND, 'bound authority set and values')
    for rel, digest in BOUND.items():
        require(sha(reader(rel)) == digest, 'bound authority bytes ' + rel)
    require((r['amendment_id'], r['milestone'], r['classification'], r['record_kind'], r['authority_commit'],
             r['scientific_scope_change'], r['deterministic_implementation_clarification'], r['is_m6_closure'],
             r['historical_evidence_rewritten'], r['frozen_docx_edited']) ==
            (AMENDMENT_ID, MILESTONE, EXCLUSION_CLASS, RECORD_KIND, AUTHORITY, True, False, False, False, False),
            'identity')
    require(r['status'] == ['OWNER_APPROVED_DIFFFAS_FAMILY_ONLY', 'ADDITIVE', 'NON_DESTRUCTIVE', 'PROSPECTIVE'], 'status')
    require({'BLOCKED_BY_SOURCE_GAP', 'MODEL_FAILURE', 'OOM', 'TECHNICAL_IMPLEMENTATION_FAILURE',
             'BASELINE_REPLACEMENT', 'SILENT_DELETION'} <= set(r['not_classified_as']), 'not classified as')
    require(r['scope']['methods'] == ['E07c', 'E07b'] and r['scope']['family'] == 'DiffFAS' and
            set(r['scope']['not_in_scope']) == {'E06b', 'E06c', 'E01-E05', 'GPAT'}, 'scope = DiffFAS family only')
    e = r['E07c']
    require((e['method_name'], e['m6_gate_status'], e['blocker_or_exclusion_reason'], e['fidelity_class'],
             e['deviation'], e['new_fidelity_class'], e['new_deviation'], e['eligible_for_future_bank'],
             e['completed_scientific_seeds'], e['scientific_training_completed'], e['seeds_never_started'],
             e['m6d6jr_launch_authorization']) ==
            ('DiffFAS-BIN-IDFREE', 'blocked', REASON, 'CONTROLLED_ADAPTATION', 'DEV-021', False, False, False, 0,
             False, [1337, 2026], 'SUPERSEDED_PROSPECTIVELY_BY_A9'), 'E07c state')
    require(e['attempts'] == [FAILED_ATTEMPT, PARTIAL_ATTEMPT], 'both seed-42 attempts preserved')
    require(e['partial_checkpoint'] == PARTIAL_CHECKPOINT, 'partial checkpoint metadata and retention')
    require({'rerun seed 42', 'start seed 1337', 'start seed 2026', 'use the partial checkpoint scientifically',
             'generate an E07c M8 bank', 'use E07c for downstream evaluation',
             'report partial-run numbers as benchmark results', 'silently remove E07c from a table schema'}
            <= set(e['forbidden']), 'E07c forbidden actions')
    require(r['E07b'] == E07B, 'E07b excluded with the DiffFAS family; fidelity not assessed')
    require(r['E06b'] == E06B, 'E06b out of A9 scope; ACTIVE M6 work; not excluded; not resource-blocked')
    require(r['m6'] == M6, 'M6 remains OPEN; method_status.csv not created')
    ss = r['stage_state_json']
    require((ss['path'], ss['status'], ss['current_milestone_authority'], ss['modified_by_m6a9'], ss['sha256']) ==
            (STAGE_STATE, 'STALE_HISTORICAL_FIXTURE', False, False, BOUND[STAGE_STATE]), 'STAGE_STATE stale fixture')
    require(r['smoke_test'] == SMOKE and r['decision_operation'] == ZERO, 'smoke gate / no runtime activity')
    require(r['unchanged'] == {'gpat_configs': True, 'other_baselines': True, 'frozen_configs': True, 'data': True,
                               'split': True, 'pairs': True, 'probe': True, 'metrics': True}, 'unchanged scope')
    doc = reader(DOC).decode()
    for token in (EXCLUSION_CLASS, REASON, 'DEV-021', 'CONTROLLED_ADAPTATION', 'NOT_ASSESSED_NOT_IMPLEMENTED',
                  '`BLOCKED_BY_SOURCE_GAP` is **not** used', 'Attempt 1', 'Attempt 2', '54,836', '0 of 3',
                  'Seeds 1337 and 2026 are never started', 'model_050000.pt', 'RETAIN', 'NOT_ELIGIBLE_FOR_SCIENCE',
                  'b4ae6325bd04fdf615db24ca0cfb883680401c494b9c9d39f4830dbc8d775cae', 'E07b has no bank',
                  'not owner-excluded', 'not resource-blocked', 'remains ACTIVE M6 WORK', 'E06b DSDG-NATIVE remains active M6 work',
                  '`M6_CLOSED = false`', 'M6 remains OPEN', 'STALE_HISTORICAL_FIXTURE', 'HARD'):
        require(token in doc, 'amendment document token ' + token)
    require('OWNER_EXCLUDED_SECONDARY_TRACK' not in doc and 'TRACK_B_DEFERRED_NOT_IMPLEMENTED' not in doc,
            'no withdrawn E06b exclusion wording')
    return r


def build_evidence_from(r, d):
    return {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'authority_commit': AUTHORITY,
            'amendment_id': AMENDMENT_ID, 'exclusion_class': EXCLUSION_CLASS, 'record_path': RECORD,
            'record_sha256': RECORD_SHA, 'amendment_document_sha256': r['amendment_document']['sha256'],
            'scope': r['scope'], 'owner_scope_change': True,
            'E07c': {'m6_gate_status': 'blocked', 'reason': REASON, 'fidelity_class': 'CONTROLLED_ADAPTATION',
                     'deviation': 'DEV-021', 'completed_scientific_seeds': 0, 'seeds_never_started': [1337, 2026],
                     'attempts': [FAILED_ATTEMPT, PARTIAL_ATTEMPT], 'partial_checkpoint': PARTIAL_CHECKPOINT,
                     'eligible_for_future_bank': False},
            'E07b': E07B, 'E06b': E06B, 'm6': M6, 'smoke_test': SMOKE, 'decision_operation': ZERO,
            'stage_state_json': r['stage_state_json'], 'static_derivation': d,
            'bound_authority_sha256': BOUND, 'next_step': 'E06b DSDG-NATIVE M6 audit/decision packet (M6F-A); M7 NOT started'}


def check_evidence(reader=worktree_reader):
    r = check_record(reader)
    d = derive(reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(r, d), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in ('M6_CLOSED = false', EXCLUSION_CLASS, REASON, RECORD_SHA, 'E06b', 'ACTIVE M6 WORK', 'E07b', 'E07c',
                  '54,836', 'DEV-021', 'NOT_ASSESSED_NOT_IMPLEMENTED', 'method_status.csv', 'M7 HAS NOT STARTED',
                  'STALE_HISTORICAL_FIXTURE'):
        require(token in md, 'evidence report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6jF authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([CONFIG_STATUS, *bookkeeping]),
            'only CONFIG_STATUS + ledger/index may differ from the authority: ' + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    require(not [p for p in M6E_CLOSURE_ARTIFACTS if (ROOT / p).exists()], 'provisional M6E closure artifacts not revived')
    base, now = at_authority(CONFIG_STATUS), read(CONFIG_STATUS)
    require(sha(base) == CONFIG_STATUS_AUTHORITY_SHA and now.startswith(base) and len(now) > len(base),
            'CONFIG_STATUS is append-only (historical bytes preserved)')
    added = now[len(base):].decode()
    for token in ('M6A9', 'Amendment A9', EXCLUSION_CLASS, 'E07c', 'E07b', 'E06b', 'ACTIVE M6 WORK',
                  'M6_CLOSED = false', 'method_status.csv', 'M7 HAS NOT STARTED'):
        require(token in added, 'CONFIG_STATUS M6A9 note token ' + token)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (CONFIG_STATUS, *bookkeeping)]),
            'worktree holds exactly the M6A9 candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, CONFIG_STATUS):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no whitespace before line endings ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    require(rows[CONFIG_STATUS]['sha256'] == CONFIG_STATUS_AUTHORITY_SHA, 'baseline CONFIG_STATUS row')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in (*NEW, CONFIG_STATUS):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
    'record_kind': RECORD_KIND, 'amendment_created': True, 'amendment_id': AMENDMENT_ID, 'owner_scope_change': True,
    'exclusion_class': EXCLUSION_CLASS, 'scope_methods': ['E07c', 'E07b'], 'E07c_m6_gate_status': 'blocked',
    'E07c_exclusion_reason': REASON, 'E07c_fidelity_class': 'CONTROLLED_ADAPTATION', 'E07c_deviation': 'DEV-021',
    'E07c_completed_scientific_seeds': 0, 'E07c_scientific_training_completed': False, 'E07c_partial_seed': 42,
    'E07c_last_completed_step': 54836, 'E07c_seeds_never_started': [1337, 2026], 'E07c_terminal_checkpoint': False,
    'E07c_selected_checkpoint': False, 'E07c_future_bank_eligible': False,
    'E07c_partial_checkpoint_sha256': PARTIAL_CHECKPOINT['sha256'], 'E07c_partial_checkpoint_retained': True,
    'E07b_m6_gate_status': 'blocked', 'E07b_exclusion_reason': REASON, 'E07b_fidelity_class': 'NOT_ASSESSED_NOT_IMPLEMENTED',
    'E07b_future_bank_eligible': False, 'E06b_in_a9_scope': False, 'E06b_status': 'ACTIVE_M6_WORK',
    'E06b_owner_excluded': False, 'E06b_resource_blocked': False, 'M6_closed': False, 'method_status_csv_created': False,
    'M7_started': False, 'smoke_test_required': True, 'smoke_test_completed': False,
    'blocks_full_M8_plus_execution': True, 'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False,
    'GPU_contacted': False, 'M8_bank': False, 'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0,
    'checkpoint_writes': 0, 'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'record_sha256': RECORD_SHA, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': [CONFIG_STATUS]}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, CONFIG_STATUS)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == t['errors'] == 0 for t in row['tests']), 'ledger tests passed')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'Amendment A9: discontinue the DiffFAS family (E07c, E07b) for resource constraints; M6 stays open',
            'notes': ('A9 OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION, DiffFAS family only. E07c blocked, fidelity '
                      'CONTROLLED_ADAPTATION/DEV-021 unchanged; seed-42 attempts 1 (RUN_CONTEXT_OPEN, 0 steps) and 2 '
                      '(owner-interrupted at 54836/884000) preserved; 0 completed seeds; 1337/2026 never started; '
                      'model_050000.pt retained, never selected/final, not eligible. E07b excluded with the family '
                      '(NOT_ASSESSED_NOT_IMPLEMENTED). E06b out of scope and ACTIVE M6 work. M6_CLOSED=false; '
                      'method_status.csv not created. No runtime activity. M7 not started.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest tests.test_m6a9_difffas_scope_exclusion; python3 -B '
                        'tools/m6a9_difffas_scope_exclusion_preflight.py --write-evidence-json / --before-ledger / '
                        '--append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, CONFIG_STATUS))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_evidence()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA and
            current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
    worktree(bookkeeping)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [CONFIG_STATUS, *bookkeeping]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        ev = build_evidence_from(check_record(), derive())
        (ROOT / EV_JSON).write_text(json.dumps(ev, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        print(json.dumps({'status': 'WRITTEN', 'path': EV_JSON}))
        return
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.append_ledger:
        verify('before_ledger')
        row = ledger_row(json.loads(args.tests))
        with open(ROOT / LEDGER, 'ab') as fh:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
            fh.flush()
            os.fsync(fh.fileno())
        result = verify('before_index')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
