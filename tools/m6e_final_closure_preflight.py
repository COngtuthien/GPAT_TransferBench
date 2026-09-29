#!/usr/bin/env python3
"""Verify the M6E candidate: the FINAL M6 baseline closure (M6_CLOSED = true at implementation/qualification level).

The frozen M6 gate (spec section 25) is "method adapters + method_status.csv; each row tagged faithful/adapted/blocked;
no silent simplification". It is not a scientific-execution gate: baseline_full_scientific_execution_complete = false.
M6E creates outputs/audit/method_status.csv and reconciles the current-state records (deviation register, registry,
CONFIG_STATUS). It trains nothing, writes no checkpoint or bank, reads no data and does not start M7.

The row universe is DERIVED here, not assumed: spec section 17 third-party baseline rows (section 8 status tags apply to
third-party reimplementations) plus Amendment A1's Track-A additions, classified by A1 (fair_track_v1) and A9.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no checkpoint, no model.
Candidate-time checks run only here; derive / check_rows / check_records / check_evidence take readers (no HEAD lock).

  --write-method-status / --write-evidence-json   candidate creation only
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
AUTHORITY = '57c6a99035c2d45f47882c2104751a4cc0b923b1'
BRANCH = 'm6-baselines'
MILESTONE = 'M6E'
CLASSIFICATION = 'M6E_FINAL_M6_BASELINE_CLOSURE'
RECORD_KIND = 'MILESTONE_CLOSURE / STATUS_RECONCILIATION'
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 134
LEDGER_PREFIX_SHA = 'ab1e5b46e0ab5d8be139729481ec7d15826c299948cd682aa83c8f6d65235555'
INDEX_BASELINE_ROWS = 805
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
DEVIATIONS = 'outputs/audit/deviation_report.md'
REGISTRY = 'third_party/registry.yaml'
MODIFIED = tuple(sorted((CONFIG_STATUS, DEVIATIONS, REGISTRY)))
METHOD_STATUS = 'outputs/audit/method_status.csv'
EV_JSON = 'outputs/audit/M6E_FINAL_M6_CLOSURE.json'
EV_MD = 'outputs/audit/M6E_FINAL_M6_CLOSURE.md'
TESTS = 'tests/test_m6e_final_closure.py'
PREFLIGHT = 'tools/m6e_final_closure_preflight.py'
NEW = tuple(sorted((METHOD_STATUS, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
WITHDRAWN_M6E_ARTIFACTS = ('outputs/audit/M6E_BASELINE_CLOSURE.json', 'outputs/audit/M6E_BASELINE_CLOSURE.md',
                           'tools/m6e_baseline_closure_preflight.py', 'tests/test_m6e_baseline_closure.py')
STAGE_STATE = 'outputs/audit/STAGE_STATE.json'
A1 = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md'
A9_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A9_DiffFAS_Family_Resource_Constrained_Scope_Exclusion.md'
A9_RECORD = 'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'
FAIR_TRACK = 'configs/frozen/fair_track_v1.yaml'
TRACK_A_MATRIX = 'outputs/audit/M4_TRACK_A_METHOD_MATRIX.md'
JR_RECORD = 'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml'
M6A9_EV = 'outputs/audit/M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION.json'
M6FC_EV = 'outputs/audit/M6FC_E06B_GPU_QUALIFICATION.json'
M6FD_EV = 'outputs/audit/M6FD_E06B_PRODUCTION_QUALIFICATION.json'
M6D2B_EV = 'outputs/audit/M6D2B_E03_RUNTIME_QUALIFICATION.json'
E06C_LOCK = 'environments/e06c.lock.json'
SOURCE_PINS = 'third_party/source_pins.json'
METHOD_CONFIGS = {'E01': 'configs/methods/e01_fas_aug.yaml', 'E02': 'configs/methods/e02_freqsub.yaml',
                  'E03': 'configs/methods/e03_stdn.yaml', 'E04': 'configs/methods/e04_physics_std.yaml',
                  'E05': 'configs/methods/e05_pcgan.yaml', 'E06b': 'configs/methods/e06b_dsdg_native.yaml',
                  'E06c': 'configs/methods/e06c_dsdg_bin_idfree.yaml',
                  'E07c': 'configs/methods/e07c_difffas_bin_idfree.yaml'}
GPAT_B0 = 'configs/methods/gpat_b0.yaml'
GPAT_ABSENT = ('configs/methods/gpat_b1.yaml', 'configs/methods/gpat_b2.yaml', 'configs/methods/gpat_b3.yaml')
E06B_CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
DSDG_PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
DIFFFAS_PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
BOUND = {SPEC: SPEC_SHA,
         A1: '03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472',
         A9_DOC: '2ea13c1c9f070c7d0e47d27d3ee77688bf1d9e1a92febc2aa8f1f5325a93873e',
         A9_RECORD: '5ddd59f3560eea63471dd5bc1b4f65281e8f529a74a1e9d4d990ddbb427f3644',
         FAIR_TRACK: 'd9b02214d1e4701fe7d53972e94be0c2e68c6199a3378cacd69f2c1c87826d9d',
         METHOD_CONFIGS['E01']: '87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0',
         METHOD_CONFIGS['E02']: 'fdf74776cc100d5b5c1ed5f58b84ebb0dfd534782fbb7fc34382919d38d67bf0',
         METHOD_CONFIGS['E03']: '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0',
         METHOD_CONFIGS['E04']: '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a',
         METHOD_CONFIGS['E05']: '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de',
         METHOD_CONFIGS['E06b']: E06B_CONFIG_SHA,
         METHOD_CONFIGS['E06c']: '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002',
         METHOD_CONFIGS['E07c']: 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
         GPAT_B0: '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
         'configs/run_logging_v1.yaml': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
         TRACK_A_MATRIX: 'deeb63ca600559a58ed8c2928ab9c88c7528e2edadd5e03480b57212c86610f8',
         JR_RECORD: '2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830',
         M6A9_EV: 'c87505e36bfa9b234132268e30d9c4e4eb91b0c780ef557e88d5a120971f615a',
         M6FC_EV: 'bc638cf3548ac170e3d9d7d0f7263d768aebabf1cc7a99c1b743f54d931f12b2',
         M6FD_EV: '8853f6e850d15491b6b6baadfa80d9c5caf2303bf967dd068e79146048f113d3',
         M6D2B_EV: '4fa74d00b945cf6b36de42e3a55682d52a3bdf61a3a3cc27b57c5f6812f826eb',
         E06C_LOCK: '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95',
         SOURCE_PINS: '484558032f53b536cb7832803980c5273264c262ab74a6eb2c67f27641d72cec',
         STAGE_STATE: 'a8abf8288feaba561916b7ffe513014cbe2b5831be86d048702f2718059dd50d'}

# ----------------------------------------------------------------- canonical universe (derived and re-checked below)
SPEC_17_IDS = ['E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06a', 'E06b', 'E07a', 'E07b', 'E08', 'E09', 'E10', 'E11']
UNIVERSE = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06a', 'E06b', 'E06c', 'E07a', 'E07b', 'E07c']
NON_ROWS = {
    'E00': {'classification': 'NOT_AN_M6_BASELINE_NO_SYNTHETIC_METHOD',
            'basis': 'spec 17 synthetic source "None / real-only": a downstream lower bound (M10/M11) with no generator; '
                     'spec 8 status tags describe third-party reimplementations, so faithful/adapted/blocked has no '
                     'object for it',
            'owning_milestones': ['M10', 'M11']},
    'E08': {'classification': 'NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7',
            'basis': 'GPAT-B0 is the frozen proposed method (spec 9 "GPAT exact architecture - frozen proposed method"); '
                     'its deliverable is the spec 25 M7 row; spec 8 third-party status tags do not apply',
            'owning_milestones': ['M7']},
    'E09': {'classification': 'NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7', 'basis': 'GPAT-B1 (spec 5.3, 17); M7 deliverable',
            'owning_milestones': ['M7', 'M12']},
    'E10': {'classification': 'NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7', 'basis': 'GPAT-B2 (spec 5.3, 17); M7 deliverable',
            'owning_milestones': ['M7', 'M12']},
    'E11': {'classification': 'NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7', 'basis': 'GPAT-B3 (spec 5.3, 17); M7 deliverable',
            'owning_milestones': ['M7', 'M12']}}
GATES = ('faithful', 'adapted', 'blocked')
SCOPES = ('ACTIVE_M6_BASELINE', 'SUPERSEDED_BY_ADOPTED_AMENDMENT', 'OWNER_EXCLUDED')
COLUMNS = ['experiment_id', 'method', 'track', 'scope_status', 'gate', 'fidelity_class', 'spec_intended_status',
           'deviation_or_authority', 'implementation_status', 'runtime_qualification_status',
           'production_qualification_status', 'scientific_execution_status', 'completed_scientific_seeds',
           'checkpoint_status', 'future_bank_eligible', 'block_or_supersession_reason', 'disclosures',
           'latest_authoritative_milestone']
E06B_DISCLOSURES = ('SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED', 'CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED',
                    'RESUME_UNQUALIFIED_FRESH_ONLY', 'SCIENTIFIC_TRAINING_NOT_EXECUTED')
A9_REASON = 'OWNER_EXCLUDED_RESOURCE_CONSTRAINT'
E07C_CHECKPOINT = ('model_050000.pt NON_SCIENTIFIC_EVIDENCE_ONLY (selected=false; final=false; scientific_eligible=false; '
                   'sha256 b4ae6325bd04fdf615db24ca0cfb883680401c494b9c9d39f4830dbc8d775cae)')
E07C_EXECUTION = ('DISCONTINUED_BY_A9; seed 42 attempt 1 FAILED_BEFORE_OPTIMIZATION (RUN_CONTEXT_OPEN; 0 steps); '
                  'seed 42 attempt 2 OWNER_INTERRUPTED (last completed step 54836; epoch 25); seeds 1337 and 2026 '
                  'NEVER_STARTED')


def _row(eid, method, track, scope, gate, fidelity, intended, authority, impl, runtime, production, execution,
         checkpoint, bank, reason, disclosures, latest):
    return dict(zip(COLUMNS, [eid, method, track, scope, gate, fidelity, intended, authority, impl, runtime, production,
                              execution, '0', checkpoint, bank, reason, disclosures, latest]))


NOT_QUALIFIED_REAL = 'NOT_QUALIFIED_ON_REAL_TRAIN'
ROWS = [
    _row('E01', 'FAS-Aug', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'faithful',
         'FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION', 'FAITHFUL_OFFICIAL', 'A2-01; A2-05',
         'IMPLEMENTED (M6C1; official backend; synthetic pixel smoke PASS)', 'CORE_ENVIRONMENT_QUALIFIED (M6D1; synthetic)',
         NOT_QUALIFIED_REAL + ' (non-learned; real-data bank path not exercised)', 'NOT_EXECUTED',
         'NOT_APPLICABLE_NON_LEARNED', 'true', 'NONE',
         'non-learned but seed-dependent per-pair operator draws; asset order lexicographic (A2-01)', 'M6D1'),
    _row('E02', 'Frequency Substitution', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'adapted',
         'SPEC_DEFINED', 'CONTROLLED_ADAPTATION', 'spec 8.2; A2-02; A2-03; A2-04; M6A5b label normalization',
         'IMPLEMENTED (M6C1)', 'CORE_ENVIRONMENT_QUALIFIED (M6D1; synthetic)',
         NOT_QUALIFIED_REAL + ' (non-learned; real-data bank path not exercised)', 'NOT_EXECUTED',
         'NOT_APPLICABLE_NON_LEARNED', 'true', 'NONE',
         'gate adapted because spec 8.2 tags it CONTROLLED_ADAPTATION; SPEC_DEFINED (M6A5b) is the provenance label '
         '(method defined by the frozen spec itself; not an adaptation of an external method)', 'M6D1'),
    _row('E03', 'STDN', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'faithful',
         'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'FAITHFUL_OFFICIAL', 'E03 runtime compatibility addendum (M6D2b)',
         'IMPLEMENTED (M6C2a)', 'RUNTIME_QUALIFIED (M6D2a environment; M6D2b official graph; synthetic)',
         NOT_QUALIFIED_REAL, 'NOT_EXECUTED', 'NO_SCIENTIFIC_CHECKPOINT', 'true', 'NONE',
         'frozen config field FAITHFUL_OFFICIAL unchanged (historical); compatibility TF1 runtime; checkpoint rule '
         'OFFICIAL_LATEST_FINAL_CKPT_50', 'M6D2b'),
    _row('E04', 'Physics-STD', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'adapted',
         'CONTROLLED_ADAPTATION', 'FAITHFUL_PAPER', 'A3 section 4; A4; A2-06; M6D3d owner resolution addendum',
         'IMPLEMENTED (M6C2b1)', 'TRAINING_GRAPH_QUALIFIED (M6D3a geometry; M6D3e training graph; synthetic)',
         NOT_QUALIFIED_REAL, 'NOT_EXECUTED', 'NO_SCIENTIFIC_CHECKPOINT', 'true', 'NONE',
         'controlled reconstruction (3DDFA_V2 geometry; Q=140; depth M0); not a faithful reproduction; checkpoint '
         'BASELINE_FINAL_STATE_V1 at 150000', 'M6D3e'),
    _row('E05', 'PCGAN', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'adapted',
         'CONTROLLED_ADAPTATION', 'FAITHFUL_PAPER_OR_CONTROLLED_ADAPTATION', 'A2-07; A5; A2-06; M6D4c owner resolution addendum',
         'IMPLEMENTED (M6C2b2)', 'TRAINING_RUNNER_QUALIFIED (M6D4a; M6D4d; M6D4e clean requalification; synthetic)',
         NOT_QUALIFIED_REAL, 'NOT_EXECUTED', 'NO_SCIENTIFIC_CHECKPOINT', 'true', 'NONE',
         'executable architecture basis = pinned swapping-autoencoder (A2-07); never called native or faithful PCGAN; '
         'checkpoint BASELINE_FINAL_STATE_V1 at 4000', 'M6D4e'),
    _row('E06a', 'DSDG-BIN', 'NONE (spec 17 binary-adapted row; in neither A1 track)', 'SUPERSEDED_BY_ADOPTED_AMENDMENT',
         'blocked', 'NOT_ASSESSED_NOT_IMPLEMENTED', 'ADAPTED_BINARY_VARIANT (spec 5.1)', 'DEV-019 (Amendment A1)',
         'NOT_IMPLEMENTED', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'NOT_EXECUTED', 'NONE', 'false',
         'SUPERSEDED_BY_A1: its spec 8.6 same-subject identity pairing is a forbidden Track-A sampling rule (A1; '
         'fair_track_v1) and A1 places it in no track; replaced for Track A by E06c (DEV-020)',
         'row kept visible (not deleted); no adapter was ever built for it', 'M4 (Amendment A1)'),
    _row('E06b', 'DSDG-NATIVE', 'B_NATIVE_FULL_SECONDARY', 'ACTIVE_M6_BASELINE', 'faithful',
         'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'FAITHFUL_OFFICIAL',
         'M6FD owner fidelity decision; no deviation (execution runtime compatibility only)',
         'CONFIG_FROZEN; STATIC_ADAPTER_IMPLEMENTED (M6FB)',
         'GPU_GRAPH_QUALIFIED; EXECUTION_MAPPING_QUALIFIED; WORKER_DETERMINISM_QUALIFIED (M6FC)',
         'PRODUCTION_DATA_PATH_QUALIFIED; PRODUCTION_BATCH_PATH_QUALIFIED; SCIENTIFIC_CLI_PREFLIGHT_QUALIFIED '
         '(M6FD; real CASIA+MSU TRAIN)', 'SCIENTIFIC_TRAINING_NOT_EXECUTED', 'NO_SCIENTIFIC_CHECKPOINT', 'true', 'NONE',
         'SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED; CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED; '
         'RESUME_UNQUALIFIED_FRESH_ONLY; SCIENTIFIC_TRAINING_NOT_EXECUTED; N_syn DEFERRED_TO_M8; CASIA+MSU only (SiW '
         'NOT_INSTANTIABLE_MISSING_SUBJECT_ID); frozen YAML fidelity field PENDING_M6F_C_RUNTIME_QUALIFICATION is '
         'historical; environment gpat-m6-e06c', 'M6FD'),
    _row('E06c', 'DSDG-BIN-IDFREE', 'A_FAIR_COMMON_IDENTITY_FREE', 'ACTIVE_M6_BASELINE', 'adapted',
         'CONTROLLED_ADAPTATION', 'A1_TRACK_A_ADDITION', 'DEV-020; M6D5c; M6D5d',
         'IMPLEMENTED (M6C2a)', 'RUNTIME_ARCHITECTURE_QUALIFIED; GLOBAL_BATCH_240_EXECUTION_RESOLVED (M6D5a-M6D5d)',
         'PRODUCTION_RUNNER_QUALIFIED_ON_REAL_TRAIN (M6D5e; incl. checkpoint writer and resume)', 'NOT_EXECUTED',
         'NO_SCIENTIFIC_CHECKPOINT', 'true', 'NONE',
         'lambda_pair=0 and loss_cls degenerate with one spoof class (DEV-020); unchanged by M6F and A9', 'M6D5e'),
    _row('E07a', 'DiffFAS-BIN', 'NONE (spec 17 binary-adapted row; in neither A1 track)', 'SUPERSEDED_BY_ADOPTED_AMENDMENT',
         'blocked', 'NOT_ASSESSED_NOT_IMPLEMENTED', 'ADAPTED_BINARY_VARIANT (spec 5.1)', 'DEV-019 (Amendment A1)',
         'NOT_IMPLEMENTED', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'NOT_EXECUTED', 'NONE', 'false',
         'SUPERSEDED_BY_A1: its spec 8.7 same-identity reconstruction-pair structure is a forbidden Track-A sampling '
         'rule (A1; fair_track_v1) and A1 places it in no track; replaced for Track A by E07c (DEV-021)',
         'row kept visible (not deleted); no adapter was ever built for it', 'M4 (Amendment A1)'),
    _row('E07b', 'DiffFAS-NATIVE', 'B_NATIVE_FULL_SECONDARY', 'OWNER_EXCLUDED', 'blocked',
         'NOT_ASSESSED_NOT_IMPLEMENTED', 'FAITHFUL_OFFICIAL', 'A9', 'NOT_IMPLEMENTED', 'NOT_APPLICABLE',
         'NOT_APPLICABLE', 'NOT_EXECUTED', 'NONE', 'false',
         A9_REASON + ' (A9; DiffFAS family discontinued)',
         'not a source gap; no bank; no downstream scientific use; later tables keep the row as N/A', 'M6A9'),
    _row('E07c', 'DiffFAS-BIN-IDFREE', 'A_FAIR_COMMON_IDENTITY_FREE', 'OWNER_EXCLUDED', 'blocked',
         'CONTROLLED_ADAPTATION', 'A1_TRACK_A_ADDITION', 'DEV-021; A3 section 5; A6; A7; A8; A9',
         'IMPLEMENTED (M6C2b3)', 'RUNTIME_AND_MAIN_GRAPH_QUALIFIED (M6D6a-M6D6i)',
         'MAIN_PRODUCTION_RUNNER_QUALIFIED (M6D6j)', E07C_EXECUTION, E07C_CHECKPOINT, 'false',
         A9_REASON + ' (A9)',
         'auxiliary conditioning encoder trained once and frozen (M6D6g/M6D6h) - not a main scientific seed; '
         'downstream scientific use false; not a source gap', 'M6A9')]
KNOWN_ENVIRONMENTAL_TEST_ERRORS = [
    {'node': 'test_m6c2a_dsdg.TestDSDG (setUpClass)', 'asset': 'LightCNN_29Layers_V2_checkpoint.pth.tar'},
    {'node': 'test_m6c2b1_physics_std.TestE04Geometry (setUpClass)', 'asset': 'tddfa_v2/configs/bfm_noneck_v3.pkl'},
    {'node': 'test_m6c2b1_physics_std.TestE04Plans (setUpClass)', 'asset': 'tddfa_v2/configs/bfm_noneck_v3.pkl'},
    {'node': 'test_m6c2b1_physics_std.TestE04SourcesAssets.test_actual_assets_verified_without_deserialization',
     'asset': 'tddfa_v2/configs/bfm_noneck_v3.pkl'}]
ENV_CAUSE = 'PreparationError: required external asset unavailable: /media/cong/Data/... (external drive not mounted)'
SMOKE = {'smoke_test_required': True, 'smoke_test_completed': False, 'smoke_test_waived': False,
         'status': 'REQUIRED_NOT_COMPLETED', 'gate': 'HARD_GATE_BEFORE_FIRST_FULL_M8_PLUS_EXECUTION',
         'm6_closure_requirement': False, 'blocks_M7_generator_training': False,
         'source': 'M6D6jR O3 STAGED_MILESTONE_PRECEDENCE_CLARIFICATION (unchanged)'}
M6 = {'M6_CLOSED': True, 'closure_level': 'IMPLEMENTATION_AND_QUALIFICATION',
      'baseline_full_scientific_execution_complete': False, 'scientific_baseline_training_required_for_M6_gate': False,
      'method_status_csv_created': True, 'method_status_rows': len(UNIVERSE)}
M7 = {'status': 'NOT_STARTED', 'entry_blockers': ['DEV-003 lambda_dir UNAPPROVED', 'Q-05 GRL identity-adversary semantics',
                                                  'configs/methods/gpat_b1.yaml, gpat_b2.yaml, gpat_b3.yaml not created',
                                                  'zero-residual, DWT->IDWT and VAL-only selection gates not implemented'],
      'm6_blockers': False}
ZERO = {'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'banks_generated': 0,
        'model_executions': 0, 'TRAIN_image_reads': 0, 'VAL_image_reads': 0, 'TEST_image_reads': 0,
        'GPU_contacted': False}
CURRENT_AUTHORITY = ['outputs/audit/EXECUTION_LEDGER.jsonl', 'adopted amendments (docs/spec/amendments, configs/amendments)',
                     'milestone evidence (outputs/audit/M6*)', METHOD_STATUS]


def require(ok, message):
    if not ok:
        raise ValueError('M6E: ' + message)


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


def registry_entry(text, method_id):
    start = text.index(f'  - method_id: {method_id}\n')
    end = text.find('\n  - method_id: ', start + 1)
    return text[start:end]


def yaml_scalar(text, key):
    found = re.findall(rf'^{key}: (\S+)$', text, flags=re.M)
    require(len(found) == 1, 'single top-level key ' + key)
    return found[0]


# ----------------------------------------------------------------- universe derivation (reader-scoped)
def derive(reader=worktree_reader):
    for rel, digest in BOUND.items():
        require(sha(reader(rel)) == digest, 'bound authority bytes ' + rel)
    for mid, rel in METHOD_CONFIGS.items():
        require(reader(rel) == reader('frozen_config_snapshot/' + rel), 'snapshot byte-identical ' + mid)
    paras = spec_paragraphs(reader(SPEC))
    i, j = paras.index('17. Required experiment matrix'), paras.index('17.1 GPAT architectural ablation')
    cells = paras[i + 1:j]
    require(cells[:5] == ['Experiment ID', 'Synthetic source', 'Track', 'Evaluator', 'Purpose'] and len(cells) % 5 == 0,
            'spec 17 table shape')
    table = [cells[k:k + 5] for k in range(5, len(cells), 5)]
    require([r[0] for r in table] == SPEC_17_IDS, 'spec 17 experiment ids')
    source = {r[0]: r[1] for r in table}
    k = paras.index('M6 Baselines')
    require(paras[k + 1:k + 3] == ['method adapters + method_status.csv',
                                   'each row tagged faithful/adapted/blocked; no silent simplification'], 'spec 25 M6 gate')
    k = paras.index('M7 GPAT')
    require(paras[k + 1] == 'B0/B1/B2/B3 generator checkpoints', 'spec 25 M7 owns GPAT B0-B3')
    require('☐ method_status.csv identifies faithful/adapted/blocked state for all methods.' in paras, 'spec 32')
    require('A full deterministic smoke test on a 2-subject-per-class toy subset must finish before full runs.' in paras,
            'spec 26 smoke')
    k = paras.index('8. Third-party reimplementation policy and method cards')
    require(paras[k + 1].startswith('Source-of-truth hierarchy for third-party methods'), 'spec 8 scope = third-party')
    require([p for p in paras if re.match(r'^8\.\d ', p)] ==
            ['8.1 FAS-Aug', '8.2 Frequency Substitution — controlled implementation of Huang et al. synthesis idea',
             '8.3 STDN — On Disentangling Spoof Trace for Generic FAS',
             '8.4 Physics-Guided Spoof Trace Disentanglement / PAMI', '8.5 PCGAN artifact pattern conversion',
             '8.6 DSDG', '8.7 DiffFAS'], 'spec 8 method cards')
    require('9. GPAT exact architecture — frozen proposed method' in paras, 'spec 9 GPAT = proposed method')
    spec_tokens = {'e02_controlled': 'Status: CONTROLLED_ADAPTATION because the paper describes random high-frequency',
                   'e06b_faithful': 'Status: FAITHFUL_OFFICIAL for DSDG-NATIVE using FaceX-Zoo addition_module/DSDG.',
                   'dsdg_identity_pairing': 'Training live/spoof identity pairing uses only TRAIN identities',
                   'difffas_bin_pair_structure': 'preserving the same live/spoof reconstruction-pair structure'}
    for name, token in spec_tokens.items():
        require(any(token in p for p in paras), 'spec token ' + name)
    fair = reader(FAIR_TRACK).decode()
    track_a = re.search(r'^  methods: \[(.*)\]', fair.split('track_b:')[0], flags=re.M).group(1).split(', ')
    track_b = re.search(r'^  methods: \[(.*)\]', fair.split('track_b:')[1], flags=re.M).group(1).split(', ')
    require(track_a == ['E01', 'E02', 'E03', 'E04', 'E05', 'E06c', 'E07c', 'E08'] and
            track_b == ['E06b', 'E07b', 'E09', 'E10', 'E11'], 'fair_track_v1 track membership')
    for token in ('  identity_supervision: FORBIDDEN', '  forbidden_inputs_scope: [model_input, loss, class_target, sampling_rule]',
                  '  dataset_dropping_allowed: false'):
        require(token in fair, 'fair_track_v1 ' + token.strip())
    matrix = reader(TRACK_A_MATRIX).decode()
    require('| E06a | DSDG-BIN | collapses only the spoof-type target and still inherits the official same-subject '
            'pairing — superseded for Track A by E06c |' in matrix and
            '| E07a | DiffFAS-BIN | collapses `style_id` but keeps the same-ID reconstruction structure — superseded '
            'for Track A by E07c |' in matrix, 'M4 (A1) records E06a/E07a superseded')
    a9 = json.loads(reader(A9_RECORD))
    require(a9['scope']['methods'] == ['E07c', 'E07b'] and a9['E06b']['owner_excluded'] is False, 'A9 scope')
    non_method = [e for e in SPEC_17_IDS if source[e] == 'None / real-only']
    gpat = [e for e in SPEC_17_IDS if source[e].startswith('GPAT-')]
    baselines = [e for e in SPEC_17_IDS if e not in non_method + gpat]
    a1_added = [e for e in track_a + track_b if e not in SPEC_17_IDS]
    universe = sorted(baselines + a1_added, key=lambda e: (e[:3], e[3:]))
    require(non_method == ['E00'] and gpat == ['E08', 'E09', 'E10', 'E11'] and a1_added == ['E06c', 'E07c'], 'partition')
    require(universe == UNIVERSE and sorted(NON_ROWS) == sorted(non_method + gpat), 'derived universe == canonical rows')
    superseded = [e for e in baselines if e not in track_a + track_b]
    excluded = list(a9['scope']['methods'])
    require(superseded == ['E06a', 'E07a'], 'superseded = spec 17 baselines in no A1 track')
    scope = {e: ('OWNER_EXCLUDED' if e in excluded else 'SUPERSEDED_BY_ADOPTED_AMENDMENT' if e in superseded
                 else 'ACTIVE_M6_BASELINE') for e in universe}
    return {'spec_17_ids': SPEC_17_IDS, 'non_rows': sorted(non_method + gpat), 'third_party_spec_17_baselines': baselines,
            'a1_track_a_additions': a1_added, 'universe': universe, 'row_count': len(universe),
            'superseded_by_a1': superseded, 'a9_owner_excluded': sorted(excluded), 'scope_status': scope,
            'track_a': track_a, 'track_b': track_b}


def render(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def check_rows(reader=worktree_reader, rows=None):
    rows = ROWS if rows is None else rows
    d = derive(reader)
    raw = reader(METHOD_STATUS)
    require(raw == render(rows), 'method_status.csv bytes == canonical rows')
    parsed = list(csv.DictReader(io.StringIO(raw.decode())))
    ids = [r['experiment_id'] for r in parsed]
    require(ids == d['universe'] and len(set(ids)) == len(ids) == d['row_count'], 'rows = derived universe, no duplicate')
    by = {r['experiment_id']: r for r in parsed}
    for r in parsed:
        e = r['experiment_id']
        require(r['gate'] in GATES, 'gate vocabulary ' + e)
        require(r['scope_status'] == d['scope_status'][e], 'scope_status derived from A1/A9 ' + e)
        require(r['completed_scientific_seeds'] == '0', 'honest seed count ' + e)
        require(r['scientific_execution_status'] in ('NOT_EXECUTED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED', E07C_EXECUTION),
                'no scientific completion claim ' + e)
        require(r['future_bank_eligible'] in ('true', 'false'), 'bank vocabulary ' + e)
        require((r['future_bank_eligible'] == 'false') == (r['scope_status'] != 'ACTIVE_M6_BASELINE'), 'bank eligibility ' + e)
        require((r['gate'] == 'blocked') == (r['scope_status'] != 'ACTIVE_M6_BASELINE'), 'blocked iff not active ' + e)
    for e in ('E01', 'E02', 'E03', 'E04', 'E05'):
        require(by[e]['production_qualification_status'].startswith(NOT_QUALIFIED_REAL), 'no real-TRAIN claim ' + e)
    expected_gate = {'E01': 'faithful', 'E02': 'adapted', 'E03': 'faithful', 'E04': 'adapted', 'E05': 'adapted',
                     'E06a': 'blocked', 'E06b': 'faithful', 'E06c': 'adapted', 'E07a': 'blocked', 'E07b': 'blocked',
                     'E07c': 'blocked'}
    require({e: by[e]['gate'] for e in by} == expected_gate, 'gate map')
    for e in ('E01', 'E02', 'E04', 'E05', 'E06c', 'E07c'):
        require(yaml_scalar(reader(METHOD_CONFIGS[e]).decode(), 'fidelity_class') == by[e]['fidelity_class'],
                'fidelity_class == frozen config ' + e)
    require(yaml_scalar(reader(METHOD_CONFIGS['E03']).decode(), 'fidelity_class') == 'FAITHFUL_OFFICIAL' and
            json.loads(reader(M6D2B_EV))['fidelity_disclosure'] == by['E03']['fidelity_class'], 'E03 runtime disclosure')
    require(by['E02']['spec_intended_status'] == 'CONTROLLED_ADAPTATION' and by['E02']['fidelity_class'] == 'SPEC_DEFINED',
            'E02 SPEC_DEFINED maps to adapted per spec 8.2')
    for e, dev in (('E06c', 'DEV-020'), ('E07c', 'DEV-021')):
        require(yaml_scalar(reader(METHOD_CONFIGS[e]).decode(), 'deviation') == dev and
                by[e]['deviation_or_authority'].startswith(dev), 'deviation ' + e)
    fd = json.loads(reader(M6FD_EV))
    require(fd['fidelity']['final_method_fidelity'] == by['E06b']['fidelity_class'] ==
            'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY' and fd['fidelity']['new_scientific_deviation_found'] is False,
            'E06b final fidelity from M6FD')
    require((fd['scientific_seeds_completed'], fd['scientific_checkpoints_created'], fd['scientific_bank_created'],
             fd['scientific_training_completed'], fd['environment_of_record']) == (0, 0, False, False, 'gpat-m6-e06c'),
            'E06b M6FD execution facts')
    require(all(t in fd['statuses'] and t in by['E06b']['disclosures'] for t in E06B_DISCLOSURES), 'E06b disclosures')
    require(yaml_scalar(reader(METHOD_CONFIGS['E06b']).decode(), 'fidelity_class') ==
            'PENDING_M6F_C_RUNTIME_QUALIFICATION' and 'N_syn DEFERRED_TO_M8' in by['E06b']['disclosures'],
            'E06b frozen YAML pending field kept as history; N_syn deferred')
    a9 = json.loads(reader(M6A9_EV))
    ck, (at1, at2) = a9['E07c']['partial_checkpoint'], a9['E07c']['attempts']
    require((ck['selected'], ck['final'], ck['selected_for_final'], ck['eligible_for_M8'], ck['sha256'][:8]) ==
            (False, False, False, False, 'b4ae6325') and ck['sha256'] in by['E07c']['checkpoint_status'] and
            'selected=false; final=false; scientific_eligible=false' in by['E07c']['checkpoint_status'],
            'E07c partial checkpoint non-scientific')
    require((at1['failure_phase'], at1['scientific_optimizer_steps'], at2['interruption'],
             at2['last_completed_global_step'], at2['epoch'], a9['E07c']['seeds_never_started'],
             a9['E07c']['completed_scientific_seeds']) ==
            ('RUN_CONTEXT_OPEN', 0, 'OWNER_MANUAL_INTERRUPT', 54836, 25, [1337, 2026], 0) and
            by['E07c']['scientific_execution_status'] == E07C_EXECUTION, 'E07c seed history')
    for e in ('E07c', 'E07b'):
        require(by[e]['block_or_supersession_reason'].startswith(A9_REASON) and 'A9' in by[e]['deviation_or_authority'],
                'A9 reason ' + e)
    require([e for e in by if A9_REASON in by[e]['block_or_supersession_reason']] == ['E07b', 'E07c'],
            'only E07c/E07b are A9 owner-excluded')
    require(by['E07b']['fidelity_class'] == 'NOT_ASSESSED_NOT_IMPLEMENTED', 'E07b not assessed')
    for e in ('E06a', 'E07a'):
        require(by[e]['block_or_supersession_reason'].startswith('SUPERSEDED_BY_A1') and
                by[e]['deviation_or_authority'] == 'DEV-019 (Amendment A1)', 'supersession ' + e)
    return {'rows': len(parsed), 'gates': {g: [e for e in by if by[e]['gate'] == g] for g in GATES},
            'method_status_sha256': sha(raw)}


# ----------------------------------------------------------------- current-state record reconciliation
DEVIATION_TOKENS = ('## M6E — M6 baseline deviation-authority reconciliation (2026-09-29)',
                    '**No new DEV number is invented.**', 'A2-01', 'A2-07', 'A3 §4', 'A3 §5', 'A4', 'A5', 'A6', 'A9',
                    'E03 runtime compatibility addendum (M6D2b)', 'M6FC/M6FD owner fidelity decision',
                    '`new_scientific_deviation_found = false`', 'DEV-019', 'DEV-020', 'DEV-021', 'M6D5c/M6D5d',
                    'M6A5b label normalization', 'outputs/audit/method_status.csv', 'DEV-003 (`lambda_dir`) remains UNAPPROVED')
REGISTRY_FIELDS = {
    'E06b': {'track_b_status': 'M6_IMPLEMENTED_AND_QUALIFIED_SCIENTIFIC_TRAINING_NOT_EXECUTED', 'm6_gate': 'faithful',
             'm6_final_fidelity': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'm6_status_milestone': 'M6FD'},
    'E07b': {'track_b_status': 'OWNER_EXCLUDED_A9_RESOURCE_CONSTRAINT', 'm6_gate': 'blocked',
             'm6_final_fidelity': 'NOT_ASSESSED_NOT_IMPLEMENTED', 'm6_status_milestone': 'M6A9'},
    'E07c': {'m6_gate': 'blocked', 'm6_final_fidelity': 'CONTROLLED_ADAPTATION', 'm6_status_milestone': 'M6A9'}}
CONFIG_STATUS_TOKENS = ('**M6E (2026-09-29) — FINAL M6 BASELINE CLOSURE: M6_CLOSED = true', 'supersedes every current-state',
                        '`configs/methods/dsdg_native.yaml — NOT CREATED (M6)`',
                        '**dsdg_native is now implemented and qualified through M6F-D**', '**A9 owner-excluded**',
                        '**will not continue scientific training**', '**unchanged**',
                        '`baseline_full_scientific_execution_complete = false`', 'Canonical row universe: 11 rows',
                        '**Full scientific experiment execution remains pending**', 'STAGE_STATE.json',
                        '**M6_CLOSED = true. M7 HAS NOT STARTED.**', 'hard gate before the first full M8+')


def appended(reader, authority_reader, rel):
    base, now = authority_reader(rel), reader(rel)
    require(now.startswith(base) and len(now) > len(base), rel + ' append-only (historical bytes preserved)')
    return base.decode(), now[len(base):].decode()


def check_records(reader=worktree_reader, authority_reader=at_authority):
    base, added = appended(reader, authority_reader, DEVIATIONS)
    for token in DEVIATION_TOKENS:
        require(token in added, 'deviation register token ' + token)
    require(set(re.findall(r'DEV-\d+', added)) <= set(re.findall(r'DEV-\d+', base)), 'no new DEV number')
    require(set(re.findall(r'\*\*Status:\*\* (\w+)', added)) == {'APPROVED'}, 'appended status line(s) APPROVED only')
    for r in ROWS:
        require(f'| {r["experiment_id"]} ' in added or r['experiment_id'] in added, 'deviation register row ' + r['experiment_id'])
    _, cs_added = appended(reader, authority_reader, CONFIG_STATUS)
    for token in CONFIG_STATUS_TOKENS:
        require(token in cs_added, 'CONFIG_STATUS token ' + token)
    for r in ROWS:
        require(f'| {r["experiment_id"]} ' in cs_added and f'| {r["gate"]} |' in cs_added, 'CONFIG_STATUS row ' +
                r['experiment_id'])
    old, new = authority_reader(REGISTRY).decode(), reader(REGISTRY).decode()
    head_old, head_new = old[:old.index('schema_version:')], new[:new.index('schema_version:')]
    require(head_new.startswith(head_old) and 'M6E final M6 baseline closure' in head_new[len(head_old):] and
            'http' not in head_new[len(head_old):], 'registry header: additive M6E comment only')
    strip_old, strip_new = old[old.index('schema_version:'):], new[new.index('schema_version:'):]
    for mid in REGISTRY_FIELDS:
        strip_old = strip_old.replace(registry_entry(strip_old, mid), '')
        strip_new = strip_new.replace(registry_entry(strip_new, mid), '')
    require(strip_new == strip_old, 'registry changes confined to the E06b/E07b/E07c entries')
    for mid, fields in REGISTRY_FIELDS.items():
        o, n = registry_entry(old, mid), registry_entry(new, mid)
        for line in o.splitlines():
            if line.startswith(('    pinned_commit:', '    implementation_status:', '    source_status:',
                                '    official_repo:', '    config_file:', '    deviation:', '    notes:')):
                require(line in n.splitlines(), f'registry {mid} kept {line.strip()[:40]}')
        for k, v in fields.items():
            require(f'    {k}: {v}\n' in n + '\n', f'registry {mid} {k}')
        require('http' not in ''.join(line for line in n.splitlines() if line not in o.splitlines()), 'no new URL')
    require('PENDING_M6F_C' not in registry_entry(new, 'E06b') and 'GPU_GRAPH_NOT_YET_QUALIFIED' not in
            registry_entry(new, 'E06b'), 'E06b stale M6FB state removed')
    require('DEFERRED_TO_M6_SECONDARY_TRACK\n' not in registry_entry(new, 'E07b') + '\n' and
            'SOURCE_GAP' not in ''.join(registry_entry(new, m) for m in ('E07b', 'E07c')), 'E07b/E07c current state; no '
            'DiffFAS source gap')
    require(registry_entry(new, 'E06b').count(DSDG_PIN) == 1 and registry_entry(new, 'E07b').count(DIFFFAS_PIN) == 1 and
            registry_entry(new, 'E07c').count(DIFFFAS_PIN) == 1 and 'gpat-m6-e06c (environments/e06c.lock.json)' in
            registry_entry(new, 'E06b'), 'pins and E06b environment of record')
    require(new.count('    implementation_status: NOT_STARTED\n') == old.count('    implementation_status: NOT_STARTED\n'),
            'M0-pinned implementation_status untouched')
    return {'deviation_register': {'path': DEVIATIONS, 'change': 'APPENDED_M6E_RECONCILIATION_SECTION',
                                   'new_dev_numbers': [], 'sha256': sha(reader(DEVIATIONS))},
            'config_status': {'path': CONFIG_STATUS, 'change': 'APPENDED_M6E_CURRENT_STATE_SECTION',
                              'sha256': sha(reader(CONFIG_STATUS))},
            'registry': {'path': REGISTRY, 'change': 'E06b/E07b/E07c current-state fields + header comment; pins unchanged',
                         'fields': REGISTRY_FIELDS, 'sha256': sha(reader(REGISTRY))}}


def build_evidence(d, rows_result, records):
    return {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'record_kind': RECORD_KIND, 'status': 'PASS',
            'authority_commit': AUTHORITY, 'spec_sha256': SPEC_SHA,
            'm6_gate': {'spec_section': '25', 'deliverable': 'method adapters + method_status.csv',
                        'acceptance': 'each row tagged faithful/adapted/blocked; no silent simplification',
                        'allowed_gate_labels': list(GATES)},
            'm6': M6, 'universe_derivation': d, 'non_rows': NON_ROWS, 'method_status': rows_result,
            'E06b': {'gate': 'faithful', 'final_fidelity': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
                     'latest_authority': 'M6FD', 'owner_excluded': False, 'in_a9_scope': False,
                     'disclosures': list(E06B_DISCLOSURES), 'n_syn': 'DEFERRED_TO_M8',
                     'frozen_config_sha256': E06B_CONFIG_SHA, 'environment_of_record': 'gpat-m6-e06c'},
            'a9_scope': ['E07c', 'E07b'], 'E07c_history_preserved': {
                'attempt_1': 'FAILED_BEFORE_OPTIMIZATION (RUN_CONTEXT_OPEN)', 'attempt_2': 'OWNER_INTERRUPTED',
                'last_completed_step': 54836, 'epoch': 25, 'partial_checkpoint': 'model_050000.pt',
                'selected': False, 'final': False, 'scientific_eligible': False, 'completed_seeds': 0,
                'seeds_never_started': [1337, 2026], 'future_bank_eligible': False},
            'smoke_test': SMOKE, 'm7': M7, 'records': records,
            'stage_state_json': {'path': STAGE_STATE, 'sha256': BOUND[STAGE_STATE], 'modified_by_m6e': False,
                                 'status': 'STALE_HISTORICAL_FIXTURE', 'current_milestone_authority': False},
            'current_milestone_authority': CURRENT_AUTHORITY,
            'known_environmental_test_errors': {'count': len(KNOWN_ENVIRONMENTAL_TEST_ERRORS),
                                                'classification': 'PRE_EXISTING_ENVIRONMENTAL', 'cause': ENV_CAUSE,
                                                'nodes': KNOWN_ENVIRONMENTAL_TEST_ERRORS},
            'withdrawn_provisional_m6e_artifacts_revived': False, 'decision_operation': ZERO,
            'bound_authority_sha256': BOUND}


def check_evidence(reader=worktree_reader, authority_reader=at_authority):
    d = derive(reader)
    rows_result = check_rows(reader)
    records = check_records(reader, authority_reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence(d, rows_result, records), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in ('M6_CLOSED = true', '`baseline_full_scientific_execution_complete = false`',
                  '`scientific_baseline_training_required_for_M6_gate = false`', '11 rows', 'E06a', 'E07a', 'E00',
                  'E08', 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'OWNER_EXCLUDED_RESOURCE_CONSTRAINT',
                  'REQUIRED_NOT_COMPLETED', 'HARD_GATE_BEFORE_FIRST_FULL_M8_PLUS_EXECUTION', 'STALE_HISTORICAL_FIXTURE',
                  'M7 HAS NOT STARTED', '54,836', 'model_050000.pt', 'No new DEV number', rows_result['method_status_sha256']):
        require(token in md, 'evidence report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6F-D authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only the reconciled records + ledger/index differ: ' +
            json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    require(not [p for p in WITHDRAWN_M6E_ARTIFACTS if (ROOT / p).exists()], 'withdrawn provisional M6E artifacts absent')
    require(not [p for p in GPAT_ABSENT if (ROOT / p).exists()] and
            [p.name for p in (ROOT / 'methods/gpat').iterdir()] == ['.gitkeep'], 'no M7 artifact')
    require(read(STAGE_STATE) == at_authority(STAGE_STATE), 'STAGE_STATE byte-unchanged')


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M6E candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no trailing whitespace ' + rel)
    for rel in MODIFIED:
        raw = read(rel)
        added = raw[len(at_authority(rel)):] if raw.startswith(at_authority(rel)) else raw
        require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF, no trailing whitespace ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in MODIFIED:
        require(rows[p]['sha256'] == sha(at_authority(p)), 'baseline row ' + p)
    for p in (*NEW, *MODIFIED):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
    'record_kind': RECORD_KIND, 'M6_closed': True, 'baseline_full_scientific_execution_complete': False,
    'scientific_baseline_training_required_for_M6_gate': False, 'method_status_csv_created': True,
    'method_status_rows': len(UNIVERSE), 'method_status_universe': UNIVERSE,
    'method_status_non_rows': sorted(NON_ROWS), 'gates': {r['experiment_id']: r['gate'] for r in ROWS},
    'E06b_final_fidelity': 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'E06b_owner_excluded': False,
    'a9_scope': ['E07c', 'E07b'], 'completed_scientific_seeds_total': 0, 'scientific_training_completed': False,
    'smoke_test_required': True, 'smoke_test_completed': False, 'smoke_test_waived': False, 'M7_started': False,
    'amendment_created': False, 'new_deviation_numbers': [], 'stage_state_modified': False,
    'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False, 'GPU_contacted': False, 'M8_bank': False,
    'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'commit': False,
    'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': list(MODIFIED)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'], 'ledger tests recorded')
    for t in row['tests']:
        require(t['failures'] == 0, 'no test failure: ' + t['scope'])
        require(t['errors'] == 0 or (t.get('errors_preexisting_environmental') and
                                     t['errors'] == len(KNOWN_ENVIRONMENTAL_TEST_ERRORS)), 'errors classified: ' + t['scope'])


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6E: final M6 baseline closure (implementation/qualification level); method_status.csv created',
            'notes': ('Spec 25 gate met: 11-row method_status.csv (spec 17 third-party baselines E01-E07b + A1 E06c/E07c; '
                      'E00 and GPAT E08-E11 are not M6 baselines). Gates: faithful E01/E03/E06b; adapted E02/E04/E05/E06c; '
                      'blocked E06a/E07a (superseded by A1, DEV-019) and E07c/E07b (A9 owner exclusion). E06b final '
                      'fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY. 0 completed scientific seeds anywhere; '
                      'baseline_full_scientific_execution_complete=false. Deviation register, registry and CONFIG_STATUS '
                      'reconciled additively; no new DEV number. Smoke REQUIRED_NOT_COMPLETED (hard gate before full M8+). '
                      'STAGE_STATE unchanged. M7 not started.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest discover -s tests -p test_m6e_final_closure.py; '
                        'python3 -B tools/m6e_final_closure_preflight.py --write-method-status / --write-evidence-json / '
                        '--before-ledger / --append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
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
            'artifact_rows_added': len(NEW), 'modified_existing_files': [*MODIFIED, *bookkeeping],
            'method_status_rows': len(UNIVERSE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-method-status', action='store_true')
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.write_method_status:
        require(not (ROOT / METHOD_STATUS).exists(), 'fresh method_status.csv')
        derive()
        (ROOT / METHOD_STATUS).write_bytes(render(ROWS))
        check_rows()
        print(json.dumps({'status': 'WRITTEN', 'path': METHOD_STATUS, 'rows': len(ROWS)}))
        return
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        ev = build_evidence(derive(), check_rows(), check_records())
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
