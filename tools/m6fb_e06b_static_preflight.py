#!/usr/bin/env python3
"""Verify the M6FB candidate: E06b DSDG-NATIVE frozen config + static adapter; ledger/index LAST.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no model. Every change to
an existing file is proven to be exactly the intended additive hunk against the M6FA authority bytes, so E06c,
the M6B config set and all historical evidence stay byte/behaviour unchanged. Candidate-time checks run only here;
check_evidence takes a reader (no moving-HEAD lock).

  --write-evidence-json   candidate creation only (this is also the E06b freeze record read by
                          methods/common/config.py::load_method_config)
  --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
"""
import argparse
import ast
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '4ee7ac29ca16687aa4eb2a492d2a25554ce29253'
BRANCH = 'm6-baselines'
MILESTONE = 'M6FB'
CLASSIFICATION = 'M6FB_E06B_STATIC_IMPLEMENTATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 131
LEDGER_PREFIX_SHA = '2d1aeb69bff31da170ada076aec4ff1840f5fe5057c334d636e63b1ca88504d1'
INDEX_BASELINE_ROWS = 777
CONFIG = 'configs/methods/e06b_dsdg_native.yaml'
SNAPSHOT = 'frozen_config_snapshot/configs/methods/e06b_dsdg_native.yaml'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
NATIVE = 'methods/dsdg/native.py'
EV_JSON = 'outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json'
EV_MD = 'outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.md'
TESTS = 'tests/test_m6fb_e06b_static.py'
PREFLIGHT = 'tools/m6fb_e06b_static_preflight.py'
NEW = tuple(sorted((CONFIG, SNAPSHOT, NATIVE, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
REGISTRY = 'third_party/registry.yaml'
COMMON_CONFIG = 'methods/common/config.py'
COMMON_LEARNED = 'methods/common/learned.py'
MODIFIED = tuple(sorted((CONFIG_STATUS, REGISTRY, COMMON_CONFIG, COMMON_LEARNED)))
CONTRACT = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json'
CONTRACT_SHA = 'fe287ebe1ec9c19e5f63a40f1498e41f5b7044014c2cc7eb6279a32604555058'
UNCHANGED = {  # E06c, the M6B set and A9 remain byte-identical to the authority
    'configs/methods/e06c_dsdg_bin_idfree.yaml': '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002',
    'frozen_config_snapshot/configs/methods/e06c_dsdg_bin_idfree.yaml':
        '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002',
    'configs/frozen/dsdg_bin_idfree_v1.yaml': 'a0f84fac2e893158410aab6d3cd11f464f6385edad2403c1f8ce4a66615b15de',
    'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml':
        '5ddd59f3560eea63471dd5bc1b4f65281e8f529a74a1e9d4d990ddbb427f3644',
    'outputs/audit/M6B_CONFIG_FREEZE.json': None, 'tools/m6b_validate_configs.py': None,
    'methods/dsdg/adapter.py': None, 'methods/dsdg/training_graph.py': None,
    'methods/dsdg/microbatch_execution.py': None, 'methods/dsdg/microbatch_execution_v2.py': None,
    'methods/dsdg/runner.py': None, 'methods/dsdg/runner_io.py': None, CONTRACT: CONTRACT_SHA}
STATUSES = ['CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_NOT_YET_QUALIFIED',
            'PRODUCTION_RUNNER_NOT_YET_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED']
# exact additive hunks (authority text -> candidate text)
CONFIG_PY_HUNKS = [
    ('    "E07c": "e07c_difffas_bin_idfree.yaml",\n}\n',
     '    "E07c": "e07c_difffas_bin_idfree.yaml",\n}\n\n'
     '#: Methods frozen AFTER M6B: method_id -> (config filename, freeze record whose `config_sha256` pins the\n'
     '#: bytes). Kept separate so METHOD_FILES and the M6B freeze record stay exactly as M6B froze them.\n'
     'LATER_FROZEN_METHOD_FILES = {\n'
     '    "E06b": ("e06b_dsdg_native.yaml", "outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json"),\n}\n'),
    ('    if method_id not in METHOD_FILES:\n'
     '        raise FrozenConfigError(f"unknown method_id {method_id!r}; expected one of "\n'
     '                                f"{sorted(METHOD_FILES)}")\n'
     '    fn = METHOD_FILES[method_id]\n',
     '    if method_id in METHOD_FILES:\n'
     '        fn, freeze_record = METHOD_FILES[method_id], None\n'
     '    elif method_id in LATER_FROZEN_METHOD_FILES:\n'
     '        fn, freeze_record = LATER_FROZEN_METHOD_FILES[method_id]\n'
     '    else:\n'
     '        raise FrozenConfigError(f"unknown method_id {method_id!r}; expected one of "\n'
     '                                f"{sorted(METHOD_FILES) + sorted(LATER_FROZEN_METHOD_FILES)}")\n'),
    ('    frozen = json.loads((ROOT / "outputs/audit/M6B_CONFIG_FREEZE.json").read_text())\n'
     '    expected = next(m["config_sha256"] for m in frozen["methods"] if m["method_id"] == method_id)\n',
     '    if freeze_record is None:\n'
     '        frozen = json.loads((ROOT / "outputs/audit/M6B_CONFIG_FREEZE.json").read_text())\n'
     '        expected = next(m["config_sha256"] for m in frozen["methods"] if m["method_id"] == method_id)\n'
     '    else:\n'
     '        record_path = ROOT / freeze_record\n'
     '        if not record_path.is_file():\n'
     '            raise FrozenConfigError(f"{ctx}: freeze record {freeze_record} not found")\n'
     '        record = json.loads(record_path.read_text())\n'
     '        if record.get("method_id") != method_id:\n'
     '            raise FrozenConfigError(f"{ctx}: freeze record names another method")\n'
     '        expected = record["config_sha256"]\n')]
LEARNED_PY_HUNKS = [
    ("    elif cfg['method_id'] == 'E06c':\n        epoch = cfg['training']['all_epochs']\n",
     "    elif cfg['method_id'] in ('E06c', 'E06b'):   # both follow the pinned DSDG official epoch-200 generator rule\n"
     "        epoch = cfg['training']['all_epochs']\n")]
REGISTRY_OLD = '    config_file: configs/methods/dsdg_native.yaml (not yet created)\n'
REGISTRY_NEW = '    config_file: configs/methods/e06b_dsdg_native.yaml\n'
REGISTRY_ADDED_KEYS = ('m6_status', 'm6_status_milestone', 'm6_status_note')
CONFIG_TOKENS = (
    'method_id: E06b\n', 'status: CONFIG_FROZEN\n', 'milestone_status: NOT_TRAINED ', 'logging_contract: run_logging_v1\n',
    'track: B_NATIVE_FULL_SECONDARY\n', 'target_fidelity: FAITHFUL_OFFICIAL ',
    'final_execution_fidelity: PENDING_M6F_C_RUNTIME_QUALIFICATION\n',
    f'contract_resolution_sha256: {CONTRACT_SHA}\n', '  pinned_commit: 16b793a7564a4b9308cf94e62bdb2ffacb3a725a\n',
    '  new_source_pin: false\n', '  train_datasets: [casia_fasd, msu_mfsd]\n',
    '      status: NOT_INSTANTIABLE_MISSING_SUBJECT_ID\n', '      rows_used: 0\n',
    '    total: {spoof: 3720, live: 1240, subjects: 60, print: 2080, replay: 1640}\n',
    '  semantics: SAME_SUBJECT_ONLINE_RANDOM\n', '  materialized_pair_list: false\n', '  native_manifest: NOT_CREATED ',
    '  experiment_seeds: [42, 1337, 2026]\n', '  effective_batch_size: 240\n', '  all_epochs: 200\n', '  hdim: 128\n',
    '  attack_type: 2\n', '  workers: 8\n', '  learning_rate: 2.0e-4\n', '  lambda_mmd: 50\n', '  lambda_ip: 1000\n',
    '  lambda_type: 10\n', '  lambda_ort: 1\n', '  lambda_pair: 5\n', '  removed: []\n',
    '  vocabulary: [print, replay]\n', '  class_index: {print: 0, replay: 1}\n', '  num_spoof_classes: 2\n',
    '  num_workers: 8\n', '  persistent_workers: false\n', '  drop_last: false\n', '  rule: OFFICIAL_GENERATOR_EPOCH_200\n',
    '  microbatches_full_batch: 12\n', '  microbatches_tail_batch: 6\n', '  n_syn_intended: DEFERRED_TO_M8\n',
    '    sha256: d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964\n',
    '    TEST:\n      used_for: []\n      allowed: false\n', '      code_path_present: false\n')


def require(ok, message):
    if not ok:
        raise ValueError('M6FB: ' + message)


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


def apply_hunks(text, hunks):
    for old, new in hunks:
        require(text.count(old) == 1, 'authority hunk anchor present exactly once')
        text = text.replace(old, new)
    return text


def registry_entry(text, method_id):
    start = text.index(f'  - method_id: {method_id}\n')
    end = text.find('\n  - method_id: ', start + 1)
    return text[start:end]


# ----------------------------------------------------------------- reader-scoped derivation
def derive(reader=worktree_reader, authority_reader=at_authority):
    cfg = reader(CONFIG)
    require(sha(cfg) == CONFIG_SHA and reader(SNAPSHOT) == cfg, 'config SHA256 pinned; snapshot byte-identical')
    text = cfg.decode()
    for token in CONFIG_TOKENS:
        require(token in text, 'config token ' + token.strip()[:60])
    require(sha(reader(CONTRACT)) == CONTRACT_SHA, 'M6FA contract unchanged')
    for rel, digest in UNCHANGED.items():
        require(reader(rel) == authority_reader(rel), 'unchanged file ' + rel)
        require(digest is None or sha(reader(rel)) == digest, 'unchanged SHA256 ' + rel)
    require(reader(COMMON_CONFIG).decode() == apply_hunks(authority_reader(COMMON_CONFIG).decode(), CONFIG_PY_HUNKS),
            'methods/common/config.py differs from the authority only by the E06b additive hunks')
    require(reader(COMMON_LEARNED).decode() == apply_hunks(authority_reader(COMMON_LEARNED).decode(), LEARNED_PY_HUNKS),
            'methods/common/learned.py differs from the authority only by the E06b checkpoint-branch hunk')
    reg_old, reg_new = authority_reader(REGISTRY).decode(), reader(REGISTRY).decode()
    require(reg_new.replace(registry_entry(reg_new, 'E06b'), '') == reg_old.replace(registry_entry(reg_old, 'E06b'), ''),
            'registry changes are confined to the E06b entry')
    old_e, new_e = registry_entry(reg_old, 'E06b'), registry_entry(reg_new, 'E06b')
    added = [line for line in new_e.splitlines() if line not in old_e.splitlines()]
    require(REGISTRY_NEW.rstrip('\n') in added and REGISTRY_OLD.rstrip('\n') not in new_e and
            sorted(line.split(':')[0].strip() for line in added if not line.startswith('    config_file')) ==
            sorted(REGISTRY_ADDED_KEYS), 'registry E06b: config_file + m6 status fields only')
    require('    implementation_status: NOT_STARTED\n' in new_e and
            '    pinned_commit: 16b793a7564a4b9308cf94e62bdb2ffacb3a725a\n' in new_e, 'registry: M0 field and pin kept')
    native = reader(NATIVE).decode()
    tree = ast.parse(native)
    top_imports = {a.name.split('.')[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '.')])}
    require(not {'torch', 'yaml', 'cv2', 'PIL'} & top_imports, 'native.py imports no torch/yaml/image lib at module level')
    for token in ('from .adapter import DSDGAdapter', 'from . import microbatch_execution_v2 as mb2',
                  "return rng.choice(self.pools[self.spoof[index][1]])", 'manifests/dsdg_identity_pairs_v1.parquet is never written',
                  "filters=[('split', '==', 'TRAIN'), ('dataset', 'in', list(DATASETS))]",
                  "FINAL_EXECUTION_FIDELITY = 'PENDING_M6F_C_RUNTIME_QUALIFICATION'"):
        require(token in native, 'native.py token ' + token[:50])
    for forbidden in ('FROZEN_LAMBDAS', 'execution_guard(', 'IDFreePairDataset', 'pairs_train_v1', 'to_parquet',
                      'write_table', "'TEST'"):
        require(forbidden not in native, 'native.py must not use ' + forbidden)
    return {'config_sha256': CONFIG_SHA, 'snapshot_byte_identical': True, 'contract_sha256': CONTRACT_SHA,
            'native_sha256': sha(reader(NATIVE)), 'common_config_additive_only': True,
            'common_learned_additive_only': True, 'registry_change_confined_to_e06b': True,
            'unchanged_files': sorted(UNCHANGED)}


def build_evidence_from(d):
    return {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'method_id': 'E06b',
        'method_name': 'DSDG-NATIVE', 'track': 'B_NATIVE_FULL_SECONDARY', 'authority_commit': AUTHORITY,
        'config_path': CONFIG, 'config_sha256': CONFIG_SHA, 'snapshot_path': SNAPSHOT, 'snapshot_byte_identical': True,
        'freeze_record_role': 'methods/common/config.py LATER_FROZEN_METHOD_FILES anchors E06b to config_sha256 here',
        'statuses': STATUSES, 'target_fidelity': 'FAITHFUL_OFFICIAL',
        'final_execution_fidelity': 'PENDING_M6F_C_RUNTIME_QUALIFICATION',
        'native_population': {'train_datasets': ['casia_fasd', 'msu_mfsd'],
                              'siwmv2': 'NOT_INSTANTIABLE_MISSING_SUBJECT_ID',
                              'casia_fasd': {'spoof': 2520, 'live': 840, 'subjects': 35, 'print': 1680, 'replay': 840},
                              'msu_mfsd': {'spoof': 1200, 'live': 400, 'subjects': 25, 'print': 400, 'replay': 800},
                              'total': {'spoof': 3720, 'live': 1240, 'subjects': 60, 'print': 2080, 'replay': 1640},
                              'verified_by': 'tests/test_m6fb_e06b_static.py (TRAIN-filtered allowlisted split_v1 read)'},
        'spoof_type': {'vocabulary': ['print', 'replay'], 'class_index': {'print': 0, 'replay': 1}, 'K': 2,
                       'unexpected_macro': 'REJECT_NO_REMAP'},
        'native_relation': {'semantics': 'SAME_SUBJECT_ONLINE_RANDOM', 'rng': 'module random in the seeded worker',
                            'materialized': False, 'native_manifest_created': False,
                            'order': 'ascending sample_id bytewise (spoof index and per-subject live pools)'},
        'loader': {'batch_size': 240, 'num_workers': 8, 'persistent_workers': False, 'drop_last': False, 'shuffle': True,
                   'worker_init_fn': 'methods.common.learned.seed_torch_worker',
                   'worker_seed': '(per-epoch loader base seed + worker_id) mod 2**32',
                   'batch_assignment': 'batch b is served by worker b % 8'},
        'batch_plan': {'rows': 3720, 'batch_sizes': [240] * 15 + [120], 'optimizer_steps_per_epoch': 16,
                       'total_optimizer_steps': 3200, 'microbatch': 20, 'chunks_full_batch': 12, 'chunks_tail_batch': 6,
                       'execution_status': 'CANDIDATE_NOT_YET_QUALIFIED'},
        'losses': {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10, 'lambda_ort': 1, 'lambda_pair': 5,
                   'loss_cls_active': True, 'loss_pair_active': True},
        'reuse': {'from_e06c_unchanged': ['methods/dsdg/adapter.py DSDGAdapter.prepare/validate_source/verify_lightcnn/'
                                          'official_components (inherited)', 'methods/dsdg/source.py',
                                          'methods/dsdg/microbatch_execution.py (LOCAL/GLOBAL terms)',
                                          'methods/dsdg/microbatch_execution_v2.py epoch_batch_plan/chunk_slices_v2',
                                          'methods/common/learned.py seed_plan/seed_torch_worker/verify_source/mapping'],
                  'not_reused': ['E06c A1 guards (lambda_pair == 0, attack_type == 1)', 'training_graph.FROZEN_LAMBDAS',
                                 'IDFreePairDataset / pairs_train_v1 relation', 'M6D5d execution_guard (8838 rows)']},
        'shared_module_changes': {COMMON_CONFIG: 'additive LATER_FROZEN_METHOD_FILES (E06b) + freeze-record anchor',
                                  COMMON_LEARNED: "checkpoint_plan official-epoch branch accepts 'E06b'"},
        'registry': {'path': REGISTRY, 'change': 'E06b config_file + m6_status fields; no new source pin',
                     'implementation_status_m0_field': 'NOT_STARTED (pinned by historical test_m0_bootstrap)'},
        'generation': {'n_syn_intended': 'DEFERRED_TO_M8', 'blocks_m6': False},
        'static_derivation': d,
        'decision_operation': {'training_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'model_executions': 0,
                               'GPU_contacted': False, 'image_reads': 0, 'TEST_rows_materialized': 0,
                               'banks_generated': 0},
        'm6': {'M6_CLOSED': False, 'M7_started': False}}


def check_evidence(reader=worktree_reader, authority_reader=at_authority):
    d = derive(reader, authority_reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(d), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in ('M6FB', CONFIG_SHA, 'CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_NOT_YET_QUALIFIED',
                  'PRODUCTION_RUNNER_NOT_YET_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED',
                  'PENDING_M6F_C_RUNTIME_QUALIFICATION', '15 × 240 + 120', 'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in md, 'evidence report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6H authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]),
            'only the intended files may differ from the authority: ' + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for p in ('manifests/dsdg_identity_pairs_v1.parquet', 'outputs/audit/method_status.csv'):
        require(not (ROOT / p).exists(), 'must not exist: ' + p)
    base, now = at_authority(CONFIG_STATUS), read(CONFIG_STATUS)
    require(now.startswith(base) and len(now) > len(base), 'CONFIG_STATUS append-only')
    added = now[len(base):].decode()
    for token in ('M6FB', 'E06b', CONFIG, 'STATIC_ADAPTER_IMPLEMENTED', 'PENDING_M6F_C_RUNTIME_QUALIFICATION',
                  'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in added, 'CONFIG_STATUS M6FB note token ' + token)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M6FB candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, *MODIFIED):
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
    for p in MODIFIED:
        require(p in rows, 'modified file indexed at authority ' + p)
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
    'record_kind': 'CONFIG_FREEZE / STATIC_IMPLEMENTATION', 'method_id': 'E06b', 'method_name': 'DSDG-NATIVE',
    'track': 'B_NATIVE_FULL_SECONDARY', 'config_path': CONFIG, 'config_sha256': CONFIG_SHA, 'statuses': STATUSES,
    'target_fidelity': 'FAITHFUL_OFFICIAL', 'final_execution_fidelity': 'PENDING_M6F_C_RUNTIME_QUALIFICATION',
    'train_datasets': ['casia_fasd', 'msu_mfsd'], 'K': 2, 'lambda_pair': 5, 'train_spoof_rows': 3720,
    'batch_plan': '15 x 240 + 120; microbatch 20: 12 / 6', 'num_workers': 8, 'n_syn_intended': 'DEFERRED_TO_M8',
    'amendment_created': False, 'new_deviation': False, 'new_source_pin': False, 'gpu_graph_qualified': False,
    'production_runner_qualified': False, 'M6_closed': False, 'M7_started': False, 'TEST_access': False,
    'GPU_contacted': False, 'M8_bank': False, 'training_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0,
    'image_reads': 0, 'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': LEDGER_PREFIX_SHA,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
    'modified_existing_files': list(MODIFIED)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all((t['failures'] == 0 or t.get('failures_preexisting_at_authority')) and
                                 (t['errors'] == 0 or t.get('errors_preexisting_environmental')) for t in row['tests']),
            'ledger tests: failures/errors only when proven pre-existing at the authority')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6F-B: freeze the E06b DSDG-NATIVE config and implement the static adapter (no training)',
            'notes': ('configs/methods/e06b_dsdg_native.yaml frozen from the M6FA contract (snapshot byte-identical); '
                      'methods/dsdg/native.py: CASIA+MSU native same-subject relation (online random.choice, no '
                      'manifest), attack_macro K=2 {print:0, replay:1}, lambda_pair=5, loader 8 workers, plan 15x240+120 '
                      '(microbatch 20: 12/6). Shared-module changes additive only (config.py later-frozen registry; '
                      'learned.py checkpoint branch). GPU graph and production runner NOT qualified; fidelity pending '
                      'M6F-C. N_syn DEFERRED_TO_M8. M6 open; M7 not started.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest tests.test_m6fb_e06b_static; python3 -B '
                        'tools/m6fb_e06b_static_preflight.py --write-evidence-json / --before-ledger / --append-ledger / '
                        '--rebuild-index'),
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
            'artifact_rows_added': len(NEW), 'modified_existing_files': [*MODIFIED, *bookkeeping]}


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
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence_from(derive()), indent=2, sort_keys=True,
                                               ensure_ascii=False) + '\n')
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
