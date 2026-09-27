#!/usr/bin/env python3
"""Verify the M6D6iR E07c MAIN checkpoint-retention OWNER DECISION candidate; ledger/index LAST.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no checkpoint. The checkpoint
count, cadence, payload and retention clauses are re-derived from the pinned FAS_train.py AST and from committed
config/amendment/evidence text, never merely copied from the record. The decision record is the JSON subset of YAML.

Candidate-time checks (authority, branch, exact worktree, tracked files vs the M6D6i authority) run only here, once.
check_record/check_evidence take a reader so tests validate the state at the M6D6iR commit, not a moving HEAD.

  --before-ledger   decision + evidence + candidate checks, ledger not yet appended
  --rebuild-index   after the one-row ledger append: rebuild ARTIFACT_INDEX.csv LAST (CRLF, sorted)
  (default)         final check
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'a433b28f861631b1e7828ff247c32f3e3d37bcca'
BRANCH = 'm6-baselines'
MILESTONE = 'M6D6iR'
CLASSIFICATION = 'M6D6IR_E07C_MAIN_CHECKPOINT_RETENTION_OWNER_DECISION'
DECISION_CLASS = 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION'
RECORD_KIND = 'OWNER_DECISION / CHECKPOINT_RETENTION'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
SOURCE_COMMIT = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TRAIN_SCRIPT = 'third_party/source_cache/difffas/FAS_train.py'
TRAIN_SCRIPT_SHA = 'c5eb42a1193f7708b9e972db0faac602c82ffe2dc10578b6c9013d81e26e4f84'   # A7 overlay files_sha256
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 124
INDEX_BASELINE_ROWS = 735

RECORD = 'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml'
RECORD_SHA = '94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Checkpoint_Retention_Decision_M6D6iR.md'
EV_JSON = 'outputs/audit/M6D6IR_E07C_CHECKPOINT_RETENTION_DECISION.json'
EV_MD = 'outputs/audit/M6D6IR_E07C_CHECKPOINT_RETENTION_DECISION.md'
TESTS = 'tests/test_m6d6ir_e07c_checkpoint_retention.py'
PREFLIGHT = 'tools/m6d6ir_e07c_checkpoint_retention_preflight.py'
NEW = tuple(sorted((RECORD, DOC, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
BOOKKEEPING = (INDEX, LEDGER)
CONFIG = 'configs/methods/e07c_difffas_bin_idfree.yaml'
ADAPTATION = 'configs/frozen/difffas_bin_idfree_v1.yaml'
LOGGING = 'configs/run_logging_v1.yaml'
A2 = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md'
M4_AUDIT = 'outputs/audit/M4_IDFREE_MANIFEST_AUDIT.json'
BOUND = {
    'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx': SPEC_SHA,
    CONFIG: 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
    ADAPTATION: 'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
    LOGGING: 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md':
        '03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472',
    A2: 'b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d',
    'configs/amendments/e07c_a7_execution_policy.yaml': '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3',
    'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml': '6230e0b9531d47664f9eb14a7f87b24f67b0bf231a995adaba1294f944ec7693',
    M4_AUDIT: None,
    'methods/difffas/main_graph.py': 'cd85bf2e28ca88cf48c9a745f84cb1d66d5271e17a4a2bd5b083f1a1863c3411',
    'methods/difffas/main_graph_qualification.py': 'be0df3098c76c83f770957267cccffa5f03ab977164c9fbb4e4a6d4d3a5f2ebd',
    'outputs/audit/M6D6I_E07C_MAIN_GRAPH_QUALIFICATION.json': '089af4d51f9c1cd9592a81686115dfcd7683ec4fc04291db790515e66f79c1b8',
    'outputs/audit/M6D6I_E07C_MAIN_GRAPH_QUALIFICATION.md': '827e4d6145a7b6c15ecb20c7a969fc6762e5a43e9c3a48a96db32bc8876526a9',
    'outputs/audit/M6D6I_E07C_PROCESS_1.json': 'd6d15ec734938c3862c324aec73c4f39958ea999770e5813b22a57f8b5d2cf5a',
    'outputs/audit/M6D6I_E07C_PROCESS_2.json': '7bc909635f19be5f564c4fafd5194e82cd7b1ca7340c8a55277753c18fb75384',
    TRAIN_SCRIPT: TRAIN_SCRIPT_SHA,
}
PROTECTED = ('methods/difffas/main_graph.py', 'methods/difffas/main_graph_qualification.py',
             'methods/difffas/execution_policy.py', 'methods/difffas/aux_checkpoint.py', 'methods/difffas/aux_runner.py',
             'methods/difffas/aux_runner_io.py', 'methods/difffas/aux_resume.py', 'tools/run_e07c_aux.py', CONFIG,
             ADAPTATION, LOGGING, A2, 'configs/amendments/e07c_a7_execution_policy.yaml',
             'configs/amendments/e07c_a8_aux_resume_policy.yaml', 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml',
             'environments/e07c.lock.json', 'tests/test_m6d6i_e07c_main_graph.py',
             'tools/m6d6i_e07c_main_graph_preflight.py')
PAYLOAD = ['model', 'ema', 'scheduler', 'optimizer', 'conf']
SEEDS = [42, 1337, 2026]
INDEX_FIELDS = ['path', 'epoch', 'global_step', 'file_size_bytes', 'sha256', 'checkpoint_type', 'selected_for_final',
                'selection_reason']
GATE = ['checkpoint file creation completed', 'file closed successfully', 'path recorded', 'epoch recorded',
        'global_step recorded', 'file_size_bytes recorded', 'SHA256 computed from the completed file',
        'checkpoint_index.json entry durably written', 'SHA256/index verification passed']
QUALIFIED = ['E07c_MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN', 'MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN']
NOT_QUALIFIED = ['MAIN_PRODUCTION_RUNNER', 'MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
ZERO_OPERATION = {'training_runs': 0, 'scientific_runs': 0, 'qualification_optimizer_steps': 0, 'optimizer_steps': 0,
                  'backward_calls': 0, 'checkpoint_creations': 0, 'checkpoint_deletions': 0, 'checkpoint_moves': 0,
                  'checkpoint_deserializations': 0, 'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False,
                  'M8_outputs': 0, 'gpu_contacted': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6iR: ' + message)


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


# ----------------------------------------------------------------- static derivation (authority text + pinned AST)
def derive(reader=worktree_reader):
    """Re-derive cadence, trigger order, payload, iteration and checkpoint counts from source and frozen text."""
    raw = read(TRAIN_SCRIPT)                    # git-ignored pinned cache; identity fixed by the A7-bound SHA256
    require(sha(raw) == TRAIN_SCRIPT_SHA, 'pinned FAS_train.py SHA256')
    mod = ast.parse(raw)
    train = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    init = [n.lineno for n in train.body if isinstance(n, ast.Assign) and ast.unparse(n) == 'iters = 0']
    epoch = next(n for n in train.body if isinstance(n, ast.For))
    batch = next(n for n in epoch.body if isinstance(n, ast.For))
    stmts = {ast.unparse(s) if not isinstance(s, ast.If) else 'if ' + ast.unparse(s.test): s for s in batch.body}
    inc, save = stmts['iters = iters + 1'], stmts['if iters % args.save_checkpoints_every_iters == 0']
    require(init == [30] and epoch.lineno == 37 and inc.lineno == 42 and save.lineno == 102 and
            batch.body.index(inc) == 0 and batch.body.index(inc) < batch.body.index(save),
            'iters = 0 before the epoch loop; increment precedes the save test; never reset')
    call = next(n for n in ast.walk(save) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save')
    keys = [k.value for k in call.args[0].keys]
    require(keys == PAYLOAD and ast.unparse(call.args[0].values[0]) == 'model_module.state_dict()' and
            [ast.unparse(v) for v in call.args[0].values[1:]] ==
            ['ema.state_dict()', 'scheduler.state_dict()', 'optimizer.state_dict()', 'conf'], 'source save payload')
    defaults = {}
    for n in ast.walk(mod):
        if (isinstance(n, ast.Call) and ast.unparse(n.func) == 'parser.add_argument' and n.args and
                isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith('--')):
            kw = {k.arg: k.value for k in n.keywords}
            if 'default' in kw:
                defaults[n.args[0].value[2:]] = ast.literal_eval(kw['default'])
    loader = next(n for n in ast.walk(mod) if isinstance(n, ast.Assign) and n.lineno == 205)
    loader_kw = {k.arg: ast.unparse(k.value) for k in loader.value.keywords}
    require(loader_kw == {'batch_size': 'args.batch_size', 'shuffle': 'True'}, 'DataLoader passes no drop_last')
    cfg = reader(CONFIG).decode()
    for token in ('  batch_size: 4\n', '  max_epochs: 400\n', '  cadence: save_checkpoints_every_10000_iters\n',
                  '  rule: BASELINE_FINAL_STATE_V1\n', '  terminal_checkpoint_creation_required_if_cadence_misses_it: true\n',
                  '  additional_optimizer_steps_permitted: false\n', '  experiment_seeds: [42, 1337, 2026]\n',
                  '  checkpoint_bytes: RETAIN_PER_OFFICIAL_CADENCE\n',
                  '  never_delete: [authoritative_final_checkpoint, selected_checkpoint, officially_required_checkpoint]\n',
                  '  pruning_in_m6b: false\n', '  if_pruned_later: metadata_and_sha256_records_must_remain\n',
                  '  selection_uses_val: false\n', '  selection_uses_test: false\n', 'n_syn_intended: 8838\n'):
        require(token in cfg, 'E07c config ' + token.strip())
    require(defaults['batch_size'] == 4 and defaults['max_epochs'] == 400 and
            defaults['save_checkpoints_every_iters'] == 10000, 'pinned argparse defaults 4 / 400 / 10000')
    adaptation = reader(ADAPTATION).decode()
    require(re.search(r'\ntraining_manifest:\n(?:  .*\n)*?  rows: 8838\n', adaptation), 'A1 training_manifest.rows 8838')
    n_train = json.loads(reader(M4_AUDIT))['difffas']['rows']
    require(n_train == 8838, 'M4 id-free manifest audit rows 8838')
    logging = reader(LOGGING).decode()
    for token in ('  silent_pruning: FORBIDDEN\n', '  if_pruned_later: "metadata and SHA256 records MUST remain in '
                  'checkpoint_index.json"\n', '  pruning_performed_in_m6b: false\n',
                  '  record_every_checkpoint_actually_written: true\n'):
        require(token in logging, 'run_logging_v1 ' + token.strip())
    block = logging.split('checkpoint_history:', 1)[1].split('checkpoint_type_enum', 1)[0]
    require(re.findall(r'\n    - (\w+)', block) == INDEX_FIELDS, 'run_logging_v1 checkpoint_history.required_fields')
    a2 = reader(A2).decode()
    require('If periodic upstream checkpoint saving does not save the exact terminal state, save **one\n   additional '
            'TERMINAL checkpoint** immediately after the final scheduled training step/epoch, with\n   **ZERO additional '
            'optimizer steps**.' in a2 and '| `E07c-5` | E07c (DiffFAS-BIN-IDFREE) |' in a2, 'A2-06 clause 3 / E07c-5')
    batch_size, epochs, cadence = defaults['batch_size'], defaults['max_epochs'], defaults['save_checkpoints_every_iters']
    per_epoch = math.ceil(n_train / batch_size)            # drop_last=False: the tail batch is kept
    total = per_epoch * epochs
    periodic = list(range(cadence, total + 1, cadence))
    terminal_extra = total % cadence != 0
    return {'train_rows': n_train, 'batch_size': batch_size, 'drop_last': False, 'iterations_per_epoch': per_epoch,
            'last_batch_size': n_train - (per_epoch - 1) * batch_size, 'epochs': epochs, 'total_iterations': total,
            'cadence_iterations': cadence, 'save_at_global_step_0': False,
            'periodic_global_steps': {'first': periodic[0], 'last': periodic[-1], 'step': cadence},
            'periodic_checkpoint_count': len(periodic), 'terminal_global_step': total,
            'terminal_remainder': total % cadence, 'terminal_checkpoint_count': int(terminal_extra),
            'created_checkpoint_count': len(periodic) + int(terminal_extra), 'payload_keys': keys,
            'source_lines': {'iters_init': 30, 'epoch_loop': 37, 'iters_increment': 42, 'save_test': 102,
                             'torch_save': call.lineno, 'dataloader': 205},
            'pinned_defaults': {k: defaults[k] for k in ('batch_size', 'max_epochs', 'save_checkpoints_every_iters')},
            'experiment_seeds': SEEDS, 'index_required_fields': INDEX_FIELDS}


# ----------------------------------------------------------------- decision record / evidence (reader-scoped)
def check_record(reader=worktree_reader):
    raw = reader(RECORD)
    require(sha(raw) == RECORD_SHA, 'decision record SHA256 == pinned constant')
    r = json.loads(raw)
    require(r['record_document'] == {'path': DOC, 'sha256': sha(reader(DOC))}, 'record document binding')
    for rel, digest in r['bound_authority_sha256'].items():
        body = read(rel) if rel == TRAIN_SCRIPT else reader(rel)
        require(sha(body) == digest and (BOUND[rel] in (None, digest)), 'bound authority ' + rel)
    require(sorted(r['bound_authority_sha256']) == sorted(BOUND), 'bound authority set')
    require((r['milestone'], r['method_id'], r['classification'], r['record_kind'], r['authority_commit'],
             r['amendment_created'], r['a9_created'], r['fidelity'], r['fidelity_class'], r['deviation'],
             r['new_deviation'], r['new_fidelity_class'], r['historical_configs_rewritten']) ==
            (MILESTONE, 'E07c', DECISION_CLASS, RECORD_KIND, AUTHORITY, False, False,
             {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
              'new_fidelity_class': False}, 'CONTROLLED_ADAPTATION', 'DEV-021', False, False, False), 'identity')
    require(r['source']['commit'] == SOURCE_COMMIT and r['source']['official_source_modified'] is False, 'source pin')
    s = r['scope']
    require(s['applies_to'] == 'E07c main scientific runs' and s['experiment_seeds'] == SEEDS and
            s['run_root_template'] == '<runtime_root>/runs/m6/E07c/seed_<seed>/' and
            {'M6D6g auxiliary encoder run', 'encoder_final.pkl', 'auxiliary resume sidecars', 'qualification artifacts',
             'E01', 'E02', 'E03', 'E04', 'E05', 'E06c', 'M7', 'M8', 'any other method'} == set(s['excludes']), 'scope')
    d = derive(reader)
    c = r['checkpoint_creation']
    for k in ('train_rows', 'batch_size', 'drop_last', 'iterations_per_epoch', 'last_batch_size', 'epochs',
              'total_iterations', 'cadence_iterations', 'save_at_global_step_0', 'periodic_global_steps',
              'periodic_checkpoint_count', 'terminal_global_step', 'terminal_checkpoint_count', 'created_checkpoint_count'):
        require(c[k] == d[k], 'record checkpoint_creation.' + k + ' == source/authority derivation')
    require(c['unchanged'] and not c['skip_save_authorized'] and not c['cadence_change_authorized'], 'creation unchanged')
    p = r['payload']
    require(p['unchanged'] and p['keys'] == PAYLOAD == d['payload_keys'] and
            not any(p[k] for k in p if k.endswith('_authorized')) and
            {'weights_only_authorized', 'optimizer_strip_authorized', 'dtype_change_authorized', 'compression_authorized',
             'serialization_api_change_authorized', 'payload_reduction_authorized', 'payload_rewrite_authorized'} <= set(p),
            'payload unchanged; no rewrite / weights-only / dtype / compression authority')
    sel = r['selection']
    require((sel['rule'], sel['final_checkpoint'], sel['selected_checkpoint'], sel['val_selects'], sel['test_selects'],
             sel['loss_selects'], sel['early_stopping'], sel['changed']) ==
            ('BASELINE_FINAL_STATE_V1', 'terminal', 'terminal', False, False, False, False, False), 'selection unchanged')
    pr = r['pruning']
    require(pr['authorized'] is True and pr['successor_gate'] == GATE and pr['successor_gate_all_required'] is True and
            pr['gated_on'] == 'direct successor C_{k+1}' and pr['prune_before_successor_verification'] == 'FORBIDDEN' and
            pr['prune_immediately_after_own_save'] == 'FORBIDDEN' and
            pr['terminal_transition']['last_periodic_global_step'] == d['periodic_global_steps']['last'] and
            {'background deletion', 'unlogged cleanup', 'glob deletion', 'removal of unknown paths',
             'early predecessor deletion to make room'} <= set(pr['forbidden']), 'pruning authorization + successor gate')
    f = pr['failure_policy']
    require([f[k] for k in ('successor_write_failure', 'successor_hash_failure', 'successor_index_failure',
                            'successor_verification_failure')] == ['KEEP_PREDECESSOR'] * 4 and
            f['prune_failure'].startswith('STOP_AND_REPORT') and
            f['insufficient_space_for_successor'].startswith('STOP_AND_REPORT'), 'fail-closed')
    pt = r['protected']
    require(set(pt['roles']) == {'authoritative_final_checkpoint', 'selected_checkpoint', 'officially_required_checkpoint',
                                 'terminal'} and pt['pruning'] == 'FORBIDDEN' and
            pt['one_physical_file_may_satisfy_all_roles'] is True and pt['duplicate_byte_copies_required'] is False and
            str(d['terminal_global_step']) in pt['resolve_to'], 'terminal protection + role deduplication')
    m = r['metadata_retention']
    require(m['records_never_removed'] is True and m['required_fields_unchanged'] == INDEX_FIELDS == d['index_required_fields']
            and m['retained_fields'] == {'bytes_present': True, 'bytes_pruned': False} and
            (m['pruned_fields']['bytes_present'], m['pruned_fields']['bytes_pruned'], m['pruned_fields']['prune_reason']) ==
            (False, True, 'EXPLICIT_E07C_PERIODIC_RETENTION_POLICY') and
            {'pruned_utc', 'successor_checkpoint_global_step', 'successor_checkpoint_sha256'} <= set(m['pruned_fields']),
            'metadata survives pruning')
    require(r['silent_pruning'] == 'FORBIDDEN', 'silent pruning forbidden')
    require(set(r['science_unchanged'].values()) == {False} and len(r['science_unchanged']) == 18, 'no science change')
    require(r['resume'] == {'MAIN_CHECKPOINT_RESUME': 'UNQUALIFIED', 'pruning_proves_resume': False,
                            'future_resume_may_rely_only_on_present_bytes': True,
                            'conflict_policy': 'STOP_AND_RESOLVE_PROSPECTIVELY'}, 'resume unqualified')
    fi = r['future_implementation']
    require(fi['milestone'] == 'M6D6j' and len(fi['obligations']) == 10 and len(fi['required_tests']) == 10, 'M6D6j contract')
    sr = r['storage_rationale']
    require(sr['label'] == 'RESOURCE_PLANNING_ESTIMATE' and sr['binding'] is False and
            sr['previous_feasibility_verdict'] == 'GO_IF_OWNER_APPROVES_RETENTION_CHANGE', 'storage rationale non-binding')
    st = r['statuses']
    require(st['qualified'] == QUALIFIED and st['not_qualified'] == NOT_QUALIFIED and
            st['m6d6i_statuses_requalified'] is False and st['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'statuses')
    require(r['decision_operation'] == ZERO_OPERATION, 'no runtime / scientific activity')
    doc = reader(DOC).decode()
    for token in ('DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'no A9 is created', '884,000', '88', '89', '10,000',
                  '880,000', 'MUST NOT be pruned', 'STOP_AND_REPORT', 'FORBIDDEN', 'MAIN_CHECKPOINT_RESUME remains unqualified',
                  'RESOURCE_PLANNING_ESTIMATE', 'EXPLICIT_E07C_PERIODIC_RETENTION_POLICY', 'DEV-021'):
        require(token in doc, 'decision document token ' + token)
    return r, d


def check_evidence(reader=worktree_reader):
    record, d = check_record(reader)
    ev = json.loads(reader(EV_JSON))
    require((ev['milestone'], ev['classification'], ev['decision_classification'], ev['record_kind'], ev['status'],
             ev['authority_commit'], ev['record_sha256'], ev['record_document_sha256'], ev['amendment_created']) ==
            (MILESTONE, CLASSIFICATION, DECISION_CLASS, RECORD_KIND, 'PASS', AUTHORITY, RECORD_SHA,
             record['record_document']['sha256'], False), 'evidence identity')
    require(ev['static_derivation'] == d, 'evidence derivation == live static derivation')
    require(ev['decision_operation'] == ZERO_OPERATION and ev['qualified_statuses'] == QUALIFIED and
            ev['not_qualified'] == NOT_QUALIFIED and ev['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'evidence statuses')
    require(ev['storage_rationale'] == record['storage_rationale'], 'evidence storage rationale (non-binding)')
    md = reader(EV_MD).decode()
    for token in (*QUALIFIED[:1], *NOT_QUALIFIED, RECORD_SHA, 'RESOURCE_PLANNING_ESTIMATE', 'GO_IF_OWNER_APPROVES_RETENTION_CHANGE',
                  'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', '884000', 'M6D6j', 'IMPLEMENTED_NOT_EXECUTED'):
        require(token in md, 'evidence report token ' + token)
    return ev


def build_evidence():
    record, d = check_record()
    return {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'decision_classification': DECISION_CLASS,
            'record_kind': RECORD_KIND, 'status': 'PASS', 'method_id': 'E07c', 'authority_commit': AUTHORITY,
            'record_path': RECORD, 'record_sha256': RECORD_SHA, 'record_document_path': DOC,
            'record_document_sha256': record['record_document']['sha256'],
            'bound_authority_sha256': record['bound_authority_sha256'], 'amendment_created': False,
            'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
            'new_fidelity_class': False, 'static_derivation': d,
            'derivation_method': ('stdlib ast of pinned FAS_train.py (SHA256-checked) + committed config / A1 / A2 / '
                                  'run_logging_v1 / M4 evidence text; no torch, no data, no manifest decode'),
            'storage_rationale': record['storage_rationale'], 'decision_operation': ZERO_OPERATION,
            'qualified_statuses': QUALIFIED, 'not_qualified': NOT_QUALIFIED, 'method_status': 'IMPLEMENTED_NOT_EXECUTED',
            'm6d6i_statuses_requalified': False}


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6i authority')


def tracked_unchanged(expected_changed):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted(expected_changed), 'only ledger/index may differ from the authority: ' + json.dumps(changed))
    for rel in PROTECTED:
        require(read(rel) == at_authority(rel), 'protected byte-identical ' + rel)
    for rel in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').decode().splitlines():
        if Path(rel).name.startswith(tuple(f'M6D6{c}_' for c in 'ABCDEFGHI')):
            require(read(rel) == at_authority(rel), 'historical evidence byte-identical ' + rel)
    require(not [p for p in git('ls-files', '--others', '--exclude-standard').decode().split()
                 if 'Amendment_A9' in p or '_a9_' in p], 'no A9')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def worktree(bookkeeping_modified):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    expected = ['?? ' + p for p in NEW] + ([' M ' + p for p in BOOKKEEPING] if bookkeeping_modified else [])
    require(status == sorted(expected), 'worktree holds exactly the M6D6iR candidate: ' + json.dumps(status))


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
    'final_status': 'PASS', 'decision_classification': DECISION_CLASS, 'record_kind': RECORD_KIND,
    'amendment_created': False, 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021',
    'new_deviation': False, 'new_fidelity_class': False, 'checkpoint_cadence_changed': False,
    'checkpoint_payload_changed': False, 'checkpoint_selection_changed': False, 'training_changed': False,
    'periodic_pruning_authorized': True, 'silent_pruning_forbidden': True,
    'successor_verification_required_before_prune': True, 'terminal_checkpoint_permanently_retained': True,
    'periodic_checkpoint_count_per_seed': 88, 'terminal_checkpoint_count_per_seed': 1, 'checkpoints_created_per_seed': 89,
    'scientific_runs': 0, 'optimizer_steps': 0, 'backward_calls': 0, 'checkpoints_created_this_milestone': 0,
    'checkpoints_pruned_this_milestone': 0, 'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False,
    'M8_bank': False, 'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'qualified_statuses': QUALIFIED[:1], 'not_qualified': NOT_QUALIFIED, 'method_status': 'IMPLEMENTED_NOT_EXECUTED',
    'record_sha256': RECORD_SHA, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(row['committed_prefix_sha256'] == sha(prefix), 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted(NEW), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)


def verify(stage):
    authority()
    tracked_unchanged({'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage])
    whitespace()
    check_evidence()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and current.startswith(prefix),
            f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
        worktree(False)
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
        worktree(True)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'new_files': list(NEW),
            'modified_existing_files': {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage],
            'record_sha256': RECORD_SHA, 'torch_imported': False, 'checkpoint_bytes_opened': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--write-evidence-json', action='store_true',
                        help='candidate creation only: write the JSON evidence from the static derivation')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence(), indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': 'WRITTEN', 'path': EV_JSON}))
        return
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
