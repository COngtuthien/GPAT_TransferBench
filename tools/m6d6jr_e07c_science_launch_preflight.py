#!/usr/bin/env python3
"""Verify the M6D6jR E07c SCIENCE-LAUNCH OWNER DECISION candidate (O3 smoke timing, O4 resume); ledger/index LAST.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no checkpoint. The O3/O4 evidence is
re-derived from the frozen specification bytes (milestone order, the section 26 / 32 smoke language, the section 27
same-seed failure policy) and from the committed M6D6j CLI / runner / contract / evidence, never merely copied.
Candidate-time checks run only here; check_record / check_evidence take a reader (no moving-HEAD lock).

  --write-evidence-json   candidate creation only
  --before-ledger / --rebuild-index / (default final)
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'ade5900db1dbff73917a2eb2280f8f2fe7afc418'
BRANCH = 'm6-baselines'
MILESTONE = 'M6D6jR'
CLASSIFICATION = 'M6D6JR_E07C_SCIENCE_LAUNCH_OWNER_DECISION'
DECISION_CLASS = 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION'
RECORD_KIND = 'OWNER_DECISION / SCIENCE_LAUNCH_POLICY'
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
FROZEN_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 126
INDEX_BASELINE_ROWS = 752
RECORD = 'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml'
RECORD_SHA = '2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Science_Launch_Decision_M6D6jR.md'
EV_JSON = 'outputs/audit/M6D6JR_E07C_SCIENCE_LAUNCH_DECISION.json'
EV_MD = 'outputs/audit/M6D6JR_E07C_SCIENCE_LAUNCH_DECISION.md'
TESTS = 'tests/test_m6d6jr_e07c_science_launch_decision.py'
PREFLIGHT = 'tools/m6d6jr_e07c_science_launch_preflight.py'
NEW = tuple(sorted((RECORD, DOC, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
BOOKKEEPING = (INDEX, LEDGER)
CONTRACT = 'configs/amendments/e07c_m6d6j_main_production_runner_contract.yaml'
M6D6J_EVIDENCE = 'outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.json'
CONFIG = 'configs/methods/e07c_difffas_bin_idfree.yaml'
CLI = 'tools/run_e07c_main.py'
RUNNER = 'methods/difffas/main_runner.py'
RUNNER_IO = 'methods/difffas/main_runner_io.py'
BOUND = {SPEC: SPEC_SHA, CONFIG: 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
         'configs/run_logging_v1.yaml': 'd33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15',
         'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml':
             '6230e0b9531d47664f9eb14a7f87b24f67b0bf231a995adaba1294f944ec7693',
         'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml':
             '94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6',
         CONTRACT: '9528d53e536e3e6002db30d9d5300c4abb72ef1e653b1d53ba81dc3af5c7c422',
         M6D6J_EVIDENCE: 'd8a3ae9c6aa6876006606b6b89fcbca02d56e60ebb35d19f8d151e2b0c3d668f',
         CLI: '7defa308b166a9aa09effb97f424c1ae625ca0fc69f73f12740a2e7d9ca75569',
         RUNNER: 'cb2f17f9b3c6d4b86215e2846a26ac968f8806c57996277fe815f20d71fdc708',
         RUNNER_IO: '3bf0239e3809a3cf2f82223e02db6f5aad1a613e708afc31c018f0394379bc20',
         'methods/difffas/main_checkpoint.py': '416dcbeecf041af253f0ab598fed02784d1b47077ba86cf5d2b5c6e1ff915d31'}
MILESTONES = ['M6 Baselines', 'M7 GPAT', 'M8 Banks', 'M9 Generator eval', 'M10 ResNet downstream', 'M11 DINOv3 downstream']
SMOKE_CLAUSE = 'A full deterministic smoke test on a 2-subject-per-class toy subset must finish before full runs.'
SMOKE_CHECKLIST = 'smoke test passes for generator → bank → evaluator → metrics → plots.'
FAILURE_ROW = ('One seed fails', 'rerun same seed after technical fix; never replace with a different seed')
QUALIFIED = ['E07c_MAIN_SCIENTIFIC_LAUNCH_POLICY_RESOLVED', 'E07c_MAIN_SCIENTIFIC_RUNS_AUTHORIZED_TO_LAUNCH']
NOT_QUALIFIED = ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
ZERO = {'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'TRAIN_image_reads': 0,
        'VAL_image_reads': 0, 'TEST_image_reads': 0, 'GPU_contacted': False}
RUN_CONTRACT = {'experiment_seeds': [42, 1337, 2026], 'epochs': 400, 'batch_size': 4, 'drop_last': False,
                'iterations_per_epoch': 2210, 'total_iterations': 884000, 'entrypoint': CLI,
                'visualization': 'exact source every 1000 steps (M6D6j D1)', 'periodic_checkpoint_every': 10000,
                'terminal_checkpoint': '884000 after the final visualization (M6D6j D2)',
                'retention': 'M6D6iR successor-gated pruning', 'changed': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6jR: ' + message)


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


# ----------------------------------------------------------------- static derivation (reader-scoped)
def derive(reader=worktree_reader):
    raw = reader(SPEC)
    require(sha(raw) == SPEC_SHA, 'frozen specification SHA256')
    paras = spec_paragraphs(raw)
    pos = [paras.index(m) for m in MILESTONES]
    require(pos == sorted(pos), 'frozen milestone order M6 < M7 < M8 < M9 < M10 < M11')
    smoke, check = paras.index(SMOKE_CLAUSE), [i for i, p in enumerate(paras) if p.endswith(SMOKE_CHECKLIST)]
    require(len(check) == 1 and paras.index('32. Final go/no-go checklist before any full training') < check[0],
            'section 32 smoke checklist item')
    fail = paras.index(FAILURE_ROW[0])
    require(paras[fail + 1] == FAILURE_ROW[1], 'section 27 failure policy: rerun same seed after technical fix')
    cli = ast.parse(reader(CLI))
    options = [ast.literal_eval(n.args[0]) for n in ast.walk(cli) if isinstance(n, ast.Call) and
               ast.unparse(n.func) == 'parser.add_argument']
    require(options == ['--seed', '--execution-config', '--preflight-only'], 'CLI options: no resume path')
    require("raise Refused(f'{run_dir} already exists; it is never overwritten and MAIN_CHECKPOINT_RESUME is unqualified')"
            in ast.unparse(cli), 'CLI refuses an existing scientific run root (fresh only)')
    io_tree = ast.parse(reader(RUNNER_IO))
    refused = next(ast.literal_eval(n.value) for n in io_tree.body if isinstance(n, ast.Assign) and
                   ast.unparse(n.targets[0]) == 'REFUSED_OPTIONS')
    resume_flags = sorted(f for f in refused if 'resume' in f or 'pretrain' in f)
    require({'--resume', '--resume-latest', '--resume-state', '--resume-from', '--auto-resume', '--pretrain-path'}
            <= set(resume_flags), 'resume flags refused before parsing')
    seeds = next(ast.literal_eval(n.value) for n in io_tree.body if isinstance(n, ast.Assign) and
                 ast.unparse(n.targets[0]) == 'EXPERIMENT_SEEDS')
    require(seeds == (42, 1337, 2026) and '  experiment_seeds: [42, 1337, 2026]\n' in reader(CONFIG).decode(), 'seeds')
    runner = reader(RUNNER).decode()
    require("'overwritten; MAIN_CHECKPOINT_RESUME is unqualified)')" in runner and
            'self.method_id, self.config, self.runtime_root, self.resume = (mio.METHOD_ID, config, '
            'Path(runtime_root), False)' in ast.unparse(ast.parse(runner)),
            'run context: fresh only (resume = False; existing root refused)')
    contract = json.loads(reader(CONTRACT))
    ev = json.loads(reader(M6D6J_EVIDENCE))
    require('MAIN_PRODUCTION_RUNNER' in ev['qualified_statuses'] and ev['status'] == 'PASS' and
            ev['MAIN_CHECKPOINT_RESUME'] == 'UNQUALIFIED' and ev['not_qualified'] == NOT_QUALIFIED and
            ev['FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION'] is True and
            contract['full_science_blockers'] == {'FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION': True,
                                                  'O3_smoke_test': 'PENDING', 'O4_resume_before_science': 'PENDING'} and
            contract['resume'] == {'MAIN_CHECKPOINT_RESUME': 'UNQUALIFIED', 'pretrain_path': None, 'cli': 'refused'},
            'M6D6j: runner qualified; resume unqualified; O3/O4 were the recorded pending blockers')
    loop = contract['loop']
    require((loop['batch_size'], loop['drop_last'], loop['iterations_per_epoch'], loop['epochs'], loop['total_iterations'],
             contract['checkpoint']['cadence'], contract['checkpoint']['terminal_step'],
             contract['owner_decisions']['D1_visualization']['cadence']) ==
            (4, False, 2210, 400, 884000, 10000, 884000, 1000), 'M6D6j run contract values')
    return {'spec_milestone_paragraphs': dict(zip(MILESTONES, pos)), 'spec_smoke_clause_paragraph': smoke,
            'spec_smoke_checklist_paragraph': check[0], 'spec_failure_policy_paragraph': fail,
            'milestone_order_verified': True, 'smoke_clause_present': True, 'smoke_checklist_present': True,
            'failure_policy_same_seed_rerun': True, 'cli_options': options, 'cli_refused_resume_flags': resume_flags,
            'cli_fresh_only': True, 'experiment_seeds': list(seeds), 'm6d6j_production_runner_qualified': True,
            'm6d6j_resume': 'UNQUALIFIED', 'm6d6j_pending_blockers': contract['full_science_blockers'],
            'run_contract': {k: loop[k] for k in ('batch_size', 'drop_last', 'iterations_per_epoch', 'epochs',
                                                   'total_iterations')} | {
                'checkpoint_cadence': 10000, 'terminal_step': 884000, 'visualization_cadence': 1000}}


def check_record(reader=worktree_reader):
    raw = reader(RECORD)
    require(sha(raw) == RECORD_SHA, 'decision record SHA256 == pinned constant')
    r = json.loads(raw)
    require(r['record_document'] == {'path': DOC, 'sha256': sha(reader(DOC))}, 'record document binding')
    require(r['bound_authority_sha256'] == BOUND, 'bound authority set and values')
    for rel, digest in BOUND.items():
        require(sha(reader(rel)) == digest, 'bound authority bytes ' + rel)
    require((r['milestone'], r['method_id'], r['classification'], r['record_kind'], r['authority_commit'],
             r['amendment_created'], r['a9_created'], r['fidelity'], r['new_deviation'], r['new_fidelity_class'],
             r['historical_configs_rewritten']) ==
            (MILESTONE, 'E07c', DECISION_CLASS, RECORD_KIND, AUTHORITY, False, False,
             {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
              'new_fidelity_class': False}, False, False, False), 'identity')
    require(r['frozen_encoder'] == {'sha256': FROZEN_SHA, 'bytes': 185136819} and
            r['source']['commit'] == '23f40519ec25a833ebc06842aa6fbab74fad4d15', 'frozen encoder / source pin')
    o3 = r['O3_smoke_test']
    require((o3['decision'], o3['resolution'], o3['smoke_test_required'], o3['smoke_test_completed'],
             o3['smoke_test_waived'], o3['smoke_test_deleted'], o3['blocks_E07c_M6_training'],
             o3['blocks_full_M8_plus_execution_until_completed'], o3['must_exercise'], o3['m6d6j_is_the_smoke_test'],
             o3['scientific_results_from_smoke'], o3['spec_milestone_order'], o3['spec_smoke_clause'],
             o3['spec_checklist_item']) ==
            ('STAGED_MILESTONE_PRECEDENCE_CLARIFICATION', 'RESOLVED_STAGED_MILESTONE_PRECEDENCE', True, False, False,
             False, False, True, ['generator', 'bank', 'evaluator', 'metrics', 'plots'], False, 'FORBIDDEN', MILESTONES,
             SMOKE_CLAUSE, SMOKE_CHECKLIST), 'O3')
    o4 = r['O4_resume']
    require((o4['decision'], o4['resolution'], o4['resume_required_before_science'], o4['resume_qualified'],
             o4['MAIN_CHECKPOINT_RESUME'], o4['m6d6k_prerequisite_for_science'], o4['scientific_runs'],
             o4['resume_flag_or_path_allowed'], o4['recovery_policy']) ==
            ('RESUME_NOT_REQUIRED_BEFORE_SCIENCE', 'RESOLVED_NOT_REQUIRED', False, False, 'UNQUALIFIED', False,
             'FRESH_ONLY', False, 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX'), 'O4')
    require({'substitute another seed', 'rerun to obtain a better result'} <= set(o4['forbidden']), 'O4 forbidden')
    require(r['launch'] == {'FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION': False,
                            'E07c_main_scientific_training': 'AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED',
                            'scientific_training_completed': False}, 'launch status')
    require(r['scientific_run_contract_unchanged'] == RUN_CONTRACT, 'scientific run contract unchanged')
    st = r['statuses']
    require(st['qualified'] == QUALIFIED and st['remains_qualified'] == ['MAIN_PRODUCTION_RUNNER'] and
            st['not_qualified'] == NOT_QUALIFIED and st['method_status'] == 'IMPLEMENTED_NOT_EXECUTED', 'statuses')
    require(r['decision_operation'] == ZERO, 'no runtime activity')
    d = derive(reader)
    doc = reader(DOC).decode()
    for token in ('STAGED_MILESTONE_PRECEDENCE_CLARIFICATION', 'RESUME_NOT_REQUIRED_BEFORE_SCIENCE', 'no A9 is created',
                  'HARD GATE', 'NOT deleted or waived', 'FRESH_ONLY', 'SAME experiment seed', 'DEV-021',
                  'AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED', '884000'):
        require(token in doc, 'decision document token ' + token)
    return r, d


def build_evidence():
    return build_evidence_from(*check_record())


def check_evidence(reader=worktree_reader):
    r, d = check_record(reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(r, d), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in (*QUALIFIED, *NOT_QUALIFIED, 'MAIN_PRODUCTION_RUNNER', RECORD_SHA, 'RESOLVED_STAGED_MILESTONE_PRECEDENCE',
                  'RESOLVED_NOT_REQUIRED', 'smoke_test_completed = false', 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX',
                  'tools/run_e07c_main.py --seed 42', 'NOT EXECUTED'):
        require(token in md, 'evidence report token ' + token)
    return ev


def build_evidence_from(r, d):
    return {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'decision_classification': DECISION_CLASS,
            'record_kind': RECORD_KIND, 'status': 'PASS', 'method_id': 'E07c', 'authority_commit': AUTHORITY,
            'record_path': RECORD, 'record_sha256': RECORD_SHA, 'record_document_sha256': r['record_document']['sha256'],
            'bound_authority_sha256': BOUND, 'static_derivation': d, 'O3': r['O3_smoke_test']['resolution'],
            'O4': r['O4_resume']['resolution'], 'launch': r['launch'], 'qualified_statuses': QUALIFIED,
            'not_qualified': NOT_QUALIFIED, 'method_status': 'IMPLEMENTED_NOT_EXECUTED', 'decision_operation': ZERO,
            'amendment_created': False, 'new_deviation': False, 'new_fidelity_class': False}


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6j authority')


def tracked_unchanged(expected_changed):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted(expected_changed), 'only ledger/index may differ from the authority: ' + json.dumps(changed))
    require(not [p for p in git('ls-files', '--others', '--exclude-standard').decode().split()
                 if 'Amendment_A9' in p or '_a9_' in p], 'no A9')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in bookkeeping]),
            'worktree holds exactly the M6D6jR candidate: ' + json.dumps(status))


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
    'amendment_created': False, 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
    'new_fidelity_class': False, 'O3_smoke_test': 'RESOLVED_STAGED_MILESTONE_PRECEDENCE',
    'O4_resume_before_science': 'RESOLVED_NOT_REQUIRED', 'FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION': False,
    'smoke_test_required': True, 'smoke_test_completed': False, 'blocks_E07c_M6_training': False,
    'blocks_full_M8_plus_execution_until_completed': True, 'resume_required_before_science': False,
    'resume_qualified': False, 'recovery_policy': 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX',
    'MAIN_CHECKPOINT_RESUME': 'UNQUALIFIED', 'MAIN_PRODUCTION_RUNNER': 'QUALIFIED',
    'E07c_main_scientific_training': 'AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED',
    'scientific_training_completed': False, 'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0,
    'checkpoint_writes': 0, 'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False, 'GPU_contacted': False,
    'M8_bank': False, 'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'qualified_statuses': QUALIFIED, 'not_qualified': NOT_QUALIFIED, 'method_status': 'IMPLEMENTED_NOT_EXECUTED',
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
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW), 'modified_existing_files': changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--write-evidence-json', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence(), indent=2, sort_keys=True, ensure_ascii=False) + '\n')
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
