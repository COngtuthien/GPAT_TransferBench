#!/usr/bin/env python3
"""Verify the M6D6jF E07c SCIENTIFIC RunContext seed-metadata HOTFIX candidate; ledger/index LAST.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU, no data, no checkpoint. The runner change is
re-derived from the AST of main_runner.py at the authority and in the candidate: every top-level node other than
E07cMainRunContext is identical, and inside it only the seed-metadata methods differ. check_record / check_evidence take
a reader (no moving-HEAD lock); candidate-time checks (authority HEAD, worktree, staging) run only here.

  --write-evidence-json   candidate creation only
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
AUTHORITY = '8358d8b6fe468478ad86a715616becb81bb9b339'
BRANCH = 'm6-baselines'
MILESTONE = 'M6D6jF'
CLASSIFICATION = 'M6D6JF_E07C_SCIENTIFIC_CONTEXT_HOTFIX'
DECISION_CLASS = 'DETERMINISTIC_IMPLEMENTATION_CORRECTION'
RECORD_KIND = 'TECHNICAL_HOTFIX / IMPLEMENTATION_BUGFIX'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 127
INDEX_BASELINE_ROWS = 758
RECORD = 'configs/amendments/e07c_m6d6jf_scientific_context_hotfix.yaml'
RECORD_SHA = '946ca134ef514650002f368a8d87715f16a80855d10725587d9e6e855747af13'
EV_JSON = 'outputs/audit/M6D6JF_E07C_SCIENTIFIC_CONTEXT_HOTFIX.json'
EV_MD = 'outputs/audit/M6D6JF_E07C_SCIENTIFIC_CONTEXT_HOTFIX.md'
TESTS = 'tests/test_m6d6jf_e07c_scientific_context_hotfix.py'
PREFLIGHT = 'tools/m6d6jf_e07c_scientific_context_preflight.py'
NEW = tuple(sorted((RECORD, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
RUNNER = 'methods/difffas/main_runner.py'
RUNNER_AUTHORITY_SHA = 'cb2f17f9b3c6d4b86215e2846a26ac968f8806c57996277fe815f20d71fdc708'
CONTEXT_CLASS = 'E07cMainRunContext'
CHANGED_METHODS = ['_manifest', '_open_locked', '_resolved_config', '_seed_metadata', 'close', 'log_event']
SEED_METADATA_BODY = ["metadata = {'experiment_seed': self.seed}", 'metadata[self.seed_field] = self.run_seed',
                      'return metadata']
M6D6JR_BOUND = {  # the M6D6jR record's bound authority (runner excluded: it is the hotfixed file)
    'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml':
        '2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830',
    'outputs/audit/M6D6JR_E07C_SCIENCE_LAUNCH_DECISION.json':
        'bfd3320f4d1f4a335706f288f81eb79faa7bae3afeabf776e8097fc3ccb2a15e',
    'configs/amendments/e07c_m6d6j_main_production_runner_contract.yaml':
        '9528d53e536e3e6002db30d9d5300c4abb72ef1e653b1d53ba81dc3af5c7c422',
    'outputs/audit/M6D6J_E07C_MAIN_RUNNER_QUALIFICATION.json':
        'd8a3ae9c6aa6876006606b6b89fcbca02d56e60ebb35d19f8d151e2b0c3d668f',
    'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml':
        '94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6',
    'configs/methods/e07c_difffas_bin_idfree.yaml': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
    'tools/run_e07c_main.py': '7defa308b166a9aa09effb97f424c1ae625ca0fc69f73f12740a2e7d9ca75569',
    'methods/difffas/main_runner_io.py': '3bf0239e3809a3cf2f82223e02db6f5aad1a613e708afc31c018f0394379bc20',
    'methods/difffas/main_checkpoint.py': '416dcbeecf041af253f0ab598fed02784d1b47077ba86cf5d2b5c6e1ff915d31'}
FAILED_ATTEMPT = {'seed': 42, 'attempt': 1, 'authority': AUTHORITY, 'exit_status': 1, 'failure_phase': 'RUN_CONTEXT_OPEN',
                  'scientific_optimizer_steps': 0, 'scientific_checkpoint_writes': 0, 'scientific_result': False,
                  'experiment_seed_consumed_as_completed_run': False,
                  'recovery': 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX'}
NOT_QUALIFIED = ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
ZERO = {'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'TRAIN_image_reads': 0,
        'VAL_image_reads': 0, 'TEST_image_reads': 0, 'GPU_contacted': False}
RUN_CONTRACT = {'experiment_seeds': [42, 1337, 2026], 'epochs': 400, 'batch_size': 4, 'drop_last': False,
                'iterations_per_epoch': 2210, 'total_iterations': 884000, 'visualize_every': 1000,
                'checkpoint_every': 10000, 'retention': 'M6D6iR', 'optimizer_scheduler_ema_rng_ddpm_encoder_changed': False,
                'train_manifest_changed': False, 'firewall_changed': False, 'cli_fresh_only': True,
                'resume': 'UNQUALIFIED', 'changed': False}


def require(ok, message):
    if not ok:
        raise ValueError('M6D6jF: ' + message)


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


# ----------------------------------------------------------------- static derivation (reader-scoped)
def runner_scope(new_raw, old_raw):
    """AST diff of main_runner.py: only E07cMainRunContext seed-metadata methods may differ."""
    new, old = ast.parse(new_raw), ast.parse(old_raw)
    require(sha(old_raw) == RUNNER_AUTHORITY_SHA, 'authority runner SHA256')
    rest = lambda t: [ast.dump(n) for n in t.body if getattr(n, 'name', None) != CONTEXT_CLASS]  # noqa: E731
    require(rest(new) == rest(old), 'runner: nothing outside E07cMainRunContext changed')

    def members(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CONTEXT_CLASS)
        return cls, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    (ncls, nm), (ocls, om) = members(new), members(old)
    require([ast.dump(b) for b in ncls.bases] == [ast.dump(b) for b in ocls.bases] and
            ast.get_docstring(ncls) == ast.get_docstring(ocls), 'context class header unchanged')
    changed = sorted(k for k in nm.keys() | om.keys() if k not in om or k not in nm or
                     ast.dump(nm[k]) != ast.dump(om[k]))
    require(changed == CHANGED_METHODS and '_seed_metadata' not in om, 'runner: only seed-metadata methods changed')
    helper = nm['_seed_metadata']
    require(ast.get_docstring(helper) and [ast.unparse(n) for n in helper.body[1:]] == SEED_METADATA_BODY,
            'canonical seed-metadata helper')
    for name in CHANGED_METHODS:
        if name == '_seed_metadata':
            continue
        src = ast.unparse(nm[name])
        require('self._seed_metadata()' in src and 'self.seed_field' not in src and
                'experiment_seed=self.seed' not in src and "'experiment_seed': self.seed" not in src,
                'single seed-metadata construction in ' + name)
        old_src = ast.unparse(om[name])
        require('self.seed_field' in old_src, 'authority site used seed_field: ' + name)
    for name in set(nm) - set(CHANGED_METHODS):
        require('self.seed_field' not in ast.unparse(nm[name]) or name == 'seed_field', 'no other seed_field use')
    return {'changed_methods': changed, 'unchanged_outside_context': True, 'canonical_helper': '_seed_metadata',
            'authority_runner_sha256': RUNNER_AUTHORITY_SHA, 'candidate_runner_sha256': sha(new_raw)}


def check_record(reader=worktree_reader):
    raw = reader(RECORD)
    require(sha(raw) == RECORD_SHA, 'hotfix record SHA256 == pinned constant')
    r = json.loads(raw)
    require((r['milestone'], r['method_id'], r['classification'], r['record_kind'], r['authority_commit'],
             r['amendment_created'], r['a9_created'], r['historical_records_rewritten'],
             r['m6d6jr_remains_historically_correct_at_its_commit']) ==
            (MILESTONE, 'E07c', DECISION_CLASS, RECORD_KIND, AUTHORITY, False, False, False, True), 'identity')
    require(r['fidelity'] == {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
                              'new_fidelity_class': False}, 'fidelity unchanged')
    require(r['supersedes_prospectively']['path'] == RUNNER and
            r['supersedes_prospectively']['sha256_at_authority'] == RUNNER_AUTHORITY_SHA and
            r['supersedes_prospectively']['other_implementation_files_changed'] is False, 'supersession scope')
    require({k: r['failed_attempt'][k] for k in FAILED_ATTEMPT} == FAILED_ATTEMPT, 'failed attempt semantics')
    require(r['failed_attempt']['other_runtime_facts'].startswith('NOT_RECORDED'), 'no fabricated runtime facts')
    sites = {s['method']: s for s in r['defect_sites']}
    require(sorted(sites) == sorted(m for m in CHANGED_METHODS if m != '_seed_metadata') and
            [m for m, s in sites.items() if s['observed_in_launch']] == ['_manifest'] and
            sites['_open_locked']['kind'] == 'DUPLICATE_KWARG_TYPEERROR' and
            sites['close']['kind'] == 'SUMMARY_IDENTITY_REFUSAL', 'defect sites')
    require(r['fix']['used_by'] == [m for m in ('_open_locked', '_resolved_config', '_manifest', 'log_event', 'close')] and
            r['fix']['QUALIFICATION'] == {'experiment_seed': None, 'qualification_seed': 60608} and
            r['fix']['qualification_output_changed'] is False, 'fix')
    require(r['scientific_run_contract_unchanged'] == RUN_CONTRACT, 'scientific run contract unchanged')
    require(r['statuses']['not_qualified'] == NOT_QUALIFIED and r['statuses']['MAIN_CHECKPOINT_RESUME'] == 'UNQUALIFIED'
            and r['statuses']['scientific_training_completed'] is False, 'statuses')
    require(r['hotfix_operation'] == ZERO, 'no runtime activity')
    require((r['regression_test'], r['preflight']) == (TESTS, PREFLIGHT), 'bound test / preflight')
    for rel, digest in M6D6JR_BOUND.items():
        require(sha(reader(rel)) == digest, 'M6D6j/M6D6jR historical bytes ' + rel)
    scope = runner_scope(reader(RUNNER), at_authority(RUNNER))
    return r, scope


def build_evidence_from(r, scope):
    return {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'decision_classification': DECISION_CLASS,
            'record_kind': RECORD_KIND, 'status': 'PASS', 'method_id': 'E07c', 'authority_commit': AUTHORITY,
            'record_path': RECORD, 'record_sha256': RECORD_SHA, 'runner_scope': scope,
            'defect_sites': r['defect_sites'], 'failed_attempt': r['failed_attempt'], 'fix': r['fix'],
            'historical_bytes_unchanged': M6D6JR_BOUND, 'scientific_run_contract_unchanged': RUN_CONTRACT,
            'statuses': r['statuses'], 'hotfix_operation': ZERO, 'amendment_created': False, 'a9_created': False,
            'fidelity': r['fidelity']}


def check_evidence(reader=worktree_reader):
    r, scope = check_record(reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(r, scope), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in (RECORD_SHA, "multiple values for keyword argument 'experiment_seed'", '_manifest', '_open_locked',
                  'summary may not overwrite run identity', 'RUN_CONTEXT_OPEN', 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX',
                  'MAIN_CHECKPOINT_RESUME', 'UNQUALIFIED', 'DEV-021', 'tools/run_e07c_main.py --seed 42', 'NOT EXECUTED'):
        require(token in md, 'evidence report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6D6jR authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([RUNNER, *bookkeeping]), 'only runner + ledger/index may differ: ' + json.dumps(changed))
    require(not [p for p in git('ls-files', '--others', '--exclude-standard').decode().split()
                 if 'Amendment_A9' in p or '_a9_' in p], 'no A9')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (RUNNER, *bookkeeping)]),
            'worktree holds exactly the M6D6jF candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, RUNNER):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no whitespace before line endings ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    require(rows[RUNNER]['sha256'] == RUNNER_AUTHORITY_SHA, 'baseline runner row')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in (*NEW, RUNNER):
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
    'amendment_created': False, 'a9_created': False, 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021',
    'new_deviation': False, 'new_fidelity_class': False, 'failed_attempt': FAILED_ATTEMPT,
    'MAIN_CHECKPOINT_RESUME': 'UNQUALIFIED', 'scientific_training_completed': False, 'training_runs': 0,
    'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'TRAIN_access': False, 'VAL_access': False,
    'TEST_access': False, 'GPU_contacted': False, 'M8_bank': False, 'commit': False, 'push': False,
    'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'not_qualified': NOT_QUALIFIED,
    'method_status': 'IMPLEMENTED_NOT_EXECUTED', 'record_sha256': RECORD_SHA, 'modified_implementation_files': [RUNNER],
    'runner_sha256_before': RUNNER_AUTHORITY_SHA, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(row['committed_prefix_sha256'] == sha(prefix), 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, RUNNER)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == t['errors'] == 0 for t in row['tests']), 'ledger tests passed')


def ledger_row(tests):
    prefix = at_authority(LEDGER)
    return {**LEDGER_EXPECTED, 'purpose': 'fix duplicate experiment_seed seed metadata in the SCIENTIFIC E07c RunContext',
            'notes': ('Implementation bugfix (DETERMINISTIC_IMPLEMENTATION_CORRECTION), not an amendment/A9. Seed-42 '
                      'attempt 1 failed at RUN_CONTEXT_OPEN with 0 scientific optimizer steps; recovery is a fresh '
                      'same-seed rerun after this fix is committed. No scientific value changed.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest; python3 -B '
                        'tools/m6d6jf_e07c_scientific_context_preflight.py --write-evidence-json / --before-ledger / '
                        '--append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, RUNNER))},
            'committed_prefix_sha256': sha(prefix), 'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(),
            'user': getpass.getuser(), 'git_dirty': True,
            'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
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
    worktree(bookkeeping)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
            'artifact_index_rows_expected': count, 'artifact_rows_added': len(NEW),
            'modified_existing_files': [RUNNER, *bookkeeping]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--write-evidence-json', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence_from(*check_record()), indent=2, sort_keys=True,
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
