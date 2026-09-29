#!/usr/bin/env python3
"""Verify the M6H candidate: scope the two historical no-A9 assertions to their own milestone trees; ledger/index LAST.

HISTORICAL TEST-SCOPING HOTFIX ONLY. tests/test_m6d6ir_e07c_checkpoint_retention.py::test_06_no_a9 and
tests/test_m6d6jr_e07c_science_launch_decision.py::test_14_no_a9 listed `git ls-files docs configs` (the CURRENT
index/HEAD), so the legitimately later Amendment A9 (M6A9, 8e9ccc3) broke them. The only change is
CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE (`git ls-tree -r --name-only <commit that added the test>`).
No record, document, scientific value or other assertion changes.

STATIC: stdlib only; no GPU, data, model or TEST. check_evidence takes readers (no moving-HEAD lock).

  --write-evidence-json / --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'e936ec252de20ac9dd28bcc8bff0282b5b65d2da'
M6A9_COMMIT = '8e9ccc371f6c53b75ec6b473d2306aefa9aa672b'
BRANCH = 'm6-baselines'
MILESTONE = 'M6H'
CLASSIFICATION = 'M6H_HISTORICAL_TEST_SCOPE_HOTFIX'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 130
LEDGER_PREFIX_SHA = '75b64bb01bc7fad5c55267adb967a307d3dcff3b32574c00b7a52d4eef2c7158'
INDEX_BASELINE_ROWS = 773
EV_JSON = 'outputs/audit/M6H_HISTORICAL_TEST_SCOPE_HOTFIX.json'
EV_MD = 'outputs/audit/M6H_HISTORICAL_TEST_SCOPE_HOTFIX.md'
TESTS = 'tests/test_m6h_historical_test_scope.py'
PREFLIGHT = 'tools/m6h_historical_test_scope_preflight.py'
NEW = tuple(sorted((EV_JSON, EV_MD, TESTS, PREFLIGHT)))
IR_TEST = 'tests/test_m6d6ir_e07c_checkpoint_retention.py'
JR_TEST = 'tests/test_m6d6jr_e07c_science_launch_decision.py'
MODIFIED = (IR_TEST, JR_TEST)
FIXED = {IR_TEST: ('M6D6iR', 'm6d6ir', 'test_06_no_a9', '39508719'), JR_TEST: ('M6D6jR', 'm6d6jr', 'test_14_no_a9', '8358d8b6')}
OLD_ASSERT = "        self.assertFalse([p for p in git('ls-files', 'docs', 'configs').stdout.decode().split() if 'A9' in p])\n"
UNCHANGED_RECORDS = {
    'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml': '94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6',
    'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml': '2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830',
    'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml':
        '5ddd59f3560eea63471dd5bc1b4f65281e8f529a74a1e9d4d990ddbb427f3644',
    'configs/amendments/e07c_m6d6jf_scientific_context_hotfix.yaml':
        '946ca134ef514650002f368a8d87715f16a80855d10725587d9e6e855747af13',
    'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json': 'fe287ebe1ec9c19e5f63a40f1498e41f5b7044014c2cc7eb6279a32604555058',
    'configs/methods/e07c_difffas_bin_idfree.yaml': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c'}
A9_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A9_DiffFAS_Family_Resource_Constrained_Scope_Exclusion.md'


def require(ok, message):
    if not ok:
        raise ValueError('M6H: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    return (ROOT / p).read_bytes()


def worktree_reader(rel):
    return read(rel)


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


def hunks(tag, fn):
    helper = (f"def tracked_at_{fn}(*paths):\n"
              f'    """Tracked paths at the {tag} state: the tree of the commit that added this test, else the (candidate) index."""\n'
              f"    commit = {fn}_commit()\n"
              f"    args = ('ls-tree', '-r', '--name-only', commit, *paths) if commit else ('ls-files', *paths)\n"
              f"    return git(*args).stdout.decode().split()\n\n\n")
    return [("def at_authority(rel):\n", helper + "def at_authority(rel):\n"),
            (OLD_ASSERT, f"        self.assertFalse([p for p in tracked_at_{fn}('docs', 'configs') if 'A9' in p])\n")]


def expected_fixed(authority_text, tag, fn):
    text = authority_text
    for old, new in hunks(tag, fn):
        require(text.count(old) == 1, f'{tag}: authority anchor present exactly once')
        text = text.replace(old, new)
    return text


def derive(reader=worktree_reader, authority_reader=at_authority):
    out = {}
    for rel, (tag, fn, test, commit_prefix) in FIXED.items():
        cur = reader(rel).decode()
        require(cur == expected_fixed(authority_reader(rel).decode(), tag, fn),
                f'{rel} differs from the authority ONLY by the tracked_at_{fn} helper and the {test} scope line')
        require(OLD_ASSERT not in cur and f"tracked_at_{fn}('docs', 'configs')" in cur, f'{test} no longer reads HEAD')
        require("self.assertEqual((self.r['amendment_created'], self.r['a9_created']), (False, False))" in cur,
                f'{test} still asserts amendment_created = a9_created = false')
        out[rel] = {'milestone': tag, 'test': test, 'historical_commit_prefix': commit_prefix,
                    'before': "git('ls-files', 'docs', 'configs')  # CURRENT HEAD index",
                    'after': f"tracked_at_{fn}('docs', 'configs')  # git ls-tree -r --name-only <{fn}_commit()>"}
    for rel, digest in UNCHANGED_RECORDS.items():
        require(sha(reader(rel)) == digest and reader(rel) == authority_reader(rel), 'record unchanged ' + rel)
    require(reader(A9_DOC) == authority_reader(A9_DOC), 'A9 document unchanged')
    return {'fixed_assertions': out, 'unchanged_records': sorted(UNCHANGED_RECORDS), 'a9_document_unchanged': True}


def build_evidence_from(d):
    return {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'authority_commit': AUTHORITY,
        'record_kind': 'HISTORICAL_TEST_SCOPING_HOTFIX', 'is_scientific_amendment': False,
        'defect': {'pre_existed_before_m6fb': True, 'introduced_by_commit': M6A9_COMMIT,
                   'introduced_by': 'M6A9 legitimately tracked the Amendment A9 document (later milestone)',
                   'reproduced_at_authority': AUTHORITY, 'reproduction': 'clean detached worktree at e936ec2',
                   'root_cause': 'the historical no-A9 assertions listed the CURRENT HEAD index (git ls-files) instead '
                                 'of their own milestone tree',
                   'failing_tests': ['tests/test_m6d6ir_e07c_checkpoint_retention.py::TestM6D6iRDecision::test_06_no_a9',
                                     'tests/test_m6d6jr_e07c_science_launch_decision.py::TestM6D6jRDecision::test_14_no_a9']},
        'correction': 'CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE', 'static_derivation': d,
        'historical_facts_retained': ['record amendment_created = false', 'record a9_created = false',
                                      'M6D6iR / M6D6jR did not create A9', 'their own milestone trees contain no A9 path',
                                      'authority bindings unchanged'],
        'scientific_state_changed': False, 'a9_changed': False, 'e07c_state_changed': False,
        'e06b_state_changed': False, 'm6fb_candidate_mixed_in': False,
        'decision_operation': {'training_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'model_executions': 0,
                               'GPU_contacted': False, 'image_reads': 0, 'data_reads': 0, 'TEST_access': False},
        'm6': {'M6_CLOSED': False, 'M7_started': False}}


def check_evidence(reader=worktree_reader, authority_reader=at_authority):
    d = derive(reader, authority_reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(d), 'evidence == live derivation')
    md = reader(EV_MD).decode()
    for token in ('M6H', 'CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE', 'test_06_no_a9', 'test_14_no_a9',
                  '8e9ccc3', 'e936ec2', 'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in md, 'report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on e936ec2')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only the two tests + ledger/index differ: ' + json.dumps(changed))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for p in ('configs/methods/e06b_dsdg_native.yaml', 'methods/dsdg/native.py',
              'outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json'):
        require(not (ROOT / p).exists(), 'parked M6F-B file must not be present: ' + p)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M6H candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, *MODIFIED):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF, no trailing whitespace ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
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
    'record_kind': 'HISTORICAL_TEST_SCOPING_HOTFIX', 'correction': 'CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE',
    'defect_pre_existed_before_m6fb': True, 'defect_introduced_by_commit': M6A9_COMMIT,
    'reproduced_at_authority': AUTHORITY, 'scientific_state_changed': False, 'a9_changed': False,
    'e07c_state_changed': False, 'e06b_state_changed': False, 'amendment_created': False, 'M6_closed': False,
    'M7_started': False, 'TEST_access': False, 'GPU_contacted': False, 'training_runs': 0, 'optimizer_steps': 0,
    'checkpoint_writes': 0, 'image_reads': 0, 'commit': False, 'push': False, 'authority_commit': AUTHORITY,
    'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': LEDGER_PREFIX_SHA,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
    'modified_existing_files': list(MODIFIED)}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == 0 and (t['errors'] == 0 or t.get('errors_preexisting_environmental'))
                                 for t in row['tests']), 'ledger tests: no failures; errors only if pre-existing environmental')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6H: scope the historical M6D6iR/M6D6jR no-A9 assertions to their own milestone trees',
            'notes': ('test_06_no_a9 / test_14_no_a9 listed the CURRENT HEAD index and failed since M6A9 (8e9ccc3) '
                      'legitimately tracked A9; reproduced at e936ec2. Now they list git ls-tree of the commit that added '
                      'each test. Records, documents, science and all other assertions unchanged. Parked M6F-B candidate '
                      'not mixed in.'),
            'command': ('laptop only: .venv/bin/python -B -m unittest (affected modules, M6A9, E07c static regression, '
                        'M6H); python3 -B tools/m6h_historical_test_scope_preflight.py'),
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
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'modified_existing_files': [*MODIFIED, *bookkeeping]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        (ROOT / EV_JSON).write_text(json.dumps(build_evidence_from(derive()), indent=2, sort_keys=True) + '\n')
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
