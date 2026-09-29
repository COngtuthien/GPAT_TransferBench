#!/usr/bin/env python3
"""Verify the M6FD candidate: E06b DSDG-NATIVE production runner, PRODUCTION_PATH_QUALIFICATION_ONLY on real TRAIN data.

The summary evidence is DERIVED from the four GPU-produced JSONs (copied back byte-exact); no measured value is written
on the laptop. The runner / harness / CLI / capture bytes are pinned to the bytes that ran on the GPU host.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow). check_evidence takes readers (no moving HEAD).

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
AUTHORITY = '4b54280c53f87c391151a5436b18e5fc51db88fe'
BRANCH = 'm6-baselines'
MILESTONE = 'M6FD'
CLASSIFICATION = 'M6FD_E06B_PRODUCTION_QUALIFICATION'
LABEL = 'PRODUCTION_PATH_QUALIFICATION_ONLY'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 133
LEDGER_PREFIX_SHA = '69f1f00fa102f44d01b24aa55b938eeb78cd9f28ff551ea2b8de2cf3c8e206e8'
INDEX_BASELINE_ROWS = 793
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
CODE_SHA = {  # the bytes that ran on the GPU host (staged uncommitted, removed afterwards)
    'methods/dsdg/native_runner.py': 'ecbef580f71c02eca26ccae2a47ae6ca201eaccb332995db0f47003093dc28cd',
    'methods/dsdg/native_runner_qualification.py': 'e88bd32fbc8a558596f80286d31a742345a577c2dbd5ecb0da91dfd5fb9eceb6',
    'tools/run_e06b.py': '7c797c47f45641f2919780463e67d3ae15933e736c74cdd331802b568e62c4ef',
    'tools/m6fd_e06b_cli_preflight_capture.py': 'c0ac00c0511795b7c0cdf1ad8a720986a8bafd4a5741396c63055113b4012199'}
GPU = {'loader': 'outputs/audit/M6FD_E06B_LOADER.json', 'b240': 'outputs/audit/M6FD_E06B_B240.json',
       'tail120': 'outputs/audit/M6FD_E06B_TAIL120.json', 'cli': 'outputs/audit/M6FD_E06B_CLI_PREFLIGHT.json'}
GPU_SHA = {'loader': 'dcc2fdd847ede8ac878e98aeddd8abe04b0d4a81aa1edce424a79d1e33844a8f',
           'b240': '04271cca88055223580bab04b60b57200e993d90e860e88219cd7b5fcee7b64c',
           'tail120': '5f78d8796f34f4c18702b53a4b397b292631bfa053113e78c47884b39c26fb26',
           'cli': '638c637fd84a26d78ebe82353c8c3c6dacdb7082d424dbe3d93dc9fd74e6bc76'}
EV_JSON = 'outputs/audit/M6FD_E06B_PRODUCTION_QUALIFICATION.json'
EV_MD = 'outputs/audit/M6FD_E06B_PRODUCTION_QUALIFICATION.md'
TESTS = 'tests/test_m6fd_e06b_production_qualification.py'
PREFLIGHT = 'tools/m6fd_e06b_production_qualification_preflight.py'
NEW = tuple(sorted((*CODE_SHA, *GPU.values(), EV_JSON, EV_MD, TESTS, PREFLIGHT)))
CONFIG = 'configs/methods/e06b_dsdg_native.yaml'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
SPLIT_SHA = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
SEED = 60801
RUNTIME_FIDELITY = 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY'
POPULATION = {'spoof': 3720, 'live': 1240, 'subjects': 60, 'print': 2080, 'replay': 1640}
STATUSES = ['CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_QUALIFIED', 'EXECUTION_MAPPING_QUALIFIED',
            'WORKER_DETERMINISM_QUALIFIED', 'PRODUCTION_DATA_PATH_QUALIFIED', 'PRODUCTION_BATCH_PATH_QUALIFIED',
            'SCIENTIFIC_CLI_PREFLIGHT_QUALIFIED', 'SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED',
            'CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED', 'RESUME_UNQUALIFIED_FRESH_ONLY',
            'SCIENTIFIC_TRAINING_NOT_EXECUTED']


def require(ok, message):
    if not ok:
        raise ValueError('M6FD: ' + message)


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


def common(name, r):
    require(r['label'] == LABEL and r['milestone'] == 'M6F-D', name + ' label / milestone')
    if name == 'cli':
        return       # pass criteria (preflight_pass, all_refused, no run root, no torch) are checked in derive()
    require(r['status'] == 'PASS', name + ' PASS')
    require((r['qualification_seed'], r['method_id'], r['scientific']) == (SEED, 'E06b', 'NON_SCIENTIFIC'), name + ' id')
    ids = r['identities']
    require((ids['config_sha256'], ids['environment_lock_sha256'], ids['lightcnn_sha256'], ids['split_manifest_sha256'],
             ids['runtime_fidelity_assessment']) == (CONFIG_SHA, LOCK_SHA, LIGHTCNN_SHA, SPLIT_SHA, RUNTIME_FIDELITY),
            name + ' bound identities')
    require(r['population']['total'] == POPULATION and r['siw_rows_present'] is False, name + ' native population')
    require(all(r['gates'].values()) and all(r['firewall_gates'].values()), name + ' gates')
    a = r['access_audit']
    require(a['denied_events'] == a['VAL_accesses'] == a['TEST_accesses'] == a['non_train_face_accesses'] ==
            a['other_manifest_accesses'] == a['scientific_run_root_accesses'] == 0 and
            a['train_faces_all_in_native_relation'], name + ' access audit')
    require((r['scientific_training_completed'], r['scientific_seeds_completed'], r['scientific_checkpoints_created'],
             r['scientific_bank_created'], r['native_manifest_created'], r['scientific_run_root_exists']) ==
            (False, 0, 0, False, False, False), name + ' no science artifacts')
    require(r['counters']['checkpoint_saves'] == r['counters']['autograd_grad_calls'] == 0, name + ' no save')


def derive(reader=worktree_reader):
    runs = {}
    for name, rel in GPU.items():
        raw = reader(rel)
        require(sha(raw) == GPU_SHA[name], 'GPU evidence bytes ' + rel)
        runs[name] = json.loads(raw)
        common(name, runs[name])
    for rel, digest in CODE_SHA.items():
        require(sha(reader(rel)) == digest, 'code bytes == the bytes that ran: ' + rel)
    require(sha(reader(CONFIG)) == CONFIG_SHA, 'frozen config unchanged')
    ld, cli = runs['loader'], runs['cli']
    require(cli['all_refused'] and cli['preflight_pass'] and not cli['run_dir_exists'] and not cli['torch_imported']
            and cli['resume'] == 'NOT_QUALIFIED_FRESH_ONLY' and len(cli['refusals']) == 8, 'CLI preconditions/refusals')
    cases = {}
    for name in ('b240', 'tail120'):
        r = runs[name]
        B = r['logical_batch_size']
        require((B, r['chunks'], r['chunk_sizes']) == ({'b240': 240, 'tail120': 120}[name], B // 20, [20] * (B // 20)),
                name + ' plan')
        cases[name] = {
            'logical_batch_size': B, 'physical_microbatch': r['physical_microbatch'], 'chunks': r['chunks'],
            'epoch_branch': r['epoch_branch'], 'labels_count': r['labels_count'],
            'subjects': r['subset']['summary']['total']['subjects'], 'group_slice': r['subset']['group_slice'],
            'independent_case': r['subset']['independent_case'], 'input_range': r['input_range'],
            'losses': r['losses'], 'total_loss': r['total_loss'], 'owned_gradients': r['owned_gradients'],
            'netCls_grad_nonzero': r['netCls_grad_nonzero'], 'optimizer_applications': r['optimizer_applications'],
            'backward_calls': r['backward_calls'], 'peak_allocated_bytes': r['peak_allocated_bytes'],
            'peak_reserved_bytes': r['peak_reserved_bytes'], 'step_seconds': r['step_seconds'],
            'batch_load_seconds': r['batch_load_seconds'], 'workers_configured': r['workers_configured'],
            'pairs_first_5': r['pairs'][:5], 'pairs_count': len(r['pairs']),
            'pre_execution_row_assertions': len(r['pre_execution_row_assertions']),
            'train_face_opens': r['access_audit']['counts_by_category']['TRAIN_FACE'],
            'train_faces_distinct': r['access_audit']['train_faces_opened_distinct'],
            'processes_logged': r['access_audit']['processes_logged'], 'gates': r['gates'],
            'lightcnn_load': r['lightcnn_load'], 'environment_torch': r['environment']['torch']}
    env = runs['b240']['environment']
    return {'environment': {'gpu': env['gpu_driver'], 'python': env['python'], 'torch': env['torch'],
                            'cuda': env['cuda'], 'cudnn': env['cudnn'], 'executable': env['executable'],
                            'environment_lock_sha256': LOCK_SHA},
            'source': runs['b240']['source'], 'lightcnn': runs['b240']['lightcnn'],
            'population': ld['population'],
            'loader': {'gates': ld['gates'], 'batch_sizes': ld['loader']['batch_sizes'],
                       'distinct_worker_ids': ld['loader']['distinct_worker_ids'], 'seconds': ld['loader']['seconds'],
                       'pre_execution_row_assertions': ld['pre_execution_row_assertions'],
                       'image_pairs': [{k: p[k] for k in ('spoof_id', 'live_id', 'dataset', 'subject_id_global',
                                                          'attack_macro', 'class_index', 'same_subject')} |
                                       {'tensors': [{k: t[k] for k in ('role', 'shape', 'dtype', 'min', 'max')}
                                                    for t in p['tensors']]} for p in ld['image_check']['pairs']],
                       'production_dataset_item': ld['image_check']['production_dataset_item'],
                       'access': {k: ld['access_audit'][k] for k in ('counts_by_category', 'train_faces_opened_distinct',
                                                                    'processes_logged', 'denied_events')}},
            'cases': cases, 'cli': {k: cli[k] for k in ('run_dir', 'run_dir_exists', 'torch_imported', 'resume',
                                                        'refusals', 'all_refused', 'dirty_injected_false_reason')},
            'started_utc': {n: r['started_utc'] for n, r in runs.items() if 'started_utc' in r}}


def build_evidence_from(d):
    return {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'method_id': 'E06b',
        'label': LABEL, 'authority_commit': AUTHORITY, 'gpu_authority_commit': AUTHORITY, 'qualification_seed': SEED,
        'config_sha256': CONFIG_SHA, 'config_unchanged': True, 'environment_of_record': 'gpat-m6-e06c',
        'code_sha256': CODE_SHA, 'gpu_evidence_sha256': {GPU[k]: v for k, v in GPU_SHA.items()}, 'derived': d,
        'statuses': STATUSES,
        'fidelity': {
            'target_source_fidelity': 'FAITHFUL_OFFICIAL', 'runtime_fidelity_assessment': RUNTIME_FIDELITY,
            'new_scientific_deviation_found': False,
            'final_method_fidelity': RUNTIME_FIDELITY,
            'final_method_fidelity_decision': 'OWNER_DECISION_AT_M6F_D_REVIEW',
            'controlled_adaptation': False,
            'frozen_config_field': 'final_execution_fidelity stays PENDING_M6F_C_RUNTIME_QUALIFICATION in the frozen '
                                   'YAML (historical; never rewritten); the final current fidelity lives here and in '
                                   'the later method_status.csv',
            'owner_rationale': ['real CASIA/MSU TRAIN production data path passed',
                                'native same-subject pairing passed on real data',
                                'K = 2 and lambda_pair = 5 unchanged', 'all official losses unchanged and active',
                                'qualified effective batch remains 240',
                                'physical microbatch 20 is only runtime compatibility',
                                'logical 240 and tail 120 paths passed on real images',
                                'worker = 8 native pairing determinism passed',
                                'no scientific configuration, loss, architecture or optimization semantics changed',
                                'no new scientific deviation introduced'],
            'lifecycle_disclosure_not_fidelity': 'the inherited E06c checkpoint/run-log writers were not exercised at '
                                                 'runtime for E06b and no scientific run lifecycle was executed; this '
                                                 'is an implementation/lifecycle disclosure, not a fidelity downgrade',
            'basis': ['real CASIA+MSU TRAIN native relation, K = 2, lambda_pair = 5 and every frozen value used unchanged',
                      'the production batch function is the M6F-C-qualified run_global_batch_v2 (240 = 12 x 20, '
                      '120 = 6 x 20) with finite losses/gradients and one Adam step',
                      'real worker draws equal the pure-Python model on the full real relation'],
            'disclosures_not_scientific_deviations': [
                'input = frozen canonical faces_256 (preprocess_v1) instead of the official scene bbox crop: the '
                'benchmark dataset adapter shared by every method (spec 8: dataset adapters may change)',
                'scientific runs are FRESH ONLY: resume is not qualified for E06b (a failed seed reruns the same seed '
                'from the start, spec 27); the engineering resume sidecar of E06c is not written',
                'the run-log (RunContext) and official checkpoint writers are reused E06c-qualified functions but are '
                'not runtime-exercised by M6F-D for E06b (no checkpoint may be written in qualification)',
                'split_v1 is opened once and read with a TRAIN + CASIA/MSU pyarrow filter on allowlisted columns; no '
                'TEST/VAL row is materialized']},
        'scientific_training_completed': False, 'scientific_seeds_completed': 0, 'scientific_checkpoints_created': 0,
        'scientific_bank_created': False, 'VAL_access': False, 'TEST_access': False, 'siw_access': False,
        'native_manifest_created': False, 'm6': {'M6_CLOSED': False, 'M7_started': False}}


def check_evidence(reader=worktree_reader):
    d = derive(reader)
    ev = json.loads(reader(EV_JSON))
    require(ev == build_evidence_from(d), 'summary evidence == derivation from the GPU JSONs')
    md = reader(EV_MD).decode()
    for token in ('M6FD', LABEL, CONFIG_SHA, RUNTIME_FIDELITY, 'new_scientific_deviation_found = false',
                  'final_method_fidelity = `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`', *STATUSES,
                  'scientific_training_completed = false', 'FRESH ONLY', 'M6_CLOSED = false', 'M7 HAS NOT STARTED'):
        require(token in md, 'report token ' + token)
    return ev


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6F-C authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([CONFIG_STATUS, *bookkeeping]), 'only CONFIG_STATUS + ledger/index differ: ' + json.dumps(changed))
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    require(read(CONFIG) == at_authority(CONFIG), 'frozen E06b config byte-identical')
    base, now = at_authority(CONFIG_STATUS), read(CONFIG_STATUS)
    require(now.startswith(base) and len(now) > len(base), 'CONFIG_STATUS append-only')
    added = now[len(base):].decode()
    for token in ('M6FD', 'E06b', 'PRODUCTION_DATA_PATH_QUALIFIED', 'PRODUCTION_BATCH_PATH_QUALIFIED',
                  'final_method_fidelity = FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'M6_CLOSED = false',
                  'M7 HAS NOT STARTED'):
        require(token in added, 'CONFIG_STATUS M6FD note token ' + token)
    for p in ('manifests/dsdg_identity_pairs_v1.parquet', 'outputs/audit/method_status.csv'):
        require(not (ROOT / p).exists(), 'must not exist ' + p)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (CONFIG_STATUS, *bookkeeping)]),
            'worktree holds exactly the M6FD candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, CONFIG_STATUS):
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
    'record_kind': LABEL, 'method_id': 'E06b', 'qualification_seed': SEED, 'config_sha256': CONFIG_SHA,
    'environment_lock_sha256': LOCK_SHA, 'code_sha256': CODE_SHA, 'gpu_cases_pass': ['loader', 'b240', 'tail120', 'cli'],
    'real_train_datasets': ['casia_fasd', 'msu_mfsd'], 'statuses': STATUSES,
    'runtime_fidelity_assessment': RUNTIME_FIDELITY, 'new_scientific_deviation_found': False,
    'final_method_fidelity': RUNTIME_FIDELITY, 'controlled_adaptation': False, 'scientific_training_completed': False,
    'scientific_seeds_completed': 0, 'scientific_checkpoints_created': 0, 'scientific_bank_created': False,
    'VAL_access': False, 'TEST_access': False, 'siw_access': False, 'GPU_contacted': True, 'M6_closed': False,
    'M7_started': False, 'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
    'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': LEDGER_PREFIX_SHA,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
    'modified_existing_files': [CONFIG_STATUS]}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, CONFIG_STATUS)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'] and all(t['failures'] == 0 and (t['errors'] == 0 or t.get('errors_preexisting_environmental'))
                                 for t in row['tests']), 'ledger tests: no failures; errors only if pre-existing environmental')


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M6F-D: E06b DSDG-NATIVE production runner, PRODUCTION_PATH_QUALIFICATION_ONLY on real TRAIN',
            'notes': ('Real CASIA+MSU TRAIN (split_v1 TRAIN filter): loader (8 workers, full real relation, draws == model; '
                      '16 real pairs decoded), independent real b240 (12x20) and tail120 (6x20) through the production '
                      'batch function, one Adam step each, weights discarded; CLI preconditions + 8 refusals. No VAL/TEST/SiW, '
                      'no checkpoint, no bank, no scientific run root. No new scientific deviation. Owner decision: final '
                      'method fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY (not CONTROLLED_ADAPTATION). '
                      'Scientific run lifecycle not executed; checkpoint writer not E06b-runtime-exercised; resume '
                      'unqualified (fresh only). M6 open; M7 not started.'),
            'command': ('GPU (gpat-m6-e06c): methods/dsdg/native_runner_qualification.py --case loader|b240|tail120; '
                        'tools/m6fd_e06b_cli_preflight_capture.py; laptop: tools/m6fd_e06b_production_qualification_preflight.py'),
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
