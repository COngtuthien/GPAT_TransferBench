"""Verify the M7C4 candidate: GPAT production runner + real-TRAIN data path + checkpoint/resume qualification.

M7C4 adds the production runner (methods/gpat/runner*.py), the execution asset config, the frozen CLI wiring
(gpatbench/cli.py, additive), the qualification tool, its evidence and tests. No scientific run, no 60/10-epoch
training, no scientific checkpoint candidate, no bank, no VAL/TEST access. The frozen spec, A1-A10, M7B, M7C2a, M7C3,
the static core and B0-B3 are unchanged.

STATIC: stdlib only. Candidate-time checks (HEAD lock, worktree, ledger, index) run only here; check_* take readers so
the regression tests can pin them to the commit that added them.

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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'cff688dd35524bf7f01c39916ef87cd4987b6470'
BRANCH = 'm6-baselines'
MILESTONE = 'M7C4'
CLASSIFICATION = 'M7C4_GPAT_PRODUCTION_RUNNER'
RECORD_KIND = 'PRODUCTION_RUNNER_IMPLEMENTATION / REAL_TRAIN_DATA_PATH_QUALIFICATION / CHECKPOINT_RESUME_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 140
INDEX_BASELINE_ROWS = 877
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
CLI = 'gpatbench/cli.py'
MODIFIED = (CONFIG_STATUS, CLI)
RUNNER = ('methods/gpat/runner.py', 'methods/gpat/runner_io.py', 'methods/gpat/runner_data.py',
          'methods/gpat/runner_checkpoint.py', 'methods/gpat/runner_cli.py')
ASSET_CONFIG = 'configs/execution/gpat_m7_assets_3090.yaml'
EV_JSON = 'outputs/audit/M7C4_GPAT_RUNNER_QUALIFICATION.json'
EV_MD = 'outputs/audit/M7C4_GPAT_RUNNER.md'
QUAL_TOOL = 'tools/m7c4_gpat_runner_qualification.py'
PREFLIGHT = 'tools/m7c4_gpat_runner_preflight.py'
TESTS = 'tests/test_m7c4_gpat_runner.py'
RECORD = 'configs/amendments/gpat_m7c4_runner_resolution.yaml'
EV_JSON_SHA = '036ef14c7bd18af5793c3591050d316b81bfd7c4c99ffae5d814797ebd5ee011'
NEW = tuple(sorted((*RUNNER, ASSET_CONFIG, EV_JSON, EV_MD, QUAL_TOOL, PREFLIGHT, TESTS, RECORD)))
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
STATIC_CORE = tuple('methods/gpat/' + n for n in (
    '__init__.py', 'artifact_encoder.py', 'batching.py', 'composition.py', 'config.py', 'discriminator.py', 'ema.py',
    'generator.py', 'grl.py', 'heads.py', 'highpass.py', 'identity_labels.py', 'losses.py', 'model.py',
    'naf_source.py', 'runtime_contract.py', 'schedule.py', 'spectral.py', 'teacher_preprocess.py', 'wavelet.py'))
PROTECTED = ('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
             'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
             'configs/amendments/gpat_m7b_owner_clarifications.yaml',
             'configs/amendments/gpat_m7c2a_implementation_resolution.yaml',
             'configs/amendments/gpat_m7c3_gpu_runtime_resolution.yaml',
             *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA), *STATIC_CORE,
             'environments/gpat_m7_cpu.lock.json', 'environments/gpat_m7_gpu.lock.json',
             'outputs/audit/method_status.csv', 'outputs/audit/deviation_report.md', 'outputs/audit/STAGE_STATE.json',
             'manifests/pairs_train_v1.parquet', 'manifests/split_v1.parquet', 'models/registry.yaml',
             'gpatbench/preprocess/aux_models.py', 'configs/execution/m5_gpu_3090.yaml', 'configs/run_logging_v1.yaml',
             'methods/difffas/aux_runner_io.py', 'methods/common/runlog.py', 'methods/common/learned.py')
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'configs/methods/', 'configs/amendments/', 'frozen_config_snapshot/',
                      'manifests/', 'models/', 'runs/', 'data/', 'cache/', 'environments/', 'third_party/')
STATUS_PHRASES = ('AMP_OVERFLOW_FAIL_CLOSED', 'PRODUCTION_RUNNER_IMPLEMENTED', 'REAL_TRAIN_DATA_PATH_QUALIFIED', 'CHECKPOINT_RESUME_QUALIFIED',
                  'WARMUP_RUNNER_QUALIFIED', 'SCIENTIFIC_TRAINING_READY_NOT_STARTED', 'GPU_RUNTIME_QUALIFIED',
                  'R04_LEVEL2_TEACHER_PARITY_QUALIFIED', 'M6_CLOSED = true')
REQUIRED_GATES = ('orders', 'b0_regular', 'b3_regular', 'b3_tail', 'dev022', 'warmup_regular', 'warmup_tail',
                  'warmup_handoff', 'resume', 'warmup_resume', 'candidate_writer', 'firewall', 'all_scenarios_exit_0',
                  'no_oom_below_24GiB')
WEIGHT_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors', '.pkl')


def require(ok, message):
    if not ok:
        raise ValueError('M7C4: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in WEIGHT_SUFFIXES + ('.png', '.jpg'), 'no weight/image ' + path)
    return (ROOT / p).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- reader-scoped checks
def check_protected(reader, base):
    for rel in PROTECTED:
        require(reader(rel) == base(rel), 'protected unchanged ' + rel)
    for rel, digest in CONFIG_SHA.items():
        require(sha(reader(rel)) == digest, 'B0-B3 ' + rel)


def check_cli(reader, base):
    old, new = base(CLI).decode().splitlines(), reader(CLI).decode().splitlines()
    import difflib
    removed = [l for l in difflib.ndiff(old, new) if l.startswith('- ')]
    require(all('`train-probe` (M5)' in l for l in removed) and len(removed) <= 1,
            'gpatbench/cli.py: only the module docstring line may change; everything else is additive')
    added = '\n'.join(l[2:] for l in difflib.ndiff(old, new) if l.startswith('+ '))
    require('runner_cli.add_parser(sub)' in added and 'from methods.gpat import runner_cli' in added, 'CLI wiring')


def check_evidence(reader):
    ev = json.loads(reader(EV_JSON))
    require(ev['milestone'] == MILESTONE and ev['status'] == 'PASS' and ev['authority_commit'] == AUTHORITY,
            'evidence status')
    require(set(REQUIRED_GATES) <= set(ev['gates']) and all(ev['gates'][k] for k in REQUIRED_GATES), 'evidence gates')
    require(ev['qualification_seed'] == 70404 and 'QUALIFICATION_ONLY' in ev['labels'], 'qualification only')
    f = ev['firewall']
    require(all(f[k] == 0 for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata', 'test_metadata')),
            'firewall counters')
    require(sha('\n'.join(sorted(ev['opened_train_ids'])).encode()) == f['unique_ids_sha256'] and
            len(ev['opened_train_ids']) == f['unique_ids'], 'opened TRAIN id list')
    require(all(ev['resume_equivalence'].values()) and all(ev['warmup_resume_equivalence'].values()), 'resume')
    sc = ev['scenarios']
    require(sc['b3_tail']['group_record']['microbatch_sizes'] == [4, 2] and
            sc['b3_tail']['group_record']['sample_weights'] == [4 / 6, 2 / 6], 'tail [4, 2]')
    require(sc['warmup_regular']['record']['batch_size'] == 64 and sc['warmup_tail']['record']['batch_size'] == 6,
            'warmup 64 / 6')
    require(sc['candidate_writer']['metadata']['selected'] is False and
            sc['candidate_writer']['bytes_deleted_after_hash'] is True, 'candidate writer')
    for name, r in sc.items():
        if isinstance(r, dict) and 'run_dir' in r:
            require('/qualification/m7/M7C4/' in r['run_dir'] and '/runs/m7/' not in r['run_dir'],
                    'qualification roots only: ' + name)
    require(ev['peak_reserved_bytes_max'] < 24 * 2 ** 30, 'no OOM')
    return ev


ORDER_SHA = {'generator|SCIENTIFIC|42|1': '2b8961cea6327719bc7ae7768d1fd6cfa8caf4d0bb25b23ec8a4f6cbc3396fda',
             'generator|SCIENTIFIC|1337|1': '4591f8e1c18ab3e4a41cbcdb6a2506b639f31060858d08d87785322a28d1b444',
             'generator|SCIENTIFIC|2026|1': '11d4b830b4af9badb0f0e4d7758b1d678f33a143666e2e5dc6f8f62cbbc3aad7',
             'generator|QUALIFICATION|70404|1': '118216183a328a9373cd2b66446d9dc567c56ce7428228f45b58f0df7a2304d9'}


def check_record(reader):
    rec = json.loads(reader(RECORD))
    require(rec['record_kind'] == 'ADDITIVE_RUNNER_IMPLEMENTATION_RESOLUTION' and rec['authority_commit'] == AUTHORITY
            and rec['new_deviation'] is None and 'NO_NEW_DEVIATION' in rec['status'], 'record kind')
    for rel, digest in rec['bound_authority_sha256'].items():
        require(sha(reader(rel)) == digest, 'record binding ' + rel)
    r = rec['resolutions']
    eo = r['EPOCH_ORDER']
    require(eo['class'] == 'OWNER_IMPLEMENTATION_CLARIFICATION' and eo['payload'] == 'GPAT-M7|{stage}|{MODE}|{seed}|{epoch}'
            and eo['MODE'] == ['SCIENTIFIC', 'QUALIFICATION'], 'epoch-order preimage')
    for key, value in ORDER_SHA.items():
        require(eo['order_sha256'][key] == value, 'frozen order hash ' + key)
    require(r['AMP_OVERFLOW']['policy'] == 'FAIL_CLOSED_AMP_OVERFLOW', 'AMP policy')
    require(r['ASSET_BINDING']['classification'] == 'EXECUTION_ONLY_HOST_BINDING', 'asset binding')
    require(r['QUALIFICATION_EVIDENCE']['sha256'] == sha(reader(EV_JSON)) == EV_JSON_SHA, 'qualification evidence bound')
    require(all(r['POST_EDIT_FINITE_SMOKE']['result'].values()), 'post-edit finite smoke bitwise equal')
    require(r['SEEDS'] == {'scientific': [42, 1337, 2026], 'qualification': 70404,
                           'qualification_is_never_scientific': True}, 'seeds')
    require('AMP_OVERFLOW_FAIL_CLOSED' in rec['m7_status_after'], 'status')
    return rec


def check_no_val_path(reader):
    for rel in (*RUNNER, QUAL_TOOL):
        src = reader(rel).decode()
        for forbidden in ('val_pairs_v1', 'artifact_probe_v1', 'ArtifactProbe(', 'select_checkpoint', 'generate_bank',
                          'test_pairs'):
            require(forbidden not in src, f'{rel}: {forbidden}')


def check_config_status(reader, base):
    old, now = base(CONFIG_STATUS), reader(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in STATUS_PHRASES:
        require(phrase in added, 'CONFIG_STATUS M7C4 ' + phrase)
    for bad in ('SCIENTIFIC_TRAINING_COMPLETE', 'M7 COMPLETE'):
        require(bad not in added, 'CONFIG_STATUS must not claim ' + bad)
    return len(now) - len(old)


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7C3 authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS, cli.py + ledger/index differ: '
            + json.dumps(changed))
    require(git('ls-files', *('*' + s for s in WEIGHT_SUFFIXES)).decode().strip() == '', 'no weights tracked')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in (*changed, *NEW):
        require(rel == RECORD or not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7C4 candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, CLI):
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF, no trailing ws ' + rel)
    added = read(CONFIG_STATUS)[len(at_authority(CONFIG_STATUS)):]
    require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF ' + CONFIG_STATUS)


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


def ledger_expected(prefix_sha):
    return {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
        'record_kind': RECORD_KIND, 'exit_code': 0, 'M6_closed': True, 'M7_production_runner_implemented': True,
        'M7_real_train_data_path_qualified': True, 'M7_checkpoint_resume_qualified': True,
        'M7_warmup_runner_qualified': True, 'M7_gpu_runtime_qualified': True,
        'M7_r04_level2_teacher_parity_qualified': True, 'M7_scientific_training_ready': True,
        'M7_scientific_training_started': False, 'M7_amp_overflow_fail_closed': True, 'scientific_runs': 0, 'scientific_checkpoint_candidates': 0,
        'TRAIN_access': 'QUALIFICATION_ROWS_ONLY', 'VAL_access': False, 'TEST_access': False, 'GPU_contacted': True,
        'M8_bank': False, 'downstream_training': False, 'new_deviation_numbers': [], 'amendment_created': False,
        'method_status_modified': False, 'deviation_report_modified': False, 'stage_state_modified': False,
        'config_status_modified': True, 'commit': False, 'push': False, 'authority_commit': AUTHORITY,
        'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': prefix_sha,
        'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
        'modified_existing_files': list(MODIFIED), 'output_artifacts': sorted((*NEW, *MODIFIED))}


def check_ledger_row(row, prefix):
    for k, v in ledger_expected(sha(prefix)).items():
        require(row[k] == v, 'ledger field ' + k)
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'], 'ledger tests recorded')
    for t in row['tests']:
        require(t['failures'] == 0 or t.get('failures_preexisting'), 'failures classified: ' + t['scope'])
        require(t['errors'] == 0 or t.get('errors_preexisting_environmental'), 'errors classified: ' + t['scope'])


def ledger_row(tests, prefix_sha, notes):
    return {**ledger_expected(prefix_sha),
            'purpose': ('M7C4: GPAT production runner (B1/B3 attack warmup + B0-B3 generator stage, recovery/resume, '
                        'EMA candidate writer, run logging, frozen train-generator CLI) and its qualification on '
                        'selected real TRAIN rows with seed 70404 in isolated qualification roots; no scientific run'),
            'notes': notes,
            'command': ('GPU host (gpat-m7-gpu): python -B tools/m7c4_gpat_runner_qualification.py --all --workdir ...; '
                        'laptop: --assemble; python3 -B tools/m7c4_gpat_runner_preflight.py --before-ledger / '
                        '--append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_protected(read, at_authority)
    check_cli(read, at_authority)
    check_evidence(read)
    check_no_val_path(read)
    check_record(read)
    status_bytes = check_config_status(read, at_authority)
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
    require(not {'torch', 'numpy', 'PIL', 'cv2'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'config_status_bytes_added': status_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests')
    parser.add_argument('--notes', default='')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.append_ledger:
        verify('before_ledger')
        prefix = at_authority(LEDGER)
        row = ledger_row(json.loads(args.tests), sha(prefix), args.notes)
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
