"""Verify the M7D1-N1 candidate: GPAT AMP overflow diagnosis + ATOMIC_AMP_BACKOFF_RETRY runtime resolution.

M7D1-N1 modifies methods/gpat/runner.py additively (AMP policy constants, PreGroupState, the retry loop, Trainer wiring,
in-flight checkpoint guard), adds the additive record configs/amendments/gpat_m7d1_amp_retry_resolution.yaml, the
qualification tool, its evidence and tests. The failed scientific attempt 2 (GPAT-B0/E08/seed 42) is preserved
read-only; no scientific run is started or resumed; no VAL/TEST access. The frozen spec, A1-A10, M7B, M7C2a, M7C3, M7C4,
the static core, the other runner modules and B0-B3 are unchanged.

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
AUTHORITY = '6cf271ebd0d658dcc5384da00265326e7bac6476'
BRANCH = 'm6-baselines'
MILESTONE = 'M7D1-N1'
CLASSIFICATION = 'M7D1_GPAT_AMP_RETRY_RESOLUTION'
RECORD_KIND = 'ADDITIVE_RUNTIME_NUMERICAL_STABILITY_RESOLUTION / AMP_OVERFLOW_DIAGNOSIS / ATOMIC_RETRY_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 141
INDEX_BASELINE_ROWS = 889
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
RUNNER_PY = 'methods/gpat/runner.py'
MODIFIED = (CONFIG_STATUS, RUNNER_PY)
RUNNER_UNCHANGED = ('methods/gpat/runner_io.py', 'methods/gpat/runner_data.py', 'methods/gpat/runner_checkpoint.py',
                    'methods/gpat/runner_cli.py', 'configs/execution/gpat_m7_assets_3090.yaml', 'gpatbench/cli.py',
                    'configs/amendments/gpat_m7c4_runner_resolution.yaml', 'tools/m7c4_gpat_runner_qualification.py',
                    'tools/m7c4_gpat_runner_preflight.py', 'tests/test_m7c4_gpat_runner.py',
                    'outputs/audit/M7C4_GPAT_RUNNER_QUALIFICATION.json', 'outputs/audit/M7C4_GPAT_RUNNER.md')
EV_JSON = 'outputs/audit/M7D1_GPAT_AMP_RETRY_QUALIFICATION.json'
EV_MD = 'outputs/audit/M7D1_GPAT_AMP_RETRY.md'
QUAL_TOOL = 'tools/m7d1_gpat_amp_retry_qualification.py'
PREFLIGHT = 'tools/m7d1_gpat_amp_retry_preflight.py'
TESTS = 'tests/test_m7d1_gpat_amp_retry.py'
RECORD = 'configs/amendments/gpat_m7d1_amp_retry_resolution.yaml'
NEW = tuple(sorted((EV_JSON, EV_MD, QUAL_TOOL, PREFLIGHT, TESTS, RECORD)))
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
             'configs/amendments/gpat_m7c3_gpu_runtime_resolution.yaml', *RUNNER_UNCHANGED,
             *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA), *STATIC_CORE,
             'environments/gpat_m7_cpu.lock.json', 'environments/gpat_m7_gpu.lock.json',
             'outputs/audit/method_status.csv', 'outputs/audit/deviation_report.md', 'outputs/audit/STAGE_STATE.json',
             'manifests/pairs_train_v1.parquet', 'manifests/split_v1.parquet', 'models/registry.yaml',
             'gpatbench/preprocess/aux_models.py', 'configs/execution/m5_gpu_3090.yaml', 'configs/run_logging_v1.yaml',
             'methods/difffas/aux_runner_io.py', 'methods/common/runlog.py', 'methods/common/learned.py')
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'configs/methods/', 'configs/amendments/', 'frozen_config_snapshot/',
                      'manifests/', 'models/', 'runs/', 'data/', 'cache/', 'environments/', 'third_party/')
STATUS_PHRASES = ('SCIENTIFIC_TRAINING_BLOCKED_NUMERICAL_POLICY_RESOLVED_READY_TO_RESTART', 'ATOMIC_AMP_BACKOFF_RETRY',
                  'FAIL_CLOSED_AMP_OVERFLOW', 'LOSS_SCALE_OVERFLOW_CONFIRMED', 'M6_CLOSED = true')
REQUIRED_GATES = ('A_failed_run_preserved', 'B_overflow_reproduced_exactly', 'C_snapshot_captured',
                  'D_probe_classification', 'D_probe_snapshot_reproduces_failure', 'D_probe_base_unaltered',
                  'F_restore_equivalence', 'F_failed_attempt_leaves_params_and_optimizer_state', 'G_exact_group_retry',
                  'G_post_failure_updates', 'finite_path_unchanged_under_new_code', 'firewall', 'all_scenarios_exit_0')
WEIGHT_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors', '.pkl')


def require(ok, message):
    if not ok:
        raise ValueError('M7D1: ' + message)


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


def check_runner_additive(reader, base):
    """runner.py: the M7C4 boundary scan, losses, schedules and GradScaler construction are kept; only the AMP policy
    machinery is added (the M7C4 step body is renamed `attempt`)."""
    old, new = base(RUNNER_PY).decode(), reader(RUNNER_PY).decode()
    for keep in ("raise AmpOverflowStop(f'non-finite gradient after unscale at update {u}'",
                 'd_norm = float(torch.nn.utils.clip_grad_norm_(self.d_params, CLIP))',
                 "torch.amp.GradScaler('cuda')", 'CLIP = 1.0', 'BETAS = (0.5, 0.999)',
                 "raise NumericalPostStepStop(f'non-finite parameter after update {u}'"):
        require(keep in old and keep in new, 'runner keeps ' + keep)
    require(new.count("torch.amp.GradScaler('cuda')") == old.count("torch.amp.GradScaler('cuda')") == 4, 'scalers')
    for add in ("AMP_ATOMIC_RETRY = 'ATOMIC_AMP_BACKOFF_RETRY'", 'class PreGroupState', 'def run_atomic_retry',
                "failure_type = 'FAIL_CLOSED_AMP_OVERFLOW_FINAL'", 'PRODUCTION_AMP_POLICY = AMP_ATOMIC_RETRY'):
        require(add in new and add not in old, 'runner adds ' + add)


def check_evidence(reader):
    ev = json.loads(reader(EV_JSON))
    require(ev['milestone'] == MILESTONE and ev['status'] == 'PASS' and ev['authority_commit'] == AUTHORITY,
            'evidence status')
    require(set(REQUIRED_GATES) <= set(ev['gates']) and all(ev['gates'][k] for k in REQUIRED_GATES), 'evidence gates')
    require(ev['classification'] == 'LOSS_SCALE_OVERFLOW_CONFIRMED', 'diagnosis classification')
    require('QUALIFICATION_ONLY' in ev['labels'], 'qualification only')
    f = ev['firewall_qualification']
    require(all(f[k] == 0 for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata', 'test_metadata')),
            'firewall counters')
    for name, r in ev['scenarios'].items():
        if name.startswith('preserve'):
            require(r['run_dir'].endswith('/runs/m7/E08/seed_42'), 'preserved scientific root')
        else:
            require('/qualification/m7/M7D1_N1/' in r['run_dir'] and '/runs/m7/' not in r['run_dir'],
                    'qualification roots only: ' + name)
    return ev


def check_record(reader):
    rec = json.loads(reader(RECORD))
    require(rec['classification'] == 'ADDITIVE_RUNTIME_NUMERICAL_STABILITY_RESOLUTION' and
            rec['parent_commit'] == AUTHORITY and rec['new_deviation'] is None, 'record kind')
    for flag in ('NOT_A_FROZEN_SPEC_REWRITE', 'NOT_AN_ARCHITECTURE_CHANGE', 'NOT_AN_OPTIMIZER_LR_LOSS_CHANGE',
                 'NO_NEW_DEVIATION'):
        require(flag in rec['status'], 'record status ' + flag)
    for rel, digest in rec['bound_authority_sha256'].items():
        require(sha(reader(rel)) == digest, 'record binding ' + rel)
    require(rec['qualification_evidence']['sha256'] == sha(reader(EV_JSON)), 'qualification evidence bound')
    inv = rec['invariants']
    require(inv['amp_fp16_enabled'] and inv['grad_scaler_enabled'] and inv['successful_scientific_updates'] == 66300
            and not inv['overflowed_attempts_count_as_updates'] and not inv['scientific_group_skipped'], 'invariants')
    require(rec['policy']['name'] == 'ATOMIC_AMP_BACKOFF_RETRY' and rec['policy']['backoff_factor'] == 0.5, 'policy')
    return rec


def check_no_val_path(reader):
    for rel in (RUNNER_PY, QUAL_TOOL):
        src = reader(rel).decode()
        for forbidden in ('val_pairs_v1', 'artifact_probe_v1', 'ArtifactProbe(', 'select_checkpoint', 'generate_bank',
                          'test_pairs'):
            require(forbidden not in src, f'{rel}: {forbidden}')
        require("'VAL'" not in src, f'{rel}: VAL')


def check_config_status(reader, base):
    old, now = base(CONFIG_STATUS), reader(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in STATUS_PHRASES:
        require(phrase in added, 'CONFIG_STATUS M7D1 ' + phrase)
    for bad in ('SCIENTIFIC_TRAINING_COMPLETE', 'M7 COMPLETE', 'SCIENTIFIC_TRAINING_STARTED_AGAIN'):
        require(bad not in added, 'CONFIG_STATUS must not claim ' + bad)
    return len(now) - len(old)


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7C4 authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS, runner.py + ledger/index differ: '
            + json.dumps(changed))
    require(git('ls-files', *('*' + s for s in WEIGHT_SUFFIXES)).decode().strip() == '', 'no weights tracked')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in (*changed, *NEW):
        require(rel == RECORD or not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7D1 candidate: ' + json.dumps(status))


def whitespace():
    for rel in (*NEW, RUNNER_PY):
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
        'M7_amp_overflow_diagnosed': 'LOSS_SCALE_OVERFLOW_CONFIRMED', 'M7_amp_atomic_retry_qualified': True,
        'M7_amp_overflow_fail_closed_fallback': True, 'M7_scientific_training_ready': True,
        'M7_scientific_training_status': 'SCIENTIFIC_TRAINING_BLOCKED_NUMERICAL_POLICY_RESOLVED_READY_TO_RESTART',
        'M7_scientific_restart_started': False, 'failed_attempt_preserved': 'GPAT-B0/E08/seed_42 attempt 2',
        'scientific_runs_started_in_milestone': 0, 'scientific_checkpoint_candidates': 0,
        'TRAIN_access': 'QUALIFICATION_REPLAY_OF_SCIENTIFIC_EPOCH2_GROUPS_1_TO_886', 'VAL_access': False,
        'TEST_access': False, 'GPU_contacted': True, 'M8_bank': False, 'downstream_training': False,
        'new_deviation_numbers': [], 'amendment_created': False, 'method_status_modified': False,
        'deviation_report_modified': False, 'stage_state_modified': False, 'config_status_modified': True,
        'commit': False, 'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY,
        'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': prefix_sha,
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
            'purpose': ('M7D1-N1: diagnose the GPAT-B0/E08/seed-42 attempt-2 FAIL_CLOSED_AMP_OVERFLOW at update 1971 '
                        'by an exact qualification replay, and add + qualify the additive ATOMIC_AMP_BACKOFF_RETRY '
                        'runtime policy; no scientific run started or resumed'),
            'notes': notes,
            'command': ('GPU host (gpat-m7-gpu, PYTHONHASHSEED=42): python -B tools/m7d1_gpat_amp_retry_qualification.py '
                        '--all --workdir ...; laptop: --assemble; python3 -B tools/m7d1_gpat_amp_retry_preflight.py '
                        '--before-ledger / --append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_protected(read, at_authority)
    check_runner_additive(read, at_authority)
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
