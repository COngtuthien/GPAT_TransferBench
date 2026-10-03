"""Verify the M7C2b candidate: GPAT static core implementation + synthetic CPU qualification.

M7C2b adds the static GPAT core under methods/gpat/ (config loader, wavelet, high-pass, E_art, heads, GRL, G_res,
composition/artifact map, PatchGAN D, spectra, pure losses, EMA utility, schedule/batching/identity-map helpers and the
GPATCore facade), the synthetic CPU evidence and its tests. No training loop, optimizer, DataLoader, checkpoint, bank,
GPU, teacher Level-2 run or TRAIN/VAL/TEST sample. The frozen spec, A1-A10, M7B, M7C2a and B0-B3 are unchanged.

STATIC: stdlib only (no torch/numpy). Candidate-time checks (HEAD lock, worktree, ledger, index) run only here;
check_* take readers so the regression tests can pin them to the commit that added them.

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
AUTHORITY = 'c9a12e10eff959a31aaa361cbff98469aae1e7c2'
BRANCH = 'm6-baselines'
MILESTONE = 'M7C2b'
CLASSIFICATION = 'M7C2B_GPAT_STATIC_CORE_IMPLEMENTATION'
RECORD_KIND = 'STATIC_CORE_IMPLEMENTATION / SYNTHETIC_CPU_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 138
LEDGER_PREFIX_SHA = '9398aadb8bfd0800f59910c4740bf31b1dd93263f1a1822ec312c35d6925188b'
INDEX_BASELINE_ROWS = 841
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
PACKAGE_INIT = 'methods/gpat/__init__.py'          # docstring-only current-state update (owner review)
MODIFIED = (CONFIG_STATUS, PACKAGE_INIT)
RESOLUTION = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'
RESOLUTION_SHA = 'b61555a2ca15954bdd8929d92c28d0d957561c595ea38eeef496d06c871e8e70'
LOCK = 'environments/gpat_m7_cpu.lock.json'
LOCK_SHA = 'd52b5e7ccc8ee0f6c289db9f31d4521e4b241a5e22664912a6185e073be9c23a'
EV_JSON = 'outputs/audit/M7C2B_GPAT_STATIC_CORE.json'
EV_MD = 'outputs/audit/M7C2B_GPAT_STATIC_CORE.md'
STATIC_CORE = ('artifact_encoder.py', 'batching.py', 'composition.py', 'config.py', 'discriminator.py', 'ema.py',
               'generator.py', 'grl.py', 'heads.py', 'highpass.py', 'identity_labels.py', 'losses.py', 'model.py',
               'schedule.py', 'spectral.py', 'wavelet.py')
CODE = tuple('methods/gpat/' + n for n in STATIC_CORE)
M7C2A_CODE = ('methods/gpat/naf_source.py', 'methods/gpat/runtime_contract.py', 'methods/gpat/teacher_preprocess.py')
TESTS = 'tests/test_m7c2b_gpat_static_core.py'
EV_TOOL = 'tools/m7c2b_gpat_static_core_evidence.py'
PREFLIGHT = 'tools/m7c2b_gpat_static_core_preflight.py'
NEW = tuple(sorted((*CODE, EV_JSON, EV_MD, TESTS, EV_TOOL, PREFLIGHT)))
GPAT_DIR = sorted(['.gitkeep', '__init__.py', 'naf_source.py', 'runtime_contract.py', 'teacher_preprocess.py',
                   *STATIC_CORE])
FORBIDDEN_GPAT_FILES = ('train.py', 'training.py', 'trainer.py', 'runner.py', 'checkpoint.py', 'bank.py')
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
PROTECTED = (SPEC, 'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
             'configs/amendments/gpat_m7b_owner_clarifications.yaml', RESOLUTION,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
             *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA), 'outputs/audit/method_status.csv',
             'outputs/audit/M6E_FINAL_M6_CLOSURE.json', 'outputs/audit/M6E_FINAL_M6_CLOSURE.md',
             'outputs/audit/M7B_GPAT_CONFIG_FREEZE.json', 'outputs/audit/M7C2A_GPAT_CPU_QUALIFICATION.json',
             'outputs/audit/STAGE_STATE.json', 'outputs/audit/deviation_report.md', 'configs/frozen/fair_track_v1.yaml',
             'third_party/registry.yaml', 'third_party/source_pins.json', 'models/registry.yaml',
             'environments/gpat_m7_cpu.conda-explicit.txt', 'environments/gpat_m7_cpu.pip-freeze.txt',
             'environments/gpat_m7_cpu.runtime.json', LOCK, 'gpatbench/preprocess/aux_models.py',
             'gpatbench/probe/preprocess.py', *M7C2A_CODE)
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'configs/methods/', 'configs/amendments/', 'frozen_config_snapshot/',
                      'gpatbench/', 'manifests/', 'models/', 'runs/', 'data/', 'cache/', 'environments/', 'third_party/')
STATUS_PHRASES = ('STATIC_CORE_IMPLEMENTED', 'CPU_SYNTHETIC_QUALIFIED', 'GPU_RUNTIME_NOT_QUALIFIED',
                  'SCIENTIFIC_TRAINING_NOT_ALLOWED', 'R-04 Level 2', 'M6_CLOSED = true')
WEIGHT_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors', '.pkl')


def require(ok, message):
    if not ok:
        raise ValueError('M7C2b: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg'}, 'no weight/data ' + path)
    return (ROOT / p).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- reader-scoped checks
def check_protected(reader, base):
    for rel in PROTECTED:
        require(reader(rel) == base(rel), 'protected unchanged ' + rel)
    for rel, digest in CONFIG_SHA.items():
        require(sha(reader(rel)) == digest and sha(reader('frozen_config_snapshot/' + rel)) == digest, 'B0-B3 ' + rel)
    require(sha(reader(RESOLUTION)) == RESOLUTION_SHA, 'M7C2a resolution hash')
    require(sha(reader(LOCK)) == LOCK_SHA, 'gpat-m7-cpu lock hash')
    rec = json.loads(reader(RESOLUTION))
    require(list(rec['remaining_deferred']) == ['R-04_LEVEL2'] and
            rec['resolutions']['R-04']['level2']['executed_in_m7c2a'] is False, 'R-04 Level 2 still deferred')


def check_evidence(reader):
    ev = json.loads(reader(EV_JSON))
    require(ev['status'] == 'PASS' and ev['milestone'] == MILESTONE and ev['authority_commit'] == AUTHORITY,
            'evidence status')
    require(all(ev['gates'].values()), 'evidence gates')
    require(ev['gpu_used'] is False and ev['environment']['cuda_available'] is False, 'no GPU')
    require(ev['dataset_images_read'] == 0 and ev['synthetic_only'] is True, 'no dataset access')
    require(ev['teacher_weights_loaded'] is False and ev['imagenet_weight_file_read'] is False, 'no teacher/asset weights')
    require(ev['training_steps'] == 0 and ev['checkpoint_writes'] == 0, 'no training/checkpoint')
    require(ev['environment']['ptwt'] == '1.0.1' and ev['environment']['PyWavelets_distribution'] == '1.9.0' and
            ev['environment']['torch'] == '2.12.1+cu130', 'gpat-m7-cpu environment')
    require({k: v['total'] for k, v in ev['parameter_counts'].items()} == EXPECTED_COUNTS, 'parameter counts')
    require(ev['config_sha256'] == {f'B{i}': CONFIG_SHA[f'configs/methods/gpat_b{i}.yaml'] for i in range(4)},
            'config hashes')
    require(ev['wavelet']['D11_max_abs'] < 1e-5, 'D11')
    require(ev['highpass']['max_abs_vs_cv2_reflect101'] <= 1e-6, 'HP parity')
    d12 = ev['d12']
    require(d12['zero_residual_x_hat_minus_idwt_dwt_x_t_max_abs'] < 1e-5 and
            d12['zero_residual_x_hat_minus_x_t_max_abs'] < 1e-5 and d12['source_independence_max_abs'] < 1e-5 and
            d12['non_vacuous_live_source_dependence_max_abs'] > 1e-5 and d12['gamma0_LL_syn_equals_LL_t_exact'], 'D12')
    heads = {v: (t['attack_logits'], t['identity_logits']) for v, t in ev['shape_traces'].items()}
    require(heads == {'B0': (None, None), 'B1': ([1, 6], None), 'B2': (None, [1, 60]), 'B3': ([1, 6], [1, 60])},
            'variant heads')
    require(all(t['D_logits'] == [1, 1, 30, 30] and t['x_hat'] == [1, 3, 256, 256] and t['A'] == [1, 1, 256, 256]
                for t in ev['shape_traces'].values()), 'shape traces')
    b = ev['batching']
    require((b['microbatches'], b['optimizer_groups'], b['last_group']) == (2210, 1105, [4, 2]), 'batching')
    ll, oc = ev['l_low'], ev['owner_clarifications']
    require(ll['zero_residual'] < 1e-5 and ll['perturbed_x_hat'] > 1e-3 and ll['grad_to_x_hat_abs_sum'] > 0 and
            ll['selection_lferr_gamma0_max'] == 0.0, 'L_low literal / LFErr exact zero')
    require(oc['L_low_training']['definition'] == 'mean |DWT(x_hat).LL - LL_t|' and
            oc['L_low_training']['uses_internal_LL_syn'] is False and
            oc['LFErr_selection']['used_as_training_loss'] is False, 'L_low vs LFErr contracts distinct')
    require(oc['patchgan']['class'] == 'IMPLEMENTATION_CLARIFICATION' and oc['patchgan']['new_deviation'] is None and
            oc['patchgan']['sigmoid'] is False, 'PatchGAN convention')
    require(oc['dev022_group_normalization']['status'] == 'RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED' and
            oc['dev022_group_normalization']['microbatch_sample_weight_applied_to_idadv'] is False, 'DEV-022 runner')
    return ev


def check_scope(listdir):
    names = listdir('methods/gpat')
    require(names == GPAT_DIR, 'methods/gpat holds exactly the M7C2a infrastructure + static core')
    require(not set(names) & set(FORBIDDEN_GPAT_FILES), 'no training runner')


def check_config_status(reader, base):
    old, now = base(CONFIG_STATUS), reader(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in STATUS_PHRASES:
        require(phrase in added, 'CONFIG_STATUS M7C2b ' + phrase)
    return len(now) - len(old)


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7C2a authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS + ledger/index differ: ' + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('ls-files', *('*' + s for s in WEIGHT_SUFFIXES)).decode().strip() == '', 'no weights tracked')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in (*changed, *NEW):
        require(not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def no_run_artifacts():
    """No GPAT run, checkpoint or bank anywhere in the runtime trees (ignored files included)."""
    for top in ('runs', 'outputs', 'models', 'cache'):
        base = ROOT / top
        if not base.exists():
            continue
        for p in base.rglob('*'):
            low = str(p.relative_to(ROOT)).lower()
            if top == 'runs':
                require('gpat' not in low and not any(q.startswith(('m7', 'e08', 'e09', 'e10', 'e11'))
                                                      for q in p.relative_to(base).parts), 'GPAT run ' + low)
            require(not ('gpat' in low and (p.suffix in WEIGHT_SUFFIXES or 'bank' in low)),
                    'GPAT checkpoint/bank artifact ' + low)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7C2b candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF, no trailing ws ' + rel)
    added = read(CONFIG_STATUS)[len(at_authority(CONFIG_STATUS)):]
    require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF, no trailing ws ' + CONFIG_STATUS)
    raw = read(PACKAGE_INIT)
    require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF ' + PACKAGE_INIT)


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
    'record_kind': RECORD_KIND, 'exit_code': 0, 'M6_closed': True, 'M7_started': True, 'M7_configs_frozen': True,
    'M7_runtime_contract_frozen': True, 'M7_cpu_environment_qualified': True, 'M7_static_core_started': True,
    'M7_static_core_implemented': True, 'M7_cpu_synthetic_qualified': True, 'M7_gpu_runtime_qualified': False,
    'M7_gpu_runtime_started': False, 'M7_scientific_training_allowed': False, 'M7_scientific_training_started': False,
    'remaining_deferred': ['R-04_LEVEL2'], 'r04_level2_executed': False, 'new_deviation_numbers': [],
    'amendment_created': False, 'gpat_code_created': True, 'gpat_architecture_created': True,
    'training_loop_created': False, 'environment_created': False, 'environment_name': 'gpat-m7-cpu',
    'existing_environments_mutated': False, 'packages_installed': False, 'method_status_modified': False,
    'deviation_report_modified': False, 'stage_state_modified': False, 'config_status_modified': True,
    'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False, 'GPU_contacted': False, 'M8_bank': False,
    'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'commit': False,
    'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': list(MODIFIED),
    'parameter_counts': EXPECTED_COUNTS, 'config_sha256': CONFIG_SHA,
    'input_artifacts': [SPEC, 'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
                        'configs/amendments/gpat_m7b_owner_clarifications.yaml', RESOLUTION, *CONFIG_SHA, LOCK,
                        *M7C2A_CODE, 'models/registry.yaml', CONFIG_STATUS],
    'output_artifacts': sorted((*NEW, *MODIFIED))}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'], 'ledger tests recorded')
    focused = [t for t in row['tests'] if t['scope'].startswith(TESTS + ' (gpat-m7-cpu')]
    require(len(focused) == 1 and focused[0]['failures'] == focused[0]['errors'] == focused[0]['skipped'] == 0 and
            focused[0]['ran'] > 0, 'mandatory focused tests pass in gpat-m7-cpu without skips')
    for t in row['tests']:
        require(t['failures'] == 0 or t.get('failures_preexisting'), 'failures classified: ' + t['scope'])
        require(t['errors'] == 0 or t.get('errors_preexisting_environmental'), 'errors classified: ' + t['scope'])


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': ('M7C2b: GPAT static core implementation (config loader, ptwt wavelet, HP, E_art, attack/identity '
                        'heads, GRL, NAFResidualUNet G_res with FiLM, composition, artifact map, PatchGAN D, spectra, pure '
                        'losses, EMA, schedule/batching/identity-map helpers, GPATCore facade) and synthetic CPU '
                        'qualification; no training loop, checkpoint, bank, GPU or dataset access'),
            'notes': ('Owner review corrections: training L_low = mean |DWT(x_hat).LL - LL_t| (frozen spec 10.1 literal, '
                      'fresh differentiable DWT of x_hat); N-08 selection LFErr stays on the internal LL_syn (exact 0 at '
                      'gamma 0); the two contracts are separate functions. PatchGAN pix2pix norm=instance convention '
                      'frozen as IMPLEMENTATION_CLARIFICATION. DEV-022 group normalization sum CE / sum labelled count, '
                      'never re-weighted by n_j / n_group: RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED. '
                      'methods/gpat/__init__.py docstring updated to the current state (no code change). Counts: G_res 31,677,421; E_art 11,204,736 (no FC); D 2,767,809; attack head 3,078; identity head '
                      '30,780 (all trainable). FiLM affine zero-initialized (identity modulation; owner M7C2b section 12, '
                      'no contrary authority). Bottleneck 16->8 pool = AvgPool2d(2, 2) (owner M7C2b section 13; exact '
                      'adaptive-pool equivalent, CPU forward/grad bitwise equal). PatchGAN follows the pix2pix '
                      'NLayerDiscriminator norm=instance convention (InstanceNorm2d affine=False, conv bias kept). '
                      'artifact_scale in {0, 1} is the A10 D12 qualification hook, not a hyperparameter. E_art '
                      'IMAGENET1K_V1 init reads only the SHA-256 verified local file (never downloads); synthetic tests '
                      'inject a 3-channel state. R-04 Level 2 remains deferred (blocks GPU/scientific training only).'),
            'command': ('laptop only: ~/.venvs/gpat-m7-cpu/bin/python -B tools/m7c2b_gpat_static_core_evidence.py --write; '
                        '~/.venvs/gpat-m7-cpu/bin/python -B -m unittest discover -s tests -p '
                        'test_m7c2b_gpat_static_core.py; python3 -B tools/m7c2b_gpat_static_core_preflight.py '
                        '--before-ledger / --append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def listdir_worktree(rel):
    return sorted(q.name for q in (ROOT / rel).iterdir() if q.name != '__pycache__')


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_protected(read, at_authority)
    check_evidence(read)
    check_scope(listdir_worktree)
    no_run_artifacts()
    status_bytes = check_config_status(read, at_authority)
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
    require(not {'torch', 'numpy', 'PIL', 'cv2'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [*MODIFIED, *bookkeeping],
            'config_status_bytes_added': status_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
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
