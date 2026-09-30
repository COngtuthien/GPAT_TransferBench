"""Verify the M7C2a candidate: GPAT implementation/runtime contract freeze, CPU environment, pinned NAFNet source.

M7C2a adds the owner-resolution record configs/amendments/gpat_m7c2a_implementation_resolution.yaml (layered on A10 and
the M7B record; not an amendment, no DEV), the gpat-m7-cpu environment capture, the NAFNet source pin (additive entry in
third_party/source_pins.json), the runtime/source qualification infrastructure under methods/gpat/ (no architecture,
losses or training loop) and the CPU qualification evidence. No GPU, no TRAIN/VAL/TEST sample, no training, checkpoint
or bank.

STATIC: stdlib only (no torch/numpy/PIL). Candidate-time checks (HEAD lock, worktree, ledger, index) run only here;
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
AUTHORITY = '5f04912b652ae93f72ef6fc2dcdaa41dea55760d'
BRANCH = 'm6-baselines'
MILESTONE = 'M7C2a'
CLASSIFICATION = 'M7C2A_GPAT_RUNTIME_CONTRACT_FREEZE'
RECORD_KIND = 'IMPLEMENTATION_RUNTIME_CONTRACT_FREEZE / CPU_ENVIRONMENT_QUALIFICATION / PINNED_SOURCE_QUALIFICATION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 137
LEDGER_PREFIX_SHA = '6aeb19eeb9dadaa322d8f09f39758f522e1a93da992d707073dfc448552262bb'
INDEX_BASELINE_ROWS = 827
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
PINS = 'third_party/source_pins.json'
MODIFIED = (CONFIG_STATUS, PINS)
RECORD = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'
EV_JSON = 'outputs/audit/M7C2A_GPAT_CPU_QUALIFICATION.json'
EV_MD = 'outputs/audit/M7C2A_GPAT_RUNTIME_CONTRACT.md'
LOCK = 'environments/gpat_m7_cpu.lock.json'
ENV_FILES = ('environments/gpat_m7_cpu.conda-explicit.txt', 'environments/gpat_m7_cpu.pip-freeze.txt',
             'environments/gpat_m7_cpu.runtime.json', LOCK)
CODE = ('methods/gpat/__init__.py', 'methods/gpat/naf_source.py', 'methods/gpat/runtime_contract.py',
        'methods/gpat/teacher_preprocess.py')
TESTS = 'tests/test_m7c2a_gpat_runtime_contract.py'
QUAL_TOOL = 'tools/m7c2a_gpat_cpu_qualification.py'
PREFLIGHT = 'tools/m7c2a_gpat_runtime_contract_preflight.py'
NEW = tuple(sorted((RECORD, EV_JSON, EV_MD, *ENV_FILES, *CODE, TESTS, QUAL_TOOL, PREFLIGHT)))
GPAT_DIR = ['.gitkeep', '__init__.py', 'naf_source.py', 'runtime_contract.py', 'teacher_preprocess.py']
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
NAFNET_COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'
NAF_SHA = {'basicsr/models/archs/NAFNet_arch.py': '01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967',
           'basicsr/models/archs/arch_util.py': '5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005',
           'basicsr/models/archs/local_arch.py': 'c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26',
           'LICENSE': 'a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad'}
RESOLUTIONS = ['ADVERSARIAL_BCE', 'IDENTITY_CLASS_ORDER', 'N-01', 'N-03', 'N-04', 'N-05', 'N-06', 'N-07', 'N-08',
               'N-09', 'R-04', 'R-05', 'SHAPE_TRACE']
LEVEL1_GATES = {'facexformer_matrix_vs_library_float_operator_max_abs': 1e-6,
                'facexformer_clip_emulating_vs_frozen_pil_uint8_max_lsb': 1.13,
                'facexformer_clip_emulating_vs_frozen_normalized_max_abs': 0.0198,
                'adaface_matrix_vs_library_float_operator_max_abs': 1e-6,
                'adaface_adapter_vs_frozen_cv2_uint8_max_abs': 0.5 / 127.5 + 1e-6,
                'highpass_float_path_max_abs': 1e-6,
                'artifact_probe_float_preprocessing_max_abs': 1e-6}
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
PROTECTED = (SPEC, 'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
             'configs/amendments/gpat_m7b_owner_clarifications.yaml',
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
             *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA), 'outputs/audit/method_status.csv',
             'outputs/audit/M6E_FINAL_M6_CLOSURE.json', 'outputs/audit/M6E_FINAL_M6_CLOSURE.md',
             'outputs/audit/STAGE_STATE.json', 'outputs/audit/deviation_report.md', 'configs/frozen/fair_track_v1.yaml',
             'third_party/registry.yaml', 'models/registry.yaml', 'environments/m6_core_gpu.conda-explicit.txt',
             'environments/m6_core_gpu.pip-freeze.txt', 'environments/m6_core_gpu.lock.json',
             'environments/m6_core_gpu.runtime.json', 'gpatbench/preprocess/aux_models.py', 'gpatbench/probe/preprocess.py')
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'configs/methods/', 'frozen_config_snapshot/', 'gpatbench/',
                      'manifests/', 'models/', 'runs/', 'data/', 'cache/')


def require(ok, message):
    if not ok:
        raise ValueError('M7C2a: ' + message)


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
def check_record(reader):
    rec = json.loads(reader(RECORD))
    require(rec['authority_commit'] == AUTHORITY and rec['milestone'] == MILESTONE, 'record authority')
    for rel, digest in rec['bound_authority_sha256'].items():
        require(sha(reader(rel)) == digest, 'bound authority ' + rel)
    for rel, digest in CONFIG_SHA.items():
        require(rec['bound_authority_sha256'][rel] == digest, 'config binding ' + rel)
    require(sorted(rec['resolutions']) == RESOLUTIONS, 'resolution set')
    require(rec['new_deviation'] is None and 'NO_NEW_DEVIATION' in rec['status'], 'no DEV')
    for flag in ('a10_edited', 'm7b_record_edited', 'frozen_spec_edited', 'gpat_configs_edited'):
        require(rec[flag] is False, flag)
    r = rec['resolutions']
    require({k: v['threshold'] for k, v in r['R-04']['level1']['gates'].items()} == LEVEL1_GATES, 'Level-1 gates')
    require(r['R-04']['facexformer_adapter']['name'] == 'CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY', 'R-04 FX')
    require(r['R-04']['level2']['executed_in_m7c2a'] is False and
            r['R-04']['level2']['thresholds_invented_now'] is False, 'R-04 Level 2 deferred')
    require(r['R-05']['option'] == 'D' and r['R-05']['pinned_commit'] == NAFNET_COMMIT and
            r['R-05']['files_sha256'] == NAF_SHA, 'R-05')
    require(r['N-01']['imagenet_normalization'] is False and r['N-01']['applies_to'] == ['E_art', 'F_art', 'PatchGAN_D'],
            'N-01')
    require('nearest-exact' in r['N-03']['resize'], 'N-03')
    require(r['N-04']['scheme'] == 'PRE_UPDATE_JOINT_GRADIENT' and r['N-04']['g_reforward_through_updated_d'] is False,
            'N-04')
    require(r['N-05']['d_trains_from_generator_update'] == 1, 'N-05')
    require(r['N-06']['sync_batchnorm'] == 'FORBIDDEN', 'N-06')
    require(r['N-07']['main_lr']['endpoints'] == {'1': 0.0, '5525': 0.0002, '5526': 0.0002, '66300': 2e-06}, 'N-07 LR')
    require(r['N-07']['attack_warmup_lr']['endpoints'] == {'1': 0.0001, '1390': 0.0}, 'N-07 warmup LR')
    require(r['N-08']['lferr_removed'] is False and r['N-09']['adopted_in_m7c2a'] is False, 'N-08/N-09')
    require(r['SHAPE_TRACE']['bottleneck_film'] is False, 'shape trace')
    require(r['ADVERSARIAL_BCE']['L_D'] == '0.5 * (L_D_real + L_D_fake)' and
            r['ADVERSARIAL_BCE']['L_D_class'] == 'OWNER_IMPLEMENTATION_CLARIFICATION', 'L_D formula')
    require(rec['residual_open_items']['M7C2A-OBS-01']['status'] == 'RESOLVED_M7C2A_OWNER', 'OBS-01 resolved')
    obs2 = rec['residual_open_items']['M7C2A-OBS-02']
    require(obs2['class'] == 'PACKAGING_METADATA_ANOMALY' and obs2['distribution_metadata_version'] == '1.9.0' and
            obs2['pywt___version___patched'] is False, 'OBS-02 recorded')
    require(rec['unresolved_blockers_for_m7c2b'] == [] and list(rec['remaining_deferred']) == ['R-04_LEVEL2'],
            'only R-04 Level 2 remains deferred')
    require('BEFORE' in r['R-04']['level2']['requantization_replicas']['meaning'], 'Level-2 dither semantics')
    for token in ('STATIC_CORE_IMPLEMENTATION_ALLOWED', 'GPU_RUNTIME_NOT_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_ALLOWED'):
        require(token in rec['m7_status_after'], 'status ' + token)
    return rec


def check_pins(reader, old_reader):
    raw = reader(PINS)
    pins, old = json.loads(raw), json.loads(old_reader(PINS))
    require(raw == (json.dumps(pins, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode(), 'pins format')
    require({k: v for k, v in pins['sources'].items() if k != 'nafnet'} == old['sources'], 'pins additive: sources')
    require({k: v for k, v in pins.items() if k != 'sources'} == {k: v for k, v in old.items() if k != 'sources'},
            'pins additive: top level')
    nf = pins['sources']['nafnet']
    require(nf['pinned_commit'] == NAFNET_COMMIT and nf['repository'] == 'https://github.com/megvii-research/NAFNet',
            'nafnet pin')
    require({k: v['sha256'] for k, v in nf['cited_files'].items()} == NAF_SHA, 'nafnet cited files')
    require(nf['model_weights_downloaded'] is False and nf['binary_weight_files_present_now'] == [], 'no weights')
    return nf


def check_environment(reader):
    lock = json.loads(reader(LOCK))
    require(lock['environment_name'] == 'gpat-m7-cpu' and lock['gpu_environment_created'] is False, 'env identity')
    require(lock['existing_environments_mutated'] is False, 'existing environments untouched')
    require(lock['versions']['python'] == '3.11.16' and lock['versions']['torch'] == '2.12.1+cu130' and
            lock['versions']['torchvision'] == '0.27.1+cu130' and lock['versions']['numpy'] == '2.4.6' and
            lock['versions']['Pillow'] == '12.3.0' and lock['versions']['opencv'] == '5.0.0' and
            lock['versions']['ptwt'] == '1.0.1' and lock['versions']['PyWavelets'] == '1.9.0', 'env versions')
    base = reader('environments/m6_core_gpu.pip-freeze.txt').decode().splitlines()
    new = reader('environments/gpat_m7_cpu.pip-freeze.txt').decode().splitlines()
    require(sorted(set(new) - set(base)) == ['PyWavelets==1.9.0', 'ptwt==1.0.1'] and not set(base) - set(new),
            'freeze = base + approved additions only')
    require(lock['freeze_diff_vs_base'] == {'added': ['PyWavelets==1.9.0', 'ptwt==1.0.1'], 'removed': [], 'changed': []},
            'lock freeze diff')
    for rel, digest in lock['capture_sha256'].items():
        require(sha(reader(rel)) == digest, 'env capture ' + rel)
    for name, add in lock['approved_additions'].items():
        require(add['sha256'] == add['pypi_sha256'], 'wheel digest ' + name)
    require(lock['base']['pip_freeze_sha256'] == sha(reader('environments/m6_core_gpu.pip-freeze.txt')), 'base binding')
    return lock


def check_evidence(reader):
    ev = json.loads(reader(EV_JSON))
    require(ev['status'] == 'PASS' and ev['milestone'] == MILESTONE, 'evidence status')
    require(ev['level2_executed'] is False and ev['gpu_used'] is False and ev['dataset_images_read'] == 0, 'scope')
    require(ev['environment']['python'] == '3.11.16' and ev['environment']['cuda_available'] is False, 'evidence env')
    l1 = ev['r04_level1']
    require(l1['gates'] == LEVEL1_GATES, 'evidence gates')
    for k, g in LEVEL1_GATES.items():
        require(l1['measured'][k] <= g and l1['pass'][k] is True, 'Level-1 ' + k)
    require(l1['repeat_bitwise_identical'] is True, 'Level-1 repeatability')
    for n, case in ev['r04_gradient_structure']['cases'].items():
        require(case['input_grad_finite'] and case['input_grad_abs_sum'] > 0 and case['teacher_param_grads_none'] and
                case['targets_detached'], 'gradient structure ' + n)
    naf = ev['nafnet']
    require(naf['commit'] == NAFNET_COMMIT and naf['files_sha256'] == NAF_SHA, 'evidence NAF source')
    require(naf['gradcheck_fp64'] == {'LayerNormFunction': True, 'NAFBlock': True}, 'gradcheck')
    require(naf['saved_variables_verbatim'] is True and naf['basicsr_or_lmdb_imported'] is False, 'NAF loader')
    require(ev['ptwt']['D11_max_abs'] < 1e-5, 'D11')
    require(ev['n09_cpu_pre_evidence']['status'] == 'CPU_PRE_EVIDENCE_ONLY_NOT_ADOPTED', 'N-09 not adopted')
    pv = ev['pywavelets_provenance']
    require(pv['importlib_metadata_version'] == '1.9.0' and pv['pywt___version__'] == '1.8.0' and
            pv['module_in_environment'] is True and pv['environment_name'] == 'gpat-m7-cpu' and
            pv['installed_version_py_matches_record'] is True and pv['patched'] is False, 'PyWavelets provenance')
    require(ev['adversarial_bce']['abs_difference_float64'] <= 1e-12, 'L_D concatenated-mean equivalence')
    return ev


def check_scope(listdir):
    require(listdir('methods/gpat') == GPAT_DIR, 'methods/gpat holds only the M7C2a infrastructure')


def check_config_status(reader, base):
    old, now = base(CONFIG_STATUS), reader(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in ('IMPLEMENTATION_RUNTIME_CONTRACT_FROZEN', 'CPU_ENVIRONMENT_QUALIFIED', 'STATIC_CORE_NOT_STARTED',
                   'GPU_RUNTIME_NOT_STARTED', 'SCIENTIFIC_TRAINING_NOT_STARTED', 'STATIC_CORE_IMPLEMENTATION_ALLOWED',
                   'GPU_RUNTIME_NOT_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_ALLOWED', 'M6_CLOSED = true'):
        require(phrase in added, 'CONFIG_STATUS M7C2a ' + phrase)
    return len(now) - len(old)


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7B authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS + source_pins + ledger/index differ: '
            + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in PROTECTED:
        require(git('diff', '--name-only', AUTHORITY, '--', rel).decode().strip() == '', 'protected ' + rel)
    for rel in (*changed, *NEW):
        require(not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7C2a candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF, no trailing ws ' + rel)
    for rel in MODIFIED:
        added = read(rel)[len(at_authority(rel)):] if rel == CONFIG_STATUS else read(rel)
        require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF, no trailing ws ' + rel)


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
    'M7_runtime_contract_frozen': True, 'M7_cpu_environment_qualified': True, 'M7_static_core_started': False,
    'M7_gpu_runtime_started': False, 'M7_implementation_started': False, 'M7_scientific_training_started': False,
    'owner_resolutions': RESOLUTIONS, 'resolved_observations': {'M7C2A-OBS-01': 'L_D = 0.5 * (L_D_real + L_D_fake)'},
    'recorded_anomalies': {'M7C2A-OBS-02': 'PACKAGING_METADATA_ANOMALY'}, 'remaining_deferred': ['R-04_LEVEL2'],
    'unresolved_blockers_for_m7c2b': [], 'M7_static_core_implementation_allowed': True,
    'M7_gpu_runtime_qualified': False, 'M7_scientific_training_allowed': False,
    'r04_level1_frozen': True, 'r04_level2_executed': False, 'r05_option': 'D', 'nafnet_commit': NAFNET_COMMIT,
    'nafnet_files_sha256': NAF_SHA, 'new_deviation_numbers': [], 'amendment_created': False,
    'gpat_code_created': True, 'gpat_architecture_created': False, 'environment_created': True,
    'environment_name': 'gpat-m7-cpu', 'packages_installed_into_new_environment_only': True,
    'approved_additions': {'PyWavelets': '1.9.0', 'ptwt': '1.0.1'}, 'existing_environments_mutated': False,
    'method_status_modified': False, 'stage_state_modified': False, 'config_status_modified': True,
    'source_pins_modified': 'ADDITIVE_NAFNET_ENTRY_ONLY', 'TEST_access': False, 'TRAIN_access': False,
    'VAL_access': False, 'GPU_contacted': False, 'M8_bank': False, 'training_runs': 0, 'scientific_runs': 0,
    'optimizer_steps': 0, 'checkpoint_writes': 0, 'commit': False, 'push': False, 'authority_commit': AUTHORITY,
    'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS, 'committed_prefix_sha256': LEDGER_PREFIX_SHA,
    'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW),
    'modified_existing_files': list(MODIFIED),
    'input_artifacts': [SPEC, 'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
                        'configs/amendments/gpat_m7b_owner_clarifications.yaml', *CONFIG_SHA,
                        'environments/m6_core_gpu.conda-explicit.txt', 'environments/m6_core_gpu.pip-freeze.txt',
                        'gpatbench/preprocess/aux_models.py', 'gpatbench/probe/preprocess.py', PINS, CONFIG_STATUS],
    'output_artifacts': sorted((*NEW, *MODIFIED))}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'], 'ledger tests recorded')
    for t in row['tests']:
        require(t['failures'] == 0 or t.get('failures_preexisting'), 'failures classified: ' + t['scope'])
        require(t['errors'] == 0 or t.get('errors_preexisting_environmental'), 'errors classified: ' + t['scope'])


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': ('M7C2a: GPAT implementation/runtime contract freeze (owner resolutions R-04, R-05, N-01, N-03..N-09, '
                        'adversarial BCE, identity class order, shape trace), gpat-m7-cpu environment and pinned NAFNet '
                        'source qualification; no GPAT architecture, training, checkpoint or bank'),
            'notes': ('R-04: FaceXFormer CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY and AdaFace area-matrix adapters; Level-1 '
                      'gates frozen and passed on the 44-image synthetic corpus; Level 2 contract frozen, executed later '
                      'on GPU (TRAIN only; dither before the teacher uint8 quantization, 8 replicas). R-05: option D, AST-selected SimpleGate/NAFBlock/LayerNormFunction/LayerNorm2d '
                      'from the hash-verified third_party/source_cache/nafnet pin, saved_variables verbatim. gpat-m7-cpu = '
                      'exact gpat-m5 capture + ptwt 1.0.1 + PyWavelets 1.9.0. M7C2A-OBS-01 resolved: L_D = 0.5 * (L_D_real + '
                      'L_D_fake) (= balanced concatenated mean BCE); L_Gadv = mean BCE(D(x_hat), 1). M7C2A-OBS-02 '
                      'PACKAGING_METADATA_ANOMALY (dist 1.9.0, pywt.__version__ 1.8.0, wheel hash pinned). Remaining '
                      'deferred: R-04 Level 2 (blocks GPU/scientific training only).'),
            'command': ('laptop only: ~/.venvs/gpat-m7-cpu/bin/python -B tools/m7c2a_gpat_cpu_qualification.py --write; '
                        'python3 -B tools/m7c2a_gpat_runtime_contract_preflight.py --before-ledger / --append-ledger / '
                        '--rebuild-index; ~/.venvs/gpat-m7-cpu/bin/python -B -m unittest '
                        'tests/test_m7c2a_gpat_runtime_contract.py'),
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
    check_record(read)
    check_pins(read, at_authority)
    check_environment(read)
    check_evidence(read)
    check_scope(listdir_worktree)
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
