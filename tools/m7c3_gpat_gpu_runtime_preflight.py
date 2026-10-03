"""Verify the M7C3 candidate: GPAT GPU runtime qualification + R-04 Level-2 teacher parity.

M7C3 adds the gpat-m7-gpu environment capture/lock, the synthetic GPU qualification harness, the TRAIN-only R-04
Level-2 parity tool, their evidence and tests. No production runner, scientific training, checkpoint, bank, VAL or TEST
image. The frozen spec, A1-A10, M7B, the M7C2a resolution, the M7C2b static core and B0-B3 are unchanged.

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
AUTHORITY = '33955b05ff70289c42386c3ac1623bae4a5fae5e'
BRANCH = 'm6-baselines'
MILESTONE = 'M7C3'
CLASSIFICATION = 'M7C3_GPAT_GPU_RUNTIME_QUALIFICATION'
RECORD_KIND = 'GPU_RUNTIME_QUALIFICATION / R04_LEVEL2_TEACHER_PARITY / GPU_ENVIRONMENT_LOCK'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 139
INDEX_BASELINE_ROWS = 862
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
TEACHER_PRE = 'methods/gpat/teacher_preprocess.py'    # M7C3 exact-forward FaceXFormer adapter (owner decision)
MODIFIED = (CONFIG_STATUS, TEACHER_PRE)
ENV = ('environments/gpat_m7_gpu.conda-explicit.txt', 'environments/gpat_m7_gpu.pip-freeze.txt',
       'environments/gpat_m7_gpu.pip-requirements.txt', 'environments/gpat_m7_gpu.runtime.json',
       'environments/gpat_m7_gpu.lock.json')
LOCK = 'environments/gpat_m7_gpu.lock.json'
EV_GPU = 'outputs/audit/M7C3_GPAT_GPU_QUALIFICATION.json'
EV_REP = 'outputs/audit/M7C3_GPAT_GPU_REPEATABILITY.json'
EV_R04 = 'outputs/audit/M7C3_R04_LEVEL2_TEACHER_PARITY.json'
EV_R04_A1 = 'outputs/audit/M7C3_R04_LEVEL2_ATTEMPT1.json'
EV_R04_A1_SHA = '146de245d093fdc6af7598e608c4f0604c7d4f99a494406679fb4ac90f1a3391'
EV_MD = 'outputs/audit/M7C3_GPAT_GPU_RUNTIME.md'
HARNESS = 'tools/m7c3_gpat_gpu_qualification.py'
R04_TOOL = 'tools/m7c3_r04_level2_parity.py'
PREFLIGHT = 'tools/m7c3_gpat_gpu_runtime_preflight.py'
TESTS = 'tests/test_m7c3_gpat_gpu_runtime.py'
RECORD = 'configs/amendments/gpat_m7c3_gpu_runtime_resolution.yaml'
NEW = tuple(sorted((*ENV, EV_GPU, EV_REP, EV_R04, EV_R04_A1, EV_MD, HARNESS, R04_TOOL, PREFLIGHT, TESTS, RECORD)))
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
RESOLUTION = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'
RESOLUTION_SHA = 'b61555a2ca15954bdd8929d92c28d0d957561c595ea38eeef496d06c871e8e70'
CPU_LOCK = 'environments/gpat_m7_cpu.lock.json'
CPU_LOCK_SHA = 'd52b5e7ccc8ee0f6c289db9f31d4521e4b241a5e22664912a6185e073be9c23a'
BASE_FREEZE = 'environments/m6_core_gpu.pip-freeze.txt'
BASE_CONDA = 'environments/m6_core_gpu.conda-explicit.txt'
SELECTION_SHA = 'd02a8a96eedd18b067fd4c7bf809368ced68b7aff14c88daac053b999baa855f'
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}
TEACHER_SHA = {'adaface_weight': '52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8',
               'facexformer_weight': '327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d',
               'f_art_resnet18_imagenet1k_v1': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
               'artifact_probe_v1': 'b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff'}
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
STATIC_CORE = tuple('methods/gpat/' + n for n in (
    '__init__.py', 'artifact_encoder.py', 'batching.py', 'composition.py', 'config.py', 'discriminator.py', 'ema.py',
    'generator.py', 'grl.py', 'heads.py', 'highpass.py', 'identity_labels.py', 'losses.py', 'model.py',
    'naf_source.py', 'runtime_contract.py', 'schedule.py', 'spectral.py', 'wavelet.py'))
PROTECTED = (SPEC, 'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
             'configs/amendments/gpat_m7b_owner_clarifications.yaml', RESOLUTION,
             'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
             *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA), 'outputs/audit/method_status.csv',
             'outputs/audit/deviation_report.md', 'outputs/audit/STAGE_STATE.json', 'outputs/audit/M7C2B_GPAT_STATIC_CORE.json',
             'third_party/source_pins.json', 'models/registry.yaml', CPU_LOCK, BASE_FREEZE, BASE_CONDA,
             'gpatbench/preprocess/aux_models.py', 'gpatbench/probe/preprocess.py', 'manifests/pairs_train_v1.parquet',
             'manifests/split_v1.parquet', *STATIC_CORE)
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'configs/methods/', 'configs/amendments/', 'frozen_config_snapshot/',
                      'gpatbench/', 'manifests/', 'models/', 'runs/', 'data/', 'cache/', 'methods/', 'third_party/')
STATUS_PHRASES = ('STATIC_CORE_IMPLEMENTED', 'CPU_SYNTHETIC_QUALIFIED', 'GPU_RUNTIME_QUALIFIED',
                  'R04_LEVEL2_TEACHER_PARITY_QUALIFIED', 'PRODUCTION_RUNNER_NOT_IMPLEMENTED',
                  'REAL_TRAIN_QUALIFICATION_NOT_DONE', 'SCIENTIFIC_TRAINING_NOT_ALLOWED', 'M6_CLOSED = true')
WEIGHT_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors', '.pkl')


def require(ok, message):
    if not ok:
        raise ValueError('M7C3: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.png', '.jpg'}, 'no weight/image ' + path)
    return (ROOT / p).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


# ----------------------------------------------------------------- reader-scoped checks
def check_protected(reader, base):
    for rel in PROTECTED:
        require(reader(rel) == base(rel), 'protected unchanged ' + rel)
    for rel, digest in CONFIG_SHA.items():
        require(sha(reader(rel)) == digest and sha(reader('frozen_config_snapshot/' + rel)) == digest, 'B0-B3 ' + rel)
    require(sha(reader(RESOLUTION)) == RESOLUTION_SHA and sha(reader(CPU_LOCK)) == CPU_LOCK_SHA, 'M7C2a record/CPU lock')


def check_environment(reader):
    lock = json.loads(reader(LOCK))
    require(lock['environment_name'] == 'gpat-m7-gpu' and lock['kind'] == 'GPU_EXECUTION_ENVIRONMENT', 'env identity')
    require(lock['existing_environments_mutated'] is False and lock['protected_environment_fingerprints_identical'] is True,
            'protected envs untouched')
    base, new = reader(BASE_FREEZE).decode().splitlines(), reader('environments/gpat_m7_gpu.pip-freeze.txt').decode().splitlines()
    require(sorted(set(new) - set(base)) == ['PyWavelets==1.9.0', 'ptwt==1.0.1'] and not set(base) - set(new),
            'freeze = base + approved additions only')
    conda = lambda raw: sorted(l for l in raw.decode().splitlines() if l.startswith('http'))   # noqa: E731
    require(conda(reader('environments/gpat_m7_gpu.conda-explicit.txt')) == conda(reader(BASE_CONDA)), 'same conda layer')
    req = reader('environments/gpat_m7_gpu.pip-requirements.txt').decode()
    require('ptwt==1.0.1 --hash=sha256:a6f832ee73decd7741ca8a0d1585a18188d773c428107973475e43393707d4af' in req and
            'PyWavelets==1.9.0 --hash=sha256:d76b7fa8fc500b09201d689b4f15bf5887e30ffbe2e1f338eb8470590eb4521a' in req,
            'hash-pinned requirements')
    for rel, digest in lock['capture_sha256'].items():
        require(sha(reader(rel)) == digest, 'env capture ' + rel)
    v = lock['versions']
    require((v['python'], v['torch'], v['torchvision'], v['cuda'], v['cudnn'], v['numpy'], v['Pillow'],
             v['opencv_python_headless'], v['ptwt'], v['PyWavelets'], v['pywt___version__']) ==
            ('3.11.16', '2.12.1+cu130', '0.27.1+cu130', '13.0', 92000, '2.4.6', '12.3.0', '5.0.0.93', '1.0.1', '1.9.0',
             '1.8.0'), 'env versions')
    return lock


def check_gpu_evidence(reader):
    ev = json.loads(reader(EV_GPU))
    require(ev['status'] == 'PASS' and ev['milestone'] == MILESTONE and ev['authority_commit'] == AUTHORITY, 'GPU status')
    require(all(ev['gates'].values()), 'GPU gates')
    require(ev['dataset_images_read'] == 0 and ev['checkpoint_writes'] == 0 and ev['bank_writes'] == 0 and
            ev['training_runs'] == 0 and ev['synthetic_only'] is True, 'GPU harness scope')
    require({k: v['total'] for k, v in ev['static_authority']['parameter_counts'].items()} == EXPECTED_COUNTS, 'counts')
    det = ev['determinism']['observed']
    require(det == {**det, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'cudnn_benchmark': False, 'cudnn_deterministic': True,
                    'tf32_matmul': False, 'tf32_cudnn': False, 'use_deterministic_algorithms': True,
                    'deterministic_warn_only': False}, 'deterministic configuration')
    require(ev['d11_d12']['D11_max_abs'] < 1e-5, 'D11')
    for mode in ev['d11_d12']['D12'].values():
        require(mode['zero_residual_vs_idwt_dwt_x_t_max_abs'] < 1e-5 and mode['source_independence_max_abs'] < 1e-5, 'D12')
    require(ev['environment']['gpu'] == 'NVIDIA GeForce RTX 3090' and ev['environment']['torch'] == '2.12.1+cu130', 'GPU env')
    require(ev['vram']['physical_batch'] == 4 and ev['vram']['oom'] is False, 'batch 4, no OOM')
    rep = json.loads(reader(EV_REP))
    require(rep['bitwise_identical'] is True and rep['process_1_sha256'] == rep['process_2_sha256'], 'repeatability')
    return ev


def check_teacher_preprocess(reader, base):
    """M7C3 may only ADD the exact-forward adapter: every M7C2a definition keeps its exact source text."""
    import ast
    old, new = base(TEACHER_PRE).decode(), reader(TEACHER_PRE).decode()
    told, tnew = ast.parse(old), ast.parse(new)
    defs_old = {n.name: ast.get_source_segment(old, n) for n in told.body if isinstance(n, ast.FunctionDef)}
    defs_new = {n.name: ast.get_source_segment(new, n) for n in tnew.body if isinstance(n, ast.FunctionDef)}
    for name, src in defs_old.items():
        require(defs_new.get(name) == src, 'M7C2a definition unchanged: ' + name)
    added = sorted(set(defs_new) - set(defs_old))
    require(added == sorted(['_pil_bicubic', 'pil_bicubic_fixed_point', 'teacher_uint8', '_clip8_pass',
                             'facexformer_uint8_exact', 'facexformer_normalize_uint8', '_exact_surrogate_function',
                             'facexformer_input_exact']), 'additive exact-forward definitions: ' + json.dumps(added))
    require("FACEXFORMER_ADAPTER_CLASS = 'EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY'" in new, 'adapter class')
    return added


def check_r04(reader):
    a1_raw = reader(EV_R04_A1)
    require(sha(a1_raw) == EV_R04_A1_SHA, 'attempt-1 evidence preserved byte-identical')
    a1 = json.loads(a1_raw)
    require(a1['status'] == 'FAIL' and a1['level2']['gates'] == {'adaface': True, 'landmarks': False, 'parsing': True},
            'attempt-1 recorded as failed on landmarks')
    ev = json.loads(reader(EV_R04))
    require(ev['milestone'] == MILESTONE and ev['authority_commit'] == AUTHORITY and ev['status'] == 'PASS', 'R-04 status')
    s1 = ev['attempt_1']
    require(s1['evidence_sha256'] == EV_R04_A1_SHA and s1['result'] == 'FAIL' and s1['failed_gate'] == 'landmark_abs_max',
            'attempt-1 summary')
    require(s1['floor'] == a1['level2']['thresholds_noise_floor'] and s1['candidate'] == a1['level2']['candidate'],
            'attempt-1 values exact')
    require({k: ev['inventory'][k]['sha256'] for k in TEACHER_SHA} == TEACHER_SHA, 'teacher hashes')
    require({k: a1['inventory'][k]['sha256'] for k in TEACHER_SHA} == TEACHER_SHA, 'attempt-1 teacher hashes')
    a2 = ev['attempt_2']
    require(a2['adapter'] == 'EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY', 'attempt-2 adapter')
    require(a2['floors_from_attempt_1'] == a1['level2']['thresholds_noise_floor'] and a2['floors_recomputed'] is False,
            'original floors verbatim')
    require(a2['selection_sha256'] == a1['selection']['selected_list_sha256'] == SELECTION_SHA, 'same selection')
    require(a2['input_parity']['uint8_bitwise_equal'] == a2['input_parity']['images'] == 192, '192/192 uint8 parity')
    require(ev['level1_exact_gpu']['uint8_bitwise_equal'] == ev['level1_exact_gpu']['images'] == 44, '44/44 corpus')
    f, c = a2['floors_from_attempt_1'], a2['candidate']
    require(c['adaface_cos'] >= f['adaface_cos'] and c['landmark_mean'] <= f['landmark_mean'] and
            c['landmark_abs'] <= f['landmark_abs'] and c['parsing_argmax'] >= f['parsing_argmax'] and
            c['parsing_dice'] >= f['parsing_dice'] and a2['pass'] is True, 'attempt-2 gates against attempt-1 floors')
    sb = ev['surrogate_backward']
    require(sb['finite'] and sb['nonzero'] and sb['bitwise_equal'], 'surrogate backward equals approved VJP')
    require(ev['backward_repeatability']['bitwise_identical'] is True, 'backward two-process repeatability')
    require(ev['access']['all_train'] and ev['access']['opened_equals_selection'] and
            ev['access']['val_images_opened'] == 0 and ev['access']['test_images_opened'] == 0, 'TRAIN-only access')
    require(ev['teacher_gradient']['pass'] is True and ev['teacher_gradient']['memory']['oom'] is False, 'teacher gradient')
    require(ev['laptop_data_partition']['unmounted'] is True, 'laptop partition unmounted')
    return ev


def check_record(reader):
    rec = json.loads(reader(RECORD))
    require(rec['record_kind'] == 'ADDITIVE_RUNTIME_QUALIFICATION_RECORD' and rec['authority_commit'] == AUTHORITY,
            'record kind/authority')
    require(rec['new_deviation'] is None and 'NOT_AN_A10_AMENDMENT' in rec['status'] and 'NO_NEW_DEVIATION' in rec['status'],
            'not an amendment, no DEV')
    for flag in ('frozen_spec_edited', 'a10_edited', 'm7b_record_edited', 'm7c2a_resolution_edited', 'gpat_configs_edited'):
        require(rec[flag] is False, flag)
    for rel, digest in rec['bound_authority_sha256'].items():
        require(sha(reader(rel)) == digest, 'record binding ' + rel)
    require(rec['bound_authority_sha256'][RESOLUTION] == RESOLUTION_SHA, 'M7C2a resolution bound')
    res = rec['resolutions']
    q = res['FACEXFORMER_TEACHER_INPUT_QUANTIZATION']
    require(q['class'] == 'OWNER_RUNTIME_COMPATIBILITY_CLARIFICATION' and
            q['rule'] == 'q = 255 * (x + 1) / 2; u8 = clip(round_half_to_even(q), 0, 255); cast to uint8', 'quantization rule')
    ad = res['FACEXFORMER_TEACHER_ADAPTER']
    require(ad['name'] == 'EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY' and ad['forward']['name'] == 'EXACT_FROZEN_FORWARD'
            and ad['backward']['name'] == 'APPROVED_SURROGATE_BACKWARD' and ad['forward']['pil_call_in_gpu_forward'] is False,
            'adapter contract')
    r4 = res['R04_LEVEL2']
    a1, a2 = json.loads(reader(EV_R04_A1)), json.loads(reader(EV_R04))
    require(r4['final_status'] == 'PASS' and r4['selected_list_sha256'] == SELECTION_SHA, 'R-04 final / selection')
    require(r4['attempt_1']['record_sha256'] == EV_R04_A1_SHA and r4['attempt_1']['result'] == 'FAIL', 'attempt 1')
    require(r4['attempt_2']['record_sha256'] == sha(reader(EV_R04)) and r4['attempt_2']['result'] == 'PASS', 'attempt 2')
    require(r4['floors_from_attempt_1'] == a1['level2']['thresholds_noise_floor'] == a2['attempt_2']['floors_from_attempt_1']
            and r4['floors_recomputed'] is False, 'floors immutable')
    require(r4['replica_seed_list_sha256'] == a2['attempt_2']['replica_seed_list_sha256'], 'seed list')
    c = r4['attempt_2']['candidate']
    require((c['landmark_mean'], c['landmark_abs'], c['parsing_argmax'], c['parsing_dice']) == (0.0, 0.0, 1.0, 1.0),
            'attempt-2 frozen results')
    ip = r4['attempt_2']['input_parity']
    require(ip['uint8_bitwise_equal'] == 192 and (ip['normalized_max_abs'], ip['normalized_mean_abs'],
                                                  ip['normalized_p99_abs']) == (0.0, 0.0, 0.0), 'input parity')
    sb = r4['attempt_2']['surrogate_backward']
    require(sb['max_abs'] == 0.0 and sb['bitwise_equal'] and sb['finite'] and sb['nonzero'] and
            r4['attempt_2']['backward_repeatability_bitwise'] and r4['attempt_2']['teacher_parameter_grads_none'],
            'surrogate backward gate')
    g = res['GPU_RUNTIME']
    require(g['driver'] == '580.178.04' and g['status'] == 'QUALIFIED' and
            g['environment_lock_sha256'] == sha(reader(LOCK)), 'GPU runtime binding')
    for token in ('GPU_RUNTIME_QUALIFIED', 'R04_LEVEL2_TEACHER_PARITY_QUALIFIED', 'PRODUCTION_RUNNER_NOT_IMPLEMENTED',
                  'SCIENTIFIC_TRAINING_NOT_ALLOWED'):
        require(token in rec['m7_status_after'], 'status ' + token)
    return rec


def check_config_status(reader, base):
    old, now = base(CONFIG_STATUS), reader(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in STATUS_PHRASES:
        require(phrase in added, 'CONFIG_STATUS M7C3 ' + phrase)
    return len(now) - len(old)


def check_no_runner(reader):
    for rel in (HARNESS, R04_TOOL):
        src = reader(rel).decode()
        for forbidden in ('torch.save', 'DataLoader', 'for epoch', 'resume', 'save_checkpoint', 'val_pairs', 'test_pairs'):
            require(forbidden not in src, f'{rel}: {forbidden}')


# ----------------------------------------------------------------- candidate-time checks
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7C2b authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS + ledger/index differ: ' + json.dumps(changed))
    require(git('ls-files', *('*' + s for s in WEIGHT_SUFFIXES)).decode().strip() == '', 'no weights tracked')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in (*changed, *NEW):
        require(rel in (TEACHER_PRE, RECORD) or not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7C3 candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF, no trailing ws ' + rel)
    added = read(CONFIG_STATUS)[len(at_authority(CONFIG_STATUS)):]
    require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF ' + CONFIG_STATUS)
    raw = read(TEACHER_PRE)
    require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw, 'LF ' + TEACHER_PRE)


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
        'record_kind': RECORD_KIND, 'exit_code': 0, 'M6_closed': True, 'M7_started': True,
        'M7_static_core_implemented': True, 'M7_cpu_synthetic_qualified': True, 'M7_gpu_runtime_qualified': True,
        'M7_r04_level2_teacher_parity_qualified': True, 'M7_production_runner_implemented': False,
        'M7_real_train_qualification_done': False, 'M7_scientific_training_allowed': False,
        'M7_scientific_training_started': False, 'r04_level2_executed': True, 'remaining_deferred': [],
        'new_deviation_numbers': [], 'amendment_created': False, 'environment_created': True,
        'environment_name': 'gpat-m7-gpu', 'existing_environments_mutated': False, 'GPU_contacted': True,
        'TRAIN_access': 'R04_LEVEL2_SELECTED_FACES_ONLY', 'VAL_access': False, 'TEST_access': False, 'M8_bank': False,
        'training_runs': 0, 'scientific_runs': 0, 'qualification_optimizer_steps': {'G': 1, 'D': 1},
        'checkpoint_writes': 0, 'method_status_modified': False, 'deviation_report_modified': False,
        'stage_state_modified': False, 'config_status_modified': True, 'commit': False, 'push': False,
        'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
        'committed_prefix_sha256': prefix_sha, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
        'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': list(MODIFIED),
        'output_artifacts': sorted((*NEW, *MODIFIED))}


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
            'purpose': ('M7C3: GPAT GPU runtime qualification (gpat-m7-gpu environment, pinned NAFNet on CUDA, deterministic '
                        'CUDA, AMP/fp32 boundaries, batch-4 forward and one synthetic accumulation-group optimizer smoke, '
                        'DEV-022, VRAM, repeatability) and R-04 Level-2 teacher parity on TRAIN-only selected faces; '
                        'no production runner, scientific training, checkpoint or bank'),
            'notes': notes,
            'command': ('GPU host (gpat-m7-gpu): python -B tools/m7c3_gpat_gpu_qualification.py --out ...; --digests x2; '
                        'python -B tools/m7c3_r04_level2_parity.py --out ...; laptop: python3 -B '
                        'tools/m7c3_gpat_gpu_runtime_preflight.py --before-ledger / --append-ledger / --rebuild-index'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    check_protected(read, at_authority)
    check_environment(read)
    check_gpu_evidence(read)
    check_r04(read)
    check_teacher_preprocess(read, at_authority)
    check_record(read)
    check_no_runner(read)
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
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
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
