"""E05 PCGAN (controlled architecture resolution) production-runner I/O: contract, run roots, TRAIN pairs, records.

Imports no deep-learning framework (the scientific CLI refusal and --preflight-only paths use it).

Binds, unchanged: the frozen E05 config (M6B) + snapshot, A2-07/A2-06, the A5 blur overlay, the M6D4c training-runner
resolution, the M6D4e clean synthetic requalification, the gpat-m6-e05 environment lock and its compatibility-patch
manifest, and the frozen common TRAIN pair manifest (manifests/pairs_train_v1.parquet). Nothing here changes a
scientific value; every value below is either read from those inputs or checked against them.

Pair roles (spec 8.5, M6D4c `pair`): source_spoof_id supplies the spoof-artifact (pattern) representation and
target_live_id the facial-content representation, i.e. x_src = face(source_spoof_id), x_tgt = face(target_live_id),
m = G(z_pat_src, z_con_tgt). The relation is read through a column allowlist (subject columns are never materialized).

Two modes share the engine and nothing else:
  SCIENTIFIC     seeds 42/1337/2026, 4000 benchmark iterations, <runtime_root>/runs/m6/E05/seed_<seed>/, fresh only,
                 launched only by tools/run_e05.py from a clean worktree.
  QUALIFICATION  seed 60901 (never an experiment seed), <runtime_root>/qualification/e05_production/E05/<case>-<run_id>/,
                 launched only by methods/pcgan/production_qualification.py; never scientific.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError

METHOD_ID = 'E05'
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
SCIENTIFIC_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60901          # engineering only; never an experiment seed; unused by any earlier harness
QUALIFICATION_PARTS = ('qualification', 'e05_production', 'E05')
QUALIFICATION_CASES = ('train', 'terminal_probe')
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_SYNTHETIC_BANK',
                        'NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION', 'NOT_ELIGIBLE_FOR_REPORTING')
TERMINAL_PROBE_LABEL = 'QUALIFICATION_TERMINAL_SEMANTICS_PROBE_NOT_A_SCIENTIFIC_COMPLETION'
QUALIFICATION_ITERATIONS = 3        # iteration 1 = the periodic checkpoint; 2-3 prove no extra step around it

CONFIG_PATH = 'configs/methods/e05_pcgan.yaml'
CONFIG_SHA = '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'
SNAPSHOT_PATH = 'frozen_config_snapshot/' + CONFIG_PATH
SPEC_PATH = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
A5_OVERLAY_PATH, A5_OVERLAY_SHA = ('configs/amendments/e05_a5_blur_operator_resolution.yaml',
                                   'af28a48b43828652a7423235aba6be239325e3fbfeeaf8fded9bbb57c11a7ca1')
M6D4C_OVERLAY_PATH, M6D4C_OVERLAY_SHA = ('configs/amendments/e05_training_runner_resolution.yaml',
                                         '8eee350c5642f2de38c92af366b99e80f7b6dab4404433ff88ca71da87b92828')
M6D4E_EVIDENCE_PATH = 'outputs/audit/M6D4E_E05_CLEAN_REQUALIFICATION.json'
LOCK_PATH, LOCK_SHA = 'environments/e05.lock.json', '849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50'
COMPAT_MANIFEST_PATH, COMPAT_MANIFEST_SHA = ('environments/e05.compatibility_patch_manifest.json',
                                             'b5d4d279c3332e6a3e3044e526d0b1304e0084cfc1450b38ce1a368a578052ac')
LOGGING_CONTRACT_PATH = 'configs/run_logging_v1.yaml'
SOURCE_COMMIT, SOURCE_TREE = '6baa180f1184ee79a6b967f9d80ee0e02a979ac7', '25a434f1282a0177d6afdcc68ebfcf40bd805621'
ENVIRONMENT_NAME = 'gpat-m6-e05'
PYTHON_EXECUTABLE = '/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python'
EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'
BUILD_PARTS = ('builds', 'e05_pcgan')

PAIR_MANIFEST = 'manifests/pairs_train_v1.parquet'
PAIR_MANIFEST_SHA = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
PAIR_ROWS = 8838
ALLOWLIST = ('pair_id', 'source_spoof_id', 'target_live_id', 'dataset', 'split')
NEVER_READ = ('source_subject', 'target_subject')
DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')
SAMPLE_ID = re.compile(r'[0-9a-f]{64}')
FACE_SIZE = 256

TOTAL_ITERATIONS, BATCH_SIZE = 4000, 1
LEARNING_RATE, BETAS = 1e-6, (0.9, 0.999)
CHECKPOINT_RULE = 'BASELINE_FINAL_STATE_V1'
PERIODIC_REASON = 'Official cadence; not final selection'     # methods.common.learned.checkpoint_metadata wording
# Pinned predecessor cadence (util/iter_counter.py:12,56-57 + train.py:46-54): save after the step taken at
# steps_so_far % save_freq < batch_size (save_freq 50000 images, batch 1) and once more after training. Within the
# frozen 4000-iteration budget that is benchmark iteration 1 (periodic) and iteration 4000 (terminal, selected).
UPSTREAM_SAVE_FREQ = 50000
PASS_INDEX = 1                     # 4000 < 8838 pairs: every iteration lies in the first (partial) pass
LOADER = {'batch_size': 1, 'shuffle': True, 'num_workers': 0, 'pin_memory': False, 'drop_last': False}
INPUT_RANGE = (-1.0, 1.0)          # pinned data/base_dataset.py:137-141 ToTensor + Normalize((0.5,)*3, (0.5,)*3)
G_TERMS = ('L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat')
D_TERMS = ('L_D_real', 'L_D_rec', 'L_D_mix', 'L_D_patch_real', 'L_D_patch_fake')
TRAIN_LOSS_KEYS = G_TERMS + ('L_G_total',) + D_TERMS + ('L_D_image', 'L_D_patch', 'L_D_total')
GROUP_TENSORS = {'D': 80, 'G': 59}

# exact launch environment of record (M6D4e launch commands); PYTHONHASHSEED is the run seed
LAUNCH_ENVIRONMENT = {
    'CUDA_VISIBLE_DEVICES': '0', 'CUDA_HOME': '/home/student20261/miniconda3/envs/gpat-m6-e05',
    'TORCH_EXTENSIONS_DIR': '/home/student20261/workdir/GPAT_TransferBench_runtime/builds/e05_pcgan/torch_extensions',
    'TORCH_CUDA_ARCH_LIST': '8.6', 'MAX_JOBS': '2', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
    'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

MISSING_REASONS = {
    'train_metrics': 'E05 exposes no train metric beyond its D/G loss terms; none is fabricated.',
    'val_losses': 'No VAL pass exists in this runner (VAL is diagnostics-only and never opened here).',
    'val_metrics': 'No VAL pass exists in this runner (VAL is diagnostics-only and never opened here).'}

REFUSED_OPTIONS = (
    '--iterations', '--total-iterations', '--total_iterations', '--max-iterations', '--steps', '--epochs',
    '--batch-size', '--batch_size', '--lr', '--learning-rate', '--learning_rate', '--beta1', '--beta2', '--betas',
    '--resolution', '--crop-size', '--crop_size', '--load-size', '--image-size', '--lambda', '--loss-weights',
    '--weight-rec', '--weight-recblur', '--weight-adv', '--weight-pat', '--alpha', '--beta', '--amp', '--fp16',
    '--bf16', '--half', '--tf32', '--allow-tf32', '--precision', '--cudnn-benchmark', '--checkpoint-selection',
    '--select-checkpoint', '--selection-split', '--best', '--best-checkpoint', '--save-freq', '--save_freq',
    '--checkpoint-every', '--datasets', '--dataset', '--split', '--splits', '--val', '--test', '--include-siw',
    '--pair-manifest', '--pairs', '--manifest', '--faces-root', '--output', '--output-dir', '--run-root',
    '--run-dir', '--runtime-root', '--qualification', '--qualification-only', '--qualification-seed',
    '--qualification-iterations', '--limit', '--subset', '--max-pairs', '--resume', '--resume-state',
    '--resume-from', '--auto-resume', '--resume-latest', '--continue-train', '--continue_train', '--workers',
    '--num-workers', '--shuffle', '--no-shuffle', '--seed-offset', '--build-root')


def require(value, message):
    if not value:
        raise PreparationError('E05 production: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def order_sha256(items):
    return sha('\n'.join(items).encode('utf-8'))


# ----------------------------------------------------------------- modes / seeds / roots
def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'integer seed required')
    if mode == SCIENTIFIC:
        require(seed in SCIENTIFIC_SEEDS, f'scientific seed must be one of {SCIENTIFIC_SEEDS}; got {seed}')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED and seed not in SCIENTIFIC_SEEDS,
                f'qualification seed must be {QUALIFICATION_SEED}')
    else:
        raise PreparationError('unknown runner mode ' + repr(mode))


def run_root(runtime_root, mode, seed, *, case=None, run_id=None):
    """Scientific: <rt>/runs/m6/E05/seed_<seed>. Qualification: <rt>/qualification/e05_production/E05/<case>-<id>."""
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    require(rt.is_absolute(), 'absolute runtime root')
    if mode == SCIENTIFIC:
        require(case is None and run_id is None, 'scientific roots take no case/run id')
        return rt / 'runs' / 'm6' / METHOD_ID / f'seed_{seed}'
    require(case in QUALIFICATION_CASES, 'qualification case')
    require(isinstance(run_id, str) and re.fullmatch(r'[0-9a-f]{16}', run_id) is not None, 'qualification run id')
    root = rt.joinpath(*QUALIFICATION_PARTS, f'{case}-q{seed}-{run_id}')
    require(not root.is_relative_to(rt / 'runs'), 'qualification root outside scientific runs')
    return root


def build_root(runtime_root):
    return Path(runtime_root).joinpath(*BUILD_PARTS)


# ----------------------------------------------------------------- frozen contract (framework-free)
def verify_contract(config=None):
    """Frozen config + snapshot + spec + A2/A5 + M6D4c + M6D4e gates; every frozen value is checked, none is set."""
    from methods.common.config import FrozenConfigError, load_method_config
    from methods.common.learned import authoritative
    from methods.pcgan.contract import validate_contract
    from methods.pcgan.training_runner import training_contract
    try:
        config = authoritative(config) if config is not None else load_method_config(METHOD_ID)
    except FrozenConfigError as exc:
        raise PreparationError('E05 production: frozen config refused: ' + str(exc)) from exc
    require(config['method_id'] == METHOD_ID and config['_runtime']['config_sha256'] == CONFIG_SHA,
            'frozen E05 config SHA256')
    require((ROOT / CONFIG_PATH).read_bytes() == (ROOT / SNAPSHOT_PATH).read_bytes(), 'config/snapshot byte-identical')
    require(config['spec_sha256'] == SPEC_SHA, 'frozen spec SHA256 named by the config')
    require(sha256_file(ROOT / A5_OVERLAY_PATH) == A5_OVERLAY_SHA, 'A5 blur overlay SHA256')
    contracts = validate_contract(config)                       # A2 + A5 bytes bound to the owner contract commit
    m6d4c = training_contract(ROOT)                             # M6D4c overlay bytes (8eee350c...)
    t, o, a, lo, cp = (config[k] for k in ('training', 'optimizer', 'architecture_at_256', 'losses', 'checkpoint'))
    require((t['total_iterations'], t['batch_size'], t['input_resolution']) == (TOTAL_ITERATIONS, BATCH_SIZE, 256),
            '4000 iterations, batch 1, 256 input')
    require((o['name'], o['learning_rate'], tuple(o['betas'])) == ('Adam', LEARNING_RATE, BETAS), 'Adam 1e-6 (.9,.999)')
    require((a['netE_num_downsampling_sp'], a['spatial_code_ch'], a['z_pat']) == (1, 8, '8 x 128 x 128'),
            'architecture at 256')
    require(lo['total'] == ' + '.join(G_TERMS) and lo['weighting'] == 'ALL_UNIT_WEIGHTED' and
            (lo['alpha'], lo['beta']) == (0.2, 1e-6) and lo['coefficients_retuned_in_m6b'] is False, 'five unit losses')
    require((cp['rule'], cp['terminal_state'], cp['selection_scope'], cp['selection_uses_val'], cp['selection_uses_test'])
            == (CHECKPOINT_RULE, f'iteration_{TOTAL_ITERATIONS}', 'WITHIN_SEED', False, False), 'checkpoint rule')
    require((config['fidelity_class'], config['reporting_label']) ==
            ('CONTROLLED_ADAPTATION', 'PCGAN (controlled architecture resolution)'), 'fidelity / reporting label')
    require(config['attack_type_labels'] == 'FORBIDDEN' and config['gpat_identity_or_landmark_losses_in_main_row']
            is False and config['data']['splits']['TEST']['allowed'] is False and
            config['data']['splits']['VAL']['may_select_checkpoint'] is False, 'supervision / data firewall')
    require((config['data']['pair_manifest'], config['data']['pair_manifest_sha256']) ==
            (PAIR_MANIFEST, PAIR_MANIFEST_SHA), 'frozen pair manifest identity')
    require(config['source']['executable_architecture_basis']['pinned_commit'] == SOURCE_COMMIT, 'source pin')
    require(m6d4c['iteration']['order'] == ['D', 'G'] and m6d4c['iteration']['applications_per_iteration'] ==
            {'D': 1, 'G': 1, 'total': 2} and m6d4c['iteration']['counter']['terminal'] == TOTAL_ITERATIONS and
            m6d4c['checkpoint']['rule'] == CHECKPOINT_RULE and m6d4c['checkpoint']['extra_optimizer_applications'] == 0
            and m6d4c['pair']['batch_size'] == 1 and m6d4c['pair']['distinct_source_target'] is True,
            'M6D4c D->G iteration contract')
    m6d4e = json.loads((ROOT / M6D4E_EVIDENCE_PATH).read_text())
    require('E05_CLEAN_REQUALIFICATION_PASS' in m6d4e['final_status'] and
            m6d4e['environment_lock_sha256'] == LOCK_SHA and m6d4e['overlay_sha256'] == M6D4C_OVERLAY_SHA,
            'M6D4e clean synthetic requalification PASS')
    return config, contracts


def identities(config, contracts):
    lock_raw = (ROOT / LOCK_PATH).read_bytes()
    require(sha(lock_raw) == LOCK_SHA, 'E05 environment lock SHA256')
    lock = json.loads(lock_raw)
    require(sha256_file(ROOT / COMPAT_MANIFEST_PATH) == COMPAT_MANIFEST_SHA == lock['compatibility_patch_manifest_sha256'],
            'compatibility patch manifest SHA256')
    require(lock['source_commit'] == SOURCE_COMMIT and lock['environment_name'] == ENVIRONMENT_NAME and
            lock['identity']['executable'] == PYTHON_EXECUTABLE, 'lock source/environment identity')
    return {'config_sha256': CONFIG_SHA, 'spec_sha256': SPEC_SHA,
            'a2_sha256': contracts['A2']['sha256'], 'a5_amendment_sha256': contracts['A5']['sha256'],
            'a5_overlay_sha256': A5_OVERLAY_SHA, 'm6d4c_overlay_sha256': M6D4C_OVERLAY_SHA,
            'm6d4e_evidence_sha256': sha256_file(ROOT / M6D4E_EVIDENCE_PATH),
            'environment_lock_sha256': LOCK_SHA, 'compatibility_manifest_sha256': COMPAT_MANIFEST_SHA,
            'pair_manifest_sha256': PAIR_MANIFEST_SHA, 'logging_contract_sha256': sha256_file(ROOT / LOGGING_CONTRACT_PATH),
            'source_commit': SOURCE_COMMIT, 'source_tree': SOURCE_TREE, 'fidelity_class': 'CONTROLLED_ADAPTATION',
            'production_files_sha256': production_file_hashes()}


PRODUCTION_FILES = ('methods/pcgan/production_io.py', 'methods/pcgan/production_runner.py', 'tools/run_e05.py')


def production_file_hashes():
    return {rel: sha256_file(ROOT / rel) for rel in PRODUCTION_FILES if (ROOT / rel).is_file()}


def verify_source(adapter):
    """Pinned swapping-autoencoder checkout: commit, tree and required file hashes (no import, no CUDA build)."""
    source = adapter.validate_source()
    require(source['commit'] == SOURCE_COMMIT and source.get('tree', SOURCE_TREE) == SOURCE_TREE, 'source pin/tree')
    return source


def faces_root_from_exec_config(path=None):
    """Faces root through the M2 containment firewall (the E06c production reader of record, unchanged)."""
    from methods.dsdg import runner_io as rio
    require((path or EXEC_CONFIG) == EXEC_CONFIG, 'execution config must be ' + EXEC_CONFIG)
    return rio.faces_root_from_exec_config(EXEC_CONFIG)


def environment_differences(lock_identity, live, seed):
    """Live environment record vs the E05 lock identity (M6D4a/e). Every field must be equal, with two exceptions:
    PYTHONHASHSEED is the run seed, and the host NVIDIA driver *version* may differ (it is host-level, outside the
    gpat-m6-e05 environment and its CUDA 13.0 runtime); the GPU model and UUID must still be identical, and both
    driver versions are returned for disclosure."""
    expected = json.loads(json.dumps(lock_identity))
    expected['launch_environment']['PYTHONHASHSEED'] = str(seed)
    diff = sorted(k for k in set(expected) | set(live) if k != 'gpu_driver' and expected.get(k) != live.get(k))
    old, new = (str(x.get('gpu_driver', '')).split(', ') for x in (expected, live))
    if len(old) != 3 or len(new) != 3 or (old[0], old[2]) != (new[0], new[2]):
        diff.append('gpu_driver(device model/uuid)')
    driver = {'lock_driver_version': old[1] if len(old) == 3 else None,
              'live_driver_version': new[1] if len(new) == 3 else None,
              'gpu_model': new[0] if len(new) == 3 else None, 'gpu_uuid': new[2] if len(new) == 3 else None}
    driver['driver_version_changed'] = driver['lock_driver_version'] != driver['live_driver_version']
    return diff, driver


def launch_environment_errors(environ, seed):
    expected = dict(LAUNCH_ENVIRONMENT, PYTHONHASHSEED=str(seed))
    return sorted(f'{k}={v}' for k, v in expected.items() if environ.get(k) != v)


def refused_arguments(argv):
    """Override/tuning/resume/qualification flags the scientific CLI refuses before parsing (nothing executed)."""
    return sorted({a.split('=', 1)[0] for a in argv if a.split('=', 1)[0] in REFUSED_OPTIONS})


# ----------------------------------------------------------------- TRAIN pair manifest
def read_pair_manifest(path=None, *, expected_sha256=PAIR_MANIFEST_SHA, expected_rows=PAIR_ROWS, pq=None):
    """SHA256 before parse; allowlisted columns only; TRAIN-only; distinct source/target; one dataset per sample id.

    Returns (rows, sample_datasets, evidence). Row order is the stored manifest order (pair_id ascending).
    """
    path = Path(path) if path is not None else ROOT / PAIR_MANIFEST
    require(sha256_file(path) == expected_sha256, 'pair manifest SHA256 differs from the frozen value')
    if pq is None:
        import pyarrow.parquet as pq
    schema = pq.read_schema(path).names
    missing = [c for c in ALLOWLIST if c not in schema]
    require(not missing, 'required allowlisted columns absent: ' + ', '.join(missing))
    table = pq.read_table(path, columns=list(ALLOWLIST))
    require(tuple(table.column_names) == ALLOWLIST, 'only allowlisted columns materialized')
    rows = tuple(table.to_pylist())
    require(len(rows) == expected_rows, f'pair manifest must have {expected_rows} rows; got {len(rows)}')
    datasets = {}
    for r in rows:
        require(r['split'] == 'TRAIN', 'non-TRAIN pair row refused: ' + str(r['pair_id']))
        require(r['dataset'] in DATASETS, 'dataset outside the frozen set: ' + str(r['dataset']))
        require(isinstance(r['pair_id'], str) and r['pair_id'], 'nonempty pair_id')
        require(r['source_spoof_id'] != r['target_live_id'], 'source and target must be distinct samples')
        for key in ('source_spoof_id', 'target_live_id'):
            sid = r[key]
            require(isinstance(sid, str) and SAMPLE_ID.fullmatch(sid) is not None, 'sample_id pattern')
            require(datasets.setdefault(sid, r['dataset']) == r['dataset'], 'sample_id bound to two datasets')
    ids = [r['pair_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'unique pair_id')
    require(ids == sorted(ids), 'stored manifest order is pair_id ascending')
    sources, targets = {r['source_spoof_id'] for r in rows}, {r['target_live_id'] for r in rows}
    require(not sources & targets, 'no sample is both a spoof source and a live target')
    by_dataset = {}
    for r in rows:
        by_dataset[r['dataset']] = by_dataset.get(r['dataset'], 0) + 1
    evidence = {'path': PAIR_MANIFEST, 'sha256': expected_sha256, 'sha256_checked_before_parse': True,
                'schema_columns': len(schema), 'columns_read': list(ALLOWLIST),
                'subject_columns_present_in_file': [c for c in NEVER_READ if c in schema],
                'subject_columns_read': [c for c in NEVER_READ if c in table.column_names],
                'rows': len(rows), 'splits': sorted({r['split'] for r in rows}),
                'rows_by_dataset': dict(sorted(by_dataset.items())), 'unique_source_ids': len(sources),
                'unique_target_ids': len(targets), 'unique_sample_ids': len(datasets),
                'pair_id_order_sha256': order_sha256(ids)}
    return rows, datasets, evidence


def pair_key(row):
    return f"{row['pair_id']}|{row['source_spoof_id']}>{row['target_live_id']}"


def canonical_input(rgb):
    """uint8 RGB [256,256,3] -> float32 CHW in [-1,1]: (x/255 - 0.5)/0.5, the pinned ToTensor + Normalize(0.5, 0.5).

    No resize, crop, flip or other augmentation: the canonical 256 face is the frozen input (spec 8.5 full 256 RGB).
    """
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (FACE_SIZE, FACE_SIZE, 3),
            'canonical uint8 RGB256 required')
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / np.float32(255.0)
    return (x - np.float32(0.5)) / np.float32(0.5)


# ----------------------------------------------------------------- checkpoint cadence / semantics
def needs_saving(iteration):
    """Pinned cadence: the step taken at steps_so_far = iteration - 1 saves when steps_so_far % 50000 < 1; plus the
    unconditional final save. Only benchmark iterations 1..4000 exist."""
    require(type(iteration) is int and 1 <= iteration <= TOTAL_ITERATIONS, 'benchmark iteration 1..4000')
    return (iteration - 1) % UPSTREAM_SAVE_FREQ < BATCH_SIZE or iteration == TOTAL_ITERATIONS


def checkpoint_iterations():
    return [t for t in range(1, TOTAL_ITERATIONS + 1) if needs_saving(t)]


def checkpoint_kind(iteration):
    """(checkpoint_type, selected_for_final, selection_reason): only the iteration-4000 state is selected."""
    require(needs_saving(iteration), 'iteration is not on the checkpoint cadence')
    if iteration == TOTAL_ITERATIONS:
        return 'selected', True, CHECKPOINT_RULE
    return 'periodic', False, PERIODIC_REASON


def checkpoint_basename(iteration):
    return f'iteration_{iteration:04d}.pth'


# ----------------------------------------------------------------- trajectory records
def finite(values):
    return all(isinstance(v, float) and v == v and abs(v) != float('inf') for v in values)


def step_record(*, iteration, learning_rates, losses, wall_clock_seconds, step_seconds, gpu_memory_bytes, pair,
                sampler_index, applications, gradients, mode, seed, input_range):
    """One trajectory record per complete D -> G benchmark iteration (run_logging_v1 fields + explicit null reasons)."""
    train_losses = {k: float(losses[k]) for k in TRAIN_LOSS_KEYS}
    require(finite(train_losses.values()), 'non-finite loss')
    require(set(learning_rates) == {'D', 'G'} and all(v == LEARNING_RATE for v in learning_rates.values()),
            'learning rate = frozen 1e-6 for both optimizers')
    record = {'epoch': PASS_INDEX, 'global_step': iteration, 'benchmark_iteration': iteration,
              'learning_rate': LEARNING_RATE, 'learning_rates': dict(learning_rates),
              'train_losses': train_losses,
              'train_losses_semantics': ('D terms from the D step on the pre-D E/G paths (fakes detached); G terms '
                                         'from the fresh post-D E/G paths; L_D_total = L_D_image + L_D_patch, '
                                         'L_G_total = unit sum of the five G terms (M6D4c)'),
              'train_metrics': None, 'val_losses': None, 'val_metrics': None,
              'wall_clock_seconds': wall_clock_seconds, 'step_seconds': step_seconds,
              'gpu_memory_bytes': gpu_memory_bytes,
              'gpu_memory_semantics': 'torch.cuda.max_memory_allocated() within this benchmark iteration',
              'epoch_semantics': 'pass index over the 8838-row pair manifest; 4000 < 8838 so always pass 1',
              'record_granularity': 'BENCHMARK_ITERATION_D_THEN_G', 'batch_size': BATCH_SIZE,
              'pair_id': pair['pair_id'], 'source_spoof_id': pair['source_spoof_id'],
              'target_live_id': pair['target_live_id'], 'dataset': pair['dataset'],
              'pair_sha256': sha(pair_key(pair).encode('utf-8')), 'sampler_index': sampler_index,
              'input_range': list(input_range), 'optimizer_applications': dict(applications),
              'gradients': gradients, 'runner_mode': mode, 'method_id': METHOD_ID,
              'missing_field_reasons': dict(MISSING_REASONS)}
    if gpu_memory_bytes is None:
        record['missing_field_reasons']['gpu_memory_bytes'] = 'CUDA memory not measured: no CUDA device in this process'
    record['experiment_seed' if mode == SCIENTIFIC else 'qualification_seed'] = seed
    if mode == QUALIFICATION:
        record['labels'] = list(QUALIFICATION_LABELS)
    return record


def trajectory(metrics_path):
    raw = Path(metrics_path).read_text(encoding='utf-8')
    require(not raw or raw.endswith('\n'), 'metrics.jsonl has a partial final line')
    return [json.loads(line) for line in raw.splitlines()]


def resolved_overlays():
    """The execution choices this runner adds on top of the frozen YAML (recorded in resolved_config._resolved)."""
    return {
        'classification': 'PRODUCTION_INTEGRATION_OF_THE_QUALIFIED_M6D4C_RUNNER',
        'pair_roles': 'x_src = face(source_spoof_id) [pattern/artifact], x_tgt = face(target_live_id) [content]; '
                      'spec 8.5 + M6D4c pair (m = G(z_pat_src, z_con_tgt))',
        'pair_order': ('torch DataLoader(batch_size=1, shuffle=True, num_workers=0, drop_last=False) on the stored '
                       'manifest order with generator = torch.Generator().manual_seed(seed) '
                       '(methods.common.learned.torch_loader_options); iteration t consumes sampler position t-1 of '
                       'the first pass: one int64 base-seed draw, then randperm(8838); 4000 distinct pairs, none '
                       'repeated; shuffle provenance: pinned data/__init__.py:93 (train phase shuffles)'),
        'input_mapping': 'canonical uint8 RGB -> float32 (x/255-0.5)/0.5 in [-1,1]; pinned data/base_dataset.py:137-141; '
                         'no resize/crop/flip (canonical faces are already 256; no augmentation)',
        'workers': 'num_workers = 0 (workers_policy IMPLEMENTATION_ADAPTABLE_NO_SCIENTIFIC_IMPACT; the dataset has '
                   'no randomness, so no worker RNG exists)',
        'checkpoint_cadence': ('pinned predecessor cadence util/iter_counter.py save_freq=50000 (+ final save): '
                               'benchmark iteration 1 periodic, iteration 4000 terminal/selected'),
        'checkpoint_content': 'state_dicts of Encoder, Generator, ImageD, PatchD + JSON provenance (no optimizer '
                              'state: resume is not qualified)',
        'resume': 'NOT_QUALIFIED_FRESH_ONLY'}
