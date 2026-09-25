"""M6D5e E06c production-runner I/O: contract, run roots, TRAIN relation, canonical reader, records.

Imports no deep-learning framework (the static preflight and CLI refusal paths use it).
The TRAIN relation is the frozen common pair manifest read through a column allowlist, so
subject columns are never materialized; every row is projected by the unchanged
DSDGAdapter.project_pair. Canonical faces are resolved exactly as the M2 writer stored them
(faces_256_root/<dataset>/<sample_id>.png) and the reader receives a sample_id only.
Qualification (seed 60506) and scientific (42/1337/2026) roots can never coincide.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError
from methods.common.runlog import RunDirectoryError, atomic_write_json
from methods.dsdg.adapter import IDFreePairDataset

CONTRACT_PATH = 'configs/amendments/e06c_m6d5e_production_runner_contract.yaml'
METHOD_ID = 'E06c'
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
SCIENTIFIC_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60506          # M6D5e-r clean requalification; engineering only, never an experiment seed
INITIAL_QUALIFICATION_SEED = 60505  # M6D5e initial run (procedural firewall deviation); root retained, never reused
QUALIFICATION_PARTS = ('qualification', 'm6d5e', 'E06c')
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'
TRAIN_RELATION = 'manifests/pairs_train_v1.parquet'
TRAIN_RELATION_SHA256 = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
TRAIN_ROWS = 8838
ALLOWLIST = ('pair_id', 'source_spoof_id', 'target_live_id', 'dataset', 'split', 'seed', 'source_sha256',
             'target_sha256')
NEVER_READ = ('source_subject', 'target_subject')
FACE_SIZE = 256
SAMPLE_ID = re.compile(r'[0-9a-f]{64}')
EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'
LOADER = {'batch_size': 240, 'shuffle': True, 'num_workers': 8, 'pin_memory': True, 'drop_last': False}
ALL_EPOCHS, SAVE_EPOCH = 200, 10
# train_generator.py:197-211 training-sample visualization block (after the epoch loop, before the checkpoint save).
# It is NOT validation and selects nothing; the pinned argparse name of its cadence is bound here only.
VISUALIZATION_CADENCE_KEY, VISUALIZATION_EVERY = 'test_epoch', 10
VISUALIZATION_DIR = 'diagnostics/visualization'
VISUALIZATION_FILES = ('img_spoof', 'img_live', 'rec_spoof', 'rec_live', 'fake_spoof', 'fake_live')  # lines 206-211
VISUALIZATION_NOISE_SHAPE = (240, 128)   # torch.zeros(args.batch_size, args.hdim).normal_(0, 1), CPU, twice
OFFICIAL_FILES = (('netE_nir', 'netE_spoof_'), ('netE_vis', 'netE_live_'), ('netG', 'netG_'))  # train_generator.py:215-217
SELECTED_MODEL = 'netG'
RESUME_KIND = 'E06C_ENGINEERING_RESUME_STATE'
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_SYNTHETIC_BANK',
                        'NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION', 'NOT_ELIGIBLE_FOR_REPORTING')
MISSING_REASONS = {
    'train_metrics': 'Upstream E06c (train_generator.py) exposes no separate train metric; none is fabricated.',
    'val_losses': 'No VAL pass exists in the pinned trainer or this runner; VAL is never opened.',
    'val_metrics': 'No VAL pass exists in the pinned trainer or this runner; VAL is never opened.'}
LOSS_KEYS = ('loss_rec', 'loss_kl', 'loss_mmd', 'loss_ip', 'loss_pair', 'loss_cls', 'loss_ort')
M6D5E_FILES = ('configs/amendments/e06c_m6d5e_production_runner_contract.yaml', 'methods/dsdg/runner_io.py',
               'methods/dsdg/runner.py', 'methods/dsdg/runner_qualification.py', 'tools/run_e06c.py')


def require(value, message):
    if not value:
        raise PreparationError('E06c runner: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_contract():
    """The M6D5e contract is the JSON subset of YAML; bound input hashes are verified here."""
    contract = json.loads((ROOT / CONTRACT_PATH).read_text())
    require(contract['method_id'] == METHOD_ID and contract['execution_mode'] == EXECUTION_MODE, 'contract identity')
    for rel, digest in contract['bound_inputs_sha256'].items():
        require(sha256_file(ROOT / rel) == digest, 'bound input SHA256 ' + rel)
    return contract


def identities(contract):
    """Every hash a run binds: frozen config, A1, V1/V2 overlays, M6D5e contract, lock, relation, source, LightCNN."""
    b = contract['bound_inputs_sha256']
    return {'config_sha256': b['configs/methods/e06c_dsdg_bin_idfree.yaml'],
            'a1_adaptation_sha256': b['configs/frozen/dsdg_bin_idfree_v1.yaml'],
            'm6d5c_v1_overlay_sha256': b['configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml'],
            'm6d5d_v2_overlay_sha256': b['configs/amendments/e06c_m6d5d_tail_batch_execution_resolution.yaml'],
            'm6d5e_contract_sha256': sha256_file(ROOT / CONTRACT_PATH),
            'logging_contract_sha256': b['configs/run_logging_v1.yaml'],
            'environment_lock_sha256': b['environments/e06c.lock.json'],
            'train_relation_sha256': b[TRAIN_RELATION],
            'source_commit': contract['source_pin']['commit'], 'source_tree': contract['source_pin']['tree'],
            'lightcnn_sha256': contract['lightcnn']['sha256']}


def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'integer seed required')
    if mode == SCIENTIFIC:
        require(seed in SCIENTIFIC_SEEDS, f'scientific seed must be one of {SCIENTIFIC_SEEDS}; got {seed}')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED and seed not in SCIENTIFIC_SEEDS,
                f'qualification seed must be {QUALIFICATION_SEED} ({INITIAL_QUALIFICATION_SEED} is the retained initial run)')
    else:
        raise PreparationError('unknown runner mode ' + repr(mode))


def qualification_run_id(run_id):
    return f'q{QUALIFICATION_SEED}-{run_id}'


def run_root(runtime_root, mode, seed, run_id=None):
    """Scientific: <rt>/runs/m6/E06c/seed_<seed>. Qualification: <rt>/qualification/m6d5e/E06c/q60506-<run_id>."""
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    require(rt.is_absolute(), 'absolute runtime root')
    if mode == SCIENTIFIC:
        return rt / 'runs' / 'm6' / METHOD_ID / f'seed_{seed}'
    require(isinstance(run_id, str) and re.fullmatch(r'[0-9a-f]{16}', run_id), 'qualification run id')
    root = rt.joinpath(*QUALIFICATION_PARTS, qualification_run_id(run_id))
    require(not root.is_relative_to(rt / 'runs'), 'qualification root outside scientific runs')
    return root


def faces_root_from_exec_config(path=None):
    """Faces root through the M2 containment firewall; must equal <runtime_root>/data/processed/faces_256."""
    import yaml
    from gpatbench.preprocess import m2b
    path = ROOT / (path or EXEC_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    resolved = m2b.resolve_roots(cfg)
    runtime_root = Path(resolved['runtime_root'])
    faces = Path(resolved['roots']['faces_256_root'])
    require(faces == runtime_root / 'data' / 'processed' / 'faces_256', 'faces root = <runtime_root>/data/processed/faces_256')
    require(not faces.is_symlink() and faces.resolve() == faces, 'faces root is not a symlink')
    return {'exec_config': str(path.relative_to(ROOT)), 'exec_config_sha256': sha256_file(path),
            'runtime_root': str(runtime_root), 'faces_256_root': str(faces)}


# ----------------------------------------------------------------- TRAIN relation
def read_train_relation(adapter, path=None, *, pq=None):
    """SHA256 before parse; allowlisted columns only; 8838 TRAIN rows projected by the unchanged adapter."""
    path = Path(path) if path is not None else ROOT / TRAIN_RELATION
    require(sha256_file(path) == TRAIN_RELATION_SHA256, 'TRAIN relation SHA256 differs from the frozen value')
    if pq is None:
        import pyarrow.parquet as pq
    schema = pq.read_schema(path).names
    missing = [c for c in ALLOWLIST if c not in schema]
    require(not missing, 'required allowlisted columns absent: ' + ', '.join(missing))
    table = pq.read_table(path, columns=list(ALLOWLIST))
    require(tuple(table.column_names) == ALLOWLIST, 'only allowlisted columns materialized')
    rows = table.to_pylist()
    require(len(rows) == TRAIN_ROWS, f'TRAIN relation must have {TRAIN_ROWS} rows; got {len(rows)}')
    records = tuple(adapter.project_pair(r) for r in rows)   # TRAIN-only, frozen datasets, nonempty IDs
    require(len({r['pair_id'] for r in records}) == TRAIN_ROWS, 'unique pair_id')
    datasets = {}
    for r in records:
        for key in ('source_spoof_id', 'target_live_id'):
            sid = r[key]
            require(SAMPLE_ID.fullmatch(sid) is not None, 'sample_id pattern')
            require(datasets.setdefault(sid, r['dataset']) == r['dataset'], 'sample_id bound to two datasets')
    by_dataset = {}
    for r in records:
        by_dataset[r['dataset']] = by_dataset.get(r['dataset'], 0) + 1
    evidence = {'path': TRAIN_RELATION, 'sha256': TRAIN_RELATION_SHA256, 'sha256_checked_before_parse': True,
                'schema_columns': len(schema), 'columns_read': list(ALLOWLIST),
                'subject_columns_present_in_file': [c for c in NEVER_READ if c in schema],
                'subject_columns_read': [c for c in NEVER_READ if c in table.column_names],
                'rows': len(records), 'splits': sorted({r['split'] for r in records}),
                'rows_by_dataset': dict(sorted(by_dataset.items())),
                'unique_source_ids': len({r['source_spoof_id'] for r in records}),
                'unique_target_ids': len({r['target_live_id'] for r in records}),
                'unique_sample_ids': len(datasets),
                'pair_id_order_sha256': order_sha256(r['pair_id'] for r in records)}
    return records, datasets, evidence


def order_sha256(pair_ids):
    return sha('\n'.join(pair_ids).encode('utf-8'))


# ----------------------------------------------------------------- canonical reader
class CanonicalFaceReader:
    """reader(sample_id) -> canonical uint8 RGB [256,256,3]; no crop/resize/augmentation/normalization.

    Only sample_ids of the TRAIN relation resolve; the dataset comes from the relation's
    allowlisted `dataset` column. Traversal, symlinks and non-canonical bytes fail closed.
    """
    def __init__(self, faces_root, sample_datasets, datasets=('casia_fasd', 'msu_mfsd', 'siwmv2'), decode=None):
        root = Path(faces_root)
        require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'canonical faces root')
        self.root = root.resolve()
        self._datasets = dict(sample_datasets)
        require(set(self._datasets.values()) <= set(datasets), 'dataset outside the frozen set')
        self._decode = decode or _cv2_decode

    def __call__(self, sample_id):
        require(isinstance(sample_id, str) and SAMPLE_ID.fullmatch(sample_id) is not None, 'sample_id pattern')
        dataset = self._datasets.get(sample_id)
        require(dataset is not None, 'sample_id is not in the TRAIN relation')
        directory = self.root / dataset
        path = directory / f'{sample_id}.png'
        require(not directory.is_symlink() and not path.is_symlink(), 'symlink refused')
        require(path.resolve(strict=True).parent == directory and directory.resolve() == directory,
                'path escapes the canonical faces root')
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as fh:
            raw = fh.read()
        rgb = self._decode(raw)
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (FACE_SIZE, FACE_SIZE, 3),
                'canonical uint8 RGB256 required; no implicit recrop/rescale')
        return rgb


def _cv2_decode(raw):
    import cv2
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    require(bgr is not None and bgr.dtype == np.uint8 and bgr.shape == (FACE_SIZE, FACE_SIZE, 3),
            'canonical PNG must decode to uint8 [256,256,3]')
    return np.ascontiguousarray(bgr[:, :, ::-1])   # M2 encoded RGB[:, :, ::-1]


class IndexedPairDataset(IDFreePairDataset):
    """IDFreePairDataset unchanged plus the row index, used only for pair-order accounting."""
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['index'] = np.int64(index)
        return item


class IndexOnlyDataset:
    """Same length as the TRAIN relation; yields row indices only (loader-order checks, no image bytes)."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return np.int64(index)


# ----------------------------------------------------------------- checkpoint policy
def is_checkpoint_epoch(epoch, save_epoch=SAVE_EPOCH):
    """train_generator.py:214: `if epoch % args.save_epoch == 0 or epoch == 1` (the epoch-1 exception kept)."""
    return epoch % save_epoch == 0 or epoch == 1


def checkpoint_epochs(all_epochs=ALL_EPOCHS, save_epoch=SAVE_EPOCH):
    return [e for e in range(1, all_epochs + 1) if is_checkpoint_epoch(e, save_epoch)]


def official_basename(prefix, epoch):
    """misc/util.py::save_checkpoint: name + 'model_epoch_{}_iter_{}.pth'.format(epoch, 0)."""
    return prefix + 'model_epoch_{}_iter_{}.pth'.format(epoch, 0)


def checkpoint_kind(model, epoch, all_epochs=ALL_EPOCHS):
    """(checkpoint_type, selected_for_final): only netG at the final epoch is selected."""
    if epoch == all_epochs:
        return ('selected', True) if model == SELECTED_MODEL else ('terminal', False)
    return 'periodic', False


def resume_state_name(epoch):
    return f'runner_state_epoch_{epoch}.pth'


# ----------------------------------------------------------------- upstream visualization block
def visualization_every(config):
    require(config['training'][VISUALIZATION_CADENCE_KEY] == VISUALIZATION_EVERY, 'visualization cadence = frozen 10')
    return VISUALIZATION_EVERY


def is_visualization_epoch(epoch, every=VISUALIZATION_EVERY):
    """train_generator.py:198: the cadence test `epoch % <cadence> == 0 or epoch == 1` (TRAIN-sample grids only)."""
    return epoch % every == 0 or epoch == 1


def visualization_basename(epoch, name):
    """train_generator.py:206-211: '{}/Epoch_{:03d}_<name>.png'.format(out_path, epoch)."""
    return 'Epoch_{:03d}_{}.png'.format(epoch, name)


# ----------------------------------------------------------------- trajectory records
def finite(values):
    return all(isinstance(v, float) and v == v and abs(v) != float('inf') for v in values)


def step_record(*, epoch, global_step, iteration, learning_rate, losses, total_loss, wall_clock_seconds,
                gpu_memory_bytes, batch_size, chunk_sizes, batch_pair_sha256, epsilon_sha256, mode, seed, extra=None):
    """One trajectory record per GLOBAL optimizer step (run_logging_v1 fields + explicit null reasons)."""
    train_losses = {k: float(losses[k]) for k in LOSS_KEYS}
    train_losses['total_loss'] = float(total_loss)
    require(finite(train_losses.values()), 'non-finite loss')
    record = {'epoch': epoch, 'global_step': global_step, 'iteration_in_epoch': iteration,
              'learning_rate': learning_rate, 'train_losses': train_losses,
              'train_losses_semantics': 'global-batch values (local terms sum m/B over chunks; mmd/ort from pass-1 '
                                        'global statistics); total_loss = pinned train_generator.py:174-178 assembly',
              'train_metrics': None, 'val_losses': None, 'val_metrics': None,
              'wall_clock_seconds': wall_clock_seconds, 'gpu_memory_bytes': gpu_memory_bytes,
              'gpu_memory_semantics': 'torch.cuda.max_memory_allocated() within this global step',
              'record_granularity': 'GLOBAL_OPTIMIZER_STEP', 'execution_mode': EXECUTION_MODE,
              'actual_global_batch_size': batch_size, 'microbatch_sizes': list(chunk_sizes),
              'batch_pair_id_sha256': batch_pair_sha256, 'epsilon_sha256': dict(epsilon_sha256),
              'runner_mode': mode, 'missing_field_reasons': dict(MISSING_REASONS)}
    if gpu_memory_bytes is None:
        record['missing_field_reasons']['gpu_memory_bytes'] = 'CUDA memory not measured: no CUDA device in this process'
    record['experiment_seed' if mode == SCIENTIFIC else 'qualification_seed'] = seed
    if mode == QUALIFICATION:
        record['labels'] = list(QUALIFICATION_LABELS)
    record.update(extra or {})
    return record


def trajectory(metrics_path):
    """Parsed metrics.jsonl (each line must be a JSON object and newline-terminated)."""
    raw = Path(metrics_path).read_text(encoding='utf-8')
    if raw and not raw.endswith('\n'):
        raise RunDirectoryError('metrics.jsonl has a partial final line; explicit owner reconciliation required')
    return [json.loads(line) for line in raw.splitlines()]


def reconcile_metrics(metrics_path, *, completed_epoch, global_step):
    """Metrics must agree exactly with the resume state; disagreement is STOP (never truncate)."""
    rows = [r for r in trajectory(metrics_path) if r.get('record_type') == 'epoch']
    steps = [r['global_step'] for r in rows]
    if steps != list(range(1, global_step + 1)):
        raise RunDirectoryError(f'metrics/resume disagreement: {len(steps)} trajectory records, resume global_step '
                                f'{global_step}; STOP_AND_REPORT (no truncation, no overwrite)')
    if rows and rows[-1]['epoch'] != completed_epoch:
        raise RunDirectoryError('metrics last epoch differs from resume completed_epoch')
    return {'trajectory_records': len(rows), 'last_global_step': steps[-1] if steps else 0,
            'last_epoch': rows[-1]['epoch'] if rows else 0, 'agreement': True}


# ----------------------------------------------------------------- resume-state index
def resume_index_path(run_dir):
    return Path(run_dir) / 'resume_state_index.json'


def init_resume_index(run_dir, *, run_id, mode, seed):
    path = resume_index_path(run_dir)
    if not path.exists():
        atomic_write_json(path, {
            'method_id': METHOD_ID, 'run_id': run_id, 'runner_mode': mode,
            ('experiment_seed' if mode == SCIENTIFIC else 'qualification_seed'): seed,
            'kind': RESUME_KIND, 'scientific_checkpoint': False, 'selection_candidate': False,
            'note': 'Engineering resume sidecars only. They are not official model checkpoints, never replace '
                    'checkpoint_index.json entries and are never eligible for checkpoint selection.',
            'entries': []})
    return path


def record_resume_state(run_dir, *, path, epoch, global_step, resolved_config_sha256):
    index = resume_index_path(run_dir)
    data = json.loads(index.read_text())
    file = Path(run_dir) / path
    data['entries'].append({'path': str(path), 'completed_epoch': epoch, 'global_step': global_step,
                            'file_size_bytes': file.stat().st_size, 'sha256': sha256_file(file),
                            'state_kind': RESUME_KIND, 'scientific_checkpoint': False,
                            'selection_candidate': False, 'resolved_config_sha256': resolved_config_sha256})
    atomic_write_json(index, data)
    return data['entries'][-1]


def resume_entry(run_dir, resume_path):
    """Exact path only: it must be inside <run_dir>/checkpoints and indexed with the same SHA256."""
    run_dir = Path(run_dir).resolve()
    p = Path(resume_path)
    p = (p if p.is_absolute() else run_dir / p).resolve()
    require(p.parent == run_dir / 'checkpoints' and p.is_file() and not Path(resume_path).is_symlink(),
            'resume state must be an explicit file inside <run_dir>/checkpoints')
    rel = str(p.relative_to(run_dir))
    entries = [e for e in json.loads(resume_index_path(run_dir).read_text())['entries'] if e['path'] == rel]
    require(len(entries) == 1, 'resume state must have exactly one resume_state_index.json entry')
    e = entries[0]
    require(p.stat().st_size == e['file_size_bytes'] and sha256_file(p) == e['sha256'], 'resume state bytes/hash')
    return p, e


# ----------------------------------------------------------------- production CLI refusals
REFUSED_OPTIONS = ('--epochs', '--all-epochs', '--all_epochs', '--pre-epoch', '--pre_epoch', '--lr',
                   '--learning-rate', '--batch-size', '--batch_size', '--microbatch', '--micro-batch',
                   '--max-microbatch', '--drop-last', '--drop_last', '--amp', '--fp16', '--bf16', '--half', '--tf32',
                   '--allow-tf32', '--checkpoint-selection', '--select-checkpoint', '--selection-split',
                   '--save-epoch', '--save_epoch', '--workers', '--qualification', '--qualification-seed',
                   '--lambda-mmd', '--lambda-ip', '--lambda-pair', '--lambda-type', '--lambda-ort', '--hdim',
                   '--attack-type', '--activation-checkpointing', '--val', '--test', '--resume-latest')


def refused_arguments(argv):
    """Tuning/override flags the production CLI refuses before parsing (fail closed, nothing executed)."""
    return sorted({a.split('=', 1)[0] for a in argv if a.split('=', 1)[0] in REFUSED_OPTIONS})


class Tee:
    """Mirror a stream into an append-only run log (stdout.log / stderr.log)."""
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, 'a', encoding='utf-8')

    def write(self, text):
        self.stream.write(text)
        self.fh.write(text)
        self.fh.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def tee_run_logs(run_dir):
    out, err = Tee(sys.stdout, Path(run_dir) / 'stdout.log'), Tee(sys.stderr, Path(run_dir) / 'stderr.log')
    sys.stdout, sys.stderr = out, err
    return out, err


def untee(streams):
    out, err = streams
    sys.stdout, sys.stderr = out.stream, err.stream
    out.close()
    err.close()


def m6d5e_file_hashes():
    return {rel: sha256_file(ROOT / rel) for rel in M6D5E_FILES if (ROOT / rel).is_file()}
