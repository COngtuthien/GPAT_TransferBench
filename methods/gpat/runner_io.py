"""M7C4 GPAT production-runner I/O: modes, seeds, roots, TRAIN relation, epoch order, layout, identity map, assets.

Imports no deep-learning framework (the static preflight, the CLI refusal path and metadata tests use it).

TRAIN relation: the frozen common pair manifest manifests/pairs_train_v1.parquet (8838 TRAIN rows; A10-bound SHA-256).
It is SHA-256 checked before it is parsed and read through a column allowlist. Its own split column (all TRAIN) and its
bound split_manifest_sha256 prove TRAIN membership, so the training process never opens split_v1, VAL or TEST metadata.
Canonical faces are read by the M6D6e production CanonicalFaceReader (<faces_256_root>/<dataset>/<sample_id>.png,
O_NOFOLLOW, no enumeration) restricted to the relation's ids, behind an access-logging firewall.

Epoch order (OWNER_IMPLEMENTATION_CLARIFICATION, configs/amendments/gpat_m7c4_runner_resolution.yaml; no Python/NumPy
global RNG, independent of workers / PYTHONHASHSEED / filesystem):
    payload = UTF-8 'GPAT-M7|<stage>|<MODE>|<seed>|<epoch>', stage in {generator, warmup}, MODE in {SCIENTIFIC,
              QUALIFICATION} (upper case, the constants below), decimal seed, 1-based decimal epoch
    seed64  = int(SHA256(payload).hexdigest()[:16], 16)          (first 16 hex characters = 64 bits)
    perm    = numpy.random.Generator(PCG64(seed64)).permutation(8838)
Generator microbatches are consecutive chunks of 4 of the permuted rows (2210; tail 2), optimizer groups are
consecutive pairs of microbatches (1105; tail group [4, 2]). Warmup batches are chunks of 64 (139; tail 6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError

SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
EXPERIMENT_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 70404                      # engineering only; never valid for a scientific run
VARIANTS = {'GPAT-B0': ('B0', 'E08'), 'GPAT-B1': ('B1', 'E09'), 'GPAT-B2': ('B2', 'E10'), 'GPAT-B3': ('B3', 'E11')}
# M7D1-A1: scientific runs under the amended curriculum get a fresh tree; the v1.0-curriculum roots under runs/m7/
# (e.g. B0/E08/seed_42 run_id 7b799fbd6d0426da) stay immutable historical evidence and are never resumed.
SCIENTIFIC_PARTS = ('runs', 'm7_a1')
QUALIFICATION_PARTS = ('qualification', 'm7', 'M7C4')
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_SCIENTIFIC', 'NOT_ELIGIBLE_FOR_BANK', 'NOT_ELIGIBLE_FOR_SELECTION',
                        'NOT_ELIGIBLE_FOR_PAPER_RESULT')
RELATION = 'manifests/pairs_train_v1.parquet'
RELATION_SHA256 = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
SPLIT_MANIFEST_SHA256 = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'   # bound inside the relation
RELATION_COLUMNS = ('pair_id', 'dataset', 'source_spoof_id', 'target_live_id', 'attack_macro', 'source_subject',
                    'split', 'split_manifest_sha256')
RELATION_NEVER_READ = ('target_subject', 'd_pose', 'd_scale', 'd_luma', 'seed', 'd_pair', 'source_video_id',
                       'target_video_id', 'source_content_group_id', 'target_content_group_id', 'source_sha256',
                       'target_sha256', 'candidate_count_eligible', 'candidate_count_evaluated', 'pairs_config_sha256',
                       'face_area_fraction_source', 'face_area_fraction_target')
DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')
ROWS_BY_DATASET = {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118}
TRAIN_ROWS = 8838
ATTACK_CLASSES = ('makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay')
SAMPLE_ID = re.compile(r'[0-9a-f]{64}')
PAIR_ID = re.compile(r'PTR\d{6}')
PHYSICAL_BATCH, GRAD_ACCUM = 4, 2
MICROBATCHES_PER_EPOCH = -(-TRAIN_ROWS // PHYSICAL_BATCH)              # 2210
GROUPS_PER_EPOCH = -(-MICROBATCHES_PER_EPOCH // GRAD_ACCUM)           # 1105
GENERATOR_EPOCHS = 60
TOTAL_UPDATES = GROUPS_PER_EPOCH * GENERATOR_EPOCHS                   # 66300
EMA_START_EPOCH = 5                                                    # EMA initialized at the end of epoch 5
CANDIDATE_EPOCHS = tuple(range(10, GENERATOR_EPOCHS + 1))             # 10..60 (51)
WARMUP_BATCH, WARMUP_EPOCHS = 64, 10
WARMUP_STEPS_PER_EPOCH = -(-TRAIN_ROWS // WARMUP_BATCH)               # 139
WARMUP_TOTAL_STEPS = WARMUP_STEPS_PER_EPOCH * WARMUP_EPOCHS           # 1390
LOADER = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2}   # execution only
EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'
ASSET_CONFIG = 'configs/execution/gpat_m7_assets_3090.yaml'
GPU_LOCK = 'environments/gpat_m7_gpu.lock.json'
GPU_LOCK_SHA256 = '24c983ebbb308acacd63f32837532e5f9166114962f524e993f243f2ff746114'
M7C3_RECORD = 'configs/amendments/gpat_m7c3_gpu_runtime_resolution.yaml'
M7C3_RECORD_SHA256 = 'f89a91206ffe1c82a55b600f87fc824c91b1840d4222f2b45346121d6af82717'
TEACHER_SHA256 = {'adaface_weight': '52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8',
                  'facexformer_weight': '327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d',
                  'f_art_resnet18_imagenet1k_v1': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
                  'e_art_resnet18_imagenet1k_v1': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'}
LAUNCH_ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'NVIDIA_TF32_OVERRIDE': '0'}


def require(value, message):
    if not value:
        raise PreparationError('GPAT runner: ' + message)


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------- modes, seeds, roots
def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'seed must be an int')
    if mode == SCIENTIFIC:
        require(seed in EXPERIMENT_SEEDS, f'scientific seed must be one of {EXPERIMENT_SEEDS} (got {seed}; the '
                                          f'qualification seed {QUALIFICATION_SEED} is never scientific)')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED, f'qualification seed must be {QUALIFICATION_SEED}')
    else:
        require(False, 'mode must be SCIENTIFIC or QUALIFICATION')
    return mode, seed


def variant_of(method):
    require(method in VARIANTS, f'unknown GPAT method {method!r} (GPAT-B0..GPAT-B3)')
    return VARIANTS[method]


def run_root(runtime_root, mode, method, seed, label=None):
    validate_mode_seed(mode, seed)
    variant, experiment = variant_of(method)
    root = Path(runtime_root)
    require(root.is_absolute(), 'runtime root must be absolute')
    if mode == SCIENTIFIC:
        require(label is None, 'scientific roots carry no label')
        return root.joinpath(*SCIENTIFIC_PARTS, experiment, f'seed_{seed}')
    require(isinstance(label, str) and re.fullmatch(r'[a-z0-9_]+', label), 'qualification label')
    return root.joinpath(*QUALIFICATION_PARTS, f'{experiment}_{label}_q{seed}')


def run_id(method, seed, config_sha256, commit):
    """run_logging_v1: sha256(method_id|seed|config_sha256|git_commit)[:16]."""
    return sha(f'{method}|{seed}|{config_sha256}|{commit}'.encode('utf-8'))[:16]


# ----------------------------------------------------------------------------- TRAIN relation
def read_relation(path=None, *, pq=None):
    """Allowlisted, SHA-checked rows of the GPAT TRAIN relation (index = manifest row order)."""
    rel = Path(path) if path is not None else ROOT / RELATION
    require(sha256_file(rel) == RELATION_SHA256, 'TRAIN relation SHA-256 mismatch')
    if pq is None:
        import pyarrow.parquet as pq
    table = pq.read_table(rel, columns=list(RELATION_COLUMNS))
    require(table.column_names == list(RELATION_COLUMNS), 'column allowlist')
    rows = table.to_pylist()
    require(len(rows) == TRAIN_ROWS, f'{TRAIN_ROWS} TRAIN rows')
    records = []
    for i, r in enumerate(rows):
        require(r['split'] == 'TRAIN' and r['split_manifest_sha256'] == SPLIT_MANIFEST_SHA256, 'TRAIN-only relation')
        require(r['dataset'] in DATASETS and PAIR_ID.fullmatch(r['pair_id']), 'dataset / pair id')
        require(SAMPLE_ID.fullmatch(r['source_spoof_id']) and SAMPLE_ID.fullmatch(r['target_live_id']), 'sample ids')
        require(r['attack_macro'] in ATTACK_CLASSES, 'source attack_macro in the frozen 6-class order')
        subject = r['source_subject']
        if r['dataset'] == 'siwmv2':
            require(subject is None, 'SiW-Mv2 source_subject is masked (DEV-022)')
        else:
            require(isinstance(subject, str) and subject, 'CASIA/MSU source_subject present')
        records.append({'index': i, 'pair_id': r['pair_id'], 'dataset': r['dataset'],
                        'source_spoof_id': r['source_spoof_id'], 'target_live_id': r['target_live_id'],
                        'attack_macro': r['attack_macro'], 'attack_index': ATTACK_CLASSES.index(r['attack_macro']),
                        'source_subject': subject})
    counts = {d: sum(r['dataset'] == d for r in records) for d in DATASETS}
    require(counts == ROWS_BY_DATASET, f'rows by dataset {counts}')
    require(len({r['pair_id'] for r in records}) == TRAIN_ROWS, 'unique pair ids')
    require(len({r['source_spoof_id'] for r in records}) == TRAIN_ROWS, 'one row per TRAIN source spoof frame')
    return records


def sample_datasets(records):
    """sample_id -> dataset for every source and target id of the relation (the reader's whole universe)."""
    out = {}
    for r in records:
        for key in ('source_spoof_id', 'target_live_id'):
            prev = out.setdefault(r[key], r['dataset'])
            require(prev == r['dataset'], 'sample id bound to two datasets')
    return out


# ----------------------------------------------------------------------------- identity map (B2/B3)
def identity_map(records, *, strict=True):
    from methods.gpat import identity_labels
    recs = [{'dataset': r['dataset'], 'source_subject': r['source_subject']} for r in records]
    m = identity_labels.build_identity_map(recs, RELATION_SHA256, strict=strict)
    return m


def identity_labels_for(records, imap):
    """(labels, valid) per relation row; SiW rows -> (-1, False). No pseudo identity."""
    from methods.gpat import identity_labels
    return identity_labels.labels_for([{'dataset': r['dataset'], 'source_subject': r['source_subject']}
                                       for r in records], imap)


# ----------------------------------------------------------------------------- epoch order and layout
def order_key(stream, mode, seed, epoch) -> int:
    require(stream in ('generator', 'warmup'), 'order stream')
    validate_mode_seed(mode, seed)
    require(type(epoch) is int and epoch >= 1, 'epoch index (1-based)')
    digest = sha(f'GPAT-M7|{stream}|{mode}|{seed}|{epoch}'.encode('utf-8'))
    return int(digest[:16], 16)


def epoch_permutation(stream, mode, seed, epoch, n=TRAIN_ROWS):
    perm = np.random.Generator(np.random.PCG64(order_key(stream, mode, seed, epoch))).permutation(n)
    return [int(i) for i in perm]


def order_sha256(records, perm):
    return sha('\n'.join(records[i]['pair_id'] for i in perm).encode('utf-8'))


def microbatches(perm, size=PHYSICAL_BATCH):
    return [perm[i:i + size] for i in range(0, len(perm), size)]


def groups(perm):
    mbs = microbatches(perm)
    return [mbs[i:i + GRAD_ACCUM] for i in range(0, len(mbs), GRAD_ACCUM)]


def warmup_batches(perm):
    return microbatches(perm, WARMUP_BATCH)


def sample_weights(group):
    """Generic per-sample-mean weights m_j / sum(m) of one optimizer group ([1/2, 1/2]; tail [4/6, 2/6])."""
    n = sum(len(mb) for mb in group)
    return [len(mb) / n for mb in group]


def layout_report(perm):
    gs = groups(perm)
    mbs = microbatches(perm)
    wb = warmup_batches(perm)
    return {'rows': len(perm), 'microbatches': len(mbs), 'optimizer_groups': len(gs),
            'regular_group': [len(mb) for mb in gs[0]], 'tail_group': [len(mb) for mb in gs[-1]],
            'regular_weights': sample_weights(gs[0]), 'tail_weights': sample_weights(gs[-1]),
            'warmup_batches': len(wb), 'warmup_tail': len(wb[-1]),
            'covered_exactly_once': sorted(i for mb in mbs for i in mb) == list(range(len(perm)))}


def update_index(epoch, group):
    """1-based global generator update of 1-based (epoch, group)."""
    require(1 <= epoch <= GENERATOR_EPOCHS and 1 <= group <= GROUPS_PER_EPOCH, 'epoch/group range')
    return (epoch - 1) * GROUPS_PER_EPOCH + group


# ----------------------------------------------------------------------------- execution assets
def faces_root(exec_config=None):
    from methods.difffas import aux_runner_io as aio     # M6D6e containment resolver, reused unchanged
    return aio.faces_root_from_exec_config(exec_config or EXEC_CONFIG)


def load_assets(path=None, *, verify_bytes=True):
    """Teacher / backbone asset paths of the execution host, each SHA-256 verified (never downloaded)."""
    import yaml
    cfg_path = ROOT / (path or ASSET_CONFIG)
    cfg = yaml.safe_load(cfg_path.read_text())
    require(cfg['kind'] == 'EXECUTION_INFRASTRUCTURE' and cfg['scientific_values'] is False and
            cfg['classification'] == 'EXECUTION_ONLY_HOST_BINDING', 'asset config kind')
    for name, entry in cfg['assets'].items():
        require(all(entry.get(k) for k in ('path', 'identity', 'role')), f'asset {name}: path, identity and role')
    out = {'asset_config': str(cfg_path.relative_to(ROOT)), 'asset_config_sha256': sha256_file(cfg_path), 'assets': {}}
    for name, want in TEACHER_SHA256.items():
        entry = cfg['assets'][name]
        require(entry['sha256'] == want, f'asset config records the frozen hash for {name}')
        p = Path(entry['path'])
        require(p.is_absolute(), f'{name} path must be absolute')
        if verify_bytes:
            require(p.is_file() and not p.is_symlink(), f'{name} missing: {p}')
            require(sha256_file(p) == want, f'{name} SHA-256 mismatch')
        out['assets'][name] = {'path': str(p), 'sha256': want, 'verified': bool(verify_bytes)}
    for name in ('facexformer_code', 'adaface_code'):
        out['assets'][name] = {'path': cfg['assets'][name]['path'], 'verified_by': 'aux_models.verify_code'}
    return out


# ----------------------------------------------------------------------------- firewall
class AccessLog:
    """Every image opened by the training process: TRAIN relation ids only; VAL/TEST counters must stay 0."""

    def __init__(self, records):
        self.allowed = sample_datasets(records)
        self.opened = []
        self.counters = {'train_images': 0, 'non_train_images': 0, 'val_images': 0, 'test_images': 0,
                         'val_metadata': 0, 'test_metadata': 0}

    def check(self, sample_id, role):
        if sample_id not in self.allowed:
            self.counters['non_train_images'] += 1
            raise PreparationError(f'GPAT runner firewall: {sample_id} is not a TRAIN relation id')
        self.counters['train_images'] += 1
        self.opened.append((role, sample_id))
        return self.allowed[sample_id]

    def report(self):
        ids = sorted({s for _, s in self.opened})
        return {**self.counters, 'opened_events': len(self.opened), 'unique_ids': len(ids),
                'unique_ids_sha256': sha('\n'.join(ids).encode('utf-8'))}
