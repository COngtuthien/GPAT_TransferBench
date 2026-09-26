"""M6D6e E07c auxiliary-encoder runner I/O: contract, run roots, TRAIN population, canonical reader, records.

Imports no deep-learning framework (the static preflight and the CLI refusal path use it).
The auxiliary population is the frozen TRAIN split of manifests/split_v1.parquet restricted to
the frozen K7 taxonomy of manifests/artifact_probe_classes_v1.json (A3 5.3): 14467 rows,
live included. It is NOT manifests/difffas_bin_idfree_train_v1.parquet (the 8838-row MAIN
DiffFAS spoof relation), which is never opened here. Both inputs are SHA256-checked before
they are parsed; the split manifest is read through a column allowlist with a TRAIN filter, so
subject/video/raw-attack columns and VAL/TEST rows never reach Python. Canonical faces are
resolved exactly as the M2 writer stored them (faces_256_root/<dataset>/<sample_id>.png) and
the reader accepts a TRAIN sample_id only.
Qualification (seed 60605) and scientific (auxiliary seed 42) roots can never coincide.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError

CONTRACT_PATH = 'configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml'
METHOD_ID = 'E07c'
RUN_LABEL = 'E07c/aux_encoder'        # run_logging_v1 run-id method label; never collides with main E07c
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
SCIENTIFIC_SEED = 42                   # A3 5.4b auxiliary_encoder_training_seed (not an experiment-seed run)
QUALIFICATION_SEED = 60605             # engineering only; never 42/1337/2026
EXPERIMENT_SEEDS = (42, 1337, 2026)
SCIENTIFIC_PARTS = ('runs', 'm6', 'E07c', 'aux_encoder')
QUALIFICATION_PARTS = ('qualification', 'm6d6e', 'E07c_aux')
SPLIT_MANIFEST = 'manifests/split_v1.parquet'
SPLIT_MANIFEST_SHA256 = 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d'
CLASS_MAP = 'manifests/artifact_probe_classes_v1.json'
CLASS_MAP_SHA256 = 'c7d23e3e3e6a526ae412a50125d0d56379ce98bca99d4133ead3fda7139a3812'
MAIN_RELATION = 'manifests/difffas_bin_idfree_train_v1.parquet'   # 8838 spoof rows: NOT the auxiliary population
EXEC_CONFIG = 'configs/execution/m5_gpu_3090.yaml'
ALLOWLIST = ('sample_id', 'dataset', 'attack_macro', 'split', 'm2_status')
NEVER_READ = ('subject_id_global', 'video_id', 'frame_index', 'label_binary', 'attack_raw', 'sha256', 'sha256_kind',
              'content_group_id', 'allocation_group_id')
TRAIN_FILTER = [('split', '==', 'TRAIN')]
CLASSES = ('live', 'makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay')
CLASS_COUNTS = {'live': 5629, 'makeup': 759, 'mask_2d': 96, 'mask_3d': 1056, 'partial': 1911, 'print': 2838,
                'replay': 2178}
EXCLUDED = ('other_spoof',)
DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')
TRAIN_ROWS = 14467
K = 7
FACE_SIZE = 256
SAMPLE_ID = re.compile(r'[0-9a-f]{64}')
# pinned models/pretrain_classifier.py:26-29 (A3 5.4)
BATCH_SIZE, SHUFFLE, WORKERS, DROP_LAST = 256, True, 6, True
LR, MOMENTUM, WEIGHT_DECAY = 0.002, 0.9, 5e-3
EPOCHS = 200
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE              # 56
CONSUMED_PER_EPOCH = STEPS_PER_EPOCH * BATCH_SIZE       # 14336
DROPPED_PER_EPOCH = TRAIN_ROWS - CONSUMED_PER_EPOCH     # 131
EPOCH_LOSS_DENOMINATOR = TRAIN_ROWS                     # len(train_dataset), pretrain_classifier.py:43
CHECKPOINT_NAME = 'encoder_final.pkl'
CHECKPOINT_RULE = 'FINAL_STATE_AFTER_EPOCH_200'
LAUNCH_ENVIRONMENT = {'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_A_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_BANK',
                        'NOT_ELIGIBLE_FOR_DOWNSTREAM', 'NOT_ELIGIBLE_FOR_REPORTING')
MISSING_REASONS = {
    'train_metrics': 'Pinned pretrain_classifier.py computes no train metric (loss only); none is fabricated.',
    'val_losses': 'NOT_USED: the pinned auxiliary trainer has no VAL pass; VAL is never opened (A3: no VAL selection).',
    'val_metrics': 'NOT_USED: the pinned auxiliary trainer has no VAL pass; VAL is never opened (A3: no VAL selection).'}
M6D6E_FILES = (CONTRACT_PATH, 'methods/difffas/aux_runner_io.py', 'methods/difffas/aux_runner.py',
               'methods/difffas/aux_runner_qualification.py', 'tools/run_e07c_aux.py')


def require(value, message):
    if not value:
        raise PreparationError('E07c aux runner: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def order_sha256(sample_ids):
    return sha('\n'.join(sample_ids).encode('utf-8'))


def load_contract():
    """The M6D6e contract is the JSON subset of YAML; every bound input SHA256 is verified here."""
    contract = json.loads((ROOT / CONTRACT_PATH).read_text())
    require((contract['method_id'], contract['role'], contract['milestone']) ==
            (METHOD_ID, 'AUXILIARY_CONDITIONING_ENCODER', 'M6D6e'), 'contract identity')
    require(contract['fidelity_class'] == 'CONTROLLED_ADAPTATION' and contract['deviation'] == 'DEV-021' and
            not contract['new_deviation'] and not contract['owner_decision_made'], 'contract fidelity')
    for rel, digest in contract['bound_inputs_sha256'].items():
        require(sha256_file(ROOT / rel) == digest, 'bound input SHA256 ' + rel)
    require(contract['bound_inputs_sha256'][SPLIT_MANIFEST] == SPLIT_MANIFEST_SHA256 and
            contract['bound_inputs_sha256'][CLASS_MAP] == CLASS_MAP_SHA256 and
            MAIN_RELATION not in contract['bound_inputs_sha256'], 'population inputs')
    return contract


def identities(contract):
    b = contract['bound_inputs_sha256']
    return {'config_sha256': b['configs/methods/e07c_difffas_bin_idfree.yaml'],
            'a1_adaptation_sha256': b['configs/frozen/difffas_bin_idfree_v1.yaml'],
            'a3_sha256': b['docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md'],
            'a6_overlay_sha256': b['configs/amendments/e07c_a6_feature_interface_source_correction.yaml'],
            'a7_overlay_sha256': b['configs/amendments/e07c_a7_execution_policy.yaml'],
            'm6d6e_contract_sha256': sha256_file(ROOT / CONTRACT_PATH),
            'logging_contract_sha256': b['configs/run_logging_v1.yaml'],
            'environment_lock_sha256': b['environments/e07c.lock.json'],
            'split_manifest_sha256': b[SPLIT_MANIFEST], 'class_map_sha256': b[CLASS_MAP],
            'source_commit': contract['source_pin']['commit'], 'source_tree': contract['source_pin']['tree']}


# ----------------------------------------------------------------- modes, seeds, roots
def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'integer seed required')
    if mode == SCIENTIFIC:
        require(seed == SCIENTIFIC_SEED, f'the auxiliary encoder has exactly one scientific seed ({SCIENTIFIC_SEED}); got {seed}')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED and seed not in EXPERIMENT_SEEDS, f'qualification seed must be {QUALIFICATION_SEED}')
    else:
        raise PreparationError('unknown runner mode ' + repr(mode))


def run_id(seed, config_sha256, commit):
    """run_logging_v1 formula with the method label E07c/aux_encoder."""
    return sha(f'{RUN_LABEL}|{seed}|{config_sha256}|{commit}'.encode('utf-8'))[:16]


def run_root(runtime_root, mode, seed, rid=None):
    """Scientific: <rt>/runs/m6/E07c/aux_encoder/seed_42. Qualification: <rt>/qualification/m6d6e/E07c_aux/q60605-<run_id>."""
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    require(rt.is_absolute() and not rt.resolve().is_relative_to(ROOT), 'absolute runtime root outside the repository')
    if mode == SCIENTIFIC:
        return rt.joinpath(*SCIENTIFIC_PARTS, f'seed_{seed}')
    require(isinstance(rid, str) and re.fullmatch(r'[0-9a-f]{16}', rid) is not None, 'qualification run id')
    root = rt.joinpath(*QUALIFICATION_PARTS, f'q{seed}-{rid}')
    require(not root.is_relative_to(rt / 'runs'), 'qualification root outside scientific runs')
    return root


def checkpoint_path(run_dir):
    return Path(run_dir) / 'checkpoints' / CHECKPOINT_NAME


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


# ----------------------------------------------------------------- TRAIN population
def read_class_map(path=None):
    """SHA256 before parse; the frozen K7 order must equal A3 / the method config."""
    raw = Path(path if path is not None else ROOT / CLASS_MAP).read_bytes()
    require(sha(raw) == CLASS_MAP_SHA256, 'class map SHA256 differs from the frozen value')
    cmap = json.loads(raw)
    require(tuple(cmap['classes']) == CLASSES and cmap['num_classes'] == K and
            cmap['class_index'] == {c: i for i, c in enumerate(CLASSES)} and tuple(cmap['excluded']) == EXCLUDED and
            cmap['split_manifest_sha256'] == SPLIT_MANIFEST_SHA256, 'frozen K7 class map')
    require(list(CLASSES) == sorted(CLASSES), 'K7 order is alphabetical (ImageFolder class_to_idx equivalence)')
    return cmap


def read_train_population(path=None, class_map_path=None, *, pq=None):
    """14467 TRAIN rows in ImageFolder-equivalent order; VAL/TEST rows never reach Python.

    Returns (records, sample_datasets, evidence). records[i] = {index, sample_id, dataset,
    attack_macro, label, split}. The filter is applied inside pyarrow; the returned table is
    required to hold TRAIN rows only.
    """
    cmap = read_class_map(class_map_path)
    path = Path(path) if path is not None else ROOT / SPLIT_MANIFEST
    require(sha256_file(path) == SPLIT_MANIFEST_SHA256, 'split manifest SHA256 differs from the frozen value')
    if pq is None:
        import pyarrow.parquet as pq
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path).names
    missing = [c for c in ALLOWLIST if c not in schema]
    require(not missing, 'required allowlisted columns absent: ' + ', '.join(missing))
    table = pq.read_table(path, columns=list(ALLOWLIST), filters=TRAIN_FILTER)
    require(tuple(table.column_names) == ALLOWLIST, 'only allowlisted columns materialized')
    rows = table.to_pylist()
    require({r['split'] for r in rows} == {'TRAIN'}, 'only TRAIN rows materialized')
    require(len(rows) == TRAIN_ROWS, f'TRAIN population must have {TRAIN_ROWS} rows; got {len(rows)}')
    index = cmap['class_index']
    for r in rows:
        require(SAMPLE_ID.fullmatch(r['sample_id'] or '') is not None, 'sample_id pattern')
        require(r['dataset'] in DATASETS, 'dataset outside the frozen set')
        require(r['m2_status'] == 'COMPLETE', 'M2 COMPLETE canonical face required')
        require(r['attack_macro'] in index, f"attack_macro {r['attack_macro']!r} outside the frozen K7 taxonomy")
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == TRAIN_ROWS, 'unique sample_id')
    counts = {c: 0 for c in CLASSES}
    by_dataset = {d: 0 for d in DATASETS}
    for r in rows:
        counts[r['attack_macro']] += 1
        by_dataset[r['dataset']] += 1
    require(counts == CLASS_COUNTS, 'frozen K7 TRAIN class counts: ' + json.dumps(counts))
    rows.sort(key=lambda r: (index[r['attack_macro']], r['sample_id']))    # ImageFolder-equivalent order
    records = tuple({'index': i, 'sample_id': r['sample_id'], 'dataset': r['dataset'],
                     'attack_macro': r['attack_macro'], 'label': index[r['attack_macro']], 'split': 'TRAIN'}
                    for i, r in enumerate(rows))
    datasets = {r['sample_id']: r['dataset'] for r in records}
    evidence = {'path': SPLIT_MANIFEST, 'sha256': SPLIT_MANIFEST_SHA256, 'sha256_checked_before_parse': True,
                'class_map': CLASS_MAP, 'class_map_sha256': CLASS_MAP_SHA256,
                'file_rows_total_from_footer': meta.num_rows, 'row_groups': meta.num_row_groups,
                'schema_columns': len(schema), 'columns_read': list(ALLOWLIST),
                'columns_never_read': [c for c in NEVER_READ if c in schema],
                'filter': 'split == TRAIN (pyarrow, native)', 'physical_read_disclosure': (
                    'pyarrow read the footer and decoded the allowlisted column chunks of the single row group '
                    '(all splits) natively, then filtered; only TRAIN rows were materialized in Python'),
                'train_rows_materialized': len(records), 'val_rows_materialized': 0, 'test_rows_materialized': 0,
                'class_counts': counts, 'other_spoof_rows': 0, 'rows_by_dataset': by_dataset,
                'order': 'IMAGEFOLDER_EQUIVALENT (class index, sample_id)',
                'population_order_sha256': order_sha256(r['sample_id'] for r in records),
                'labels_sha256': sha(bytes(r['label'] for r in records)),
                'main_relation_opened': False}
    return records, datasets, evidence


# ----------------------------------------------------------------- canonical reader / dataset
class CanonicalFaceReader:
    """reader(sample_id) -> PIL RGB 256x256 exactly as ImageFolder's pil_loader would return it.

    Only TRAIN-population sample_ids resolve; the dataset comes from the population row.
    Traversal, symlinks and non-canonical images fail closed; nothing is enumerated.
    """
    def __init__(self, faces_root, sample_datasets):
        root = Path(faces_root)
        require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'canonical faces root')
        self.root = root.resolve()
        self._datasets = dict(sample_datasets)
        require(set(self._datasets.values()) <= set(DATASETS), 'dataset outside the frozen set')

    def path(self, sample_id):
        require(isinstance(sample_id, str) and SAMPLE_ID.fullmatch(sample_id) is not None, 'sample_id pattern')
        dataset = self._datasets.get(sample_id)
        require(dataset is not None, 'sample_id is not in the TRAIN population')
        return self.root / dataset / f'{sample_id}.png'

    def raw(self, sample_id):
        path = self.path(sample_id)
        directory = path.parent
        require(not directory.is_symlink() and not path.is_symlink(), 'symlink refused')
        require(directory.resolve() == directory and path.resolve(strict=True).parent == directory,
                'path escapes the canonical faces root')
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as fh:
            return fh.read()

    def __call__(self, sample_id):
        from PIL import Image
        img = Image.open(io.BytesIO(self.raw(sample_id)))
        require(img.format == 'PNG' and img.mode == 'RGB' and img.size == (FACE_SIZE, FACE_SIZE),
                'canonical RGB 256x256 PNG required; no implicit conversion/recrop')
        return img.convert('RGB')      # torchvision pil_loader; a no-op copy for an RGB image


class AuxTrainDataset:
    """ImageFolder-equivalent map-style dataset over the TRAIN population: (tensor, label, index).

    The index travels only for sample accounting; the training step consumes data[0] and data[1]
    exactly as pretrain_classifier.py:34.
    """
    def __init__(self, records, reader, transform):
        require(len(records) == TRAIN_ROWS and all(r['split'] == 'TRAIN' for r in records), 'TRAIN-only records')
        require([r['index'] for r in records] == list(range(len(records))), 'record index order')
        self.records, self.reader, self.transform = tuple(records), reader, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return self.transform(self.reader(r['sample_id'])), r['label'], index


# ----------------------------------------------------------------- trajectory records
def finite(values):
    return all(isinstance(v, float) and v == v and abs(v) != float('inf') for v in values)


def step_record(*, mode, seed, epoch, global_step, iteration, learning_rate, loss, running_loss, batch_size,
                batch_sample_sha256, batch_class_histogram, wall_clock_seconds, gpu_memory_bytes, extra=None):
    """One trajectory record per optimizer step (run_logging_v1 fields + explicit null reasons)."""
    require(finite([float(loss), float(running_loss)]), 'non-finite loss')
    record = {'epoch': epoch, 'global_step': global_step, 'iteration_in_epoch': iteration,
              'learning_rate': learning_rate, 'batch_size': batch_size,
              'train_losses': {'cross_entropy': float(loss)},
              'train_losses_semantics': 'CrossEntropyLoss on the fourth forward output (pretrain_classifier.py:37-38)',
              'running_loss': float(running_loss),
              'running_loss_semantics': 'sum over this epoch of loss.item() * inputs.size(0) (pretrain_classifier.py:42)',
              'train_metrics': None, 'val_losses': None, 'val_metrics': None,
              'wall_clock_seconds': wall_clock_seconds, 'gpu_memory_bytes': gpu_memory_bytes,
              'gpu_memory_semantics': 'torch.cuda.max_memory_allocated() within this optimizer step',
              'batch_sample_id_sha256': batch_sample_sha256, 'batch_class_histogram': list(batch_class_histogram),
              'record_granularity': 'OPTIMIZER_STEP', 'runner_mode': mode,
              'missing_field_reasons': dict(MISSING_REASONS)}
    if gpu_memory_bytes is None:
        record['missing_field_reasons']['gpu_memory_bytes'] = 'CUDA memory not measured: no CUDA device in this process'
    record['auxiliary_encoder_training_seed' if mode == SCIENTIFIC else 'qualification_seed'] = seed
    if mode == QUALIFICATION:
        record['labels'] = list(QUALIFICATION_LABELS)
    record.update(extra or {})
    return record


def epoch_loss(batch_losses, batch_size=BATCH_SIZE):
    """Source-native epoch loss: running_loss / len(train_dataset) (14467, NOT the 14336 consumed)."""
    running = 0.0
    for value in batch_losses:
        running += value * batch_size
    return {'running_loss': running, 'denominator': EPOCH_LOSS_DENOMINATOR,
            'source_epoch_loss': running / EPOCH_LOSS_DENOMINATOR,
            'consumed_examples': len(batch_losses) * batch_size,
            'dropped_examples': EPOCH_LOSS_DENOMINATOR - len(batch_losses) * batch_size,
            'denominator_source': 'len(train_dataset), pretrain_classifier.py:43 (drop_last tail included)'}


# ----------------------------------------------------------------- production CLI refusals
REFUSED_OPTIONS = ('--epochs', '--num-epochs', '--num_epochs', '--max-epochs', '--batch-size', '--batch_size', '--bs',
                   '--workers', '--num-workers', '--num_workers', '--drop-last', '--drop_last', '--no-drop-last',
                   '--shuffle', '--no-shuffle', '--lr', '--learning-rate', '--momentum', '--weight-decay',
                   '--weight_decay', '--wd', '--scheduler', '--lr-scheduler', '--warmup', '--amp', '--fp16', '--bf16',
                   '--half', '--mixed-precision', '--tf32', '--allow-tf32', '--microbatch', '--micro-batch',
                   '--accumulate', '--grad-accum', '--gradient-accumulation', '--activation-checkpointing',
                   '--checkpointing', '--classes', '--num-classes', '--label-map', '--k', '--train-manifest',
                   '--manifest', '--split', '--data-root', '--faces-root', '--val', '--test', '--select-checkpoint',
                   '--checkpoint-selection', '--best', '--resume', '--resume-state', '--pretrained',
                   '--qualification', '--qualification-seed', '--balance', '--weighted-sampler', '--augment')


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


def m6d6e_file_hashes():
    return {rel: sha256_file(ROOT / rel) for rel in M6D6E_FILES if (ROOT / rel).is_file()}
