"""M6D6j E07c MAIN DiffFAS production-runner I/O: contract, modes/seeds/roots, A1 relation, TRAIN membership, dataset.

Imports no deep-learning framework (the static preflight and the CLI refusal path use it).

The MAIN training relation is the frozen A1 manifest manifests/difffas_bin_idfree_train_v1.parquet (8838 TRAIN spoof GT
rows). It is SHA256-checked before it is parsed and read through a column allowlist; identity and attack-taxonomy
columns do not exist in it and are never requested. Every GT / guide / content id must be a member of the frozen TRAIN
split of manifests/split_v1.parquet (read with a TRAIN filter, allowlisted columns only), with the relation's dataset.
Canonical faces are read by the M6D6e production CanonicalFaceReader (<faces_256_root>/<dataset>/<sample_id>.png,
O_NOFOLLOW, no enumeration, TRAIN ids only). The guide is the frozen guide_spoof_id: nothing is resampled and no Python
RNG is consumed (A1 replaces FAS_dataset.py's random.choice).
"""
import hashlib
import json
from pathlib import Path
import re

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError
from methods.difffas import aux_runner_io as aio   # M6D6e production reader / exec-config resolver / Tee (unchanged)

CONTRACT_PATH = 'configs/amendments/e07c_m6d6j_main_production_runner_contract.yaml'
METHOD_ID = 'E07c'
SCIENTIFIC, QUALIFICATION = 'SCIENTIFIC', 'QUALIFICATION'
EXPERIMENT_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60608            # engineering only; never 42/1337/2026 and never the auxiliary seed
AUXILIARY_SEED = 42
QUALIFICATION_RUN_LABEL = 'E07c/main_qualification'
SCIENTIFIC_PARTS = ('runs', 'm6', 'E07c')
QUALIFICATION_PARTS = ('qualification', 'm6d6j', 'E07c')
RELATION = 'manifests/difffas_bin_idfree_train_v1.parquet'
RELATION_SHA256 = '0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6'
SPLIT_MANIFEST = aio.SPLIT_MANIFEST
SPLIT_MANIFEST_SHA256 = aio.SPLIT_MANIFEST_SHA256
EXEC_CONFIG = aio.EXEC_CONFIG
RELATION_COLUMNS = ('track_pair_id', 'dataset', 'gt_spoof_id', 'guide_spoof_id', 'content_live_id', 'style_id',
                    'split', 'use_pair', 'selection_policy', 'gt_sha256', 'guide_sha256')
RELATION_NEVER_READ = ('seed', 'eligible_guide_count', 'self_guide', 'content_training_role', 'split_manifest_sha256',
                       'common_pair_manifest_sha256', 'adaptation_config_sha256', 'official_source_commit')
FORBIDDEN_COLUMNS = ('subject_id_global', 'attack_raw', 'attack_macro', 'label_binary', 'video_id')
SPLIT_COLUMNS = ('sample_id', 'dataset', 'split', 'm2_status')
TRAIN_FILTER = [('split', '==', 'TRAIN')]
DATASETS = aio.DATASETS
ROWS_BY_DATASET = {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118}
TRAIN_ROWS = 8838
STYLE_ID = 'SPOOF_BINARY'
SELECTION_POLICY = 'DETERMINISTIC_RAW_BYTE_SHA256_RANKING'
SAMPLE_ID = aio.SAMPLE_ID
TRACK_ID = re.compile(r'DFA\d{6}')
# pinned FAS_train.py:205 + argparse defaults (verified from the AST by main_runner.verify_production_source)
BATCH_SIZE, SHUFFLE, WORKERS, DROP_LAST = 4, True, 0, False
ITERATIONS_PER_EPOCH = -(-TRAIN_ROWS // BATCH_SIZE)          # 2210 (drop_last=False keeps the tail)
TAIL_BATCH = TRAIN_ROWS - (ITERATIONS_PER_EPOCH - 1) * BATCH_SIZE   # 2
EPOCHS = 400
TOTAL_ITERATIONS = ITERATIONS_PER_EPOCH * EPOCHS             # 884000
SAVE_EVERY, VISUALIZE_EVERY, PRINT_EVERY = 10000, 1000, 1
PERIODIC_STEPS = tuple(range(SAVE_EVERY, TOTAL_ITERATIONS + 1, SAVE_EVERY))   # 10000 .. 880000 (88)
TERMINAL_STEP = TOTAL_ITERATIONS
SAMPLE_ALGORITHM, SAMPLE_INITIAL_NOISE, COND_SCALE, DDIM_SKIP = 'ddpm', 250, 2, 10
LAUNCH_ENVIRONMENT = aio.LAUNCH_ENVIRONMENT
QUALIFICATION_LABELS = ('QUALIFICATION_ONLY', 'NOT_A_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_BANK',
                        'NOT_ELIGIBLE_FOR_DOWNSTREAM', 'NOT_ELIGIBLE_FOR_REPORTING', 'NOT_A_SCIENTIFIC_RUN')
MISSING_REASONS = {
    'train_metrics': 'Pinned FAS_train.py computes no train metric (loss, mse, vb only); none is fabricated.',
    'val_losses': 'NOT_USED: pinned FAS_train.py has no VAL pass (its val_loader argument is the TRAIN loader); VAL '
                  'is never opened.',
    'val_metrics': 'NOT_USED: pinned FAS_train.py has no VAL pass (its val_loader argument is the TRAIN loader); VAL '
                   'is never opened.'}
M6D6J_FILES = (CONTRACT_PATH, 'methods/difffas/main_runner_io.py', 'methods/difffas/main_checkpoint.py',
               'methods/difffas/main_runner.py', 'methods/difffas/main_runner_qualification.py', 'tools/run_e07c_main.py')
CanonicalFaceReader = aio.CanonicalFaceReader     # the M6D6e production reader, reused unchanged
Tee, tee_run_logs, untee = aio.Tee, aio.tee_run_logs, aio.untee
faces_root_from_exec_config = aio.faces_root_from_exec_config


def require(value, message):
    if not value:
        raise PreparationError('E07c main runner: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_contract():
    """The M6D6j contract is the JSON subset of YAML; every bound input SHA256 is verified here."""
    contract = json.loads((ROOT / CONTRACT_PATH).read_text())
    require((contract['method_id'], contract['milestone'], contract['role']) == (METHOD_ID, 'M6D6j', 'MAIN_DIFFFAS'),
            'contract identity')
    require((contract['fidelity_class'], contract['deviation'], contract['new_deviation'],
             contract['new_fidelity_class'], contract['amendment_created']) ==
            ('CONTROLLED_ADAPTATION', 'DEV-021', False, False, False), 'contract fidelity')
    for rel, digest in contract['bound_inputs_sha256'].items():
        require(sha256_file(ROOT / rel) == digest, 'bound input SHA256 ' + rel)
    require(contract['bound_inputs_sha256'][RELATION] == RELATION_SHA256 and
            contract['bound_inputs_sha256'][SPLIT_MANIFEST] == SPLIT_MANIFEST_SHA256, 'data inputs')
    d = contract['owner_decisions']
    require(d['D1_visualization']['decision'] == 'EXECUTE_EXACT_SOURCE' and
            d['D2_terminal_position']['decision'] == 'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION', 'D1/D2')
    return contract


def identities(contract):
    b = contract['bound_inputs_sha256']
    return {'config_sha256': b['configs/methods/e07c_difffas_bin_idfree.yaml'],
            'a1_adaptation_sha256': b['configs/frozen/difffas_bin_idfree_v1.yaml'],
            'a7_overlay_sha256': b['configs/amendments/e07c_a7_execution_policy.yaml'],
            'm6d6h_freeze_record_sha256': b['configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml'],
            'm6d6ir_retention_record_sha256': b['configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml'],
            'm6d6j_contract_sha256': sha256_file(ROOT / CONTRACT_PATH),
            'logging_contract_sha256': b['configs/run_logging_v1.yaml'],
            'environment_lock_sha256': b['environments/e07c.lock.json'],
            'relation_sha256': b[RELATION], 'split_manifest_sha256': b[SPLIT_MANIFEST],
            'main_graph_sha256': b['methods/difffas/main_graph.py'],
            'source_commit': contract['source_pin']['commit'], 'source_tree': contract['source_pin']['tree']}


# ----------------------------------------------------------------- modes, seeds, roots
def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'integer seed required')
    if mode == SCIENTIFIC:
        require(seed in EXPERIMENT_SEEDS, f'scientific E07c main runs use experiment seeds {EXPERIMENT_SEEDS}; got {seed}')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED, f'the M6D6j qualification seed is {QUALIFICATION_SEED}; got {seed}')
    else:
        raise PreparationError('unknown runner mode ' + repr(mode))


def qualification_id(config_sha256, commit):
    """run_logging_v1 formula with a qualification label; never a scientific run id."""
    return sha(f'{QUALIFICATION_RUN_LABEL}|{QUALIFICATION_SEED}|{config_sha256}|{commit}'.encode('utf-8'))[:16]


def run_root(runtime_root, mode, seed, rid=None):
    """Scientific: <rt>/runs/m6/E07c/seed_<seed>. Qualification: <rt>/qualification/m6d6j/E07c/q60608-<id>."""
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    require(rt.is_absolute() and not rt.resolve().is_relative_to(ROOT), 'absolute runtime root outside the repository')
    if mode == SCIENTIFIC:
        require(rid is None, 'the scientific root is fixed by run_logging_v1')
        return rt.joinpath(*SCIENTIFIC_PARTS, f'seed_{seed}')
    require(isinstance(rid, str) and re.fullmatch(r'[0-9a-f]{16}', rid) is not None, 'qualification id')
    root = rt.joinpath(*QUALIFICATION_PARTS, f'q{seed}-{rid}')
    require(not root.is_relative_to(rt / 'runs'), 'qualification root outside scientific runs')
    return root


# ----------------------------------------------------------------- A1 relation + TRAIN membership
def read_relation(path=None, *, pq=None):
    """8838 A1 rows in frozen file order ([dataset, gt_spoof_id]); allowlisted columns only; SHA256 before parse."""
    path = Path(path) if path is not None else ROOT / RELATION
    require(sha256_file(path) == RELATION_SHA256, 'A1 relation SHA256 differs from the frozen value')
    if pq is None:
        import pyarrow.parquet as pq
    schema = pq.read_schema(path).names
    require(not [c for c in RELATION_COLUMNS if c not in schema], 'allowlisted relation columns present')
    require(not [c for c in FORBIDDEN_COLUMNS if c in schema], 'identity / attack-taxonomy columns absent')
    table = pq.read_table(path, columns=list(RELATION_COLUMNS))
    require(tuple(table.column_names) == RELATION_COLUMNS, 'only allowlisted columns materialized')
    rows = table.to_pylist()
    require(len(rows) == TRAIN_ROWS, f'A1 relation must have {TRAIN_ROWS} rows; got {len(rows)}')
    by_dataset = {d: 0 for d in DATASETS}
    for r in rows:
        require(r['split'] == 'TRAIN' and r['use_pair'] is False and r['style_id'] == STYLE_ID and
                r['selection_policy'] == SELECTION_POLICY, 'TRAIN / use_pair=false / SPOOF_BINARY / frozen guide policy')
        require(r['dataset'] in DATASETS and TRACK_ID.fullmatch(r['track_pair_id'] or '') is not None, 'dataset / id')
        for key in ('gt_spoof_id', 'guide_spoof_id', 'content_live_id'):
            require(SAMPLE_ID.fullmatch(r[key] or '') is not None, key + ' pattern')
        by_dataset[r['dataset']] += 1
    require(by_dataset == ROWS_BY_DATASET, 'rows by dataset ' + json.dumps(by_dataset))
    keys = [(r['dataset'], r['gt_spoof_id']) for r in rows]
    require(keys == sorted(keys) and len(set(keys)) == TRAIN_ROWS, 'frozen row order [dataset, gt_spoof_id], unique GT')
    tracks = [r['track_pair_id'] for r in rows]
    require(tracks == sorted(tracks) and len(set(tracks)) == TRAIN_ROWS, 'track_pair_id order')
    records = tuple({'index': i, 'track_pair_id': r['track_pair_id'], 'dataset': r['dataset'],
                     'gt_spoof_id': r['gt_spoof_id'], 'guide_spoof_id': r['guide_spoof_id'],
                     'content_live_id': r['content_live_id']} for i, r in enumerate(rows))
    evidence = {'path': RELATION, 'sha256': RELATION_SHA256, 'sha256_checked_before_parse': True,
                'rows': len(records), 'rows_by_dataset': by_dataset, 'columns_read': list(RELATION_COLUMNS),
                'columns_never_read': [c for c in RELATION_NEVER_READ if c in schema],
                'forbidden_columns_in_schema': [c for c in FORBIDDEN_COLUMNS if c in schema],
                'split_values': ['TRAIN'], 'style_ids': [STYLE_ID], 'use_pair': False,
                'selection_policy': SELECTION_POLICY, 'self_guide_rows': sum(r['gt_spoof_id'] == r['guide_spoof_id']
                                                                              for r in records),
                'unique_guides': len({r['guide_spoof_id'] for r in records}),
                'unique_content': len({r['content_live_id'] for r in records}),
                'order': 'frozen file order [dataset, gt_spoof_id]',
                'order_sha256': sha('\n'.join(r['track_pair_id'] for r in records).encode('utf-8')),
                'gt_guide_sha256_semantics': ('split_v1 frame-lineage hashes (original frame bytes), NOT canonical-face '
                                              'PNG hashes; recorded, not used to verify face bytes')}
    return records, evidence


def read_train_membership(records, path=None, *, pq=None):
    """Every GT / guide / content id must be a TRAIN, M2-COMPLETE sample of split_v1; returns {sample_id: dataset}."""
    path = Path(path) if path is not None else ROOT / SPLIT_MANIFEST
    require(sha256_file(path) == SPLIT_MANIFEST_SHA256, 'split manifest SHA256 differs from the frozen value')
    if pq is None:
        import pyarrow.parquet as pq
    table = pq.read_table(path, columns=list(SPLIT_COLUMNS), filters=TRAIN_FILTER)
    require(tuple(table.column_names) == SPLIT_COLUMNS, 'only allowlisted split columns materialized')
    train = {}
    for r in table.to_pylist():
        require(r['split'] == 'TRAIN', 'only TRAIN rows materialized')
        train[r['sample_id']] = (r['dataset'], r['m2_status'])
    needed = {}
    for r in records:
        for role in ('gt_spoof_id', 'guide_spoof_id', 'content_live_id'):
            sid = r[role]
            require(sid in train, f'{role} {sid} is not a TRAIN sample')
            dataset, status = train[sid]
            require(status == 'COMPLETE', 'M2 COMPLETE canonical face required')
            if role != 'content_live_id':
                require(dataset == r['dataset'], 'GT / guide belong to the relation dataset (same-dataset pool)')
            require(needed.setdefault(sid, dataset) == dataset, 'one dataset per sample id')
    evidence = {'path': SPLIT_MANIFEST, 'sha256': SPLIT_MANIFEST_SHA256, 'columns_read': list(SPLIT_COLUMNS),
                'filter': 'split == TRAIN (pyarrow, native)', 'train_rows_materialized': len(train),
                'val_rows_materialized': 0, 'test_rows_materialized': 0, 'relation_ids': len(needed),
                'relation_ids_all_train': True, 'gt_guide_same_dataset': True}
    return needed, evidence


class MainTrainDataset:
    """Map-style A1 dataset shaped like FAS_dataset.py items: {'content', 'style_spoof', 'GT'} (no label key).

    Read order content, GT, guide and transform order content, style, GT follow FAS_dataset.py:34-45. The guide is the
    frozen guide_spoof_id (no random.choice). `on_item` is an accounting callback; it never changes the item.
    """
    def __init__(self, records, reader, transform, on_item=None):
        require(len(records) == TRAIN_ROWS and [r['index'] for r in records] == list(range(TRAIN_ROWS)), 'records')
        self.records, self.reader, self.transform, self.on_item = tuple(records), reader, transform, on_item

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        content_img = self.reader(r['content_live_id'])
        gt_img = self.reader(r['gt_spoof_id'])
        style_img = self.reader(r['guide_spoof_id'])
        if self.on_item is not None:
            self.on_item(r)
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)
        gt_img = self.transform(gt_img)
        return {'content': content_img, 'style_spoof': style_img, 'GT': gt_img}


# ----------------------------------------------------------------- trajectory records
def finite(values):
    return all(isinstance(v, float) and v == v and abs(v) != float('inf') for v in values)


def step_record(*, mode, seed, epoch, global_step, iteration, learning_rate, loss, mse, vb, batch_size,
                batch_track_sha256, wall_clock_seconds, gpu_memory_bytes, extra=None):
    """One trajectory record per GLOBAL optimizer step (run_logging_v1 fields + explicit null reasons)."""
    require(finite([float(loss), float(mse), float(vb)]), 'non-finite loss')
    record = {'epoch': epoch, 'global_step': global_step, 'iteration_in_epoch': iteration,
              'learning_rate': learning_rate, 'batch_size': batch_size,
              'train_losses': {'loss': float(loss), 'mse': float(mse), 'vb': float(vb)},
              'train_losses_semantics': 'loss = mse + vb (learned-range), FAS_train.py:70-72 means',
              'train_metrics': None, 'val_losses': None, 'val_metrics': None,
              'wall_clock_seconds': wall_clock_seconds, 'gpu_memory_bytes': gpu_memory_bytes,
              'gpu_memory_semantics': 'torch.cuda.max_memory_allocated() within this optimizer step',
              'batch_track_pair_id_sha256': batch_track_sha256, 'record_granularity': 'OPTIMIZER_STEP',
              'runner_mode': mode, 'missing_field_reasons': dict(MISSING_REASONS)}
    record['experiment_seed' if mode == SCIENTIFIC else 'qualification_seed'] = seed
    if mode == QUALIFICATION:
        record['experiment_seed'] = None
        record['missing_field_reasons']['experiment_seed'] = 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN'
        record['labels'] = list(QUALIFICATION_LABELS)
    record.update(extra or {})
    return record


# ----------------------------------------------------------------- production CLI refusals
REFUSED_OPTIONS = ('--epochs', '--max-epochs', '--max_epochs', '--batch-size', '--batch_size', '--bs', '--workers',
                   '--num-workers', '--num_workers', '--drop-last', '--drop_last', '--shuffle', '--no-shuffle', '--lr',
                   '--learning-rate', '--scheduler', '--warmup', '--amp', '--fp16', '--bf16', '--half', '--tf32',
                   '--microbatch', '--accumulate', '--grad-accum', '--activation-checkpointing', '--checkpoint',
                   '--checkpoint-path', '--pretrain-path', '--pretrain_path', '--save-every', '--save_checkpoints_every_iters',
                   '--checkpoint-every', '--visualization', '--disable-visualization', '--no-visualization',
                   '--visualization-frequency', '--save_images_every_iters', '--save-images-every', '--sample-algorithm',
                   '--sample_algorithm', '--sample-initial-noise', '--sample_initial_noise', '--ddim', '--manifest',
                   '--train-manifest', '--data-root', '--faces-root', '--split', '--val', '--test', '--encoder',
                   '--pretrain-classifier', '--pretrain_classifier', '--encoder-sha256', '--resume', '--resume-latest',
                   '--resume-state', '--resume-from', '--auto-resume', '--qualification', '--qualification-seed',
                   '--guidance-prob', '--guidance_prob', '--use-pair', '--use_pair', '--means-size', '--var-size',
                   '--select-checkpoint', '--keep-checkpoints', '--no-prune', '--prune', '--run-root', '--output')


def refused_arguments(argv):
    """Override flags the production CLI refuses before parsing (fail closed; nothing imported or executed)."""
    return sorted({a.split('=', 1)[0] for a in argv if a.split('=', 1)[0] in REFUSED_OPTIONS})


def m6d6j_file_hashes():
    return {rel: sha256_file(ROOT / rel) for rel in M6D6J_FILES if (ROOT / rel).is_file()}
