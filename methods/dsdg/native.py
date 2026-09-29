"""E06b DSDG-NATIVE (Track B): static adapter, native same-subject relation, loader contract, batch plan.

Reuses, unchanged: the pinned-source validation, LightCNN verification, argv mapping and checkpoint plan of
the E06c DSDGAdapter; the M6D5c/M6D5d microbatch chunk helpers; the common seeded loader hooks. It never
reuses the E06c Amendment-A1 guards (lambda_pair == 0, attack_type == 1, common-pair relation, physical
batch 240 only).

Official relation (addition_module/DSDG/data/generation_dataset.py:48,87,93): the dataset is indexed by
SPOOF frames; each load draws the live partner with the module-level `random.choice` over the SAME subject's
live frames. Nothing is materialized: manifests/dsdg_identity_pairs_v1.parquet is never written.

STATIC: importing this module loads no model, image, GPU or torch. `load_train_rows` reads only the
TRAIN-filtered, allowlisted split metadata (no image, no TEST row materialized). M6F-B qualifies nothing
at runtime: the GPU graph and the production runner are NOT yet qualified.
"""
import json
import random

import numpy as np

from methods.common.config import ROOT, sha256_file
from methods.common.learned import PreparationError, authoritative, checkpoint_plan, seed_plan
from . import microbatch_execution as mb1
from . import microbatch_execution_v2 as mb2
from .adapter import DSDGAdapter

METHOD_ID = 'E06b'
DATASETS = ('casia_fasd', 'msu_mfsd')
NON_INSTANTIABLE = {'siwmv2': 'NOT_INSTANTIABLE_MISSING_SUBJECT_ID'}
VOCABULARY = ('print', 'replay')
CLASS_INDEX = {'print': 0, 'replay': 1}
LIVE_MACRO = 'live'
COLUMNS = ('sample_id', 'dataset', 'subject_id_global', 'label_binary', 'attack_macro', 'split', 'm2_status')
SPLIT_MANIFEST = 'manifests/split_v1.parquet'
CONTRACT = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json'
GLOBAL_BATCH = 240
MICROBATCH = mb2.MAX_MICROBATCH
WORKERS = 8
STATUSES = ('CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_NOT_YET_QUALIFIED',
            'PRODUCTION_RUNNER_NOT_YET_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED')
FINAL_EXECUTION_FIDELITY = 'PENDING_M6F_C_RUNTIME_QUALIFICATION'


def byte_order(sample_id):
    """Ascending bytewise order (A2-01 precedent); independent of filesystem or input order."""
    return sample_id.encode('utf-8')


# ----------------------------------------------------------------- frozen contract
def native_semantics(config):
    """Validate the frozen E06b config against its own invariants and the M6FA contract; returns semantics."""
    c = config
    contract_path = ROOT / c['contract_resolution']
    if sha256_file(contract_path) != c['contract_resolution_sha256']:
        raise PreparationError('M6FA contract bytes differ from the SHA256 frozen in the E06b config')
    k = json.loads(contract_path.read_text())
    pop, rel, sts = c['native_population'], c['native_relation'], c['spoof_type_supervision']
    if (c['method_id'], c['track'], k['method_id'], k['track']) != (METHOD_ID, 'B_NATIVE_FULL_SECONDARY',
                                                                    METHOD_ID, 'B_NATIVE_FULL_SECONDARY'):
        raise PreparationError('E06b identity/track mismatch')
    if tuple(pop['train_datasets']) != DATASETS or pop['non_instantiable']['siwmv2']['status'] != NON_INSTANTIABLE['siwmv2']:
        raise PreparationError('E06b native TRAIN scope must be CASIA + MSU; SiW not instantiable')
    if (tuple(sts['vocabulary']), sts['class_index'], sts['num_spoof_classes'], c['training']['attack_type']) != \
            (VOCABULARY, CLASS_INDEX, len(VOCABULARY), len(VOCABULARY)) or sts['unexpected_macro'] != 'REJECT_NO_REMAP':
        raise PreparationError('native spoof-type space must be attack_macro {print: 0, replay: 1}, K = 2')
    if (sts['vocabulary'], sts['class_index']) != (k['spoof_type_supervision']['vocabulary'],
                                                   k['spoof_type_supervision']['class_index']):
        raise PreparationError('config/contract spoof-type disagreement')
    lo = c['losses']
    for name in ('lambda_mmd', 'lambda_ip', 'lambda_pair', 'lambda_type', 'lambda_ort'):
        if lo[name] != k['losses'][name]:
            raise PreparationError(f'config/contract disagreement on {name}')
    if not (lo['lambda_pair'] > 0 and lo['loss_pair_active'] and lo['loss_cls_active'] and lo['removed'] == []):
        raise PreparationError('native DSDG keeps every official loss (identity pair and spoof-type active)')
    t = c['training']
    for key in ('all_epochs', 'effective_batch_size', 'hdim', 'workers', 'pre_epoch', 'test_epoch'):
        if t[key] != k['training'][key]:
            raise PreparationError(f'config/contract disagreement on training.{key}')
    if c['optimizer']['learning_rate'] != k['training']['learning_rate']:
        raise PreparationError('config/contract disagreement on learning rate')
    if (rel['semantics'], rel['materialized_pair_list'], rel['native_manifest'], rel['spoof_index_order'],
            rel['live_pool_order']) != ('SAME_SUBJECT_ONLINE_RANDOM', False, 'NOT_CREATED',
                                        'ASCENDING_SAMPLE_ID_BYTEWISE', 'ASCENDING_SAMPLE_ID_BYTEWISE_WITHIN_SUBJECT'):
        raise PreparationError('native relation must be the official online same-subject draw, never materialized')
    ld = c['loader']
    if (ld['batch_size'], ld['num_workers'], ld['persistent_workers'], ld['drop_last'], ld['shuffle']) != \
            (GLOBAL_BATCH, WORKERS, False, False, True) or t['workers'] != WORKERS:
        raise PreparationError('E06b loader contract: batch 240, 8 workers, non-persistent, drop_last False')
    if c['checkpoint']['rule'] != k['checkpoint']['rule'] or c['seeds']['experiment_seeds'] != k['seeds']['experiment_seeds']:
        raise PreparationError('config/contract disagreement on checkpoint rule or seeds')
    if c['generation']['n_syn_intended'] != 'DEFERRED_TO_M8' or c['final_execution_fidelity'] != FINAL_EXECUTION_FIDELITY:
        raise PreparationError('bank budget must stay DEFERRED_TO_M8; fidelity must stay pending M6F-C')
    return {'datasets': DATASETS, 'vocabulary': VOCABULARY, 'class_index': dict(CLASS_INDEX),
            'expected': pop['expected'], 'contract_sha256': c['contract_resolution_sha256']}


class DSDGNativeAdapter(DSDGAdapter):
    """E06b: E06c's generic preparation (source, LightCNN, argv mapping) with the native contract."""
    method_id = METHOD_ID

    def _semantics(self):
        return native_semantics(authoritative(self.config))

    def spoof_class(self, attack_macro):
        if attack_macro not in CLASS_INDEX:
            raise PreparationError(f'unexpected spoof attack_macro {attack_macro!r}; no silent remap')
        return CLASS_INDEX[attack_macro]

    def lambdas(self):
        """Loss coefficients for the (parameterized) M6D5c graph; the E06c frozen-lambda guard is NOT used."""
        lo = authoritative(self.config)['losses']
        return {k: lo[k] for k in ('lambda_mmd', 'lambda_ip', 'lambda_type', 'lambda_ort', 'lambda_pair')}

    def validate_batch(self, **_ignored):
        """Static candidate mapping only; M6F-C must qualify it before any runtime use."""
        plan = epoch_plan(authoritative(self.config)['native_population']['expected']['total']['spoof'])
        return dict(plan, execution_status='CANDIDATE_NOT_YET_QUALIFIED', runtime_accumulation_executed=False)

    def build_training_plan(self, seed, *, selection_split=None, **_batch_options):
        self.prepare()
        cfg = authoritative(self.config)
        settings = self.settings + [{'target': '--batch_size', 'value': cfg['training']['effective_batch_size'],
                                     'config_field': 'training.effective_batch_size (global batch)',
                                     'source_file': 'addition_module/DSDG/train_generator.sh:7'}]
        return {'method_id': METHOD_ID, 'status': 'STATIC_PREPARED_NOT_EXECUTED', 'statuses': list(STATUSES),
                'experiment_seed': seed, 'config_sha256': cfg['_runtime']['config_sha256'],
                'target_fidelity': cfg['target_fidelity'], 'final_execution_fidelity': FINAL_EXECUTION_FIDELITY,
                'source': self.source, 'lightcnn': self.lightcnn, 'settings_mapping': settings,
                'argument_vector_is_launch_command': False, 'loss_contract': self.lambdas(),
                'spoof_type': dict(CLASS_INDEX), 'seed_plan': seed_plan(cfg, seed, 'torch'),
                'loader_contract': loader_contract(cfg), 'batch_plan': self.validate_batch(),
                'checkpoint': checkpoint_plan(cfg, seed, selection_split=selection_split),
                'training_launched': False, 'checkpoint_created': False, 'gpu_graph_qualified': False,
                'production_runner_qualified': False}


# ----------------------------------------------------------------- native relation
class NativeRelation:
    """Validated CASIA+MSU TRAIN population: spoof index + same-subject live pools (in memory only)."""

    def __init__(self, rows):
        seen, spoof, pools = set(), [], {}
        for r in rows:
            if tuple(sorted(r)) != tuple(sorted(COLUMNS)):
                raise PreparationError('rows must carry exactly the allowlisted split columns')
            if r['split'] != 'TRAIN':
                raise PreparationError('E06b native relation is TRAIN only')
            if r['dataset'] in NON_INSTANTIABLE:
                raise PreparationError(f"{r['dataset']} is {NON_INSTANTIABLE[r['dataset']]} for E06b")
            if r['dataset'] not in DATASETS:
                raise PreparationError(f"dataset {r['dataset']!r} outside the E06b native scope")
            if r['m2_status'] != 'COMPLETE':
                raise PreparationError('only M2-COMPLETE canonical faces are admissible')
            sid, subject = r['sample_id'], r['subject_id_global']
            if not (isinstance(sid, str) and sid) or sid in seen:
                raise PreparationError('sample_id must be a unique nonempty string')
            if not (isinstance(subject, str) and subject.startswith(r['dataset'] + '::')):
                raise PreparationError('a trustworthy dataset subject_id_global is required; no pseudo identity')
            seen.add(sid)
            key = (r['dataset'], subject)
            if r['label_binary'] == 0 and r['attack_macro'] == LIVE_MACRO:
                pools.setdefault(key, []).append(sid)
            elif r['label_binary'] == 1 and r['attack_macro'] != LIVE_MACRO:
                spoof.append((sid, key, self._class(r['attack_macro'])))
            else:
                raise PreparationError('label_binary/attack_macro inconsistency')
        self.pools = {k: tuple(sorted(v, key=byte_order)) for k, v in pools.items()}
        missing = sorted({k for _, k, _ in spoof if not self.pools.get(k)})
        if missing:
            raise PreparationError(f'spoof subjects without a same-subject live pool: {missing[:3]}')
        self.spoof = tuple(sorted(spoof, key=lambda item: byte_order(item[0])))

    @staticmethod
    def _class(macro):
        if macro not in CLASS_INDEX:
            raise PreparationError(f'unexpected spoof attack_macro {macro!r}; no silent remap')
        return CLASS_INDEX[macro]

    def __len__(self):
        return len(self.spoof)

    def summary(self):
        out = {}
        for ds in DATASETS:
            sp = [s for s in self.spoof if s[1][0] == ds]
            pools = {k: v for k, v in self.pools.items() if k[0] == ds}
            out[ds] = {'spoof': len(sp), 'live': sum(len(v) for v in pools.values()),
                       'subjects': len({k for _, k, _ in sp} | set(pools)),
                       **{m: sum(1 for s in sp if s[2] == i) for m, i in CLASS_INDEX.items()}}
        out['total'] = {k: sum(out[ds][k] for ds in DATASETS) for k in ('spoof', 'live', 'subjects', *VOCABULARY)}
        return out

    def draw_live(self, index, rng=random):
        """Official get_pair: random.choice over the same subject's live pool (worker-seeded module RNG)."""
        return rng.choice(self.pools[self.spoof[index][1]])


def canonical_chw(array, size=256):
    array = np.asarray(array)
    if array.shape != (size, size, 3) or array.dtype != np.uint8:
        raise PreparationError('canonical uint8 RGB256 required; no implicit recrop/rescale')
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32) / 255


class NativePairDataset:
    """DataLoader-compatible GenDataset_s replacement: spoof-indexed, live partner redrawn on every load.

    The injected reader receives sample IDs only. Keys follow the official item: '0' spoof, '1' live,
    'type' the zero-based attack_macro class index.
    """

    def __init__(self, relation, canonical_rgb_reader):
        self.relation = relation
        self.reader = canonical_rgb_reader

    def __len__(self):
        return len(self.relation)

    def __getitem__(self, index):
        spoof_id, _, cls = self.relation.spoof[index]
        live_id = self.relation.draw_live(index)          # drawn first, as in the official __getitem__
        return {'0': canonical_chw(self.reader(spoof_id)), '1': canonical_chw(self.reader(live_id)),
                'type': np.int64(cls)}


# ----------------------------------------------------------------- loader / randomness contract
def loader_contract(config):
    ld = config['loader']
    return {'batch_size': ld['batch_size'], 'shuffle': ld['shuffle'], 'num_workers': ld['num_workers'],
            'persistent_workers': ld['persistent_workers'], 'drop_last': ld['drop_last'],
            'pin_memory': ld['pin_memory'], 'generator': 'torch.Generator().manual_seed(experiment_seed)',
            'worker_init_fn': 'methods.common.learned.seed_torch_worker',
            'worker_python_random_seed': '(per-epoch loader base seed + worker_id) mod 2**32',
            'worker_batch_assignment': worker_batch_assignment(1, ld['num_workers'])['rule'],
            'num_workers_affects_live_draws': True}


def worker_batch_assignment(n_batches, workers=WORKERS):
    """In-order multi-process DataLoader: worker w serves global batches w, w + workers, ..."""
    return {'rule': f'batch b is served by worker b % {workers}',
            'assignment': {w: list(range(w, n_batches, workers)) for w in range(workers)}}


def simulate_live_draws(relation, batches, worker_seeds):
    """Pure-Python model of the per-worker live draws for one epoch (M6F-C proves it against torch).

    `batches` are index lists in loader order; worker w seeds one random.Random(worker_seeds[w]) and serves
    its batches in order, drawing items in index order, exactly like the module RNG in a seeded worker.
    """
    workers = len(worker_seeds)
    streams = [random.Random(s) for s in worker_seeds]
    draws = [None] * len(batches)
    for w in range(workers):
        for b in range(w, len(batches), workers):
            draws[b] = [relation.draw_live(i, streams[w]) for i in batches[b]]
    return draws


# ----------------------------------------------------------------- batch plan
def epoch_plan(rows):
    """15 x 240 + 120 for 3720 rows; each global batch as microbatch-20 chunks (M6D5c/M6D5d helpers)."""
    plan = mb2.epoch_batch_plan(rows, GLOBAL_BATCH, False)
    chunks = [len(mb2.chunk_slices_v2(b, MICROBATCH)) for b in plan['batch_sizes']]
    if set(mb1.LOCAL_TERMS) != {'loss_rec', 'loss_kl', 'loss_ip', 'loss_cls', 'loss_pair'} or \
            set(mb1.GLOBAL_TERMS) != {'loss_mmd', 'loss_ort'}:
        raise PreparationError('M6D5c chunk objective no longer weights loss_cls/loss_pair as per-sample means')
    tail = chunks[-1] if plan['batch_sizes'][-1] != GLOBAL_BATCH else None
    return dict(plan, microbatch=MICROBATCH, chunks_per_batch=chunks, chunks_full_batch=chunks[0],
                chunks_tail_batch=tail)


# ----------------------------------------------------------------- TRAIN metadata (runner plumbing)
def load_train_rows(config, split_path=None):
    """SHA256 before parse; TRAIN + CASIA/MSU filter at read; allowlisted columns; no image, no TEST row."""
    import pyarrow.parquet as pq
    path = ROOT / (split_path or config['data']['split_manifest'])
    if sha256_file(path) != config['data']['split_manifest_sha256']:
        raise PreparationError('split manifest SHA256 differs from the frozen config')
    rows = pq.read_table(path, columns=list(COLUMNS),
                         filters=[('split', '==', 'TRAIN'), ('dataset', 'in', list(DATASETS))]).to_pylist()
    if any(r['split'] != 'TRAIN' or r['dataset'] not in DATASETS for r in rows):
        raise PreparationError('split filter leaked a non-TRAIN or out-of-scope row')
    return rows
