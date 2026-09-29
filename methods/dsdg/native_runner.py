"""E06b DSDG-NATIVE production runner (M6F-D).

Target FAITHFUL_OFFICIAL; runtime fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY (M6F-C owner decision).

Binds, unchanged: the frozen E06b config (M6F-B) and the M6F-A contract, the pinned FaceX-Zoo DSDG networks/misc.util,
LightCNN-29 v2 (frozen, eval, SHA-verified), the DSDG environment of record (environments/e06c.lock.json) and the
M6F-C-qualified execution mapping (the unchanged M6D5c/M6D5d GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 functions:
global batch 240, microbatch 20, drop_last=False, 15 x 240 + 120 = 16 optimizer steps per epoch).

Reused from the E06c production runner WITHOUT modification: loader verification, precision/seeding hooks,
environment record, the pinned visualization block and the canonical face reader. Replaced for E06b only: the
per-global-batch gates (K = 2 active CE, lambda_pair = 5 active, spoof-indexed native relation), the epoch coverage
check and the checkpoint event (fresh-only: no engineering resume sidecar; resume is not qualified for E06b).

Two modes share this engine and nothing else:
  SCIENTIFIC     seeds 42/1337/2026, 200 epochs, <runtime_root>/runs/m6/E06b/seed_<seed>/, launched only by
                 tools/run_e06b.py from a clean worktree (fresh only).
  QUALIFICATION  seed 60801, <runtime_root>/qualification/m6fd/E06b/<case>/, launched only by
                 methods/dsdg/native_runner_qualification.py; never scientific, never a checkpoint.
Torch is imported by the caller or inside functions; importing this module is static.
"""
import json
import os
from pathlib import Path
import time

import numpy as np

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import PreparationError, seed_torch_worker, torch_loader_options
from methods.dsdg import microbatch_execution as v1
from methods.dsdg import microbatch_execution_v2 as v2
from methods.dsdg import native
from methods.dsdg import runner as e06c_runner
from methods.dsdg import runner_io as rio
from methods.dsdg import training_graph as tg

METHOD_ID = native.METHOD_ID
SCIENTIFIC, QUALIFICATION = rio.SCIENTIFIC, rio.QUALIFICATION
SCIENTIFIC_SEEDS = (42, 1337, 2026)
QUALIFICATION_SEED = 60801       # M6F-D engineering seed; never an experiment seed; unused by any earlier harness
QUALIFICATION_PARTS = ('qualification', 'm6fd', 'E06b')
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
CONFIG_PATH = 'configs/methods/e06b_dsdg_native.yaml'
SNAPSHOT_PATH = 'frozen_config_snapshot/' + CONFIG_PATH
CONTRACT_PATH, CONTRACT_SHA = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json', \
    'fe287ebe1ec9c19e5f63a40f1498e41f5b7044014c2cc7eb6279a32604555058'
M6FC_EVIDENCE_PATH = 'outputs/audit/M6FC_E06B_GPU_QUALIFICATION.json'
LOCK_PATH, LOCK_SHA = 'environments/e06c.lock.json', '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
EXEC_CONFIG = rio.EXEC_CONFIG
EXECUTION_MODE = v2.EXECUTION_MODE
RUNTIME_FIDELITY = 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY'
GLOBAL_BATCH, MICROBATCH, ROWS = 240, 20, 3720
LAMBDAS = {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10, 'lambda_ort': 1, 'lambda_pair': 5}
LOADER = dict(rio.LOADER)        # batch 240, shuffle, 8 workers, pin_memory, drop_last=False (persistent_workers False)
ALL_EPOCHS = 200
EXTRA_REFUSED = ('--resume', '--resume-state', '--resume-from', '--auto-resume', '--qualification-only',
                 '--qualification-max-logical-batches', '--k', '--num-classes', '--datasets', '--include-siw',
                 '--subset', '--limit')
require = rio.require


def refused_arguments(argv):
    """The E06c production refusals plus resume/qualification/scope flags (the scientific CLI is fresh-only)."""
    bad = set(rio.refused_arguments(argv))
    bad |= {a.split('=', 1)[0] for a in argv if a.split('=', 1)[0] in EXTRA_REFUSED}
    return sorted(bad)


def validate_mode_seed(mode, seed):
    require(type(seed) is int, 'integer seed required')
    if mode == SCIENTIFIC:
        require(seed in SCIENTIFIC_SEEDS, f'scientific seed must be one of {SCIENTIFIC_SEEDS}; got {seed}')
    elif mode == QUALIFICATION:
        require(seed == QUALIFICATION_SEED and seed not in SCIENTIFIC_SEEDS,
                f'qualification seed must be {QUALIFICATION_SEED}')
    else:
        raise PreparationError('unknown runner mode ' + repr(mode))


def run_root(runtime_root, mode, seed, case=None):
    """Scientific: <rt>/runs/m6/E06b/seed_<seed>. Qualification: <rt>/qualification/m6fd/E06b/<case> (never runs/)."""
    validate_mode_seed(mode, seed)
    rt = Path(runtime_root)
    require(rt.is_absolute(), 'absolute runtime root')
    if mode == SCIENTIFIC:
        return rt / 'runs' / 'm6' / METHOD_ID / f'seed_{seed}'
    require(case in ('loader', 'b240', 'tail120'), 'qualification case')
    root = rt.joinpath(*QUALIFICATION_PARTS, case)
    require(not root.is_relative_to(rt / 'runs'), 'qualification root outside scientific runs')
    return root


# ----------------------------------------------------------------- identities / contract (framework-free)
def verify_contract(config=None):
    """Frozen-config + snapshot + contract gates; K, class index and lambdas are checked on the live config."""
    config = config or load_method_config(METHOD_ID)
    require(config['_runtime']['config_sha256'] == CONFIG_SHA, 'frozen E06b config SHA256')
    require((ROOT / CONFIG_PATH).read_bytes() == (ROOT / SNAPSHOT_PATH).read_bytes(), 'config/snapshot byte-identical')
    require(sha256_file(ROOT / CONTRACT_PATH) == CONTRACT_SHA, 'M6F-A contract SHA256')
    sem = native.native_semantics(config)
    t, lo, sts = config['training'], config['losses'], config['spoof_type_supervision']
    require(t['attack_type'] == len(sem['vocabulary']) == 2 and sts['num_spoof_classes'] == 2, 'K = 2')
    require(sts['class_index'] == {'print': 0, 'replay': 1}, 'class index print=0, replay=1')
    require({k: lo[k] for k in LAMBDAS} == LAMBDAS, 'lambda_pair = 5 and the official coefficients')
    require((t['effective_batch_size'], t['workers'], t['all_epochs'], t['hdim']) == (GLOBAL_BATCH, 8, ALL_EPOCHS, 128),
            'effective batch 240, workers 8, 200 epochs, hdim 128')
    require(config['checkpoint']['rule'] == 'OFFICIAL_GENERATOR_EPOCH_200', 'checkpoint rule epoch 200')
    ld = config['loader']
    require((ld['batch_size'], ld['num_workers'], ld['persistent_workers'], ld['drop_last'], ld['shuffle']) ==
            (240, 8, False, False, True) and
            {k: LOADER[k] for k in ('batch_size', 'num_workers', 'drop_last', 'shuffle')} ==
            {'batch_size': 240, 'num_workers': 8, 'drop_last': False, 'shuffle': True}, 'loader contract')
    m6fc = json.loads((ROOT / M6FC_EVIDENCE_PATH).read_text())
    require(m6fc['fidelity']['runtime_fidelity_assessment'] == RUNTIME_FIDELITY and m6fc['status'] == 'PASS',
            'M6F-C runtime qualification of the execution mapping')
    return config


def identities(config):
    lock_raw = (ROOT / LOCK_PATH).read_bytes()
    require(rio.sha(lock_raw) == LOCK_SHA, 'DSDG environment lock SHA256')
    lock = json.loads(lock_raw)
    return {'config_sha256': CONFIG_SHA, 'contract_sha256': CONTRACT_SHA,
            'freeze_record_sha256': sha256_file(ROOT / 'outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json'),
            'm6fc_evidence_sha256': sha256_file(ROOT / M6FC_EVIDENCE_PATH),
            'm6d5c_v1_overlay_sha256': sha256_file(ROOT / v1.RESOLUTION_PATH),
            'm6d5d_v2_overlay_sha256': sha256_file(ROOT / v2.RESOLUTION_PATH),
            'logging_contract_sha256': sha256_file(ROOT / 'configs/run_logging_v1.yaml'),
            'environment_lock_sha256': LOCK_SHA, 'split_manifest_sha256': config['data']['split_manifest_sha256'],
            'source_commit': lock['source']['commit'], 'source_tree': lock['source'].get('tree'),
            'source_files_sha256': lock['source']['files_sha256'], 'lightcnn_sha256': LIGHTCNN_SHA,
            'runtime_fidelity_assessment': RUNTIME_FIDELITY}


def verify_source(adapter, ids):
    source = adapter.validate_source()
    require(source['commit'] == ids['source_commit'] and source['files_sha256'] == ids['source_files_sha256'],
            'source identity equals the DSDG environment lock')
    return source


def lightcnn_identity(path):
    ident = e06c_runner.lightcnn_identity(path)      # unchanged SHA256/byte gate
    require(ident['sha256'] == LIGHTCNN_SHA, 'LightCNN SHA256')
    return ident


# ----------------------------------------------------------------- native TRAIN relation + dataset
def load_relation(config):
    """split_v1 TRAIN + CASIA/MSU (SHA before parse; allowlisted columns) -> the validated native relation."""
    rows = native.load_train_rows(config)
    relation = native.NativeRelation(rows)
    summary = relation.summary()
    require(summary['total'] == config['native_population']['expected']['total'], 'native population = frozen counts')
    sample_datasets = {sid: key[0] for sid, key, _ in relation.spoof}
    for (ds, _), pool in relation.pools.items():
        sample_datasets.update({sid: ds for sid in pool})
    return rows, relation, sample_datasets, summary


def assert_native_rows(relation, rows, indices):
    """Independent pre-execution assertions for the rows a batch will draw from (TRAIN, CASIA/MSU, same subject)."""
    by_id = {r['sample_id']: r for r in rows}
    out = []
    for i in indices:
        sid, key, cls = relation.spoof[i]
        spoof = by_id[sid]
        require(spoof['split'] == 'TRAIN' and spoof['dataset'] in native.DATASETS and spoof['label_binary'] == 1,
                'spoof row TRAIN, CASIA/MSU')
        require(spoof['attack_macro'] in native.CLASS_INDEX and native.CLASS_INDEX[spoof['attack_macro']] == cls,
                'attack_macro exactly print/replay with the frozen index')
        require(spoof['subject_id_global'] == key[1] and key[1].startswith(spoof['dataset'] + '::'), 'real subject id')
        for live_id in relation.pools[key]:
            live = by_id[live_id]
            require(live['split'] == 'TRAIN' and live['dataset'] == spoof['dataset'] and live['label_binary'] == 0 and
                    live['attack_macro'] == 'live' and live['subject_id_global'] == spoof['subject_id_global'],
                    'every candidate live partner is TRAIN, same dataset, same subject')
        out.append({'spoof_id': sid, 'dataset': spoof['dataset'], 'subject_id_global': key[1],
                    'attack_macro': spoof['attack_macro'], 'class_index': cls,
                    'live_pool_size': len(relation.pools[key])})
    return out


class E06bDataset(native.NativePairDataset):
    """The M6F-B NativePairDataset (official spoof-indexed item, live redraw on every load) plus provenance IDs."""

    def __getitem__(self, index):
        spoof_id, _, cls = self.relation.spoof[index]
        live_id = self.relation.draw_live(index)          # module random, seeded per worker by seed_torch_worker
        return {'0': native.canonical_chw(self.reader(spoof_id)), '1': native.canonical_chw(self.reader(live_id)),
                'type': np.int64(cls), 'index': np.int64(index), 'spoof_id': spoof_id, 'live_id': live_id}


def build_loader(torch, dataset, mode, seed, config):
    """Production DataLoader (batch 240, shuffle, 8 workers, pin_memory, drop_last=False, non-persistent)."""
    validate_mode_seed(mode, seed)
    if mode == SCIENTIFIC:
        opts = torch_loader_options(config, seed)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        opts = {'generator': generator, 'worker_init_fn': seed_torch_worker}
    loader = torch.utils.data.DataLoader(dataset, **LOADER, **opts)
    evidence = e06c_runner.verify_loader(torch, loader, opts['generator'], len(dataset))  # unchanged E06c gate
    return loader, opts['generator'], evidence


# ----------------------------------------------------------------- trainer (native gates)
class E06bTrainer(e06c_runner.Trainer):
    """E06c Trainer engine with the E06b per-batch gates; the V2 global step itself is the unchanged M6D5d code."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require(self.lam == LAMBDAS, 'native lambdas (lambda_pair = 5)')

    def global_step_run(self, batch, epoch, iteration, capture=False, keep_visual=False):
        torch, gate = self.torch, e06c_runner.gate
        indices = batch['index'].tolist()
        spoof_ids, live_ids = list(batch['spoof_id']), list(batch['live_id'])
        x_spoof, x_live = batch['0'].to(self.device), batch['1'].to(self.device)
        label = batch['type'].to(self.device)
        B = x_spoof.shape[0]
        gate(len(set(indices)) == B == len(spoof_ids) == len(live_ids) and 1 <= B <= GLOBAL_BATCH,
             'batch rows unique, 1 <= B <= 240')
        gate(all(lid in self.records.pools[self.records.spoof[i][1]] for i, lid in zip(indices, live_ids)),
             'every live partner is from the same subject pool')
        gate(tuple(x_spoof.shape) == tuple(x_live.shape) == (B, 3, 256, 256) and
             x_spoof.dtype == x_live.dtype == torch.float32, 'inputs FP32 [B,3,256,256]')
        gate(label.dtype == torch.int64 and tuple(label.shape) == (B,) and
             set(label.tolist()) <= {0, 1} and label.tolist() == [self.records.spoof[i][2] for i in indices],
             'native CE index in {0 print, 1 replay} equals the relation class')
        lo, hi = min(float(x_spoof.min()), float(x_live.min())), max(float(x_spoof.max()), float(x_live.max()))
        gate(0.0 <= lo and hi <= 1.0, 'inputs in [0,1] (ToTensor /255 only)')
        chunks = [s.stop - s.start for s in v2.chunk_slices_v2(B)]
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        eps = v2.draw_epsilon(torch, B, self.device)
        eps_sha = {k: e06c_runner.tsha(eps[k]) for k in v2.EPS_ORDER}
        seen = {}

        def before_step():
            grads = [p.grad for _, p in self.owned]
            present = [g for g in grads if g is not None]
            finite_count = int(torch.stack([torch.isfinite(g).all() for g in present]).sum()) if present else 0
            seen['owned'] = {'tensors': len(grads), 'non_none': len(present), 'finite': finite_count}
            gate(len(grads) == len(present) == finite_count == 55,
                 '55/55 owned finite gradients: ' + repr(seen['owned']))
            gate(all(p.grad is None for p in self.models['netIP'].parameters()), 'netIP receives no gradient')
            cls_grads = [p.grad for p in self.models['netCls'].parameters()]
            gate(all(g is not None and bool(torch.isfinite(g).all()) for g in cls_grads), 'netCls gradients finite')
            seen['netCls_grad_nonzero'] = sum(int(torch.count_nonzero(g)) for g in cls_grads)
        recs = []

        def on_chunk(i, out):
            recs.append((out['rec_nir'].detach().clone(), out['rec_vis'].detach().clone()))
        before_apps = self.optimizer_applications
        try:
            run = v2.run_global_batch_v2(torch, self.F, self.util, self.nets, self.optimizer, x_spoof, x_live, label,
                                         eps, self.lam, self.criterion_type, self.criterionL2, epoch,
                                         on_chunk=on_chunk if keep_visual else None, before_step=before_step)
        except torch.OutOfMemoryError as error:
            raise e06c_runner.TrainingStop(
                f'CUDA OOM at epoch {epoch} iteration {iteration} (B={B}): {error}') from error
        gate(self.optimizer_applications == before_apps + 1, 'exactly one optimizer.step() per global batch')
        gate(run['chunk_sizes'] == chunks and run['global_batch'] == B, 'executed chunk plan')
        self.backward_calls += len(chunks)
        if keep_visual:
            gate(len(recs) == len(chunks), 'one reconstruction per chunk')
            size = (128, 128)
            self.visual = {'epoch': epoch, 'global_step': self.global_step + 1, 'batch_size': B, 'grids': {
                'img_spoof': self.F.interpolate(x_spoof, size=size, mode='bilinear'),
                'img_live': self.F.interpolate(x_live, size=size, mode='bilinear'),
                'rec_spoof': torch.cat([r[0] for r in recs]), 'rec_live': torch.cat([r[1] for r in recs])}}
        losses = run['global_losses']
        total = tg.total_loss(losses, epoch)
        gate(rio.finite([float(v) for v in losses.values()] + [float(total)]), 'finite global losses')
        self.global_step += 1
        mem = torch.cuda.max_memory_allocated() if self.device == 'cuda' else None
        reserved = torch.cuda.max_memory_reserved() if self.device == 'cuda' else None
        self.peak_step_gpu_memory_bytes = max(self.peak_step_gpu_memory_bytes, mem or 0)
        pair_keys = [f'{s}>{l}' for s, l in zip(spoof_ids, live_ids)]
        return {'epoch': epoch, 'global_step': self.global_step, 'iteration': iteration, 'batch_size': B,
                'chunk_sizes': chunks, 'indices': indices, 'spoof_ids': spoof_ids, 'live_ids': live_ids,
                'labels': label.tolist(), 'batch_pair_sha256': rio.order_sha256(pair_keys),
                'epsilon_sha256': eps_sha, 'losses': {k: float(v) for k, v in losses.items()},
                'total_loss': float(total), 'step_seconds': time.monotonic() - t0, 'gpu_memory_bytes': mem,
                'gpu_reserved_bytes': reserved, 'owned_gradients': seen['owned'],
                'netCls_grad_nonzero': seen['netCls_grad_nonzero'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'input_range': [lo, hi]}

    def run_epoch(self, epoch, ctx=None, t_start=None):
        """One full pass: 15 x 240 + 120, one trajectory record per global optimizer step, every spoof row once."""
        self.set_modes()
        sizes, order = [], []
        t_start = time.monotonic() if t_start is None else t_start
        last = len(self.loader) - 1
        visual_epoch = rio.is_visualization_epoch(epoch, self.visualization_every)
        for iteration, batch in enumerate(self.loader):
            r = self.global_step_run(batch, epoch, iteration, keep_visual=visual_epoch and iteration == last)
            sizes.append(r['batch_size'])
            order.extend(r['spoof_ids'])
            if ctx is not None:
                ctx.log_epoch(rio.step_record(
                    epoch=epoch, global_step=r['global_step'], iteration=iteration, learning_rate=r['learning_rate'],
                    losses=r['losses'], total_loss=r['total_loss'], wall_clock_seconds=time.monotonic() - t_start,
                    gpu_memory_bytes=r['gpu_memory_bytes'], batch_size=r['batch_size'], chunk_sizes=r['chunk_sizes'],
                    batch_pair_sha256=r['batch_pair_sha256'], epsilon_sha256=r['epsilon_sha256'], mode=self.mode,
                    seed=self.seed, extra={'method_id': METHOD_ID, 'step_seconds': r['step_seconds'],
                                           'owned_gradients_finite': r['owned_gradients']['finite'],
                                           'netCls_grad_nonzero': r['netCls_grad_nonzero'],
                                           'optimizer_applications_total': self.optimizer_applications,
                                           'missing_field_reasons': dict(MISSING_REASONS)}))
        plan = v2.epoch_batch_plan(len(self.records))
        e06c_runner.gate(sizes == plan['batch_sizes'] == [GLOBAL_BATCH] * 15 + [120], f'epoch plan {sizes}')
        ids = {s[0] for s in self.records.spoof}
        e06c_runner.gate(len(order) == len(ids) == len(set(order)) and set(order) == ids,
                         'every TRAIN spoof row exactly once (no drop, no duplicate)')
        self.completed_epoch = epoch
        summary = {'epoch': epoch, 'optimizer_steps': len(sizes), 'rows': len(order),
                   'epoch_spoof_order_sha256': rio.order_sha256(order), 'global_step_end': self.global_step}
        if ctx is not None:
            ctx.log_event('e06b_epoch_complete', summary)
        return summary

    def checkpoint_event(self, epoch, ctx):
        """Pinned misc/util.py::save_checkpoint for netE_spoof/netE_live/netG (fresh-only: no resume sidecar)."""
        require(rio.is_checkpoint_epoch(epoch) and ctx.mode == SCIENTIFIC, 'official checkpoint epoch, scientific only')
        written = []
        for model, prefix in rio.OFFICIAL_FILES:
            self.util.save_checkpoint(str(ctx.run_dir / 'checkpoints') + '/', self.models[model], epoch, 0, prefix)
            rel = 'checkpoints/' + rio.official_basename(prefix, epoch)
            kind, selected = rio.checkpoint_kind(model, epoch)
            p = ctx.run_dir / rel
            meta = dict(path=rel, epoch=epoch, global_step=self.global_step, file_size_bytes=p.stat().st_size,
                        sha256=sha256_file(p), checkpoint_type=kind, selected_for_final=selected,
                        selection_reason=ctx.config['checkpoint']['rule'] if selected else
                        'Official cadence; not final selection')
            ctx.record_checkpoint(**meta)
            written.append(dict(meta, model=model))
        ctx.log_event('e06b_checkpoint_event', {'epoch': epoch, 'global_step': self.global_step,
                                                'official': [{k: w[k] for k in ('path', 'sha256', 'checkpoint_type')}
                                                             for w in written], 'resume_sidecar': None})
        return written

    def visualize(self, epoch, ctx):
        """The unchanged E06c visualization block, with its event recorded under the E06b name."""
        return super().visualize(epoch, _EventRename(ctx))


class _EventRename:
    """ctx proxy: identical run directory and writers; only the reused engine's event names become e06b_*."""

    def __init__(self, ctx):
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def log_event(self, name, payload):
        return self._ctx.log_event(name.replace('e06c_', 'e06b_', 1), payload)


MISSING_REASONS = {
    'train_metrics': 'Upstream DSDG (train_generator.py) exposes no separate train metric; none is fabricated.',
    'val_losses': 'No VAL pass exists in the pinned trainer or this runner; VAL is never opened.',
    'val_metrics': 'No VAL pass exists in the pinned trainer or this runner; VAL is never opened.'}


def environment_record(torch):
    env, reasons = e06c_runner.environment_record(torch)       # same environment of record (gpat-m6-e06c)
    env['gpu_note'] = ('single RTX 3090; physical B=240 does not fit (E06c M6D5b); '
                       'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 240 = 12 x 20, tail 120 = 6 x 20 '
                       '(M6F-C: FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY)')
    reasons = {'tensorflow_version': 'E06b is a PyTorch method; TensorFlow is not part of gpat-m6-e06c'}
    return env, reasons


def build_models(torch, modules, source, config):
    """Pinned define_G(hdim, attack_type=2) / define_IP + LightCNN load, via the M6F-C builder (Cls(128, 2))."""
    from methods.dsdg import native_qualification as m6fc
    from methods.dsdg import runtime as m6d5a
    root = Path(source['root']) / config['source']['relevant_path']
    models, binding = m6fc.build_models(torch, modules['networks'], root, config)
    lightcnn = m6d5a.load_lightcnn(torch, models['netIP'])
    return models, binding, lightcnn


# ================================================================= scientific entry (tools/run_e06b.py)
def run_scientific(*, seed, runtime_root, faces_root, command_line=None):
    """Full 200-epoch SCIENTIFIC run, fresh only. Never called by M6F-D (qualification uses its own harness)."""
    import torch
    import torch.nn.functional as F
    from methods.common.learned_runlog import LearnedRunContext
    from methods.common.runlog import git_dirty
    from methods.common.upstream import upstream_modules
    from methods.dsdg import runtime as m6d5a
    from methods.dsdg import training_qualification as m6d5b
    mode = SCIENTIFIC
    validate_mode_seed(mode, seed)
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    adapter = native.DSDGNativeAdapter()
    config = verify_contract(adapter.config)
    ids = identities(config)
    precision = e06c_runner.configure_precision(torch)
    seeding = _seed_scientific(torch, seed, config)
    source = verify_source(adapter, ids)
    lightcnn = lightcnn_identity(m6d5a.LIGHTCNN)
    rows, relation, sample_datasets, population = load_relation(config)
    reader = rio.CanonicalFaceReader(faces_root, sample_datasets, datasets=native.DATASETS)
    dataset = E06bDataset(relation, reader)
    env, reasons = environment_record(torch)
    run_dir = run_root(runtime_root, mode, seed)
    require(not run_dir.exists(), f'{run_dir} exists; E06b scientific runs are fresh only (resume is not qualified)')
    ctx = LearnedRunContext(method_id=METHOD_ID, seed=seed, config=config, runtime_root=runtime_root,
                            command_line=command_line, resume=False, environment=env,
                            missing_environment_reasons=reasons)
    with upstream_modules(Path(source['root']) / config['source']['relevant_path'], m6d5a.UPSTREAM_MODULES,
                          m6d5a.UPSTREAM_ROOTS) as modules:
        models, binding, lightcnn_load = build_models(torch, modules, source, config)
        optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                       config['optimizer']['learning_rate'])
        m6d5b.optimizer_evidence(torch, optimizer, models)
        loader, generator, loader_evidence = build_loader(torch, dataset, mode, seed, config)
        trainer = E06bTrainer(torch, F, modules['misc.util'], models, optimizer, loader, generator, relation, mode,
                              seed, dict(LAMBDAS), visualization_every=rio.visualization_every(config))
        with ctx:
            streams = rio.tee_run_logs(ctx.run_dir)
            try:
                ctx.log_event('e06b_run_start', {'seeding': seeding, 'precision': precision, 'identities': {
                    k: v for k, v in ids.items() if k != 'source_files_sha256'}, 'loader': loader_evidence,
                    'population': population, 'lightcnn': lightcnn, 'binding': binding,
                    'execution_mode': EXECUTION_MODE, 'runtime_fidelity_assessment': RUNTIME_FIDELITY})
                t0 = time.monotonic()
                for epoch in range(1, ALL_EPOCHS + 1):
                    trainer.run_epoch(epoch, ctx, t0)
                    if rio.is_visualization_epoch(epoch, trainer.visualization_every):
                        trainer.visualize(epoch, ctx)            # source order: visualization, then checkpoint
                    if rio.is_checkpoint_epoch(epoch):
                        trainer.checkpoint_event(epoch, ctx)
                final = ctx.run_dir / 'checkpoints' / 'netG_model_epoch_200_iter_0.pth'
                ctx.close(completion_status='completed', summary={
                    'final_or_selected_checkpoint_path': str(final),
                    'final_or_selected_checkpoint_sha256': sha256_file(final),
                    'checkpoint_selection_rule': config['checkpoint']['rule'],
                    'training_duration_seconds': time.monotonic() - t0,
                    'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                    'optimizer_applications': trainer.optimizer_applications,
                    'missing_field_reasons': {
                        'seed_level_evaluation_metrics': 'Downstream evaluation is a later milestone'}})
            finally:
                rio.untee(streams)
    return ctx.run_dir


def _seed_scientific(torch, seed, config):
    from methods.common.learned import apply_framework_seed
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    plan = apply_framework_seed(config, seed, 'torch', cuda=True)
    return {'seed': seed, 'role': 'EXPERIMENT_SEED', 'plan': 'methods.common.learned.apply_framework_seed',
            'cudnn_deterministic': plan['cudnn_deterministic']}


def seed_qualification(torch, seed):
    """The same calls apply_framework_seed makes, with the qualification seed (never an experiment seed)."""
    import random
    validate_mode_seed(QUALIFICATION, seed)
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    return {'seed': seed, 'role': 'QUALIFICATION_SEED_NOT_EXPERIMENT_SEED'}
