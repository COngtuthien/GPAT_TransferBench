"""M7C4 GPAT production runner: B1/B3 attack warmup + B0..B3 generator training (scientific and qualification modes).

Authorities: frozen spec, A1-A10, M7B configs, M7C2a runtime contract (N-04 order, N-05/N-06/N-07 schedules, EMA,
DEV-022), M7C2b static core (all formulas; nothing is re-implemented here) and M7C3 (gpat-m7-gpu, exact-forward
FaceXFormer adapter, deterministic CUDA, AMP boundaries).

Generator update (one optimizer group = 2 physical microbatches of 4; tail [4, 2]; u = 1..66300):
  lr = main_lr(u) for G_OPT and D_OPT; s_hf / lambda_adv / lambda_con / lambda_spec = curriculum(u)
  per microbatch j (weight w_j = m_j / n_group):
    forward GPAT once (fp16 autocast for the neural modules; fp32 composition/x_hat)
    G side, D parameters gradient-disabled, x_hat NOT detached, frozen fp32 teachers (x_t targets under no_grad):
      L_G_j = w_j * assemble(L_id, L_lm, L_parse, L_low, L_artcon, L_spec, L_Gadv, [L_type], L_budget, L_TV, L_bg)
              + lambda_idadv * idadv_share(CE_sum_j, labelled_count_of_group)       (DEV-022; never times w_j)
      G_SCALER.scale(L_G_j).backward()
    D side, D trainable: L_D_j = w_j * 0.5 * (BCE(D(x_source), 1) + BCE(D(x_hat.detach()), 0));
      D_SCALER.scale(L_D_j).backward()
  boundary (FAIL_CLOSED_AMP_OVERFLOW, owner M7C4): D unscale, G unscale, then every D- and G-owned gradient is
  scanned; any NaN/Inf -> AmpOverflowStop BEFORE either optimizer steps (no step, no update-index/schedule advance,
  no EMA, no recovery). Only with all gradients finite: D clip 1.0 -> step -> update; G clip 1.0 -> step -> update;
  post-step parameter finiteness (else NumericalPostStepStop = FAILED_NUMERICAL_POST_STEP); zero both; EMA update
  (E_art, G_res) when active (u > 5525; initialized at the end of epoch 5). The D/G group is one atomic unit.
  AMP policy (M7D1, configs/amendments/gpat_m7d1_amp_retry_resolution.yaml): the production Trainer runs
  ATOMIC_AMP_BACKOFF_RETRY. Before a group the mutable state a never-stepped attempt can change is preserved (all
  module buffers incl. E_art BN running stats, Python/NumPy/CPU/CUDA RNG, both scaler states). If the boundary scan
  finds a non-finite unscaled gradient, neither optimizer steps, the pre-group state is restored, gradients are zeroed,
  ONLY each offending scaler backs off (x0.5, growth tracker 0, i.e. standard GradScaler backoff) and the SAME group
  is recomputed at the SAME update index (an `amp_retry` event, never an optimizer update; data are not re-read or
  re-logged). An offending scaler already at scale <= 1.0 -> AmpOverflowFinalStop (FAIL_CLOSED_AMP_OVERFLOW_FINAL).
  The M7C4 policy FAIL_CLOSED_AMP_OVERFLOW (stop at the first overflow) remains the step default and the fallback.
Epoch end: counts verified; EMA init at epoch 5; EMA candidate at epochs 10..60; recovery/latest.pt; epoch record.
Warmup (B1/B3): 10 epochs x 139 batches of 64 TRAIN source spoof frames, AdamW(E_art + attack head, lr
attack_warmup_lr(s), wd 1e-4), fp16 forward, fp32 CE, WARMUP_SCALER, clip 1.0, BN train. Handoff carries E_art +
attack head (incl. BN buffers) only; WARMUP_OPT / WARMUP_SCALER are discarded; G_OPT / G_SCALER start fresh.
Never loaded: VAL/TEST metadata or images, val_pairs, ArtifactProbe, any selector. Fail closed on non-finite loss or
parameter, and on any non-finite gradient after unscale (AMP overflow is never a silently skipped step; the last
safe-boundary recovery stays the only resumable authority). SIGINT/SIGTERM: the current optimizer group completes,
then recovery is written at that boundary.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpatbench.preprocess import aux_models
from methods.common.runlog import atomic_write_json, atomic_write_yaml, git_commit, git_dirty
from methods.gpat import composition, losses, runtime_contract as rc, spectral, teacher_preprocess as tp, wavelet
from methods.gpat import runner_checkpoint as ck
from methods.gpat import runner_io as rio
from methods.gpat.artifact_encoder import encoder_input, imagenet_state_dict
from methods.gpat.config import load_config
from methods.gpat.ema import ModelEMA
from methods.gpat.highpass import highpass
from methods.gpat.model import GPATCore

DEV = 'cuda'
FX_PARSE, FX_LM = aux_models.FX_TASKS['parsing'], aux_models.FX_TASKS['landmarks']
CLIP = 1.0
BETAS = (0.5, 0.999)
WARMUP_WD = 1e-4
EMA_START_UPDATE = rc.EMA_START_UPDATE


class TrainingStop(RuntimeError):
    """Fail-closed stop: non-finite loss/parameter, identity/firewall/provenance violation."""


class NumericalStop(TrainingStop):
    """Numerical fail-closed stop; `record` holds the failure type, optimizer, offending parameters and scales."""

    def __init__(self, message, record):
        super().__init__('GPAT runner STOP: ' + message)
        self.record = record


class NonFiniteLossStop(NumericalStop):
    failure_type = 'NON_FINITE_LOSS'


class AmpOverflowStop(NumericalStop):
    failure_type = 'FAIL_CLOSED_AMP_OVERFLOW'


class AmpOverflowFinalStop(AmpOverflowStop):
    failure_type = 'FAIL_CLOSED_AMP_OVERFLOW_FINAL'


class NumericalPostStepStop(NumericalStop):
    failure_type = 'FAILED_NUMERICAL_POST_STEP'


# AMP overflow policies: M7C4 fail-closed (the step default and the fallback) and the M7D1 atomic backoff-and-retry
# (the production Trainer). GradScaler construction is unchanged (init 65536, growth 2, backoff 0.5, interval 2000).
AMP_FAIL_CLOSED = 'FAIL_CLOSED_AMP_OVERFLOW'
AMP_ATOMIC_RETRY = 'ATOMIC_AMP_BACKOFF_RETRY'
AMP_POLICIES = (AMP_FAIL_CLOSED, AMP_ATOMIC_RETRY)
PRODUCTION_AMP_POLICY = AMP_ATOMIC_RETRY
AMP_BACKOFF = 0.5
AMP_MIN_SCALE = 1.0


def nonfinite(named_params, attr='grad'):
    """Names of parameters whose gradient (attr='grad') or value (attr=None) holds NaN/Inf."""
    bad = []
    for name, p in named_params:
        t = p.grad if attr == 'grad' else p
        if t is not None and not bool(torch.isfinite(t).all()):
            bad.append(name)
    return bad


def gate(ok, message):
    if not ok:
        raise TrainingStop('GPAT runner STOP: ' + message)


def finite(t):
    return bool(torch.isfinite(t).all())


def _digest(t):
    import hashlib
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _rng_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_rng_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_rng_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


class PreGroupState:
    """Mutable state that an attempt which never reaches an optimizer step can change: every module buffer of the
    core (E_art BatchNorm running_mean/running_var/num_batches_tracked in train mode, ...), the Python/NumPy/torch CPU/
    CUDA RNG and the GradScaler states. Parameters, optimizer states, EMA, schedule and position change only after the
    boundary scan has passed, so a failed attempt cannot touch them. `scalers` maps the optimizer name to its scaler."""

    def __init__(self, core, scalers):
        self.buffers = [(name, b, b.detach().clone()) for name, b in core.named_buffers()]
        self.rng = ck.rng_state()
        self.scalers = dict(scalers)
        self.scaler_states = {k: s.state_dict() for k, s in self.scalers.items()}

    def back_off(self, name, factor):
        """Standard GradScaler backoff of one offending scaler: scale * factor, growth tracker reset to 0."""
        st = self.scaler_states[name]
        st['scale'] = float(st['scale']) * factor
        st['_growth_tracker'] = 0
        return st['scale']

    def restore(self, params):
        for p in params:
            p.grad = None
        with torch.no_grad():
            for _, b, saved in self.buffers:
                b.copy_(saved)
        ck.restore_rng(self.rng)
        for k, s in self.scalers.items():
            st = self.scaler_states[k]
            s.update(new_scale=float(st['scale']))      # drops the failed attempt's per-optimizer unscale/inf records
            s.load_state_dict(st)
        self.verify(params)

    def verify(self, params):
        gate(all(p.grad is None for p in params), 'retry restore: gradients not cleared')
        gate(all(torch.equal(b, saved) for _, b, saved in self.buffers), 'retry restore: module buffers differ')
        gate(_rng_equal(ck.rng_state(), self.rng), 'retry restore: RNG state differs')
        for k, s in self.scalers.items():
            got, want = s.state_dict(), self.scaler_states[k]
            gate(got == want, f'retry restore: {k} scaler state {got} != {want}')
            gate(not s._per_optimizer_states, f'retry restore: {k} scaler kept per-optimizer records')


@contextlib.contextmanager
def fp32():
    with torch.autocast(device_type='cuda', enabled=False):
        yield


# ============================================================================= teachers
class Teachers:
    """Frozen fp32 eval teachers (M7C3-qualified assets): AdaFace, FaceXFormer (exact-forward adapter), F_art."""

    def __init__(self, assets, device=DEV):
        a = assets['assets']
        self.fx = aux_models.FaceXFormerAdapter(a['facexformer_code']['path'], a['facexformer_weight']['path'],
                                                device=device)
        self.ada = aux_models.AdaFaceAdapter(a['adaface_code']['path'], a['adaface_weight']['path'], device=device)
        import torchvision
        r18 = torchvision.models.resnet18(weights=None)
        r18.load_state_dict(imagenet_state_dict(a['f_art_resnet18_imagenet1k_v1']['path']))
        self.f_art = nn.Sequential(*list(r18.children())[:-1], nn.Flatten()).to(device)
        for m in (self.fx.model, self.ada.model, self.f_art):
            m.eval().requires_grad_(False)
        self.device = device

    def modules(self):
        return {'facexformer': self.fx.model, 'adaface': self.ada.model, 'f_art': self.f_art}

    def facexformer(self, x):
        with fp32():
            inp = tp.facexformer_input_exact(x.float())
            n = x.shape[0]
            seg = self.fx.model(inp, None, torch.full((n,), FX_PARSE, device=x.device))[7]
            lm = self.fx.model(inp, None, torch.full((n,), FX_LM, device=x.device))[0].view(n, 68, 2)
        return seg.float(), lm.float()

    def adaface(self, x):
        with fp32():
            feat, _ = self.ada.model(tp.adaface_input(x.float()))
            return F.normalize(feat.float(), dim=1)

    def art(self, x):
        with fp32():
            return self.f_art(highpass(x.float())).float()


# ============================================================================= optimizers
def generator_optimizer(core):
    params = [p for m in core.generator_modules().values() for p in m.parameters()]
    return torch.optim.Adam(params, lr=0.0, betas=BETAS, weight_decay=0.0)


def discriminator_optimizer(core):
    return torch.optim.Adam(core.discriminator.parameters(), lr=0.0, betas=BETAS, weight_decay=0.0)


def warmup_optimizer(core):
    gate(core.attack_head is not None, 'warmup exists only for B1/B3')
    params = list(core.e_art.parameters()) + list(core.attack_head.parameters())
    return torch.optim.AdamW(params, lr=rc.ATTACK_WARMUP_LR, weight_decay=WARMUP_WD)


def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


def grad_norm(params):
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.norm(torch.stack(gs))) if gs else 0.0


# ============================================================================= generator step
class GeneratorStep:
    """One optimizer group under the N-04 pre-update joint-gradient scheme. Owns no data, no checkpoint."""

    def __init__(self, core, teachers, cfg, g_opt, d_opt, g_scaler, d_scaler, ema=None, amp_policy=AMP_FAIL_CLOSED):
        gate(amp_policy in AMP_POLICIES, f'AMP policy {amp_policy!r}')
        self.amp_policy = amp_policy
        self.on_retry = None                            # callable(event) per amp_retry (Trainer: metrics.jsonl)
        self.active = False                             # True while a group (incl. its retries) is in flight
        self.core, self.t, self.cfg = core, teachers, cfg
        self.g_opt, self.d_opt, self.g_scaler, self.d_scaler = g_opt, d_opt, g_scaler, d_scaler
        self.ema = ema                                  # {'e_art': ModelEMA, 'g_res': ModelEMA} once active
        self.g_named = [(f'{k}.{n}', p) for k, m in core.generator_modules().items() for n, p in m.named_parameters()]
        self.d_named = [(f'discriminator.{n}', p) for n, p in core.discriminator.named_parameters()]
        self.g_params = [p for _, p in self.g_named]
        self.d_params = [p for _, p in self.d_named]
        self.capture = None                             # qualification: per-microbatch records (None = off)
        self.inspect = None                             # qualification: callable(stage, step) after unscale_

    def components(self, mb, out, cur):
        x_s, x_t = mb['x_source'], mb['x_target']
        with torch.no_grad():                           # detached x_t targets / source features
            seg_t, lm_t = self.t.facexformer(x_t)
            id_t = self.t.adaface(x_t)
            art_s, art_t = self.t.art(x_s), self.t.art(x_t)
            face = losses.face_mask_dilated(seg_t.argmax(dim=1))
        seg_h, lm_h = self.t.facexformer(out.x_hat)
        comps = {'id': losses.l_id(id_t, self.t.adaface(out.x_hat)), 'lm': losses.l_lm(lm_t, lm_h),
                 'parse': losses.l_parse(seg_t, seg_h), 'low': losses.l_low(out.x_hat, out.target_bands[0]),
                 'artcon': losses.l_artcon(self.t.art(out.x_hat), art_s, art_t),
                 'spec': spectral.spec_loss(out.x_hat, x_s), 'budget': losses.l_budget(out.A),
                 'tv': losses.l_tv(out.M, out.delta_LH, out.delta_HL, out.delta_HH),
                 'bg': losses.l_bg(out.x_hat, x_t, face)}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            d_fake_for_g = self.core.discriminator(out.x_hat)
        comps['gadv'] = losses.l_gadv(d_fake_for_g)
        if self.cfg.attack_type_head:
            comps['type'] = losses.l_type(out.attack_logits, mb['attack'])
        return comps

    def __call__(self, group, u):
        """One scientific update u. Under ATOMIC_AMP_BACKOFF_RETRY an overflowed attempt is undone and the same group is
        recomputed at u with the offending scaler(s) backed off; exactly one attempt ever reaches the optimizer steps."""
        gate(not self.active, 're-entrant optimizer group')
        self.active = True
        try:
            if self.amp_policy == AMP_FAIL_CLOSED:
                return dict(self.attempt(group, u), amp_policy=self.amp_policy, amp_attempts=1)
            scalers = {'D_OPT': self.d_scaler, 'G_OPT': self.g_scaler}
            return run_atomic_retry(lambda: self.attempt(group, u), PreGroupState(self.core, scalers),
                                    list(self.core.parameters()), self, u, 'global_update_attempted')
        finally:
            self.active = False

    def attempt(self, group, u):
        cur = rc.curriculum(u)
        lr = rc.main_lr(u)
        set_lr(self.g_opt, lr)
        set_lr(self.d_opt, lr)
        sizes = [int(mb['x_source'].shape[0]) for mb in group]
        weights = [s / sum(sizes) for s in sizes]
        labelled = sum(int(mb['identity_valid'].sum()) for mb in group) if self.cfg.identity_adversary else 0
        self.g_opt.zero_grad(set_to_none=True)
        self.d_opt.zero_grad(set_to_none=True)
        sums = {}
        d_real = d_fake = 0.0
        t0 = time.monotonic()
        for j, mb in enumerate(group):
            self.core.discriminator.requires_grad_(False)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = self.core(mb['x_source'], mb['x_target'], scale_hf=cur['s_hf'])
            comps = self.components(mb, out, cur)
            total, record = losses.assemble_generator_loss(comps, cur, lambda_type=self.cfg.lambda_type,
                                                           lambda_idadv=0.0)
            l_g = weights[j] * total
            if self.cfg.identity_adversary:
                ce_sum, _ = losses.idadv_microbatch(out.identity_logits, mb['identity'], mb['identity_valid'])
                share = losses.idadv_share(ce_sum, labelled)
                comps['idadv_share'] = share
                l_g = l_g + self.cfg.lambda_idadv * share
            if not (finite(l_g) and all(finite(v) for v in comps.values())):
                raise NonFiniteLossStop(f'non-finite G loss at update {u}', {
                    'failure_type': NonFiniteLossStop.failure_type, 'global_update_attempted': u, 'microbatch': j,
                    'side': 'G', 'components': {k: float(v) for k, v in comps.items()}})
            self.g_scaler.scale(l_g).backward()
            self.core.discriminator.requires_grad_(True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                real_logits = self.core.discriminator(mb['x_source'])
                fake_logits = self.core.discriminator(out.x_hat.detach())
            lr_, lf_ = losses.l_d_terms(real_logits, fake_logits)
            l_d = losses.l_d(real_logits, fake_logits)
            if not finite(l_d):
                raise NonFiniteLossStop(f'non-finite D loss at update {u}', {
                    'failure_type': NonFiniteLossStop.failure_type, 'global_update_attempted': u, 'microbatch': j,
                    'side': 'D', 'value': float(l_d)})
            self.d_scaler.scale(weights[j] * l_d).backward()
            d_real += weights[j] * float(lr_)
            d_fake += weights[j] * float(lf_)
            for k, v in comps.items():
                sums[k] = sums.get(k, 0.0) + (float(v) if k == 'idadv_share' else weights[j] * float(v))
            sums['G_total'] = sums.get('G_total', 0.0) + float(l_g)
            if self.capture is not None:
                self.capture.append({'index': mb['index'].tolist(), 'x_source': _digest(mb['x_source']),
                                     'x_target': _digest(mb['x_target']), 'x_hat': _digest(out.x_hat),
                                     'l_g': float(l_g), 'l_d': float(l_d), 'weight': weights[j],
                                     'components': {k: float(v) for k, v in comps.items()}})
        # ---- boundary: D first, then G (both gradients from the same pre-update state)
        scales = (float(self.d_scaler.get_scale()), float(self.g_scaler.get_scale()))
        self.d_scaler.unscale_(self.d_opt)
        if self.inspect is not None:
            self.inspect('D', self)
        self.g_scaler.unscale_(self.g_opt)
        if self.inspect is not None:
            self.inspect('G', self)
        bad_d, bad_g = nonfinite(self.d_named), nonfinite(self.g_named)
        if bad_d or bad_g:                              # atomic D/G group: neither optimizer may step
            raise AmpOverflowStop(f'non-finite gradient after unscale at update {u}', {
                'failure_type': AmpOverflowStop.failure_type, 'global_update_attempted': u,
                'offending_optimizers': [n for n, bad in (('D_OPT', bad_d), ('G_OPT', bad_g)) if bad],
                'offending_parameter_count': {'D_OPT': len(bad_d), 'G_OPT': len(bad_g)},
                'offending_parameters': {'D_OPT': bad_d[:50], 'G_OPT': bad_g[:50]},
                'D_scale': scales[0], 'G_scale': scales[1], 'optimizer_steps_taken': 0})
        d_norm = float(torch.nn.utils.clip_grad_norm_(self.d_params, CLIP))
        self.d_scaler.step(self.d_opt)
        self.d_scaler.update()
        g_norm = float(torch.nn.utils.clip_grad_norm_(self.g_params, CLIP))
        self.g_scaler.step(self.g_opt)
        self.g_scaler.update()
        gate(float(self.d_scaler.get_scale()) >= scales[0] and float(self.g_scaler.get_scale()) >= scales[1],
             f'a scaler backed off at update {u} although every gradient was finite')
        self.g_opt.zero_grad(set_to_none=True)
        self.d_opt.zero_grad(set_to_none=True)
        bad_pd, bad_pg = nonfinite(self.d_named, None), nonfinite(self.g_named, None)
        if bad_pd or bad_pg:
            raise NumericalPostStepStop(f'non-finite parameter after update {u}', {
                'failure_type': NumericalPostStepStop.failure_type, 'global_update_attempted': u,
                'offending_parameters': {'D_OPT': bad_pd[:50], 'G_OPT': bad_pg[:50]},
                'D_scale': scales[0], 'G_scale': scales[1], 'optimizer_steps_taken': 2})
        ema_updated = False
        if self.ema is not None and u > EMA_START_UPDATE:
            self.ema['e_art'].update_from(self.core.e_art)
            self.ema['g_res'].update_from(self.core.g_res)
            ema_updated = True
        return {'global_update': u, 'microbatch_sizes': sizes, 'group_samples': sum(sizes), 'sample_weights': weights,
                'labelled_identity_count': labelled, 'learning_rate': lr, 'scale_hf': cur['s_hf'],
                'lambda_adv': cur['lambda_adv'], 'lambda_con': cur['lambda_con'], 'lambda_spec': cur['lambda_spec'],
                'train_losses': {**{k: v for k, v in sums.items()}, 'D_real': d_real, 'D_fake': d_fake,
                                 'D_total': 0.5 * (d_real + d_fake)},
                'G_grad_norm': g_norm, 'D_grad_norm': d_norm, 'D_scale_before': scales[0],
                'G_scale_before': scales[1], 'D_scale': float(self.d_scaler.get_scale()),
                'G_scale': float(self.g_scaler.get_scale()),
                'group_status': 'COMPLETE', 'ema_updated': ema_updated,
                'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                'wall_seconds': time.monotonic() - t0}


# ============================================================================= warmup step
def run_atomic_retry(attempt, pre, params, step, index, index_key):
    """ATOMIC_AMP_BACKOFF_RETRY loop shared by the generator group (D_OPT/G_OPT) and the warmup step (WARMUP_OPT).
    AmpOverflowStop is raised only by the boundary scan, i.e. before any optimizer step, scaler update, position or
    schedule advance, EMA update or checkpoint; any other stop propagates unchanged."""
    events = []
    while True:
        n_capture = len(step.capture) if getattr(step, 'capture', None) is not None else None
        try:
            rec = attempt()
        except AmpOverflowFinalStop:
            raise
        except AmpOverflowStop as exc:
            if n_capture is not None:
                del step.capture[n_capture:]                   # the failed attempt's records are not group records
            r = exc.record
            offending = list(r['offending_optimizers'])
            gate(offending and set(offending) <= set(pre.scalers), 'retry: offending optimizer set')
            gate(r['optimizer_steps_taken'] == 0, 'retry: an optimizer stepped in an overflowed attempt')
            old = {k: float(pre.scalers[k].get_scale()) for k in offending}
            gate(all(old[k] == float(pre.scaler_states[k]['scale']) for k in offending), 'retry: scale bookkeeping')
            if any(old[k] <= AMP_MIN_SCALE for k in offending):
                raise AmpOverflowFinalStop(f'non-finite gradient after unscale at {index_key} {index} with the offending '
                                           f'scale at {AMP_MIN_SCALE}', {
                                               **r, 'failure_type': AmpOverflowFinalStop.failure_type,
                                               'amp_policy': AMP_ATOMIC_RETRY, 'amp_retries': len(events),
                                               'scale_at_final_attempt': old}) from exc
            for k in offending:
                gate(pre.scalers[k].get_backoff_factor() == AMP_BACKOFF, f'{k} backoff factor')
            new = {k: pre.back_off(k, AMP_BACKOFF) for k in offending}
            pre.restore(params)
            event = {index_key: index, 'retry_number': len(events) + 1, 'amp_policy': AMP_ATOMIC_RETRY,
                     'offending_optimizers': offending,
                     'offending_parameter_count': {k: r['offending_parameter_count'][k] for k in offending},
                     'offending_parameters': {k: r['offending_parameters'][k] for k in offending},
                     'old_scale': old, 'new_scale': new,
                     'scales_after_restore': {k: float(s.get_scale()) for k, s in pre.scalers.items()},
                     'optimizer_steps_taken': 0, 'optimizer_update': False, 'pre_group_state_restored': True}
            events.append(event)
            if step.on_retry is not None:
                step.on_retry(event)
            continue
        return dict(rec, amp_policy=AMP_ATOMIC_RETRY, amp_attempts=len(events) + 1)


class WarmupStep:
    def __init__(self, core, optimizer, scaler, amp_policy=AMP_FAIL_CLOSED):
        gate(amp_policy in AMP_POLICIES, f'AMP policy {amp_policy!r}')
        self.amp_policy = amp_policy
        self.on_retry = None
        self.active = False
        self.core, self.opt, self.scaler = core, optimizer, scaler
        self.named = ([(f'e_art.{n}', p) for n, p in core.e_art.named_parameters()] +
                      [(f'attack_head.{n}', p) for n, p in core.attack_head.named_parameters()])
        self.params = [p for _, p in self.named]
        self.inspect = None                             # qualification: callable(step) after unscale_

    def __call__(self, batch, s):
        gate(not self.active, 're-entrant warmup step')
        self.active = True
        try:
            if self.amp_policy == AMP_FAIL_CLOSED:
                return dict(self.attempt(batch, s), amp_policy=self.amp_policy, amp_attempts=1)
            return run_atomic_retry(lambda: self.attempt(batch, s), PreGroupState(self.core, {'WARMUP_OPT': self.scaler}),
                                    list(self.core.parameters()), self, s, 'warmup_step_attempted')
        finally:
            self.active = False

    def attempt(self, batch, s):
        lr = rc.attack_warmup_lr(s)
        set_lr(self.opt, lr)
        self.core.e_art.train()
        self.opt.zero_grad(set_to_none=True)
        t0 = time.monotonic()
        x_s = batch['x_source']
        bands = wavelet.dwt(x_s)
        inp = encoder_input(x_s, *bands[1:])
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, _, z_a = self.core.e_art(inp)
            logits = self.core.attack_head(z_a)
        ce = losses.l_type(logits, batch['attack'])
        if not finite(ce):
            raise NonFiniteLossStop(f'non-finite warmup CE at step {s}', {
                'failure_type': NonFiniteLossStop.failure_type, 'warmup_step_attempted': s, 'value': float(ce)})
        scale = float(self.scaler.get_scale())
        self.scaler.scale(ce).backward()
        self.scaler.unscale_(self.opt)
        if self.inspect is not None:
            self.inspect(self)
        bad = nonfinite(self.named)
        if bad:
            raise AmpOverflowStop(f'non-finite warmup gradient after unscale at step {s}', {
                'failure_type': AmpOverflowStop.failure_type, 'warmup_step_attempted': s,
                'offending_optimizers': ['WARMUP_OPT'], 'offending_parameter_count': {'WARMUP_OPT': len(bad)},
                'offending_parameters': {'WARMUP_OPT': bad[:50]}, 'WARMUP_scale': scale, 'optimizer_steps_taken': 0})
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, CLIP))
        self.scaler.step(self.opt)
        self.scaler.update()
        gate(float(self.scaler.get_scale()) >= scale, f'WARMUP_SCALER backed off at step {s} with finite gradients')
        self.opt.zero_grad(set_to_none=True)
        bad = nonfinite(self.named, None)
        if bad:
            raise NumericalPostStepStop(f'non-finite parameter after warmup step {s}', {
                'failure_type': NumericalPostStepStop.failure_type, 'warmup_step_attempted': s,
                'offending_parameters': {'WARMUP_OPT': bad[:50]}, 'optimizer_steps_taken': 1})
        return {'warmup_step': s, 'batch_size': int(x_s.shape[0]), 'learning_rate': lr, 'cross_entropy': float(ce),
                'grad_norm': norm, 'scale_before': scale, 'scale': float(self.scaler.get_scale()),
                'step_status': 'COMPLETE',
                'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()), 'wall_seconds': time.monotonic() - t0}


def handoff(core):
    """Warmup -> generator stage: E_art + attack head (with BN buffers) carry in place; fresh G_OPT / G_SCALER."""
    return generator_optimizer(core), torch.amp.GradScaler('cuda')


# ============================================================================= process setup
def build_core(cfg, assets, seed):
    torch.manual_seed(seed)
    core = GPATCore(cfg, weight_path=assets['assets']['e_art_resnet18_imagenet1k_v1']['path'],
                    with_discriminator=True).to(DEV)
    core.train()
    return core


def to_device(batch):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def provenance(cfg, assets):
    return {'method': cfg.method, 'config_sha256': cfg.sha256, 'code_commit': git_commit(),
            'gpu_env_lock_sha256': rio.GPU_LOCK_SHA256, 'm7c3_record_sha256': rio.M7C3_RECORD_SHA256,
            'source_manifest_sha256': rio.RELATION_SHA256,
            'teacher_sha256': {k: v['sha256'] for k, v in assets['assets'].items() if 'sha256' in v},
            'gamma': cfg.gamma, 'ema_decay': rc.EMA_DECAY, 'curriculum_id': rc.CURRICULUM_ID}


class StopFlag:
    """SIGINT/SIGTERM request a stop at the next optimizer boundary; nothing mid-accumulation is ever saved."""

    def __init__(self):
        self.requested = False
        self._old = {}

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        self.requested = True

    def restore(self):
        for sig, h in self._old.items():
            signal.signal(sig, h)


# ============================================================================= run context (run_logging_v1 layout)
class GPATRunContext:
    """<runtime_root>/runs/m7_a1/<E08..E11>/seed_<seed>/ (scientific, M7D1-A1) or <runtime_root>/qualification/m7/M7C4/... ;
    run_logging_v1 file names; append-only metrics.jsonl; atomic JSON/YAML; checkpoint_index.json of every write."""

    FILES = {'resolved_config': 'resolved_config.yaml', 'run_manifest': 'run_manifest.json',
             'metrics': 'metrics.jsonl', 'checkpoint_index': 'checkpoint_index.json',
             'run_summary': 'run_summary.json', 'identity_map': 'identity_class_map.json',
             'stdout': 'stdout.log', 'stderr': 'stderr.log'}

    def __init__(self, *, mode, method, seed, runtime_root, cfg, provenance_record, label=None, resume=False):
        rio.validate_mode_seed(mode, seed)
        self.mode, self.method, self.seed, self.cfg = mode, method, seed, cfg
        self.run_dir = rio.run_root(runtime_root, mode, method, seed, label)
        self.provenance = provenance_record
        self.run_id = rio.run_id(method, seed, cfg.sha256, provenance_record['code_commit'])
        self.resume = resume
        self._fh = None

    def path(self, key):
        return self.run_dir / self.FILES[key]

    @property
    def ckpt_dir(self):
        return self.run_dir / 'checkpoints'

    def open(self):
        existed = self.run_dir.exists() and any(self.run_dir.iterdir())
        gate(not existed or self.resume, f'run root exists and is never overwritten: {self.run_dir}')
        if existed:
            manifest = json.loads(self.path('run_manifest').read_text())
            gate(manifest['run_id'] == self.run_id and manifest['config_sha256'] == self.cfg.sha256,
                 'resume refused: run identity differs')
        (self.ckpt_dir / 'ema_candidates').mkdir(parents=True, exist_ok=True)
        (self.ckpt_dir / 'recovery').mkdir(parents=True, exist_ok=True)
        labels = list(rio.QUALIFICATION_LABELS) if self.mode == rio.QUALIFICATION else []
        if not existed:
            atomic_write_yaml(self.path('resolved_config'), {
                **json.loads(json.dumps(dict(_plain(self.cfg.raw)))),
                '_resolved': {'method': self.method, 'experiment_id': self.cfg.experiment_id, 'runner_mode': self.mode,
                              'experiment_seed': self.seed if self.mode == rio.SCIENTIFIC else None,
                              'qualification_seed': self.seed if self.mode == rio.QUALIFICATION else None,
                              'labels': labels, 'config_path': self.cfg.path, 'config_sha256': self.cfg.sha256,
                              'run_dir': str(self.run_dir), 'provenance': self.provenance,
                              'scientific_override_applied': False}})
            atomic_write_json(self.path('checkpoint_index'), {'method': self.method, 'run_id': self.run_id,
                                                             'labels': labels, 'checkpoints': []})
        atomic_write_json(self.path('run_manifest'), self._manifest('running'))
        for key in ('stdout', 'stderr'):
            self.path(key).touch(exist_ok=True)
        self._fh = self.path('metrics').open('a', encoding='utf-8')
        if existed:
            self.log('event', {'event': 'resume_reconciliation', 'policy': 'APPEND_OR_EXPLICITLY_RECONCILE'})
        return self

    def _manifest(self, status):
        return {'method_id': self.method, 'experiment_id': self.cfg.experiment_id, 'run_id': self.run_id,
                'runner_mode': self.mode, 'experiment_seed': self.seed if self.mode == rio.SCIENTIFIC else None,
                'qualification_seed': self.seed if self.mode == rio.QUALIFICATION else None,
                'labels': list(rio.QUALIFICATION_LABELS) if self.mode == rio.QUALIFICATION else [],
                'git_commit': self.provenance['code_commit'], 'git_dirty': git_dirty(),
                'config_path': self.cfg.path, 'config_sha256': self.cfg.sha256, 'logging_contract': 'run_logging_v1',
                'environment_lock_path': rio.GPU_LOCK, 'dependency_fingerprint': rio.GPU_LOCK_SHA256,
                'provenance': self.provenance, 'python_version': sys.version.split()[0],
                'pytorch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'command_line': ' '.join(sys.argv), 'completion_status': status, 'amp_policy': PRODUCTION_AMP_POLICY,
                'val_split_accessed': False, 'test_split_accessed': False}

    def log(self, record_type, record):
        def check(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    gate('test' not in str(k).lower() and str(k).lower() not in ('image', 'pixels', 'embedding'),
                         f'forbidden log field {k}')
                    check(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    check(v)
        check(record)
        self._fh.write(json.dumps({'record_type': record_type, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                   **record}, sort_keys=True, allow_nan=False) + '\n')
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record_checkpoint(self, entry):
        idx = json.loads(self.path('checkpoint_index').read_text())
        rel = os.path.relpath(entry['path'], self.run_dir)
        if entry.get('role') in ('recovery', 'warmup_recovery'):
            idx['checkpoints'] = [e for e in idx['checkpoints'] if e['path'] != rel]   # rolling latest.pt
        idx['checkpoints'].append({**entry, 'path': rel})
        atomic_write_json(self.path('checkpoint_index'), idx)

    def close(self, status, summary):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        atomic_write_json(self.path('run_manifest'), self._manifest(status))
        atomic_write_json(self.path('run_summary'), {
            'method_id': self.method, 'run_id': self.run_id, 'runner_mode': self.mode,
            'experiment_seed': self.seed if self.mode == rio.SCIENTIFIC else None,
            'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
            'checkpoint_selection_rule': 'A10 D14/D16 separate post-training VAL selection over 51 EMA candidates',
            'seed_level_evaluation_metrics': None, 'completion_status': status,
            'missing_field_reasons': {'final_or_selected_checkpoint_path': 'selection is a separate later process',
                                      'seed_level_evaluation_metrics': 'the training process never evaluates'},
            'val_split_accessed': False, 'test_split_accessed': False, **summary})


def _plain(node):
    if hasattr(node, 'items'):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, tuple):
        return [_plain(v) for v in node]
    return node


# ============================================================================= trainer (scientific + qualification)
class Trainer:
    def __init__(self, *, mode, method, seed, runtime_root, faces_root, assets, label=None, resume=False,
                 num_workers=None, load_teachers=True):
        rio.validate_mode_seed(mode, seed)
        self.mode, self.method, self.seed = mode, method, seed
        self.cfg = load_config(rio.variant_of(method)[0])
        self.assets = assets
        self.records = rio.read_relation()
        self.access = rio.AccessLog(self.records)
        self.imap = rio.identity_map(self.records) if self.cfg.identity_adversary else None
        ident = rio.identity_labels_for(self.records, self.imap) if self.imap is not None else None
        from methods.difffas.aux_runner_io import CanonicalFaceReader
        from methods.gpat import runner_data as rd
        reader = CanonicalFaceReader(faces_root, rio.sample_datasets(self.records))
        self.rd = rd
        self.gen_sampler, self.warm_sampler = rd.PlanSampler(), rd.PlanSampler()
        self.gen_loader = rd.make_loader(rd.GeneratorPairs(self.records, reader, ident), self.gen_sampler, seed,
                                         num_workers=num_workers)
        self.warm_loader = (rd.make_loader(rd.WarmupSources(self.records, reader), self.warm_sampler, seed,
                                           num_workers=num_workers) if self.cfg.attack_type_head else None)
        self.provenance = provenance(self.cfg, assets)
        self.ctx = GPATRunContext(mode=mode, method=method, seed=seed, runtime_root=runtime_root, cfg=self.cfg,
                                  provenance_record=self.provenance, label=label, resume=resume)
        self.core = build_core(self.cfg, assets, seed)
        self.teachers = Teachers(assets) if load_teachers else None      # warmup needs no teacher
        self.g_opt, self.d_opt = generator_optimizer(self.core), discriminator_optimizer(self.core)
        self.g_scaler, self.d_scaler = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
        self.ema = None
        self.step = GeneratorStep(self.core, self.teachers, self.cfg, self.g_opt, self.d_opt, self.g_scaler,
                                  self.d_scaler, amp_policy=PRODUCTION_AMP_POLICY)
        self.position = {'stage': 'warmup' if self.cfg.attack_type_head else 'generator', 'epoch': 1,
                         'next_group': 1, 'global_update': 0, 'warmup_epoch': 1, 'warmup_next_batch': 1,
                         'warmup_step': 0}
        self.warm = None
        if self.cfg.attack_type_head:
            self.w_opt, self.w_scaler = warmup_optimizer(self.core), torch.amp.GradScaler('cuda')
            self.warm = WarmupStep(self.core, self.w_opt, self.w_scaler, amp_policy=PRODUCTION_AMP_POLICY)

    # ---------------------------------------------------------------- data
    def log_access(self, batch, roles):
        for i in batch['index'].tolist():
            r = self.records[i]
            for role in roles:
                self.access.check(r[role], role)

    def generator_plan(self, epoch):
        return rio.groups(rio.epoch_permutation('generator', self.mode, self.seed, epoch))

    def warmup_plan(self, epoch):
        return rio.warmup_batches(rio.epoch_permutation('warmup', self.mode, self.seed, epoch))

    # ---------------------------------------------------------------- failure (fail closed)
    def last_safe_recovery(self, role):
        idx = json.loads(self.ctx.path('checkpoint_index').read_text())
        safe = [e for e in idx['checkpoints'] if e.get('role') == role]
        return safe[-1] if safe else None

    def record_failure(self, exc, *, stage, epoch, attempted, role):
        """Log the numerical failure; the position is NOT advanced and no checkpoint is written for this group."""
        safe = self.last_safe_recovery(role)
        self.ctx.log('numerical_failure', {
            **exc.record, 'stage': stage, 'epoch': epoch, 'attempted_group_or_batch': attempted,
            'position_unchanged': dict(self.position),
            'last_safe_recovery': None if safe is None else {k: safe[k] for k in ('path', 'sha256', 'global_step')},
            'resume_policy': 'owner intervention required; the failed group is never skipped, and it is retried only '
                             'by the in-group ATOMIC_AMP_BACKOFF_RETRY policy before this stop'})

    # ---------------------------------------------------------------- checkpoints
    def in_flight(self):
        return bool(getattr(self.step, 'active', False) or getattr(self.warm, 'active', False))

    def save_recovery(self):
        gate(not self.in_flight(), 'no recovery checkpoint while an optimizer group or retry is in flight')
        modules = {'e_art': self.core.e_art, 'g_res': self.core.g_res, 'discriminator': self.core.discriminator,
                   'attack_head': self.core.attack_head, 'identity_head': self.core.identity_head}
        u = self.position['global_update']
        pos = dict(self.position, lr=rc.main_lr(u) if u else 0.0, curriculum=rc.curriculum(u) if u else None)
        payload = ck.recovery_payload(
            modules=modules, optimizers={'G_OPT': self.g_opt, 'D_OPT': self.d_opt},
            scalers={'G_SCALER': self.g_scaler, 'D_SCALER': self.d_scaler}, ema=self.ema, position=pos,
            provenance=self.provenance, identity_map_sha256=self.imap['mapping_sha256'] if self.imap else None,
            order={'algorithm': 'runner_io.epoch_permutation', 'mode': self.mode, 'seed': self.seed})
        info = ck.atomic_save(payload, self.ctx.ckpt_dir / 'recovery' / 'latest.pt')
        self.ctx.record_checkpoint({**info, 'epoch': self.position['epoch'], 'global_step': u, 'role': 'recovery',
                                    'checkpoint_type': 'periodic', 'selected_for_final': False,
                                    'selection_reason': 'RECOVERY_ONLY: never eligible for selection/bank/paper',
                                    'labels': list(ck.RECOVERY_LABELS)})
        return info

    def save_warmup_recovery(self):
        gate(not self.in_flight(), 'no warmup recovery checkpoint while a warmup step or retry is in flight')
        payload = ck.warmup_recovery_payload(e_art=self.core.e_art, attack_head=self.core.attack_head,
                                             optimizer=self.w_opt, scaler=self.w_scaler, position=dict(self.position),
                                             provenance=self.provenance,
                                             order={'algorithm': 'runner_io.epoch_permutation(warmup)',
                                                    'mode': self.mode, 'seed': self.seed})
        info = ck.atomic_save(payload, self.ctx.ckpt_dir / 'recovery' / 'warmup_latest.pt')
        self.ctx.record_checkpoint({**info, 'epoch': self.position['warmup_epoch'],
                                    'global_step': self.position['warmup_step'], 'role': 'warmup_recovery',
                                    'checkpoint_type': 'periodic', 'selected_for_final': False,
                                    'selection_reason': 'WARMUP_RECOVERY_ONLY', 'labels': list(ck.RECOVERY_LABELS)})
        return info

    def load_recovery(self, path):
        p = ck.load(path)
        gate(p['kind'] == ck.RECOVERY_KIND, 'not a generator recovery checkpoint')
        ck.check_provenance(p, self.provenance)
        gate(p['identity_map_sha256'] == (self.imap['mapping_sha256'] if self.imap else None), 'identity map differs')
        for k, m in (('e_art', self.core.e_art), ('g_res', self.core.g_res), ('discriminator', self.core.discriminator),
                     ('attack_head', self.core.attack_head), ('identity_head', self.core.identity_head)):
            gate((m is None) == (k not in p['modules']), f'module set differs: {k}')
            if m is not None:
                m.load_state_dict(p['modules'][k], strict=True)
        self.g_opt.load_state_dict(p['optimizers']['G_OPT'])
        self.d_opt.load_state_dict(p['optimizers']['D_OPT'])
        self.g_scaler.load_state_dict(p['scalers']['G_SCALER'])
        self.d_scaler.load_state_dict(p['scalers']['D_SCALER'])
        if p['ema'] is not None:
            self.activate_ema()
            for k in ('e_art', 'g_res'):
                self.ema[k].load_state_dict(p['ema'][k])
        self.position = {k: v for k, v in p['position'].items() if k not in ('lr', 'curriculum')}
        ck.restore_rng(p['rng'])
        return p

    def load_warmup_recovery(self, path):
        p = ck.load(path)
        gate(p['kind'] == ck.WARMUP_RECOVERY_KIND, 'not a warmup recovery checkpoint')
        ck.check_provenance(p, self.provenance)
        self.core.e_art.load_state_dict(p['modules']['e_art'], strict=True)
        self.core.attack_head.load_state_dict(p['modules']['attack_head'], strict=True)
        self.w_opt.load_state_dict(p['optimizers']['WARMUP_OPT'])
        self.w_scaler.load_state_dict(p['scalers']['WARMUP_SCALER'])
        self.position = dict(p['position'])
        ck.restore_rng(p['rng'])
        return p

    def activate_ema(self):
        self.ema = {'e_art': ModelEMA(self.core.e_art), 'g_res': ModelEMA(self.core.g_res)}
        self.ema['e_art'].initialize_from(self.core.e_art)
        self.ema['g_res'].initialize_from(self.core.g_res)
        self.step.ema = self.ema

    def write_candidate(self, epoch):
        meta = dict(self.provenance, seed=self.seed, runner_mode=self.mode, epoch=epoch,
                    global_update=self.position['global_update'])
        payload = ck.candidate_payload(e_art_ema=self.ema['e_art'], g_res_ema=self.ema['g_res'], metadata=meta)
        entry = ck.write_candidate(self.ctx.ckpt_dir / 'ema_candidates', epoch, payload)
        self.ctx.record_checkpoint(entry)
        return entry

    # ---------------------------------------------------------------- loops
    def run_warmup_batches(self, stop, limit=None):
        """Warmup from the current position; `limit` (qualification) bounds the number of batches."""
        done = 0
        while self.position['warmup_epoch'] <= rio.WARMUP_EPOCHS:
            e = self.position['warmup_epoch']
            plan = self.warmup_plan(e)
            start = self.position['warmup_next_batch']
            self.warm_sampler.set_plan(plan[start - 1:])
            for b, batch in enumerate(self.warm_loader, start=start):
                gate(batch['index'].tolist() == plan[b - 1], 'warmup batch differs from the plan')
                self.log_access(batch, ('source_spoof_id',))
                s = self.position['warmup_step'] + 1
                self.warm.on_retry = lambda ev, _e=e, _b=b: self.ctx.log('amp_retry', dict(ev, stage='warmup',
                                                                                         warmup_epoch=_e, warmup_batch=_b))
                try:
                    rec = self.warm(to_device(batch), s)
                except NumericalStop as exc:
                    self.record_failure(exc, stage='warmup', epoch=e, attempted=b, role='warmup_recovery')
                    raise
                self.position.update(warmup_step=s, warmup_next_batch=b + 1)
                self.ctx.log('warmup_step', dict(rec, warmup_epoch=e, warmup_batch=b))
                done += 1
                if stop.requested or (limit is not None and done >= limit):
                    return done
            self.position.update(warmup_epoch=e + 1, warmup_next_batch=1)
            self.save_warmup_recovery()
        gate(self.position['warmup_step'] == rio.WARMUP_TOTAL_STEPS, 'warmup step accounting 1390')
        return done

    def drop_warmup(self):
        """WARMUP_OPT / WARMUP_SCALER state never carries into the generator stage."""
        self.w_opt = self.w_scaler = self.warm = None
        self.position.update(stage='generator')

    def finish_warmup(self):
        """Handoff: discard WARMUP_OPT / WARMUP_SCALER; fresh G_OPT / G_SCALER; E_art + attack head carried."""
        self.drop_warmup()
        self.g_opt, self.g_scaler = handoff(self.core)
        self.step = GeneratorStep(self.core, self.teachers, self.cfg, self.g_opt, self.d_opt, self.g_scaler,
                                  self.d_scaler, self.ema, amp_policy=PRODUCTION_AMP_POLICY)
        self.position.update(stage='generator')

    def run_generator_groups(self, stop, limit=None, only_groups=None):
        """Generator training from the current position; `limit` bounds groups (qualification)."""
        done = 0
        while self.position['epoch'] <= rio.GENERATOR_EPOCHS:
            e = self.position['epoch']
            plan = self.generator_plan(e)
            start = self.position['next_group']
            self.gen_sampler.set_plan([mb for g in plan[start - 1:] for mb in g])
            it = iter(self.gen_loader)
            for g in range(start, rio.GROUPS_PER_EPOCH + 1):
                group = [next(it) for _ in plan[g - 1]]
                gate([b['index'].tolist() for b in group] == plan[g - 1], 'group differs from the plan')
                for b in group:
                    self.log_access(b, ('source_spoof_id', 'target_live_id'))
                u = rio.update_index(e, g)
                gate(u == self.position['global_update'] + 1, 'global update continuity')
                if hasattr(self.step, 'on_retry'):
                    self.step.on_retry = lambda ev, _e=e, _g=g: self.ctx.log('amp_retry', dict(ev, stage='generator',
                                                                                             epoch=_e, group=_g))
                try:
                    rec = self.step([to_device(b) for b in group], u)
                except NumericalStop as exc:
                    self.record_failure(exc, stage='generator', epoch=e, attempted=g, role='recovery')
                    raise
                self.position.update(global_update=u, next_group=g + 1)
                self.ctx.log('optimizer_group', dict(rec, epoch=e, group=g))
                done += 1
                if stop.requested or (limit is not None and done >= limit):
                    del it
                    return done
            del it
            self.end_epoch(e)
        return done

    def end_epoch(self, e):
        gate(self.position['global_update'] == e * rio.GROUPS_PER_EPOCH, 'epoch update accounting 1105')
        if e == rio.EMA_START_EPOCH:
            self.activate_ema()
        if e in rio.CANDIDATE_EPOCHS:
            self.write_candidate(e)
        self.position.update(epoch=e + 1, next_group=1)
        self.save_recovery()
        self.ctx.log('epoch', {'epoch': e, 'global_step': self.position['global_update'],
                               'microbatches': rio.MICROBATCHES_PER_EPOCH, 'optimizer_groups': rio.GROUPS_PER_EPOCH,
                               'rows': rio.TRAIN_ROWS, 'ema_active': self.ema is not None,
                               'val_losses': None, 'val_metrics': None,
                               'missing_field_reasons': {'val_losses': 'training never reads VAL',
                                                         'val_metrics': 'training never reads VAL'}})


# ============================================================================= scientific entry point (not executed in M7C4)
def run_scientific(*, method, seed, runtime_root, faces_root, assets, resume=False):
    rio.validate_mode_seed(rio.SCIENTIFIC, seed)
    rc.apply_qualification_determinism(seed, gpu=True)          # N-09 determinism policy (CUDA)
    tr = Trainer(mode=rio.SCIENTIFIC, method=method, seed=seed, runtime_root=runtime_root, faces_root=faces_root,
                 assets=assets, resume=resume)
    stop = StopFlag().install()
    tr.ctx.open()
    if tr.imap is not None:
        atomic_write_json(tr.ctx.path('identity_map'), {k: tr.imap[k] for k in ('classes', 'mapping_sha256',
                                                                                 'source_manifest_sha256')})
    status = 'running'
    try:
        if resume:
            rec = tr.ctx.ckpt_dir / 'recovery' / 'latest.pt'
            warm = tr.ctx.ckpt_dir / 'recovery' / 'warmup_latest.pt'
            if rec.exists():
                tr.drop_warmup()                                  # generator stage: restored G_OPT is kept
                tr.load_recovery(rec)
            elif warm.exists():
                tr.load_warmup_recovery(warm)
        if tr.position['stage'] == 'warmup':
            tr.run_warmup_batches(stop)
            if stop.requested:
                tr.save_warmup_recovery()
                status = 'interrupted'
                return tr.ctx.run_dir
            tr.finish_warmup()
        tr.run_generator_groups(stop)
        if stop.requested:
            tr.save_recovery()
            status = 'interrupted'
            return tr.ctx.run_dir
        gate(tr.position['global_update'] == rio.TOTAL_UPDATES, 'total update accounting 66300')
        status = 'completed'
        return tr.ctx.run_dir
    except NumericalStop as exc:
        status = exc.failure_type                    # e.g. FAIL_CLOSED_AMP_OVERFLOW / FAILED_NUMERICAL_POST_STEP
        raise
    except BaseException:
        status = 'failed'
        raise
    finally:
        stop.restore()
        tr.ctx.close(status, {'access': tr.access.report(), 'global_update': tr.position['global_update']})
