"""Pure GPAT loss functions (spec 10.1-10.4, A10 D01-D03/D08-D10, M7C2a N-03/N-07/ADVERSARIAL_BCE).

Every function takes already-computed tensors (teacher embeddings, landmarks, parsing logits, F_art features, D logits)
and returns fp32 scalars; no teacher, F_art or D is instantiated here, nothing calls backward or steps an optimizer.
Definitions already frozen in `runtime_contract` (parsing, spectra, BCE, face mask) are reused, not re-implemented.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.gpat import runtime_contract as rc
from methods.gpat import spectral, wavelet

# Frozen weights (spec 10.3, gpat_b0.yaml); lambda_con / lambda_spec / lambda_adv come from the curriculum and
# lambda_type / lambda_idadv from the variant config.
FIXED_WEIGHTS = {'id': 1.0, 'lm': 1.0, 'parse': 0.5, 'low': 2.0, 'budget': 0.5, 'tv': 0.05, 'bg': 1.0}
CURRICULUM_WEIGHTS = {'artcon': 'lambda_con', 'spec': 'lambda_spec', 'gadv': 'lambda_adv'}
ARTIFACT_MIN = 0.01
ARTIFACT_MAX = 0.25
TV_DELTA_SCALE = 0.25
TRIPLET_MARGIN = 0.2


def _f32(t):
    return t.float()


def _fp32_region(t):
    return torch.autocast(device_type=t.device.type, enabled=False)


# ----------------------------------------------------------------------------- geometry
def l_id(target_embedding, hat_embedding):
    """1 - cos(F_id(x_t), F_id(x_hat)), batch mean (512-D frozen teacher embeddings)."""
    with _fp32_region(hat_embedding):
        return (1.0 - F.cosine_similarity(_f32(target_embedding), _f32(hat_embedding), dim=1)).mean()


def l_lm(target_landmarks, hat_landmarks):
    """Mean absolute error of FaceXFormer native normalized coordinates [N, 68, 2] (A10 D08; no heatmaps)."""
    if target_landmarks.shape[-2:] != (68, 2) or hat_landmarks.shape != target_landmarks.shape:
        raise ValueError('landmarks must be [N, 68, 2]')
    with _fp32_region(hat_landmarks):
        return (_f32(hat_landmarks) - _f32(target_landmarks)).abs().mean()


def l_parse(target_logits, hat_logits, *, components: bool = False):
    """L_dice + 0.1 * KL(p_t || p_h) on [N, 11, 224, 224] logits (M7C2a N-07)."""
    if target_logits.dim() != 4 or target_logits.shape[1] != 11 or hat_logits.shape != target_logits.shape:
        raise ValueError('parsing logits must be [N, 11, H, W]')
    with _fp32_region(hat_logits):
        dice = rc.soft_dice_loss(target_logits, hat_logits)
        kl = rc.kl_loss(target_logits, hat_logits)
        total = dice + rc.KL_WEIGHT * kl
    return (total, {'dice': dice, 'kl': kl}) if components else total


def l_low(x_hat, ll_target):
    """TRAINING L_low = mean |LL(x_hat) - LL_t| (frozen spec 10.1, literal): a fresh differentiable fp32 DWT of x_hat
    (the frozen ptwt Haar level-1 reflect operator); LL_t is the LL band of the original DWT(x_t). It never reads the
    internal LL_syn, so gamma = 0 does not bypass the DWT (the value is then only fp32 reconstruction noise)."""
    if x_hat.dim() != 4 or tuple(x_hat.shape[1:]) != (3, 256, 256):
        raise ValueError('L_low expects x_hat [N, 3, 256, 256]')
    ll_hat = wavelet.dwt(x_hat)[0]
    with _fp32_region(ll_hat):
        return (ll_hat - _f32(ll_target)).abs().mean()


def lferr_selection(ll_syn_internal, ll_target):
    """SELECTION metric only (M7C2a N-08), never a training loss: per-sample ||LL_syn_internal - LL_t||_1 /
    (||LL_t||_1 + 1e-8) on the internal LL coefficients; exactly 0 for every candidate when gamma = 0."""
    return rc.lferr_internal(ll_syn_internal, ll_target)


# ----------------------------------------------------------------------------- artifact
def l_artcon(f_hat, f_source, f_target, margin: float = TRIPLET_MARGIN):
    """1 - cos(f_hat, f_s) + max(0, cos(f_hat, f_t) - cos(f_hat, f_s) + 0.2), batch mean (F_art features, fp32)."""
    with _fp32_region(f_hat):
        c_s = F.cosine_similarity(_f32(f_hat), _f32(f_source), dim=1)
        c_t = F.cosine_similarity(_f32(f_hat), _f32(f_target), dim=1)
        return (1.0 - c_s + torch.clamp(c_t - c_s + margin, min=0.0)).mean()


l_spec = spectral.spec_loss


def l_type(attack_logits, labels):
    """Unweighted mean CE over the spoof rows the caller passes (A10 D05; no live class, never on x_hat)."""
    if attack_logits.dim() != 2 or attack_logits.shape[1] != 6:
        raise ValueError('attack logits must be [N, 6]')
    with _fp32_region(attack_logits):
        return F.cross_entropy(_f32(attack_logits), labels, reduction='mean')


# ----------------------------------------------------------------------------- identity adversary (DEV-022)
def idadv_microbatch(identity_logits, labels, valid):
    """(sum of per-row CE over labelled rows, labelled count) for one physical microbatch; not normalized here.

    Masked rows (SiW-Mv2, no source_subject) never enter the CE; their label value is ignored. With no labelled row
    the sum is an exact zero that stays attached to the graph.
    """
    if identity_logits.dim() != 2 or identity_logits.shape[1] != 60:
        raise ValueError('identity logits must be [N, 60]')
    valid = valid.to(torch.bool)
    if valid.shape != identity_logits.shape[:1]:
        raise ValueError('valid mask must be [N]')
    with _fp32_region(identity_logits):
        logits = _f32(identity_logits)
        count = int(valid.sum())
        if count == 0:
            return logits.sum() * 0.0, 0
        return F.cross_entropy(logits[valid], labels[valid], reduction='sum'), count


def idadv_group(sums, counts):
    """Optimizer-group L_idadv = total labelled CE sum / total labelled count; exact differentiable zero if none."""
    sums, total = list(sums), int(sum(counts))
    if not sums:
        raise ValueError('an optimizer group has at least one microbatch')
    acc = sums[0]
    for s in sums[1:]:
        acc = acc + s
    return acc * 0.0 if total == 0 else acc / total


# DEV-022 runner contract (owner M7C2b review): RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED. Per optimizer group,
# L_idadv = sum_j ce_sum_j / sum_j labelled_count_j (exact differentiable zero when no row is labelled). A microbatch
# contributes idadv_share(ce_sum_j, group count); it is NEVER multiplied again by the sample weight n_j / n_group.
IDADV_RUNNER_CONTRACT = 'RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED'


def idadv_share(microbatch_sum, group_labelled_count: int):
    """One microbatch's additive share of the group L_idadv: its CE sum / the group's labelled count (0 -> zero)."""
    return microbatch_sum * 0.0 if group_labelled_count == 0 else microbatch_sum / int(group_labelled_count)


# ----------------------------------------------------------------------------- regularization
def l_budget(a_map):
    """Per-image hinge max(0, 0.01 - mean(A)) + max(0, mean(A) - 0.25), batch mean (A10 D10.6)."""
    with _fp32_region(a_map):
        m = _f32(a_map).flatten(1).mean(dim=1)
        return (torch.clamp(ARTIFACT_MIN - m, min=0.0) + torch.clamp(m - ARTIFACT_MAX, min=0.0)).mean()


def tv(t):
    """Anisotropic mean L1 total variation: mean |d/dx| + mean |d/dy| (A10 D10.3)."""
    t = _f32(t)
    return (t[..., :, 1:] - t[..., :, :-1]).abs().mean() + (t[..., 1:, :] - t[..., :-1, :]).abs().mean()


def l_tv(m, delta_lh, delta_hl, delta_hh):
    """TV(M) + 0.25 * (TV(dLH) + TV(dHL) + TV(dHH)), all at 128x128."""
    with _fp32_region(m):
        return tv(m) + TV_DELTA_SCALE * (tv(delta_lh) + tv(delta_hl) + tv(delta_hh))


def l_bg(x_hat, x_t, face_mask_dilated):
    """mean |(1 - M_face_dilated) * (x_hat - x_t)| with M_face_dilated [N, 1, 256, 256]."""
    if face_mask_dilated.dim() != 4 or face_mask_dilated.shape[1] != 1:
        raise ValueError('M_face_dilated must be [N, 1, H, W]')
    with _fp32_region(x_hat):
        return ((1.0 - _f32(face_mask_dilated)) * (_f32(x_hat) - _f32(x_t))).abs().mean()


def face_mask_dilated(parsing_argmax_224):
    """[N, 224, 224] parser classes -> [N, 1, 256, 256] {0, 1}: classes 1..10, nearest-exact, 15x15 dilation (N-03)."""
    if parsing_argmax_224.dim() != 3 or tuple(parsing_argmax_224.shape[1:]) != (224, 224):
        raise ValueError('parser mask must be [N, 224, 224]')
    return rc.face_mask_dilated(parsing_argmax_224)


# ----------------------------------------------------------------------------- adversarial (M7C2a ADVERSARIAL_BCE)
l_d_terms = rc.d_bce_terms          # (mean BCE(real, 1), mean BCE(fake.detach(), 0))
l_d = rc.d_loss                     # 0.5 * (L_D_real + L_D_fake)
l_gadv = rc.g_adv_bce               # mean BCE(D(x_hat), 1); x_hat not detached


# ----------------------------------------------------------------------------- generator objective
def assemble_generator_loss(components: dict, curriculum: dict, *, lambda_type: float, lambda_idadv: float):
    """L_G = sum of weighted components (spec 10.3 as resolved by A10 D02: + lambda_idadv * L_idadv, the GRL is the
    only reversal). `components` holds already-computed scalars; `curriculum` holds lambda_con/spec/adv of the update.
    Returns (total, detached components and weights). No backward here."""
    weights = dict(FIXED_WEIGHTS)
    for key, name in CURRICULUM_WEIGHTS.items():
        weights[key] = float(curriculum[name])
    for key, lam in (('type', lambda_type), ('idadv', lambda_idadv)):
        if lam:
            weights[key] = float(lam)
        elif key in components:
            raise ValueError(f'L_{key} supplied for a variant whose lambda_{key} is 0')
    missing = sorted(set(weights) - set(components))
    unknown = sorted(set(components) - set(weights))
    if missing or unknown:
        raise ValueError(f'generator components missing {missing} / unknown {unknown}')
    total = None
    for key in sorted(weights):
        term = weights[key] * components[key].float()
        total = term if total is None else total + term
    record = {'weights': weights, 'components': {k: components[k].detach().float() for k in sorted(components)}}
    return total, record
