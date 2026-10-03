"""M7C2b GPAT static-core synthetic CPU qualification evidence (run in ~/.venvs/gpat-m7-cpu).

Builds every static GPAT module on synthetic tensors only (no dataset, no teacher weights, no GPU, no training) and
records architecture/parameter counts, the B0..B3 shape traces and the measured numerical gates. E_art is initialized
from a deterministic synthetic 3-channel ResNet-18 state (the IMAGENET1K_V1 file is never read or downloaded here).

  ~/.venvs/gpat-m7-cpu/bin/python -B tools/m7c2b_gpat_static_core_evidence.py [--write]
"""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                                    # noqa: E402
import torchvision                                                              # noqa: E402

from methods.gpat import batching, composition, losses, runtime_contract as rc, schedule, spectral, wavelet  # noqa
from methods.gpat.artifact_encoder import ArtifactEncoder                       # noqa: E402
from methods.gpat.config import VARIANTS, load_config                           # noqa: E402
from methods.gpat.discriminator import PatchGANDiscriminator                    # noqa: E402
from methods.gpat.generator import NAFResidualUNet                              # noqa: E402
from methods.gpat.model import GPATCore, parameter_counts                       # noqa: E402

EVIDENCE = 'outputs/audit/M7C2B_GPAT_STATIC_CORE.json'
AUTHORITY = 'c9a12e10eff959a31aaa361cbff98469aae1e7c2'
SEED = 20261003
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}
M7C1_APPROX = {'g_res': '~31.68M', 'e_art': '~11.2M without FC', 'discriminator': '~2.77M'}


def synthetic_resnet_state(seed: int = SEED) -> dict:
    """Deterministic random ResNet-18 (3-channel) state standing in for IMAGENET1K_V1 in synthetic tests."""
    torch.manual_seed(seed)
    return torchvision.models.resnet18(weights=None).state_dict()


def build(variant: str, seed: int = SEED, *, with_discriminator: bool = True) -> GPATCore:
    state = synthetic_resnet_state(seed)
    torch.manual_seed(seed + 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return GPATCore(load_config(variant), pretrained_state=state, with_discriminator=with_discriminator)


def images(n: int = 1, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.rand(n, 3, 256, 256, generator=g) * 2 - 1 for _ in range(3))


def shape_trace(model: GPATCore, x_s, x_t, scale_hf=0.15) -> dict:
    trace, hooks = {}, []

    def hook(name):
        def fn(_module, _inputs, output):
            out = output if isinstance(output, torch.Tensor) else output[-1]
            trace[name] = list(out.shape)                                     # returns None (never replaces output)
        return fn

    named = {'e_art.' + n: getattr(model.e_art, n) for n in ('conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4')}
    g = model.g_res
    named['g_res.intro'] = g.intro
    for i in range(4):
        named[f'g_res.enc{i + 1}'] = g.encoders[i]
        named[f'g_res.down{i + 1}'] = g.downs[i]
    named.update({'g_res.code_proj': g.code_proj, 'g_res.code_pool': g.code_pool, 'g_res.fuse': g.fuse,
                  'g_res.middle': g.middle})
    for i in range(4):
        named[f'g_res.up_dec{4 - i}'] = g.ups[i]
        named[f'g_res.dec{4 - i}'] = g.decoders[i]
    named['g_res.ending'] = g.ending
    for name, mod in named.items():
        hooks.append(mod.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            out = model(x_s, x_t, scale_hf=scale_hf)
    finally:
        for h in hooks:
            h.remove()
    trace.update({
        'x_source': list(x_s.shape), 'x_target': list(x_t.shape),
        'source_bands': [list(b.shape) for b in out.source_bands], 'target_bands': [list(b.shape) for b in out.target_bands],
        'e_art_input': list(out.encoder_input.shape), 'spatial_code': list(out.spatial_code.shape),
        'layer4': list(out.layer4.shape), 'z_a': list(out.z_a.shape), 'g_res_raw': list(out.raw.shape),
        **{k: list(getattr(out, k).shape) for k in ('delta_LL', 'delta_LH', 'delta_HL', 'delta_HH', 'M', 'LL_syn',
                                                     'LH_syn', 'HL_syn', 'HH_syn', 'x_hat', 'A')},
        'attack_logits': None if out.attack_logits is None else list(out.attack_logits.shape),
        'identity_logits': None if out.identity_logits is None else list(out.identity_logits.shape)})
    if model.discriminator is not None:
        with torch.no_grad():
            trace['D_logits'] = list(model.discriminator(out.x_hat).shape)
    return trace


def architecture(model: GPATCore) -> dict:
    g, d = model.g_res, model.discriminator
    return {
        'e_art': {'backbone': 'torchvision.models.resnet18 body (conv1 12->64 k7 s2 p3 no bias, bn1, relu, maxpool, '
                              'layer1..layer4); no avgpool/fc', 'conv1': list(model.e_art.conv1.weight.shape),
                  'has_fc': hasattr(model.e_art, 'fc'), 'z_a': 'F.normalize(GAP(layer4), p=2, dim=1)'},
        'g_res': {'intro': repr(g.intro), 'encoder_channels': [e.film.channels for e in g.encoders],
                  'encoder_blocks': [len(e.blocks) for e in g.encoders], 'downs': [repr(m) for m in g.downs],
                  'code_proj': repr(g.code_proj), 'code_pool': repr(g.code_pool), 'fuse': repr(g.fuse),
                  'middle_blocks': len(g.middle), 'ups': [repr(u.conv) for u in g.ups],
                  'decoder_channels': [dd.film.channels for dd in g.decoders],
                  'decoder_blocks': [len(dd.blocks) for dd in g.decoders], 'ending': repr(g.ending),
                  'film': [repr(e.film.affine) for e in (*g.encoders, *g.decoders)], 'bottleneck_film': False,
                  'naf_block_type': type(g.middle[0]).__module__ + '.' + type(g.middle[0]).__name__},
        'discriminator': [repr(m) for m in d.model] if d is not None else None}


def counts(model: GPATCore) -> dict:
    out = {n: parameter_counts(getattr(model, n)) for n in ('g_res', 'e_art', 'discriminator')}
    for n in ('attack_head', 'identity_head'):
        mod = getattr(model, n)
        out[n] = parameter_counts(mod) if mod is not None else None
    return out


def wavelet_checks() -> dict:
    g = torch.Generator().manual_seed(42)
    x = torch.rand(100, 3, 256, 256, generator=g) * 2 - 1                      # A10 D11 corpus
    rec = wavelet.idwt(*wavelet.dwt(x))
    d11 = float((rec - x).abs().max())
    stripes = torch.zeros(1, 3, 256, 256)
    stripes[..., 1::2, :] = 1.0                                                # horizontal stripes (vary along rows)
    h = [float(b.abs().max()) for b in wavelet.dwt(stripes)[1:]]
    v = [float(b.abs().max()) for b in wavelet.dwt(stripes.transpose(-1, -2).contiguous())[1:]]
    return {'D11_max_abs': d11, 'D11_pass_if_less_than': 1e-5, 'horizontal_stripes_LH_HL_HH_absmax': h,
            'vertical_stripes_LH_HL_HH_absmax': v, 'mapping': {'LL': 'cA', 'LH': 'cH', 'HL': 'cV', 'HH': 'cD'}}


def highpass_check() -> dict:
    import cv2
    import numpy as np
    from methods.gpat.highpass import highpass
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(2, 3, 256, 256, generator=g) * 2 - 1
    hp = highpass(x)
    ref = np.stack([np.stack([c - cv2.GaussianBlur(c, (9, 9), 1.5, borderType=cv2.BORDER_REFLECT_101)
                              for c in img]) for img in x.numpy()])
    return {'max_abs_vs_cv2_reflect101': float(np.abs(hp.numpy() - ref).max()), 'threshold': 1e-6}


def d12_checks(model: GPATCore) -> dict:
    x_s1, x_s2, x_t = images(2, SEED + 7)
    with torch.no_grad():
        a = model(x_s1, x_t, scale_hf=0.15, artifact_scale=0.0)
        b = model(x_s2, x_t, scale_hf=0.15, artifact_scale=0.0)
        live1 = model(x_s1, x_t, scale_hf=0.15)
        live2 = model(x_s2, x_t, scale_hf=0.15)
        recon = wavelet.idwt(*wavelet.dwt(x_t))
    return {'zero_residual_x_hat_minus_idwt_dwt_x_t_max_abs': float((a.x_hat - recon).abs().max()),
            'zero_residual_x_hat_minus_x_t_max_abs': float((a.x_hat - x_t).abs().max()),
            'source_independence_max_abs': float((a.x_hat - b.x_hat).abs().max()),
            'threshold': 1e-5,
            'non_vacuous_live_source_dependence_max_abs': float((live1.x_hat - live2.x_hat).abs().max()),
            'gamma0_LL_syn_equals_LL_t_exact': bool(torch.equal(live1.LL_syn, live1.target_bands[0]))}


def l_low_checks(model: GPATCore) -> dict:
    """Training L_low (literal spec 10.1, re-DWT of x_hat) vs N-08 selection LFErr (internal LL), synthetic only."""
    x_s, _, x_t = images(1, SEED + 11)
    with torch.no_grad():
        zero = model(x_s, x_t, scale_hf=0.15, artifact_scale=0.0)
        live = model(x_s, x_t, scale_hf=0.15)
    ll_t = live.target_bands[0]
    yy, xx = torch.meshgrid(torch.arange(256.), torch.arange(256.), indexing='ij')
    perturbed = live.x_hat + 0.1 * torch.sin(2 * torch.pi * (xx + 2 * yy) / 64)
    x = live.x_hat.clone().requires_grad_(True)
    losses.l_low(x, ll_t).backward()
    return {'definition': 'mean |DWT(x_hat).LL - LL_t| (fresh fp32 ptwt Haar level-1 reflect DWT of x_hat)',
            'zero_residual': float(losses.l_low(zero.x_hat, ll_t)),
            'random_init_residual': float(losses.l_low(live.x_hat, ll_t)),
            'perturbed_x_hat': float(losses.l_low(perturbed, ll_t)),
            'internal_ll_value_same_x_hat': float((live.LL_syn - ll_t).abs().mean()),
            'grad_to_x_hat_abs_sum': float(x.grad.abs().sum()),
            'grad_to_x_hat_finite': bool(torch.isfinite(x.grad).all()),
            'selection_lferr_gamma0_max': float(losses.lferr_selection(live.LL_syn, ll_t).abs().max())}


PATCHGAN_CONVENTION = {
    'class': 'IMPLEMENTATION_CLARIFICATION', 'status': 'OWNER_APPROVED_M7C2B', 'new_deviation': None,
    'layers': [
        {'conv': 'Conv4x4 s2 6->64 pad1 bias', 'norm': None, 'activation': 'LeakyReLU(0.2)'},
        {'conv': 'Conv4x4 s2 64->128 pad1 bias', 'norm': 'InstanceNorm2d(affine=False, track_running_stats=False)',
         'activation': 'LeakyReLU(0.2)'},
        {'conv': 'Conv4x4 s2 128->256 pad1 bias', 'norm': 'InstanceNorm2d(affine=False, track_running_stats=False)',
         'activation': 'LeakyReLU(0.2)'},
        {'conv': 'Conv4x4 s1 256->512 pad1 bias', 'norm': 'InstanceNorm2d(affine=False, track_running_stats=False)',
         'activation': 'LeakyReLU(0.2)'},
        {'conv': 'Conv4x4 s1 512->1 pad1 bias', 'norm': None, 'activation': None}],
    'output': [None, 1, 30, 30], 'sigmoid': False}
OWNER_CLARIFICATIONS = {
    'L_low_training': {'definition': 'mean |DWT(x_hat).LL - LL_t|', 'basis': 'frozen spec 10.1 literal',
                       'uses_internal_LL_syn': False, 'differentiable_to_x_hat': True},
    'LFErr_selection': {'definition': '||LL_syn_internal - LL_t||_1 / (||LL_t||_1 + 1e-8)', 'basis': 'M7C2a N-08',
                        'gamma0_value': 0.0, 'used_as_training_loss': False},
    'patchgan': PATCHGAN_CONVENTION,
    'dev022_group_normalization': {
        'status': 'RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED',
        'rule': 'L_idadv_group = sum_j ce_sum_j / sum_j labelled_count_j; exact differentiable zero if no labelled row',
        'microbatch_sample_weight_applied_to_idadv': False}}


def spectral_checks() -> dict:
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(2, 3, 256, 256, generator=g) * 2 - 1
    r, o = spectral.s_radial(x), spectral.s_orient(x)
    yy, xx = torch.meshgrid(torch.arange(256.), torch.arange(256.), indexing='ij')
    horiz = torch.cos(2 * torch.pi * 32 * yy / 256).expand(1, 3, 256, 256).contiguous()   # varies along rows
    vert = torch.cos(2 * torch.pi * 32 * xx / 256).expand(1, 3, 256, 256).contiguous()    # varies along columns
    return {'radial_shape': list(r.shape), 'orient_shape': list(o.shape),
            'radial_sum_max_dev': float((r.sum(-1) - 1).abs().max()), 'orient_sum_max_dev': float((o.sum(-1) - 1).abs().max()),
            'repeat_bitwise_identical': bool(torch.equal(r, spectral.s_radial(x)) and torch.equal(o, spectral.s_orient(x))),
            'rows_pattern_orient_argmax': int(spectral.s_orient(horiz)[0, 0].argmax()),
            'cols_pattern_orient_argmax': int(spectral.s_orient(vert)[0, 0].argmax())}


def schedule_checks() -> dict:
    return {'main_lr': {u: schedule.main_lr(u) for u in (1, 2, 5525, 5526, 66300)},
            'attack_warmup_lr': {s: schedule.attack_warmup_lr(s) for s in (1, 1390)},
            'curriculum': {u: schedule.curriculum(u) for u in (1, 5525, 5526, 16575, 16576, 66300)}}


def run() -> dict:
    det = rc.apply_qualification_determinism(SEED, gpu=False)
    b0 = build('B0')
    traces = {}
    for v in VARIANTS:
        model = b0 if v == 'B0' else build(v)
        traces[v] = shape_trace(model, *images(1)[:2])
    b3 = build('B3')
    ev = {
        'milestone': 'M7C2b', 'status': 'PASS', 'authority_commit': AUTHORITY, 'seed': SEED,
        'synthetic_only': True, 'dataset_images_read': 0, 'teacher_weights_loaded': False,
        'imagenet_weight_file_read': False, 'gpu_used': False, 'training_steps': 0, 'checkpoint_writes': 0,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'ptwt': md.version('ptwt'),
                        'PyWavelets_distribution': md.version('PyWavelets'), 'cuda_available': torch.cuda.is_available(),
                        'sys_prefix': sys.prefix, 'determinism': det},
        'config_sha256': {v: VARIANTS[v][3] for v in VARIANTS},
        'parameter_counts': counts(b3), 'parameter_counts_expected': EXPECTED_COUNTS, 'm7c1_approximate': M7C1_APPROX,
        'architecture': architecture(b3), 'shape_traces': traces,
        'wavelet': wavelet_checks(), 'highpass': highpass_check(), 'd12': d12_checks(b0),
        'spectral': spectral_checks(), 'l_low': l_low_checks(b0), 'owner_clarifications': OWNER_CLARIFICATIONS, 'schedule': schedule_checks(), 'batching': batching.accounting()}
    pc = ev['parameter_counts']
    gates = {
        'counts': all(pc[k]['total'] == EXPECTED_COUNTS[k] for k in EXPECTED_COUNTS),
        'D11': ev['wavelet']['D11_max_abs'] < 1e-5,
        'highpass': ev['highpass']['max_abs_vs_cv2_reflect101'] <= 1e-6,
        'zero_residual': ev['d12']['zero_residual_x_hat_minus_idwt_dwt_x_t_max_abs'] < 1e-5
        and ev['d12']['zero_residual_x_hat_minus_x_t_max_abs'] < 1e-5,
        'source_independence': ev['d12']['source_independence_max_abs'] < 1e-5,
        'non_vacuous': ev['d12']['non_vacuous_live_source_dependence_max_abs'] > 1e-5,
        'gamma0': ev['d12']['gamma0_LL_syn_equals_LL_t_exact'],
        'patchgan_30x30': all(t['D_logits'] == [1, 1, 30, 30] for t in traces.values()),
        'l_low_literal': ev['l_low']['zero_residual'] < 1e-5 and ev['l_low']['perturbed_x_hat'] > 1e-3
        and ev['l_low']['grad_to_x_hat_abs_sum'] > 0 and ev['l_low']['grad_to_x_hat_finite'],
        'lferr_gamma0_exact_zero': ev['l_low']['selection_lferr_gamma0_max'] == 0.0,
        'cpu_only': not ev['environment']['cuda_available']}
    ev['gates'] = gates
    if not all(gates.values()):
        ev['status'] = 'FAIL'
    return ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    ev = run()
    text = json.dumps(ev, indent=1, sort_keys=True) + '\n'
    if args.write:
        (ROOT / EVIDENCE).write_text(text)
    print(json.dumps({'status': ev['status'], 'gates': ev['gates'], 'sha256': hashlib.sha256(text.encode()).hexdigest()},
                     indent=1))
    if ev['status'] != 'PASS':
        sys.exit(1)


if __name__ == '__main__':
    main()
