"""M6D5c E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1 (CONTROLLED_EXECUTION_ADAPTATION).

Owner-approved execution resolution after the M6D5b physical-B=240 CUDA OOM. One
global batch of 240 is executed as 12 microbatches of 20 with ONE optimizer step:

  epsilon  eps_cls, eps_nir, eps_vis [240,128] drawn ONCE per global batch in the
           pinned order cls -> nir -> vis and replayed in both passes
           (STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION).
  pass 1   torch.no_grad encoders only: global delta = mean_240(z_nir - z_vis) and
           ort_mean = mean_240(sum_d z_cls*z_nir); their signs are detached constants.
  pass 2   zero_grad(set_to_none=False) once, then per chunk the full graph with the
           per-sample mean terms weighted by m/240 and the linear MMD/orthogonality
           surrogates whose sums equal (and whose gradients are) the global terms.

This is NOT naive accumulation and NOT a bitwise reproduction of the impossible
physical-B=240 step. The M6D5b transcription (training_graph.py) is imported and
unchanged. No dataset, image reader, manifest, epoch loop or checkpoint writer
exists here. Torch is passed in by the caller; importing this module is static.
"""
import json

from methods.common.config import ROOT
from methods.common.learned import PreparationError
from methods.dsdg import training_graph as tg

RESOLUTION_PATH = 'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml'
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1'
CLASSIFICATION = 'CONTROLLED_EXECUTION_ADAPTATION'
DRAW_POLICY = 'STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION'
GLOBAL_BATCH = 240
MICROBATCH = 20
MICROBATCHES = 12
HDIM = 128
EPS_ORDER = ('cls', 'nir', 'vis')     # train_generator.py:125-127
GLOBAL_TERMS = ('loss_mmd', 'loss_ort')
LOCAL_TERMS = ('loss_rec', 'loss_kl', 'loss_ip', 'loss_cls', 'loss_pair')
FORBIDDEN_EXECUTION = ('autocast', 'fp16', 'bf16', 'tf32', 'activation_checkpointing', 'cpu_offload',
                       'parameter_offload', 'optimizer_offload', 'reduced_resolution', 'microbatch_averaging')


def load_resolution(path=None):
    """The owner overlay is the JSON subset of YAML; parsed with the standard library only."""
    return json.loads((ROOT / (path or RESOLUTION_PATH)).read_text())


def execution_guard(resolution, *, global_batch_size=GLOBAL_BATCH, microbatch_size=MICROBATCH, **forbidden):
    """Exactly 240 = 12 x 20 in FP32 under the owner overlay; every other workaround is refused."""
    unknown = sorted(set(forbidden) - set(FORBIDDEN_EXECUTION))
    if unknown:
        raise PreparationError('unknown execution option(s): ' + ', '.join(unknown))
    used = sorted(k for k, v in forbidden.items() if v)
    if used:
        raise PreparationError('M6D5c forbids execution workaround(s): ' + ', '.join(used))
    r = resolution
    fixed = (r['execution_mode'], r['classification'], r['fidelity_class'], r['global_batch_size'],
             r['microbatch_size'], r['microbatches_per_step'], r['optimizer_steps_per_global_batch'],
             r['precision'], r['amp'], r['tf32'], r['activation_checkpointing'], r['cpu_parameter_offload'],
             r['optimizer_offload'], r['naive_microbatch_loss_averaging'], tuple(r['global_statistic_terms']),
             r['physical_batch_240'])
    if fixed != (EXECUTION_MODE, CLASSIFICATION, 'CONTROLLED_ADAPTATION', GLOBAL_BATCH, MICROBATCH, MICROBATCHES,
                 1, 'FP32', False, False, False, False, False, 'forbidden', GLOBAL_TERMS, 'OOM_RETAINED'):
        raise PreparationError('M6D5c owner overlay differs from the frozen execution resolution')
    if (global_batch_size, microbatch_size) != (GLOBAL_BATCH, MICROBATCH):
        raise PreparationError(f'{EXECUTION_MODE} requires global batch {GLOBAL_BATCH} and microbatch {MICROBATCH}; '
                               'microbatch tuning is forbidden (OOM at 20 is STOP_AND_REPORT)')
    return {'execution_mode': EXECUTION_MODE, 'classification': CLASSIFICATION,
            'global_batch_size': GLOBAL_BATCH, 'microbatch_size': MICROBATCH,
            'microbatches_per_step': MICROBATCHES, 'optimizer_steps_per_global_batch': 1,
            'precision': 'FP32', 'amp': False, 'tf32': False, 'physical_batch_240': 'OOM_RETAINED'}


def chunk_slices(global_batch=GLOBAL_BATCH, microbatch=MICROBATCH):
    """Contiguous, ordered, exhaustive and disjoint chunks; the batch must divide evenly."""
    if global_batch % microbatch:
        raise PreparationError('global batch must be a whole number of microbatches')
    return [slice(i, i + microbatch) for i in range(0, global_batch, microbatch)]


def draw_epsilon(torch, global_batch=GLOBAL_BATCH, hdim=HDIM, device='cuda'):
    """Three standard-normal FP32 draws in pinned order cls -> nir -> vis, once per global batch.

    `empty(...).normal_()` on the CUDA default generator is the same call pinned
    misc/util.py::reparameterize makes through torch.cuda.FloatTensor(size).normal_().
    """
    return {k: torch.empty((global_batch, hdim), dtype=torch.float32, device=device).normal_() for k in EPS_ORDER}


def replay_latent(eps, mu, logvar):
    """misc/util.py::reparameterize with the epsilon supplied instead of drawn: z = mu + eps*std."""
    std = logvar.mul(0.5).exp_()
    return eps.mul(std).add_(mu)


def encode(nets, img_nir, img_vis, eps, s):
    """train_generator.py:121-127 for one chunk, with the replayed epsilon slices."""
    netE_nir, netE_vis = nets[0], nets[1]
    mu_nir, logvar_nir, mu_a, logvar_a = netE_nir(img_nir)
    mu_vis, logvar_vis = netE_vis(img_vis)
    z_cls = replay_latent(eps['cls'][s], mu_a, logvar_a)
    z_nir = replay_latent(eps['nir'][s], mu_nir, logvar_nir)
    z_vis = replay_latent(eps['vis'][s], mu_vis, logvar_vis)
    return dict(mu_nir=mu_nir, logvar_nir=logvar_nir, mu_a=mu_a, logvar_a=logvar_a,
                mu_vis=mu_vis, logvar_vis=logvar_vis, z_cls=z_cls, z_nir=z_nir, z_vis=z_vis)


def pass1_statistics(torch, nets, x_spoof, x_live, eps, slices, lam, keep_latents=False):
    """No optimizer, no backward, no netCls/netG/netIP: FP32 global MMD/orthogonality statistics."""
    B = x_spoof.shape[0]
    dtype = eps['nir'].dtype  # FP32 in execution (draw_epsilon); float64 only in CPU exactness tests
    delta_sum = torch.zeros(eps['nir'].shape[1], dtype=dtype, device=x_spoof.device)
    ort_sum = torch.zeros((), dtype=dtype, device=x_spoof.device)
    latents = []
    with torch.no_grad():
        for s in slices:
            e = encode(nets, x_spoof[s], x_live[s], eps, s)
            delta_sum += (e['z_nir'] - e['z_vis']).sum(dim=0)
            ort_sum += (e['z_cls'] * e['z_nir']).sum(dim=1).sum()
            if keep_latents:
                latents.append({k: e[k].clone() for k in ('z_cls', 'z_nir', 'z_vis')})
            del e
        delta = delta_sum / B
        ort_mean = ort_sum / B
        stats = {'delta': delta, 'ort_mean': ort_mean,
                 'mmd_sign': torch.sign(delta), 'ort_sign': torch.sign(ort_mean),   # sign(0) = 0
                 'loss_mmd_global': lam['lambda_mmd'] * delta.abs().mean(),
                 'loss_ort_global': lam['lambda_ort'] * ort_mean.abs(),
                 'grad_enabled_inside': torch.is_grad_enabled()}
    if keep_latents:
        stats['latents'] = latents
    return stats


def mmd_surrogate(z_nir, z_vis, mmd_sign, lam, global_batch=GLOBAL_BATCH):
    """lambda_mmd/(B*D) * sum(sign(delta) * (z_nir - z_vis)); summed over chunks = lambda_mmd*|delta|.mean()."""
    return lam['lambda_mmd'] / (global_batch * z_nir.shape[1]) * (mmd_sign.detach() * (z_nir - z_vis)).sum()


def ort_surrogate(z_cls, z_nir, ort_sign, lam, global_batch=GLOBAL_BATCH):
    """lambda_ort*sign(ort_mean)/B * sum_b(sum_d z_cls*z_nir); summed over chunks = lambda_ort*|ort_mean|."""
    return lam['lambda_ort'] * ort_sign.detach() / global_batch * (z_cls * z_nir).sum(dim=1).sum()


def chunk_forward(torch, F, util, nets, img_nir, img_vis, label_spoof, eps, s, lam, criterion_type, criterionL2,
                  stats, global_batch=GLOBAL_BATCH):
    """train_generator.py:118-171 for one chunk (nir = spoof, vis = live), differing ONLY in:

    replayed epsilon; target LightCNN features under no_grad (approved dead-gradient
    elimination); MMD/orthogonality as the pass-1-signed global surrogates.
    """
    netE_nir, netE_vis, netG, netCls, netIP = nets
    reconstruction_loss, kl_loss, rgb2gray = util.reconstruction_loss, util.kl_loss, util.rgb2gray
    img = torch.cat((img_nir, img_vis), 1)
    e = encode(nets, img_nir, img_vis, eps, s)
    z_cls, z_nir, z_vis = e['z_cls'], e['z_nir'], e['z_vis']
    pre_spoof = netCls(z_cls)
    rec = netG(torch.cat((z_cls, z_nir, z_vis), dim=1))
    loss_rec = reconstruction_loss(rec, img, True) / 2.0
    loss_kl = (kl_loss(e['mu_nir'], e['logvar_nir']).mean() + kl_loss(e['mu_vis'], e['logvar_vis']).mean()
               + kl_loss(e['mu_a'], e['logvar_a']).mean()) / 3.0
    loss_mmd_surrogate = mmd_surrogate(z_nir, z_vis, stats['mmd_sign'], lam, global_batch)
    loss_cls = lam['lambda_type'] * criterion_type(pre_spoof, label_spoof)
    loss_ort_surrogate = ort_surrogate(z_cls, z_nir, stats['ort_sign'], lam, global_batch)
    rec_nir = rec[:, 0:3, :, :]
    rec_vis = rec[:, 3:6, :, :]
    img_nir = F.interpolate(img_nir, size=(128, 128), mode='bilinear')
    img_vis = F.interpolate(img_vis, size=(128, 128), mode='bilinear')
    rec_nir = F.interpolate(rec_nir, size=(128, 128), mode='bilinear')
    rec_vis = F.interpolate(rec_vis, size=(128, 128), mode='bilinear')
    with torch.no_grad():  # targets: no input gradient, frozen netIP, detached in loss_ip
        nir_fc = netIP(rgb2gray(img_nir))
        vis_fc = netIP(rgb2gray(img_vis))
    rec_nir_fc = netIP(rgb2gray(rec_nir))  # NOT no_grad: loss_ip -> frozen LightCNN -> rec -> netG/encoders
    rec_vis_fc = netIP(rgb2gray(rec_vis))
    nir_fc = F.normalize(nir_fc, p=2, dim=1)
    vis_fc = F.normalize(vis_fc, p=2, dim=1)
    rec_nir_fc = F.normalize(rec_nir_fc, p=2, dim=1)
    rec_vis_fc = F.normalize(rec_vis_fc, p=2, dim=1)
    loss_ip = lam['lambda_ip'] * (
            criterionL2(rec_nir_fc, nir_fc.detach()) + criterionL2(rec_vis_fc, vis_fc.detach())) / 2.0
    loss_pair = lam['lambda_pair'] * criterionL2(rec_nir_fc, rec_vis_fc)
    return dict(e, pre_spoof=pre_spoof, rec=rec, rec_nir=rec_nir, rec_vis=rec_vis, nir_fc=nir_fc, vis_fc=vis_fc,
                rec_nir_fc=rec_nir_fc, rec_vis_fc=rec_vis_fc, loss_rec=loss_rec, loss_kl=loss_kl, loss_ip=loss_ip,
                loss_cls=loss_cls, loss_pair=loss_pair, loss_mmd_surrogate=loss_mmd_surrogate,
                loss_ort_surrogate=loss_ort_surrogate)


def chunk_objective(out, chunk_size, epoch, global_batch=GLOBAL_BATCH):
    """Per-microbatch backward objective; local mean terms weighted by m/B, surrogates unweighted."""
    w = chunk_size / global_batch
    scale = tg.WARMUP_SCALE if epoch < tg.WARMUP_BELOW_EPOCH else 1.0
    return (w * out['loss_rec']
            + scale * (w * out['loss_kl']) + scale * (w * out['loss_ip']) + scale * (w * out['loss_cls'])
            + scale * (w * out['loss_pair'])
            + scale * out['loss_mmd_surrogate'] + scale * out['loss_ort_surrogate'])


def run_global_batch(torch, F, util, nets, optimizer, x_spoof, x_live, label_spoof, eps, lam, criterion_type,
                     criterionL2, epoch, microbatch=MICROBATCH, on_pass1=None, on_chunk=None, after_backward=None,
                     before_step=None, keep_latents=False):
    """Pass 1, zero_grad(set_to_none=False) once, 12 x (forward + backward), one optimizer.step().

    Observers never touch tensors used by the objective: `on_pass1(stats)`,
    `on_chunk(i, out)` before chunk i's backward (hooks), `after_backward(i)`
    (memory), `before_step()` after the last backward and before the single step.
    Returns global loss values as Python floats (float64 accumulation for logging).
    """
    B = x_spoof.shape[0]
    slices = chunk_slices(B, microbatch)
    stats = pass1_statistics(torch, nets, x_spoof, x_live, eps, slices, lam, keep_latents=keep_latents)
    if on_pass1 is not None:
        on_pass1(stats)
    optimizer.zero_grad(set_to_none=False)
    acc = dict.fromkeys(LOCAL_TERMS + ('loss_mmd_surrogate', 'loss_ort_surrogate', 'chunk_total'), 0.0)
    pass2_latents = []
    for i, s in enumerate(slices):
        m = s.stop - s.start
        out = chunk_forward(torch, F, util, nets, x_spoof[s], x_live[s], label_spoof[s], eps, s, lam,
                            criterion_type, criterionL2, stats, B)
        total = chunk_objective(out, m, epoch, B)
        for k in LOCAL_TERMS:
            acc[k] += m / B * out[k].item()
        acc['loss_mmd_surrogate'] += out['loss_mmd_surrogate'].item()
        acc['loss_ort_surrogate'] += out['loss_ort_surrogate'].item()
        acc['chunk_total'] += total.item()
        if keep_latents:
            pass2_latents.append({k: out[k].detach().clone() for k in ('z_cls', 'z_nir', 'z_vis')})
        if on_chunk is not None:
            on_chunk(i, out)
        total.backward()
        del out, total
        if after_backward is not None:
            after_backward(i)
    if before_step is not None:
        before_step()
    optimizer.step()
    g = {k: acc[k] for k in LOCAL_TERMS}
    g['loss_mmd'] = stats['loss_mmd_global'].item()
    g['loss_ort'] = stats['loss_ort_global'].item()
    return {'global_losses': g,
            'epoch1_total': tg.total_loss(g, 1), 'postwarmup_total_algebraic_only': tg.total_loss(g, 2),
            'surrogate_sums': {'loss_mmd_surrogate': acc['loss_mmd_surrogate'],
                               'loss_ort_surrogate': acc['loss_ort_surrogate']},
            'sum_of_chunk_objectives': acc['chunk_total'], 'stats': stats, 'chunks': len(slices),
            'pass2_latents': pass2_latents}
