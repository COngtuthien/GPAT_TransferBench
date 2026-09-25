"""M6D5d E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 (CONTROLLED_EXECUTION_ADAPTATION).

Owner-approved additive extension of M6D5c V1 (methods/dsdg/microbatch_execution.py,
imported unchanged) to the source-consistent final partial global batch. Pinned
train_generator.py:94-97 builds DataLoader(batch_size=args.batch_size, shuffle=True,
num_workers=args.workers, pin_memory=True) with no drop_last, so the frozen TRAIN
relation of 8838 rows yields 36 global batches of 240 and one of 198 per epoch.

V2 changes ONLY the chunk plan: ordered contiguous chunks of at most 20 over any
1 <= B <= 240 (B=198 -> 9x20 + 18). Every per-chunk formula is the V1 function with
the actual B: local mean terms weighted m/B, MMD surrogate lambda_mmd/(B*D), ORT
surrogate lambda_ort*sign/B, epsilon drawn once as [B,128] in order cls -> nir -> vis.
For B=240 the chunk plan equals V1's and the executed calls are V1's.

No dataset, DataLoader, image reader, manifest reader, epoch loop or checkpoint writer
exists here. Torch is passed in by the caller; importing this module is static.
"""
import json

from methods.common.config import ROOT
from methods.common.learned import PreparationError
from methods.dsdg import microbatch_execution as v1
from methods.dsdg import training_graph as tg

RESOLUTION_PATH = 'configs/amendments/e06c_m6d5d_tail_batch_execution_resolution.yaml'
EXECUTION_MODE = 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'
CLASSIFICATION = v1.CLASSIFICATION
NOMINAL_GLOBAL_BATCH = 240
MAX_MICROBATCH = 20
HDIM = v1.HDIM
EPS_ORDER = v1.EPS_ORDER
FINAL_BATCH_POLICY = 'EXACT_REMAINDER_NO_PADDING_NO_DUPLICATION'
EXPECTED_TRAIN_ROWS = 8838
DROP_LAST = False
# train_generator.py:94-97 (stripped), the only DataLoader construction in the pinned trainer
PINNED_DATALOADER = {
    94: 'train_loader = torch.utils.data.DataLoader(',
    95: 'GenDataset_s(img_root=args.img_root, list_file=args.train_list, attack_type=args.attack_type),',
    96: 'batch_size=args.batch_size, shuffle=True,',
    97: 'num_workers=args.workers, pin_memory=True)',
}


def load_resolution(path=None):
    """The owner overlay is the JSON subset of YAML; parsed with the standard library only."""
    return json.loads((ROOT / (path or RESOLUTION_PATH)).read_text())


def verify_pinned_dataloader(source_text):
    """[] when lines 94-97 are verbatim and the pinned trainer never passes drop_last."""
    lines = source_text.splitlines()
    bad = [{'line': n, 'expected': s, 'found': lines[n - 1].strip() if n <= len(lines) else None}
           for n, s in PINNED_DATALOADER.items() if n > len(lines) or lines[n - 1].strip() != s]
    if 'drop_last' in source_text:
        bad.append({'line': None, 'expected': 'no drop_last argument', 'found': 'drop_last present'})
    if source_text.count('DataLoader(') != 1:
        bad.append({'line': None, 'expected': 'exactly one DataLoader(', 'found': source_text.count('DataLoader(')})
    return bad


def epoch_batch_plan(rows=EXPECTED_TRAIN_ROWS, batch_size=NOMINAL_GLOBAL_BATCH, drop_last=DROP_LAST):
    """torch DataLoader batch sizes for one epoch; drop_last=False keeps the exact remainder."""
    if drop_last is not False:
        raise PreparationError('E06c keeps the source-consistent final partial batch (drop_last=False)')
    full, tail = divmod(rows, batch_size)
    sizes = [batch_size] * full + ([tail] if tail else [])
    return {'rows': rows, 'batch_size': batch_size, 'drop_last': False, 'full_batches': full,
            'final_batch': tail if tail else batch_size, 'batch_sizes': sizes,
            'optimizer_steps_per_epoch': len(sizes), 'rows_covered': sum(sizes)}


def chunk_slices_v2(B, max_microbatch=MAX_MICROBATCH):
    """Contiguous, ordered, exhaustive, disjoint slices of size <= max_microbatch; no empty chunk."""
    if not isinstance(B, int) or not isinstance(max_microbatch, int) or isinstance(B, bool):
        raise PreparationError('integer batch and microbatch required')
    if not 1 <= B <= NOMINAL_GLOBAL_BATCH or max_microbatch < 1:
        raise PreparationError(f'global batch must satisfy 1 <= B <= {NOMINAL_GLOBAL_BATCH}')
    return [slice(i, min(i + max_microbatch, B)) for i in range(0, B, max_microbatch)]


def execution_guard(resolution, *, max_microbatch=MAX_MICROBATCH, **forbidden):
    """V2 overlay binding; max microbatch 20 only; every memory workaround refused."""
    unknown = sorted(set(forbidden) - set(v1.FORBIDDEN_EXECUTION + ('drop_last', 'padding', 'duplication')))
    if unknown:
        raise PreparationError('unknown execution option(s): ' + ', '.join(unknown))
    used = sorted(k for k, v in forbidden.items() if v)
    if used:
        raise PreparationError('M6D5d forbids execution workaround(s): ' + ', '.join(used))
    r = resolution
    fixed = (r['execution_mode'], r['classification'], r['fidelity_class'], r['nominal_global_batch'],
             r['max_microbatch'], r['drop_last'], r['partial_global_batch_allowed'], r['final_batch_policy'],
             r['expected_train_rows'], r['full_batches_per_epoch'], r['final_batch_size'],
             r['optimizer_steps_per_epoch'], r['precision'], r['amp'], r['tf32'], r['activation_checkpointing'],
             r['cpu_parameter_offload'], r['optimizer_offload'], r['naive_microbatch_loss_averaging'],
             tuple(r['global_statistic_terms']), r['physical_batch_240'])
    if fixed != (EXECUTION_MODE, CLASSIFICATION, 'CONTROLLED_ADAPTATION', NOMINAL_GLOBAL_BATCH, MAX_MICROBATCH,
                 False, True, FINAL_BATCH_POLICY, EXPECTED_TRAIN_ROWS, 36, 198, 37, 'FP32', False, False, False,
                 False, False, 'forbidden', v1.GLOBAL_TERMS, 'OOM_RETAINED'):
        raise PreparationError('M6D5d owner overlay differs from the frozen tail-batch execution resolution')
    if max_microbatch != MAX_MICROBATCH:
        raise PreparationError('max microbatch is fixed at 20; tuning is forbidden (OOM is STOP_AND_REPORT)')
    return {'execution_mode': EXECUTION_MODE, 'classification': CLASSIFICATION,
            'nominal_global_batch': NOMINAL_GLOBAL_BATCH, 'max_microbatch': MAX_MICROBATCH, 'drop_last': False,
            'final_batch_policy': FINAL_BATCH_POLICY, 'precision': 'FP32', 'amp': False, 'tf32': False}


def draw_epsilon(torch, B, device='cuda'):
    """Exactly [B,128] x3 for the ACTUAL global batch (never [240,128] truncated); V1 draw call."""
    chunk_slices_v2(B)
    return v1.draw_epsilon(torch, B, HDIM, device)


def run_global_batch_v2(torch, F, util, nets, optimizer, x_spoof, x_live, label_spoof, eps, lam, criterion_type,
                        criterionL2, epoch, max_microbatch=MAX_MICROBATCH, on_pass1=None, on_chunk=None,
                        after_backward=None, before_step=None, keep_latents=False):
    """V1 run_global_batch with chunk_slices_v2: pass 1 over all B, zero_grad(set_to_none=False) once,
    one forward+backward per chunk, one optimizer.step(). Observers as in V1."""
    B = x_spoof.shape[0]
    if any(eps[k].shape[0] != B for k in EPS_ORDER):
        raise PreparationError('epsilon must be drawn for the actual global batch [B,128]')
    slices = chunk_slices_v2(B, max_microbatch)
    stats = v1.pass1_statistics(torch, nets, x_spoof, x_live, eps, slices, lam, keep_latents=keep_latents)
    if on_pass1 is not None:
        on_pass1(stats)
    optimizer.zero_grad(set_to_none=False)
    acc = dict.fromkeys(v1.LOCAL_TERMS + ('loss_mmd_surrogate', 'loss_ort_surrogate', 'chunk_total'), 0.0)
    pass2_latents = []
    for i, s in enumerate(slices):
        m = s.stop - s.start
        out = v1.chunk_forward(torch, F, util, nets, x_spoof[s], x_live[s], label_spoof[s], eps, s, lam,
                               criterion_type, criterionL2, stats, B)
        total = v1.chunk_objective(out, m, epoch, B)
        for k in v1.LOCAL_TERMS:
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
    g = {k: acc[k] for k in v1.LOCAL_TERMS}
    g['loss_mmd'] = stats['loss_mmd_global'].item()
    g['loss_ort'] = stats['loss_ort_global'].item()
    return {'global_losses': g, 'global_batch': B, 'chunk_sizes': [s.stop - s.start for s in slices],
            'epoch1_total': tg.total_loss(g, 1), 'postwarmup_total_algebraic_only': tg.total_loss(g, 2),
            'surrogate_sums': {'loss_mmd_surrogate': acc['loss_mmd_surrogate'],
                               'loss_ort_surrogate': acc['loss_ort_surrogate']},
            'sum_of_chunk_objectives': acc['chunk_total'], 'stats': stats, 'chunks': len(slices),
            'pass2_latents': pass2_latents}
