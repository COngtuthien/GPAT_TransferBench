"""Explicit-pair FP32 D->G iteration, bound to the immutable M6D4c contract.

Architecture modules and pinned crop callable are injected by the qualified
runtime. This module has no dataset, upstream loop or checkpoint writer.
"""
import hashlib
import json
from pathlib import Path

from .training_losses import generator_losses, discriminator_losses
from methods.common.config import ROOT

OVERLAY = 'configs/amendments/e05_training_runner_resolution.yaml'
OVERLAY_SHA = '8eee350c5642f2de38c92af366b99e80f7b6dab4404433ff88ca71da87b92828'
COUNTS = dict(Encoder=929960, Generator=3699233, ImageD=28859521, PatchD=24522145)


def require(ok, message):
    if not ok:
        raise RuntimeError('M6D4d: ' + message)


def training_contract(root=ROOT):
    raw = (root / OVERLAY).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OVERLAY_SHA, 'immutable M6D4c overlay')
    return json.loads(raw)


def parameter_groups(models):
    """Preserve every named identity; reject aliases within or across components."""
    require(set(models) == set(COUNTS), 'four architecture components')
    groups = {}
    for group, components in [('G', ('Encoder', 'Generator')), ('D', ('ImageD', 'PatchD'))]:
        entries = []
        for component in components:
            entries.extend((component + '.' + n, p) for n, p in
                           models[component].named_parameters(remove_duplicate=False))
        require(len({id(p) for _, p in entries}) == len(entries), 'duplicate parameter identity')
        groups[group] = entries
    require(not {id(p) for _, p in groups['G']} & {id(p) for _, p in groups['D']},
            'optimizer group overlap')
    return groups


def make_optimizers(groups):
    import torch
    return {key: torch.optim.Adam([p for _, p in entries], lr=1e-6, betas=(0.9, 0.999))
            for key, entries in groups.items()}


def explicit_pair(encoder, generator, src, tgt):
    import torch
    for x in (src, tgt):
        require(tuple(x.shape) == (1, 3, 256, 256) and x.dtype == torch.float32,
                'explicit batch-one FP32 pair')
    require(src.device == tgt.device and not torch.equal(src, tgt), 'distinct source/target')
    sp_src, gl_src = encoder(src)
    sp_tgt, gl_tgt = encoder(tgt)
    for x in (sp_src, sp_tgt):
        require(tuple(x.shape) == (1, 8, 128, 128), 'spatial code shape')
    for x in (gl_src, gl_tgt):
        require(tuple(x.shape) == (1, 2048), 'global code shape')
    rec, mix = generator(sp_src, gl_src), generator(sp_src, gl_tgt)
    for x in (rec, mix):
        require(x.shape == src.shape and x.dtype == torch.float32, 'RGB output')
    return dict(z_pat_src=sp_src, z_con_src=gl_src, z_pat_tgt=sp_tgt,
                z_con_tgt=gl_tgt, r_src=rec, m=mix)


def finite_gradients(named):
    import torch
    for name, p in named:
        require(p.grad is not None, 'missing expected gradient: ' + name)
        require(torch.isfinite(p.grad).all().item(), 'nonfinite gradient: ' + name)


class TrainingRunner:
    def __init__(self, models, crop, *, root=ROOT):
        import torch
        self.contract = training_contract(root)
        self.models, self.crop = models, crop
        self.groups = parameter_groups(models)
        for name, model in models.items():
            require(sum(p.numel() for p in model.parameters()) == COUNTS[name], 'live count ' + name)
            require(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()),
                    'initial trainable FP32 parameters')
        self.optimizers = make_optimizers(self.groups)
        self.benchmark_iteration = 0
        self.applications = dict(D=0, G=0)
        self.failed = False

    def trainable(self, group):
        for key, entries in self.groups.items():
            for _, p in entries:
                p.requires_grad_(key == group)

    def cycle(self, src, tgt, observer=None):
        """One complete cycle; failed partial iterations cannot silently be retried.

        Observer receives live tensors at stage boundaries and must not mutate
        them or consume RNG. It enables qualification without changing losses.
        """
        import torch
        require(not self.failed and self.benchmark_iteration < 4000, 'iteration boundary')
        require(not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
                'autocast forbidden')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                and not torch.backends.cudnn.benchmark, 'FP32 execution policy')
        def emit(stage, **state):
            if observer is not None:
                observer(stage, self, state)
        E, G, D, P = (self.models[n] for n in COUNTS)
        try:
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            self.trainable('D')
            emit('before_D')
            with torch.no_grad():
                paths = explicit_pair(E, G, src, tgt)
            losses, tensors = discriminator_losses(src, tgt, paths['r_src'], paths['m'], D, P, self.crop)
            emit('D_forward', paths=paths, losses=losses, tensors=tensors)
            require(all(torch.isfinite(x).item() for x in losses.values()), 'finite D losses')
            losses['L_D_total'].backward()
            finite_gradients(self.groups['D'])
            emit('D_backward')
            self.optimizers['D'].step()
            self.applications['D'] += 1
            emit('after_D')
            # Release the complete D graph before fresh post-update E/G evaluation.
            del paths, losses, tensors
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            self.trainable('G')
            emit('before_G')
            paths = explicit_pair(E, G, src, tgt)
            losses, tensors = generator_losses(src, tgt, paths['r_src'], paths['m'], D, P, self.crop)
            emit('G_forward', paths=paths, losses=losses, tensors=tensors)
            require(all(torch.isfinite(x).item() for x in losses.values()), 'finite G losses')
            losses['L_G_total'].backward()
            finite_gradients(self.groups['G'])
            emit('G_backward')
            self.optimizers['G'].step()
            self.applications['G'] += 1
            self.benchmark_iteration += 1
            emit('after_G')
        except BaseException:
            self.failed = True
            raise
        finally:
            for model in self.models.values():
                model.requires_grad_(True)
