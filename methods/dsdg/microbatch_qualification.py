"""M6D5c E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1 qualification, never a training runner.

Two modes, each one fresh GPU process:

  reference  B=4 synthetic batch, identical initial bytes / inputs / epsilon:
             Path A = unchanged M6D5b full-batch graph (training_graph.training_forward),
             Path B = two-pass microbatch graph with microbatch 2. Compares every loss,
             the epoch-1 total, optimizer-owned gradients and one Adam update.
  b240       one SYNTHETIC global batch of 240 = 12 x 20: pass-1 statistics, 12 backward
             calls, one Adam step. OOM at microbatch 20 is STOP_OOM (no smaller retry).

No dataset, image reader, manifest, epoch loop or checkpoint writer exists here.
M6D5a/M6D5b helpers are imported unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.dsdg import runtime as m6d5a  # noqa: E402  (unchanged M6D5a helpers)
from methods.dsdg import training_graph as tg  # noqa: E402  (unchanged M6D5b transcription)
from methods.dsdg import training_qualification as m6d5b  # noqa: E402  (unchanged M6D5b helpers)
from methods.dsdg import microbatch_execution as mb  # noqa: E402

SEED = 60503  # qualification only; not an experiment seed (42/1337/2026)
EPOCH = 1     # warmup branch of train_generator.py:174
BUILD_PARTS = ('builds', 'e06c_dsdg_m6d5c')
LOCK = ROOT / 'environments/e06c.lock.json'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
CLEAN_GPU_MAX_USED_MIB = 1024
REFERENCE_BATCH, REFERENCE_MICROBATCH = 4, 2
# M6D5b B=240 synthetic inputs (outputs/audit/M6D5B_E06C_SYNTHETIC_PROCESS_1.json); same analytic construction.
M6D5B_INPUT_SHA256 = {'x_spoof': 'e4ec45dbe45375cb21c8c63004b266faa7bba246df4f058bc874af6acd1012c8',
                      'x_live': 'b4fea619f2d81a0b4f46dd6db2cec9e4bb953047d312c1022ed1ec93bdcd42c0'}
EXIT = {'PASS': 0, 'STOP_OOM': 3, 'STOP_RESOURCE_CONTAMINATION': 4, 'STOP_REFERENCE_GATE': 5}


def require(value, message):
    if not value:
        raise RuntimeError('M6D5c gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tsha(t):
    return sha(t.detach().cpu().contiguous().numpy().tobytes())


def instrument(torch, counters):
    """Count backward/Adam/zero_grad; forbid autograd.grad, torch.save and activation checkpointing."""
    tensor_backward, autograd_backward = torch.Tensor.backward, torch.autograd.backward
    adam_init, zero_grad = torch.optim.Adam.__init__, torch.optim.Optimizer.zero_grad

    def backward(self, *a, **k):
        counters['backward_calls'] += 1
        return tensor_backward(self, *a, **k)

    def autograd(*a, **k):
        counters['autograd_backward_calls'] += 1
        return autograd_backward(*a, **k)

    def adam(self, *a, **k):
        counters['optimizer_constructions'] += 1
        return adam_init(self, *a, **k)

    def zero(self, set_to_none=True):
        counters['zero_grad_calls'].append({'set_to_none': set_to_none,
                                            'backward_calls_before': counters['backward_calls']})
        return zero_grad(self, set_to_none=set_to_none)

    def forbid(name):
        def forbidden(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('M6D5c forbids ' + name)
        return forbidden
    torch.Tensor.backward, torch.autograd.backward = backward, autograd
    torch.optim.Adam.__init__, torch.optim.Optimizer.zero_grad = adam, zero
    torch.autograd.grad = forbid('autograd_grad_calls')
    torch.save = forbid('checkpoint_saves')
    torch.utils.checkpoint.checkpoint = forbid('activation_checkpoint_calls')


def new_counters():
    c = dict.fromkeys(('optimizer_constructions', 'optimizer_step_entries', 'optimizer_applications',
                       'backward_calls', 'autograd_backward_calls', 'autograd_grad_calls', 'checkpoint_saves',
                       'activation_checkpoint_calls'), 0)
    c['zero_grad_calls'] = []
    return c


def count_steps(optimizer, counters):
    optimizer.register_step_pre_hook(lambda *a: counters.__setitem__(
        'optimizer_step_entries', counters['optimizer_step_entries'] + 1))
    optimizer.register_step_post_hook(lambda *a: counters.__setitem__(
        'optimizer_applications', counters['optimizer_applications'] + 1))


def pinned_util_shim(util, eps):
    """Pinned util with reparameterize replaying eps in pinned draw order (Path A only)."""
    draws = iter(mb.EPS_ORDER)

    class U:
        kl_loss, reconstruction_loss, rgb2gray = util.kl_loss, util.reconstruction_loss, util.rgb2gray

        @staticmethod
        def reparameterize(mu, logvar):
            return mb.replay_latent(eps[next(draws)], mu, logvar)
    return U


def epsilon_draw_proof(torch, util, batch):
    """draw_epsilon from CUDA state S == pinned reparameterize(0, 0) x3 from the same S (bitwise)."""
    state = torch.cuda.get_rng_state()
    eps = mb.draw_epsilon(torch, batch)
    after_ours = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    zero = torch.zeros((batch, mb.HDIM), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        pinned = [util.reparameterize(zero, zero) for _ in mb.EPS_ORDER]
    after_pinned = torch.cuda.get_rng_state()
    proof = {'order': list(mb.EPS_ORDER), 'shape': [batch, mb.HDIM], 'dtype': str(eps['cls'].dtype),
             'bitwise_equal_to_pinned_reparameterize_draws': {k: bool(torch.equal(eps[k], p))
                                                               for k, p in zip(mb.EPS_ORDER, pinned)},
             'rng_state_after_equal': bool(torch.equal(after_ours, after_pinned)),
             'sha256': {k: tsha(eps[k]) for k in mb.EPS_ORDER},
             'moments': {k: {'mean': float(eps[k].double().mean()), 'std': float(eps[k].double().std())}
                         for k in mb.EPS_ORDER},
             'method': 'torch.empty((B,128), float32, cuda).normal_() == torch.cuda.FloatTensor(size).normal_()'}
    require(all(proof['bitwise_equal_to_pinned_reparameterize_draws'].values()) and proof['rng_state_after_equal'],
            'epsilon draws equal pinned reparameterize draws')
    return eps, proof


def setup(build, mode):
    """Shared identity/firewall/environment gates (M6D5b order); returns context or a STOP result."""
    firewall = m6d5a.Firewall(build)
    sys.addaudithook(firewall)
    source, adapter = m6d5a.source_identity()
    cfg = adapter.config
    lam = tg.frozen_lambdas(cfg)
    resolution = mb.load_resolution()
    policy = mb.execution_guard(resolution)
    try:
        adapter.validate_batch(physical_batch_size=mb.MICROBATCH, gradient_accumulation_steps=mb.MICROBATCHES)
        generic_refused = False
    except Exception:
        generic_refused = True
    require(generic_refused, 'generic adapter accumulation remains refused')
    root = Path(source['root']) / cfg['source']['relevant_path']
    mismatch = tg.verify_pinned_statements((root / 'train_generator.py').read_text())
    require(not mismatch, 'pinned statements')
    lock_raw = LOCK.read_bytes()
    require(sha(lock_raw) == LOCK_SHA, 'environment lock SHA256')
    lock = json.loads(lock_raw)
    require(source['commit'] == lock['source']['commit'] and source['files_sha256'] == lock['source']['files_sha256'],
            'source identity equals environment lock')
    result = {'mode': mode, 'execution_mode': mb.EXECUTION_MODE, 'classification': mb.CLASSIFICATION,
              'draw_policy': mb.DRAW_POLICY, 'diagnostic_seed': SEED, 'epoch_semantics': EPOCH,
              'contract': m6d5a.contract(adapter), 'lambdas': lam, 'execution_policy': policy,
              'resolution_sha256': sha((ROOT / mb.RESOLUTION_PATH).read_bytes()),
              'adapter_generic_accumulation_refused': generic_refused,
              'pinned_statements_verified': len(tg.PINNED_STATEMENTS), 'pinned_statement_mismatches': mismatch,
              'source_before': source, 'asset_before': m6d5a.asset_identity(adapter),
              'environment_lock_sha256': sha(lock_raw), 'gpu_before': m6d5b.gpu_snapshot()}
    foreign = [p for p in result['gpu_before']['compute_processes'] if int(p['pid']) != os.getpid()]
    result['resource_clean'] = not foreign and result['gpu_before']['used_mib'] <= CLEAN_GPU_MAX_USED_MIB
    if not result['resource_clean']:
        result.update(status='STOP_RESOURCE_CONTAMINATION', foreign_compute_processes=foreign)
        return result, None
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint  # noqa: F401  (so it can be forbidden)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    counters = new_counters()
    instrument(torch, counters)
    env = m6d5a.environment()
    result['environment_before'] = env
    lock_diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
                 if k != 'launch_environment' and env.get(k) != v}
    require(not lock_diff, 'runtime identity equals environment lock: ' + json.dumps(lock_diff))
    require(not env['precision']['matmul_tf32'] and not env['precision']['cudnn_tf32'] and
            not env['precision']['autocast_cuda'] and env['precision']['default_dtype'] == 'torch.float32',
            'FP32, no TF32, no autocast')
    result['environment_lock_identity_match'] = sorted(k for k in lock['identity'] if k != 'launch_environment')
    torch.cuda.reset_peak_memory_stats()
    return result, dict(torch=torch, F=F, firewall=firewall, source=source, adapter=adapter, cfg=cfg, lam=lam,
                        root=root, counters=counters, env=env)


def finish(result, ctx):
    torch, counters, firewall = ctx['torch'], ctx['counters'], ctx['firewall']
    result['counters'] = counters
    result['gpu_after'] = m6d5b.gpu_snapshot()
    result['source_after'] = m6d5a.source_identity()[0]
    require(result['source_after'] == ctx['source'], 'source-cache integrity after execution')
    result['asset_after'] = m6d5a.asset_identity(ctx['adapter'])
    require(result['asset_after'] == result['asset_before'], 'LightCNN unchanged')
    result['environment_after'] = m6d5a.environment()
    require(result['environment_after'] == ctx['env'], 'environment stable')
    require(counters['autograd_grad_calls'] == counters['checkpoint_saves']
            == counters['activation_checkpoint_calls'] == 0, 'no forbidden call')
    require(not torch.is_autocast_enabled('cuda'), 'autocast off')
    result.update(checkpoint_created=False, benchmark_training=False, benchmark_data_access=False, TEST_access=False,
                  synthetic_bank=False, fidelity='CONTROLLED_ADAPTATION', compatibility_patch='NONE',
                  firewall={'denied': firewall.denied, 'event_counts': firewall.events,
                            'lightcnn_read_opens': firewall.lightcnn_opens})
    require(not firewall.denied, 'firewall denials')
    return result


def models_ready(torch, ctx, result, networks):
    models, result['binding'] = m6d5b.build_models(torch, networks, ctx['root'], ctx['cfg'])
    result['lightcnn'] = m6d5a.load_lightcnn(torch, models['netIP'])
    models['netE_nir'].train(); models['netE_vis'].train(); models['netG'].train(); models['netIP'].eval()
    params = {n: m6d5a.parameter_report(m) for n, m in models.items()}
    for n, (count, tensors) in m6d5a.EXPECTED_PARAMETERS.items():
        require((params[n]['parameters'], params[n]['parameter_tensors']) == (count, tensors), 'live count ' + n)
    require(params['netIP']['requires_grad_parameters'] == 0 and not models['netIP'].training, 'netIP frozen eval')
    result['parameters_initial'] = params
    result['modes'] = {n: m.training for n, m in models.items()}
    result['batch_coupling'] = batch_coupling(torch, models)
    return models, params


def batch_coupling(torch, models):
    """Evidence that every layer is per-sample: no BatchNorm, InstanceNorm without running stats."""
    rows = {}
    for n, m in models.items():
        kinds = {}
        for mod in m.modules():
            k = type(mod).__name__
            if 'Norm' in k or 'Dropout' in k:
                kinds.setdefault(k, set()).add((getattr(mod, 'affine', None), getattr(mod, 'track_running_stats', None)))
        rows[n] = {k: sorted(map(list, v)) for k, v in kinds.items()}
        rows[n]['buffers'] = len(list(m.buffers()))
    bn = [n for n, m in models.items() for mod in m.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    inorm = [mod for m in models.values() for mod in m.modules() if isinstance(mod, torch.nn.InstanceNorm2d)]
    require(not bn and all(not i.track_running_stats and not i.affine for i in inorm), 'no batch-coupled layers')
    return {'per_model': rows, 'batchnorm_modules': len(bn), 'instancenorm_modules': len(inorm),
            'instancenorm_track_running_stats': False, 'instancenorm_affine': False,
            'consequence': 'per-sample layers; MMD and orthogonality are the only batch-coupled terms'}


def owned_params(models):
    return [(n + '.' + k, p) for n in tg.OPTIMIZER_OWNED for k, p in models[n].named_parameters()]


def compare_vectors(torch, a_list, b_list):
    """Aggregate cosine / relative-L2 over concatenated tensors (float64 accumulation)."""
    dot = na = nb = nd = 0.0
    sign_agree = nonzero = 0
    for a, b in zip(a_list, b_list):
        a64, b64 = a.double(), b.double()
        dot += float((a64 * b64).sum()); na += float(a64.pow(2).sum()); nb += float(b64.pow(2).sum())
        nd += float((a64 - b64).pow(2).sum())
        mask = (a != 0) | (b != 0)
        nonzero += int(mask.sum()); sign_agree += int(((torch.sign(a) == torch.sign(b)) & mask).sum())
    return {'cosine': dot / max((na * nb) ** 0.5, 1e-300), 'relative_l2': (nd ** 0.5) / max(na ** 0.5, 1e-300),
            'l2_a': na ** 0.5, 'l2_b': nb ** 0.5, 'sign_agreement_fraction': sign_agree / max(nonzero, 1),
            'nonzero_elements': nonzero}


# ================================================================= reference (B=4)
def reference(build):
    result, ctx = setup(build, 'reference')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    gates = mb.load_resolution()['qualification']['reference_check']['gates']
    B, m = REFERENCE_BATCH, REFERENCE_MICROBATCH
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: {k: v.detach().clone() for k, v in mdl.state_dict().items()} for n, mdl in models.items()}
        criterion_type, criterionL2 = tg.criteria(torch)
        x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
        label = torch.zeros(B, dtype=torch.long, device='cuda')
        eps, result['epsilon'] = epsilon_draw_proof(torch, util, B)
        result['inputs'] = {'x_spoof': m6d5b.summary(x_spoof), 'x_live': m6d5b.summary(x_live), 'batch': B,
                            'microbatch': m, 'chunks': B // m}
        owned = owned_params(models)

        # ---- Path A: unchanged M6D5b full-batch graph, pinned order forward -> zero_grad -> backward -> step
        opt_a = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        count_steps(opt_a, counters)
        out = tg.training_forward(torch, F, pinned_util_shim(util, eps), nets, x_spoof, x_live, label, lam,
                                  criterion_type, criterionL2)
        loss_a = tg.total_loss(out, EPOCH)
        path_a = {'losses': {k: out[k].item() for k in tg.LOSS_TERMS}, 'epoch1_total': loss_a.item()}
        z_a = {k: out[k].detach().clone() for k in ('z_cls', 'z_nir', 'z_vis')}
        opt_a.zero_grad()
        loss_a.backward()
        grads_a = [p.grad.detach().clone() for _, p in owned]
        path_a['owned_grad_non_none'] = sum(p.grad is not None for _, p in owned)
        opt_a.step()
        delta_a = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
        del out, loss_a, opt_a

        # ---- restore identical initial bytes; fresh gradients
        for n, mdl in models.items():
            mdl.load_state_dict(initial[n])
            for p in mdl.parameters():
                p.grad = None
        restored = {n: m6d5a.parameter_report(mdl)['sha256'] for n, mdl in models.items()}
        require(restored == {n: p['sha256'] for n, p in params.items()}, 'identical initial model bytes restored')

        # ---- Path B: two-pass global-statistic-preserving microbatch graph
        opt_b = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        count_steps(opt_b, counters)
        grads_b = []
        run = mb.run_global_batch(torch, F, util, nets, opt_b, x_spoof, x_live, label, eps, lam, criterion_type,
                                  criterionL2, EPOCH, microbatch=m, keep_latents=True,
                                  before_step=lambda: grads_b.extend(p.grad.detach().clone() for _, p in owned))
        delta_b = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
        z_b1 = {k: torch.cat([c[k] for c in run['stats']['latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
        z_b2 = {k: torch.cat([c[k] for c in run['pass2_latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
    g = run['global_losses']
    rel = lambda a, b: abs(a - b) / max(abs(a), 1e-30)  # noqa: E731
    grad_cmp = compare_vectors(torch, grads_a, grads_b)
    step_cmp = compare_vectors(torch, delta_a, delta_b)
    per_module = {}
    for n in tg.OPTIMIZER_OWNED:
        idx = [i for i, (k, _) in enumerate(owned) if k.startswith(n + '.')]
        per_module[n] = {'gradient': compare_vectors(torch, [grads_a[i] for i in idx], [grads_b[i] for i in idx]),
                         'update': compare_vectors(torch, [delta_a[i] for i in idx], [delta_b[i] for i in idx])}
    result['path_a_full_batch'] = path_a
    result['path_b_microbatch'] = {'losses': g, 'epoch1_total': run['epoch1_total'],
                                   'surrogate_sums': run['surrogate_sums'],
                                   'sum_of_chunk_objectives': run['sum_of_chunk_objectives'],
                                   'chunks': run['chunks'], 'owned_grad_non_none': len(grads_b)}
    result['comparison'] = {
        'loss_abs_diff': {k: abs(path_a['losses'][k] - g[k]) for k in tg.LOSS_TERMS},
        'loss_rel_diff': {k: rel(path_a['losses'][k], g[k]) for k in tg.LOSS_TERMS},
        'total_rel_error': rel(path_a['epoch1_total'], run['epoch1_total']),
        'sum_of_chunk_objectives_rel_error': rel(path_a['epoch1_total'], run['sum_of_chunk_objectives']),
        'gradient': grad_cmp, 'update': step_cmp, 'per_module': per_module,
        'latents_max_abs_diff': {'path_a_vs_pass1': {k: float((z_a[k] - z_b1[k]).abs().max()) for k in z_a},
                                 'path_a_vs_pass2': {k: float((z_a[k] - z_b2[k]).abs().max()) for k in z_a}}}
    finite = all(v == v and abs(v) != float('inf') for v in list(path_a['losses'].values()) + list(g.values()))
    result['gates'] = {
        'scalar_losses_finite': finite,
        'total_loss_relative_error': result['comparison']['total_rel_error'] <= gates['total_loss_relative_error_max'],
        'aggregate_gradient_cosine': grad_cmp['cosine'] >= gates['aggregate_gradient_cosine_min'],
        'aggregate_gradient_relative_l2': grad_cmp['relative_l2'] <= gates['aggregate_gradient_relative_l2_max'],
        'post_step_parameter_direction': step_cmp['cosine'] >= gates['post_step_parameter_delta_cosine_min'],
        'owned_gradient_coverage': path_a['owned_grad_non_none'] == len(grads_b) == 55}
    result['gate_thresholds'] = gates
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['status'] = 'PASS' if all(result['gates'].values()) else 'STOP_REFERENCE_GATE'
    finish(result, ctx)
    require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_calls'])
            == (2, 2, 1 + B // m), 'reference accounting: 2 optimizers, 2 steps, 1 + chunks backward')
    result.update(reference_global_batch=B, reference_microbatch=m, optimizer_constructions=2,
                  optimizer_applications=2, backward_calls=1 + B // m)
    return result


# ================================================================= B=240
def b240(build):
    result, ctx = setup(build, 'b240')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    mem, timing, hooks_seen, rngs = [], {}, [], {}
    peaks = {'allocated': 0, 'reserved': 0}
    phase = {'name': 'model_construction', 't': time.time()}

    def mark(name):
        timing[phase['name']] = round(time.time() - phase['t'], 4)
        phase.update(name=name, t=time.time())

    def snap(label, reset=False):
        torch.cuda.synchronize()
        row = m6d5b.memory(torch, label)
        peaks['allocated'] = max(peaks['allocated'], row['peak_allocated_bytes'])
        peaks['reserved'] = max(peaks['reserved'], row['peak_reserved_bytes'])
        mem.append(row)
        if reset:
            torch.cuda.reset_peak_memory_stats()

    def on_pass1(stats):
        require(stats['grad_enabled_inside'] is False and not stats['delta'].requires_grad, 'pass 1 under no_grad')
        require(counters['backward_calls'] == 0 and not counters['zero_grad_calls'], 'pass 1 has no backward')
        snap('pass1_peak', reset=True)
        rngs['after_pass1'] = m6d5b.rng(torch)
        mark('pass2')

    def on_chunk(i, out):
        row = {'chunk': i, 'rows': [i * mb.MICROBATCH, (i + 1) * mb.MICROBATCH],
               'nir_fc_requires_grad': out['nir_fc'].requires_grad, 'vis_fc_requires_grad': out['vis_fc'].requires_grad,
               'rec_nir_fc_requires_grad': out['rec_nir_fc'].requires_grad,
               'rec_vis_fc_requires_grad': out['rec_vis_fc'].requires_grad,
               'rec_nir_fc_grad_fn': type(out['rec_nir_fc'].grad_fn).__name__,
               'losses': {k: out[k].item() for k in mb.LOCAL_TERMS + ('loss_mmd_surrogate', 'loss_ort_surrogate')},
               'grads': {}}

        def hook(name):
            def capture(g):
                d = g.detach()
                row['grads'][name] = {'shape': list(g.shape), 'finite': bool(torch.isfinite(d).all()),
                                      'nonzero': int(torch.count_nonzero(d)), 'l2': float(torch.linalg.vector_norm(d))}
            return capture
        for name in ('rec_nir', 'rec_vis', 'rec_nir_fc', 'rec_vis_fc', 'pre_spoof', 'z_cls', 'z_nir', 'z_vis'):
            out[name].register_hook(hook(name))
        hooks_seen.append(row)

    def after_backward(i):
        snap(f'pass2_chunk_{i:02d}_peak', reset=True)

    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: m6d5b.cpu_copy(models[n]) for n in models}
        snap('after_model_construction')
        run = optimizer = None
        try:
            mark('optimizer_construction')
            optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                           ctx['cfg']['optimizer']['learning_rate'])
            count_steps(optimizer, counters)
            result['optimizer'] = m6d5b.optimizer_evidence(torch, optimizer, models)
            criterion_type, criterionL2 = tg.criteria(torch)
            mark('input_allocation')
            x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, mb.GLOBAL_BATCH, 'cpu'))
            label = torch.zeros(mb.GLOBAL_BATCH, dtype=torch.long, device='cuda')
            rows = [sha(r.cpu().numpy().tobytes()) for x in (x_spoof, x_live) for r in x]
            result['inputs'] = {'x_spoof': m6d5b.summary(x_spoof), 'x_live': m6d5b.summary(x_live),
                                'label_spoof': m6d5b.summary(label), 'distinct_rows': len(set(rows)), 'rows': len(rows),
                                'label': 'SYNTHETIC_GLOBAL_BATCH_240',
                                'construction': 'methods/dsdg/training_graph.py::synthetic_pair (unchanged M6D5b)'}
            result['inputs']['equal_to_m6d5b_inputs'] = {
                k: result['inputs'][k]['sha256'] == M6D5B_INPUT_SHA256[k] for k in M6D5B_INPUT_SHA256}
            require(len(set(rows)) == 2 * mb.GLOBAL_BATCH and all(result['inputs']['equal_to_m6d5b_inputs'].values()),
                    '480 distinct synthetic rows equal to the M6D5b inputs')
            snap('after_optimizer_and_input_allocation')
            mark('epsilon_allocation')
            rngs['before_epsilon'] = m6d5b.rng(torch)
            eps, result['epsilon'] = epsilon_draw_proof(torch, util, mb.GLOBAL_BATCH)
            rngs['after_epsilon'] = m6d5b.rng(torch)
            snap('after_epsilon_allocation', reset=True)
            grads_before = sum(p.grad is not None for mdl in models.values() for p in mdl.parameters())
            mark('pass1')
            state = {}

            def before_step():
                snap('before_optimizer_step')
                state['inventory'] = {n: m6d5b.grad_inventory(mdl) for n, mdl in models.items()}
                state['grad_sha256'] = {k: tsha(p.grad) for k, p in owned_params(models)}
                rngs['before_step'] = m6d5b.rng(torch)
                mark('optimizer_step')
            run = mb.run_global_batch(torch, F, util, nets, optimizer, x_spoof, x_live, label, eps, lam,
                                      criterion_type, criterionL2, EPOCH, on_pass1=on_pass1, on_chunk=on_chunk,
                                      after_backward=after_backward, before_step=before_step, keep_latents=True)
            snap('after_optimizer_step')
            rngs['after_step'] = m6d5b.rng(torch)
            mark('post_step_audit')
        except torch.OutOfMemoryError as error:
            stats = torch.cuda.memory_stats()
            result['oom'] = {'phase': phase['name'], 'error_type': type(error).__name__, 'error': str(error),
                             'chunks_completed': sum(1 for r in mem if r['phase'].startswith('pass2_chunk')),
                             'allocator': {k: stats.get(k) for k in (
                                 'allocated_bytes.all.current', 'allocated_bytes.all.peak',
                                 'reserved_bytes.all.current', 'reserved_bytes.all.peak',
                                 'num_alloc_retries', 'num_ooms', 'inactive_split_bytes.all.current')},
                             'memory_at_oom': m6d5b.memory(torch, 'at_oom'), 'gpu_at_oom': m6d5b.gpu_snapshot(),
                             'memory_summary': torch.cuda.memory_summary(),
                             'optimizer_application_occurred': counters['optimizer_applications'] > 0}
            run = None
            torch.cuda.empty_cache()
            change = {n: m6d5b.parameter_change(initial[n], models[n]) for n in models}
            result['oom']['any_parameter_changed'] = any(c['changed_elements'] for c in change.values())
        result['phase_seconds'] = timing
        result['memory_by_phase'] = mem
        result['overall_peak'] = {'allocated_bytes': peaks['allocated'], 'reserved_bytes': peaks['reserved']}
        result['rng'] = rngs
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if 'oom' in result:
            result['status'] = 'STOP_OOM'
        else:
            result.update(post_step(torch, models, optimizer, run, state, initial, params, hooks_seen, rngs,
                                    grads_before, counters, lam))
            result['status'] = 'PASS'
    finish(result, ctx)
    passed = result['status'] == 'PASS'
    if passed:
        require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_calls'],
                 counters['autograd_backward_calls']) == (1, 1, 12, 12), 'one optimizer, one step, 12 backward calls')
    result.update(global_batch_size=mb.GLOBAL_BATCH, microbatch_size=mb.MICROBATCH,
                  microbatches_per_step=mb.MICROBATCHES, optimizer_constructions=counters['optimizer_constructions'],
                  optimizer_applications=counters['optimizer_applications'], backward_calls=counters['backward_calls'],
                  physical_batch_240='OOM_RETAINED', global_batch_240_executed=passed)
    return result


def post_step(torch, models, optimizer, run, state, initial, params, hooks_seen, rngs, grads_before, counters, lam):
    r = {}
    g = run['global_losses']
    require(all(v == v and abs(v) != float('inf') for v in g.values()), 'finite global losses')
    require(g['loss_cls'] == 0.0 and g['loss_pair'] == 0.0, 'cls/pair exactly zero')
    st = run['stats']
    z1 = {k: torch.cat([c[k] for c in st['latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
    z2 = {k: torch.cat([c[k] for c in run['pass2_latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
    zn, zv, zc = (z1[k].double().cpu() for k in ('z_nir', 'z_vis', 'z_cls'))
    delta64 = zn.mean(0) - zv.mean(0)
    ort64 = (zc * zn).sum(1).mean()
    chunks = mb.chunk_slices()
    naive_mmd = sum(float(lam['lambda_mmd'] * (zn[s].mean(0) - zv[s].mean(0)).abs().mean()) for s in chunks) / len(chunks)
    naive_ort = sum(float(lam['lambda_ort'] * (zc[s] * zn[s]).sum(1).mean().abs()) for s in chunks) / len(chunks)
    p2 = {k: v.double().cpu() for k, v in z2.items()}
    delta_p2 = p2['z_nir'].mean(0) - p2['z_vis'].mean(0)
    r['losses'] = {'global_components': g, 'epoch1_total_reconstructed': run['epoch1_total'],
                   'postwarmup_total_algebraic_only': run['postwarmup_total_algebraic_only'],
                   'postwarmup_executed': False,
                   'sum_of_chunk_objectives': run['sum_of_chunk_objectives'],
                   'epoch1_formula': 'rec + 0.01*(kl + mmd + ip + pair + cls + ort)',
                   'per_chunk': [h['losses'] for h in hooks_seen]}
    r['global_statistics'] = {
        'delta': {'shape': list(st['delta'].shape), 'sha256': tsha(st['delta']), 'l1_mean': float(st['delta'].abs().mean()),
                  'zero_coordinates': int((st['delta'] == 0).sum()), 'positive': int((st['delta'] > 0).sum()),
                  'negative': int((st['delta'] < 0).sum())},
        'ort_mean': st['ort_mean'].item(), 'ort_sign': st['ort_sign'].item(),
        'mmd_sign_sha256': tsha(st['mmd_sign']),
        'loss_mmd_global_fp32': st['loss_mmd_global'].item(), 'loss_ort_global_fp32': st['loss_ort_global'].item(),
        'loss_mmd_global_float64_recomputed': float(lam['lambda_mmd'] * delta64.abs().mean()),
        'loss_ort_global_float64_recomputed': float(lam['lambda_ort'] * ort64.abs()),
        'surrogate_sums_pass2': run['surrogate_sums'],
        'diagnostic_naive_microbatch_average_not_used': {'loss_mmd': naive_mmd, 'loss_ort': naive_ort},
        'pass2_latents_bitwise_equal_pass1': {k: bool(torch.equal(z1[k], z2[k])) for k in z1},
        'pass2_vs_pass1_latent_max_abs_diff': {k: float((z1[k] - z2[k]).abs().max()) for k in z1},
        'pass2_recomputed_sign_mismatches': int((torch.sign(delta_p2) != torch.sign(delta64)).sum()),
        'pass2_recomputed_ort_sign_equal': bool(torch.sign((p2['z_cls'] * p2['z_nir']).sum(1).mean()) == torch.sign(ort64))}
    gs = r['global_statistics']
    require(abs(gs['loss_mmd_global_fp32'] - gs['loss_mmd_global_float64_recomputed'])
            <= 1e-4 * max(1.0, gs['loss_mmd_global_float64_recomputed']), 'MMD global float64 agreement')
    require(abs(gs['loss_ort_global_fp32'] - gs['loss_ort_global_float64_recomputed'])
            <= 1e-4 * max(1.0, gs['loss_ort_global_float64_recomputed']), 'ORT global float64 agreement')
    for k, v in (('loss_mmd_surrogate', gs['loss_mmd_global_fp32']), ('loss_ort_surrogate', gs['loss_ort_global_fp32'])):
        require(abs(run['surrogate_sums'][k] - v) <= 1e-4 * max(1.0, abs(v)), 'surrogate sum equals global ' + k)
    r['latents'] = {k: m6d5b.summary(v) for k, v in z1.items()}
    r['chunk_evidence'] = hooks_seen
    require(len(hooks_seen) == mb.MICROBATCHES, '12 pass-2 chunks')
    for h in hooks_seen:
        require(not h['nir_fc_requires_grad'] and not h['vis_fc_requires_grad'], 'target features under no_grad')
        require(h['rec_nir_fc_requires_grad'] and h['rec_vis_fc_requires_grad'], 'reconstructed features keep graph')
        require(all(h['grads'][k]['nonzero'] > 0 and h['grads'][k]['finite'] for k in ('rec_nir', 'rec_vis', 'rec_nir_fc')),
                'gradient through frozen LightCNN into reconstruction, chunk %d' % h['chunk'])
        require(h['grads']['pre_spoof']['nonzero'] == 0 and h['grads']['pre_spoof']['shape'] == [20, 1],
                'one-logit CE zero gradient')
    inv = state['inventory']
    r['gradient_inventory'] = inv
    r['gradient_sha256'] = state['grad_sha256']
    for n in tg.OPTIMIZER_OWNED:
        require(inv[n]['non_none'] == inv[n]['tensors'] == inv[n]['finite'] and inv[n]['nonzero_tensors'] > 0,
                'finite non-None connected gradients ' + n)
    require(sum(inv[n]['non_none'] for n in tg.OPTIMIZER_OWNED) == 55, '55/55 owned gradient coverage')
    require(inv['netIP']['non_none'] == 0 and inv['netCls']['nonzero_elements'] == 0, 'netIP no grad; netCls zero grad')
    zg = counters['zero_grad_calls']
    r['zero_grad'] = {'calls': zg, 'set_to_none': False, 'grads_before_first_zero_grad': grads_before,
                      'called_before_first_backward': bool(zg) and zg[0]['backward_calls_before'] == 0}
    require(len(zg) == 1 and zg[0] == {'set_to_none': False, 'backward_calls_before': 0},
            'zero_grad(set_to_none=False) exactly once before the first microbatch')
    require(rngs['after_epsilon'] == rngs['after_pass1'] == rngs['before_step'] == rngs['after_step'],
            'no RNG draw after the single epsilon draw (no redraw in pass 2)')
    r['rng_no_redraw'] = True
    owned = [p for n in tg.OPTIMIZER_OWNED for p in models[n].parameters()]
    steps = sorted({float(s['step']) for s in optimizer.state.values()})
    r['optimizer_state'] = {'entries': len(optimizer.state), 'step_values': steps,
                            'only_owned': set(optimizer.state) == set(owned),
                            'keys': sorted({k for s in optimizer.state.values() for k in s}),
                            'exp_avg_sha256': {k: tsha(optimizer.state[p]['exp_avg']) for k, p in owned_params(models)},
                            'finite_moments': all(bool(torch.isfinite(s['exp_avg']).all() and
                                                       torch.isfinite(s['exp_avg_sq']).all())
                                                  for s in optimizer.state.values())}
    require(len(optimizer.state) == 55 and r['optimizer_state']['only_owned'] and steps == [1.0] and
            r['optimizer_state']['finite_moments'], 'Adam state 55 x step 1, owned only')
    change = {n: m6d5b.parameter_change(initial[n], models[n]) for n in models}
    r['parameter_change'] = change
    require(all(change[n]['changed_tensors'] > 0 for n in tg.OPTIMIZER_OWNED), 'optimizer-owned parameters changed')
    require(change['netCls']['changed_elements'] == change['netIP']['changed_elements'] == 0, 'netCls/netIP unchanged')
    r['parameters_after_sha256'] = {n: m6d5a.parameter_report(mdl)['sha256'] for n, mdl in models.items()}
    require(all(not p.requires_grad for p in models['netIP'].parameters()) and not models['netIP'].training,
            'netIP still frozen/eval')
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reference', 'b240'), required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-2:] == BUILD_PARTS, 'dedicated build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    build.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    result = (reference if args.mode == 'reference' else b240)(build)
    result['started_utc'], result['ended_utc'] = started, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
