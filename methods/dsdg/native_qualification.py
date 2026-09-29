"""M6F-C E06b DSDG-NATIVE GPU qualification — QUALIFICATION_ONLY_SYNTHETIC, NON_SCIENTIFIC; never a training runner.

Four modes, each one fresh process:

  reference  small batches where both paths fit (B, max microbatch, epoch) in REFERENCE_CASES, with identical initial
             bytes / inputs / labels / epsilon: Path A = the unchanged pinned full-batch transcription
             (training_graph.training_forward), Path B = the unchanged M6D5c/M6D5d global-statistic microbatch
             graph (microbatch_execution_v2.run_global_batch_v2). Compares every loss, the total, optimizer-owned
             gradients and one Adam update against PRE-REGISTERED gates (the owner-approved M6D5c gates, verified
             byte-for-byte at runtime). Also isolates loss_cls and loss_pair to prove both are active.
  b240       one synthetic logical batch of 240 = 12 x 20, both spoof classes, one Adam step.
  tail120    one synthetic logical tail batch of 120 = 6 x 20, both spoof classes, one Adam step.
  workers    CPU only: a real torch DataLoader (8 workers, seed_torch_worker) over a MOCK relation with the exact
             E06b cardinalities; every live-partner draw must equal methods.dsdg.native.simulate_live_draws.
             Per epoch the loader generator is consumed as: int64 base seed (_BaseDataLoaderIter), randperm(n)
             (RandomSampler), and a second randperm(n) sliced [:n % n] when the sampler is exhausted.

Synthetic analytic tensors and mock identifiers only: no face image, manifest, VAL or TEST, no checkpoint, no bank.
OOM at microbatch 20 is STOP_OOM (no smaller retry). The frozen E06b config is only read.
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

from methods.dsdg import microbatch_execution as mb1  # noqa: E402  (unchanged M6D5c)
from methods.dsdg import microbatch_execution_v2 as mb2  # noqa: E402  (unchanged M6D5d)
from methods.dsdg import microbatch_qualification as m6d5c  # noqa: E402  (unchanged M6D5c helpers)
from methods.dsdg import native  # noqa: E402  (M6F-B)
from methods.dsdg import runtime as m6d5a  # noqa: E402  (unchanged M6D5a helpers)
from methods.dsdg import training_graph as tg  # noqa: E402  (unchanged M6D5b transcription)
from methods.dsdg import training_qualification as m6d5b  # noqa: E402  (unchanged M6D5b helpers)

MILESTONE = 'M6F-C'
LABEL = 'QUALIFICATION_ONLY_SYNTHETIC'
SCIENTIFIC = 'NON_SCIENTIFIC'
SEED = 60701  # qualification only; never an experiment seed (42/1337/2026); not used by any earlier harness
BUILD_PARTS = ('builds', 'e06b_dsdg_m6fc')
SHORT_TMP = Path('/tmp/gpat-m6fc')   # AF_UNIX socket path limit for DataLoader workers (M6D5e lesson)
LOCK = ROOT / 'environments/e06c.lock.json'   # owner decision (M6F-C): E06b binds the qualified DSDG env
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
CLEAN_GPU_MAX_USED_MIB = 1024
K = 2
LAMBDAS = {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10, 'lambda_ort': 1, 'lambda_pair': 5}
EXPECTED_PARAMETERS = dict(m6d5a.EXPECTED_PARAMETERS, netCls=(128 * K + K, 2))
MICROBATCH = 20
EPOCH = 2    # post-warmup branch (train_generator.py:178): every term at full weight
# (B, max microbatch, epoch). B=6/m=4 gives unequal chunks [4, 2] like a tail; epoch 1 checks the warm-up branch.
REFERENCE_CASES = ((4, 2, 2), (6, 4, 2), (4, 2, 1))
# PRE-REGISTERED before any E06b run: the owner-approved M6D5c gates (verified equal to the resolution record at
# runtime) plus a per-term bound equal to the registered total-loss bound (FP32 reductions over <= 1e6 terms).
GATES = {'scalar_losses_finite': True, 'total_loss_relative_error_max': 1e-4,
         'aggregate_gradient_cosine_min': 0.999, 'aggregate_gradient_relative_l2_max': 0.01,
         'post_step_parameter_delta_cosine_min': 0.99}
PER_TERM_RELATIVE_ERROR_MAX = 1e-4
LOGICAL = {'b240': 240, 'tail120': 120}
MOCK = {'casia_fasd': {'subjects': 35, 'live': 24, 'print': 48, 'replay': 24},
        'msu_mfsd': {'subjects': 25, 'live': 16, 'print': 16, 'replay': 32}}
WORKER_EPOCHS = 2
EXIT = {'PASS': 0, 'STOP_OOM': 3, 'STOP_RESOURCE_CONTAMINATION': 4, 'STOP_REFERENCE_GATE': 5, 'STOP_WORKER_GATE': 6}


def require(value, message):
    if not value:
        raise RuntimeError('M6F-C gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Firewall(m6d5a.Firewall):
    """The unchanged M6D5a firewall plus DataLoader IPC: /dev/shm and the short worker TMPDIR only."""

    def __call__(self, event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = os.path.abspath(os.fsdecode(args[0]))
            if path.startswith('/dev/shm/') or path.startswith(str(SHORT_TMP) + '/'):
                return
        return super().__call__(event, args)


def labels(torch, batch, device='cuda'):
    """Alternating print (0) / replay (1): both native classes in every batch and every chunk."""
    return torch.tensor([i % K for i in range(batch)], dtype=torch.long, device=device)


def e06b_contract(adapter):
    """Live frozen-config checks (the E06b analogue of M6D5a contract); nothing is constructed here."""
    cfg = adapter.config
    require(cfg['_runtime']['config_sha256'] == CONFIG_SHA, 'frozen E06b config SHA256')
    sem = native.native_semantics(cfg)
    t, o, lo = cfg['training'], cfg['optimizer'], cfg['losses']
    require((t['input_resolution'], t['hdim'], t['attack_type'], t['all_epochs'], t['effective_batch_size'],
             t['workers']) == (256, 128, K, 200, 240, 8), 'frozen training contract')
    require({k: lo[k] for k in LAMBDAS} == LAMBDAS == adapter.lambdas(), 'frozen native loss coefficients')
    require((o['name'], o['learning_rate'], o['scope'], o['netCls_in_optimizer']) ==
            ('Adam', 2e-4, 'netE_nir + netE_vis + netG', False), 'frozen optimizer contract')
    require(cfg['checkpoint']['rule'] == 'OFFICIAL_GENERATOR_EPOCH_200' and
            cfg['seeds']['experiment_seeds'] == [42, 1337, 2026], 'checkpoint rule / seeds')
    require((cfg['target_fidelity'], cfg['final_execution_fidelity']) ==
            ('FAITHFUL_OFFICIAL', 'PENDING_M6F_C_RUNTIME_QUALIFICATION'), 'fidelity fields')
    require(cfg['data']['splits']['TEST']['allowed'] is False, 'TEST forbidden')
    return {'config_sha256': CONFIG_SHA, 'hdim': t['hdim'], 'attack_type': t['attack_type'],
            'vocabulary': list(sem['vocabulary']), 'class_index': sem['class_index'], 'lambdas': LAMBDAS,
            'optimizer': {k: o[k] for k in ('name', 'learning_rate', 'scope', 'netCls_in_optimizer')},
            'effective_batch_size': t['effective_batch_size'], 'workers': t['workers'],
            'target_fidelity': cfg['target_fidelity'],
            'final_execution_fidelity_before': cfg['final_execution_fidelity'],
            'e06c_guards_used': False}


def setup(build, mode, *, cuda=True):
    firewall = Firewall(build)
    sys.addaudithook(firewall)
    adapter = native.DSDGNativeAdapter()
    source = adapter.validate_source()
    root = Path(source['root']) / adapter.config['source']['relevant_path']
    mismatch = tg.verify_pinned_statements((root / 'train_generator.py').read_text())
    require(not mismatch, 'pinned statements (incl. train_generator.py:171 loss_pair, :144 loss_cls)')
    lock_raw = LOCK.read_bytes()
    require(sha(lock_raw) == LOCK_SHA, 'environment lock SHA256')
    lock = json.loads(lock_raw)
    require(source['commit'] == lock['source']['commit'] and source['files_sha256'] == lock['source']['files_sha256'],
            'source identity equals the DSDG environment lock')
    registered = mb1.load_resolution()['qualification']['reference_check']['gates']
    require(registered == GATES, 'pre-registered gates equal the owner-approved M6D5c gates')
    result = {'milestone': MILESTONE, 'mode': mode, 'label': LABEL, 'scientific': SCIENTIFIC,
              'qualification_seed': SEED, 'method_id': 'E06b', 'contract': e06b_contract(adapter),
              'execution_mode': mb2.EXECUTION_MODE, 'pinned_statements_verified': len(tg.PINNED_STATEMENTS),
              'source_before': source, 'environment_lock': str(LOCK.relative_to(ROOT)),
              'environment_lock_sha256': LOCK_SHA, 'pre_registered_gates': GATES,
              'per_term_relative_error_max': PER_TERM_RELATIVE_ERROR_MAX}
    if cuda:
        result['asset_before'] = m6d5a.asset_identity(adapter)
        result['gpu_before'] = m6d5b.gpu_snapshot()
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
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    counters = m6d5c.new_counters()
    m6d5c.instrument(torch, counters)
    ctx = dict(torch=torch, F=F, firewall=firewall, source=source, adapter=adapter, cfg=adapter.config, root=root,
               counters=counters, lam=dict(LAMBDAS))
    if cuda:
        torch.cuda.manual_seed_all(SEED)
        env = m6d5a.environment()
        lock_diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
                     if k != 'launch_environment' and env.get(k) != v}
        require(not lock_diff, 'runtime identity equals the DSDG environment lock: ' + json.dumps(lock_diff))
        require(not env['precision']['matmul_tf32'] and not env['precision']['cudnn_tf32'] and
                not env['precision']['autocast_cuda'] and env['precision']['default_dtype'] == 'torch.float32',
                'FP32, no TF32, no autocast')
        result['environment_before'] = env
        ctx['env'] = env
        torch.cuda.reset_peak_memory_stats()
    return result, ctx


def finish(result, ctx, *, cuda=True):
    torch, counters, firewall = ctx['torch'], ctx['counters'], ctx['firewall']
    result['counters'] = counters
    result['source_after'] = ctx['adapter'].validate_source()
    require(result['source_after'] == ctx['source'], 'source-cache integrity after execution')
    if cuda:
        result['gpu_after'] = m6d5b.gpu_snapshot()
        result['asset_after'] = m6d5a.asset_identity(ctx['adapter'])
        require(result['asset_after'] == result['asset_before'], 'LightCNN unchanged')
        result['environment_after'] = m6d5a.environment()
        require(result['environment_after'] == ctx['env'], 'environment stable')
        require(not torch.is_autocast_enabled('cuda'), 'autocast off')
    require(counters['autograd_grad_calls'] == counters['checkpoint_saves'] == counters['activation_checkpoint_calls']
            == 0, 'no forbidden call (autograd.grad, torch.save, activation checkpoint)')
    result.update(checkpoint_created=False, scientific_training_completed=False, benchmark_data_access=False,
                  benchmark_image_reads=0, VAL_access=False, TEST_access=False, synthetic_bank=False,
                  native_manifest_created=False, compatibility_patch='NONE',
                  firewall={'denied': firewall.denied, 'event_counts': firewall.events,
                            'lightcnn_read_opens': firewall.lightcnn_opens})
    require(not firewall.denied, 'firewall denials')
    return result


def build_models(torch, networks, root, cfg):
    """M6D5b build_models with the native classifier: Cls(128, K = 2); pinned define_G/define_IP unchanged."""
    t = cfg['training']
    netE_nir, netE_vis, netG, netCls = networks.define_G(hdim=t['hdim'], attack_type=t['attack_type'])
    netIP = networks.define_IP(is_train=False)
    models = dict(netE_nir=netE_nir, netE_vis=netE_vis, netG=netG, netCls=netCls, netIP=netIP)
    binding = {}
    for name, model in models.items():
        inner = type(model.module)
        require(type(model) is torch.nn.DataParallel, 'upstream DataParallel wrapper ' + name)
        require(inner.__name__ == m6d5a.EXPECTED_CLASSES[name], 'class binding ' + name)
        require(Path(sys.modules[inner.__module__].__file__).resolve().is_relative_to(root.resolve()),
                'class from pinned source ' + name)
        binding[name] = {'wrapper': 'torch.nn.DataParallel', 'class': inner.__name__, 'module': inner.__module__,
                         'device_ids': list(model.device_ids)}
    fc = netCls.module.fc
    require((fc.out_features, fc.in_features, tuple(fc.weight.shape)) == (K, 128, (K, 128)), 'live Cls(128, 2)')
    require(netIP.module.is_train is False and not hasattr(netIP.module, 'fc2_'), 'define_IP(is_train=False)')
    binding['netCls_live'] = {'in_features': fc.in_features, 'out_features': fc.out_features,
                              'weight_shape': list(fc.weight.shape), 'bias_shape': list(fc.bias.shape)}
    return models, binding


def models_ready(torch, ctx, result, networks):
    models, result['binding'] = build_models(torch, networks, ctx['root'], ctx['cfg'])
    result['lightcnn'] = m6d5a.load_lightcnn(torch, models['netIP'])
    models['netE_nir'].train(); models['netE_vis'].train(); models['netG'].train(); models['netIP'].eval()
    params = {n: m6d5a.parameter_report(m) for n, m in models.items()}
    for n, (count, tensors) in EXPECTED_PARAMETERS.items():
        require((params[n]['parameters'], params[n]['parameter_tensors']) == (count, tensors), 'live count ' + n)
    require(params['netIP']['requires_grad_parameters'] == 0 and not models['netIP'].training, 'netIP frozen eval')
    result['parameters_initial'] = {n: {k: v for k, v in p.items() if k != 'sha256'} for n, p in params.items()}
    result['batch_coupling'] = m6d5c.batch_coupling(torch, models)
    return models, params


def restore(models, initial, params):
    for n, mdl in models.items():
        mdl.load_state_dict(initial[n])
        for p in mdl.parameters():
            p.grad = None
    require({n: m6d5a.parameter_report(m)['sha256'] for n, m in models.items()} ==
            {n: p['sha256'] for n, p in params.items()}, 'identical initial model bytes restored')


def raw_and_weighted(losses, lam):
    """Weighted = value in the pinned objective; raw = weighted / lambda (rec/kl carry no lambda)."""
    div = {'loss_mmd': lam['lambda_mmd'], 'loss_ip': lam['lambda_ip'], 'loss_cls': lam['lambda_type'],
           'loss_ort': lam['lambda_ort'], 'loss_pair': lam['lambda_pair'], 'loss_rec': 1, 'loss_kl': 1}
    return {k: {'weighted': v, 'raw': v / div[k], 'lambda': div[k]} for k, v in losses.items()}


def finite_losses(values):
    return all(v == v and abs(v) != float('inf') for v in values)


# ================================================================= reference
def reference(build):
    result, ctx = setup(build, 'reference')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    cases = []
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: {k: v.detach().clone() for k, v in mdl.state_dict().items()} for n, mdl in models.items()}
        criterion_type, criterionL2 = tg.criteria(torch)
        owned = m6d5c.owned_params(models)
        for B, m, epoch in REFERENCE_CASES:
            x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
            label = labels(torch, B)
            eps, eps_proof = m6d5c.epsilon_draw_proof(torch, util, B)
            restore(models, initial, params)
            # ---- Path A: pinned full-batch transcription, pinned order forward -> zero_grad -> backward -> step
            opt_a = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
            m6d5c.count_steps(opt_a, counters)
            out = tg.training_forward(torch, F, m6d5c.pinned_util_shim(util, eps), nets, x_spoof, x_live, label, lam,
                                      criterion_type, criterionL2)
            require(tuple(out['pre_spoof'].shape) == (B, K), 'classifier logits [B, 2]')
            loss_a = tg.total_loss(out, epoch)
            path_a = {'losses': {k: out[k].item() for k in tg.LOSS_TERMS}, 'total': loss_a.item(),
                      'pre_spoof_shape': list(out['pre_spoof'].shape),
                      'labels': label.tolist()}
            opt_a.zero_grad()
            loss_a.backward()
            grads_a = [p.grad.detach().clone() for _, p in owned]
            netcls_grad_a = m6d5b.grad_inventory(models['netCls'])
            opt_a.step()
            delta_a = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
            del out, loss_a, opt_a
            # ---- Path B: unchanged M6D5c/M6D5d global-statistic microbatch graph
            restore(models, initial, params)
            opt_b = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
            m6d5c.count_steps(opt_b, counters)
            grads_b = []
            run = mb2.run_global_batch_v2(torch, F, util, nets, opt_b, x_spoof, x_live, label, eps, lam,
                                          criterion_type, criterionL2, epoch, max_microbatch=m,
                                          before_step=lambda: grads_b.extend(p.grad.detach().clone() for _, p in owned))
            delta_b = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
            g = run['global_losses']
            total_b = tg.total_loss(g, epoch)
            rel = lambda a, b: abs(a - b) / max(abs(a), 1e-30)  # noqa: E731
            grad_cmp = m6d5c.compare_vectors(torch, grads_a, grads_b)
            step_cmp = m6d5c.compare_vectors(torch, delta_a, delta_b)
            grad_max = max(float((a - b).abs().max()) for a, b in zip(grads_a, grads_b))
            step_max = max(float((a - b).abs().max()) for a, b in zip(delta_a, delta_b))
            term_rel = {k: rel(path_a['losses'][k], g[k]) for k in tg.LOSS_TERMS}
            case = {'global_batch': B, 'max_microbatch': m, 'chunk_sizes': run['chunk_sizes'], 'epoch': epoch,
                    'labels_both_classes': sorted(set(label.tolist())) == [0, 1], 'epsilon': eps_proof,
                    'path_a_full_batch': dict(path_a, raw_and_weighted=raw_and_weighted(path_a['losses'], lam),
                                              netCls_grad_nonzero_elements=netcls_grad_a['nonzero_elements']),
                    'path_b_microbatch': {'losses': g, 'total': total_b, 'surrogate_sums': run['surrogate_sums'],
                                          'sum_of_chunk_objectives': run['sum_of_chunk_objectives']},
                    'comparison': {'loss_abs_diff': {k: abs(path_a['losses'][k] - g[k]) for k in tg.LOSS_TERMS},
                                   'loss_rel_diff': term_rel, 'total_rel_error': rel(path_a['total'], total_b),
                                   'sum_of_chunk_objectives_rel_error':
                                       rel(path_a['total'], run['sum_of_chunk_objectives']),
                                   'gradient': grad_cmp, 'gradient_max_abs_diff': grad_max,
                                   'update': step_cmp, 'update_max_abs_diff': step_max}}
            case['gates'] = {
                'scalar_losses_finite': finite_losses(list(path_a['losses'].values()) + list(g.values())),
                'total_loss_relative_error':
                    case['comparison']['total_rel_error'] <= GATES['total_loss_relative_error_max'],
                'per_term_relative_error': max(term_rel.values()) <= PER_TERM_RELATIVE_ERROR_MAX,
                'aggregate_gradient_cosine': grad_cmp['cosine'] >= GATES['aggregate_gradient_cosine_min'],
                'aggregate_gradient_relative_l2':
                    grad_cmp['relative_l2'] <= GATES['aggregate_gradient_relative_l2_max'],
                'post_step_parameter_direction': step_cmp['cosine'] >= GATES['post_step_parameter_delta_cosine_min'],
                'owned_gradient_coverage': len(grads_a) == len(grads_b) == 55,
                'both_classes_in_batch': case['labels_both_classes'],
                'loss_cls_nonzero': path_a['losses']['loss_cls'] > 0,
                'loss_pair_nonzero': path_a['losses']['loss_pair'] > 0}
            cases.append(case)
            del run, opt_b, grads_a, grads_b, delta_a, delta_b
        result['term_isolation'] = term_isolation(torch, F, util, nets, models, initial, params, owned, lam,
                                                  criterion_type, criterionL2, counters)
    result['reference_cases'] = cases
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    gates_ok = all(all(c['gates'].values()) for c in cases) and all(result['term_isolation']['gates'].values())
    result['status'] = 'PASS' if gates_ok else 'STOP_REFERENCE_GATE'
    result['peak_memory'] = m6d5b.memory(torch, 'reference_end')
    return finish(result, ctx)


def term_isolation(torch, F, util, nets, models, initial, params, owned, lam, criterion_type, criterionL2, counters):
    """loss_cls and loss_pair each backpropagated ALONE (fresh forward, B=4, both classes): active and nontrivial."""
    out_rows = {}
    for term in ('loss_cls', 'loss_pair'):
        restore(models, initial, params)
        x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, 4, 'cpu'))
        eps = mb2.draw_epsilon(torch, 4)
        out = tg.training_forward(torch, F, m6d5c.pinned_util_shim(util, eps), nets, x_spoof, x_live,
                                  labels(torch, 4), lam, criterion_type, criterionL2)
        value = out[term]
        value.backward()
        owned_l2 = sum(float(p.grad.double().pow(2).sum()) for _, p in owned if p.grad is not None) ** 0.5
        cls_inv = m6d5b.grad_inventory(models['netCls'])
        weight = lam['lambda_type' if term == 'loss_cls' else 'lambda_pair']
        out_rows[term] = {'value': value.item(), 'raw': value.item() / weight,
                          'owned_gradient_l2': owned_l2, 'owned_grad_finite': all(
                              bool(p.grad.isfinite().all()) for _, p in owned if p.grad is not None),
                          'netCls_grad_nonzero_elements': cls_inv['nonzero_elements']}
        del out, value
    restore(models, initial, params)
    gates = {'loss_cls_active': out_rows['loss_cls']['value'] > 0 and out_rows['loss_cls']['owned_gradient_l2'] > 0
             and out_rows['loss_cls']['netCls_grad_nonzero_elements'] > 0,
             'loss_pair_active': out_rows['loss_pair']['value'] > 0 and out_rows['loss_pair']['owned_gradient_l2'] > 0,
             'finite': out_rows['loss_cls']['owned_grad_finite'] and out_rows['loss_pair']['owned_grad_finite']}
    return {'terms': out_rows, 'gates': gates,
            'note': 'separate forward+backward per term; no optimizer step; counts appear in counters.backward_calls'}


# ================================================================= logical 240 / tail 120
def logical(build, mode):
    B = LOGICAL[mode]
    result, ctx = setup(build, mode)
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        before = {n: m6d5b.cpu_copy(models[n]) for n in models}
        criterion_type, criterionL2 = tg.criteria(torch)
        x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
        label = labels(torch, B)
        eps = mb2.draw_epsilon(torch, B)
        optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        m6d5c.count_steps(optimizer, counters)
        result['optimizer'] = m6d5b.optimizer_evidence(torch, optimizer, models)
        result['inputs'] = {'x_spoof': m6d5b.summary(x_spoof), 'x_live': m6d5b.summary(x_live),
                            'labels_count': {'print_0': int((label == 0).sum()), 'replay_1': int((label == 1).sum())}}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = m6d5b.memory(torch, 'before_global_batch')
        grads = {}
        t0 = time.perf_counter()
        try:
            run = mb2.run_global_batch_v2(torch, F, util, nets, optimizer, x_spoof, x_live, label, eps, lam,
                                          criterion_type, criterionL2, EPOCH, max_microbatch=MICROBATCH,
                                          before_step=lambda: grads.update(
                                              {n: m6d5b.grad_inventory(models[n]) for n in ('netE_nir', 'netE_vis',
                                                                                            'netG', 'netCls')}))
        except torch.cuda.OutOfMemoryError as exc:
            result.update(status='STOP_OOM', oom={'microbatch': MICROBATCH, 'logical_batch': B,
                                                  'error': str(exc)[:2000],
                                                  'memory': m6d5b.memory(torch, 'oom')}, smaller_retry=False)
            return finish(result, ctx)
        torch.cuda.synchronize()
        duration = time.perf_counter() - t0
        g = run['global_losses']
        change = {n: m6d5b.parameter_change(before[n], models[n]) for n in models}
        result.update(
            logical_batch_size=B, physical_microbatch=MICROBATCH, chunks=run['chunks'], chunk_sizes=run['chunk_sizes'],
            epoch_branch=EPOCH, losses=raw_and_weighted(g, lam), total_loss=tg.total_loss(g, EPOCH),
            warmup_total_algebraic_only=tg.total_loss(g, 1), surrogate_sums=run['surrogate_sums'],
            sum_of_chunk_objectives=run['sum_of_chunk_objectives'],
            gradients={n: {k: v for k, v in inv.items() if k != 'per_tensor'} for n, inv in grads.items()},
            parameter_change={n: {k: v for k, v in c.items() if k != 'per_tensor'} for n, c in change.items()},
            memory={'before': mem_before, 'after_step': m6d5b.memory(torch, 'after_step')},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_clock_seconds=duration, warnings=sorted({f'{w.category.__name__}: {w.message}' for w in caught}))
    owned_ok = all(grads[n]['finite'] == grads[n]['non_none'] == grads[n]['tensors']
                   for n in ('netE_nir', 'netE_vis', 'netG'))
    result['gates'] = {
        'chunks': (run['chunks'], run['chunk_sizes']) == (B // MICROBATCH, [MICROBATCH] * (B // MICROBATCH)),
        'losses_finite': finite_losses(list(g.values()) + [result['total_loss']]),
        'owned_gradients_finite_and_present': owned_ok,
        'netCls_gradient_nonzero_and_finite': grads['netCls']['nonzero_elements'] > 0 and
                                              grads['netCls']['finite'] == grads['netCls']['non_none'],
        'optimizer_step_applied_once': counters['optimizer_applications'] == 1 and
                                       counters['backward_calls'] == B // MICROBATCH,
        'owned_parameters_changed': all(change[n]['changed_tensors'] > 0 for n in ('netE_nir', 'netE_vis', 'netG')),
        'netCls_netIP_unchanged': change['netCls']['changed_elements'] == change['netIP']['changed_elements'] == 0,
        'both_classes': min(result['inputs']['labels_count'].values()) > 0,
        'loss_cls_and_pair_nonzero': g['loss_cls'] > 0 and g['loss_pair'] > 0}
    result['status'] = 'PASS' if all(result['gates'].values()) else 'STOP_REFERENCE_GATE'
    return finish(result, ctx)


# ================================================================= workers (CPU)
def mock_rows(shuffle_seed=None):
    rows = []
    for ds, n in MOCK.items():
        for s in range(n['subjects']):
            subject = f'{ds}::mock{s:03d}'
            for kind, label, count in (('live', 0, n['live']), ('print', 1, n['print']), ('replay', 1, n['replay'])):
                for i in range(count):
                    rows.append({'sample_id': f'mock-{ds}-{s:03d}-{kind}-{i:03d}', 'dataset': ds,
                                 'subject_id_global': subject, 'label_binary': label, 'attack_macro': kind,
                                 'split': 'TRAIN', 'm2_status': 'COMPLETE'})
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows


class DrawDataset:
    """Mock E06b item: the official draw only (no image): (spoof index, live partner, worker id, worker seed)."""

    def __init__(self, relation):
        self.relation = relation

    def __len__(self):
        return len(self.relation)

    def __getitem__(self, index):
        import torch.utils.data as tud
        live = self.relation.draw_live(index)       # module `random`, seeded by seed_torch_worker
        info = tud.get_worker_info()
        return index, live, info.id, info.seed


def identity_collate(batch):
    return batch


def workers(build):
    result, ctx = setup(build, 'workers', cuda=False)
    torch = ctx['torch']
    from methods.common.learned import seed_torch_worker
    relation = native.NativeRelation(mock_rows())
    shuffled = native.NativeRelation(mock_rows(shuffle_seed=SEED))
    lc = native.loader_contract(ctx['cfg'])

    def epochs(seed):
        loader = torch.utils.data.DataLoader(
            DrawDataset(relation), batch_size=lc['batch_size'], shuffle=lc['shuffle'],
            generator=torch.Generator().manual_seed(seed), num_workers=lc['num_workers'],
            worker_init_fn=seed_torch_worker, persistent_workers=lc['persistent_workers'], drop_last=lc['drop_last'],
            collate_fn=identity_collate, pin_memory=False)
        observed = []
        for _ in range(WORKER_EPOCHS):
            observed.append([[list(item) for item in batch] for batch in loader])
        return observed

    def expected(seed):
        gen = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(WORKER_EPOCHS):
            base = int(torch.empty((), dtype=torch.int64).random_(generator=gen).item())   # _BaseDataLoaderIter
            perm = torch.randperm(len(relation), generator=gen).tolist()                   # RandomSampler
            torch.randperm(len(relation), generator=gen)   # RandomSampler remainder draw [:n % n] at exhaustion
            batches = [perm[i:i + lc['batch_size']] for i in range(0, len(perm), lc['batch_size'])]
            wseeds = [(base + w) % 2 ** 32 for w in range(lc['num_workers'])]
            draws = native.simulate_live_draws(relation, batches, wseeds)
            out.append({'base_seed': base, 'batches': batches, 'draws': draws})
        return out

    first, replay = epochs(SEED), epochs(SEED)
    other = epochs(SEED + 1)
    model = expected(SEED)
    per_epoch = []
    for e in range(WORKER_EPOCHS):
        obs, exp = first[e], model[e]
        per_epoch.append({
            'batches': len(obs), 'batch_sizes': [len(b) for b in obs],
            'indices_match_model': [[it[0] for it in b] for b in obs] == exp['batches'],
            'worker_ids_match_b_mod_8': all(it[2] == b % lc['num_workers']
                                            for b, batch in enumerate(obs) for it in batch),
            'worker_seed_eq_base_plus_id': all(it[3] == exp['base_seed'] + it[2] for batch in obs for it in batch),
            'live_draws_match_pure_python_model': [[it[1] for it in b] for b in obs] == exp['draws'],
            'distinct_worker_ids': sorted({it[2] for batch in obs for it in batch}),
            'draws_within_same_subject': all(it[1] in relation.pools[relation.spoof[it[0]][1]]
                                             for b in obs for it in b),
            'draw_sequence_sha256': sha(json.dumps([[it[1] for it in b] for b in obs]).encode())})
    result['relation'] = {'summary': relation.summary(), 'order_invariant_to_input_order':
                          (relation.spoof, relation.pools) == (shuffled.spoof, shuffled.pools),
                          'mock_identifiers_only': True}
    result['loader_contract'] = lc
    result['epochs'] = per_epoch
    result['replay_identical'] = first == replay
    live_seq = lambda run: [[it[1] for it in b] for b in run[0]]  # noqa: E731
    result['different_seed_differs'] = live_seq(first) != live_seq(other)
    result['epoch_1_vs_2_differ'] = per_epoch[0]['draw_sequence_sha256'] != per_epoch[1]['draw_sequence_sha256']
    result['gates'] = {
        'workers_8_honored': all(p['distinct_worker_ids'] == list(range(8)) for p in per_epoch),
        'batch_plan_15x240_plus_120': all(p['batch_sizes'] == [240] * 15 + [120] for p in per_epoch),
        'indices_match_model': all(p['indices_match_model'] for p in per_epoch),
        'worker_assignment_b_mod_8': all(p['worker_ids_match_b_mod_8'] for p in per_epoch),
        'worker_seed_contract': all(p['worker_seed_eq_base_plus_id'] for p in per_epoch),
        'live_draws_match_model': all(p['live_draws_match_pure_python_model'] for p in per_epoch),
        'same_subject': all(p['draws_within_same_subject'] for p in per_epoch),
        'deterministic_replay': result['replay_identical'], 'seed_sensitive': result['different_seed_differs'],
        'redraw_every_epoch': result['epoch_1_vs_2_differ'],
        'order_invariant_relation': result['relation']['order_invariant_to_input_order'],
        'mock_cardinalities': relation.summary()['total'] == {'spoof': 3720, 'live': 1240, 'subjects': 60,
                                                              'print': 2080, 'replay': 1640}}
    result['status'] = 'PASS' if all(result['gates'].values()) else 'STOP_WORKER_GATE'
    return finish(result, ctx, cuda=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reference', 'b240', 'tail120', 'workers'), required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-3:-1] == BUILD_PARTS or build.parts[-2:] == BUILD_PARTS, 'dedicated build path')
    require('runs' not in build.parts, 'never under a scientific run root')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    require(build.is_dir(), 'build root pre-created by the launcher (no parent mkdir inside the audited process)')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.resolve() == SHORT_TMP if args.mode == 'workers' else tmp.resolve().is_relative_to(build),
            'TMPDIR: short worker dir (workers) or inside the build root')
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fn = {'reference': reference, 'workers': workers}.get(args.mode)
    result = fn(build) if fn else logical(build, args.mode)
    result['started_utc'], result['ended_utc'] = started, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
