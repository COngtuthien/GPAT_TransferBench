"""M6D5b E06c exact training-graph + physical-batch-240 qualification, never a training runner.

One fresh GPU process = one SYNTHETIC_PHYSICAL_BATCH_240 pair, one full pinned
forward (methods/dsdg/training_graph.py), one backward, one Adam step. No dataset,
image reader, manifest, epoch loop or checkpoint writer exists here. A CUDA OOM
in any phase is recorded and returned as STOP; there is no reduced-batch,
accumulation, AMP, TF32, checkpointing or offload fallback. M6D5a helpers are
imported from methods/dsdg/runtime.py unchanged.
"""
import argparse
import hashlib
import inspect
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
from methods.dsdg import training_graph as tg  # noqa: E402

SEED = 60502  # qualification only; not an experiment seed (42/1337/2026)
EPOCH = 1     # warmup branch of train_generator.py:174
BUILD_PARTS = ('builds', 'e06c_dsdg_m6d5b')
LOCK = ROOT / 'environments/e06c.lock.json'
LOCK_SHA = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
CLEAN_GPU_MAX_USED_MIB = 1024  # display-only processes; any foreign compute process is contamination
EXIT = {'PASS': 0, 'STOP_OOM': 3, 'STOP_RESOURCE_CONTAMINATION': 4}


def require(value, message):
    if not value:
        raise RuntimeError('M6D5b gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gpu_snapshot():
    """nvidia-smi view of the whole device, including processes this harness does not own."""
    total, used, free = (int(v) for v in m6d5a.command(
        'nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits').split(','))
    apps = m6d5a.command('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                         '--format=csv,noheader,nounits')
    compute = [dict(zip(('pid', 'process_name', 'used_mib'), (s.strip() for s in line.split(','))))
               for line in apps.splitlines() if line.strip()]
    return {'total_mib': total, 'used_mib': used, 'free_mib': free, 'compute_processes': compute,
            'own_pid': os.getpid()}


def memory(torch, label):
    free, total = torch.cuda.mem_get_info()
    return {'phase': label, 'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device_free_bytes': free, 'device_total_bytes': total}


def rng(torch):
    return {'cpu_sha256': sha(torch.get_rng_state().numpy().tobytes()),
            'cuda_sha256': sha(torch.cuda.get_rng_state().numpy().tobytes())}


def summary(t):
    a = t.detach().cpu().contiguous()
    f = a.double()
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'device': str(t.device),
            'finite': bool(f.isfinite().all()), 'sha256': sha(a.numpy().tobytes()),
            'min': float(f.min()), 'max': float(f.max()), 'mean': float(f.mean()),
            'l2': float(f.pow(2).sum().sqrt()), 'nonzero': int((a != 0).sum())}


def grad_inventory(model):
    rows = {}
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            rows[name] = {'present': False}
            continue
        d = g.detach().double()
        rows[name] = {'present': True, 'numel': g.numel(), 'finite': bool(d.isfinite().all()),
                      'nonzero': int((g != 0).sum()), 'l2': float(d.pow(2).sum().sqrt()),
                      'max_abs': float(d.abs().max())}
    present = [r for r in rows.values() if r['present']]
    return {'tensors': len(rows), 'non_none': len(present), 'finite': sum(r['finite'] for r in present),
            'nonzero_tensors': sum(r['nonzero'] > 0 for r in present),
            'nonzero_elements': sum(r['nonzero'] for r in present),
            'elements': sum(p.numel() for p in model.parameters()), 'per_tensor': rows}


def cpu_copy(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def parameter_change(before, model):
    rows = {}
    for n, p in model.named_parameters():
        a, b = before[n], p.detach().cpu()
        diff = (a != b)
        rows[n] = {'changed_elements': int(diff.sum()), 'numel': a.numel(),
                   'max_abs_delta': float((a.double() - b.double()).abs().max()),
                   'sha256_before': sha(a.numpy().tobytes()), 'sha256_after': sha(b.numpy().tobytes())}
    return {'tensors': len(rows), 'changed_tensors': sum(r['changed_elements'] > 0 for r in rows.values()),
            'changed_elements': sum(r['changed_elements'] for r in rows.values()),
            'elements': sum(r['numel'] for r in rows.values()),
            'max_abs_delta': max(r['max_abs_delta'] for r in rows.values()), 'per_tensor': rows}


def instrument(torch, counters):
    """Count (never forbid) the one backward/step; forbid every other training-side call."""
    tensor_backward, autograd_backward, adam_init = torch.Tensor.backward, torch.autograd.backward, torch.optim.Adam.__init__

    def backward(self, *a, **k):
        counters['backward_passes'] += 1
        return tensor_backward(self, *a, **k)

    def autograd(*a, **k):
        counters['autograd_backward_calls'] += 1
        return autograd_backward(*a, **k)

    def adam(self, *a, **k):
        counters['optimizer_constructions'] += 1
        return adam_init(self, *a, **k)

    def forbid(name):
        def forbidden(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('M6D5b forbids ' + name)
        return forbidden
    torch.Tensor.backward, torch.autograd.backward, torch.optim.Adam.__init__ = backward, autograd, adam
    torch.autograd.grad = forbid('autograd_grad_calls')
    torch.save = forbid('checkpoint_saves')
    torch.utils.checkpoint.checkpoint = forbid('activation_checkpoint_calls')


def build_models(torch, networks, root, cfg):
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
        binding[name] = {'wrapper': 'torch.nn.DataParallel', 'class': inner.__name__,
                         'module': inner.__module__, 'device_ids': list(model.device_ids)}
    require(netCls.module.fc.out_features == 1 and netCls.module.fc.in_features == 128, 'Cls(128,1)')
    require(netIP.module.is_train is False and not hasattr(netIP.module, 'fc2_'), 'define_IP(is_train=False)')
    return models, binding


def optimizer_evidence(torch, optimizer, models):
    owned = [p for n in tg.OPTIMIZER_OWNED for p in models[n].parameters()]
    group_params = [p for g in optimizer.param_groups for p in g['params']]
    ids = [id(p) for p in group_params]
    foreign = {id(p) for n in ('netCls', 'netIP') for p in models[n].parameters()}
    require(len(optimizer.param_groups) == 1, 'one param group')
    require(ids == [id(p) for p in owned], 'exact ordered ownership netE_nir+netE_vis+netG')
    require(len(set(ids)) == len(ids) == 55, 'no duplicate parameter identities')
    require(not set(ids) & foreign, 'netCls/netIP excluded')
    group = {k: (list(v) if isinstance(v, tuple) else v) for k, v in optimizer.param_groups[0].items() if k != 'params'}
    require(group['lr'] == 2e-4 and group['weight_decay'] == 0 and group['amsgrad'] is False, 'Adam lr/defaults')
    return {'class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
            'constructor_call': 'torch.optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) '
                                '+ list(netG.parameters()), lr=2e-4)',
            'param_groups': 1, 'parameter_tensors': len(ids), 'parameters': sum(p.numel() for p in group_params),
            'duplicates': len(ids) - len(set(ids)), 'overlap_netCls_netIP': len(set(ids) & foreign),
            'resolved_group_defaults': group,
            'explicit_arguments_beyond_lr': [],
            'source': 'train_generator.py:87-88'}


def zero_grad_evidence(torch):
    default = inspect.signature(torch.optim.Optimizer.zero_grad).parameters['set_to_none'].default
    return {'pinned_call': 'optimizer.zero_grad()  # train_generator.py:180, no argument',
            'historical_pytorch_1_6_default': 'zero_grad(self): p.grad.detach_(); p.grad.zero_() for p.grad not None',
            'runtime_signature_default_set_to_none': default}


def full_batch_proofs(torch, F, out, lam):
    """CPU float64 recomputation from the executed tensors; halves are a diagnostic contrast only."""
    f = lambda k: out[k].detach().to('cpu', torch.float64)  # noqa: E731
    zn, zv, zc = f('z_nir'), f('z_vis'), f('z_cls')
    B = zn.shape[0]
    mmd = lambda a, b: lam['lambda_mmd'] * (a.mean(0) - b.mean(0)).abs().mean()  # noqa: E731
    ort = lambda a, b: lam['lambda_ort'] * (a * b).sum(1).mean().abs()  # noqa: E731
    halves = (slice(0, B // 2), slice(B // 2, B))
    rec, img = f('rec'), f('img')
    sq = (rec - img).pow(2).reshape(B, -1).sum(-1)
    kl = lambda mu, lv: (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(-1)).mean()  # noqa: E731
    ip = lam['lambda_ip'] * (F.mse_loss(f('rec_nir_fc'), f('nir_fc')) + F.mse_loss(f('rec_vis_fc'), f('vis_fc'))) / 2
    rows = {
        'loss_rec': float(sq.mean() / 2), 'loss_kl': float((kl(f('mu_nir'), f('logvar_nir')) + kl(f('mu_vis'), f('logvar_vis'))
                                                          + kl(f('mu_a'), f('logvar_a'))) / 3),
        'loss_mmd': float(mmd(zn, zv)), 'loss_ort': float(ort(zc, zn)), 'loss_ip': float(ip),
        'loss_pair': float(lam['lambda_pair'] * F.mse_loss(f('rec_nir_fc'), f('rec_vis_fc')))}
    executed = {k: out[k].item() for k in rows}
    return {
        'recomputed_float64': rows, 'executed_fp32': executed,
        'abs_diff': {k: abs(rows[k] - executed[k]) for k in rows},
        'rel_diff': {k: abs(rows[k] - executed[k]) / max(abs(rows[k]), 1e-30) for k in rows},
        'loss_rec': {'reduction': 'per-sample sum over 6x256x256, mean over batch rows, /2 (size_average=True)',
                     'rows_reduced': int(sq.numel()),
                     'default_MSELoss_mean_value_not_used': float(F.mse_loss(rec, img)),
                     'ratio_to_default_mse': float(sq.mean() / 2 / F.mse_loss(rec, img))},
        'loss_mmd': {'latent_shape': list(zn.shape), 'rows_in_mean': B, 'mean_over_dim': 0,
                     'full_batch_value': float(mmd(zn, zv)),
                     'diagnostic_mean_of_two_half_batch_values_not_used': float(sum(mmd(zn[s], zv[s]) for s in halves) / 2)},
        'loss_ort': {'latent_shape': list(zc.shape), 'rows_in_mean': B, 'full_batch_value': float(ort(zc, zn)),
                     'diagnostic_mean_of_two_half_batch_values_not_used': float(sum(ort(zc[s], zn[s]) for s in halves) / 2)},
        'loss_pair': {'lambda_pair': lam['lambda_pair'],
                      'raw_mse_rec_nir_fc_rec_vis_fc': float(F.mse_loss(f('rec_nir_fc'), f('rec_vis_fc'))),
                      'executed_value': out['loss_pair'].item()}}


def qualify(build):
    firewall = m6d5a.Firewall(build)
    sys.addaudithook(firewall)
    source, adapter = m6d5a.source_identity()
    cfg = adapter.config
    lam = tg.frozen_lambdas(cfg)
    batch = tg.execution_guard(adapter)
    root = Path(source['root']) / cfg['source']['relevant_path']
    pinned_mismatch = tg.verify_pinned_statements((root / 'train_generator.py').read_text())
    require(not pinned_mismatch, 'pinned statements: ' + json.dumps(pinned_mismatch))
    lock_raw = LOCK.read_bytes()
    require(sha(lock_raw) == LOCK_SHA, 'environment lock SHA256')
    lock = json.loads(lock_raw)
    require(source['commit'] == lock['source']['commit'] and source['files_sha256'] == lock['source']['files_sha256'],
            'source identity equals environment lock')
    result = {'label': tg.LABEL, 'diagnostic_seed': SEED, 'epoch_semantics': EPOCH,
              'contract': m6d5a.contract(adapter), 'lambdas': lam, 'batch_policy': batch,
              'pinned_statements_verified': len(tg.PINNED_STATEMENTS), 'pinned_statement_mismatches': pinned_mismatch,
              'source_before': source, 'asset_before': m6d5a.asset_identity(adapter),
              'environment_lock_sha256': sha(lock_raw), 'gpu_before': gpu_snapshot()}
    foreign = [p for p in result['gpu_before']['compute_processes'] if int(p['pid']) != os.getpid()]
    result['resource_clean'] = not foreign and result['gpu_before']['used_mib'] <= CLEAN_GPU_MAX_USED_MIB
    if not result['resource_clean']:
        result.update(status='STOP_RESOURCE_CONTAMINATION', foreign_compute_processes=foreign)
        return result
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint  # noqa: F401  (so it can be forbidden)
    from methods.common.upstream import upstream_modules
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    counters = dict.fromkeys(('optimizer_constructions', 'optimizer_step_entries', 'optimizer_applications',
                              'backward_passes', 'autograd_backward_calls', 'autograd_grad_calls',
                              'checkpoint_saves', 'activation_checkpoint_calls'), 0)
    instrument(torch, counters)
    env = m6d5a.environment()
    result['environment_before'] = env
    lock_diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
                 if k != 'launch_environment' and env.get(k) != v}
    require(not lock_diff, 'runtime identity equals environment lock: ' + json.dumps(lock_diff))
    result['environment_lock_identity_match'] = {'fields_compared': sorted(k for k in lock['identity']
                                                                           if k != 'launch_environment'),
                                                 'launch_environment_not_compared': 'per-milestone seed/TMPDIR'}
    result['deterministic_algorithms_enabled'] = torch.are_deterministic_algorithms_enabled()
    torch.cuda.reset_peak_memory_stats()
    mem, rngs, timing, grads_seen = [], {'after_seed': rng(torch)}, {}, {}
    phase = {'name': 'model_construction', 't': time.time()}

    def mark(name):
        timing[phase['name']] = round(time.time() - phase['t'], 4)
        phase.update(name=name, t=time.time())

    def hook(name):
        def capture(g):  # small FP32 reductions only; hooks never return a replacement gradient
            d = g.detach()
            grads_seen[name] = {'shape': list(g.shape), 'finite': bool(torch.isfinite(d).all()),
                                'nonzero': int(torch.count_nonzero(d)), 'l2': float(torch.linalg.vector_norm(d)),
                                'max_abs': float(d.abs().max())}
        return capture
    out = optimizer = None
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(root, m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        networks, util = modules['networks'], modules['misc.util']
        models, result['binding'] = build_models(torch, networks, root, cfg)
        result['lightcnn'] = m6d5a.load_lightcnn(torch, models['netIP'])
        models['netE_nir'].train(); models['netE_vis'].train(); models['netG'].train(); models['netIP'].eval()
        params = {n: m6d5a.parameter_report(m) for n, m in models.items()}
        for n, (count, tensors) in m6d5a.EXPECTED_PARAMETERS.items():
            require((params[n]['parameters'], params[n]['parameter_tensors']) == (count, tensors), 'live count ' + n)
        require(params['netIP']['requires_grad_parameters'] == 0 and not models['netIP'].training, 'netIP frozen eval')
        result['parameters_initial'] = params
        result['modes'] = {n: m.training for n, m in models.items()}
        initial = {n: cpu_copy(models[n]) for n in models}
        mem.append(memory(torch, 'after_model_construction'))
        try:
            mark('optimizer_construction')
            optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                           cfg['optimizer']['learning_rate'])
            optimizer.register_step_pre_hook(lambda *a: counters.__setitem__(
                'optimizer_step_entries', counters['optimizer_step_entries'] + 1))
            optimizer.register_step_post_hook(lambda *a: counters.__setitem__(
                'optimizer_applications', counters['optimizer_applications'] + 1))
            result['optimizer'] = optimizer_evidence(torch, optimizer, models)
            result['zero_grad'] = zero_grad_evidence(torch)
            criterion_type, criterionL2 = tg.criteria(torch)
            mark('input_allocation')
            x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, tg.PHYSICAL_BATCH, 'cpu'))
            label_spoof = torch.zeros(tg.PHYSICAL_BATCH, dtype=torch.long, device='cuda')  # batch['type'] = 0
            rows = [sha(r.cpu().numpy().tobytes()) for x in (x_spoof, x_live) for r in x]
            result['inputs'] = {'x_spoof': summary(x_spoof), 'x_live': summary(x_live), 'label_spoof': summary(label_spoof),
                                'distinct_rows': len(set(rows)), 'rows': len(rows),
                                'construction': 'methods/dsdg/training_graph.py::synthetic_pair (analytic, CPU float64 -> FP32 CUDA)'}
            require(len(set(rows)) == 2 * tg.PHYSICAL_BATCH, 'all 480 synthetic rows distinct')
            require(list(x_spoof.shape) == list(x_live.shape) == [240, 3, 256, 256] and x_spoof.dtype == torch.float32,
                    'B=240 FP32 inputs')
            require(all(0 <= result['inputs'][k]['min'] and result['inputs'][k]['max'] <= 1 for k in ('x_spoof', 'x_live')),
                    '[0,1] inputs')
            require(int(label_spoof.sum()) == 0, 'all CE labels 0')
            rngs['after_inputs'] = rng(torch)
            mem.append(memory(torch, 'after_input_allocation'))
            mark('forward_encoders')
            out = tg.training_forward(torch, F, util, [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')],
                                      x_spoof, x_live, label_spoof, lam, criterion_type, criterionL2, mark=mark)
            del x_spoof, x_live  # pinned locals rebind img_nir/img_vis to 128px; 256px survive only inside img
            mark('loss_assembly')
            loss = tg.total_loss(out, EPOCH)
            autocast = [torch.is_autocast_enabled('cuda')]
            for name in ('rec_nir', 'rec_vis', 'rec_nir_fc', 'rec_vis_fc', 'pre_spoof', 'z_cls', 'z_nir', 'z_vis'):
                out[name].register_hook(hook(name))
            rngs['after_forward'] = rng(torch)
            mem.append(memory(torch, 'after_forward_loss_construction'))
            grads_before_zero = sum(p.grad is not None for m in models.values() for p in m.parameters())
            mark('zero_grad')
            optimizer.zero_grad()  # train_generator.py:180 (after the forward, as pinned)
            grads_after_zero = sum(p.grad is not None for m in models.values() for p in m.parameters())
            mark('backward')
            loss.backward()        # train_generator.py:181
            autocast.append(torch.is_autocast_enabled('cuda'))
            require(autocast == [False, False], 'autocast off')
            torch.cuda.synchronize()
            mem.append(memory(torch, 'after_backward'))
            inventory = {n: grad_inventory(m) for n, m in models.items()}
            mark('optimizer_step')
            optimizer.step()       # train_generator.py:182
            torch.cuda.synchronize()
            mem.append(memory(torch, 'after_optimizer_step'))
            mark('post_step_audit')
            rngs['after_step'] = rng(torch)
        except torch.OutOfMemoryError as error:
            stats = torch.cuda.memory_stats()
            result['oom'] = {'phase': phase['name'], 'error_type': type(error).__name__, 'error': str(error),
                             'allocator': {k: stats.get(k) for k in (
                                 'allocated_bytes.all.current', 'allocated_bytes.all.peak',
                                 'reserved_bytes.all.current', 'reserved_bytes.all.peak',
                                 'num_alloc_retries', 'num_ooms', 'inactive_split_bytes.all.current')},
                             'memory_at_oom': memory(torch, 'at_oom'), 'gpu_at_oom': gpu_snapshot(),
                             'memory_summary': torch.cuda.memory_summary(),
                             'optimizer_step_entered': counters['optimizer_step_entries'] > 0,
                             'optimizer_application_occurred': counters['optimizer_applications'] > 0}
            out = loss = None  # release the partial graph before the parameter comparison
            torch.cuda.empty_cache()
            change = {n: parameter_change(initial[n], models[n]) for n in models}
            result['oom']['parameters_changed'] = {n: c['changed_elements'] for n, c in change.items()}
            result['oom']['any_parameter_changed'] = any(c['changed_elements'] for c in change.values())
        result['phase_seconds'] = timing
        result['memory_by_phase'] = mem
        result['rng'] = rngs
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if 'oom' in result:
            result['status'] = 'STOP_OOM'
        else:
            result.update(post_step(torch, F, models, optimizer, out, loss, lam, inventory, initial, params,
                                    grads_seen, grads_before_zero, grads_after_zero))
            result['status'] = 'PASS'
    result['counters'] = counters
    result['gpu_after'] = gpu_snapshot()
    result['source_after'] = m6d5a.source_identity()[0]
    require(result['source_after'] == source, 'source-cache integrity after execution')
    result['asset_after'] = m6d5a.asset_identity(adapter)
    require(result['asset_after'] == result['asset_before'], 'LightCNN unchanged')
    result['environment_after'] = m6d5a.environment()
    require(result['environment_after'] == env, 'environment stable')
    require(counters['autograd_grad_calls'] == counters['checkpoint_saves']
            == counters['activation_checkpoint_calls'] == 0, 'no forbidden call')
    passed = result['status'] == 'PASS'
    if passed:
        require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_passes'],
                 counters['autograd_backward_calls']) == (1, 1, 1, 1), 'exactly one optimizer/step/backward')
    result.update(optimizer_constructed=counters['optimizer_constructions'] == 1,
                  optimizer_applications=counters['optimizer_applications'], backward_passes=counters['backward_passes'],
                  physical_batch_size=tg.PHYSICAL_BATCH, gradient_accumulation_steps=1, replica_factor=1,
                  physical_batch_240_feasible=passed, training_graph_executed=passed,
                  checkpoint_created=False, benchmark_training=False, benchmark_data_access=False,
                  TEST_access=False, synthetic_bank=False, fidelity='CONTROLLED_ADAPTATION', compatibility_patch='NONE',
                  firewall={'denied': firewall.denied, 'event_counts': firewall.events,
                            'lightcnn_read_opens': firewall.lightcnn_opens})
    require(not firewall.denied, 'firewall denials')
    return result


def post_step(torch, F, models, optimizer, out, loss, lam, inventory, initial, params, grads_seen,
              grads_before_zero, grads_after_zero):
    r = {}
    values = {k: out[k].item() for k in tg.LOSS_TERMS}
    require(all(map(lambda v: v == v and abs(v) != float('inf'), values.values())), 'finite loss components')
    with torch.no_grad():
        post = tg.total_loss({k: out[k].detach() for k in tg.LOSS_TERMS}, 2).item()
    f64 = {k: float(v) for k, v in values.items()}
    r['losses'] = {'components': values, 'epoch1_total_executed': loss.item(),
                   'epoch1_total_float64_from_components': tg.total_loss(f64, 1),
                   'postwarmup_total_no_grad_tensor': post,
                   'postwarmup_total_float64_from_components': tg.total_loss(f64, 2),
                   'postwarmup_evaluated_without_backward_or_step': True,
                   'epoch1_formula': 'rec + 0.01*(kl + mmd + ip + pair + cls + ort)',
                   'postwarmup_formula': 'rec + kl + mmd + ip + pair + cls + ort'}
    require(abs(r['losses']['epoch1_total_float64_from_components'] - loss.item()) <= 1e-5 * abs(loss.item()) + 1e-6,
            'epoch-1 assembly')
    require(values['loss_cls'] == 0.0 and values['loss_pair'] == 0.0, 'cls/pair exactly zero')
    r['full_batch'] = full_batch_proofs(torch, F, out, lam)
    fb = r['full_batch']
    require(all(fb['abs_diff'][k] <= 1e-4 * max(1.0, abs(fb['recomputed_float64'][k])) for k in fb['abs_diff']),
            'float64 recomputation agreement')
    r['tensors'] = {k: summary(out[k]) for k in ('mu_nir', 'logvar_nir', 'mu_a', 'logvar_a', 'mu_vis', 'logvar_vis',
                                                 'z_cls', 'z_nir', 'z_vis', 'pre_spoof', 'rec', 'img', 'nir_fc',
                                                 'vis_fc', 'rec_nir_fc', 'rec_vis_fc')}
    require(r['tensors']['rec']['shape'] == [240, 6, 256, 256] and r['tensors']['img']['shape'] == [240, 6, 256, 256],
            'rec/img [240,6,256,256]')
    require(all(v['finite'] for v in r['tensors'].values()), 'finite graph tensors')
    r['upstream_gradient_hooks'] = grads_seen
    r['ip_semantics'] = {
        'target_features': {'nir_fc_requires_grad': out['nir_fc'].requires_grad, 'vis_fc_requires_grad': out['vis_fc'].requires_grad,
                            'detach_applied_in_loss_ip': True},
        'reconstruction_features': {'rec_nir_fc_requires_grad': out['rec_nir_fc'].requires_grad,
                                    'rec_nir_fc_grad_fn': type(out['rec_nir_fc'].grad_fn).__name__,
                                    'rec_vis_fc_requires_grad': out['rec_vis_fc'].requires_grad,
                                    'detached': False},
        'gradient_reaching_rec_128_through_frozen_netIP': {k: grads_seen[k] for k in ('rec_nir', 'rec_vis')},
        'note': 'rec_nir/rec_vis (128px) feed only rgb2gray -> netIP, so their gradient passed through frozen LightCNN'}
    require(not out['nir_fc'].requires_grad and not out['vis_fc'].requires_grad, 'target features carry no graph')
    require(out['rec_nir_fc'].requires_grad and out['rec_vis_fc'].requires_grad, 'reconstructed features not detached')
    require(all(grads_seen[k]['nonzero'] > 0 and grads_seen[k]['finite'] for k in ('rec_nir', 'rec_vis', 'rec_nir_fc')),
            'gradient through LightCNN into reconstruction')
    r['cls_semantics'] = {'loss_cls': values['loss_cls'], 'pre_spoof_shape': list(out['pre_spoof'].shape),
                          'pre_spoof_grad': grads_seen['pre_spoof'],
                          'netCls_grad_nonzero_elements': inventory['netCls']['nonzero_elements'],
                          'status': 'EXPECTED_DEGENERACY_CONFIRMED', 'converted_to_two_logits': False}
    require(grads_seen['pre_spoof']['nonzero'] == 0 and list(out['pre_spoof'].shape) == [240, 1], 'one-logit zero gradient')
    r['gradient_inventory'] = inventory
    for n in tg.OPTIMIZER_OWNED:
        g = inventory[n]
        require(g['non_none'] == g['tensors'] == g['finite'], 'finite non-None gradients ' + n)
        require(g['nonzero_tensors'] > 0, 'gradient connectivity ' + n)
    require(inventory['netIP']['non_none'] == 0, 'netIP has no parameter gradients')
    r['zero_grad'] = {'parameters_with_grad_before_zero_grad': grads_before_zero,
                      'parameters_with_grad_after_zero_grad': grads_after_zero,
                      'first_step_semantics_identical_to_pytorch_1_6': grads_before_zero == grads_after_zero == 0,
                      'owned_tensors_with_grad_after_backward': sum(inventory[n]['non_none'] for n in tg.OPTIMIZER_OWNED),
                      'reason': ('all grads are None before the first zero_grad(); set_to_none=True and the 1.6 in-place '
                                 'zero both leave them None, so backward assigns identical fresh gradients')}
    require(grads_before_zero == grads_after_zero == 0, 'fresh gradients for step 1')
    state_ids = set(optimizer.state.keys())
    owned = [p for n in tg.OPTIMIZER_OWNED for p in models[n].parameters()]
    steps = sorted({float(s['step']) for s in optimizer.state.values()})
    r['optimizer_state'] = {'entries': len(optimizer.state), 'keys': sorted({k for s in optimizer.state.values() for k in s}),
                            'step_values': steps, 'only_owned': state_ids == set(owned),
                            'finite_moments': all(bool(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all())
                                                  for s in optimizer.state.values())}
    require(len(optimizer.state) == 55 and state_ids == set(owned) and steps == [1.0], 'Adam state 55 x step 1')
    change = {n: parameter_change(initial[n], models[n]) for n in models}
    r['parameter_change'] = change
    for n in ('netCls', 'netIP'):
        require(change[n]['changed_elements'] == 0, n + ' unchanged')
    require(all(change[n]['changed_tensors'] > 0 for n in tg.OPTIMIZER_OWNED), 'optimizer-owned parameters changed')
    r['parameters_after_sha256'] = {n: m6d5a.parameter_report(m)['sha256'] for n, m in models.items()}
    require(r['parameters_after_sha256']['netIP'] == params['netIP']['sha256'] and
            r['parameters_after_sha256']['netCls'] == params['netCls']['sha256'], 'netIP/netCls bytes unchanged')
    require(all(not p.requires_grad for p in models['netIP'].parameters()) and not models['netIP'].training,
            'netIP still frozen/eval')
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', default='process.json')
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-2:] == BUILD_PARTS, 'dedicated build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    build.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    result = qualify(build)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
