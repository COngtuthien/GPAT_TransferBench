"""M6D5d E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 qualification, never a training runner.

Three modes, each one fresh GPU process (seed 60504, qualification only):

  reference7  B=7 synthetic batch, max microbatch 3 (chunks 3,3,1): Path A = unchanged
              M6D5b full-batch graph, Path B = V2; identical initial bytes/inputs/epsilon.
  v1compat    B=240: V1 run_global_batch vs V2 run_global_batch_v2 from identical
              initial bytes/inputs/epsilon; compared before the optimizer update.
  b198        the source-consistent final partial global batch: 198 = 9x20 + 18,
              pass 1, 10 backward calls, one Adam step. OOM is STOP_OOM (no retry).

No dataset, DataLoader, image reader, manifest reader, epoch loop or checkpoint
writer exists here. M6D5a/M6D5b/M6D5c helpers are imported unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.dsdg import runtime as m6d5a  # noqa: E402  (unchanged M6D5a helpers)
from methods.dsdg import training_graph as tg  # noqa: E402  (unchanged M6D5b transcription)
from methods.dsdg import training_qualification as m6d5b  # noqa: E402  (unchanged M6D5b helpers)
from methods.dsdg import microbatch_execution as v1  # noqa: E402  (unchanged M6D5c V1)
from methods.dsdg import microbatch_qualification as mq  # noqa: E402  (unchanged M6D5c harness helpers)
from methods.dsdg import microbatch_execution_v2 as v2  # noqa: E402

SEED = 60504  # qualification only; not an experiment seed (42/1337/2026)
EPOCH = 1
BUILD_PARTS = ('builds', 'e06c_dsdg_m6d5d')
TAIL_BATCH = 198
REFERENCE_BATCH, REFERENCE_MAX_MICROBATCH = 7, 3
EXIT = {'PASS': 0, 'STOP_OOM': 3, 'STOP_RESOURCE_CONTAMINATION': 4, 'STOP_REFERENCE_GATE': 5, 'STOP_V1_COMPAT_GATE': 6}
require, sha, tsha = mq.require, mq.sha, mq.tsha


def setup(build, mode):
    """M6D5c setup (V1 overlay still binds) seeded with this milestone's qualification seed, plus V2 gates."""
    mq.SEED = SEED  # mq.setup seeds random/numpy/torch/cuda from its module global at call time
    result, ctx = mq.setup(build, mode)
    resolution = v2.load_resolution()
    result.update(execution_mode=v2.EXECUTION_MODE, diagnostic_seed=SEED,
                  v1_resolution_sha256=result.pop('resolution_sha256'),
                  v2_resolution_sha256=sha((ROOT / v2.RESOLUTION_PATH).read_bytes()),
                  v2_execution_policy=v2.execution_guard(resolution))
    root = Path(result['source_before']['root']) / 'addition_module/DSDG'
    mismatch = v2.verify_pinned_dataloader((root / 'train_generator.py').read_text())
    require(not mismatch, 'pinned DataLoader lines 94-97 without drop_last: ' + json.dumps(mismatch))
    plan = v2.epoch_batch_plan()
    require((plan['full_batches'], plan['final_batch'], plan['optimizer_steps_per_epoch'], plan['rows_covered']) ==
            (36, 198, 37, 8838), 'epoch batch plan 36 x 240 + 198')
    result['dataloader_evidence'] = {'pinned_lines': v2.PINNED_DATALOADER, 'mismatches': mismatch,
                                     'drop_last_in_source': False}
    result['epoch_batch_plan'] = {k: v for k, v in plan.items() if k != 'batch_sizes'}
    if ctx is not None:
        import inspect
        torch = ctx['torch']
        default = inspect.signature(torch.utils.data.DataLoader.__init__).parameters['drop_last'].default
        require(default is False, 'runtime DataLoader drop_last default False')
        result['dataloader_evidence']['runtime_torch_default_drop_last'] = default
    return result, ctx


def restore(models, initial):
    for n, mdl in models.items():
        mdl.load_state_dict(initial[n])
        for p in mdl.parameters():
            p.grad = None


def losses_finite(values):
    return all(v == v and abs(v) != float('inf') for v in values)


# ================================================================= reference (B=7, chunks 3,3,1)
def reference7(build):
    result, ctx = setup(build, 'reference7')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    gates = v2.load_resolution()['qualification']['reference_check']['gates']
    B, m = REFERENCE_BATCH, REFERENCE_MAX_MICROBATCH
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = mq.models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: {k: v.detach().clone() for k, v in mdl.state_dict().items()} for n, mdl in models.items()}
        criterion_type, criterionL2 = tg.criteria(torch)
        x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
        label = torch.zeros(B, dtype=torch.long, device='cuda')
        eps, result['epsilon'] = mq.epsilon_draw_proof(torch, util, B)
        rows = [sha(r.cpu().numpy().tobytes()) for x in (x_spoof, x_live) for r in x]
        require(len(set(rows)) == 2 * B, 'distinct reference rows')
        result['inputs'] = {'x_spoof': m6d5b.summary(x_spoof), 'x_live': m6d5b.summary(x_live), 'batch': B,
                            'max_microbatch': m, 'chunk_sizes': [s.stop - s.start for s in v2.chunk_slices_v2(B, m)],
                            'distinct_rows': len(set(rows))}
        require(result['inputs']['chunk_sizes'] == [3, 3, 1], 'chunks 3,3,1')
        owned = mq.owned_params(models)
        opt_a = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        mq.count_steps(opt_a, counters)
        out = tg.training_forward(torch, F, mq.pinned_util_shim(util, eps), nets, x_spoof, x_live, label, lam,
                                  criterion_type, criterionL2)
        loss_a = tg.total_loss(out, EPOCH)
        path_a = {'losses': {k: out[k].item() for k in tg.LOSS_TERMS}, 'epoch1_total': loss_a.item()}
        opt_a.zero_grad()
        loss_a.backward()
        grads_a = [p.grad.detach().clone() for _, p in owned]
        path_a['owned_grad_non_none'] = sum(p.grad is not None for _, p in owned)
        opt_a.step()
        delta_a = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
        del out, loss_a, opt_a
        restore(models, initial)
        restored = {n: m6d5a.parameter_report(mdl)['sha256'] for n, mdl in models.items()}
        require(restored == {n: p['sha256'] for n, p in params.items()}, 'identical initial model bytes restored')
        opt_b = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        mq.count_steps(opt_b, counters)
        grads_b = []
        run = v2.run_global_batch_v2(torch, F, util, nets, opt_b, x_spoof, x_live, label, eps, lam, criterion_type,
                                     criterionL2, EPOCH, max_microbatch=m,
                                     before_step=lambda: grads_b.extend(p.grad.detach().clone() for _, p in owned))
        delta_b = [(p.detach() - initial[n.split('.')[0]][n.split('.', 1)[1]]).clone() for n, p in owned]
    g = run['global_losses']
    rel = lambda a, b: abs(a - b) / max(abs(a), 1e-30)  # noqa: E731
    grad_cmp = mq.compare_vectors(torch, grads_a, grads_b)
    step_cmp = mq.compare_vectors(torch, delta_a, delta_b)
    result['path_a_full_batch'] = path_a
    result['path_b_v2'] = {'losses': g, 'epoch1_total': run['epoch1_total'], 'chunk_sizes': run['chunk_sizes'],
                           'surrogate_sums': run['surrogate_sums'], 'sum_of_chunk_objectives': run['sum_of_chunk_objectives'],
                           'owned_grad_non_none': len(grads_b)}
    result['comparison'] = {'loss_rel_diff': {k: rel(path_a['losses'][k], g[k]) for k in tg.LOSS_TERMS},
                            'total_rel_error': rel(path_a['epoch1_total'], run['epoch1_total']),
                            'gradient': grad_cmp, 'update_reported_not_gated': step_cmp}
    result['gates'] = {
        'scalar_losses_finite': losses_finite(list(path_a['losses'].values()) + list(g.values())),
        'total_loss_relative_error': result['comparison']['total_rel_error'] <= gates['total_loss_relative_error_max'],
        'aggregate_gradient_cosine': grad_cmp['cosine'] >= gates['aggregate_gradient_cosine_min'],
        'aggregate_gradient_relative_l2': grad_cmp['relative_l2'] <= gates['aggregate_gradient_relative_l2_max'],
        'owned_gradient_coverage': path_a['owned_grad_non_none'] == len(grads_b) == 55}
    result['gate_thresholds'] = gates
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['status'] = 'PASS' if all(result['gates'].values()) else 'STOP_REFERENCE_GATE'
    mq.finish(result, ctx)
    require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_calls'])
            == (2, 2, 1 + 3), 'reference accounting: 2 optimizers, 2 steps, 1 + 3 backward')
    result.update(reference_global_batch=B, reference_max_microbatch=m, optimizer_constructions=2,
                  optimizer_applications=2, backward_calls=4)
    return result


# ================================================================= V1 vs V2 at B=240
def v1compat(build):
    result, ctx = setup(build, 'v1compat')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    gates = v2.load_resolution()['qualification']['v1_compatibility']['gates']
    B = v1.GLOBAL_BATCH
    paths = {}
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = mq.models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: {k: v.detach().clone() for k, v in mdl.state_dict().items()} for n, mdl in models.items()}
        criterion_type, criterionL2 = tg.criteria(torch)
        x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
        label = torch.zeros(B, dtype=torch.long, device='cuda')
        result['inputs'] = {k: m6d5b.summary(x)['sha256'] for k, x in (('x_spoof', x_spoof), ('x_live', x_live))}
        require(result['inputs'] == mq.M6D5B_INPUT_SHA256, 'B=240 inputs equal M6D5b/M6D5c inputs')
        eps, result['epsilon'] = mq.epsilon_draw_proof(torch, util, B)
        owned = mq.owned_params(models)
        for name, runner, kw in (('v1', v1.run_global_batch, {}), ('v2', v2.run_global_batch_v2, {})):
            restore(models, initial)
            require({n: m6d5a.parameter_report(mdl)['sha256'] for n, mdl in models.items()} ==
                    {n: p['sha256'] for n, p in params.items()}, 'identical initial bytes before ' + name)
            opt = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
            mq.count_steps(opt, counters)
            cap = {}

            def before_step():
                cap['grads'] = [p.grad.detach().clone() for _, p in owned]
                cap['inventory'] = {n: {k: v for k, v in m6d5b.grad_inventory(mdl).items() if k != 'per_tensor'}
                                    for n, mdl in models.items()}
            backward_before = counters['backward_calls']
            run = runner(torch, F, util, nets, opt, x_spoof, x_live, label, eps, lam, criterion_type, criterionL2,
                         EPOCH, before_step=before_step, **kw)
            st = run['stats']
            paths[name] = {'losses': run['global_losses'], 'epoch1_total': run['epoch1_total'],
                           'chunks': run['chunks'], 'backward_calls': counters['backward_calls'] - backward_before,
                           'chunk_sizes': run.get('chunk_sizes', [s.stop - s.start for s in v1.chunk_slices()]),
                           'delta': st['delta'].detach().clone(), 'ort_mean': st['ort_mean'].item(),
                           'mmd_sign': st['mmd_sign'].detach().clone(), 'ort_sign': st['ort_sign'].item(),
                           'grads': cap['grads'], 'inventory': cap['inventory'],
                           'params_after_sha256': {k: tsha(p) for k, p in owned}}
            del opt, run
    a, b = paths['v1'], paths['v2']
    rel = lambda x, y: abs(x - y) / max(abs(x), 1e-30)  # noqa: E731
    grad_cmp = mq.compare_vectors(torch, a['grads'], b['grads'])
    cmp = {'chunk_sizes': {'v1': a['chunk_sizes'], 'v2': b['chunk_sizes']},
           'backward_calls': {'v1': a['backward_calls'], 'v2': b['backward_calls']},
           'global_losses': {'v1': a['losses'], 'v2': b['losses']},
           'global_loss_rel_diff': {k: rel(a['losses'][k], b['losses'][k]) for k in a['losses']},
           'epoch1_total': {'v1': a['epoch1_total'], 'v2': b['epoch1_total']},
           'delta_max_abs_diff': float((a['delta'] - b['delta']).abs().max()),
           'delta_bitwise_equal': bool(torch.equal(a['delta'], b['delta'])), 'delta_sha256': tsha(a['delta']),
           'ort_mean': {'v1': a['ort_mean'], 'v2': b['ort_mean']},
           'mmd_sign_equal': bool(torch.equal(a['mmd_sign'], b['mmd_sign'])),
           'ort_sign': {'v1': a['ort_sign'], 'v2': b['ort_sign']},
           'gradient': grad_cmp,
           'gradient_bitwise_equal': all(bool(torch.equal(x, y)) for x, y in zip(a['grads'], b['grads'])),
           'gradient_inventory': {'v1': a['inventory'], 'v2': b['inventory']},
           'post_step_parameters_bitwise_equal_reported_not_gated': a['params_after_sha256'] == b['params_after_sha256']}
    result['comparison'] = cmp
    result['gates'] = {
        'chunk_inventory_equal': a['chunk_sizes'] == b['chunk_sizes'] == [20] * 12 and a['chunks'] == b['chunks'] == 12,
        'global_loss_relative_diff': losses_finite(list(a['losses'].values()) + list(b['losses'].values())) and
        max(cmp['global_loss_rel_diff'].values()) <= gates['global_loss_relative_diff_max'],
        'delta_max_abs_diff': cmp['delta_max_abs_diff'] <= gates['delta_max_abs_diff'],
        'ort_mean_abs_diff': abs(a['ort_mean'] - b['ort_mean']) <= gates['ort_mean_abs_diff'],
        'mmd_sign_equal': cmp['mmd_sign_equal'], 'ort_sign_equal': a['ort_sign'] == b['ort_sign'],
        'gradient_cosine': grad_cmp['cosine'] >= gates['gradient_cosine_min'],
        'gradient_relative_l2': grad_cmp['relative_l2'] <= gates['gradient_relative_l2_max'],
        'gradient_inventory_non_none_equal': all(a['inventory'][n]['non_none'] == b['inventory'][n]['non_none']
                                                 for n in a['inventory']) and
        sum(a['inventory'][n]['non_none'] for n in tg.OPTIMIZER_OWNED) == 55}
    result['gate_thresholds'] = gates
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['status'] = 'PASS' if all(result['gates'].values()) else 'STOP_V1_COMPAT_GATE'
    mq.finish(result, ctx)
    require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_calls'])
            == (2, 2, 24), 'v1compat accounting: 2 optimizers, 2 steps, 24 backward')
    result.update(global_batch_size=B, optimizer_constructions=2, optimizer_applications=2, backward_calls=24)
    return result


# ================================================================= B=198 final partial global batch
def b198(build):
    result, ctx = setup(build, 'b198')
    if ctx is None:
        return result
    torch, F, lam, counters = ctx['torch'], ctx['F'], ctx['lam'], ctx['counters']
    from methods.common.upstream import upstream_modules
    B = TAIL_BATCH
    expected_chunks = v2.load_resolution()['final_batch_chunks']
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
        s = chunks[i]
        row = {'chunk': i, 'rows': [s.start, s.stop], 'size': s.stop - s.start,
               'nir_fc_requires_grad': out['nir_fc'].requires_grad, 'vis_fc_requires_grad': out['vis_fc'].requires_grad,
               'rec_nir_fc_requires_grad': out['rec_nir_fc'].requires_grad,
               'rec_vis_fc_requires_grad': out['rec_vis_fc'].requires_grad,
               'rec_nir_fc_grad_fn': type(out['rec_nir_fc'].grad_fn).__name__,
               'losses': {k: out[k].item() for k in v1.LOCAL_TERMS + ('loss_mmd_surrogate', 'loss_ort_surrogate')},
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

    chunks = v2.chunk_slices_v2(B)
    require([s.stop - s.start for s in chunks] == expected_chunks, 'B=198 chunks 9x20 + 18')
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(ctx['root'], m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        util = modules['misc.util']
        models, params = mq.models_ready(torch, ctx, result, modules['networks'])
        nets = [models[n] for n in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')]
        initial = {n: m6d5b.cpu_copy(models[n]) for n in models}
        snap('after_model_construction')
        run = optimizer = None
        state = {}
        try:
            mark('optimizer_construction')
            optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                           ctx['cfg']['optimizer']['learning_rate'])
            mq.count_steps(optimizer, counters)
            result['optimizer'] = m6d5b.optimizer_evidence(torch, optimizer, models)
            criterion_type, criterionL2 = tg.criteria(torch)
            mark('input_allocation')
            x_spoof, x_live = (x.cuda() for x in tg.synthetic_pair(torch, B, 'cpu'))
            label = torch.zeros(B, dtype=torch.long, device='cuda')
            rows = [sha(r.cpu().numpy().tobytes()) for x in (x_spoof, x_live) for r in x]
            result['inputs'] = {'x_spoof': m6d5b.summary(x_spoof), 'x_live': m6d5b.summary(x_live),
                                'label_spoof': m6d5b.summary(label), 'distinct_rows': len(set(rows)), 'rows': len(rows),
                                'label': 'SYNTHETIC_FINAL_PARTIAL_GLOBAL_BATCH_198',
                                'construction': 'methods/dsdg/training_graph.py::synthetic_pair(198) (unchanged M6D5b)'}
            require(len(set(rows)) == 2 * B and result['inputs']['x_spoof']['shape'] == [B, 3, 256, 256] and
                    result['inputs']['x_live']['shape'] == [B, 3, 256, 256], '396 distinct [198,3,256,256] rows')
            snap('after_optimizer_and_input_allocation')
            mark('epsilon_allocation')
            eps, result['epsilon'] = mq.epsilon_draw_proof(torch, util, B)
            require(result['epsilon']['shape'] == [B, 128], 'epsilon [198,128] (not truncated from 240)')
            rngs['after_epsilon'] = m6d5b.rng(torch)
            snap('after_epsilon_allocation', reset=True)
            grads_before = sum(p.grad is not None for mdl in models.values() for p in mdl.parameters())
            mark('pass1')

            def before_step():
                snap('before_optimizer_step')
                state['inventory'] = {n: m6d5b.grad_inventory(mdl) for n, mdl in models.items()}
                state['grad_sha256'] = {k: tsha(p.grad) for k, p in mq.owned_params(models)}
                rngs['before_step'] = m6d5b.rng(torch)
                mark('optimizer_step')
            run = v2.run_global_batch_v2(torch, F, util, nets, optimizer, x_spoof, x_live, label, eps, lam,
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
                                 'reserved_bytes.all.current', 'reserved_bytes.all.peak', 'num_alloc_retries', 'num_ooms')},
                             'memory_summary': torch.cuda.memory_summary(),
                             'optimizer_application_occurred': counters['optimizer_applications'] > 0}
            run = None
            torch.cuda.empty_cache()
        result['phase_seconds'] = timing
        result['memory_by_phase'] = mem
        result['overall_peak'] = {'allocated_bytes': peaks['allocated'], 'reserved_bytes': peaks['reserved']}
        result['rng'] = rngs
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        if 'oom' in result:
            result['status'] = 'STOP_OOM'
        else:
            result.update(post_step(torch, models, optimizer, run, state, initial, params, hooks_seen, rngs,
                                    grads_before, counters, lam, chunks))
            result['status'] = 'PASS'
    mq.finish(result, ctx)
    passed = result['status'] == 'PASS'
    if passed:
        require((counters['optimizer_constructions'], counters['optimizer_applications'], counters['backward_calls'],
                 counters['autograd_backward_calls']) == (1, 1, 10, 10), 'one optimizer, one step, 10 backward calls')
    result.update(global_batch_size=B, nominal_global_batch=v2.NOMINAL_GLOBAL_BATCH, max_microbatch=v2.MAX_MICROBATCH,
                  chunk_sizes=[s.stop - s.start for s in chunks], final_chunk=chunks[-1].stop - chunks[-1].start,
                  drop_last=False, optimizer_constructions=counters['optimizer_constructions'],
                  optimizer_applications=counters['optimizer_applications'], backward_calls=counters['backward_calls'],
                  physical_batch_240='OOM_RETAINED', tail_batch_198_executed=passed, VAL_access=False)
    return result


def post_step(torch, models, optimizer, run, state, initial, params, hooks_seen, rngs, grads_before, counters, lam,
              chunks):
    r = {}
    B = run['global_batch']
    g = run['global_losses']
    require(losses_finite(g.values()), 'finite global losses')
    require(g['loss_cls'] == 0.0 and g['loss_pair'] == 0.0, 'cls/pair exactly zero')
    require(run['chunk_sizes'] == [s.stop - s.start for s in chunks] and run['chunks'] == len(chunks) == 10,
            'executed chunk plan')
    st = run['stats']
    z1 = {k: torch.cat([c[k] for c in st['latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
    z2 = {k: torch.cat([c[k] for c in run['pass2_latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
    require(all(v.shape[0] == B for v in list(z1.values()) + list(z2.values())), 'latents cover all 198 rows')
    zn, zv, zc = (z1[k].double().cpu() for k in ('z_nir', 'z_vis', 'z_cls'))
    delta64 = zn.mean(0) - zv.mean(0)
    ort64 = (zc * zn).sum(1).mean()
    naive_mmd = sum(float(lam['lambda_mmd'] * (zn[s].mean(0) - zv[s].mean(0)).abs().mean()) for s in chunks) / len(chunks)
    naive_ort = sum(float(lam['lambda_ort'] * (zc[s] * zn[s]).sum(1).mean().abs()) for s in chunks) / len(chunks)
    p2 = {k: v.double().cpu() for k, v in z2.items()}
    delta_p2 = p2['z_nir'].mean(0) - p2['z_vis'].mean(0)
    r['losses'] = {'global_components': g, 'epoch1_total_reconstructed': run['epoch1_total'],
                   'postwarmup_total_algebraic_only': run['postwarmup_total_algebraic_only'], 'postwarmup_executed': False,
                   'sum_of_chunk_objectives': run['sum_of_chunk_objectives'],
                   'epoch1_formula': 'rec + 0.01*(kl + mmd + ip + pair + cls + ort)',
                   'local_weights': [h['size'] / B for h in hooks_seen],
                   'per_chunk': [h['losses'] for h in hooks_seen]}
    r['global_statistics'] = {
        'delta': {'shape': list(st['delta'].shape), 'sha256': tsha(st['delta']), 'l1_mean': float(st['delta'].abs().mean()),
                  'zero_coordinates': int((st['delta'] == 0).sum()), 'positive': int((st['delta'] > 0).sum()),
                  'negative': int((st['delta'] < 0).sum())},
        'ort_mean': st['ort_mean'].item(), 'ort_sign': st['ort_sign'].item(), 'mmd_sign_sha256': tsha(st['mmd_sign']),
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
    for k in ('mmd', 'ort'):
        a, b = gs['loss_%s_global_fp32' % k], gs['loss_%s_global_float64_recomputed' % k]
        require(abs(a - b) <= 1e-4 * max(1.0, abs(b)), k + ' global float64 agreement')
    for k, v in (('loss_mmd_surrogate', gs['loss_mmd_global_fp32']), ('loss_ort_surrogate', gs['loss_ort_global_fp32'])):
        require(abs(run['surrogate_sums'][k] - v) <= 1e-4 * max(1.0, abs(v)), 'surrogate sum equals global ' + k)
    r['latents'] = {k: m6d5b.summary(v) for k, v in z1.items()}
    r['chunk_evidence'] = hooks_seen
    require([h['size'] for h in hooks_seen] == [s.stop - s.start for s in chunks], '10 pass-2 chunks in order')
    for h in hooks_seen:
        require(not h['nir_fc_requires_grad'] and not h['vis_fc_requires_grad'], 'target features under no_grad')
        require(h['rec_nir_fc_requires_grad'] and h['rec_vis_fc_requires_grad'], 'reconstructed features keep graph')
        require(all(h['grads'][k]['nonzero'] > 0 and h['grads'][k]['finite'] for k in ('rec_nir', 'rec_vis', 'rec_nir_fc')),
                'gradient through frozen LightCNN into reconstruction, chunk %d' % h['chunk'])
        require(h['grads']['pre_spoof']['nonzero'] == 0 and h['grads']['pre_spoof']['shape'] == [h['size'], 1],
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
    r['zero_grad'] = {'calls': zg, 'set_to_none': False, 'grads_before_first_zero_grad': grads_before}
    require(len(zg) == 1 and zg[0] == {'set_to_none': False, 'backward_calls_before': 0},
            'zero_grad(set_to_none=False) exactly once before the first chunk')
    require(rngs['after_epsilon'] == rngs['after_pass1'] == rngs['before_step'] == rngs['after_step'],
            'no RNG draw after the single [198,128] epsilon draw')
    r['rng_no_redraw'] = True
    owned = [p for n in tg.OPTIMIZER_OWNED for p in models[n].parameters()]
    steps = sorted({float(s['step']) for s in optimizer.state.values()})
    r['optimizer_state'] = {'entries': len(optimizer.state), 'step_values': steps,
                            'only_owned': set(optimizer.state) == set(owned),
                            'keys': sorted({k for s in optimizer.state.values() for k in s}),
                            'exp_avg_sha256': {k: tsha(optimizer.state[p]['exp_avg']) for k, p in mq.owned_params(models)},
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
    require(r['parameters_after_sha256']['netCls'] == params['netCls']['sha256'] and
            r['parameters_after_sha256']['netIP'] == params['netIP']['sha256'], 'netCls/netIP bytes unchanged')
    require(all(not p.requires_grad for p in models['netIP'].parameters()) and not models['netIP'].training,
            'netIP still frozen/eval')
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reference7', 'v1compat', 'b198'), required=True)
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
    result = {'reference7': reference7, 'v1compat': v1compat, 'b198': b198}[args.mode](build)
    result['started_utc'], result['ended_utc'] = started, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
