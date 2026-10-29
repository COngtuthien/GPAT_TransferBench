"""E04 D12 stage P: ordinary production-path step replay (diagnostic only; qualification seed 60904; never a scientific
run; original graph and device placement; no RunOptions / tracing in any session call).

Builds the production Controller exactly as tools/run_e04.py would (production_runner.prepare: contract, environment,
feed pack, unchanged TrainingGraph), opens NO RunContext and writes nothing outside --output-dir (under /tmp). Then, on
real TRAIN data from the qualified Feed, for each of the protocol's iterations and each step s of the frozen order:
  * forward integrity: the D10 localization algorithm over the forward closure (all losses, D1-D4 predictions, G
    outputs) at the pre-iteration state, 3 plain repeats, with per-tensor magnitudes;
  * production replays: K times, restore the exact pre-step state (feeding every variable's own initializer input; no
    op is added to the graph), then run exactly Controller.iteration's calls for step s (the BN update calls of
    bn_paths[s], then [apply[s]] + the step losses); record the post-step state of every variable;
  * semantic replays: twice, restore the same state, run the BN update calls, then [apply[s], step gradients, losses]
    in ONE call, and check against numpy: Adam (m, v, var, beta powers), the set of variables each call changes,
    phase / iteration counters and finiteness; the two semantic replays give the gradient repeatability magnitude.
The trajectory continues from the first production replay. Static checks: gradient connectivity against graph
reachability, one ApplyAdam per present gradient targeting its variable, no stateful random op on any step path.

    python -B tools/e04_d12_production_replay.py --repo <checkout> --process P1 --output-dir /tmp/gpat_e04_d12/P/P1
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
PROTOCOL = 'outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json'
ITERATIONS = 3
PRODUCTION_REPLAYS = 3
SEMANTIC_REPLAYS = 2
FORWARD_REPEATS = 3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
RANDOM_TYPES = ('RandomUniform', 'RandomStandardNormal', 'TruncatedNormal', 'RandomShuffle', 'Multinomial',
                'RandomGamma', 'RandomPoisson', 'RandomUniformInt', 'StatelessRandomUniform')


def ulp32(x):
    import numpy as np
    x = np.abs(np.asarray(x, np.float32))
    return (np.nextafter(x, np.float32(np.inf)) - x).astype(np.float64)


def adam_expected(var, m, v, g, lr, b1p, b2p):
    """TF 1.15 ApplyAdam (use_nesterov=False) in float64 from the float32 inputs, including the float32 beta1 / beta2 /
    epsilon constants the kernel receives."""
    import numpy as np
    b1, b2, eps = (float(np.float32(x)) for x in (BETA1, BETA2, EPS))
    var, m, v, g = (np.asarray(a, np.float64) for a in (var, m, v, g))
    lr_t = float(np.float32(lr)) * (1 - float(np.float32(b2p))) ** .5 / (1 - float(np.float32(b1p)))
    m1 = m + (g - m) * (1 - b1)
    v1 = v + (g * g - v) * (1 - b2)
    return var - lr_t * m1 / (np.sqrt(v1) + eps), m1, v1


def within(actual, expected, before, ulps=4, rel=1e-5):
    """Per-element tolerance |actual - expected| <= ulps * ulp32(expected) + rel * |expected - before|."""
    import numpy as np
    a, e, b = (np.asarray(x, np.float64) for x in (actual, expected, before))
    err = np.abs(a - e)
    tol = ulps * ulp32(e) + rel * np.abs(e - b)
    return bool((err <= tol).all()), float(err.max()) if err.size else 0., float((err / np.maximum(tol, 1e-300)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--process', required=True, choices=['P1', 'P2', 'P3'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--iterations', type=int, default=ITERATIONS)
    args = ap.parse_args()
    out_dir = Path(os.path.abspath(args.output_dir))
    if not str(out_dir).startswith('/tmp/'):
        raise SystemExit('output must be under /tmp')
    out_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / 'tools'))
    sys.path.insert(0, str(repo))
    import numpy as np
    import e04_d10_numerical_probe as base
    from e04_d12_confound_probe import DiffSession, annotate, compare
    from methods.physics_std import production_io as pio
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    if errors:
        raise SystemExit('launch environment: %s' % errors)
    import tensorflow as tf
    from methods.physics_std import production_runner as pr
    from methods.physics_std.production_qualification import Firewall, RUNTIME, FACES
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, RUNTIME, FACES)
    m, graph = ctl.model, tf.get_default_graph()
    init = tf.global_variables_initializer()             # the only op this tool adds, before the count
    n_ops = len(graph.get_operations())
    fw = Firewall()
    fw.pack_ids = set(ctl.pack.samples)
    variables = ctl.variables
    vnames = [v.name for v in variables]
    index = {n: i for i, n in enumerate(vnames)}
    restore_ops = [v.initializer for v in variables]
    restore_inputs = [v.initializer.inputs[1] for v in variables]
    sha = base.sha

    # ------------------------------------------------------------------------------------------ static checks
    def closure(items):
        """Ancestor op names of tensors or operations (data and control inputs)."""
        seen, stack = set(), [getattr(x, 'op', x) for x in items]
        while stack:
            o = stack.pop()
            if o.name in seen:
                continue
            seen.add(o.name)
            stack.extend(i.op for i in o.inputs)
            stack.extend(o.control_inputs)
        return seen

    static = {'construction': list(ctl.construction), 'structure': ctl.structure}

    def differentiable_ancestors(loss):
        """Ops reachable from the loss backwards through floating-point edges, not crossing StopGradient."""
        seen, stack = set(), [loss.op]
        while stack:
            o = stack.pop()
            if o.name in seen:
                continue
            seen.add(o.name)
            if o.type in ('StopGradient', 'PreventGradient'):
                continue
            stack.extend(i.op for i in o.inputs if i.dtype.base_dtype.is_floating)
        return seen

    conn = {}
    sets = dict([(k, (m.losses[k], m.per_loss_gradients[k])) for k in m.per_loss_gradients] +
                [('step%d' % s, (m.losses[k], m.step_gradients[s])) for s, k in
                 ((1, 'loss_G_step1'), (2, 'loss_D'), (3, 'loss_G_step3'))])
    for key, (loss, pairs) in sorted(sets.items()):
        keep = differentiable_ancestors(loss)
        reach = [v.op.name in keep for _, v in pairs]
        present = [g is not None for g, _ in pairs]
        conn[key] = {'variables': len(pairs), 'present': sum(present), 'reachable': sum(reach),
                     'present_equals_reachable': present == reach,
                     'all_present_finite_shape': all(g.shape.as_list() == v.shape.as_list()
                                                     for g, v in pairs if g is not None)}
    static['gradient_connectivity'] = conn
    adam = {}
    for s in (1, 2, 3):
        keep = closure([m.apply[s]])
        applies = [graph.get_operation_by_name(n) for n in keep if graph.get_operation_by_name(n).type == 'ApplyAdam']
        targets = sorted(o.inputs[0].op.name for o in applies)
        want = sorted(v.op.name for g, v in m.step_gradients[s] if g is not None)
        grads_in = all(any(o.inputs[9].name == g.name for o in applies) for g, _ in m.step_gradients[s] if g is not None)
        adam[s] = {'apply_adam_ops': len(applies), 'present_gradients': len(want), 'targets_equal': targets == want,
                   'apply_inputs_are_step_gradients': grads_in,
                   'random_ops_on_path': sorted(n for n in keep if graph.get_operation_by_name(n).type in RANDOM_TYPES)}
        bn = closure([u for g in m.bn_paths[s] for u in g])
        adam[s]['random_ops_on_bn_path'] = sorted(n for n in bn if graph.get_operation_by_name(n).type in RANDOM_TYPES)
    static['optimizer_application'] = {str(k): v for k, v in adam.items()}

    # Which variables each call may change.
    g_names, d_names = {v.name for v in m.G}, {v.name for v in m.D}
    slot_of = {}
    for opt, group in ((m.opt_G, m.G), (m.opt_D, m.D)):
        for v in group:
            slot_of[v.name] = (opt.get_slot(v, 'm').name, opt.get_slot(v, 'v').name)
    beta = {'G': sorted(b.name for b in m.opt_G._get_beta_accumulators()),
            'D': sorted(b.name for b in m.opt_D._get_beta_accumulators())}

    def assigned(ops):
        out = set()
        for o in ops:
            for n in closure([o]):
                op = graph.get_operation_by_name(n)
                if op.type in ('Assign', 'AssignSub', 'AssignAdd', 'AssignMovingAvg') and op.inputs:
                    out.add(op.inputs[0].op.name + ':0')
        return out
    bn_allowed = {s: [assigned(u) for u in m.bn_paths[s]] for s in (1, 2, 3)}
    for s in (1, 2, 3):
        for a in bn_allowed[s]:
            pio.require(all('moving_' in n for n in a), 'BN update ops assign only moving statistics')
    static['bn_update_variables'] = {str(s): [len(a) for a in bn_allowed[s]] for s in (1, 2, 3)}

    def apply_allowed(s):
        tag = 'D' if s == 2 else 'G'
        out = set(beta[tag]) | {m.phase.name}
        if s == 3:
            out.add(m.iteration.name)
        for g, v in m.step_gradients[s]:
            if g is not None:
                out |= {v.name, *slot_of[v.name]}
        return out

    # ------------------------------------------------------------------------------------------ helpers
    summary = {'protocol_sha256': hashlib.sha256((repo / PROTOCOL).read_bytes()).hexdigest(),
               'tool_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in (
                   'tools/e04_d12_production_replay.py', 'tools/e04_d12_confound_probe.py',
                   'tools/e04_d10_numerical_probe.py')},
               'process': args.process, 'iterations': args.iterations, 'qualification_seed': pio.QUALIFICATION_SEED,
               'identities': ids, 'driver': driver, 'environment_live': {k: live[k] for k in (
                   'python', 'tensorflow', 'native_runtime', 'gpu', 'precision_env', 'session_config')},
               'static': static, 'pid': os.getpid(), 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    sys.addaudithook(fw)
    t0 = time.monotonic()

    def state(sess):
        return sess.run(variables)

    def state_sha(vals):
        return hashlib.sha256(''.join(sha(x) for x in vals).encode()).hexdigest()

    def restore(sess, vals):
        sess.run(restore_ops, dict(zip(restore_inputs, vals)))
        got = state(sess)
        return all(a.tobytes() == b.tobytes() for a, b in zip(got, vals))

    def changed(a, b):
        return {n for n, x, y in zip(vnames, a, b) if np.asarray(x).tobytes() != np.asarray(y).tobytes()}

    def finite(vals):
        return all(np.isfinite(x).all() for x in vals if np.issubdtype(np.asarray(x).dtype, np.floating))

    forward_targets = [m.losses[k] for k in sorted(m.losses)] + list(m.predictions) + \
        [m.outputs[k] for k in sorted(m.outputs)]
    forward_ops = base.closure([m.losses[k] for k in sorted(m.losses)] + list(m.predictions))
    jl = (out_dir / 'iterations.jsonl').open('w')
    with tf.Session(config=pr.session_config(tf)) as sess:
        sess.run(init)
        cur = state(sess)
        summary['initial_state_sha256'] = state_sha(cur)
        for t in range(args.iterations):
            arrays, info = ctl.feed.next()
            fd = ctl.feed_dict(arrays)
            rec = {'iteration': t, 'pair_rows': info['pair_rows'], 'input_sha256': info['input_sha256'],
                   'state_sha256_before': state_sha(cur)}
            proxy = DiffSession(sess)                         # forward integrity at the pre-iteration state
            loc = base.localize(tf, proxy, fd, forward_targets, FORWARD_REPEATS)
            rec['forward'] = {'chunks': len(proxy.chunks), 'fetched_tensors': loc['fetched_tensors'],
                              'closure_ops': loc['closure_ops'], 'varying': annotate(tf, proxy, forward_ops)}
            rec['forward_state_unchanged'] = state_sha(state(sess)) == rec['state_sha256_before']
            steps = {}
            for s in (1, 2, 3):
                keys = pio.STEP_LOSSES[s]
                pre = cur
                lr = sess.run(m.lr_D if s == 2 else m.lr_G)
                prod = []
                for r in range(PRODUCTION_REPLAYS):          # exactly Controller.iteration's calls for step s
                    ok = restore(sess, pre)
                    for updates in m.bn_paths[s]:
                        sess.run(updates, fd)
                    out = sess.run([m.apply[s]] + [m.losses[k] for k in keys], fd)
                    post = state(sess)
                    prod.append({'restore_exact': ok, 'post': post, 'losses': [float(x) for x in out[1:]]})
                sem = []
                pairs = [(g, v) for g, v in m.step_gradients[s] if g is not None]
                for r in range(SEMANTIC_REPLAYS):
                    ok = restore(sess, pre)
                    calls = []
                    before = pre
                    for i, updates in enumerate(m.bn_paths[s]):
                        sess.run(updates, fd)
                        after = state(sess)
                        calls.append({'call': 'bn_%d' % i, 'changed': sorted(changed(before, after)),
                                      'allowed': sorted(bn_allowed[s][i])})
                        before = after
                    mid = before
                    out = sess.run([m.apply[s], [g for g, _ in pairs]] + [m.losses[k] for k in keys], fd)
                    post = state(sess)
                    grads = out[1]
                    calls.append({'call': 'apply', 'changed': sorted(changed(mid, post)),
                                  'allowed': sorted(apply_allowed(s))})
                    tag = 'D' if s == 2 else 'G'
                    b1n = [n for n in beta[tag] if 'beta1_power' in n][0]
                    b2n = [n for n in beta[tag] if 'beta2_power' in n][0]
                    b1p, b2p = mid[index[b1n]], mid[index[b2n]]
                    worst = {'var': 0., 'm': 0., 'v': 0.}
                    adam_ok = True
                    for g, (_, v) in zip(grads, pairs):
                        mn, vn = slot_of[v.name]
                        e_var, e_m, e_v = adam_expected(mid[index[v.name]], mid[index[mn]], mid[index[vn]], g, lr,
                                                        b1p, b2p)
                        for lab, actual, exp, bef in (('var', post[index[v.name]], e_var, mid[index[v.name]]),
                                                      ('m', post[index[mn]], e_m, mid[index[mn]]),
                                                      ('v', post[index[vn]], e_v, mid[index[vn]])):
                            ok_, _, ratio = within(actual, exp, bef)
                            adam_ok = adam_ok and ok_
                            worst[lab] = max(worst[lab], ratio)
                    beta_ok = (np.float32(post[index[b1n]]) == np.float32(b1p) * np.float32(BETA1) and
                               np.float32(post[index[b2n]]) == np.float32(b2p) * np.float32(BETA2))
                    ph, it = int(post[index[m.phase.name]]), int(post[index[m.iteration.name]])
                    exp_ph = s if s < 3 else 0
                    exp_it = int(pre[index[m.iteration.name]]) + (1 if s == 3 else 0)
                    sem.append({'restore_exact': ok, 'calls': calls,
                                'scope_ok': all(set(c['changed']) <= set(c['allowed']) for c in calls),
                                'adam_ok': adam_ok, 'adam_worst_error_over_tolerance': worst, 'beta_powers_ok': bool(beta_ok),
                                'counters': [ph, it], 'counters_ok': (ph, it) == (exp_ph, exp_it),
                                'finite': finite(post) and all(np.isfinite(x).all() for x in grads) and
                                all(np.isfinite(x) for x in out[2:]),
                                'losses': [float(x) for x in out[2:]], 'grads': grads, 'post': post})
                ref = prod[0]['post']
                diff_vars = sorted(set().union(*[changed(ref, p['post']) for p in prod[1:]]) |
                                   set().union(*[changed(ref, x['post']) for x in sem]))
                upd = {}
                for n in diff_vars:
                    i = index[n]
                    d0 = np.asarray(ref[i], np.float64) - np.asarray(pre[i], np.float64)
                    worst_l2 = max(float(np.linalg.norm(np.asarray(p['post'][i], np.float64) - ref[i]))
                                   for p in prod[1:] + sem)
                    upd[n] = {'update_l2': float(np.linalg.norm(d0)), 'replay_diff_l2_max': worst_l2}
                gdiff = []
                for (g, v), a, b in zip(pairs, sem[0]['grads'], sem[1]['grads']):
                    c = compare(a, b)
                    if c is not None:
                        gdiff.append(dict(c, variable=v.name))
                ga = np.concatenate([np.asarray(x, np.float64).ravel() for x in sem[0]['grads']])
                gb = np.concatenate([np.asarray(x, np.float64).ravel() for x in sem[1]['grads']])
                steps['step%d' % s] = {
                    'production_restore_exact': all(p['restore_exact'] for p in prod),
                    'semantic_restore_exact': all(x['restore_exact'] for x in sem),
                    'production_post_sha256': [state_sha(p['post']) for p in prod],
                    'semantic_post_sha256': [state_sha(x['post']) for x in sem],
                    'production_losses': [p['losses'] for p in prod], 'semantic_losses': [x['losses'] for x in sem],
                    'variables_differing_across_replays': upd,
                    'semantic': [{k: x[k] for k in ('calls', 'scope_ok', 'adam_ok', 'adam_worst_error_over_tolerance',
                                                     'beta_powers_ok', 'counters', 'counters_ok', 'finite')}
                                 for x in sem],
                    'gradient_repeat': {'differing_variables': gdiff, 'l2_ref': float(np.linalg.norm(ga)),
                                        'l2_diff': float(np.linalg.norm(gb - ga)),
                                        'per_variable_l2_ref': {v.name: float(np.linalg.norm(np.asarray(a, np.float64)))
                                                                for (_, v), a in zip(pairs, sem[0]['grads'])}},
                    'lr': float(lr)}
                pio.require(steps['step%d' % s]['production_restore_exact'], 'state restore is exact')
                cur = prod[0]['post']
                pio.require(restore(sess, cur), 'trajectory continues from the first production replay')
            rec['steps'] = steps
            rec['state_finite'] = finite(cur)
            rec['state_sha256_after'] = state_sha(cur)
            rec['wall_seconds'] = time.monotonic() - t0
            jl.write(json.dumps(rec, sort_keys=True) + '\n')
            jl.flush()
    jl.close()
    summary['graph_operations_before_session'] = n_ops
    summary['graph_operations_after_session'] = len(graph.get_operations())
    summary['seconds'] = time.monotonic() - t0
    summary['firewall'] = {'denied': fw.denied, 'faces_read': len(fw.faces), 'face_opens': sum(fw.faces.values()),
                           'runtime_writes': [w for w in fw.writes if w.startswith(str(RUNTIME))]}
    summary['peak_cpu_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out_dir / 'summary.json').write_text(json.dumps(summary, sort_keys=True) + '\n')
    print(json.dumps({'process': args.process, 'status': 'DONE', 'seconds': summary['seconds'], 'denied': fw.denied}))
    return 0 if not fw.denied else 1


if __name__ == '__main__':
    sys.exit(main())
