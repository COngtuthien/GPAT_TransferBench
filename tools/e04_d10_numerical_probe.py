"""E04 D10 bounded numerical-reproducibility probe (diagnostic only; qualification seed 60904; never a scientific run).

Executes outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json for one process: builds the production Controller exactly as
tools/run_e04.py would (prepare(): contract, D9 environment check, feed pack, unchanged TrainingGraph), opens NO
RunContext, writes no checkpoint and nothing outside --output-dir (under /tmp), and runs the frozen update sequence
(Controller.iteration) for the protocol's iterations on real TRAIN data, recording:
  * iteration 0: FULL_TRACE executed devices and the first-divergence localization of every gradient tensor;
  * every iteration: in-run losses, two in-process fetches of the step-1 and L_G gradients at (S_t, X_t), state
    unchanged by those fetches, counters, beta powers, session-call log, finite flags, group norms and update norms;
  * gradient-probe iterations: 3 repeats of every per-loss and step gradient, plus the minibatch reference on X_{t+1};
  * snapshots of every global variable at the protocol's snapshot iterations.
The same Firewall as the production qualification refuses non-pack faces, manifests, runs/ and runtime writes.

    python -B tools/e04_d10_numerical_probe.py --repo <checkout> --process P1 --output-dir /tmp/gpat_e04_d10/P1 [--smoke]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
PROTOCOL = 'outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json'
SMOKE_ITERATIONS = 3
FRONTIER_CHUNK_BYTES = 1500 * 2 ** 20
LOSSES = ('L_depth', 'L_G', 'L_P', 'L_R', 'L_S', 'L_H', 'L_D')
NUMERIC = ('float16', 'float32', 'float64', 'int32', 'int64', 'bool', 'uint8', 'int8')


def sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ordered(a):
    """float32 bit patterns mapped to a monotone integer line (ULP distance = absolute difference)."""
    import numpy as np
    i = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def pair_stats(ref, other):
    """Per-variable and concatenated comparison of two gradient lists (float64 arithmetic)."""
    import numpy as np
    per, tot = [], {'l2_ref': 0., 'l2_other': 0., 'l2_diff': 0., 'dot': 0., 'max_abs': 0., 'identical': 0}
    for a, b in zip(ref, other):
        a64, b64 = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
        d = b64 - a64
        na, nb, nd = float(np.dot(a64, a64)), float(np.dot(b64, b64)), float(np.dot(d, d))
        dot, mx, amax = float(np.dot(a64, b64)), float(np.abs(d).max()), float(np.abs(a64).max())
        same = np.asarray(a).tobytes() == np.asarray(b).tobytes()
        per.append({'identical': same, 'l2_ref': na ** .5, 'l2_other': nb ** .5, 'l2_diff': nd ** .5, 'max_abs': mx,
                    'ref_abs_max': amax, 'max_ulp': int(np.abs(ordered(a) - ordered(b)).max()) if not same else 0,
                    'cosine': dot / (na * nb) ** .5 if na > 0 and nb > 0 else None,
                    'finite': bool(np.isfinite(a64).all() and np.isfinite(b64).all())})
        tot['l2_ref'] += na
        tot['l2_other'] += nb
        tot['l2_diff'] += nd
        tot['dot'] += dot
        tot['max_abs'] = max(tot['max_abs'], mx)
        tot['identical'] += same
    c = {'variables': len(per), 'identical_variables': tot['identical'], 'max_abs': tot['max_abs'],
         'l2_ref': tot['l2_ref'] ** .5, 'l2_other': tot['l2_other'] ** .5, 'l2_diff': tot['l2_diff'] ** .5,
         'cosine': tot['dot'] / (tot['l2_ref'] * tot['l2_other']) ** .5 if tot['l2_ref'] > 0 and tot['l2_other'] > 0
         else None, 'finite': all(p['finite'] for p in per)}
    return per, c


def names(fetches):
    if isinstance(fetches, (list, tuple)):
        return [names(f) for f in fetches]
    return fetches.name


class Recorder:
    """Session proxy that logs every call's fetch names and feed keys (graph-execution sequence)."""

    def __init__(self, sess):
        self.sess, self.log = sess, []

    def run(self, fetches, feed_dict=None, **kw):
        self.log.append([names(fetches), sorted(k.name for k in (feed_dict or {}))])
        return self.sess.run(fetches, feed_dict, **kw)


def bn_cancelled_biases(tf, graph, trainables, losses):
    """Biases whose every consumer inside the loss computation (ancestors of the loss tensors) is a BiasAdd feeding
    only training-mode FusedBatchNorm ops (Saver / optimizer / gradient consumers are outside the forward closure)."""
    forward, out = closure(losses), []
    for v in trainables:
        if not v.op.name.endswith('/bias'):
            continue
        reads = [c for c in v.value().consumers() if c.name in forward]
        ok = bool(reads) and all(c.type == 'BiasAdd' for c in reads)
        for c in reads:
            fwd = [u for u in c.outputs[0].consumers() if u.name in forward]
            ok = ok and bool(fwd) and all(u.type.startswith('FusedBatchNorm') and u.get_attr('is_training')
                                          for u in fwd)
        if ok:
            out.append(v.name)
    return out


def structural_signatures(graph):
    """Recursive structural signature per op (type, attributes, requested device, input and control signatures; the
    name only for stateful ops). TF's classic-optimizer common-subexpression elimination keeps an arbitrary member of
    each class of identical ops, so executed op NAMES may differ across processes while signatures do not."""
    memo = {}
    for op in graph.get_operations():                    # creation order is topological (no control flow)
        attrs = sorted((k, op.node_def.attr[k].SerializeToString(deterministic=True).hex())
                       for k in op.node_def.attr if k != '_class')
        ins = [[memo[t.op.name], t.value_index] for t in op.inputs]
        ctrl = sorted(memo[c.name] for c in op.control_inputs)
        memo[op.name] = hashlib.sha256(json.dumps([op.type, attrs, op.device, ins, ctrl,
                                                   op.name if op.op_def.is_stateful else None]).encode()).hexdigest()
    return memo


def closure(tensors):
    seen, stack = set(), [t.op for t in tensors]
    while stack:
        o = stack.pop()
        if o.name in seen:
            continue
        seen.add(o.name)
        stack.extend(i.op for i in o.inputs)
    return seen


def tensor_bytes(t):
    import numpy as np
    dims = [d if d is not None else 1 for d in (t.shape.as_list() if t.shape.dims is not None else [1])]
    return int(np.prod(dims)) * max(t.dtype.base_dtype.size, 1)


def localize(tf, sess, fd, targets, repeats):
    """First-divergence localization over the ancestor closure of `targets` (iteration-0 state, no mutation)."""
    graph = tf.get_default_graph()
    keep = closure(targets)
    ops = [o for o in graph.get_operations() if o.name in keep]
    ok = lambda t: graph.is_fetchable(t) and t.dtype.base_dtype.name in NUMERIC  # noqa: E731
    chunks, cur, cur_bytes, owner = [], {}, 0, {}
    for o in ops:
        items = [t for t in list(o.outputs) + list(o.inputs) if ok(t)]
        if not [t for t in o.outputs if ok(t)]:
            continue
        add = sum(tensor_bytes(t) for t in items if t.name not in cur)
        if cur and cur_bytes + add > FRONTIER_CHUNK_BYTES:
            chunks.append(cur)
            cur, cur_bytes = {}, 0
        for t in items:
            if t.name not in cur:
                cur[t.name] = t
                cur_bytes += tensor_bytes(t)
        owner[o.name] = len(chunks)
    if cur:
        chunks.append(cur)
    varies = []                                   # per chunk: tensor name -> differs across repeats
    for chunk in chunks:
        keys = sorted(chunk)
        digests = []
        for _ in range(repeats):
            vals = sess.run([chunk[k] for k in keys], fd)
            digests.append([sha(v) for v in vals])
        varies.append({k: len({d[i] for d in digests}) > 1 for i, k in enumerate(keys)})
    varying_tensors, origins, undetermined = set(), [], []
    for i, ch in enumerate(varies):
        varying_tensors |= {k for k, v in ch.items() if v}
    for o in ops:
        if o.name not in owner:
            continue
        ch = varies[owner[o.name]]
        if not any(ch.get(t.name) for t in o.outputs):
            continue
        hidden = [t.name for t in o.inputs if not ok(t)]
        if not any(ch.get(t.name) for t in o.inputs):
            (undetermined if hidden else origins).append({'op': o.name, 'type': o.type, 'requested_device': o.device,
                                                          'unobservable_inputs': hidden})
    by_type = {}
    for k in varying_tensors:
        t = graph.get_tensor_by_name(k).op.type
        by_type[t] = by_type.get(t, 0) + 1
    return {'closure_ops': len(ops), 'chunks': len(chunks), 'repeats': repeats,
            'fetched_tensors': len(set().union(*[set(c) for c in chunks])) if chunks else 0,
            'varying_tensors': len(varying_tensors), 'varying_tensors_by_producer_type': by_type,
            'origins': origins, 'undetermined_origins': undetermined}


def trace_devices(tf, sess, fd, targets):
    """Executed device of every graph operation (node_stats names may carry a ':<OpType>' suffix; runtime-inserted
    _Send/_Recv transfer nodes are not graph operations and are only counted)."""
    meta = tf.RunMetadata()
    sess.run(targets, fd, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=meta)
    ops = {o.name for o in tf.get_default_graph().get_operations()}
    place, order, runtime = {}, [], 0
    for d in meta.step_stats.dev_stats:
        if '/stream:' in d.device or 'memcpy' in d.device.lower():
            continue
        for n in d.node_stats:
            name = n.node_name.split(':')[0]
            if name not in ops:
                runtime += 1
                continue
            place[name] = d.device
            order.append((n.all_start_micros, name))
    order = [n for _, n in sorted(order)]
    return place, order, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--process', required=True, choices=['P1', 'P2', 'P3'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out_dir = Path(os.path.abspath(args.output_dir))
    if not str(out_dir).startswith('/tmp/'):
        raise SystemExit('output must be under /tmp')
    out_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    protocol = json.loads((repo / PROTOCOL).read_text())
    design = protocol['design']
    n_iter = SMOKE_ITERATIONS if args.smoke else design['iterations']
    gset = [t for t in design['gradient_probe_iterations'] if t < n_iter]
    sset = [t for t in design['snapshot_iterations'] if t <= n_iter]
    reps = design['gradient_probe_repeats']

    import numpy as np
    from methods.physics_std import production_io as pio
    errors = pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
    if errors:
        raise SystemExit('launch environment: %s' % errors)
    import tensorflow as tf
    from methods.physics_std import production_runner as pr
    from methods.physics_std.production_qualification import Firewall, RUNTIME, FACES
    ctl, ids, live, driver, env, reasons = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, RUNTIME, FACES)
    m, graph = ctl.model, tf.get_default_graph()
    fw = Firewall()
    fw.pack_ids = set(ctl.pack.samples)
    lookahead = pr.Feed(pio.QUALIFICATION_SEED, ctl.pack, pr.FaceReader(FACES, ctl.pack))
    la_next = 0

    # ----------------------------------------------------------------- fetch groups and state classification
    per_loss = {k: [(g, v) for g, v in m.per_loss_gradients[k] if g is not None] for k in LOSSES}
    steps = {s: [(g, v) for g, v in m.step_gradients[s] if g is not None] for s in (1, 2, 3)}
    probe_sets = dict([(k, [g for g, _ in per_loss[k]]) for k in LOSSES] +
                      [('step%d' % s, [g for g, _ in steps[s]]) for s in (1, 2, 3)])
    probe_vars = dict([(k, [v.name for _, v in per_loss[k]]) for k in LOSSES] +
                      [('step%d' % s, [v.name for _, v in steps[s]]) for s in (1, 2, 3)])
    variables = ctl.variables
    vnames = [v.name for v in variables]
    cancelled = set(bn_cancelled_biases(tf, graph, ctl.trainables, [m.losses[k] for k in sorted(m.losses)]))
    gset_names, dset_names = {v.name for v in m.G}, {v.name for v in m.D}
    slots = {}
    for opt, group, tag in ((m.opt_G, m.G, 'G'), (m.opt_D, m.D, 'D')):
        for v in group:
            for s in ('m', 'v'):
                slots[opt.get_slot(v, s).name] = 'adam_%s_%s' % (s, tag)
        for b in opt._get_beta_accumulators():
            slots[b.name] = 'beta_power_%s' % tag
    groups = {}
    for n in vnames:
        if n in gset_names:
            g = 'G_cancelled_bias' if n in cancelled else 'G_functional'
        elif n in dset_names:
            g = 'D_cancelled_bias' if n in cancelled else 'D_functional'
        elif n in slots:
            g = slots[n]
        elif 'moving_mean' in n or 'moving_variance' in n:
            g = 'bn_moving_' + ('mean' if 'moving_mean' in n else 'variance')
        elif n in ('global_iteration:0', 'minibatch_phase:0'):
            g = 'counters'
        else:
            raise SystemExit('unclassified variable %s' % n)
        groups[n] = g
    beta_names = sorted(n for n, g in groups.items() if g.startswith('beta_power'))
    all_grads = [g for k in sorted(probe_sets) for g in probe_sets[k]]

    summary = {'protocol_sha256': hashlib.sha256((repo / PROTOCOL).read_bytes()).hexdigest(), 'process': args.process,
               'smoke': args.smoke, 'iterations': n_iter, 'gradient_probe_iterations': gset,
               'snapshot_iterations': sset, 'qualification_seed': pio.QUALIFICATION_SEED, 'identities': ids,
               'construction': list(ctl.construction), 'structure': ctl.structure, 'driver': driver,
               'environment_live': {k: live[k] for k in ('python', 'tensorflow', 'native_runtime', 'gpu',
                                                         'precision_env', 'session_config')},
               'variables': vnames, 'groups': groups, 'bn_cancelled_biases': sorted(cancelled),
               'probe_variables': probe_vars, 'pid': os.getpid(), 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                                             time.gmtime())}
    sys.addaudithook(fw)
    t_start = time.monotonic()

    def state(sess):
        return sess.run(variables)

    def norms(vals):
        out = {}
        for n, x in zip(vnames, vals):
            g = groups[n]
            out[g] = out.get(g, 0.) + float(np.dot(np.asarray(x, np.float64).ravel(), np.asarray(x, np.float64).ravel()))
        return {k: v ** .5 for k, v in out.items()}

    def update_norms(a, b):
        out = {}
        for n, x, y in zip(vnames, a, b):
            d = np.asarray(y, np.float64).ravel() - np.asarray(x, np.float64).ravel()
            out[groups[n]] = out.get(groups[n], 0.) + float(np.dot(d, d))
        return {k: v ** .5 for k, v in out.items()}

    def finite(vals):
        return all(np.isfinite(x).all() for x in vals if np.issubdtype(np.asarray(x).dtype, np.floating))

    def snapshot(t, vals):
        np.savez(str(out_dir / ('snap_%04d.npz' % t)), **{'v%03d' % i: x for i, x in enumerate(vals)})

    jl = (out_dir / 'iterations.jsonl').open('w')
    with tf.Session(config=pr.session_config(tf)) as sess:
        sess.run(tf.global_variables_initializer())
        cur = state(sess)
        summary['initial_state_sha256'] = hashlib.sha256(''.join(sha(x) for x in cur).encode()).hexdigest()
        snapshot(0, cur)
        for t in range(n_iter):
            arrays, info = ctl.feed.next()
            fd = ctl.feed_dict(arrays)
            before = ctl.counters(sess)
            exp = pio.expected_learning_rates(t)
            pio.require(abs(before['lr_G'] - exp[0]) <= 1e-6 * exp[0] and abs(before['lr_D'] - exp[1]) <= 1e-6 * exp[1],
                        'learning rate schedule at iteration %d' % t)
            rec = {'iteration': t, 'pair_rows': info['pair_rows'], 'pair_passes': info['pair_passes'],
                   'hard_scales': info['hard_scales'], 'hard_keep': info['hard_keep'],
                   'input_sha256': info['input_sha256'],
                   'counters_before': {k: before[k] for k in ('iteration', 'phase', 'lr_G', 'lr_D')}}
            if t == 0:
                place, order, runtime_nodes = trace_devices(tf, sess, fd, all_grads)
                sigs = structural_signatures(graph)
                watch = {}
                for o in graph.get_operations():
                    if o.name in place and o.type in ('ResizeBilinearGrad', 'ScatterNd', 'GatherNd', 'ResizeBilinear'):
                        watch[o.name] = {'type': o.type, 'requested_device': o.device, 'executed_device': place[o.name]}
                summary['trace'] = {'executed_nodes': len(place), 'runtime_transfer_nodes': runtime_nodes,
                                    'executed_node_set_sha256': hashlib.sha256('\n'.join(sorted(place)).encode()).hexdigest(),
                                    'executed_signature_set_sha256': hashlib.sha256('\n'.join(sorted(
                                        {sigs[n] for n in place})).encode()).hexdigest(),
                                    'executed_distinct_signatures': len({sigs[n] for n in place}),
                                    'executed_order_sha256': hashlib.sha256('\n'.join(order).encode()).hexdigest(),
                                    'watched_ops': watch, 'watched_type_ops_in_closure': sorted(
                                        o for o in closure(all_grads) if graph.get_operation_by_name(o).type in
                                        ('ResizeBilinearGrad', 'ScatterNd', 'GatherNd', 'ResizeBilinear')),
                                    'executed_device_by_op_type': {}}
                for name, dev in place.items():
                    k = '%s|%s' % (graph.get_operation_by_name(name).type, 'GPU' if 'GPU' in dev.upper() else 'CPU')
                    summary['trace']['executed_device_by_op_type'][k] = \
                        summary['trace']['executed_device_by_op_type'].get(k, 0) + 1
                loc = localize(tf, sess, fd, all_grads, reps)
                for o in loc['origins'] + loc['undetermined_origins']:
                    o['executed_device'] = place.get(o['op'])
                summary['localization'] = loc
                summary['iteration0_diagnostics_state_unchanged'] = all(
                    np.array_equal(x, y) for x, y in zip(cur, state(sess)))
                pio.require(summary['iteration0_diagnostics_state_unchanged'], 'iteration-0 diagnostics changed state')
            if t in gset:
                probe = {'repeats': {}, 'minibatch_reference': {}}
                fetched = {k: [sess.run(probe_sets[k], fd) for _ in range(reps)] for k in sorted(probe_sets)}
                for k, runs in fetched.items():
                    per = [pair_stats(runs[0], r) for r in runs[1:]]
                    probe['repeats'][k] = {'per_variable_sha256_r0': [sha(x) for x in runs[0]],
                                           'distinct_per_variable': [len({sha(r[i]) for r in runs})
                                                                     for i in range(len(runs[0]))],
                                           'concat': [c for _, c in per], 'per_variable': [p for p, _ in per]}
                while la_next < t + 1:
                    lookahead.pairs.next_rows()
                    lookahead.hard.next()
                    la_next += 1
                la_arrays, la_info = lookahead.next()           # X_{t+1}, never the controller feed
                la_next += 1
                probe['lookahead_info'] = {k: la_info[k] for k in ('pair_rows', 'pair_passes', 'hard_scales',
                                                                     'hard_keep', 'input_sha256')}
                la_fd = ctl.feed_dict(la_arrays)
                for k in ('L_G', 'step1'):
                    other = sess.run(probe_sets[k], la_fd)
                    per, c = pair_stats(fetched[k][0], other)
                    probe['minibatch_reference'][k] = {'concat': c, 'per_variable_l2_diff': [p['l2_diff'] for p in per]}
                np.savez(str(out_dir / ('grad_%04d.npz' % t)),
                         **dict([('L_G_%03d' % i, x) for i, x in enumerate(fetched['L_G'][0])] +
                                [('step1_%03d' % i, x) for i, x in enumerate(fetched['step1'][0])]))
                rec['gradient_probe'] = probe
            g0 = sess.run([probe_sets['step1'], probe_sets['L_G']], fd)
            g1 = sess.run([probe_sets['step1'], probe_sets['L_G']], fd)
            mid = state(sess)
            rec['diagnostic_fetch_state_unchanged'] = all(np.array_equal(x, y) for x, y in zip(cur, mid))
            pio.require(rec['diagnostic_fetch_state_unchanged'], 'diagnostic fetch changed state at %d' % t)
            for i, k in enumerate(('step1', 'L_G')):
                per, c = pair_stats(g0[i], g1[i])
                rec['repeat_%s' % k] = {'concat': c, 'per_variable': [
                    {f: p[f] for f in ('identical', 'l2_ref', 'l2_diff', 'max_abs', 'ref_abs_max', 'max_ulp', 'cosine')}
                    for p in per]}
            proxy = Recorder(sess)
            ts = time.monotonic()
            rec['losses'] = ctl.iteration(proxy, fd)
            rec['iteration_seconds'] = time.monotonic() - ts
            rec['call_log_sha256'] = hashlib.sha256(json.dumps(proxy.log).encode()).hexdigest()
            if t == 0:
                summary['call_log_iteration0'] = proxy.log
            after = ctl.counters(sess)
            pio.require((after['iteration'], after['phase']) == (t + 1, 0), 'global_iteration +1, phase 0')
            rec['counters_after'] = {k: after[k] for k in ('iteration', 'phase', 'lr_G', 'lr_D')}
            new = state(sess)
            rec['state_finite'] = finite(new)
            rec['beta_powers'] = {n: float(new[vnames.index(n)]) for n in beta_names}
            rec['group_norms'] = norms(new)
            rec['update_norms'] = update_norms(cur, new)
            rec['state_sha256'] = hashlib.sha256(''.join(sha(x) for x in new).encode()).hexdigest()
            rec['losses_finite'] = all(np.isfinite(v) for s in rec['losses'].values() for v in s.values())
            pio.require(rec['state_finite'] and rec['losses_finite'], 'non-finite state or loss at %d' % t)
            if t + 1 in sset:
                snapshot(t + 1, new)
            cur = new
            rec['wall_seconds'] = time.monotonic() - t_start
            jl.write(json.dumps(rec, sort_keys=True) + '\n')
            jl.flush()
    jl.close()
    summary['seconds'] = time.monotonic() - t_start
    summary['firewall'] = {'denied': fw.denied, 'faces_read': len(fw.faces), 'face_opens': sum(fw.faces.values()),
                           'runtime_writes': [w for w in fw.writes if w.startswith(str(RUNTIME))]}
    summary['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out_dir / 'summary.json').write_text(json.dumps(summary, sort_keys=True) + '\n')
    print(json.dumps({'process': args.process, 'status': 'DONE', 'iterations': n_iter, 'seconds': summary['seconds'],
                      'denied': fw.denied}))
    return 0 if not fw.denied else 1


if __name__ == '__main__':
    sys.exit(main())
