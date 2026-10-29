"""E04 D11 targeted CPU gradient-placement probe (diagnostic only; qualification seed 60904; never a scientific run).

Runs tools/e04_d10_numerical_probe.main UNCHANGED (same design file outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json,
Controller update sequence, Firewall and outputs under /tmp) for one process of one arm of
outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json, through four hooks installed before it runs:
  * production_runner.prepare: after the unchanged graph is built and before any session exists, records the placement
    of the ScatterNd gradients of the CPU-requested GatherNd lookups and, for --arm cpu_scatternd, sets exactly those
    ops' requested device to /device:CPU:0 and removes only the colocation references to them from other ops' _class
    lists (those ops keep their own requested device); no source file changes; the original arm changes nothing;
  * bn_cancelled_biases: the corrected detector of tools/e04_d11_bias_detector.py (must equal the BN-sibling set);
  * trace_devices: the unchanged D10 trace, then a second FULL_TRACE of the same fetches with partition graphs;
  * Controller.counters: records the peak GPU allocator bytes (MaxBytesInUse).
Writes <output-dir>/d11_extra.json (and <output-dir>/trace/*.pb) next to the D10 outputs.

    python -B tools/e04_d11_placement_probe.py --repo <checkout> --arm {original,cpu_scatternd} --process P1
        --output-dir /tmp/gpat_e04_d11/<stage>/<arm>/P1 [--iterations N]
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
ARMS = ('original', 'cpu_scatternd')
CPU = '/device:CPU:0'


def scatter_targets(graph):
    """ScatterNd ops created by tf.gradients as the gradient of a GatherNd (scatter indices = its indices)."""
    by_indices = {}
    for o in graph.get_operations():
        if o.type == 'GatherNd':
            by_indices.setdefault(o.inputs[1].name, []).append(o)
    out, other = [], []
    for o in graph.get_operations():
        if o.type != 'ScatterNd':
            continue
        fwd = by_indices.get(o.inputs[0].name, [])
        g = fwd[0] if len(fwd) == 1 else None
        if g is None or not o.name.endswith('/%s_grad/ScatterNd' % g.name):
            other.append(o.name)
            continue
        out.append({'op': o.name, 'requested_device': o.device, 'forward_gather': g.name,
                    'forward_requested_device': g.device,
                    'forward_params': [g.inputs[0].name, g.inputs[0].dtype.name, g.inputs[0].shape.as_list()],
                    'forward_indices': [g.inputs[1].name, g.inputs[1].dtype.name, g.inputs[1].shape.as_list()],
                    'inputs': [[t.name, t.op.type, t.op.device, t.dtype.name,
                                t.shape.as_list() if t.shape.dims is not None else None] for t in o.inputs],
                    'output': [o.outputs[0].dtype.name, o.outputs[0].shape.as_list()],
                    'consumers': sorted([c.name, c.type, c.device] for c in o.outputs[0].consumers()),
                    'control_inputs': [c.name for c in o.control_inputs]})
    return out, other


def node_defs(graph):
    return {n.name: n for n in graph.as_graph_def().node}


def node_diff(before, after):
    changed = []
    for name in sorted(set(before) & set(after)):
        a, b = before[name], after[name]
        if a.SerializeToString(deterministic=True) == b.SerializeToString(deterministic=True):
            continue
        fields = [f for f, x, y in (('op', a.op, b.op), ('input', list(a.input), list(b.input)),
                                    ('device', a.device, b.device)) if x != y]
        fields += ['attr:' + k for k in sorted(set(a.attr) | set(b.attr)) if k not in a.attr or k not in b.attr or
                   a.attr[k].SerializeToString(deterministic=True) != b.attr[k].SerializeToString(deterministic=True)]
        changed.append({'node': name, 'device_before': a.device, 'device_after': b.device, 'fields': fields,
                        'only_device_differs': fields == ['device']})
    return {'changed': changed, 'added': sorted(set(after) - set(before)), 'removed': sorted(set(before) - set(after))}


def colocation_references(graph, names):
    """Ops whose '_class' colocation list names a target (TF1 gradient aggregation colocates each per-device AddN with
    its first summand, which ties that ScatterNd to the GPU-requested AddN and lets soft placement override CPU)."""
    out = []
    for o in graph.get_operations():
        if '_class' not in o.node_def.attr:
            continue
        cls = [c.decode() for c in o.node_def.attr['_class'].list.s]
        keep = [c for c in cls if not (c.startswith('loc:@') and c[5:] in names)]
        if keep != cls:
            out.append({'op': o.name, 'type': o.type, 'requested_device': o.device, 'class_before': cls,
                        'class_after': keep, 'inputs': [t.op.name for t in o.inputs]})
    return out


def downstream(ops):
    seen, stack = set(), list(ops)
    while stack:
        o = stack.pop()
        for t in o.outputs:
            for c in t.consumers():
                if c.name not in seen:
                    seen.add(c.name)
                    stack.append(c)
    return seen


def parse_trace(tf, meta):
    """The D10 trace_devices parsing, plus the runtime-inserted nodes (names, devices)."""
    ops = {o.name for o in tf.get_default_graph().get_operations()}
    place, order, runtime = {}, [], []
    for d in meta.step_stats.dev_stats:
        if '/stream:' in d.device or 'memcpy' in d.device.lower():
            continue
        for n in d.node_stats:
            name = n.node_name.split(':')[0]
            if name not in ops:
                runtime.append([n.node_name, d.device])
                continue
            place[name] = d.device
            order.append((n.all_start_micros, name))
    return place, [n for _, n in sorted(order)], runtime


def partition_summary(tf, meta, sha):
    parts = []
    for gd in meta.partition_graphs:
        nodes = []
        for n in gd.node:
            rec = {'name': n.name, 'op': n.op, 'device': n.device, 'inputs': list(n.input),
                   'attrs': {k: hashlib.sha256(n.attr[k].SerializeToString(deterministic=True)).hexdigest()[:20]
                             for k in sorted(n.attr)}}
            if n.op == 'Const':
                try:
                    v = tf.make_ndarray(n.attr['value'].tensor)
                    rec['value'] = [sha(v), str(v.dtype), list(v.shape)]
                except Exception as e:                       # pragma: no cover - recorded, never silently skipped
                    rec['value'] = ['unreadable: %s' % type(e).__name__]
            if n.op in ('_Send', '_Recv', '_HostSend', '_HostRecv'):
                rec['transfer'] = {k: (n.attr[k].s.decode() if n.attr[k].s else n.attr[k].i)
                                   for k in ('tensor_name', 'send_device', 'recv_device') if k in n.attr}
            nodes.append(rec)
        parts.append({'devices': sorted({n.device for n in gd.node}), 'nodes': nodes})
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--process', required=True, choices=['P1', 'P2', 'P3'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--iterations', type=int, help='short probe (D10 --smoke path with this many iterations)')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_dir = Path(os.path.abspath(args.output_dir))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'tools'))
    import e04_d10_numerical_probe as base
    import e04_d11_bias_detector as bias
    from methods.physics_std import production_runner as pr
    from methods.physics_std import production_io as pio

    extra = {'arm': args.arm, 'process': args.process, 'iterations_requested': args.iterations,
             'protocol_sha256': hashlib.sha256((repo / 'outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json')
                                               .read_bytes()).hexdigest(),
             'design_sha256': hashlib.sha256((repo / base.PROTOCOL).read_bytes()).hexdigest(),
             'tool_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in (
                 'tools/e04_d11_placement_probe.py', 'tools/e04_d11_bias_detector.py',
                 'tools/e04_d10_numerical_probe.py')},
             'gpu_memory_bytes': [], 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    state = {}
    original_prepare, original_trace = pr.prepare, base.trace_devices

    def prepare(tf, *a, **k):
        res = original_prepare(tf, *a, **k)
        ctl, graph = res[0], tf.get_default_graph()
        m = ctl.model
        before, sig_before = node_defs(graph), pr.construction_signature(graph)
        targets, other = scatter_targets(graph)
        pio.require(not other, 'ScatterNd ops that are not GatherNd gradients: %s' % other)
        pio.require(targets and all(t['forward_requested_device'].upper().endswith('CPU:0') for t in targets),
                    'every target is the gradient of a CPU-requested GatherNd')
        refs = colocation_references(graph, {t['op'] for t in targets})
        if args.arm == 'cpu_scatternd':
            from tensorflow.core.framework import attr_value_pb2
            for t in targets:
                graph.get_operation_by_name(t['op'])._set_device(CPU)
            for r in refs:                               # keep the op's own device; drop only the target reference
                op = graph.get_operation_by_name(r['op'])
                keep = r['class_after'] or ['loc:@' + op.name]
                op._set_attr('_class', attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(s=[c.encode() for c in keep])))
        after = node_defs(graph)
        sets = dict([(key, [g for g, _ in m.per_loss_gradients[key] if g is not None]) for key in base.LOSSES] +
                    [('step%d' % s, [g for g, _ in m.step_gradients[s] if g is not None]) for s in (1, 2, 3)])
        tops = [graph.get_operation_by_name(t['op']) for t in targets]
        down = downstream(tops)
        extra['placement'] = {
            'targets': targets, 'target_names': sorted(t['op'] for t in targets),
            'requested_device_after': {t['op']: graph.get_operation_by_name(t['op']).device for t in targets},
            'node_diff': node_diff(before, after), 'colocation_references': refs,
            'released_colocation': sorted(r['op'] for r in refs) if args.arm == 'cpu_scatternd' else [],
            'construction_before': list(ctl.construction),
            'graph_signature_at_override': {'before': list(sig_before),
                                            'after': list(pr.construction_signature(graph))},
            'probe_sets_depending_on_targets': sorted(key for key, ts in sets.items()
                                                      if base.closure(ts) & {o.name for o in tops}),
            'downstream_ops': len(down),
            'downstream_by_type': {ty: sum(graph.get_operation_by_name(n).type == ty for n in down)
                                   for ty in sorted({graph.get_operation_by_name(n).type for n in down})},
            'resize_bilinear_grad_requested': sorted([o.name, o.device] for o in graph.get_operations()
                                                     if o.type == 'ResizeBilinearGrad')}
        extra['gradients_present'] = dict(
            [(key, [g is not None for g, _ in m.per_loss_gradients[key]]) for key in base.LOSSES] +
            [('step%d' % s, [g is not None for g, _ in m.step_gradients[s]]) for s in (1, 2, 3)])
        counters = ctl.counters

        def recorded(sess):
            r = counters(sess)
            extra['gpu_memory_bytes'].append(r['gpu_memory_bytes'])
            return r
        ctl.counters = recorded
        return res

    def cancelled(tf, graph, trainables, losses):
        found = sorted(bias.bn_cancelled_biases(tf, graph, trainables, losses))
        sibling = bias.bn_sibling_biases([v.name for v in trainables])
        pio.require(found == sibling and len(found) == 31, 'corrected BN-cancelled-bias detector: %d' % len(found))
        extra['bn_cancelled_biases'] = found
        return found

    def trace(tf, sess, fd, targets):
        r1 = original_trace(tf, sess, fd, targets)
        meta = tf.RunMetadata()
        sess.run(targets, fd, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True),
                 run_metadata=meta)
        tdir = out_dir / 'trace'
        tdir.mkdir()
        for i, gd in enumerate(meta.partition_graphs):
            (tdir / ('partition_%02d.pb' % i)).write_bytes(gd.SerializeToString(deterministic=True))
        (tdir / 'step_stats_second_trace.pb').write_bytes(meta.step_stats.SerializeToString(deterministic=True))
        place2, order2, runtime2 = parse_trace(tf, meta)
        graph = tf.get_default_graph()
        names = extra['placement']['target_names']
        extra['trace'] = {
            'released_colocation_executed_device': {r['op']: [r1[0].get(r['op']), place2.get(r['op'])]
                                                    for r in extra['placement']['colocation_references']},
            'first_trace_place': r1[0], 'second_trace_place': place2,
            'second_trace_runtime_nodes': runtime2, 'first_trace_runtime_nodes': r1[2],
            'executed_set_equal_between_traces': sorted(r1[0]) == sorted(place2),
            'device_equal_between_traces': all(r1[0][n] == place2.get(n) for n in r1[0]),
            'target_executed_device': {n: [r1[0].get(n), place2.get(n)] for n in names},
            'partitions': partition_summary(tf, meta, base.sha),
            'requested_device': {n: graph.get_operation_by_name(n).device for n in r1[0]}}
        return r1

    pr.prepare, base.bn_cancelled_biases, base.trace_devices = prepare, cancelled, trace
    argv = ['e04_d10_numerical_probe.py', '--repo', str(repo), '--process', args.process, '--output-dir', str(out_dir)]
    if args.iterations is not None:
        base.SMOKE_ITERATIONS = args.iterations
        argv.append('--smoke')
    sys.argv = argv
    rc = base.main()
    extra['peak_cpu_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    extra['peak_gpu_memory_bytes'] = max([b for b in extra['gpu_memory_bytes'] if b is not None], default=None)
    extra['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    extra['rc'] = rc
    (out_dir / 'd11_extra.json').write_text(json.dumps(extra, sort_keys=True) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
