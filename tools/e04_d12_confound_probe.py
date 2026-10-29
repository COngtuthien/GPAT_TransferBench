"""E04 D12 stage C: D11 instrumentation-confound probe (diagnostic only; qualification seed 60904; original graph and
device placement unchanged; never a scientific run).

Runs one clean 1-iteration process of the D10 probe (tools/e04_d10_numerical_probe.main, unchanged) under one factor of
outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json:
  * plain        no traced session call before the localization (trace_devices returns requested devices);
  * d10          the unchanged D10 probe (one FULL_TRACE);
  * d10_trace2   the D10 trace plus D11's second FULL_TRACE of the same fetches with output_partition_graphs;
  * d11_wrapper  tools/e04_d11_placement_probe.py --arm original exactly (all D11 hooks).
In every factor the iteration-0 first-divergence localization (unchanged algorithm, chunks, repeats and session calls)
runs through a session proxy that only compares each repeat with the chunk's first repeat and records, per varying
tensor, its name, producer, chunk and magnitude. Writes <output-dir>/d12_extra.json next to the D10 outputs.

    python -B tools/e04_d12_confound_probe.py --repo <checkout> --factor plain --process P1 --output-dir /tmp/...
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
FACTORS = ('plain', 'd10', 'd10_trace2', 'd11_wrapper')
PROTOCOL = 'outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json'
DATA_MOVEMENT = ('ConcatV2', 'Identity', 'Reshape', 'Slice', 'StridedSlice', 'Pack', 'Unpack', 'Transpose', 'GatherNd',
                 'Tile', 'Squeeze', 'ExpandDims', 'Placeholder', 'VariableV2', 'Split', 'SplitV', 'StopGradient')


def compare(a, b):
    """Magnitude of b - a (float64 arithmetic); None when bit-identical."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.tobytes() == b.tobytes():
        return None
    out = {'dtype': a.dtype.name, 'elements': int(a.size), 'differing': int((a != b).sum())}
    if a.dtype.kind == 'f':
        a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        d = b64 - a64
        out.update({'max_abs': float(np.abs(d).max()), 'l2_diff': float(np.dot(d, d)) ** .5,
                    'l2_ref': float(np.dot(a64, a64)) ** .5, 'ref_abs_max': float(np.abs(a64).max()),
                    'finite': bool(np.isfinite(a64).all() and np.isfinite(b64).all())})
        if a.dtype == np.float32:
            i = [np.ascontiguousarray(x, np.float32).view(np.int32).astype(np.int64) for x in (a, b)]
            o = [np.where(x < 0, -(x & 0x7FFFFFFF), x) for x in i]
            out['max_ulp'] = int(np.abs(o[0] - o[1]).max())
    return out


def worse(old, new):
    if old is None:
        return new
    return new if (new.get('l2_diff', 0), new['differing']) > (old.get('l2_diff', 0), old['differing']) else old


class DiffSession:
    """Proxy used only by localize(): forwards every call unchanged; compares each repeat with the chunk's first."""

    def __init__(self, sess):
        self.sess, self.key, self.ref, self.chunks, self.varying = sess, None, None, [], {}

    def run(self, fetches, feed_dict=None, **kw):
        vals = self.sess.run(fetches, feed_dict, **kw)
        key = tuple(t.name for t in fetches)
        if key != self.key:
            self.key, self.ref = key, vals
            self.chunks.append(key)
            return vals
        c = len(self.chunks) - 1
        for n, a, b in zip(key, self.ref, vals):
            r = compare(a, b)
            if r is not None:
                self.varying[(c, n)] = worse(self.varying.get((c, n)), r)
        return vals


def annotate(tf, proxy, forward_ops):
    graph = tf.get_default_graph()
    names = [set(k) for k in proxy.chunks]
    var = {}
    for (c, n) in proxy.varying:
        var.setdefault(c, set()).add(n)
    rows = []
    for (c, n), r in sorted(proxy.varying.items()):
        t = graph.get_tensor_by_name(n)
        op = t.op
        ins = [i.name for i in op.inputs]
        fetched_inputs = all(i in names[c] for i in ins)
        rows.append(dict(r, tensor=n, chunk=c, op=op.name, type=op.type, requested_device=op.device,
                         forward=op.name in forward_ops, gradient=op.name.startswith('gradients'),
                         data_movement=op.type in DATA_MOVEMENT,
                         chunk_local_root=fetched_inputs and not any(i in var.get(c, ()) for i in ins),
                         inputs_fetched_in_chunk=fetched_inputs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--factor', required=True, choices=FACTORS)
    ap.add_argument('--process', required=True, choices=['P1', 'P2', 'P3', 'P4'])
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(os.path.abspath(args.output_dir))
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / 'tools'))
    sys.path.insert(0, str(repo))
    import e04_d10_numerical_probe as base
    tools = ('tools/e04_d12_confound_probe.py', 'tools/e04_d10_numerical_probe.py', 'tools/e04_d11_placement_probe.py',
             'tools/e04_d11_bias_detector.py')
    extra = {'factor': args.factor, 'process': args.process,
             'protocol_sha256': hashlib.sha256((repo / PROTOCOL).read_bytes()).hexdigest(),
             'tool_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in tools},
             'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'traced_calls_before_localization': None}
    original_localize, original_trace = base.localize, base.trace_devices
    holder = {}

    def localize(tf, sess, fd, targets, repeats):
        m = holder['model']
        forward = base.closure([m.losses[k] for k in sorted(m.losses)] + list(m.predictions))
        proxy = DiffSession(sess)
        res = original_localize(tf, proxy, fd, targets, repeats)
        rows = annotate(tf, proxy, forward)
        extra['localization'] = {'chunks': len(proxy.chunks), 'repeats': repeats,
                                 'chunk_fetch_sha256': [hashlib.sha256('\n'.join(k).encode()).hexdigest()
                                                        for k in proxy.chunks],
                                 'varying': rows, 'varying_tensors': len({r['tensor'] for r in rows}),
                                 'reported_by_d10_algorithm': {k: res[k] for k in ('varying_tensors', 'fetched_tensors',
                                                                                   'closure_ops')},
                                 'd10_origins': [o['op'] for o in res['origins']],
                                 'd10_undetermined': [o['op'] for o in res['undetermined_origins']]}
        return res

    from methods.physics_std import production_runner as pr
    original_prepare = pr.prepare

    def prepare(tf, *a, **k):
        res = original_prepare(tf, *a, **k)
        holder['model'] = res[0].model
        return res
    pr.prepare = prepare

    def plain_trace(tf, sess, fd, targets):
        extra['traced_calls_before_localization'] = 0
        graph = tf.get_default_graph()
        keep = base.closure(targets)
        extra['plain_place_is_requested_device'] = True
        return {o.name: o.device for o in graph.get_operations() if o.name in keep}, [], 0

    def trace2(tf, sess, fd, targets):
        extra['traced_calls_before_localization'] = 2
        r1 = original_trace(tf, sess, fd, targets)
        meta = tf.RunMetadata()
        sess.run(targets, fd, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True),
                 run_metadata=meta)
        tdir = out_dir / 'trace'
        tdir.mkdir()
        for i, gd in enumerate(meta.partition_graphs):
            (tdir / ('partition_%02d.pb' % i)).write_bytes(gd.SerializeToString(deterministic=True))
        (tdir / 'step_stats_second_trace.pb').write_bytes(meta.step_stats.SerializeToString(deterministic=True))
        import e04_d11_placement_probe as d11
        extra['first_trace_place'] = r1[0]
        extra['partitions'] = d11.partition_summary(tf, meta, base.sha)
        return r1

    def d10_trace(tf, sess, fd, targets):
        extra['traced_calls_before_localization'] = 1
        r1 = original_trace(tf, sess, fd, targets)
        extra['first_trace_place'] = r1[0]
        return r1

    base.localize = localize
    if args.factor == 'd11_wrapper':
        extra['traced_calls_before_localization'] = 2
    else:
        base.trace_devices = {'plain': plain_trace, 'd10': d10_trace, 'd10_trace2': trace2}[args.factor]
    if args.factor == 'd11_wrapper':
        import e04_d11_placement_probe as d11
        sys.argv = ['e04_d11_placement_probe.py', '--repo', str(repo), '--arm', 'original', '--process',
                    args.process if args.process != 'P4' else 'P1', '--output-dir', str(out_dir), '--iterations', '1']
        if args.process == 'P4':
            extra['d11_process_label'] = 'P1 (the D11 wrapper accepts P1-P3; this is clean process P4)'
        rc = d11.main()
    else:
        base.SMOKE_ITERATIONS = 1
        sys.argv = ['e04_d10_numerical_probe.py', '--repo', str(repo), '--process',
                    args.process if args.process != 'P4' else 'P1', '--output-dir', str(out_dir), '--smoke']
        rc = base.main()
    extra['rc'] = rc
    extra['peak_cpu_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    extra['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out_dir / 'd12_extra.json').write_text(json.dumps(extra, sort_keys=True) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
