"""E04 D9 investigation probe (non-scientific; synthetic only; GPU host, gpat-m6-e03 Python 3.8 / TF 1.15.5).

Runs the unchanged M6D3e qualification `methods.physics_std.training_runtime.qualify` once (qualification seed 314159,
generated fixtures, exactly one complete three-apply minibatch) under the worker's own precision environment and
file-access firewall. The only addition is a pass-through wrapper around the worker's `summarize` that keeps the raw
bytes of 1-D float32 state tensors by sha256, so the post-step BN statistics can be compared element-wise. Afterwards a
second graph is constructed (never run) to list which operations read each discriminator BN moving statistic.

    python -B tools/e04_d9_bn_probe.py --repo <checkout> --output /tmp/<dir>/probe_<k>.json

No benchmark data, checkpoint, run root or experiment seed is involved; the output must be under /tmp.
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(globals().get('__file__', '.')).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--repo', default=str(ROOT), help='checkout whose unchanged E04 worker is executed')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    sys.path.insert(0, str(root))
    if not os.path.abspath(args.output).startswith('/tmp/'):
        raise SystemExit('probe output must be under /tmp')
    from methods.physics_std import training_runtime as rt
    os.environ.update(rt.PRECISION)
    firewall = rt.Firewall()
    sys.addaudithook(firewall)
    import numpy as np
    kept, original = {}, rt.summarize

    def summarize(value):
        record = original(value)
        a = np.asarray(value)
        if a.dtype == np.float32 and a.ndim == 1:
            kept[record['sha256']] = base64.b64encode(a.tobytes()).decode()
        return record
    rt.summarize = summarize
    report = rt.qualify(root)
    rt.summarize = original
    names = sorted(n for n in report['steps'][0]['post_state'] if n.startswith('D/') and '/BN/' in n)
    states = {'initial': report['initial_state']}
    states.update({'step%d' % s['step']: s['post_state'] for s in report['steps']})
    tensors = {k: {n: {'sha256': st[n]['sha256'], 'shape': st[n]['shape'], 'f32_b64': kept.get(st[n]['sha256'])}
                   for n in names} for k, st in states.items()}

    import tensorflow as tf
    graph_module, _ = rt.import_graph_package()
    g = tf.Graph()
    with g.as_default(), tf.device('/gpu:0'):
        tf.set_random_seed(rt.SEED)
        graph_module.TrainingGraph(tf, root)
    consumers = {}
    for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if v.name.startswith('D/') and '/BN/moving_' in v.name:
            ops = set()
            for t in [o for o in g.get_operations() if v.op in [i.op for i in o.inputs]]:
                for out in t.outputs:
                    ops.update((c.type, c.name.split('/')[-1]) for c in out.consumers())
                ops.add((t.type, t.name.split('/')[-1]))
            consumers[v.name] = sorted({op for op, _ in ops})
    out = {'status': report['status'], 'repo': str(root), 'qualification_seed': report['qualification_seed'],
           'graph_def_sha256': report['graph_def_sha256'], 'fixture_sha256': report['fixture_sha256'],
           'native_runtime': report['native_runtime'], 'tensorflow': report['tensorflow'],
           'optimizer_applications': report['optimizer_applications'],
           'bn_changes_by_forward_collection': [s['bn_changes_by_forward_collection'] for s in report['steps']],
           'changed_trainables': [s['changed_trainables'] for s in report['steps']],
           'losses_before': [s['losses_before'] for s in report['steps']], 'final_losses': report['final_losses'],
           'post_state_sha256': {k: {n: r['sha256'] for n, r in st.items()} for k, st in states.items()},
           'd_bn_tensors': tensors, 'moving_statistic_consumer_op_types': consumers,
           'firewall_denied': firewall.denied}
    Path(args.output).write_text(json.dumps(out, sort_keys=True) + '\n')
    print(json.dumps({'status': out['status'], 'applies': out['optimizer_applications'],
                      'captured': sum(1 for st in tensors.values() for r in st.values() if r['f32_b64'])}))
    return 0 if report['status'] == 'PASS' and not firewall.denied else 1


if __name__ == '__main__':
    sys.exit(main())
