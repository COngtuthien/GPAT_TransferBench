"""E04 D11 corrected BN-cancelled-bias detector (isolated; the D10 probe and its evidence are not changed).

The D10 detector (tools/e04_d10_numerical_probe.bn_cancelled_biases) required op type 'BiasAdd'. Under this TF build
with TF_DETERMINISTIC_OPS=1, tf.nn.bias_add emits op type 'Add' (still named BiasAdd), so it returned an empty set.
This detector accepts BiasAdd / Add / AddV2 and checks the structure that makes the bias exactly cancelled: a per-channel
bias vector added to a rank-4 NHWC tensor whose every forward consumer is a training-mode NHWC FusedBatchNorm.

    python -B tools/e04_d11_bias_detector.py --repo <checkout>     (CPU graph construction only; prints JSON)
"""
import argparse
import json
from pathlib import Path
import sys

ADD_TYPES = ('BiasAdd', 'Add', 'AddV2')


def closure(tensors):
    seen, stack = set(), [t.op for t in tensors]
    while stack:
        o = stack.pop()
        if o.name in seen:
            continue
        seen.add(o.name)
        stack.extend(i.op for i in o.inputs)
    return seen


def attr(op, name):
    v = op.get_attr(name)
    return v.decode() if isinstance(v, bytes) else v


def bn_cancelled_biases(tf, graph, trainables, losses):
    """Same signature as the D10 detector; the forward closure is the ancestors of the loss tensors."""
    forward, out = closure(losses), []
    for v in trainables:
        if not v.op.name.endswith('/bias') or v.shape.ndims != 1:
            continue
        read = v.value()
        channels = v.shape.as_list()[0]
        adds = [c for c in read.consumers() if c.name in forward]
        ok = bool(adds)
        for c in adds:
            other = [t for t in c.inputs if t.name != read.name]
            ok = ok and c.type in ADD_TYPES and len(c.inputs) == 2 and len(other) == 1 and \
                other[0].shape.ndims == 4 and other[0].shape.as_list()[-1] == channels
            if c.type == 'BiasAdd':
                ok = ok and attr(c, 'data_format') == 'NHWC'
            fwd = [u for u in c.outputs[0].consumers() if u.name in forward]
            ok = ok and bool(fwd) and all(u.type.startswith('FusedBatchNorm') and u.get_attr('is_training') and
                                          attr(u, 'data_format') == 'NHWC' and u.inputs[0].name == c.outputs[0].name
                                          for u in fwd)
        if ok:
            out.append(v.name)
    return out


def bn_sibling_biases(names):
    """Name-based cross-check: biases whose layer scope also holds a BN/beta variable."""
    s = set(names)
    return sorted(n for n in s if n.endswith('/bias:0') and n[:-len('bias:0')] + 'BN/beta:0' in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import tensorflow as tf
    from methods.physics_std.training_graph import TrainingGraph
    with tf.device('/gpu:0'):
        m = TrainingGraph(tf, repo)
    trainables = tf.trainable_variables()
    found = sorted(bn_cancelled_biases(tf, tf.get_default_graph(), trainables, [m.losses[k] for k in sorted(m.losses)]))
    sibling = bn_sibling_biases([v.name for v in trainables])
    json.dump({'cancelled': found, 'G': sum(n.startswith('G/') for n in found), 'D': sum(n.startswith('D/') for n in found),
               'equals_bn_sibling_set': found == sibling, 'bn_sibling': sibling}, sys.stdout, sort_keys=True)
    sys.stdout.write('\n')
    return 0 if found == sibling else 1


if __name__ == '__main__':
    sys.exit(main())
