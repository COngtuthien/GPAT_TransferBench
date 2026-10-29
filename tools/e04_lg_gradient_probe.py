"""E04 diagnostic: magnitude of the in-process L_G gradient repeat difference (5 fetches, real TRAIN iteration-0
batch, qualification seed 60904) and the GPU op types found only on the L_G backward path. No RunContext, no write
except stdout. Usage (GPU host, E04 precision env): python -B tools/e04_lg_gradient_probe.py <checkout>
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from methods.physics_std import production_io as pio
assert not pio.launch_environment_errors(os.environ, pio.QUALIFICATION_SEED)
import tensorflow as tf
from methods.physics_std import production_runner as pr
rt = Path('/home/student20261/workdir/GPAT_TransferBench_runtime')
ctl, *_ = pr.prepare(tf, pio.QUALIFICATION, pio.QUALIFICATION_SEED, rt, rt / 'data/processed/faces_256')
m = ctl.model
pairs = [(g, v) for g, v in m.per_loss_gradients['L_G'] if g is not None]
others = [g for k, ps in m.per_loss_gradients.items() if k != 'L_G' for g, _ in ps if g is not None]
def closure(ts):
    seen, st = set(), [t.op for t in ts]
    while st:
        o = st.pop()
        if o.name in seen: continue
        seen.add(o.name); st.extend(i.op for i in o.inputs)
    return seen
g = tf.get_default_graph()
lg_ops, other_ops = closure([p[0] for p in pairs]), closure(others)
only = [g.get_operation_by_name(n) for n in lg_ops - other_ops]
types = {}
for o in only: types[o.type] = types.get(o.type, 0) + 1
with tf.Session(config=pr.session_config(tf)) as sess:
    sess.run(tf.global_variables_initializer())
    arrays, _ = ctl.feed.next(); fd = ctl.feed_dict(arrays)
    runs = [sess.run([p[0] for p in pairs], fd) for _ in range(5)]
    meta = tf.RunMetadata()
    sess.run([p[0] for p in pairs], fd, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=meta)
place = {n.node_name: d.device for d in meta.step_stats.dev_stats for n in d.node_stats}
only_gpu = {}
for o in only:
    if 'gpu' in (place.get(o.name) or '').lower():
        only_gpu[o.type] = only_gpu.get(o.type, 0) + 1
res = {}
for i, (_, v) in enumerate(pairs):
    base = runs[0][i].astype(np.float64)
    d = max(float(np.abs(r[i].astype(np.float64) - base).max()) for r in runs[1:])
    res[v.name] = {'max_abs_diff': d, 'abs_max': float(np.abs(base).max()),
                   'rel_l2_max': max(float(np.linalg.norm(r[i] - runs[0][i]) / max(np.linalg.norm(runs[0][i]), 1e-30)) for r in runs[1:]),
                   'distinct_of_5': len({r[i].tobytes() for r in runs})}
print(json.dumps({"per_variable": res, "variables": len(res), 'max_rel_l2': max(x['rel_l2_max'] for x in res.values()),
                  'max_abs_diff': max(x['max_abs_diff'] for x in res.values()),
                  'variables_varying': sum(x['distinct_of_5'] > 1 for x in res.values()),
                  'distinct_counts': sorted({x['distinct_of_5'] for x in res.values()}),
                  'ops_only_in_LG_backward_path_by_type': types, 'those_ops_placed_on_gpu_by_type': only_gpu}, sort_keys=True))
