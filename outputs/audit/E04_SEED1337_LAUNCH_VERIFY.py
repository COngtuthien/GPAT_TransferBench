"""Read-only early check of an E04 scientific metrics log (rule-text re-derivation, not production code).
usage: python -I -B - <run_dir> <seed>"""
import hashlib
import json
import math
import sys

import numpy as np

D, SEED = sys.argv[1], int(sys.argv[2])


def ds(dom):
    h = hashlib.sha256(('GPAT_TRANSFERBENCH|E04|%s|%d' % (dom, SEED)).encode()).digest()
    return 1 + int.from_bytes(h[:4], 'big') % 2147483647


recs = []
for line in open(D + '/metrics.jsonl').read().split('\n'):
    if line.strip():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass                                    # a partially written last line
start = [r for r in recs if r.get('event') == 'e04_run_start']
its = [r for r in recs if r.get('record_type') == 'epoch']
other = sorted({r.get('event') for r in recs if r.get('record_type') != 'epoch' and r.get('event') != 'e04_run_start'},
               key=str)
p = start[0]['payload'] if start else {}
out = {'run_start_events': len(start), 'iteration_records': len(its), 'other_events': other,
       'derived_seeds': [ds('PAIR_ORDER'), ds('HARD_SAMPLE')],
       'run_start_seeding': [p.get('seeding', {}).get(k) for k in ('pair_order_seed', 'hard_sample_seed',
                                                                    'tf_graph_seed', 'python_numpy_global')],
       'run_start_structure': p.get('structure'), 'gpu_driver': p.get('gpu_driver'),
       'environment_identity_equals_lock_except_driver': p.get('environment_identity_equals_lock_except_driver'),
       'pack': p.get('pack')}
n = len(its)
out['iterations_contiguous_from_0'] = [r['iteration'] for r in its] == list(range(n))
out['global_step_eq_iteration_plus_1'] = all(r['global_step'] == r['iteration'] + 1 for r in its)
pr = np.random.RandomState(ds('PAIR_ORDER'))
rows = np.concatenate([pr.permutation(8838) for _ in range(n * 4 // 8838 + 1)])[:n * 4]
out['pair_rows_equal_independent_stream'] = all(list(r['pair_rows']) == rows[4 * i:4 * i + 4].tolist()
                                                for i, r in enumerate(its))
h = np.random.RandomState(ds('HARD_SAMPLE'))
ok = True
for r in its:
    u = h.uniform(0, 1, (4, 1, 1, 3)).astype(np.float32)
    k = (h.uniform(0, 1, (4, 1, 1, 1)) >= 0.5).astype(np.float32)
    ok &= ((np.asarray(r['hard_scales'], np.float32) == u.reshape(-1)).all() and r['hard_keep'] == k.reshape(-1).tolist()
           and r['input_sha256']['hard_scales'] == hashlib.sha256(u.tobytes()).hexdigest())
out['hard_draws_equal_independent_stream'] = bool(ok)


def lr(t):
    return 5e-5 * 10.0 ** -(t // 45000)


out['lr_schedule'] = all(abs(r['learning_rates']['G'] - lr(r['iteration'])) <= 1e-6 * lr(r['iteration']) and
                         abs(r['learning_rates']['D'] - lr(r['iteration']) / 2) <= 1e-6 * lr(r['iteration'])
                         for r in its)
vals = [v for r in its for s in r['train_losses'].values() for v in s.values()]
out['losses_all_finite'] = all(math.isfinite(v) for v in vals)
out['loss_keys'] = {s: sorted(d) for s, d in its[0]['train_losses'].items()} if its else None
out['val_or_test_fields'] = sorted({k for r in its for k in ('val_losses', 'val_metrics') if r.get(k) is not None})
mem = [r['gpu_memory_bytes'] for r in its]
out['tf_max_bytes_in_use_max'] = max(mem) if mem else None
st = sorted(r['step_seconds'] for r in its[1:])
out['median_step_seconds'] = st[len(st) // 2] if st else None
out['last'] = {k: its[-1][k] for k in ('iteration', 'global_step', 'utc', 'train_losses')} if its else None
print(json.dumps(out, indent=1, sort_keys=True))
