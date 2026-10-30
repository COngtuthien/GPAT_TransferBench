"""Read-only independent check of the E04 seed-42 metrics log (rule text re-derivation, not production code)."""
import json
import math
import sys

import numpy as np

D = sys.argv[1]
lines = open(D + '/metrics.jsonl').read().split('\n')
recs = []
for l in lines:
    if not l.strip():
        continue
    try:
        recs.append(json.loads(l))
    except ValueError:
        pass  # a partially written last line
start = [r for r in recs if r.get('event') == 'e04_run_start']
its = [r for r in recs if r.get('record_type') == 'epoch']
other = [r for r in recs if r.get('record_type') != 'epoch' and r.get('event') != 'e04_run_start']
out = {'run_start_events': len(start), 'iteration_records': len(its),
       'other_events': sorted({r.get('event') for r in other}, key=str)}
n = len(its)
out['iterations_contiguous_from_0'] = [r['iteration'] for r in its] == list(range(n))
out['global_step_eq_iteration_plus_1'] = all(r['global_step'] == r['iteration'] + 1 for r in its)

pr = np.random.RandomState(679869707)
rows = np.concatenate([pr.permutation(8838) for _ in range(n * 4 // 8838 + 1)])[:n * 4]
out['pair_rows_equal_independent_stream'] = all(list(r['pair_rows']) == rows[4 * i:4 * i + 4].tolist()
                                                for i, r in enumerate(its))
h = np.random.RandomState(320493565)
ok = True
for r in its:
    u = h.uniform(0, 1, (4, 1, 1, 3)).astype(np.float32)
    k = (h.uniform(0, 1, (4, 1, 1, 1)) >= 0.5).astype(np.float32)
    ok &= (np.asarray(r['hard_scales'], np.float32) == u.reshape(-1)).all() and r['hard_keep'] == k.reshape(-1).tolist()
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
print(json.dumps(out, indent=1))
