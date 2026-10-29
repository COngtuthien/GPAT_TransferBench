"""E04 D10 descriptive sensitivity over the D10 probe outputs (read-only; not a protocol criterion): share of the C5
functional-group parameter deviation and of the single-iteration update carried by convolution biases that have a
sibling BN scope. Name-based stand-in for the structural BN-cancelled-bias detector, which returned an empty set in the
executed runs. Usage: python -I e04_d10_bias_sensitivity.py <probe_root> > bias_sensitivity.json
"""
import itertools, json, sys
import numpy as np
root = sys.argv[1]
S = {p: json.load(open('%s/%s/summary.json' % (root, p))) for p in ('P1', 'P2', 'P3')}
names, groups = S['P1']['variables'], S['P1']['groups']
assert all(S[p]['variables'] == names and S[p]['groups'] == groups for p in S)
allv = set(names)
def bnbias(n):
    return n.endswith('/bias:0') and (n[:-len('bias:0')] + 'BN/beta:0') in allv
idx = {g: [i for i, n in enumerate(names) if groups.get(n) == g] for g in ('G_functional', 'D_functional')}
cnt = {g: (sum(bnbias(names[i]) for i in ix), len(ix)) for g, ix in idx.items()}
def snap(p, t):
    z = np.load('%s/%s/snap_%04d.npz' % (root, p, t))
    return [z['v%03d' % i].astype(np.float64).ravel() for i in range(len(names))]
out = {'bn_sibling_biases_in_group': cnt, 'deviation': [], 'update_0_to_1': {}}
for p in S:
    a, b = snap(p, 0), snap(p, 1)
    for g, ix in idx.items():
        tot = sum(float(((b[i] - a[i]) ** 2).sum()) for i in ix)
        bi = sum(float(((b[i] - a[i]) ** 2).sum()) for i in ix if bnbias(names[i]))
        out['update_0_to_1']['%s|%s' % (p, g)] = {'total_l2': tot ** .5, 'bias_l2': bi ** .5, 'without_bias_l2': (tot - bi) ** .5}
for t in S['P1']['snapshot_iterations']:
    if t < 1:
        continue
    sn = {p: snap(p, t) for p in S}
    for pa, pb in itertools.combinations(sorted(S), 2):
        for g, ix in idx.items():
            tot = sum(float(((sn[pa][i] - sn[pb][i]) ** 2).sum()) for i in ix)
            bi = sum(float(((sn[pa][i] - sn[pb][i]) ** 2).sum()) for i in ix if bnbias(names[i]))
            out['deviation'].append({'iteration': t, 'pair': [pa, pb], 'group': g, 'total_l2': tot ** .5,
                                     'bias_l2': bi ** .5, 'without_bias_l2': (tot - bi) ** .5})
print(json.dumps(out, indent=1))
