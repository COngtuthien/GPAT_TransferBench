"""E04 D11 investigation analysis (fixed with outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json before execution).

    python -B tools/e04_d11_analysis.py short --repo <checkout> --root <S_short dir with original/ cpu_scatternd/>
    python -B tools/e04_d11_analysis.py trace --repo <checkout> --root <dir> [--root <dir> ...]   (descriptive only)
    python -B tools/e04_d11_analysis.py bounded --repo <checkout> --root <L_bounded dir> --short-root <S_short dir>
        --control-root <D10 probe dir with P1 P2 P3>

short evaluates S1-S8 (decision PROCEED_TO_L_BOUNDED or E04_D11_UNSUPPORTED); bounded evaluates the established C1-C6
with the unchanged tools/e04_d10_analysis.analyse plus L_C5_D10_grouping and L_scope and returns the protocol status.
numpy only, Python 3.8 compatible. Prints one JSON document.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ARMS = ('original', 'cpu_scatternd')
PROCESSES = ('P1', 'P2', 'P3')
C1_FIELDS = ('pair_rows', 'pair_passes', 'hard_scales', 'hard_keep', 'input_sha256', 'counters_before',
             'counters_after', 'call_log_sha256', 'beta_powers')
LOOKAHEAD_FIELDS = ('pair_rows', 'pair_passes', 'hard_scales', 'hard_keep', 'input_sha256')
FUNCTIONAL = ('G_functional', 'D_functional')
MERGE = {'G_cancelled_bias': 'G_functional', 'D_cancelled_bias': 'D_functional'}
SHORT_ITERATIONS = 25
C4_THRESHOLD = 1e-3
STATEFUL_NAMED = ('VariableV2', 'VarHandleOp')
TRANSFER = ('_Send', '_Recv', '_HostSend', '_HostRecv')
TRANSFER_ATTRS = ('_class', 'tensor_name', 'send_device', 'recv_device', 'send_device_incarnation',
                  'client_terminated')


def dev_type(d):
    return None if d is None else ('GPU' if 'GPU' in d.upper() else 'CPU')


def load(d):
    d = Path(d)
    return {'dir': d, 'summary': json.loads((d / 'summary.json').read_text()),
            'records': [json.loads(x) for x in (d / 'iterations.jsonl').read_text().splitlines()],
            'extra': json.loads((d / 'd11_extra.json').read_text())}


def quantiles(xs):
    import numpy as np
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, np.float64)
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())}


def grads(proc, t):
    import numpy as np
    z = np.load(str(proc['dir'] / ('grad_%04d.npz' % t)))
    return {k: z[k] for k in sorted(z.files)}


def concat_l2(a, b, prefix):
    import numpy as np
    keys = [k for k in a if k.startswith(prefix + '_')]
    return sum(float(np.sum((a[k].astype(np.float64) - b[k].astype(np.float64)) ** 2)) for k in keys) ** .5


def reference(repo):
    d10 = json.loads((repo / 'outputs/audit/E04_D10_INVESTIGATION.json').read_text())
    qual = json.loads((repo / 'outputs/audit/E04_PRODUCTION_QUALIFICATION.json').read_text())
    return sorted(d10['findings']['C3_localization']['origins']), qual['graph']['construction']


def scope_checks(P, arm, origins, construction, controls):
    """S1 / L_scope for the processes P (all of one arm); controls = original-arm processes for the device check."""
    out, cross = {}, []
    for name, p in P.items():
        e = p['extra']
        pl, tr = e['placement'], e['trace']
        nd = pl['node_diff']
        want = 'CPU' if arm == 'cpu_scatternd' else 'GPU'
        refs = sorted(r['op'] for r in pl['colocation_references'])
        if arm == 'cpu_scatternd':
            tch = [c for c in nd['changed'] if c['node'] in origins]
            rch = [c for c in nd['changed'] if c['node'] not in origins]
            diff_ok = [c['node'] for c in tch] == origins and all(
                c['fields'] == ['device'] and c['device_after'] == '/device:CPU:0' for c in tch) and \
                sorted(c['node'] for c in rch) == refs == pl['released_colocation'] and \
                all(c['fields'] == ['attr:_class'] for c in rch) and all(
                    not any(x.startswith('loc:@') and x[5:] in origins for x in r['class_after'])
                    for r in pl['colocation_references'])
        else:
            diff_ok = nd['changed'] == [] and pl['released_colocation'] == []
        out[name] = {
            'targets_equal_d10_origins': pl['target_names'] == origins,
            'targets_are_cpu_gathernd_gradients': all(t['forward_requested_device'].upper().endswith('CPU:0')
                                                      for t in pl['targets']),
            'graphdef_diff_exactly_scoped': diff_ok and not nd['added'] and not nd['removed'],
            'construction_before_equals_qualified': pl['construction_before'] == construction,
            'targets_executed_on_%s_both_traces' % want: all(dev_type(a) == want and dev_type(b) == want
                                                             for a, b in tr['target_executed_device'].values()),
            'arm_and_process': [e['arm'], e['process']]}
        for cname, c in controls.items():
            if c is p:
                continue
            a, b = tr['first_trace_place'], c['extra']['trace']['first_trace_place']
            bad = sorted(n for n in set(a) & set(b) if n not in origins and dev_type(a[n]) != dev_type(b[n]))
            cross.append({'process': name, 'control': cname, 'common_executed_ops': len(set(a) & set(b)),
                          'non_target_device_type_differences': bad[:20]})
    ok = all(all(v for k, v in r.items() if k != 'arm_and_process') for r in out.values()) and \
        all(not c['non_target_device_type_differences'] for c in cross)
    return ok, {'per_process': out, 'cross_arm_device_check': cross}


def short(repo, root):
    import numpy as np
    repo, root = Path(repo), Path(root)
    origins, construction = reference(repo)
    A = {arm: {p: load(root / arm / p) for p in PROCESSES} for arm in ARMS}
    allp = {'%s/%s' % (arm, p): A[arm][p] for arm in ARMS for p in PROCESSES}
    res, det = {}, {}

    # S8 completion
    res['S8_completion'] = all(x['extra']['rc'] == 0 and len(x['records']) == SHORT_ITERATIONS and
                               x['summary']['iterations'] == SHORT_ITERATIONS for x in allp.values())
    # S1 placement scope
    s1o, d1o = scope_checks(A['original'], 'original', origins, construction, {})
    s1c, d1c = scope_checks(A['cpu_scatternd'], 'cpu_scatternd', origins, construction,
                            {'original/' + p: A['original'][p] for p in PROCESSES})
    res['S1_placement_scope'] = s1o and s1c
    det['S1'] = {'original': d1o, 'cpu_scatternd': d1c}
    # S2 gradients present
    pres = {k: x['extra']['gradients_present'] for k, x in allp.items()}
    res['S2_gradients_present'] = len({json.dumps(v, sort_keys=True) for v in pres.values()}) == 1
    det['S2'] = {'missing_gradients': {k: sum(not b for b in v) for k, v in pres['original/P1'].items()}}
    # S3 contract identity across all 6 processes
    ref = allp['original/P1']
    n = min(len(x['records']) for x in allp.values())
    mism = {f: [t for t in range(n) if any(x['records'][t][f] != ref['records'][t][f] for x in allp.values())][:20]
            for f in C1_FIELDS}
    la_bad = [[k, r['iteration']] for k, x in allp.items() for r in x['records'] if 'gradient_probe' in r and
              r['iteration'] + 1 < n and any(r['gradient_probe']['lookahead_info'][f] != x['records'][r['iteration'] + 1][f]
                                             for f in LOOKAHEAD_FIELDS)]
    s3 = {'initial_state_equal': len({x['summary']['initial_state_sha256'] for x in allp.values()}) == 1,
          'firewall_clean': all(not x['summary']['firewall']['denied'] and not x['summary']['firewall']['runtime_writes']
                                for x in allp.values()),
          'diagnostics_never_changed_state': all(x['summary']['iteration0_diagnostics_state_unchanged'] and
                                                 all(r['diagnostic_fetch_state_unchanged'] for r in x['records'])
                                                 for x in allp.values()),
          'lookahead_equals_next_batch': not la_bad}
    s3.update({'per_iteration_' + f: not v for f, v in mism.items()})
    res['S3_contract_identity'] = all(s3.values())
    det['S3'] = {'checks': s3, 'first_mismatches': mism, 'lookahead_mismatches': la_bad}
    # S4 forward identity across arms (iteration 0)
    dep = A['cpu_scatternd']['P1']['extra']['placement']['probe_sets_depending_on_targets']
    sets = sorted(ref['records'][0]['gradient_probe']['repeats'])
    indep = [k for k in sets if k not in dep]
    r0 = {k: x['records'][0] for k, x in allp.items()}
    s4_loss = len({json.dumps(r['losses']['step1'], sort_keys=True) for r in r0.values()}) == 1
    s4_grad = {k: len({json.dumps(r['gradient_probe']['repeats'][k]['per_variable_sha256_r0']) for r in r0.values()}) == 1
               for k in indep}
    res['S4_forward_identity'] = s4_loss and all(s4_grad.values())
    det['S4'] = {'step1_losses_equal_across_arms': s4_loss, 'independent_sets_equal_across_arms': s4_grad,
                 'probe_sets_depending_on_targets': dep}
    # S5 cpu bitwise reproducibility
    C = A['cpu_scatternd']
    loc = {p: {k: C[p]['summary']['localization'][k] for k in ('varying_tensors', 'closure_ops', 'fetched_tensors')}
           for p in PROCESSES}
    loc_ok = all(C[p]['summary']['localization']['varying_tensors'] == 0 and not C[p]['summary']['localization']['origins']
                 and not C[p]['summary']['localization']['undetermined_origins'] for p in PROCESSES)
    rep_bad = [[p, r['iteration'], k] for p in PROCESSES for r in C[p]['records'] if 'gradient_probe' in r
               for k, v in r['gradient_probe']['repeats'].items() if max(v['distinct_per_variable']) > 1]
    rep_bad += [[p, r['iteration'], k] for p in PROCESSES for r in C[p]['records'] for k in ('repeat_L_G', 'repeat_step1')
                if r[k]['concat']['identical_variables'] != r[k]['concat']['variables']]
    state_bad = [t for t in range(n) if len({C[p]['records'][t]['state_sha256'] for p in PROCESSES}) > 1]
    loss_bad = [t for t in range(n) if len({json.dumps(C[p]['records'][t]['losses'], sort_keys=True)
                                             for p in PROCESSES}) > 1]
    gset = C['P1']['summary']['gradient_probe_iterations']
    grad_bad = []
    for t in gset:
        g = {p: grads(C[p], t) for p in PROCESSES}
        if any(any(g[p][k].tobytes() != g['P1'][k].tobytes() for k in g['P1']) for p in PROCESSES):
            grad_bad.append(t)
    res['S5_cpu_bitwise_reproducibility'] = loc_ok and not rep_bad and not state_bad and not loss_bad and not grad_bad
    det['S5'] = {'localization': loc, 'localization_clean': loc_ok, 'repeat_differences': rep_bad[:50],
                 'state_sha256_differs_at': state_bad[:20], 'losses_differ_at': loss_bad[:20],
                 'saved_gradients_differ_at': grad_bad}
    # S6 finite
    res['S6_finite'] = all(r['state_finite'] and r['losses_finite'] and r['repeat_step1']['concat']['finite'] and
                           r['repeat_L_G']['concat']['finite'] and
                           all(c['finite'] for k in r.get('gradient_probe', {}).get('repeats', {}).values()
                               for c in k['concat'])
                           for x in allp.values() for r in x['records'])
    # S7 deviation from the unchanged control at iteration 0
    s7 = []
    for pc, po in itertools.product(PROCESSES, PROCESSES):
        gc, go = grads(C[pc], 0), grads(A['original'][po], 0)
        rec = A['original'][po]['records'][0]['gradient_probe']
        for k in ('L_G', 'step1'):
            d = concat_l2(gc, go, k)
            mb = rec['minibatch_reference'][k]['concat']['l2_diff']
            gpu_rep = max(c['l2_diff'] for c in rec['repeats'][k]['concat'])
            s7.append({'cpu': pc, 'original': po, 'gradient': k, 'l2_cpu_minus_original': d, 'minibatch_l2': mb,
                       'ratio': d / mb, 'original_gpu_repeat_l2_max': gpu_rep,
                       'relative_l2': d / rec['repeats'][k]['concat'][0]['l2_ref'], 'pass': d <= C4_THRESHOLD * mb})
    res['S7_deviation_from_control'] = bool(s7) and all(x['pass'] for x in s7)
    det['S7'] = {'threshold': C4_THRESHOLD, 'points': s7, 'ratio': quantiles([x['ratio'] for x in s7]),
                 'relative_l2': quantiles([x['relative_l2'] for x in s7])}

    det['descriptive'] = descriptive_short(A)
    decision = 'PROCEED_TO_L_BOUNDED' if all(res.values()) else 'E04_D11_UNSUPPORTED'
    return {'schema_version': 'e04_d11_short_analysis_v1', 'stage': 'S_short', 'criteria': res, 'decision': decision,
            'detail': det, 'protocol_sha256': hashlib.sha256((repo / 'outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json')
                                                             .read_bytes()).hexdigest()}


def descriptive_short(A):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import e04_d10_analysis as d10a
    out = {'runtime_seconds_per_production_step': {}, 'peak_gpu_memory_bytes': {}, 'peak_cpu_rss_kib': {},
           'runtime_transfer_nodes': {}, 'executed_ops': {}}
    for arm in ARMS:
        for p in PROCESSES:
            x = A[arm][p]
            k = '%s/%s' % (arm, p)
            out['runtime_seconds_per_production_step'][k] = float(np.median([r['iteration_seconds'] for r in x['records']]))
            out['peak_gpu_memory_bytes'][k] = x['extra']['peak_gpu_memory_bytes']
            out['peak_cpu_rss_kib'][k] = x['extra']['peak_cpu_rss_kib']
            out['runtime_transfer_nodes'][k] = len(x['extra']['trace']['second_trace_runtime_nodes'])
            t = x['summary']['trace']
            out['executed_ops'][k] = {f: t[f] for f in ('executed_nodes', 'executed_distinct_signatures',
                                                       'executed_signature_set_sha256', 'executed_node_set_sha256')}
            out['executed_ops'][k]['executed_set_equal_between_traces'] = \
                x['extra']['trace']['executed_set_equal_between_traces']
    med = {arm: float(np.median([out['runtime_seconds_per_production_step']['%s/%s' % (arm, p)] for p in PROCESSES]))
           for arm in ARMS}
    out['runtime_ratio_cpu_over_original'] = med['cpu_scatternd'] / med['original']
    # parameter / optimizer / BN deviation from the control vs the control's own cross-process spread
    names, groups = A['original']['P1']['summary']['variables'], A['original']['P1']['summary']['groups']
    rows = []
    for t in A['original']['P1']['summary']['snapshot_iterations']:
        if t == 0:
            continue
        sn = {(arm, p): d10a.snapshot(A[arm][p], t) for arm in ARMS for p in PROCESSES}
        for (a, b) in [(('cpu_scatternd', 'P1'), ('original', 'P1')), (('original', 'P1'), ('original', 'P2')),
                       (('cpu_scatternd', 'P1'), ('cpu_scatternd', 'P2'))]:
            dev = d10a.group_dev(sn[a], sn[b], names, groups)
            rows.append({'iteration': t, 'pair': ['/'.join(a), '/'.join(b)],
                         'normalized': {g: (v['l2_diff'] / v['l2_ref'] if v['l2_ref'] > 0 else None)
                                        for g, v in sorted(dev.items())},
                         'l2_diff': {g: v['l2_diff'] for g, v in sorted(dev.items())}})
    out['state_deviation'] = rows
    return out


# ---------------------------------------------------------------------- descriptive op-set forensics (T_trace)
def canonical(parts):
    """Content hash of every partition-graph node (names excluded except variables; transfer edges resolved)."""
    nodes = {n['name']: n for part in parts for n in part['nodes']}
    sends = {n['transfer']['tensor_name']: n for n in nodes.values()
             if n['op'] in ('_Send', '_HostSend') and 'transfer' in n}
    deps = {}
    for name, n in nodes.items():
        d = [i.lstrip('^').split(':')[0] for i in n['inputs']]
        if n['op'] in ('_Recv', '_HostRecv') and n.get('transfer', {}).get('tensor_name') in sends:
            d.append(sends[n['transfer']['tensor_name']]['name'])
        deps[name] = [x for x in d if x in nodes]
    memo, order, state = {}, [], {}
    for root in nodes:
        stack = [(root, False)]
        while stack:
            k, done = stack.pop()
            if done:
                order.append(k)
                state[k] = 2
                continue
            if state.get(k):
                continue
            state[k] = 1
            stack.append((k, True))
            stack.extend((x, False) for x in deps[k] if not state.get(x))
    for k in order:
        n = nodes[k]
        attrs = {a: v for a, v in n['attrs'].items() if a not in TRANSFER_ATTRS}
        data = [[memo[i.split(':')[0]], i.split(':')[1] if ':' in i else '0'] for i in n['inputs']
                if not i.startswith('^') and i.split(':')[0] in memo]
        ctrl = sorted(memo[i[1:].split(':')[0]] for i in n['inputs'] if i.startswith('^') and i[1:] in memo)
        extra = [memo[x] for x in deps[k] if nodes[x]['op'] in ('_Send', '_HostSend')]
        memo[k] = hashlib.sha256(json.dumps([n['op'], dev_type(n['device']), attrs, data, ctrl, extra, n.get('value'),
                                             n['name'] if n['op'] in STATEFUL_NAMED else None]).encode()).hexdigest()
    return memo, nodes


def trace_forensics(roots):
    procs = []
    for r in roots:
        for d in sorted(Path(r).glob('**/d11_extra.json')):
            procs.append(load(d.parent))
    out = []
    canon = {}
    for x in procs:
        e, t = x['extra'], x['summary']['trace']
        key = '%s:%s/%s' % (x['dir'].parent.parent.name, e['arm'], x['dir'].name)
        memo, nodes = canonical(e['trace']['partitions'])
        retval = sorted(memo[n] for n, v in nodes.items() if v['op'] == '_Retval')
        canon[key] = (memo, nodes)
        ms = {}
        for h in memo.values():
            ms[h] = ms.get(h, 0) + 1
        out.append({'process': key, 'arm': e['arm'], 'executed_nodes': t['executed_nodes'],
                    'executed_distinct_signatures': t['executed_distinct_signatures'],
                    'executed_signature_set_sha256': t['executed_signature_set_sha256'],
                    'executed_set_equal_between_two_traces': e['trace']['executed_set_equal_between_traces'],
                    'partition_nodes': len(nodes),
                    'partition_op_counts': {op: sum(v['op'] == op for v in nodes.values())
                                            for op in sorted({v['op'] for v in nodes.values()})},
                    'partition_canonical_multiset_sha256': hashlib.sha256(json.dumps(sorted(ms.items())).encode())
                    .hexdigest(),
                    'partition_canonical_set_sha256': hashlib.sha256('\n'.join(sorted(ms)).encode()).hexdigest(),
                    'retval_canonical_sha256': hashlib.sha256('\n'.join(retval).encode()).hexdigest(),
                    'retvals': len(retval)})
    diffs = []
    keys = sorted(canon)
    for a, b in itertools.combinations(keys, 2):
        if canon[a][1] is None or a.split(':')[1].split('/')[0] != b.split(':')[1].split('/')[0]:
            continue
        ha, hb = set(canon[a][0].values()), set(canon[b][0].values())
        if ha == hb:
            continue
        only = lambda m, n, s: sorted({(v['op'], dev_type(v['device'])) for k, v in n.items() if m[k] in s})  # noqa
        diffs.append({'pair': [a, b], 'only_in_first': only(canon[a][0], canon[a][1], ha - hb),
                      'only_in_second': only(canon[b][0], canon[b][1], hb - ha),
                      'counts': [len(ha - hb), len(hb - ha)]})
    return {'processes': out, 'canonical_set_differences': diffs[:50]}


# ---------------------------------------------------------------------- bounded stage
def bounded(repo, root, short_root, control_root):
    import numpy as np
    repo = Path(repo)
    sys.path.insert(0, str(repo / 'tools'))
    import e04_d10_analysis as d10a
    origins, construction = reference(repo)
    est = d10a.analyse(repo, Path(root) / 'cpu_scatternd')
    P = {p: load(Path(root) / 'cpu_scatternd' / p) for p in PROCESSES}
    controls = {'original/' + p: load(Path(short_root) / 'original' / p) for p in PROCESSES}
    scope_ok, scope = scope_checks(P, 'cpu_scatternd', origins, construction, controls)
    # C5 with the BN-cancelled biases merged into the functional groups (the D10-executed grouping)
    names = P['P1']['summary']['variables']
    groups = {k: MERGE.get(v, v) for k, v in P['P1']['summary']['groups'].items()}
    R = {p: P[p]['records'] for p in PROCESSES}
    sset = P['P1']['summary']['snapshot_iterations']
    c5m = []
    for t in sset:
        if t == 0:
            continue
        sn = {p: d10a.snapshot(P[p], t) for p in PROCESSES}
        for a, b in itertools.combinations(PROCESSES, 2):
            dev = d10a.group_dev(sn[a], sn[b], names, groups)
            for g in FUNCTIONAL:
                cb = {v: k for k, v in MERGE.items()}[g]
                u = [float(np.median([(R[p][i]['update_norms'][g] ** 2 + R[p][i]['update_norms'].get(cb, 0.) ** 2) ** .5
                                      for i in range(max(0, t - 10), t)])) for p in (a, b)]
                c5m.append({'iteration': t, 'pair': [a, b], 'group': g, 'deviation': dev[g]['l2_diff'],
                            'update_scale': min(u), 'pass': dev[g]['l2_diff'] <= min(u)})
    c5m_ok = bool(c5m) and all(x['pass'] for x in c5m)
    n = len(R['P1'])
    bitwise = {'state_sha256_differs_at': [t for t in range(n) if len({R[p][t]['state_sha256'] for p in PROCESSES}) > 1][:20],
               'losses_differ_at': [t for t in range(n) if len({json.dumps(R[p][t]['losses'], sort_keys=True)
                                                                for p in PROCESSES}) > 1][:20]}
    bitwise['bit_identical_trajectories'] = not bitwise['state_sha256_differs_at'] and not bitwise['losses_differ_at']
    # descriptive: deviation from the D10 original-placement control trajectories
    ctrl = {p: {'dir': Path(control_root) / p, 'summary': json.loads((Path(control_root) / p / 'summary.json').read_text())}
            for p in PROCESSES}
    cgroups = ctrl['P1']['summary']['groups']
    rows = []
    for t in sset:
        if t == 0:
            continue
        sc = {p: d10a.snapshot(ctrl[p], t) for p in PROCESSES}
        sx = d10a.snapshot(P['P1'], t)
        for label, a, b in (('cpu_P1-vs-control_P1', sx, sc['P1']), ('control_P1-vs-control_P2', sc['P1'], sc['P2'])):
            dev = d10a.group_dev(a, b, names, cgroups)
            rows.append({'iteration': t, 'pair': label,
                         'normalized': {g: (v['l2_diff'] / v['l2_ref'] if v['l2_ref'] > 0 else None)
                                        for g, v in sorted(dev.items())}})
    d10_doc = json.loads((repo / 'outputs/audit/E04_D10_ANALYSIS.json').read_text())
    rt = {p: float(np.median([r['iteration_seconds'] for r in R[p]])) for p in PROCESSES}
    crit = dict(est['criteria'], L_C5_D10_grouping=c5m_ok, L_scope=scope_ok)
    c1_other = all(v for k, v in est['detail']['C1']['checks'].items() if k != 'executed_signature_set_equal')
    rest = all(v for k, v in crit.items() if k != 'C1_contract_identity')
    if all(crit.values()):
        status = 'E04_D11_READY_FOR_OWNER_APPROVAL'
    elif not crit['C1_contract_identity'] and c1_other and rest and bitwise['bit_identical_trajectories']:
        status = 'E04_D11_INCONCLUSIVE'
    else:
        status = 'E04_D11_UNSUPPORTED'
    return {'schema_version': 'e04_d11_bounded_analysis_v1', 'stage': 'L_bounded', 'status': status,
            'criteria': crit, 'established_analysis': est, 'L_C5_D10_grouping': {'points': c5m},
            'L_scope': scope, 'bitwise': bitwise, 'deviation_from_d10_control': rows,
            'runtime_seconds_per_production_step': rt,
            'runtime_ratio_vs_d10_original': float(np.median(list(rt.values()))) / float(np.median(
                list(d10_doc['seconds_per_iteration_production_step'].values()))),
            'peak_gpu_memory_bytes': {p: P[p]['extra']['peak_gpu_memory_bytes'] for p in PROCESSES},
            'peak_cpu_rss_kib': {p: P[p]['extra']['peak_cpu_rss_kib'] for p in PROCESSES},
            'protocol_sha256': hashlib.sha256((repo / 'outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json')
                                              .read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['short', 'trace', 'bounded'])
    ap.add_argument('--repo', required=True)
    ap.add_argument('--root', action='append', required=True)
    ap.add_argument('--short-root')
    ap.add_argument('--control-root')
    a = ap.parse_args()
    if a.stage == 'short':
        doc = short(a.repo, a.root[0])
    elif a.stage == 'trace':
        doc = trace_forensics(a.root)
    else:
        doc = bounded(a.repo, a.root[0], a.short_root, a.control_root)
    json.dump(doc, sys.stdout, sort_keys=True)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
