"""E04 D10 investigation analysis (fixed with outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json before execution).

Evaluates criteria C1-C6 of the protocol on the three probe processes written by tools/e04_d10_numerical_probe.py and
prints one JSON document (decision D10_SUPPORTED / STOP_AND_REPORT / DECISION_REPORT_NOT_SUPPORTED plus the descriptive
error distributions). numpy only, Python 3.8 compatible:

    python -B tools/e04_d10_analysis.py --repo <checkout> --probe-root <dir with P1 P2 P3> > analysis.json
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

PROCESSES = ('P1', 'P2', 'P3')
ALLOWED_ORIGIN_TYPES = ('ResizeBilinearGrad', 'ScatterNd')
DETERMINISTIC_PROBES = ('L_depth', 'L_P', 'L_R', 'L_S', 'L_H', 'L_D', 'step2', 'step3')
FUNCTIONAL = ('G_functional', 'D_functional')
C1_FIELDS = ('pair_rows', 'pair_passes', 'hard_scales', 'hard_keep', 'input_sha256', 'counters_before',
             'counters_after', 'call_log_sha256', 'beta_powers')
LOOKAHEAD_FIELDS = ('pair_rows', 'pair_passes', 'hard_scales', 'hard_keep', 'input_sha256')


def quantiles(xs):
    import numpy as np
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, np.float64)
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)), 'p90': float(np.quantile(a, .9)),
            'p99': float(np.quantile(a, .99)), 'max': float(a.max())}


def load(root, p):
    d = Path(root) / p
    s = json.loads((d / 'summary.json').read_text())
    recs = [json.loads(line) for line in (d / 'iterations.jsonl').read_text().splitlines()]
    return {'dir': d, 'summary': s, 'records': recs}


def snapshot(proc, t):
    import numpy as np
    z = np.load(str(proc['dir'] / ('snap_%04d.npz' % t)))
    return [z['v%03d' % i] for i in range(len(proc['summary']['variables']))]


def group_dev(a, b, names, groups):
    import numpy as np
    out = {}
    for n, x, y in zip(names, a, b):
        x64, y64 = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
        d = y64 - x64
        g = out.setdefault(groups[n], [0., 0., 0.])
        g[0] += float(np.dot(d, d))
        g[1] += float(np.dot(x64, x64))
        g[2] = max(g[2], float(np.abs(d).max()) if d.size else 0.)
    return {k: {'l2_diff': v[0] ** .5, 'l2_ref': v[1] ** .5, 'max_abs': v[2]} for k, v in out.items()}


def analyse(repo, probe_root, smoke=False):
    import numpy as np
    repo = Path(repo)
    protocol_path = repo / 'outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    design, crit = protocol['design'], protocol['criteria']
    qual = json.loads((repo / 'outputs/audit/E04_PRODUCTION_QUALIFICATION.json').read_text())
    P = {p: load(probe_root, p) for p in PROCESSES}
    S = {p: P[p]['summary'] for p in PROCESSES}
    R = {p: P[p]['records'] for p in PROCESSES}
    n = 3 if smoke else design['iterations']
    gset = [t for t in design['gradient_probe_iterations'] if t < n]
    sset = [t for t in design['snapshot_iterations'] if t <= n]
    pairs = list(itertools.combinations(PROCESSES, 2))
    res, detail = {}, {}

    # ------------------------------------------------------------------ C1 contract identity
    c1 = {'protocol_and_plan': all(S[p]['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                                   and S[p]['smoke'] is smoke and S[p]['iterations'] == n and len(R[p]) == n and
                                   S[p]['gradient_probe_iterations'] == gset and S[p]['snapshot_iterations'] == sset
                                   for p in PROCESSES),
          'construction_equals_qualified': all(S[p]['construction'] == qual['graph']['construction']
                                               for p in PROCESSES),
          'identities_equal': all(S[p]['identities'] == S['P1']['identities'] for p in PROCESSES),
          'initial_state_equal': len({S[p]['initial_state_sha256'] for p in PROCESSES}) == 1,
          'executed_signature_set_equal': len({S[p]['trace']['executed_signature_set_sha256'] for p in PROCESSES}) == 1,
          'firewall_clean': all(not S[p]['firewall']['denied'] and not S[p]['firewall']['runtime_writes']
                                for p in PROCESSES),
          'diagnostics_never_changed_state': all(S[p]['iteration0_diagnostics_state_unchanged'] and
                                                 all(r['diagnostic_fetch_state_unchanged'] for r in R[p])
                                                 for p in PROCESSES)}
    mism = {}
    for f in C1_FIELDS:
        bad = [t for t in range(min(len(R[p]) for p in PROCESSES))
               if any(R[p][t][f] != R['P1'][t][f] for p in PROCESSES)]
        mism[f] = bad[:20]
        c1['per_iteration_' + f] = not bad
    la_bad = []
    for p in PROCESSES:
        for r in R[p]:
            t = r['iteration']
            if 'gradient_probe' in r and t + 1 < n:
                if any(r['gradient_probe']['lookahead_info'][f] != R[p][t + 1][f] for f in LOOKAHEAD_FIELDS):
                    la_bad.append([p, t])
    c1['lookahead_equals_next_batch'] = not la_bad
    c1['adam_application_count'] = all(abs(R[p][t]['counters_after']['iteration'] - (t + 1)) == 0
                                       for p in PROCESSES for t in range(len(R[p])))
    res['C1_contract_identity'] = all(c1.values())
    detail['C1'] = {'checks': c1, 'first_mismatches': mism, 'lookahead_mismatches': la_bad,
                    'executed_order_sha256': {p: S[p]['trace']['executed_order_sha256'] for p in PROCESSES},
                    'executed_name_set_equal_descriptive': len({S[p]['trace']['executed_node_set_sha256']
                                                                for p in PROCESSES}) == 1}

    # ------------------------------------------------------------------ C2 finite
    fin = all(r['state_finite'] and r['losses_finite'] and r['repeat_step1']['concat']['finite'] and
              r['repeat_L_G']['concat']['finite'] and
              all(c['finite'] for k in r.get('gradient_probe', {}).get('repeats', {}).values() for c in k['concat']) and
              all(v['concat']['finite'] for v in r.get('gradient_probe', {}).get('minibatch_reference', {}).values())
              for p in PROCESSES for r in R[p])
    res['C2_finite'] = fin

    # ------------------------------------------------------------------ C3 localization
    r0 = {p: R[p][0] for p in PROCESSES}
    a_losses = all(r0[p]['losses']['step1'] == r0['P1']['losses']['step1'] for p in PROCESSES)
    a_grads = {k: all(r0[p]['gradient_probe']['repeats'][k]['per_variable_sha256_r0'] ==
                      r0['P1']['gradient_probe']['repeats'][k]['per_variable_sha256_r0'] for p in PROCESSES)
               for k in DETERMINISTIC_PROBES}
    lg_cross = sum(len({r0[p]['gradient_probe']['repeats']['L_G']['per_variable_sha256_r0'][i] for p in PROCESSES}) > 1
                   for i in range(len(r0['P1']['gradient_probe']['repeats']['L_G']['per_variable_sha256_r0'])))
    b_bad = [[p, r['iteration'], k] for p in PROCESSES for r in R[p] if 'gradient_probe' in r
             for k in DETERMINISTIC_PROBES if max(r['gradient_probe']['repeats'][k]['distinct_per_variable']) > 1]
    origins, undetermined, varying = {}, [], {}
    for p in PROCESSES:
        loc = S[p]['localization']
        varying[p] = loc['varying_tensors']
        undetermined += [[p, o['op'], o['type']] for o in loc['undetermined_origins']]
        for o in loc['origins']:
            e = origins.setdefault(o['op'], {'type': o['type'], 'requested_device': o['requested_device'],
                                             'executed_device': o['executed_device'], 'processes': []})
            e['processes'].append(p)
    c_types = sorted({o['type'] for o in origins.values()})
    c_ok = (not undetermined and set(c_types) <= set(ALLOWED_ORIGIN_TYPES) and
            all('GPU' in (o['executed_device'] or '').upper() for o in origins.values()) and
            all(bool(S[p]['localization']['origins']) == (S[p]['localization']['varying_tensors'] > 0)
                for p in PROCESSES))
    watched = S['P1']['trace']['watched_ops']
    d_ok = all(sorted(S[p]['trace']['watched_ops']) == S[p]['trace']['watched_type_ops_in_closure'] and
               {w['type'] for w in S[p]['trace']['watched_ops'].values()} >= {'GatherNd', 'ResizeBilinear'} and
               all('CPU' in w['executed_device'].upper() for w in S[p]['trace']['watched_ops'].values()
                   if w['type'] in ('GatherNd', 'ResizeBilinear')) for p in PROCESSES)
    res['C3_localization'] = a_losses and all(a_grads.values()) and not b_bad and c_ok and d_ok
    by_type = {}
    for w in watched.values():
        k = '%s|requested %s|executed %s' % (w['type'], w['requested_device'] or '(none)',
                                             'GPU' if 'GPU' in w['executed_device'].upper() else 'CPU')
        by_type[k] = by_type.get(k, 0) + 1
    detail['C3'] = {'a_iteration0_step1_losses_equal_across_processes': a_losses,
                    'a_iteration0_gradients_equal_across_processes': a_grads,
                    'iteration0_L_G_variables_differing_across_processes': lg_cross,
                    'b_in_process_repeat_differences_outside_L_G': b_bad[:50],
                    'c_origin_types': c_types, 'c_origins': origins, 'c_undetermined': undetermined,
                    'c_varying_tensors': varying,
                    'c_localization_size': {p: {k: S[p]['localization'][k] for k in ('closure_ops', 'chunks',
                                                                                    'fetched_tensors', 'repeats')}
                                            for p in PROCESSES},
                    'c_varying_tensors_by_producer_type': {p: S[p]['localization']['varying_tensors_by_producer_type']
                                                           for p in PROCESSES},
                    'd_forward_warp_and_resize_on_cpu': d_ok, 'watched_ops_by_type_and_device': by_type,
                    'executed_device_by_op_type': S['P1']['trace']['executed_device_by_op_type']}

    # ------------------------------------------------------------------ C4 fixed-state magnitude
    thr4, c4 = crit['C4_fixed_state_magnitude']['threshold'], []
    for p in PROCESSES:
        for r in R[p]:
            if 'gradient_probe' not in r:
                continue
            g = r['gradient_probe']
            for k in ('L_G', 'step1'):
                num = max(c['l2_diff'] for c in g['repeats'][k]['concat'])
                ref = g['minibatch_reference'][k]['concat']['l2_diff']
                c4.append({'process': p, 'iteration': r['iteration'], 'gradient': k, 'numerical_l2': num,
                           'minibatch_l2': ref, 'ratio': num / ref if ref > 0 else None,
                           'numerical_rel_l2': num / g['repeats'][k]['concat'][0]['l2_ref'],
                           'minibatch_rel_l2': ref / g['minibatch_reference'][k]['concat']['l2_ref'],
                           'cosine_min': min(c['cosine'] for c in g['repeats'][k]['concat']),
                           'pass': ref > 0 and num <= thr4 * ref})
    res['C4_fixed_state_magnitude'] = bool(c4) and all(x['pass'] for x in c4)
    detail['C4'] = {'threshold': thr4, 'points': c4,
                    'ratio_quantiles': quantiles([x['ratio'] for x in c4])}

    # ------------------------------------------------------------------ C5 trajectory parameters
    names, groups = S['P1']['variables'], S['P1']['groups']
    snaps = {p: {t: snapshot(P[p], t) for t in sset} for p in PROCESSES}
    c5, desc = [], []
    for t in sset:
        if t == 0:
            continue
        for a, b in pairs:
            dev = group_dev(snaps[a][t], snaps[b][t], names, groups)
            travel = group_dev(snaps[a][0], snaps[a][t], names, groups)
            row = {'iteration': t, 'pair': [a, b], 'groups': {}}
            for g, v in sorted(dev.items()):
                row['groups'][g] = dict(v, normalized=v['l2_diff'] / v['l2_ref'] if v['l2_ref'] > 0 else None,
                                        relative_to_travel=(v['l2_diff'] / travel[g]['l2_diff']
                                                            if travel[g]['l2_diff'] > 0 else None))
            for g in FUNCTIONAL:
                u = [float(np.median([R[p][i]['update_norms'][g] for i in range(max(0, t - 10), t)])) for p in (a, b)]
                row['groups'][g]['update_scale'] = min(u)
                row['groups'][g]['deviation_in_updates'] = dev[g]['l2_diff'] / min(u) if min(u) > 0 else None
                c5.append({'iteration': t, 'pair': [a, b], 'group': g, 'deviation': dev[g]['l2_diff'],
                           'update_scale': min(u), 'pass': dev[g]['l2_diff'] <= min(u)})
            desc.append(row)
    res['C5_trajectory_parameters'] = bool(c5) and all(x['pass'] for x in c5)
    slopes = {}
    for a, b in pairs:
        for g in FUNCTIONAL + ('adam_m_G', 'adam_v_G', 'bn_moving_mean', 'bn_moving_variance', 'G_cancelled_bias'):
            pts = [(r['iteration'], r['groups'][g]['l2_diff']) for r in desc if r['pair'] == [a, b] and
                   r['iteration'] >= 10 and g in r['groups'] and r['groups'][g]['l2_diff'] > 0]
            if len(pts) >= 3:
                x, y = np.log([q[0] for q in pts]), np.log([q[1] for q in pts])
                slopes['%s-%s|%s' % (a, b, g)] = float(np.polyfit(x, y, 1)[0])
    detail['C5'] = {'points': c5, 'snapshots': desc, 'loglog_growth_slope_t_ge_10': slopes,
                    'max_deviation_in_updates': {g: max(x['deviation'] / x['update_scale'] for x in c5
                                                        if x['group'] == g) for g in FUNCTIONAL}}

    # ------------------------------------------------------------------ C6 trajectory losses
    keys = [(s, k) for s in sorted(R['P1'][0]['losses']) for k in sorted(R['P1'][0]['losses'][s])]
    lo, hi = (1, n) if smoke else (900, 1000)
    thr6, c6 = crit['C6_trajectory_losses']['threshold'], []
    for a, b in pairs:
        for s, k in keys:
            diff = [abs(R[a][t]['losses'][s][k] - R[b][t]['losses'][s][k]) for t in range(lo, hi)]
            step = [abs(R[p][t]['losses'][s][k] - R[p][t - 1]['losses'][s][k]) for p in (a, b) for t in range(lo, hi)]
            md, ms = float(np.median(diff)), float(np.median(step))
            c6.append({'pair': [a, b], 'loss': '%s/%s' % (s, k), 'median_abs_diff': md, 'median_iteration_change': ms,
                       'ratio': md / ms if ms > 0 else None, 'pass': (md <= thr6 * ms) if ms > 0 else md == 0})
    res['C6_trajectory_losses'] = bool(c6) and all(x['pass'] for x in c6)
    curve = []
    for t in sorted(set(sset[1:]) | {0}):
        t = min(t, n - 1)
        rel = []
        for a, b in pairs:
            for s, k in keys:
                x, y = R[a][t]['losses'][s][k], R[b][t]['losses'][s][k]
                rel.append(abs(x - y) / max(abs(x), 1e-30))
        curve.append({'iteration': t, 'max_relative_loss_difference': max(rel),
                      'median_relative_loss_difference': float(np.median(rel)),
                      'identical_losses': sum(r == 0 for r in rel), 'compared': len(rel)})
    detail['C6'] = {'threshold': thr6, 'window': [lo, hi], 'points': c6, 'loss_difference_curve': curve,
                    'first_iteration_with_any_loss_difference': next(
                        (t for t in range(n) if any(R[a][t]['losses'] != R[b][t]['losses'] for a, b in pairs)), None)}

    # ------------------------------------------------------------------ descriptive gradient error distributions
    cancelled = set(S['P1']['bn_cancelled_biases'])
    dist = {}
    for k in ('L_G', 'step1'):
        vn = S['P1']['probe_variables'][k]
        func_rel, canc_abs, canc_ref, scale_aware, ulp, cos, concat_rel, concat_cos, varying = \
            [], [], [], [], [], [], [], [], []
        for p in PROCESSES:
            for r in R[p]:
                c = r['repeat_' + k]['concat']
                concat_rel.append(c['l2_diff'] / c['l2_ref'] if c['l2_ref'] > 0 else None)
                concat_cos.append(c['cosine'])
                varying.append(c['variables'] - c['identical_variables'])
                for name, v in zip(vn, r['repeat_' + k]['per_variable']):
                    if name in cancelled:
                        canc_abs.append(v['l2_diff'])
                        canc_ref.append(v['l2_ref'])
                        continue
                    func_rel.append(v['l2_diff'] / v['l2_ref'] if v['l2_ref'] > 0 else None)
                    scale_aware.append(v['max_abs'] / v['ref_abs_max'] if v['ref_abs_max'] > 0 else None)
                    ulp.append(v['max_ulp'])
                    if not v['identical']:
                        cos.append(v['cosine'])
        dist[k] = {'variables': len(vn), 'bn_cancelled_variables': sum(x in cancelled for x in vn),
                   'concat_relative_l2': quantiles(concat_rel), 'concat_cosine': quantiles(concat_cos),
                   'varying_variables_per_iteration': quantiles(varying),
                   'functional_per_variable_relative_l2': quantiles(func_rel),
                   'functional_per_variable_scale_aware_max_abs': quantiles(scale_aware),
                   'functional_per_variable_max_ulp': quantiles(ulp),
                   'functional_per_variable_cosine_when_differing': quantiles(cos),
                   'bn_cancelled_bias_abs_l2_diff': quantiles(canc_abs),
                   'bn_cancelled_bias_reference_l2': quantiles(canc_ref)}
    cross = []
    for t in gset:
        z = {p: np.load(str(P[p]['dir'] / ('grad_%04d.npz' % t))) for p in PROCESSES}
        for k in ('L_G', 'step1'):
            m = len(S['P1']['probe_variables'][k])
            for a, b in pairs:
                d2 = sum(float(np.sum((z[a]['%s_%03d' % (k, i)].astype(np.float64) -
                                       z[b]['%s_%03d' % (k, i)].astype(np.float64)) ** 2)) for i in range(m))
                r2 = sum(float(np.sum(z[a]['%s_%03d' % (k, i)].astype(np.float64) ** 2)) for i in range(m))
                cross.append({'iteration': t, 'gradient': k, 'pair': [a, b], 'relative_l2': (d2 / r2) ** .5})
    detail['gradient_distributions'] = dist
    detail['cross_process_gradient_on_own_trajectory'] = cross

    stop = not (res['C1_contract_identity'] and res['C2_finite'] and res['C3_localization'])
    decision = 'STOP_AND_REPORT' if stop else ('D10_SUPPORTED' if all(res.values()) else 'DECISION_REPORT_NOT_SUPPORTED')
    if smoke:
        decision = 'SMOKE_NOT_INTERPRETABLE'
    return {'schema_version': 'e04_d10_analysis_v1', 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes())
            .hexdigest(), 'criteria': res, 'decision': decision, 'detail': detail,
            'processes': {p: {k: S[p][k] for k in ('pid', 'start_utc', 'end_utc', 'seconds', 'firewall',
                                                   'initial_state_sha256', 'construction', 'driver')}
                          for p in PROCESSES},
            'bn_cancelled_biases': sorted(cancelled),
            'seconds_per_iteration_production_step': {p: float(np.median([r['iteration_seconds'] for r in R[p]]))
                                                      for p in PROCESSES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--probe-root', required=True)
    ap.add_argument('--smoke', action='store_true', help='code-path check on --smoke probes; never interpreted')
    args = ap.parse_args()
    doc = analyse(args.repo, args.probe_root, args.smoke)
    json.dump(doc, sys.stdout, sort_keys=True)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
