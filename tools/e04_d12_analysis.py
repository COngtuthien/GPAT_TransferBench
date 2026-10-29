"""E04 D12 investigation analysis: evaluates the criteria of outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json on the
raw outputs (authority.json, P/<process>/, C/<factor>/<process>/). Stdlib only; prints JSON.

    python -B tools/e04_d12_analysis.py --repo <checkout> --root /tmp/gpat_e04_d12
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e04_d11_analysis as d11a  # noqa: E402

PROTOCOL = 'outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json'
FACTORS = ('plain', 'd10', 'd10_trace2', 'd11_wrapper')
C_PROCESSES = ('P1', 'P2', 'P3', 'P4')
P_PROCESSES = ('P1', 'P2', 'P3')
BOUND = 1e-3                          # relative L2 integrity bound (forward tensors; functional-variable gradients)
QUALIFIED_CONSTRUCTION = ['61f1ed5868d6774abb809915c2e9dfd0a50c5313064672389033ca46dd951e1d', 11315]
INITIAL_STATE = '951dad42'
D11_IDENTITIES_SHA256 = 'cc77c431600f126af9c0877a0f6590dee816cc1dc09e5c175e40ec0298428291'
D11_ORIGINAL_CANONICAL_SET = '094566e10e620cf0ebf0b0586d5509359e2e781c8fcce7040eb36ffbf90816aa'
D11_ORIGINAL_RETVAL = 'fea3730eca5675fbddd01a9f165a5bfca4aa93a513d6b5bb768bfd0d299b9395'


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def ident_sha(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def cancelled(names):
    s = set(names)
    return {n for n in s if n.endswith('/bias:0') and n[:-len('bias:0')] + 'BN/beta:0' in s}


def bound_ok(row):
    if row.get('l2_ref') is None:                   # non-float tensor that differs
        return False
    return row.get('finite', False) and (row['l2_diff'] <= BOUND * row['l2_ref'] if row['l2_ref'] > 0 else False)


def forward_violations(rows):
    out = []
    for r in rows:
        if r['gradient'] or not r['forward']:
            continue
        why = []
        if r['type'] == 'Placeholder':
            why.append('fed input varies')
        if r['data_movement'] and r['chunk_local_root']:
            why.append('data-movement op varies with identical inputs in the same call')
        if not bound_ok(r):
            why.append('relative L2 above %g or non-finite' % BOUND)
        if why:
            out.append({'tensor': r['tensor'], 'type': r['type'], 'why': why,
                        'l2_diff': r.get('l2_diff'), 'l2_ref': r.get('l2_ref'), 'max_ulp': r.get('max_ulp')})
    return out


def analyse(repo, root):
    root, repo = Path(root), Path(repo)
    proto_sha = hashlib.sha256((repo / PROTOCOL).read_bytes()).hexdigest()
    crit, detail = {}, {}
    # ------------------------------------------------------------------ A: authority
    auth = json.loads((root / 'authority.json').read_text())
    a = {'verify_contract': auth['verify_contract']['ok'],
         'identities_equal_d11_record': auth['identities_sha256'] == D11_IDENTITIES_SHA256,
         'prepass': auth['prepass']['ok'] and auth['prepass']['samples'] == 12668 and auth['prepass']['all_train'],
         'feed_pack': auth['feed_pack']['ok'] and auth['feed_pack']['binds_prepass'],
         'construction_qualified': auth['production_construction'] == QUALIFIED_CONSTRUCTION and
         auth['construction_equals_reference'] and auth['prepare_identities_equal']}
    p_runs, c_runs = {}, {}
    for p in P_PROCESSES:
        d = root / 'P' / p
        p_runs[p] = {'summary': json.loads((d / 'summary.json').read_text()), 'iterations': jl(d / 'iterations.jsonl')}
    for f in FACTORS:
        for p in C_PROCESSES:
            d = root / 'C' / f / p
            c_runs[(f, p)] = {'summary': json.loads((d / 'summary.json').read_text()),
                              'extra': json.loads((d / 'd12_extra.json').read_text()), 'dir': d}
    sums = [r['summary'] for r in p_runs.values()] + [r['summary'] for r in c_runs.values()]
    a['every_process_identities'] = all(ident_sha(s['identities']) == D11_IDENTITIES_SHA256 for s in sums)
    a['every_process_construction'] = all(r['summary']['construction'] == QUALIFIED_CONSTRUCTION
                                          for r in c_runs.values()) and \
        all(r['summary']['static']['construction'] == QUALIFIED_CONSTRUCTION for r in p_runs.values())
    a['initial_state_identical'] = len({s['initial_state_sha256'] for s in sums}) == 1 and \
        sums[0]['initial_state_sha256'].startswith(INITIAL_STATE)
    a['firewall_clean'] = all(s['firewall']['denied'] == [] and s['firewall']['runtime_writes'] == [] for s in sums)
    a['protocol_and_tools'] = all(r['extra']['protocol_sha256'] == proto_sha for r in c_runs.values()) and \
        all(r['summary']['protocol_sha256'] == proto_sha for r in p_runs.values())
    detail['A'] = a
    a_proxy = all(r['extra']['localization']['varying_tensors'] ==
                  r['extra']['localization']['reported_by_d10_algorithm']['varying_tensors'] for r in c_runs.values())
    complete = all(r['extra']['rc'] == 0 for r in c_runs.values()) and \
        all(len(r['iterations']) == r['summary']['iterations'] for r in p_runs.values())
    # ------------------------------------------------------------------ B: production-path semantics (stage P)
    b = {'connectivity': True, 'optimizer_application': True, 'no_random_ops_on_step_paths': True,
         'restore_exact': True, 'scope': True, 'adam': True, 'beta_powers': True, 'counters': True, 'finite': True,
         'diagnostics_non_mutating': True, 'graph_unchanged_by_session': True}
    names = None
    for p, r in p_runs.items():
        st = r['summary']['static']
        b['connectivity'] &= all(v['present_equals_reachable'] and v['all_present_finite_shape']
                                 for v in st['gradient_connectivity'].values())
        for s, v in st['optimizer_application'].items():
            b['optimizer_application'] &= v['apply_adam_ops'] == v['present_gradients'] and v['targets_equal'] and \
                v['apply_inputs_are_step_gradients']
            b['no_random_ops_on_step_paths'] &= v['random_ops_on_path'] == [] and v['random_ops_on_bn_path'] == []
        b['graph_unchanged_by_session'] &= r['summary']['graph_operations_before_session'] == \
            r['summary']['graph_operations_after_session']
        for it in r['iterations']:
            b['diagnostics_non_mutating'] &= it['forward_state_unchanged']
            b['finite'] &= it['state_finite'] and all(x.get('finite', True) for x in it['forward']['varying'])
            for s, v in it['steps'].items():
                b['restore_exact'] &= v['production_restore_exact'] and v['semantic_restore_exact']
                for x in v['semantic']:
                    b['scope'] &= x['scope_ok']
                    b['adam'] &= x['adam_ok']
                    b['beta_powers'] &= x['beta_powers_ok']
                    b['counters'] &= x['counters_ok']
                    b['finite'] &= x['finite']
                if names is None:
                    names = list(v['gradient_repeat']['per_variable_l2_ref'])
    detail['B'] = dict(b, proxy_count_equals_d10_algorithm=a_proxy)
    # ------------------------------------------------------------------ C: integrity bounds
    c = {'production_forward_violations': [], 'production_gradient_violations': [], 'confound_forward_violations': []}
    prod_forward, grad_rows, step_var = {}, [], {}
    for p, r in p_runs.items():
        for it in r['iterations']:
            rows = it['forward']['varying']
            prod_forward['%s/t%d' % (p, it['iteration'])] = sorted({x['tensor'] for x in rows if not x['gradient']})
            c['production_forward_violations'] += [dict(v, where='%s/t%d' % (p, it['iteration']))
                                                   for v in forward_violations(rows)]
            for s, v in it['steps'].items():
                g = v['gradient_repeat']
                canc = cancelled(g['per_variable_l2_ref'])
                ref = sum(x ** 2 for n, x in g['per_variable_l2_ref'].items() if n not in canc) ** .5
                dif = sum(x['l2_diff'] ** 2 for x in g['differing_variables'] if x['variable'] not in canc) ** .5
                grad_rows.append({'where': '%s/t%d/%s' % (p, it['iteration'], s), 'functional_l2_ref': ref,
                                  'functional_l2_diff': dif, 'ratio': dif / ref if ref > 0 else None,
                                  'cancelled_bias_differing': sum(x['variable'] in canc
                                                                  for x in g['differing_variables']),
                                  'functional_differing': sum(x['variable'] not in canc
                                                              for x in g['differing_variables'])})
                if not (ref > 0 and dif <= BOUND * ref):
                    c['production_gradient_violations'].append(grad_rows[-1])
                step_var.setdefault(s, []).append({
                    'where': '%s/t%d' % (p, it['iteration']),
                    'replay_post_states_distinct': len(set(v['production_post_sha256'] + v['semantic_post_sha256'])),
                    'variables_differing': len(v['variables_differing_across_replays']),
                    'D3_variables_differing': sum('/D3/' in n for n in v['variables_differing_across_replays']),
                    'losses_identical_across_replays': len({json.dumps(x) for x in v['production_losses'] +
                                                            v['semantic_losses']}) == 1})
    conf = {}
    for (f, p), r in sorted(c_runs.items()):
        loc = r['extra']['localization']
        rows = loc['varying']
        fwd = [x for x in rows if x['forward'] and not x['gradient']]
        conf['%s/%s' % (f, p)] = {
            'traced_calls_before_localization': r['extra']['traced_calls_before_localization'],
            'varying_tensors': loc['varying_tensors'], 'forward_varying_tensors': len({x['tensor'] for x in fwd}),
            'forward_varying_by_type': {t: len({x['tensor'] for x in fwd if x['type'] == t})
                                        for t in sorted({x['type'] for x in fwd})},
            'forward_varying_scopes': sorted({'/'.join(x['op'].split('/')[:2]) for x in fwd}),
            'forward_max_relative_l2': max([x['l2_diff'] / x['l2_ref'] for x in fwd if x.get('l2_ref')], default=0.),
            'forward_max_ulp': max([x.get('max_ulp', 0) for x in fwd], default=0),
            'd10_origins': len(loc['d10_origins']), 'd10_origin_types': sorted({o.split('/')[-1] for o in
                                                                                loc['d10_origins']})}
        c['confound_forward_violations'] += [dict(v, where='%s/%s' % (f, p)) for v in forward_violations(rows)]
    detail['C'] = {k: v[:200] for k, v in c.items()}
    detail['C']['production_gradient_repeat'] = grad_rows
    detail['production_step_replay'] = step_var
    detail['production_forward_varying'] = prod_forward
    detail['confound'] = conf
    by_factor = {f: sum(conf['%s/%s' % (f, p)]['forward_varying_tensors'] > 0 for p in C_PROCESSES) for f in FACTORS}
    detail['confound_processes_with_forward_variation'] = by_factor
    # ------------------------------------------------------------------ E: graph identity and runtime rewrites
    e = {'placement_consistent': True, 'canonical_partitions_equal_d11_original': True}
    ref_place, placement_diffs = None, []
    for (f, p), r in sorted(c_runs.items()):
        place = r['extra'].get('first_trace_place')
        if f == 'd11_wrapper':
            x = json.loads((r['dir'] / 'd11_extra.json').read_text())
            place = x['trace']['first_trace_place']
        if place is None:
            continue
        dt = {n: d11a.dev_type(d) for n, d in place.items()}
        if ref_place is None:
            ref_place = dt
            continue
        diff = sorted(n for n in set(dt) & set(ref_place) if dt[n] != ref_place[n])
        if diff:
            placement_diffs.append({'process': '%s/%s' % (f, p), 'ops': diff[:20], 'count': len(diff)})
    e['placement_consistent'] = not placement_diffs
    canon = []
    for (f, p), r in sorted(c_runs.items()):
        if f not in ('d10_trace2', 'd11_wrapper'):
            continue
        if f == 'd11_wrapper':
            parts = json.loads((r['dir'] / 'd11_extra.json').read_text())['trace']['partitions']
        else:
            parts = r['extra']['partitions']
        memo, nodes = d11a.canonical(parts)
        sset = hashlib.sha256('\n'.join(sorted(set(memo.values()))).encode()).hexdigest()
        retval = hashlib.sha256('\n'.join(sorted(memo[n] for n, v in nodes.items() if v['op'] == '_Retval'))
                                .encode()).hexdigest()
        canon.append({'process': '%s/%s' % (f, p), 'canonical_set_sha256': sset, 'retval_canonical_sha256': retval})
    e['canonical_partitions_equal_d11_original'] = bool(canon) and all(
        x['canonical_set_sha256'] == D11_ORIGINAL_CANONICAL_SET and x['retval_canonical_sha256'] == D11_ORIGINAL_RETVAL
        for x in canon)
    detail['E'] = {'checks': e, 'placement_differences': placement_diffs, 'canonical': canon}
    # ------------------------------------------------------------------ criteria and decision
    crit['A_authority'] = all(a.values())
    crit['completion'] = complete
    crit['B_restore_exact'] = b['restore_exact'] and a_proxy
    crit['B_production_semantics'] = all(v for k, v in b.items() if k != 'restore_exact')
    crit['C1_production_forward_integrity'] = not c['production_forward_violations']
    crit['C2_production_gradient_integrity'] = not c['production_gradient_violations']
    crit['C3_confound_forward_integrity'] = not c['confound_forward_violations']
    crit['D1_confound_reproduced'] = by_factor['d11_wrapper'] > 0
    crit['E_graph_and_placement'] = all(e.values())
    if not (crit['A_authority'] and crit['completion'] and crit['B_restore_exact'] and crit['D1_confound_reproduced']):
        decision = 'E04_D12_INCONCLUSIVE'
    elif not all(crit[k] for k in ('B_production_semantics', 'C1_production_forward_integrity',
                                   'C2_production_gradient_integrity', 'C3_confound_forward_integrity',
                                   'E_graph_and_placement')):
        decision = 'E04_D12_BLOCKED_BY_SEMANTIC_DEFECT'
    else:
        decision = 'NO_SEMANTIC_DEFECT_FOUND'
    return {'schema_version': 'e04_d12_analysis_v1', 'protocol_sha256': proto_sha, 'criteria': crit,
            'decision': decision, 'detail': detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    json.dump(analyse(args.repo, args.root), sys.stdout, sort_keys=True)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
