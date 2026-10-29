"""E04 Physics-STD production-qualification evidence (qualification seed 60904; never a scientific result).

  collect-atime (GPU host, gpat-m6-e05 with pyarrow, `python -I -B - collect-atime < this file`): classifies the
                stat-only faces_256 atime snapshots ~/workdir/e04_prodq_atime_{pre,post}.tsv by split_v1 (no image is
                opened) and prints one JSON document.
  build         (repository): gates the collected documents in --collected (pre-pass A/B comparison and frozen
                manifest, feed pack verification on the GPU host and here, the det_a / det_b / terminal_probe /
                failure_probe qualification outputs, the D9 compatibility re-probe, the atime classification) and
                writes outputs/audit/E04_PRODUCTION_QUALIFICATION.{json,md} + E04_PRODUCTION_QUALIFICATION_SHA256.txt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

STATUS_OK, STATUS_FAIL = 'E04_PRODUCTION_QUALIFIED', 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE'
RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
REPO_GPU = '/home/student20261/workdir/GPAT_TransferBench'
OUT = {'record': 'outputs/audit/E04_PRODUCTION_QUALIFICATION.json',
       'report': 'outputs/audit/E04_PRODUCTION_QUALIFICATION.md',
       'sha256': 'outputs/audit/E04_PRODUCTION_QUALIFICATION_SHA256.txt'}
CASES = ('det_a', 'det_b', 'terminal_probe', 'failure_probe')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ============================================================================ collect-atime (GPU host)
def collect_atime(post_name='post'):
    from datetime import datetime, timezone
    import pyarrow.parquet as pq
    w = Path('/home/student20261/workdir')
    split = {r['sample_id']: r['split'] for r in pq.read_table(REPO_GPU + '/manifests/split_v1.parquet',
                                                             columns=['sample_id', 'split']).to_pylist()}

    def load(name):
        out = {}
        for line in (w / name).read_text().splitlines():
            a, m, p = line.split('\t')
            out[p] = (float(a), float(m))
        return out
    t = {k: datetime.strptime((w / ('e04_prodq_atime_%s.time' % k)).read_text().strip(), '%Y-%m-%dT%H:%M:%SZ')
         .replace(tzinfo=timezone.utc).timestamp() for k in ('pre', post_name)}
    pre, post = load('e04_prodq_atime_pre.tsv'), load('e04_prodq_atime_%s.tsv' % post_name)
    stats, manifests = {}, []
    for p, (a, m) in sorted(pre.items()):
        if '/manifests/' in p:
            manifests.append({'file': os.path.basename(p), 'mtime_unchanged': post.get(p, (0, None))[1] == m,
                              'atime_changed': post.get(p, (None, 0))[0] != a})
            continue
        sp = split.get(os.path.basename(p)[:-4], 'UNMAPPED')
        s = stats.setdefault(sp, {'files': 0, 'sensitive': 0, 'atime_changed': 0, 'mtime_changed': 0, 'missing': 0})
        s['files'] += 1
        s['sensitive'] += (a <= m) or (t['pre'] - a >= 86400)
        if p not in post:
            s['missing'] += 1
            continue
        s['atime_changed'] += post[p][0] != a
        s['mtime_changed'] += post[p][1] != m
    return {'pre_utc': (w / 'e04_prodq_atime_pre.time').read_text().strip(),
            'post_name': post_name, 'post_utc': (w / ('e04_prodq_atime_%s.time' % post_name)).read_text().strip(),
            'pre_sha256': sha256_bytes((w / 'e04_prodq_atime_pre.tsv').read_bytes()),
            'post_sha256': sha256_bytes((w / ('e04_prodq_atime_%s.tsv' % post_name)).read_bytes()),
            'files': len(pre), 'same_paths': sorted(pre) == sorted(post), 'by_split': stats, 'manifests': manifests}


# ============================================================================ build (repository)
class Gates:
    def __init__(self):
        self.results = []

    def __call__(self, name, ok, detail=None):
        self.results.append(dict({'gate': name, 'pass': bool(ok)}, **({'detail': detail} if detail is not None else {})))

    @property
    def failed(self):
        return [r['gate'] for r in self.results if not r['pass']]


def crop_summary(samples):
    out = {}
    for r in samples:
        k = '%s|%s' % (r['dataset'], r['role'])
        s = out.setdefault(k, {'faces': 0, 'outside_any': 0, 'outside_left': 0, 'outside_top': 0, 'outside_right': 0,
                               'outside_bottom': 0, 'padded_pixels': []})
        s['faces'] += 1
        s['outside_any'] += any(r['crop_outside_frame'])
        for name, flag in zip(('left', 'top', 'right', 'bottom'), r['crop_outside_frame']):
            s['outside_' + name] += flag
        s['padded_pixels'].append(r['crop_zero_padded_pixels'])
    for s in out.values():
        p = sorted(s.pop('padded_pixels'))
        s.update(padded_pixels_min=p[0], padded_pixels_median=p[len(p) // 2], padded_pixels_max=p[-1],
                 padded_faces=sum(x > 0 for x in p))
    return out


def build(root, collected):
    root, col = Path(root), Path(collected)
    sys.path.insert(0, str(root))
    from methods.physics_std import production_io as pio
    from methods.physics_std import feed_pack
    G = Gates()
    load = lambda n: json.loads((col / n).read_text())  # noqa: E731
    q = {c: load('%s.json' % c) for c in CASES}
    cmp_, frozen, pre_samples = load('prepass_compare.json'), load('prepass_manifest.json'), load('prepass_samples.json')
    pack_gpu, compat, atime = load('feed_pack_verify_gpu.json'), load('compat_probe.json'), load('atime.json')

    # A. authority
    cfg = pio.verify_contract(root)
    G('A.frozen_contract_and_owner_resolution', cfg['method_id'] == 'E04')
    G('A.pinned_manifests', pio.PREPASS_MANIFEST_SHA256 == sha256_bytes((col / 'prepass_manifest.json').read_bytes())
      and pio.PACK_MANIFEST_SHA256 == pack_gpu['manifest_sha256'])
    ids = q['det_a']['identities']
    G('A.same_production_identity_all_cases', all(q[c]['identities'] == ids for c in CASES) and
      ids['production_files_sha256'] == pio.production_file_hashes(root), ids['production_files_sha256'])
    G('A.qualification_seed_only', all(q[c]['qualification_seed'] == pio.QUALIFICATION_SEED and
                                       q[c]['experiment_seed'] is None for c in CASES))

    # B. geometry pre-pass (D1-D3)
    G('B.prepass_repeat_byte_identical', cmp_['identical'] and not cmp_['differing'], cmp_['differing'])
    G('B.prepass_population', frozen['acceptable'] and frozen['counts'] == {'samples': 12668, 'live': 3830,
                                                                            'spoof': 8838, 'VAL': 0, 'TEST': 0})
    G('B.prepass_cpu_qualified_settings', frozen['runtime']['device'] == 'cpu' and
      frozen['runtime']['torch_num_threads'] == 1 and frozen['runtime']['GPAT_E04_GEOMETRY_RUNTIME'] is not None)
    G('B.prepass_firewall', frozen['firewall'] == {'denied': [], 'face_reads': 12668})
    G('B.prepass_samples', len(pre_samples) == 12668 and all(r['split'] == 'TRAIN' for r in pre_samples) and
      all((r['m0_sha256'] is None) == (r['role'] == 'spoof') for r in pre_samples) and
      all(r['mask_pixels'] > 0 for r in pre_samples if r['role'] == 'live'))
    crops = crop_summary(pre_samples)

    # C. feed pack
    lap = feed_pack.load(col / 'e04_feed_pack_v1', pio.PACK_MANIFEST_SHA256)
    G('C.feed_pack_verified_gpu_and_laptop', pack_gpu['verified'] and lap.manifest_sha256 == pack_gpu['manifest_sha256']
      and lap.manifest['payloads'] == pack_gpu['payloads'] and lap.manifest['test_rows'] == 0)
    G('C.feed_pack_binds_prepass', lap.manifest['prepass_manifest_sha256'] == pio.PREPASS_MANIFEST_SHA256)

    # D. RNG (D4/D7): logged draws equal an independent replay; det_a == det_b
    pair, hard = pio.PairStream(pio.QUALIFICATION_SEED), pio.HardStream(pio.QUALIFICATION_SEED)
    replay = []
    for _ in range(20):
        rows, passes = pair.next_rows()
        s, k = hard.next()
        replay.append((rows, passes, [float(x) for x in s.ravel()], [float(x) for x in k.ravel()]))
    for c in ('det_a', 'det_b'):
        logged = [(r['pair_rows'], r['pair_passes'], r['hard_scales'], r['hard_keep']) for r in q[c]['per_iteration']]
        G('D.%s_rows_and_hard_draws_equal_replay' % c, logged == replay)
    G('D.det_inputs_identical', [r['input_sha256'] for r in q['det_a']['per_iteration']] ==
      [r['input_sha256'] for r in q['det_b']['per_iteration']])
    for s in (42, 1337, 2026):
        v = json.loads((root / 'outputs/audit/E04_RNG_TEST_VECTORS.json').read_text())['seeds'][str(s)]
        h = pio.stream_hashes(s)
        G('D.seed%d_full_streams_equal_approved_vectors' % s, (h['pair_sha256_int32_le'], h['hard_sha256']) ==
          (v['pair_stream']['sha256_int32_le'], v['hard_stream']['sha256_float32_scales_then_keep_per_iteration']))

    # E. graph identity
    ref, prod = q['det_a']['reference_construction'], q['det_a']['production_construction']
    G('E.construction_equals_qualified_m6d3e', ref == prod and prod[1] == 11315, [ref, prod])
    G('E.graph_inventory', all(q[c]['structure'] == q['det_a']['structure'] for c in CASES) and
      q['det_a']['structure']['trainable_parameters'] == 1763902)
    G('E.initial_state_identical', q['det_a']['initial_state_sha256'] == q['det_b']['initial_state_sha256'])

    # F. state transitions / optimizer order / BN (iteration 0 vs committed M6D3e)
    for c in ('det_a', 'det_b'):
        o = q[c]['observed_iteration0']
        G('F.%s_step_transitions_equal_m6d3e' % c, len(o['steps']) == 3 and all(
            s['equals_m6d3e_changed_trainables'] and s['equals_m6d3e_changed_nontrainables'] and
            s['phase_after'] == s['expected_phase_after'] and s['all_finite'] for s in o['steps']) and
          [s['iteration_before_after'] for s in o['steps']] == [[0, 0], [0, 0], [0, 1]])
        it0 = q[c]['per_iteration'][0]['losses']
        G('F.%s_pure_fetch_nonmutating_and_equal' % c, o['pure_fetch_mutated'] == [] and o['pure_losses'] == it0)
    G('F.iteration0_losses_identical_across_processes', q['det_a']['per_iteration'][0]['losses'] ==
      q['det_b']['per_iteration'][0]['losses'])

    # G. numerics
    finite = all(isinstance(v, float) and v == v and abs(v) != float('inf') for c in CASES
                 for r in q[c]['per_iteration'] for st in r['losses'].values() for v in st.values())
    G('G.losses_finite', finite)
    divergence = [sum(r['losses'][s][k] != r2['losses'][s][k] for s in r['losses'] for k in r['losses'][s])
                  for r, r2 in zip(q['det_a']['per_iteration'], q['det_b']['per_iteration'])]

    # H. checkpoints / lifecycle
    for c in ('det_a', 'det_b'):
        ev = q[c]['run']['checkpoints']
        G('H.%s_periodic_checkpoints' % c, [e['iteration'] for e in ev] == [10, 20] and all(
            e['state_unchanged_by_save'] and e['round_trip_equal'] == e['variables'] and not e['selected_for_final']
            and e['checkpoint_type'] == 'periodic' for e in ev) and q[c]['status'] == 'completed')
    tp = q['terminal_probe']
    e = tp['run']['checkpoints']
    G('H.terminal_probe_selected_semantics', len(e) == 1 and e[0]['selected_for_final'] and
      e[0]['label_iteration'] == 150000 and e[0]['iteration'] == 2 and pio.TERMINAL_PROBE_LABEL in e[0]['labels'] and
      e[0]['prefix'] == 'checkpoints/iteration_150000' and
      sum(r['selected_for_final'] for r in tp['checkpoint_index']['checkpoints']) == 3)
    fp = q['failure_probe']
    G('H.failure_probe', fp['status'] == 'failed' and 'QUALIFICATION_INJECTED_FAILURE' in fp['injected_failure'] and
      fp['run_summary']['completion_status'] == 'failed' and fp['checkpoints_dir_listing'] == [] and
      fp['reopen'].startswith('REFUSED'))

    # I. logging contract
    for c in CASES:
        n = len(q[c]['per_iteration'])
        G('I.%s_summary_and_records' % c, q[c]['run_summary']['records_written'] == n +
          len(q[c].get('run', {}).get('checkpoints', [])) + 1 and q[c]['run_summary']['test_split_accessed'] is False)

    # J. TEST isolation
    G('J.firewall_all_cases', all(q[c]['firewall']['denied'] == [] and
                                  q[c]['firewall']['runtime_writes_outside_qualification'] == [] for c in CASES))
    t = atime['by_split'].get('TEST', {})
    G('J.atime_test_unchanged', atime['same_paths'] and t.get('files') == 3027 and t.get('atime_changed') == 0 and
      t.get('sensitive') == 3027 and all(v['mtime_changed'] == v['missing'] == 0 for v in atime['by_split'].values())
      and 'UNMAPPED' not in atime['by_split'], atime['by_split'])
    G('J.manifests_unmodified', all(m['mtime_unchanged'] for m in atime['manifests']))
    t2 = load('atime2.json')['by_split'].get('TEST', {})
    G('J.atime_test_unchanged_after_diagnostics', t2.get('files') == 3027 and t2.get('atime_changed') == 0 and
      t2.get('sensitive') == 3027, t2)
    G('J.no_scientific_run_root', not load('host_state.json')['e04_run_root_exists'])

    # K. runtime / D9
    G('K.environment_lock_except_driver', all(q[c]['driver']['observed_driver_version'] == pio.OBSERVED_DRIVER and
                                              q[c]['driver']['lock_driver_version'] == pio.LOCK_DRIVER for c in CASES))
    variants = [json.loads((root / ('outputs/audit/E04_D9_BN_PROBE_%s.json' % v)).read_text())['post_state_sha256']
                for v in 'AB']
    # K3. real-data gradient repeatability (D9 condition 6: any new gradient discrepancy fails closed)
    rdp = [load('rdp_%d.json' % k) for k in (1, 2, 3)]
    lg = load('lg_probe.json')
    grad_findings = {}
    for k in sorted(rdp[0]['iterations'][0]['gradients']):
        g = [p['iterations'][0]['gradients'][k] for p in rdp]
        grad_findings[k] = {
            'variables': len(g[0]['per_variable_sha256']),
            'in_process_repeat_differs': max(len(x['in_process_repeat_differs']) for x in g),
            'cross_process_differs': sum(len({x['per_variable_sha256'][n] for x in g}) > 1
                                         for n in g[0]['per_variable_sha256'])}
    nondet = sorted(k for k, v in grad_findings.items() if v['in_process_repeat_differs'] or v['cross_process_differs'])
    rels = sorted(v['rel_l2_max'] for v in lg['per_variable'].values())
    gradient_detail = {
        'iteration0_identical_state_and_inputs': len({p['initial_state_sha256'] for p in rdp}) == 1 and all(
            p['iterations'][0]['input_sha256'] == rdp[0]['iterations'][0]['input_sha256'] for p in rdp),
        'per_loss': grad_findings, 'non_repeatable_losses': nondet,
        'gpu_ops_only_on_LG_backward_path': lg['those_ops_placed_on_gpu_by_type'],
        'known_nondeterministic_gpu_ops': {k: lg['those_ops_placed_on_gpu_by_type'].get(k) for k in
                                           ('ResizeBilinearGrad', 'ScatterNd')},
        'L_G_rel_l2_median': rels[len(rels) // 2], 'L_G_rel_l2_max': rels[-1], 'L_G_max_abs_diff': lg['max_abs_diff'],
        'note': ('relative differences near 1 occur only for biases of convolutions followed by training-mode BN, '
                 'whose exact gradient is zero (rounding noise ~1e-6); substantive tensors differ at float32 '
                 'rounding level. M6D3e (synthetic) recorded bit-repeatable gradients for every loss.')}
    G('K.real_data_gradient_repeatability', gradient_detail['iteration0_identical_state_and_inputs'] and not nondet,
      gradient_detail)
    G('K.d9_compatibility_reprobe_matches_accepted_variant', compat['status'] == 'PASS' and
      compat['graph_def_sha256'] == pio.QUALIFIED_WORKER_GRAPH_SHA and any(compat['post_state_sha256'] == v
                                                                         for v in variants))

    seconds = [r for r in (q['det_a']['run']['seconds'] / 20, q['det_b']['run']['seconds'] / 20)]
    failed = G.failed
    doc = {'milestone': 'E04-PRODUCTION-QUALIFICATION', 'status': STATUS_FAIL if failed else STATUS_OK,
           'failed_gates': failed, 'gates': G.results, 'gates_passed': len(G.results) - len(failed),
           'gates_total': len(G.results), 'qualification_seed': pio.QUALIFICATION_SEED,
           'scientific_seeds_launched': [], 'e04_scientific_authority': None,
           'prepass': {'manifest_sha256': pio.PREPASS_MANIFEST_SHA256, 'counts': frozen['counts'],
                       'payloads': frozen['payloads'], 'seconds': frozen['seconds'], 'runtime': frozen['runtime'],
                       'geometry': {k: v for k, v in frozen['geometry'].items() if k != 'regressor_checkpoint_keys'},
                       'compare': cmp_, 'crop_boundary': crops},
           'feed_pack': {'manifest_sha256': pio.PACK_MANIFEST_SHA256, 'counts': lap.manifest['counts'],
                         'payloads': lap.manifest['payloads']},
           'graph': {'construction': prod, 'structure': q['det_a']['structure'],
                     'qualified_worker_graph_sha256': pio.QUALIFIED_WORKER_GRAPH_SHA},
           'determinism': {'per_iteration_loss_values_differing_det_a_vs_det_b': divergence,
                           'note': 'data, RNG draws, input tensors, initial state and iteration-0 losses are '
                                   'identical across processes; later loss values may differ through the disclosed '
                                   'D3 GPU nondeterminism (M6D3e, D9)'},
           'timing': {'seconds_per_iteration': seconds,
                      'projected_150000_iteration_hours': [s * 150000 / 3600 for s in seconds]},
           'gradient_determinism': gradient_detail, 'atime': atime, 'atime_after_diagnostics': load('atime2.json'),
           'compat_probe': {k: compat[k] for k in ('status', 'graph_def_sha256', 'native_runtime')},
           'cases': {c: {k: q[c][k] for k in ('status', 'run_dir', 'run_id', 'plan', 'seconds', 'firewall',
                                               'run_summary')} for c in CASES},
           'confirmations': {'scientific_seed_launched': False, 'test_accessed': False, 'val_accessed': False,
                             'bank_generated': False, 'downstream_evaluation': False, 'graph_modified': False,
                             'environment_modified': False, 'e03_modified': False}}
    return doc


def report(doc):
    rows = ''.join('| `%s` | %s |\n' % (g['gate'], 'pass' if g['pass'] else '**FAIL**') for g in doc['gates'])
    crop = ''.join('| %s | %d | %d | %d | %d | %d |\n' % (k.replace('|', ' / '), v['faces'], v['outside_any'], v['padded_faces'],
                                                         v['padded_pixels_median'], v['padded_pixels_max'])
                   for k, v in sorted(doc['prepass']['crop_boundary'].items()))
    t = doc['timing']
    return '\n'.join([
        '# E04 Physics-STD: production qualification', '',
        '**Status:** `%s`, %d/%d gates. Qualification seed %d only; **no E04 scientific seed was launched** and no E04 '
        'scientific authority exists. Machine-readable record: `E04_PRODUCTION_QUALIFICATION.json`.'
        % (doc['status'], doc['gates_passed'], doc['gates_total'], doc['qualification_seed']), '',
        '- Frozen geometry pre-pass `%s`: %d TRAIN faces (%d live, %d spoof), two independent CPU runs byte-identical.'
        % (doc['prepass']['manifest_sha256'], doc['prepass']['counts']['samples'], doc['prepass']['counts']['live'],
           doc['prepass']['counts']['spoof']),
        '- Feed pack `%s`, verified on the GPU host and on the laptop.' % doc['feed_pack']['manifest_sha256'],
        '- Production construction `%s…` (%d ops) equals the qualified M6D3e construction.'
        % (doc['graph']['construction'][0][:16], doc['graph']['construction'][1]),
        '- %.3f / %.3f s per iteration, projected %.1f / %.1f h for 150000 iterations.'
        % (t['seconds_per_iteration'][0], t['seconds_per_iteration'][1], t['projected_150000_iteration_hours'][0],
           t['projected_150000_iteration_hours'][1]), '',
        '## Gates', '', '| gate | result |', '|---|---|', rows.rstrip('\n'), '',
        '## Crop boundary (official landmark ROI vs the 256 frame)', '',
        '| dataset/role | faces | ROI outside frame | zero-padded faces | median padded px | max padded px |',
        '|---|---|---|---|---|---|', crop.rstrip('\n'), '',
        '## Determinism', '', doc['determinism']['note'] + '.', ''] + (blocker(doc) if doc['failed_gates'] else []) + [''])


def blocker(doc):
    g = doc['gradient_determinism']
    return ['## Blocker: real-data adversarial gradient is not bit-repeatable (fail closed, D9 condition 6)', '',
            '- With identical initial state and identical real TRAIN inputs (iteration 0), every per-loss gradient is '
            'bit-repeatable in-process and across 3 processes except `%s`.' % ', '.join(g['non_repeatable_losses']),
            '- L_G (adversarial generator loss) backpropagates through the four discriminators into 58 generator '
            'variables. Its backward path alone contains GPU `ResizeBilinearGrad` (%s) and `ScatterNd` (%s; the '
            'gradient of the CPU-forward trace warp `GatherNd`), op types without deterministic TF 1.15 GPU kernels.'
            % (g['known_nondeterministic_gpu_ops']['ResizeBilinearGrad'], g['known_nondeterministic_gpu_ops']['ScatterNd']),
            '- Size: median per-variable relative L2 difference %.2e; the largest relative values (%.2f) occur only for '
            'biases of convolutions followed by training-mode BN, whose exact gradient is zero; max |diff| %.2e.'
            % (g['L_G_rel_l2_median'], g['L_G_rel_l2_max'], g['L_G_max_abs_diff']),
            '- Consequence: E04 real-data training is not bitwise reproducible across processes; the generator state '
            'differs from Step 1 of iteration 0 and later losses diverge at ~1e-7 to 1e-6 relative within 2 '
            'iterations. Data, RNG draws, inputs, initial state and iteration-0 losses stay identical.',
            '- M6D3e recorded bit-repeatable gradients for every loss on synthetic data, so this is a new gradient-level '
            'observation; the M6D3e statement that the warp has a deterministic backward path holds for its forward '
            'pass only (the gradient ScatterNd is placed on /gpu:0). It is not a driver difference (in-process).',
            '- No graph, placement or optimizer change was made. An owner decision is required.', '']


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    ca = sub.add_parser('collect-atime')
    ca.add_argument('--post', default='post')
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect-atime':
        json.dump(collect_atime(args.post), sys.stdout, sort_keys=True)
        return 0
    root = Path(args.repo_root)
    doc = build(root, args.collected)
    (root / OUT['record']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / OUT['report']).write_text(report(doc))
    (root / OUT['sha256']).write_text(''.join('%s  %s\n' % (sha256_bytes((root / OUT[k]).read_bytes()), OUT[k])
                                              for k in ('record', 'report')))
    print(doc['status'], doc['gates_passed'], '/', doc['gates_total'], doc['failed_gates'])
    return 0 if not doc['failed_gates'] else 1


if __name__ == '__main__':
    sys.exit(main())
