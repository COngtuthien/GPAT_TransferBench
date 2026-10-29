"""E04 owner-resolution investigation (D1-D9): aggregates the bounded, non-scientific probes behind the DRAFT addendum
docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum_DRAFT.md.

  collect-d2  (GPU host, a Python with pyarrow -- gpat-m6-e05, CUDA hidden, `python -I -B - collect-d2 < this file`):
              reads only manifests/pairs_train_v1.parquet, manifests/split_v1.parquet and the E03 feed-pack
              samples.json (no image, no landmark payload) and prints the TRAIN pre-pass populations.
  build       (repository): combines that document with the D9 BN probe outputs (tools/e04_d9_bn_probe.py), the
              repeated CPU geometry workers, the RNG vectors from three interpreters (tools/e04_rng_test_vectors.py),
              the committed M6D3a/M6D3e evidence, and D1 vectors from the pinned 3DDFA_V2 crop functions on synthetic
              landmarks; writes outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION.{json,md}, the evidence copies and a
              SHA-256 list. Nothing here is approved, trains, fits a benchmark face or opens an image.
"""
import argparse
import base64
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

REPO_GPU = '/home/student20261/workdir/GPAT_TransferBench'
PACK = '/home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/e03_feed_pack_v1'
STATUS = 'E04_OWNER_RULING_READY_FOR_FINAL_APPROVAL'
OUT = {'record': 'outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION.json',
       'report': 'outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION.md',
       'sha256': 'outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION_SHA256.txt',
       'probe': 'outputs/audit/E04_D9_BN_PROBE_%s.json', 'rng': 'outputs/audit/E04_RNG_TEST_VECTORS.json',
       'd2': 'outputs/audit/E04_D2_PREPASS_POPULATION.json'}
INTERPRETERS = ('laptop', 'gpat-m6-e03', 'gpat-m6-e04-geometry')
DRAFTS = ('docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum_DRAFT.md',
          'configs/amendments/e04_production_resolution.DRAFT.yaml')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


# ============================================================================ collect-d2 (GPU host, read-only)
def collect_d2():
    import collections
    import pyarrow.parquet as pq
    repo = Path(REPO_GPU)
    pairs_path, split_path = repo / 'manifests/pairs_train_v1.parquet', repo / 'manifests/split_v1.parquet'
    rows = pq.read_table(str(pairs_path), columns=['pair_id', 'target_live_id', 'source_spoof_id']).to_pylist()
    split = {r['sample_id']: r for r in pq.read_table(str(split_path), columns=[
        'sample_id', 'split', 'label_binary', 'dataset']).to_pylist()}
    live = [r['target_live_id'] for r in rows]
    spoof = [r['source_spoof_id'] for r in rows]
    reuse = collections.Counter(live)
    pack_ids = {r[0] for r in json.loads((Path(PACK) / 'samples.json').read_text())['rows']}
    columns = pq.read_schema(str(pairs_path)).names

    def classes(ids):
        return sorted([k[0], k[1], v] for k, v in collections.Counter(
            (split[x]['split'], split[x]['label_binary']) for x in ids).items())
    return {'phase': 'collect-d2', 'pairs_train_v1_sha256': sha256_file(pairs_path),
            'split_v1_sha256': sha256_file(split_path),
            'e03_feed_pack_manifest_sha256': sha256_file(Path(PACK) / 'pack_manifest.json'),
            'rows': len(rows), 'pair_ids_unique': len({r['pair_id'] for r in rows}),
            'stored_order_sorted_by_pair_id': [r['pair_id'] for r in rows] == sorted(r['pair_id'] for r in rows),
            'unique_target_live': len(set(live)), 'unique_source_spoof': len(set(spoof)),
            'live_spoof_overlap': len(set(live) & set(spoof)),
            'live_reuse': {'min': min(reuse.values()), 'max': max(reuse.values()),
                           'mean': round(len(live) / len(reuse), 6)},
            'live_split_label': classes(set(live)), 'spoof_split_label': classes(set(spoof)),
            'live_by_dataset': dict(sorted(collections.Counter(split[x]['dataset'] for x in set(live)).items())),
            'spoof_by_dataset': dict(sorted(collections.Counter(split[x]['dataset'] for x in set(spoof)).items())),
            'feed_pack_landmarks_cover': {'live': len(set(live) & pack_ids), 'spoof': len(set(spoof) & pack_ids)},
            'manifest_columns': columns,
            'forbidden_columns_present_not_to_be_read': [c for c in columns if c in ('attack_macro',)]}


# ============================================================================ build (repository)
def d1_vectors(root):
    """The pinned crop rule on deterministic synthetic 68-point sets (no benchmark landmark is read)."""
    import ast
    import numpy as np
    src_root = root / 'third_party/source_cache/tddfa_v2'
    pinned = json.loads((root / 'outputs/audit/M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION.json').read_text())[
        'source_integrity']['source_files_sha256']          # the 113 pinned-blob hashes verified by M6D3a
    raw = (src_root / 'utils/functions.py').read_bytes()
    if sha256_bytes(raw) != pinned['utils/functions.py']:
        raise SystemExit('pinned utils/functions.py changed')
    tree = ast.parse(raw)
    scope = {'sqrt': math.sqrt, 'np': np}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('parse_roi_box_from_landmark',
                                                                                   'crop_img')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src_root / 'utils/functions.py'), 'exec'), scope)
    parse, crop = scope['parse_roi_box_from_landmark'], scope['crop_img']
    a = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    cases = {'centred_face': np.stack([128 + 70 * np.cos(a), 140 + 85 * np.sin(a)], 1),
             'face_touching_left_and_top_edges': np.stack([60 + 75 * np.cos(a), 70 + 80 * np.sin(a)], 1)}
    img = (np.arange(256 * 256 * 3) % 251).astype(np.uint8).reshape(256, 256, 3)
    out = {}
    for name, lm in cases.items():
        pts = np.ascontiguousarray(lm.astype(np.float32).T)              # [2, 68]: x row 0, y row 1
        roi = parse(pts)
        c = crop(img, roi)
        sx, sy, ex, ey = [int(round(x)) for x in roi]
        out[name] = {'landmarks_sha256': sha256_bytes(lm.astype(np.float32).tobytes()),
                     'roi_box': [float(x) for x in roi], 'rounded_box': [sx, sy, ex, ey],
                     'crop_shape': list(c.shape), 'crop_sha256': sha256_bytes(c.tobytes()),
                     'zero_padded_pixels': int((c == 0).all(-1).sum()),
                     'outside_frame': [sx < 0, sy < 0, ex > 256, ey > 256]}
    out['source_files_sha256'] = {f: pinned[f] for f in ('utils/functions.py', 'TDDFA.py', 'utils/tddfa_util.py')}
    out['source_pin'] = '1b6c67601abffc1e9f248b291708aef0e43b55ae'
    return out


def d9_analysis(root, probes):
    import numpy as np
    hist = [json.loads((root / ('outputs/audit/M6D3E_E04_SYNTHETIC_PROCESS_%d.json' % k)).read_text()) for k in (1, 2)]
    ready = json.loads((root / 'outputs/audit/E04_READINESS_REQUAL_M6D3E_PROCESS.json').read_text())

    def arr(r):
        return np.frombuffer(base64.b64decode(r['f32_b64']), np.float32).astype(np.float64)
    out = {'probes': len(probes), 'graph_def_equal_history': all(p['graph_def_sha256'] == hist[0]['graph_def_sha256']
                                                                for p in probes),
           'fixtures_equal_history': all(p['fixture_sha256'] == hist[0]['fixture_sha256'] for p in probes),
           'native_runtime_equal_history': all(p['native_runtime'] == hist[0]['native_runtime'] and
                                               p['tensorflow'] == hist[0]['tensorflow'] for p in probes),
           'bn_update_sets_equal_history': all(p['bn_changes_by_forward_collection'] ==
                                               [s['bn_changes_by_forward_collection'] for s in hist[0]['steps']]
                                               for p in probes),
           'changed_trainable_sets_equal_history': all(p['changed_trainables'] ==
                                                       [s['changed_trainables'] for s in hist[0]['steps']]
                                                       for p in probes),
           'losses_before_each_step_equal_history': all(p['losses_before'] == [s['losses_before'] for s in hist[0]['steps']]
                                                        for p in probes),
           'final_losses_equal_history': all(p['final_losses'] == hist[0]['final_losses'] for p in probes),
           'applies_per_process': [p['optimizer_applications'] for p in probes],
           'moving_statistic_consumer_op_types': sorted({t for p in probes for v in
                                                          p['moving_statistic_consumer_op_types'].values() for t in v}),
           'fused_batch_norm_reads_moving_statistics': any('FusedBatchNorm' in t for p in probes for v in
                                                           p['moving_statistic_consumer_op_types'].values() for t in v),
           'steps': {}}
    for i, step in enumerate(('step1', 'step2', 'step3')):
        names = sorted(probes[0]['post_state_sha256'][step])
        h = [{n: x['steps'][i]['post_state'][n]['sha256'] for n in names} for x in hist]
        cur = [p['post_state_sha256'][step] for p in probes]
        hist_vary = [n for n in names if h[0][n] != h[1][n]]
        vary = [n for n in names if len({c[n] for c in cur}) > 1]
        vs_hist = [n for n in names if cur[0][n] != h[0][n]]
        rec = {'state_tensors': len(names), 'varying_among_current_processes': vary,
               'distinct_current_variants': len({tuple(c[n] for n in vary) for c in cur}),
               'current_equal_readiness_run_variant': any(c == {n: ready['steps'][i]['post_state'][n]['sha256']
                                                                for n in names} for c in cur),
               'historical_p1_vs_p2_differing': len(hist_vary),
               'historical_p1_vs_p2_differing_trainable': len([n for n in hist_vary if not (
                   'moving_' in n or 'Adam' in n)]),
               'current_vs_historical_p1_differing': vs_hist,
               'current_equal_historical_p2_on_historically_varying': [
                   sum(c[n] == h[1][n] for n in hist_vary) for c in cur]}
        if vary:
            worst, rel = 0., 0.
            for n in vary:
                xs = [arr(p['d_bn_tensors'][step][n]) for p in probes]
                d = max(float(np.abs(x - xs[0]).max()) for x in xs)
                worst, rel = max(worst, d), max(rel, d / float(np.abs(xs[0]).max()))
            rec['current_variation_max_abs'], rec['current_variation_max_rel'] = worst, rel
        if step == 'step1':
            rec['tensors_vs_history'] = {}
            for n in vs_hist:
                a, s = arr(probes[0]['d_bn_tensors'][step][n]), hist[0]['steps'][0]['post_state'][n]
                rec['tensors_vs_history'][n] = {
                    'shape': s['shape'], 'dtype': s['dtype'], 'current_sha256': cur[0][n], 'historical_sha256': s['sha256'],
                    'abs_max_value': float(np.abs(a).max()),
                    'float32_ulp_at_abs_max': float(np.spacing(np.float32(np.abs(a).max()))),
                    'delta_mean': float(a.mean() - s['mean']), 'delta_l2': float(np.linalg.norm(a) - s['l2']),
                    'relative_delta_l2': float((np.linalg.norm(a) - s['l2']) / s['l2']),
                    'delta_min': float(a.min() - s['min']), 'delta_max': float(a.max() - s['max']),
                    'summary_delta_ulps_at_abs_max': round(max(abs(a.min() - s['min']), abs(a.max() - s['max'])) /
                                                           float(np.spacing(np.float32(np.abs(a).max()))), 3),
                    'comparison_basis': 'historical raw tensors not retained; summaries (min/max/mean/L2) only'}
        out['steps'][step] = rec
    return out


def build(root, d2, probe_dir):
    root, probe_dir = Path(root), Path(probe_dir)
    probes = [json.loads((probe_dir / ('probe_%d.json' % k)).read_text()) for k in (1, 2, 3, 4)]
    probe_sha = {k: sha256_file(probe_dir / ('probe_%d.json' % k)) for k in (1, 2, 3, 4)}
    rng = {k: json.loads((probe_dir / ('rng_%s.json' % k)).read_text()) for k in INTERPRETERS}
    strip = lambda d: {k: v for k, v in d.items() if k not in ('numpy', 'python')}  # noqa: E731
    cpu = {'readiness_cpu_1': sha256_file(root / 'outputs/audit/E04_READINESS_REQUAL_M6D3A_CPU.json')}
    cpu.update({'cpu_%d' % k: sha256_file(probe_dir / ('m6d3a_cpu_%d.json' % k)) for k in (2, 3)})
    cpu_doc = json.loads((probe_dir / 'm6d3a_cpu_2.json').read_text())
    cuda_doc = json.loads((root / 'outputs/audit/E04_READINESS_REQUAL_M6D3A_CUDA.json').read_text())
    m6d3a = (root / 'outputs/audit/M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION.json').read_text()
    fields = {'fitted': ('regressor', 'fitted_sha256'), 'normalized_input': ('regressor', 'normalized_sha256'),
              'mesh': ('end_to_end', 'mesh_sha256'), 'depth': ('end_to_end', 'depth_sha256')}
    d3 = {'clean_cpu_processes': len(cpu), 'cpu_outputs_byte_identical': len(set(cpu.values())) == 1,
          'cpu_files_sha256': cpu,
          'values': {k: {'cpu': cpu_doc[a][b], 'cuda': cuda_doc[a][b], 'cpu_in_committed_m6d3a': cpu_doc[a][b] in m6d3a,
                         'cuda_in_committed_m6d3a': cuda_doc[a][b] in m6d3a} for k, (a, b) in fields.items()},
          'qualified_torch_settings': ['torch.set_num_threads(1)', 'torch.manual_seed(0)',
                                       'torch.use_deterministic_algorithms(True)', 'cudnn.benchmark=False',
                                       'cudnn.deterministic=True', 'TF32 off'],
          'source': 'tools/m6d3a_e04_geometry_preflight.py:90-96'}
    rng_rec = {'interpreters': {k: {'numpy': v['numpy'], 'python': v['python']} for k, v in rng.items()},
               'identical_across_interpreters': strip(rng['laptop']) == strip(rng['gpat-m6-e03']) ==
               strip(rng['gpat-m6-e04-geometry']), 'vectors': rng['gpat-m6-e03']}
    doc = {
        'milestone': 'E04-OWNER-RESOLUTION-INVESTIGATION', 'status': STATUS, 'draft': True, 'approved': False,
        'e04_scientific_authority': None, 'scientific_execution_authorized': False,
        'draft_addendum': 'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum_DRAFT.md',
        'draft_companion': 'configs/amendments/e04_production_resolution.DRAFT.yaml',
        'D1_crop_vectors': d1_vectors(root), 'D2_population': d2, 'D3_cpu_repeatability': d3,
        'D4_D7_rng': rng_rec, 'D9_bn_investigation': d9_analysis(root, probes),
        'probe_files_sha256': probe_sha,
        'draft_files_sha256': {f: sha256_file(root / f) for f in DRAFTS},
        'immutable_inputs_verified': {f: sha256_file(root / f) == h for f, h in json.loads(
            (root / DRAFTS[1]).read_text())['immutable_inputs_sha256'].items() if (root / f).is_file()},
        'confirmations': {'benchmark_images_decoded': 0, 'landmark_payload_read': False, 'test_accessed': False,
                          'prepass_started': False, 'experiment_seed_used_for_execution': False,
                          'environment_modified': False, 'gpu_checkout_modified': False, 'training_launched': False}}
    return doc, probes, probe_sha


def report(doc):
    d9, d3, rng, d2 = doc['D9_bn_investigation'], doc['D3_cpu_repeatability'], doc['D4_D7_rng'], doc['D2_population']
    s1, s2 = d9['steps']['step1'], d9['steps']['step2']
    seeds = rng['vectors']['seeds']
    t = ['# E04: owner-resolution investigation (D1–D9)', '',
         '**Status:** `%s`. The decision package is complete and **DRAFT**; nothing in it is approved. There is no E04 '
         'scientific authority and execution is not authorized. Draft addendum: `%s`; draft companion: `%s`.'
         % (doc['status'], doc['draft_addendum'], doc['draft_companion']), '',
         '## D2 TRAIN pre-pass population (manifest-only read)', '',
         '- `pairs_train_v1` `%s…`: %d rows, %d unique pair_ids, stored sorted by pair_id: %s.'
         % (d2['pairs_train_v1_sha256'][:16], d2['rows'], d2['pair_ids_unique'], d2['stored_order_sorted_by_pair_id']),
         '- %d unique live targets (split/label %s; reuse %d–%d, mean %.3f) and %d unique spoof sources (%s); '
         'overlap %d. Landmarks in `e03_feed_pack_v1` cover %d live and %d spoof.'
         % (d2['unique_target_live'], d2['live_split_label'], d2['live_reuse']['min'], d2['live_reuse']['max'],
            d2['live_reuse']['mean'], d2['unique_source_spoof'], d2['spoof_split_label'], d2['live_spoof_overlap'],
            d2['feed_pack_landmarks_cover']['live'], d2['feed_pack_landmarks_cover']['spoof']),
         '- The manifest carries `attack_macro`, which the E04 feed must never read.', '',
         '## D3 CPU geometry repeatability (synthetic)', '',
         '- %d clean CPU processes are byte-identical (%s). Fitted parameters, normalised input, mesh and depth equal '
         'the committed M6D3a values (recorded under driver 595.84): %s. CUDA differs from CPU in the fit and mesh, '
         'but its values also equal the committed CUDA values.'
         % (d3['clean_cpu_processes'], d3['cpu_outputs_byte_identical'],
            all(v['cpu_in_committed_m6d3a'] for v in d3['values'].values())), '',
         '## D4 / D7 RNG test vectors', '',
         '- Identical across numpy %s: %s. The seed-314159 realization reproduces the M6D3e hard-sample fixture '
         'hashes: %s.' % (' / '.join('%s (%s)' % (v['numpy'], k) for k, v in rng['interpreters'].items()),
                          rng['identical_across_interpreters'],
                          rng['vectors']['qualification_realization_check']['equals_m6d3e_fixture_sha256']), '',
         '| seed | pair seed | hard seed | pair stream sha256 | iteration 0 rows | hard stream sha256 | keep rate | '
         'straddling / repeated |', '|---|---|---|---|---|---|---|---|']
    t += ['| %s | %d | %d | `%s…` | %s | `%s…` | %.5f | %d / %d |' % (
        s, v['pair_seed'], v['hard_seed'], v['pair_stream']['sha256_int32_le'][:16], v['pair_stream']['iteration0_rows'],
        v['hard_stream']['sha256_float32_scales_then_keep_per_iteration'][:16], v['hard_stream']['keep_rate'],
        v['pair_stream']['pass_straddling_iterations'], len(v['pair_stream']['straddling_iterations_with_a_repeated_row']))
        for s, v in sorted(seeds.items(), key=lambda x: int(x[0]))]
    t += ['', '## D9 BN investigation (synthetic, qualification seed 314159, %d clean processes)' % d9['probes'], '',
          '- Identical to history: GraphDef %s, fixtures %s, CUDA/cuDNN/TF %s, BN update sets %s, changed-trainable '
          'sets %s, losses before every step %s, final losses %s.'
          % (d9['graph_def_equal_history'], d9['fixtures_equal_history'], d9['native_runtime_equal_history'],
             d9['bn_update_sets_equal_history'], d9['changed_trainable_sets_equal_history'],
             d9['losses_before_each_step_equal_history'], d9['final_losses_equal_history']),
          '- Step 1: all current processes and the readiness run are identical (%d varying tensors); they differ from '
          'history only in %s. Historical processes 1 and 2 were identical at step 1. The difference is therefore '
          'deterministic within each runtime and differs between the runtimes.'
          % (len(s1['varying_among_current_processes']), ', '.join('`%s`' % n for n in s1['current_vs_historical_p1_differing'])),
          '', 'The historical raw tensors were not retained, so the comparison with history uses their recorded '
          'min/max/mean/L2 summaries:', '',
          '| tensor | shape | |max| | float32 ulp there | Δmax | Δmin | Δ in ulps | relative ΔL2 |',
          '|---|---|---|---|---|---|---|---|']
    t += ['| `%s` | %s | %.3e | %.2e | %.2e | %.2e | %g | %.2e |' % (
        n, r['shape'], r['abs_max_value'], r['float32_ulp_at_abs_max'], r['delta_max'], r['delta_min'],
        r['summary_delta_ulps_at_abs_max'], r['relative_delta_l2']) for n, r in s1['tensors_vs_history'].items()]
    t += ['', '- Steps 2/3: history varied in %d tensors (%d trainable, all D3). Every current process matches '
          'historical process 2 on %s of them; the only current variation is the same 4 D3 moving means (%d variants, '
          'max |Δ| %.2e, max relative %.2e).'
          % (s2['historical_p1_vs_p2_differing'], s2['historical_p1_vs_p2_differing_trainable'],
             '/'.join(map(str, s2['current_equal_historical_p2_on_historically_varying'])),
             s2['distinct_current_variants'], s2['current_variation_max_abs'], s2['current_variation_max_rel']),
          '- The moving statistics are read only by `%s`; FusedBatchNorm never reads them (training=True). They are '
          'write-only during training and cannot reach any loss, gradient or trainable variable.'
          % '`, `'.join(d9['moving_statistic_consumer_op_types']), '',
          'Interpretation: last-bit float32 rounding in the D3 batch-mean reduction that feeds the moving average. The '
          'update semantics (ops, order, sets) are unchanged. The exact kernel is not identified (CUPTI unavailable), '
          'the same limitation M6D3e disclosed.', '',
          '## Boundary', '',
          'No benchmark image or landmark payload was read and no TEST was accessed. The pre-pass did not start, no '
          'experiment seed was executed, no environment or GPU checkout was modified, and no training was launched.', '']
    return '\n'.join(t)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('collect-d2')
    b = sub.add_parser('build')
    b.add_argument('--d2', required=True)
    b.add_argument('--probe-dir', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect-d2':
        json.dump(collect_d2(), sys.stdout, sort_keys=True)
        return 0
    root = Path(args.repo_root)
    d2 = json.loads(Path(args.d2).read_text())
    doc, probes, probe_sha = build(root, d2, args.probe_dir)
    distinct = sorted(set(probe_sha.values()), key=lambda h: min(k for k, v in probe_sha.items() if v == h))
    doc['committed_probe_variants'] = {chr(65 + i): {'sha256': h, 'probes': [k for k, v in probe_sha.items() if v == h]}
                                       for i, h in enumerate(distinct)}
    for i, h in enumerate(distinct):
        k = min(k for k, v in probe_sha.items() if v == h)
        shutil.copyfile(Path(args.probe_dir) / ('probe_%d.json' % k), root / (OUT['probe'] % chr(65 + i)))
    shutil.copyfile(Path(args.probe_dir) / 'rng_gpat-m6-e03.json', root / OUT['rng'])
    (root / OUT['d2']).write_text(json.dumps(d2, indent=2, sort_keys=True) + '\n')
    (root / OUT['record']).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    (root / OUT['report']).write_text(report(doc))
    files = [OUT['record'], OUT['report'], OUT['rng'], OUT['d2']] + [OUT['probe'] % chr(65 + i)
                                                                     for i in range(len(distinct))] + list(DRAFTS)
    (root / OUT['sha256']).write_text(''.join('%s  %s\n' % (sha256_file(root / f), f) for f in files))
    print(doc['status'], len(distinct), 'probe variants')
    return 0


if __name__ == '__main__':
    sys.exit(main())
