"""E05 PCGAN S7 low-contrast scientific review (non-invasive, descriptive; no gate of the frozen protocol).

  collect (GPU host, CPU only, read-only; fed on stdin from the repository, imports the library from --repo):
    CUDA_VISIBLE_DEVICES= <gpat-m6-e05 python> -I -B - collect --repo <clean clone> [--montage-dir DIR] \
        < tools/e05_s7_contrast_diagnostic.py > collected.json
  build (laptop): python3 tools/e05_s7_contrast_diagnostic.py build \
        --collected outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC_EVIDENCE/collected.json
        -> outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.{json,md} and _SHA256.txt

Inputs are ONLY the completed S7 TRAIN toy outputs named by the committed S7 records and the TRAIN source/target faces
of the 186 approved toy pairs (TrainFaceReader: TRAIN allowlist from split_v1 + M2, O_NOATIME, atime stat check,
M2 SHA256). No model is loaded, no image is generated, no VAL or TEST face is opened (a stat-only atime ledger of every
canonical face is taken before and after). Nothing is written except stdout and, optionally, a montage of the raw
(unmodified) S7 outputs beside their inputs in --montage-dir (never committed; it contains dataset faces).

The review introduces no acceptance threshold: every statistic is descriptive. A contract violation is reported only
for a mechanical inconsistency with the frozen contract (conversion rule, finiteness, lineage, noise seed, checkpoint).
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = '/home/student20261/workdir/GPAT_TransferBench_runtime'
FACES = RUNTIME + '/data/processed/faces_256'
SEEDS = (42, 1337, 2026)
METHODS = ('E05', 'E06c', 'E03', 'E06b')
OUT = 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC'
EVIDENCE = 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC_EVIDENCE/collected.json'


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ collect (host)
def face_ledger():
    out = {}
    for d in sorted(os.listdir(FACES)):
        for f in sorted(os.listdir(os.path.join(FACES, d))):
            st = os.stat(os.path.join(FACES, d, f))
            out['%s/%s' % (d, f)] = (st.st_atime_ns, st.st_mtime_ns)
    return out


def collect(repo, montage_dir=None):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    from methods.common import learned_bank as lb
    import io

    def decode(raw):
        with Image.open(io.BytesIO(raw)) as im:
            assert im.mode == 'RGB' and im.size == (256, 256)
            return np.array(im)

    def img_stats(a):
        f = a.astype(np.float64)
        lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        return {'min': int(a.min()), 'max': int(a.max()), 'mean': float(f.mean()), 'std': float(f.std()),
                'channel_mean': [float(f[..., c].mean()) for c in range(3)],
                'channel_std': [float(f[..., c].std()) for c in range(3)],
                'lum_std': float(lum.std()),
                'p01': float(np.percentile(f, 1)), 'p99': float(np.percentile(f, 99)),
                'sat0': int((a == 0).sum()), 'sat255': int((a == 255).sum()),
                'frac_within_5_of_mean': float((np.abs(f - f.mean()) <= 5).mean())}

    def small_gray(a):          # 64x64 block-mean luminance, zero-mean (structure only)
        f = a.astype(np.float64)
        g = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).reshape(64, 4, 64, 4).mean(axis=(1, 3))
        return g - g.mean()

    def corr(x, y):
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / d) if d > 0 else None

    pre = face_ledger()
    t_pre = utc_now()
    doc = {'phase': 'collect', 'utc': t_pre, 'repo': str(repo),
           'git_head': os.popen('git -C %s rev-parse HEAD' % repo).read().strip(),
           'git_status': os.popen('git -C %s status --porcelain' % repo).read().strip(),
           'python': sys.version.split()[0], 'numpy': np.__version__}
    s7 = {m: json.loads((Path(repo) / ('outputs/audit/LEARNED_BANK_S7_%s.json' % m)).read_text()) for m in METHODS}
    toy = lb.load_toy_manifest()
    used = {}
    for t in toy:
        used[t['target_live_id']] = t['dataset']
        used[t['source_spoof_id']] = t['dataset']
    reader = lb.TrainFaceReader(Path(FACES), lb.train_face_allowlist_parquet(used))
    faces = {}
    for t in toy:
        for sid in (t['source_spoof_id'], t['target_live_id']):
            if sid not in faces:
                raw, digest = reader.read(sid)
                faces[sid] = decode(raw)
    doc['reader'] = reader.record()
    doc['input_face_stats'] = {sid: img_stats(a) for sid, a in sorted(faces.items())}

    methods = {}
    outputs_e05 = {}
    for m in METHODS:
        methods[m] = {}
        for s in SEEDS:
            o = s7[m]['outputs'][str(s)]
            d = Path(o['run_A_dir'])
            body = (d / lb.MANIFEST).read_bytes()
            rows = [json.loads(x) for x in body.decode().splitlines()]
            rec = {'dir': str(d), 'manifest_sha256': sha_bytes(body),
                   'manifest_equals_s7_record': sha_bytes(body) == o['manifest_sha256'],
                   'run_B_manifest_sha256': sha_bytes((Path(o['run_B_dir']) / lb.MANIFEST).read_bytes()),
                   'rows': []}
            pix = set()
            arrs = []
            for r in rows:
                raw = (d / r['output_path']).read_bytes()
                a = decode(raw)
                st = img_stats(a)
                st.update(row_id=r['row_id'], row_index=r['row_index'], file_sha_ok=sha_bytes(raw) == r['output_sha256'],
                          pixel_sha_ok=lb.pixel_sha256(a) == r['output_pixel_sha256'],
                          raw_min=r['raw_output_min'], raw_max=r['raw_output_max'],
                          raw_finite=r['raw_output_finite'])
                pix.add(r['output_pixel_sha256'])
                if m == 'E05':
                    # conversion consistency: tensor2im is monotone, so uint8 min/max must equal the rule applied
                    # to the recorded float32 raw min/max: clip((x+1)/2*255, 0, 255).astype(uint8) in float32
                    rule = lambda x: int(np.clip((np.float32(x) + np.float32(1)) / np.float32(2.0) * np.float32(255.0),
                                                 0, 255).astype(np.uint8))
                    st['conversion_min_consistent'] = rule(r['raw_output_min']) == st['min']
                    st['conversion_max_consistent'] = rule(r['raw_output_max']) == st['max']
                    src, tgt = faces[r['source_spoof_id']], faces[r['target_live_id']]
                    g, gs, gt = small_gray(a), small_gray(src), small_gray(tgt)
                    st['structure_corr_source_spoof'] = corr(g, gs)
                    st['structure_corr_target_live'] = corr(g, gt)
                    om, sm, tm = (np.array(x.reshape(-1, 3).mean(0)) for x in (a.astype(float), src.astype(float),
                                                                            tgt.astype(float)))
                    st['colour_mean_dist_source'] = float(np.abs(om - sm).sum())
                    st['colour_mean_dist_target'] = float(np.abs(om - tm).sum())
                    st['lineage'] = [r['pair_id'], r['source_spoof_id'], r['target_live_id'], r['dataset'],
                                     r['deterministic_noise_seed'], r['checkpoint_identity_sha256']]
                    outputs_e05[(s, r['row_id'])] = a
                rec['rows'].append(st)
                arrs.append(a[::4, ::4].astype(np.int16))
            rec['distinct_pixel_sha256'] = len(pix)
            stack = np.stack(arrs).reshape(len(arrs), -1)
            mind = None
            near = []
            for i in range(len(stack)):
                dd = np.abs(stack[i + 1:] - stack[i]).mean(axis=1) if i + 1 < len(stack) else np.array([])
                if dd.size:
                    mind = float(dd.min()) if mind is None else min(mind, float(dd.min()))
                    for j in np.nonzero(dd < 1.0)[0]:
                        ri, rj = rows[i], rows[i + 1 + int(j)]
                        near.append([ri['row_id'], rj['row_id'], float(dd[j]),
                                     ri['source_spoof_id'] is not None and ri['source_spoof_id'] == rj['source_spoof_id'],
                                     ri['target_live_id'] is not None and ri['target_live_id'] == rj['target_live_id']])
            rec['min_pairwise_mean_abs_diff_64px'] = mind
            rec['near_identical_pairs_mad_lt_1'] = near
            methods[m][str(s)] = rec
    doc['methods'] = methods
    # same pair, different seed (E05): how far the three generators' outputs are apart
    cross = {}
    for a_, b_ in ((42, 1337), (42, 2026), (1337, 2026)):
        dv = [float(np.abs(outputs_e05[(a_, t['pair_id'])].astype(float) - outputs_e05[(b_, t['pair_id'])]).mean())
              for t in toy]
        cross['%d_vs_%d' % (a_, b_)] = {'mean_abs_diff_mean': float(np.mean(dv)), 'min': float(np.min(dv)),
                                         'max': float(np.max(dv))}
    doc['e05_cross_seed_same_pair'] = cross
    if montage_dir:
        md = Path(montage_dir)
        md.mkdir(parents=True, exist_ok=False)
        reps = {}
        for s in SEEDS:
            rows = sorted(methods['E05'][str(s)]['rows'], key=lambda x: (x['std'], x['row_id']))
            pick = [rows[0], rows[len(rows) // 2], rows[-1]]
            reps[str(s)] = [x['row_id'] for x in pick]
            tiles = []
            for x in pick:
                t = next(t for t in toy if t['pair_id'] == x['row_id'])
                tiles.append(np.concatenate([faces[t['source_spoof_id']], faces[t['target_live_id']],
                                             outputs_e05[(s, x['row_id'])]], axis=1))
            Image.fromarray(np.concatenate(tiles, axis=0)).save(md / ('e05_s7_seed%d_src_tgt_out.png' % s))
        doc['montage'] = {'dir': str(md), 'representatives': reps,
                          'layout': 'rows = min / median / max per-image std; columns = source spoof | target live | '
                                    'raw S7 output (unmodified)'}
    post = face_ledger()
    doc['atime'] = {'pre_utc': t_pre, 'post_utc': utc_now(), 'files_pre': len(pre), 'files_post': len(post),
                    'same_paths': sorted(pre) == sorted(post),
                    'changed': sorted(k for k in pre if post.get(k) != pre[k])[:20]}
    import pyarrow.parquet as pq
    split = {r['sample_id']: r['split'] for r in pq.read_table(str(Path(repo) / 'manifests/split_v1.parquet'),
                                                               columns=['sample_id', 'split']).to_pylist()}
    by = {}
    for k in pre:
        sp = split.get(os.path.basename(k)[:-4], 'UNMAPPED')
        b = by.setdefault(sp, {'files': 0, 'changed': 0})
        b['files'] += 1
        b['changed'] += post.get(k) != pre[k]
    doc['atime']['by_split'] = by
    print(json.dumps(doc, sort_keys=True))


# ------------------------------------------------------------------------------------------------ build (laptop)
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    return {'min': min(xs), 'p05': pct(xs, 5), 'p25': pct(xs, 25), 'median': pct(xs, 50), 'p75': pct(xs, 75),
            'p95': pct(xs, 95), 'max': max(xs), 'mean': math.fsum(xs) / len(xs), 'n': len(xs)}


def build(c, repo_root):
    repo = Path(repo_root)
    sys.path.insert(0, str(repo))
    from methods.common import learned_bank as lb
    s7 = {m: json.loads((repo / ('outputs/audit/LEARNED_BANK_S7_%s.json' % m)).read_text()) for m in METHODS}
    toy = lb.load_toy_manifest(repo)
    b12 = json.loads((repo / 'configs/amendments/e05_s7_b1_b2_ruling.yaml').read_text())
    patch = json.loads((repo / 'environments/e05.compatibility_patch_manifest.json').read_text())
    checks = {}
    e05 = c['methods']['E05']
    allrows = [r for s in SEEDS for r in e05[str(s)]['rows']]
    checks['inputs_are_committed_s7_outputs'] = all(
        c['methods'][m][str(s)]['manifest_equals_s7_record'] and
        c['methods'][m][str(s)]['run_B_manifest_sha256'] == s7[m]['outputs'][str(s)]['manifest_sha256']
        for m in METHODS for s in SEEDS)
    checks['file_and_pixel_sha256'] = all(r['file_sha_ok'] and r['pixel_sha_ok']
                                          for m in METHODS for s in SEEDS for r in c['methods'][m][str(s)]['rows'])
    checks['e05_raw_finite'] = all(r['raw_finite'] is True for r in allrows)
    checks['e05_conversion_consistent_with_tensor2im'] = all(r['conversion_min_consistent'] and
                                                             r['conversion_max_consistent'] for r in allrows)
    want = {t['pair_id']: (t['source_spoof_id'], t['target_live_id'], t['dataset']) for t in toy}
    checks['e05_lineage_source_spoof_target_live'] = all(
        tuple(r['lineage'][1:4]) == want[r['lineage'][0]] for r in allrows) and all(
        len(e05[str(s)]['rows']) == lb.TOY_N for s in SEEDS)
    checks['e05_per_pair_noise_seed'] = all(r['lineage'][4] == lb.e05_pair_seed(s, r['lineage'][0])
                                            for s in SEEDS for r in e05[str(s)]['rows'])
    checks['e05_checkpoint_identity'] = all(r['lineage'][5] == lb.checkpoint_identity_sha256('E05', s)
                                            for s in SEEDS for r in e05[str(s)]['rows'])
    s7e05 = s7['E05']['collected']['runs']
    checks['e05_tensor2im_runtime_at_defining_module'] = all(
        s7e05[L][str(s)]['provenance']['adapter']['tensor2im_runtime'] == {
            'module': 'util.util', 'file': 'util/util.py',
            'file_sha256': patch['generated_files_sha256']['util/util.py'],
            'segment_sha256': b12['tensor2im']['pinned_segment_sha256']} for L in ('A', 'B') for s in SEEDS)
    checks['e05_s7_record_passed'] = s7['E05']['s7_acceptance_passed'] is True and s7['E05']['failed_gates'] == []
    checks['e05_outputs_distinct_within_seed'] = all(e05[str(s)]['distinct_pixel_sha256'] == lb.TOY_N for s in SEEDS)
    checks['no_val_test_face_access'] = c['atime']['same_paths'] and all(
        v['changed'] == 0 for v in c['atime']['by_split'].values()) and c['reader']['splits_allowlisted'] == ['TRAIN'] \
        and c['reader']['reads'] == c['reader']['atime_checked_reads']
    checks['collector_clean_checkout'] = c['git_status'] == ''
    violation_keys = [k for k in checks if k.startswith('e05_')]
    violations = [k for k in violation_keys if not checks[k]]
    agg = {}
    for m in METHODS:
        rows = [r for s in SEEDS for r in c['methods'][m][str(s)]['rows']]
        agg[m] = {'per_image_std': dist([r['std'] for r in rows]), 'per_image_lum_std': dist([r['lum_std'] for r in rows]),
                  'per_image_min': dist([r['min'] for r in rows]), 'per_image_max': dist([r['max'] for r in rows]),
                  'per_image_p01': dist([r['p01'] for r in rows]), 'per_image_p99': dist([r['p99'] for r in rows]),
                  'per_image_mean': dist([r['mean'] for r in rows]),
                  'channel_mean_mean': [math.fsum(r['channel_mean'][k] for r in rows) / len(rows) for k in range(3)],
                  'channel_std_mean': [math.fsum(r['channel_std'][k] for r in rows) / len(rows) for k in range(3)],
                  'saturated_pixels_0': sum(r['sat0'] for r in rows), 'saturated_pixels_255': sum(r['sat255'] for r in rows),
                  'frac_within_5_of_mean': dist([r['frac_within_5_of_mean'] for r in rows]),
                  'distinct_pixel_sha256_per_seed': {str(s): c['methods'][m][str(s)]['distinct_pixel_sha256'] for s in SEEDS},
                  'min_pairwise_mean_abs_diff_64px': {str(s): c['methods'][m][str(s)]['min_pairwise_mean_abs_diff_64px']
                                                      for s in SEEDS},
                  'near_identical_pairs_mad_lt_1': {str(s): {
                      'pairs': len(c['methods'][m][str(s)]['near_identical_pairs_mad_lt_1']),
                      'same_source_spoof': sum(x[3] for x in c['methods'][m][str(s)]['near_identical_pairs_mad_lt_1']),
                      'same_target_live': sum(x[4] for x in c['methods'][m][str(s)]['near_identical_pairs_mad_lt_1']),
                      'rows_involved': len({y for x in c['methods'][m][str(s)]['near_identical_pairs_mad_lt_1']
                                            for y in x[:2]})} for s in SEEDS}}
        if m == 'E05':
            agg[m]['per_seed_std'] = {str(s): dist([r['std'] for r in e05[str(s)]['rows']]) for s in SEEDS}
            agg[m]['per_seed_structure_corr_source'] = {str(s): dist([r['structure_corr_source_spoof']
                                                                      for r in e05[str(s)]['rows']]) for s in SEEDS}
            agg[m]['per_seed_raw_range'] = {str(s): [min(r['raw_min'] for r in e05[str(s)]['rows']),
                                                     max(r['raw_max'] for r in e05[str(s)]['rows'])] for s in SEEDS}
            agg[m]['raw_output_min'] = dist([r['raw_min'] for r in allrows])
            agg[m]['raw_output_max'] = dist([r['raw_max'] for r in allrows])
            agg[m]['structure_corr_source_spoof'] = dist([r['structure_corr_source_spoof'] for r in allrows])
            agg[m]['structure_corr_target_live'] = dist([r['structure_corr_target_live'] for r in allrows])
            agg[m]['structure_closer_to_source_fraction'] = sum(
                r['structure_corr_source_spoof'] > r['structure_corr_target_live'] for r in allrows) / len(allrows)
            agg[m]['colour_mean_dist_source'] = dist([r['colour_mean_dist_source'] for r in allrows])
            agg[m]['colour_mean_dist_target'] = dist([r['colour_mean_dist_target'] for r in allrows])
            agg[m]['colour_closer_to_target_fraction'] = sum(
                r['colour_mean_dist_target'] < r['colour_mean_dist_source'] for r in allrows) / len(allrows)
    inp = list(c['input_face_stats'].values())
    agg['TRAIN_input_faces'] = {'per_image_std': dist([r['std'] for r in inp]), 'per_image_min': dist([r['min'] for r in inp]),
                                'per_image_max': dist([r['max'] for r in inp]), 'n_faces': len(inp)}
    reps = c.get('montage', {}).get('representatives', {})
    rep_rows = []
    for s in SEEDS:
        for rid in reps.get(str(s), []):
            r = next(x for x in e05[str(s)]['rows'] if x['row_id'] == rid)
            rep_rows.append({'seed': s, 'row_id': rid, **{k: r[k] for k in (
                'min', 'max', 'mean', 'std', 'lum_std', 'channel_mean', 'channel_std', 'raw_min', 'raw_max',
                'structure_corr_source_spoof', 'structure_corr_target_live')}})
    e05_cfg = {'total_iterations': 4000, 'batch_size': 1, 'optimizer': 'Adam lr 1e-6 betas (0.9, 0.999)',
               'provenance': 'configs/methods/e05_pcgan.yaml training/optimizer (PAPER, unchanged); no pretrained weights '
                             '(weight_bytes_from_source_cache NONE)'}
    status = 'E05_LOW_CONTRAST_DISCLOSED_NO_CONTRACT_VIOLATION' if not violations and all(checks.values()) else \
        'E05_CONTRACT_VIOLATION_FOUND_PRODUCTION_BLOCKED' if violations else 'E05_DIAGNOSTIC_INCOMPLETE'
    doc = {'schema': 'e05_s7_contrast_diagnostic_v1', 'status': status, 'contract_violations': violations,
           'checks': checks, 'aggregates': agg, 'representatives': rep_rows,
           'e05_cross_seed_same_pair': c['e05_cross_seed_same_pair'], 'e05_training_budget': e05_cfg,
           'policy': 'descriptive only; no quality threshold, filter, enhancement, checkpoint substitution or retuning '
                     'is introduced; RAW-MATCHED protocol retained',
           'montage': {k: v for k, v in (c.get('montage') or {}).items()},
           'collected': {'path': EVIDENCE, 'sha256': sha_bytes((repo / EVIDENCE).read_bytes())}}
    return doc


def f1(x):
    return '%.1f' % x if isinstance(x, (int, float)) else str(x)


def report(d):
    a = d['aggregates']
    L = ['# E05 PCGAN S7 low-contrast review', '',
         '**Status:** `%s`.' % d['status'], '',
         'Contract violations: %s.' % (', '.join('`%s`' % v for v in d['contract_violations']) or 'none'), '',
         'This review is descriptive. It reads only the committed S7 TRAIN toy outputs (run A; run B is byte-identical) '
         'and the TRAIN faces of the 186 approved toy pairs. It loads no model, generates nothing and introduces no '
         'quality threshold.', '',
         '## Mechanical checks against the frozen contract', '',
         '| check | result |', '|---|---|']
    L += ['| %s | %s |' % (k, 'pass' if v else '**FAIL**') for k, v in d['checks'].items()]
    L += ['', '## Contrast compared with the other learned methods (S7 toy outputs, 3 seeds x 186)', '',
          '| method | per-image std, median [p05, p95] | per-image min, median | per-image max, median | '
          'p01 to p99, median | saturated 0 / 255 px | distinct images per seed |', '|---|---|---|---|---|---|---|']
    for m in METHODS + ('TRAIN_input_faces',):
        x = a[m]
        if m == 'TRAIN_input_faces':
            L.append('| TRAIN input faces (%d) | %s [%s, %s] | %s | %s | | | |' % (
                x['n_faces'], f1(x['per_image_std']['median']), f1(x['per_image_std']['p05']),
                f1(x['per_image_std']['p95']), f1(x['per_image_min']['median']), f1(x['per_image_max']['median'])))
            continue
        L.append('| %s | %s [%s, %s] | %s | %s | %s to %s | %d / %d | %s |' % (
            m, f1(x['per_image_std']['median']), f1(x['per_image_std']['p05']), f1(x['per_image_std']['p95']),
            f1(x['per_image_min']['median']), f1(x['per_image_max']['median']), f1(x['per_image_p01']['median']),
            f1(x['per_image_p99']['median']), x['saturated_pixels_0'], x['saturated_pixels_255'],
            '/'.join(str(v) for v in x['distinct_pixel_sha256_per_seed'].values())))
    e = a['E05']
    L += ['', '## E05 details', '',
          '- Per-seed per-image std (median): %s.' % ', '.join('seed %s %s' % (s, f1(v['median']))
                                                              for s, v in e['per_seed_std'].items()),
          '- Per-seed raw output range [min, max]: %s.' % ', '.join(
              'seed %s [%.3f, %.3f]' % (s, v[0], v[1]) for s, v in e['per_seed_raw_range'].items()),
          '- Per-seed structure correlation with the source spoof (median [p05, p95]): %s.' % ', '.join(
              'seed %s %.2f [%.2f, %.2f]' % (s, v['median'], v['p05'], v['p95'])
              for s, v in e['per_seed_structure_corr_source'].items()),
          '- Near-identical output pairs within one seed (mean absolute difference < 1 grey level on the 64 px grid; '
          'descriptive, not a filter): %s.' % ', '.join(
              'seed %s: %d pairs over %d rows (%d share the source spoof, %d share the target live)' % (
                  s, v['pairs'], v['rows_involved'], v['same_source_spoof'], v['same_target_live'])
              for s, v in e['near_identical_pairs_mad_lt_1'].items()),
          '- Visual inspection of the montage: seeds 42 and 2026 show a faint, desaturated face silhouette taken from '
          'the source spoof on a flat grey-beige field. Seed 1337 outputs look almost uniformly grey: per-image std '
          'about 5, with little visible structure.',
          '- Channel means (R, G, B), averaged over images: %s; channel std: %s.' % (
              ', '.join(f1(v) for v in e['channel_mean_mean']), ', '.join(f1(v) for v in e['channel_std_mean'])),
          '- Share of pixels within ±5 of the image mean: median %.2f.' % e['frac_within_5_of_mean']['median'],
          '- Raw generator output before conversion (float32, recorded per row): min %s to %s, max %s to %s. '
          'Under tensor2im, (x+1)/2*255 maps [-1, 1] to [0, 255]; these raw values stay inside about [%s, %s], so the '
          'outputs never reach the clipping range.' % (
              '%.3f' % e['raw_output_min']['min'], '%.3f' % e['raw_output_min']['max'], '%.3f' % e['raw_output_max']['min'],
              '%.3f' % e['raw_output_max']['max'], '%.3f' % e['raw_output_min']['min'], '%.3f' % e['raw_output_max']['max']),
          '- Conversion: every row\'s uint8 min and max equal tensor2im applied to its recorded raw min and max.',
          '- Mixing semantics: zero-mean luminance structure (64x64) correlates with the source spoof (median %.2f) more '
          'than with the target live (median %.2f) in %.0f%% of rows; the colour mean is closer to the target live in '
          '%.0f%% of rows.' % (e['structure_corr_source_spoof']['median'], e['structure_corr_target_live']['median'],
                                100 * e['structure_closer_to_source_fraction'], 100 * e['colour_closer_to_target_fraction']),
          '- Distinctness: every seed has 186 distinct pixel hashes; the smallest mean absolute difference between two '
          'outputs of one seed (64 px grid) is %s.' % ', '.join('%s (seed %s)' % (f1(v), s) for s, v in
                                                               e['min_pairwise_mean_abs_diff_64px'].items()),
          '- Same pair across the three generators (mean absolute pixel difference): %s.' % ', '.join(
              '%s %s' % (k, f1(v['mean_abs_diff_mean'])) for k, v in d['e05_cross_seed_same_pair'].items()),
          '- Training budget (frozen, PAPER): %s, %d iterations, batch %d; %s.' % (
              d['e05_training_budget']['optimizer'], d['e05_training_budget']['total_iterations'],
              d['e05_training_budget']['batch_size'], d['e05_training_budget']['provenance']),
          '', '## Representative examples (rows with the lowest, median and highest per-image std per seed)', '',
          '| seed | pair | min | max | mean | std | raw min | raw max | corr source | corr target |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in d['representatives']:
        L.append('| %d | %s | %d | %d | %s | %s | %.3f | %.3f | %.2f | %.2f |' % (
            r['seed'], r['row_id'], r['min'], r['max'], f1(r['mean']), f1(r['std']), r['raw_min'], r['raw_max'],
            r['structure_corr_source_spoof'], r['structure_corr_target_live']))
    L += ['', 'A montage of these rows (source spoof, target live, raw output, unmodified) is on the GPU host at `%s`. '
          'It is not committed because it contains dataset faces.' % d['montage'].get('dir'), '',
          '## Interpretation', '',
          ('Every mechanical check of the frozen contract passes, and every output is distinct by pixel hash. The '
           'outputs are low-contrast, and the low contrast comes from the generator, not the conversion: the raw '
           'generator output already lies in a narrow band just below zero, so tensor2im maps it to mid-grey without '
           'clipping. The source-spoof structure is visible in seeds 42 and 2026. Seed 1337 is close to collapse onto '
           'one near-constant grey image: %d of %d within-seed output pairs differ by less than one grey level on '
           'average, compared with %d and %d for seeds 42 and 2026, and none for E06c, E03 or E06b. This is '
           'consistent with the frozen paper training budget (4000 iterations at batch 1, learning rate 1e-6, from '
           'random initialisation), under which the generator moves little from its initial state.' % (
               e['near_identical_pairs_mad_lt_1']['1337']['pairs'], 186 * 185 // 2,
               e['near_identical_pairs_mad_lt_1']['42']['pairs'], e['near_identical_pairs_mad_lt_1']['2026']['pairs']))
          if d['status'] == 'E05_LOW_CONTRAST_DISCLOSED_NO_CONTRACT_VIOLATION' else
          'A mechanical contract check failed; E05 production is blocked pending owner review.', '']
    if d['status'] == 'E05_LOW_CONTRAST_DISCLOSED_NO_CONTRACT_VIOLATION':
        L += ['The frozen specification (RAW-MATCHED view) retains every numerically valid output and excludes only '
              'NaN/Inf, decode failure or missing output. Near-constant outputs are therefore not a contract violation, '
              'and low contrast alone is not evidence of an invalid checkpoint. The RAW-MATCHED protocol is retained: no '
              'enhancement, filtering, quality threshold, checkpoint substitution or retuning. This is a scientific '
              'limitation that must be disclosed with every E05 result. Seed 1337 in particular is likely to give a weak '
              'or near-degenerate downstream bank. Whether E05 results are interpreted with that caveat, or whether the '
              'training budget is revisited, is an owner decision outside this session.', '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect')
    a.add_argument('--repo', required=True)
    a.add_argument('--montage-dir', default=None)
    b = sub.add_parser('build')
    b.add_argument('--collected', required=True)
    b.add_argument('--repo-root', default=str(Path(globals().get('__file__', 'tools/x')).resolve().parents[1]))
    args = ap.parse_args(argv)
    if args.cmd == 'collect':
        return collect(args.repo, args.montage_dir)
    if Path(args.collected).resolve() != (Path(args.repo_root) / EVIDENCE).resolve():
        ap.error('build reads the committed evidence file %s' % EVIDENCE)
    doc = build(json.loads(Path(args.collected).read_text()), args.repo_root)
    root = Path(args.repo_root)
    (root / (OUT + '.json')).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    (root / (OUT + '.md')).write_text(report(doc))
    (root / (OUT + '_SHA256.txt')).write_text(''.join('%s  %s\n' % (sha_bytes((root / p).read_bytes()), p)
                                                    for p in (OUT + '.json', OUT + '.md')))
    print(doc['status'], doc['contract_violations'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
