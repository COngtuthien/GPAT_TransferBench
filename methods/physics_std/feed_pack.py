"""E04 feed pack v1: the hash-verified TRAIN closure the Physics-STD TF1 controller reads (stdlib + numpy, Python 3.8).

Built on the GPU host by tools/e04_feed_pack.py from the E03 feed pack (pairs_train_v1 rows in stored order, sample
identities, M2 face sha256) and the frozen geometry pre-pass (e04_geometry_prepass_v1). No VAL or TEST row, sample or
metadata enters the pack; no attack_macro column exists.

  pairs_train.json   {"columns": [pair_id, target_live_id, source_spoof_id, dataset], "split": "TRAIN", "rows": [...]}
                     pairs_train_v1 stored order == ascending pair_id (D4)
  samples.json       one row per TRAIN face, sorted by sample_id:
                     [sample_id, role, dataset, split, label_binary, face_png_sha256, q140_row, m0_row, q140_xy_sha256,
                      m0_sha256]  (m0_row / m0_sha256 null for spoof: its target is the constant exact zero)
  q140_xy.f32le      float32 little-endian [n_samples, 140, 2], canonical-256 pixel xy (pre-pass rows)
  live_m0.f32le      float32 little-endian [n_live, 32, 32], A3/A4 live depth targets in [0, 1]
  pack_manifest.json schema, provenance, counts, payload sha256/bytes; not self-hashing (pinned by the consumer)

Input encoding (D8) happens in the controller: RGB uint8 / 255 once. Faces are read from the canonical faces root and
SHA256-checked against samples.json before decoding.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA, VERSION, DIRNAME = 'GPAT_TRANSFERBENCH_E04_FEED_PACK', 1, 'e04_feed_pack_v1'
MANIFEST = 'pack_manifest.json'
PAYLOADS = ('pairs_train.json', 'samples.json', 'q140_xy.f32le', 'live_m0.f32le')
PAIR_COLUMNS = ('pair_id', 'target_live_id', 'source_spoof_id', 'dataset')
SAMPLE_COLUMNS = ('sample_id', 'role', 'dataset', 'split', 'label_binary', 'face_png_sha256', 'q140_row', 'm0_row',
                  'q140_xy_sha256', 'm0_sha256')
COUNTS = {'pairs': 8838, 'samples': 12668, 'live': 3830, 'spoof': 8838}


class PackError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise PackError('E04 feed pack STOP: ' + message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def dumps(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')


def build(out_dir, *, e03_pack, prepass_dir, prepass_manifest_sha256, provenance):
    """Write a fresh pack from a verified E03 pack object and a verified frozen pre-pass directory."""
    from methods.physics_std import prepass
    out = Path(out_dir)
    require(not out.exists(), '%s exists; packs are never overwritten' % out)
    pm, pre_samples, arrays = prepass.load(prepass_dir, prepass_manifest_sha256)
    require(pm['acceptable'] and pm['counts'] == {'samples': COUNTS['samples'], 'live': COUNTS['live'],
                                                  'spoof': COUNTS['spoof'], 'VAL': 0, 'TEST': 0}, 'pre-pass counts')
    rows = [[r['pair_id'], r['target_live_id'], r['source_spoof_id'], r['dataset']] for r in e03_pack.pair_rows('TRAIN')]
    ids = [r[0] for r in rows]
    require(len(rows) == COUNTS['pairs'] and ids == sorted(ids) and len(set(ids)) == len(ids), 'TRAIN pair order')
    by_id = {r['sample_id']: r for r in pre_samples}
    used = sorted({x for r in rows for x in r[1:3]})
    require(used == sorted(by_id), 'pre-pass population == pair members')
    samples = []
    for sid in used:
        p, e = by_id[sid], e03_pack.sample(sid)
        require((p['split'], p['dataset'], p['label_binary'], p['face_png_sha256']) ==
                (e['split'], e['dataset'], e['label_binary'], e['face_png_sha256']) and p['split'] == 'TRAIN',
                'sample identity ' + sid)
        samples.append([sid, p['role'], p['dataset'], p['split'], p['label_binary'], p['face_png_sha256'], p['row'],
                        p['live_row'], p['q140_xy_sha256'], p['m0_sha256']])
    for r in rows:
        a, b = by_id[r[1]], by_id[r[2]]
        require(a['role'] == 'live' and b['role'] == 'spoof' and a['dataset'] == b['dataset'] == r[3],
                'pair roles ' + r[0])
    out.mkdir(parents=True)
    raw = {'pairs_train.json': dumps({'columns': list(PAIR_COLUMNS), 'split': 'TRAIN', 'rows': rows}),
           'samples.json': dumps({'columns': list(SAMPLE_COLUMNS), 'rows': samples}),
           'q140_xy.f32le': np.ascontiguousarray(arrays['q140_xy.f32le'], dtype='<f4').tobytes(),
           'live_m0.f32le': np.ascontiguousarray(arrays['live_m0.f32le'], dtype='<f4').tobytes()}
    for name, data in raw.items():
        (out / name).write_bytes(data)
    manifest = {'schema': SCHEMA, 'version': VERSION, 'counts': dict(COUNTS), 'test_rows': 0, 'val_rows': 0,
                'payloads': {n: {'sha256': sha256_bytes(d), 'bytes': len(d)} for n, d in raw.items()},
                'shapes': {'q140_xy.f32le': [COUNTS['samples'], 140, 2], 'live_m0.f32le': [COUNTS['live'], 32, 32]},
                'pair_id_order_sha256': sha256_bytes('\n'.join(ids).encode()),
                'prepass_manifest_sha256': prepass_manifest_sha256,
                'input_encoding': 'D8: RGB uint8 / 255 once in the controller; no second normalisation',
                'spoof_depth': 'constant exact float32 zeros 32x32; never rendered or stored',
                'forbidden_columns': ['attack_macro'], 'provenance': provenance}
    m = dumps(manifest)
    (out / MANIFEST).write_bytes(m)
    return sha256_bytes(m)


class FeedPack:
    def __init__(self, root, manifest, manifest_sha256, rows, samples, q140, m0):
        self.root, self.manifest, self.manifest_sha256 = root, manifest, manifest_sha256
        self.rows, self.samples, self._q140, self._m0 = rows, samples, q140, m0
        self._q140.setflags(write=False)
        self._m0.setflags(write=False)

    def sample(self, sample_id):
        return dict(self.samples[sample_id])

    def q140_xy(self, sample_id):
        return self._q140[self.samples[sample_id]['q140_row']].copy()

    def live_m0(self, sample_id):
        s = self.samples[sample_id]
        require(s['role'] == 'live', 'depth M0 is stored only for live faces')
        return self._m0[s['m0_row']].copy()


def load(root, expected_manifest_sha256=None):
    """Independent receiver verification; raises PackError on any mismatch, never falls back."""
    root = Path(root)
    raw_m = (root / MANIFEST).read_bytes()
    sha = sha256_bytes(raw_m)
    if expected_manifest_sha256 is not None:
        require(sha == expected_manifest_sha256, 'manifest sha256 %s != pinned %s' % (sha, expected_manifest_sha256))
    m = json.loads(raw_m.decode('utf-8'))
    require(m['schema'] == SCHEMA and m['version'] == VERSION and m['counts'] == COUNTS, 'schema / counts')
    require(sorted(p.name for p in root.iterdir()) == sorted(PAYLOADS + (MANIFEST,)), 'pack file set')
    raw = {}
    for name in PAYLOADS:
        raw[name] = (root / name).read_bytes()
        require(len(raw[name]) == m['payloads'][name]['bytes'] and sha256_bytes(raw[name]) == m['payloads'][name][
            'sha256'], 'payload integrity ' + name)
    pairs, s = (json.loads(raw[n].decode('utf-8')) for n in ('pairs_train.json', 'samples.json'))
    require(dumps(pairs) == raw['pairs_train.json'] and dumps(s) == raw['samples.json'], 'canonical JSON')
    require(pairs['columns'] == list(PAIR_COLUMNS) and pairs['split'] == 'TRAIN' and s['columns'] == list(SAMPLE_COLUMNS),
            'columns')
    q = np.frombuffer(raw['q140_xy.f32le'], '<f4').reshape(m['shapes']['q140_xy.f32le'])
    d = np.frombuffer(raw['live_m0.f32le'], '<f4').reshape(m['shapes']['live_m0.f32le'])
    require(np.isfinite(q).all() and np.isfinite(d).all() and d.min() >= 0 and d.max() <= 1, 'finite / range')
    samples = {}
    for r in s['rows']:
        rec = dict(zip(SAMPLE_COLUMNS, r))
        require(rec['split'] == 'TRAIN' and rec['role'] in ('live', 'spoof') and
                rec['label_binary'] == (0 if rec['role'] == 'live' else 1), 'sample row ' + rec['sample_id'])
        require(sha256_bytes(q[rec['q140_row']].tobytes()) == rec['q140_xy_sha256'], 'q140 row ' + rec['sample_id'])
        if rec['role'] == 'live':
            require(sha256_bytes(d[rec['m0_row']].tobytes()) == rec['m0_sha256'], 'm0 row ' + rec['sample_id'])
        else:
            require(rec['m0_row'] is None and rec['m0_sha256'] is None, 'spoof has no stored depth ' + rec['sample_id'])
        samples[rec['sample_id']] = rec
    ids = [r['sample_id'] for r in (dict(zip(SAMPLE_COLUMNS, x)) for x in s['rows'])]
    require(ids == sorted(ids) and len(ids) == COUNTS['samples'], 'samples sorted / count')
    rows = [tuple(r) for r in pairs['rows']]
    pids = [r[0] for r in rows]
    require(len(rows) == COUNTS['pairs'] and pids == sorted(pids) and len(set(pids)) == len(pids) and
            sha256_bytes('\n'.join(pids).encode()) == m['pair_id_order_sha256'], 'pair order')
    for pid, live, spoof, ds in rows:
        require(samples[live]['role'] == 'live' and samples[spoof]['role'] == 'spoof' and
                samples[live]['dataset'] == samples[spoof]['dataset'] == ds, 'pair ' + pid)
    require(m['test_rows'] == 0 == m['val_rows'], 'TRAIN only')
    return FeedPack(root, m, sha, rows, samples, q.copy(), d.copy())
