"""E03 feed pack v1: the minimal hash-verified TRAIN+VAL data closure for the STDN TF1 controller.

Owner decision D1 (2026-10-25): the TF1 runtime (Python 3.8, no pyarrow) never reads the parquet manifests or the
mixed-split M2 shards. tools/e03_feed_pack.py builds this pack on the laptop from the frozen pair manifests and the
original M2 landmarks_px256 cache; the controller verifies it here (stdlib + numpy only) before any graph exists.

Layout (one directory, every file listed in pack_manifest.json except the manifest itself):

  pairs_train.json        pairs_train_v1 rows in stored order: [pair_id, target_live_id, source_spoof_id, dataset]
  pairs_val.json          val_pairs_v1 rows in stored order, same columns
  samples.json            one row per referenced sample, sorted by sample_id (byte order):
                          [sample_id, split, dataset, label_binary, face_png_sha256, landmark_row, m2_shard,
                           m2_block_sha256, m2_npy_sha256, landmarks_le_f4_sha256]
  landmarks_px256.f32le   float32 little-endian, C order, shape [n_samples, 68, 2]; row i = samples.json row i;
                          the exact M2 landmarks_px256 values (no normalisation, rescaling or reordering)
  pack_manifest.json      schema, provenance, counts, payload sha256/bytes. Non-circular: it does not hash itself;
                          its own sha256 is pinned by the consumer (methods/stdn/production_io.PACK_MANIFEST_SHA256).

JSON files are UTF-8, json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\\n'. No TEST sample,
pair row or metadata is ever written into the pack.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'GPAT_TRANSFERBENCH_E03_FEED_PACK'
VERSION = 1
DIRNAME = 'e03_feed_pack_v1'
MANIFEST = 'pack_manifest.json'
PAYLOADS = ('pairs_train.json', 'pairs_val.json', 'samples.json', 'landmarks_px256.f32le')
PAIR_COLUMNS = ('pair_id', 'target_live_id', 'source_spoof_id', 'dataset')
SAMPLE_COLUMNS = ('sample_id', 'split', 'dataset', 'label_binary', 'face_png_sha256', 'landmark_row', 'm2_shard',
                  'm2_block_sha256', 'm2_npy_sha256', 'landmarks_le_f4_sha256')
SPLITS = {'TRAIN': 'pairs_train.json', 'VAL': 'pairs_val.json'}
DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')
LANDMARK_SHAPE = (68, 2)
LANDMARK_DTYPE = '<f4'


class PackError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise PackError('E03 feed pack STOP: ' + message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def dumps(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')


def landmark_bytes(arr):
    """Canonical payload bytes of one landmark block: float32 little-endian, C order."""
    a = np.asarray(arr)
    require(a.shape == LANDMARK_SHAPE and a.dtype.str == LANDMARK_DTYPE and np.isfinite(a).all(),
            'landmarks_px256 must be finite <f4 [68, 2]')
    return np.ascontiguousarray(a).tobytes(order='C')


class FeedPack:
    """A verified pack. Construct only through load(); every accessor returns copies."""

    def __init__(self, root, manifest, manifest_sha256, train, val, samples, landmarks):
        self.root, self.manifest, self.manifest_sha256 = root, manifest, manifest_sha256
        self.rows = {'TRAIN': train, 'VAL': val}
        self.samples = samples
        self._landmarks = landmarks
        self._landmarks.setflags(write=False)

    def pair_rows(self, split):
        return [dict(zip(PAIR_COLUMNS, r)) for r in self.rows[split]]

    def sample(self, sample_id):
        return dict(self.samples[sample_id])

    def landmarks_px256(self, sample_id):
        return self._landmarks[self.samples[sample_id]['landmark_row']].copy()


def load(root, expected_manifest_sha256=None):
    """Independent receiver verification. Raises PackError on any mismatch; never falls back."""
    root = Path(root)
    raw_manifest = (root / MANIFEST).read_bytes()
    manifest_sha = sha256_bytes(raw_manifest)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, f'manifest sha256 {manifest_sha} != pinned '
                                                          f'{expected_manifest_sha256}')
    m = json.loads(raw_manifest.decode('utf-8'))
    require(m.get('schema') == SCHEMA and m.get('version') == VERSION, 'schema/version')
    require(sorted(m['payloads']) == sorted(PAYLOADS), 'payload set')
    present = sorted(p.name for p in root.iterdir())
    require(present == sorted(PAYLOADS + (MANIFEST,)), f'unexpected pack files {present}')
    raw = {}
    for name in PAYLOADS:
        raw[name] = (root / name).read_bytes()
        require(len(raw[name]) == m['payloads'][name]['bytes'] and
                sha256_bytes(raw[name]) == m['payloads'][name]['sha256'], f'payload integrity {name}')
    docs = {n: json.loads(raw[n].decode('utf-8')) for n in PAYLOADS if n.endswith('.json')}
    for name in docs:
        require(dumps(docs[name]) == raw[name], f'non-canonical JSON {name}')
    s = docs['samples.json']
    require(s['columns'] == list(SAMPLE_COLUMNS), 'samples columns')
    ids = [r[0] for r in s['rows']]
    require(ids == sorted(ids) and len(set(ids)) == len(ids), 'samples sorted and unique')
    n = len(ids)
    require(m['counts']['unique_samples'] == n == m['counts']['landmark_rows'], 'sample count')
    lm = np.frombuffer(raw['landmarks_px256.f32le'], dtype=LANDMARK_DTYPE)
    require(lm.size == n * 68 * 2, 'landmark payload size')
    lm = lm.reshape((n,) + LANDMARK_SHAPE).copy()
    samples = {}
    for i, r in enumerate(s['rows']):
        rec = dict(zip(SAMPLE_COLUMNS, r))
        require(rec['split'] in SPLITS and rec['dataset'] in DATASETS and rec['label_binary'] in (0, 1) and
                rec['landmark_row'] == i, 'sample row ' + rec['sample_id'])
        require(sha256_bytes(landmark_bytes(lm[i])) == rec['landmarks_le_f4_sha256'], 'landmark row ' + rec['sample_id'])
        samples[rec['sample_id']] = rec
    rows = {}
    for split, name in SPLITS.items():
        d = docs[name]
        require(d['columns'] == list(PAIR_COLUMNS) and d['split'] == split, name + ' header')
        require(len(d['rows']) == m['counts']['pairs'][split], name + ' row count')
        require(sha256_bytes('\n'.join(r[0] for r in d['rows']).encode()) == m['pair_id_order_sha256'][split],
                name + ' pair order')
        for pid, live, spoof, ds in d['rows']:
            a, b = samples.get(live), samples.get(spoof)
            require(a is not None and b is not None and live != spoof, 'pair members present ' + pid)
            require(a['split'] == b['split'] == split and a['dataset'] == b['dataset'] == ds, 'pair split ' + pid)
            require(a['label_binary'] == 0 and b['label_binary'] == 1, 'target_live is live, source_spoof is spoof '
                    + pid)
        require(len({r[0] for r in d['rows']}) == len(d['rows']), name + ' unique pair ids')
        rows[split] = [tuple(r) for r in d['rows']]
    used = {x for split in rows for r in rows[split] for x in r[1:3]}
    require(used == set(samples), 'every sample is referenced by a pair (minimal pack)')
    require(sum(r['split'] == 'TRAIN' for r in samples.values()) == m['counts']['samples_by_split']['TRAIN'] and
            sum(r['split'] == 'VAL' for r in samples.values()) == m['counts']['samples_by_split']['VAL'],
            'split sample counts')
    require(m['test_rows'] == 0 and all(r['split'] != 'TEST' for r in samples.values()), 'TEST exclusion')
    return FeedPack(root, m, manifest_sha, rows['TRAIN'], rows['VAL'], samples, lm)
