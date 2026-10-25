"""Build the E03 feed pack v1 (owner decision D1) on the laptop: Python 3.11 + pyarrow, read-only on every source.

  python tools/e03_feed_pack.py build --out <staging dir> [--m2-root <cache/geometry/landmarks_px256>]

Sources (each SHA256-checked before use):
  manifests/pairs_train_v1.parquet   TRAIN relation (R1), read in stored row order
  manifests/val_pairs_v1.parquet     VAL relation (R2), read in stored row order
  manifests/split_v1.parquet         canonical split/label/dataset of every referenced sample (TEST refused)
  manifests/m2_sample_accounting.parquet   face_png_sha256 of every referenced sample (receiver face check)
  M2 landmarks_px256 shards          every shard + index byte-checked against outputs/audit/M2_CACHE_SHARD_INDEX.csv;
                                     each selected block re-verified with the M2 procedure (gpatbench/preprocess/
                                     m2b.py ChunkShardSet._finalize_all): block sha256, np.load(allow_pickle=False),
                                     shape/dtype, sha256 of the deterministic npy (version 1.0) re-encoding.

Only rows of samples referenced by the two pair relations are taken from the mixed-split shards; no TEST sample,
row or metadata reaches the pack. The values are copied bit-exactly (no normalisation; /256 and flips stay in
methods/stdn/landmarks.py at run time).
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.stdn import feed_pack as fp  # noqa: E402

M2_ROOT = Path('/media/cong/Data/GPAT_TransferBench_runtime/cache/geometry/landmarks_px256')
SHARD_EVIDENCE = 'outputs/audit/M2_CACHE_SHARD_INDEX.csv'
SOURCES = {
    'pairs_train_v1': ('manifests/pairs_train_v1.parquet',
                       'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243', 8838),
    'val_pairs_v1': ('manifests/val_pairs_v1.parquet',
                     '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872', 1905),
    'split_v1': ('manifests/split_v1.parquet', 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d',
                 None),
    'm2_sample_accounting': ('manifests/m2_sample_accounting.parquet', None, None),
}
SPLIT_OF = {'pairs_train_v1': 'TRAIN', 'val_pairs_v1': 'VAL'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def npy_bytes(arr):
    """gpatbench.preprocess.m2b.npy_bytes_any: deterministic .npy v1.0 bytes."""
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), version=(1, 0), allow_pickle=False)
    return buf.getvalue()


def read_relation(name):
    path, digest, rows = SOURCES[name]
    fp.require(sha256_file(ROOT / path) == digest, f'{path} sha256')
    t = pq.read_table(ROOT / path, columns=list(fp.PAIR_COLUMNS) + ['split'])
    fp.require(t.num_rows == rows, f'{path} rows')
    out = t.to_pylist()
    fp.require(all(r['split'] == SPLIT_OF[name] for r in out), f'{path} split column')
    return [[r[c] for c in fp.PAIR_COLUMNS] for r in out]


def m2_index(m2_root):
    """Every shard/index byte-checked against the committed M2 evidence; returns sample_id -> index row."""
    ev = [r for r in csv.DictReader((ROOT / SHARD_EVIDENCE).open()) if r['field'] == 'landmarks_px256']
    fp.require(len(ev) == 81, 'committed evidence lists 81 landmarks_px256 shards')
    expected = {r['shard'] for r in ev} | {r['shard'] + '.index.json' for r in ev}
    fp.require({p.name for p in m2_root.iterdir()} == expected, 'M2 directory holds exactly the evidenced files')
    rows, shards = {}, []
    for r in ev:
        b, i = m2_root / r['shard'], m2_root / (r['shard'] + '.index.json')
        fp.require(sha256_file(b) == r['shard_sha256'] and b.stat().st_size == int(r['bytes']), 'shard ' + r['shard'])
        fp.require(sha256_file(i) == r['index_sha256'], 'index ' + r['shard'])
        idx = json.loads(i.read_text())
        fp.require(idx['field'] == 'landmarks_px256' and idx['shard'] == r['shard'] and
                   idx['shard_sha256'] == r['shard_sha256'] and len(idx['rows']) == int(r['rows']), 'index ' + r['shard'])
        for row in idx['rows']:
            fp.require(row['sample_id'] not in rows, 'duplicate M2 sample ' + row['sample_id'])
            rows[row['sample_id']] = row
        shards.append({k: r[k] for k in ('shard', 'rows', 'bytes', 'shard_sha256', 'index_sha256')})
    return rows, shards, sha256_file(ROOT / SHARD_EVIDENCE)


def build(out, m2_root=M2_ROOT):
    out = Path(out)
    fp.require(not out.exists(), f'{out} exists; the pack is built fresh into an empty staging directory')
    relations = {name: read_relation(name) for name in SPLIT_OF}
    split_tab = pq.read_table(ROOT / SOURCES['split_v1'][0], columns=['sample_id', 'dataset', 'label_binary', 'split'])
    fp.require(sha256_file(ROOT / SOURCES['split_v1'][0]) == SOURCES['split_v1'][1], 'split_v1 sha256')
    split = {r['sample_id']: r for r in split_tab.to_pylist()}
    acc_path = ROOT / SOURCES['m2_sample_accounting'][0]
    acc = pq.read_table(acc_path, columns=['sample_id', 'face_png_sha256', 'final_status']).to_pylist()
    faces = {}
    for r in acc:
        fp.require(r['sample_id'] not in faces, 'duplicate accounting row')
        faces[r['sample_id']] = r
    needed = {}
    for name, rows in relations.items():
        s = SPLIT_OF[name]
        for pid, live, spoof, ds in rows:
            for sid, label in ((live, 0), (spoof, 1)):
                meta = split.get(sid)
                fp.require(meta is not None, 'sample in split_v1 ' + sid)
                fp.require(meta['split'] == s and meta['split'] != 'TEST', f'{sid} is {meta["split"]}, needed {s}')
                fp.require(meta['label_binary'] == label and meta['dataset'] == ds, 'role/dataset ' + sid)
                fp.require(needed.get(sid, (s, label)) == (s, label), 'sample role conflict ' + sid)
                needed[sid] = (s, label)
    test_ids = {k for k, v in split.items() if v['split'] == 'TEST'}
    fp.require(not (set(needed) & test_ids), 'no TEST sample referenced')
    index, shards, evidence_sha = m2_index(Path(m2_root))
    blobs, ordered = {}, sorted(needed)
    sample_rows, arrays, used_shards = [], [], set()
    for i, sid in enumerate(ordered):
        row = index.get(sid)
        fp.require(row is not None, 'M2 landmarks_px256 present for ' + sid)
        if row['shard'] not in blobs:
            blobs[row['shard']] = (Path(m2_root) / row['shard']).read_bytes()
        blob = blobs[row['shard']][row['byte_offset']:row['byte_offset'] + row['byte_length']]
        fp.require(len(blob) == row['byte_length'] and fp.sha256_bytes(blob) == row['block_sha256'], 'block ' + sid)
        arr = np.load(io.BytesIO(blob), allow_pickle=False)
        fp.require(list(arr.shape) == row['shape'] == [68, 2] and arr.dtype.str == row['dtype'] == '<f4' and
                   row['codec_id'] == 'npy1', 'shape/dtype ' + sid)
        fp.require(fp.sha256_bytes(npy_bytes(arr)) == row['npy_sha256'], 'decoded payload ' + sid)
        fp.require(np.isfinite(arr).all(), 'finite ' + sid)
        acc_row = faces.get(sid)
        fp.require(acc_row is not None and acc_row['final_status'] == 'COMPLETE' and
                   len(acc_row['face_png_sha256'] or '') == 64, 'M2 COMPLETE face ' + sid)
        s, label = needed[sid]
        sample_rows.append([sid, s, split[sid]['dataset'], label, acc_row['face_png_sha256'], i, row['shard'],
                            row['block_sha256'], row['npy_sha256'], fp.sha256_bytes(fp.landmark_bytes(arr))])
        arrays.append(arr)
        used_shards.add(row['shard'])
    payload = {
        'pairs_train.json': fp.dumps({'split': 'TRAIN', 'source': SOURCES['pairs_train_v1'][0],
                                      'source_sha256': SOURCES['pairs_train_v1'][1], 'columns': list(fp.PAIR_COLUMNS),
                                      'rows': relations['pairs_train_v1']}),
        'pairs_val.json': fp.dumps({'split': 'VAL', 'source': SOURCES['val_pairs_v1'][0],
                                    'source_sha256': SOURCES['val_pairs_v1'][1], 'columns': list(fp.PAIR_COLUMNS),
                                    'rows': relations['val_pairs_v1']}),
        'samples.json': fp.dumps({'columns': list(fp.SAMPLE_COLUMNS), 'rows': sample_rows}),
        'landmarks_px256.f32le': b''.join(fp.landmark_bytes(a) for a in arrays),
    }
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    by_split = {s: sum(v[0] == s for v in needed.values()) for s in ('TRAIN', 'VAL')}
    manifest = {
        'schema': fp.SCHEMA, 'version': fp.VERSION,
        'serialization': {'json': "UTF-8, json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\\n'",
                          'landmarks': 'float32 little-endian (<f4), C order, shape [unique_samples, 68, 2]; row i = '
                                       'samples.json row i; exact M2 landmarks_px256 pixel values',
                          'sample_order': 'sample_id ascending, byte order', 'pair_order': 'source parquet stored order',
                          'manifest_hashing': 'non-circular: this file is not listed in payloads; its sha256 is pinned '
                                              'by the consumer'},
        'source_repository_commit': git('rev-parse', 'HEAD'),
        'source_repository_dirty': bool(git('status', '--porcelain')),
        'builder': {'path': 'tools/e03_feed_pack.py', 'sha256': sha256_file(ROOT / 'tools/e03_feed_pack.py'),
                    'verifier': 'methods/stdn/feed_pack.py',
                    'verifier_sha256': sha256_file(ROOT / 'methods/stdn/feed_pack.py'),
                    'python': sys.version.split()[0], 'numpy': np.__version__},
        'frozen_sources': {k: {'path': v[0], 'sha256': sha256_file(ROOT / v[0])} for k, v in SOURCES.items()},
        'm2_landmarks_px256': {'evidence': SHARD_EVIDENCE, 'evidence_sha256': evidence_sha,
                               'shards_verified': len(shards), 'indexes_verified': len(shards),
                               'rows_indexed': len(index), 'shards_used': sorted(used_shards),
                               'shard_evidence': shards,
                               'block_procedure': 'block sha256 == index.block_sha256; np.load(allow_pickle=False); '
                                                  'shape [68,2], dtype <f4, codec npy1; sha256(npy v1.0 re-encode) == '
                                                  'index.npy_sha256; finite'},
        'counts': {'pairs': {'TRAIN': len(relations['pairs_train_v1']), 'VAL': len(relations['val_pairs_v1'])},
                   'unique_samples': len(ordered), 'landmark_rows': len(arrays), 'samples_by_split': by_split,
                   'samples_by_role': {'TRAIN_live': sum(v == ('TRAIN', 0) for v in needed.values()),
                                       'TRAIN_spoof': sum(v == ('TRAIN', 1) for v in needed.values()),
                                       'VAL_live': sum(v == ('VAL', 0) for v in needed.values()),
                                       'VAL_spoof': sum(v == ('VAL', 1) for v in needed.values())},
                   'm2_rows_not_taken': len(index) - len(ordered)},
        'pair_id_order_sha256': {s: fp.sha256_bytes('\n'.join(r[0] for r in relations[n]).encode())
                                 for n, s in SPLIT_OF.items()},
        'test_rows': 0, 'test_samples_referenced': len(set(needed) & test_ids),
        'payloads': {n: {'sha256': fp.sha256_bytes(b), 'bytes': len(b)} for n, b in payload.items()},
    }
    stage = out.with_name(out.name + '.partial')
    fp.require(not stage.exists(), f'{stage} exists')
    stage.mkdir(parents=True)
    for n, b in payload.items():
        (stage / n).write_bytes(b)
    (stage / fp.MANIFEST).write_bytes(fp.dumps(manifest))
    pack = fp.load(stage)                          # self-check with the receiver code before publishing
    os.replace(stage, out)
    return pack.manifest_sha256, manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--m2-root', default=str(M2_ROOT))
    v = sub.add_parser('verify')
    v.add_argument('--pack', required=True)
    v.add_argument('--manifest-sha256')
    args = ap.parse_args(argv)
    if args.cmd == 'build':
        digest, m = build(args.out, args.m2_root)
    else:
        pack = fp.load(args.pack, args.manifest_sha256)
        digest, m = pack.manifest_sha256, pack.manifest
    print(json.dumps({'manifest_sha256': digest, 'counts': m['counts'], 'test_rows': m['test_rows']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
