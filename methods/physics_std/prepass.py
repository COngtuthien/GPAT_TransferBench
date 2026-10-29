"""E04 fixed geometry pre-pass (owner resolution D1-D3, configs/amendments/e04_production_resolution.yaml).

CPU only, in gpat-m6-e04-geometry (Python 3.12), with the qualified M6D3a torch settings. TRAIN only: every
target_live and source_spoof face of pairs_train_v1, taken from the hash-verified E03 feed pack (which binds
pairs_train_v1 a5e4fdae..., split_v1 and the M2 landmarks_px256 cache); VAL rows of that pack are never touched.

Per face, unchanged qualified components only:
  verify the canonical PNG against its M2 face_png_sha256 -> cv2.imdecode(IMREAD_COLOR) (BGR, the qualified OpenCV
  path; checked equal to cv2.imread of the same file) -> pts = landmarks_px256.T float32 [2,68] -> official
  parse_roi_box_from_landmark -> official crop_img -> cv2.resize(120x120, INTER_LINEAR) -> FixedGeometryRegressor.fit
  (pinned MobileNet, NormalizeGjz, param*std+mean) -> GeometryAdapter.reconstruct (official recon_vers dense +
  similar_transform with the float roi) -> Q140 xy = dense[q140, :2].
  live only: DepthRenderer.render (official depth()/Sim3DR, A4 uint8 INTER_AREA 32x32 / 255 / clip) -> M0 + z-buffer
  mask. spoof: never rendered; M0 is the constant exact-zero target and is not stored.
Failures (non-finite landmarks, empty box, non-finite fit/mesh, empty mask, degenerate mesh) raise; there is no
fallback crop, tracker, redetection or per-sample repair.

Output directory (fresh): samples.json + raw little-endian payloads + prepass_manifest.json (non-circular).
"""
import ast
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA, VERSION = 'GPAT_TRANSFERBENCH_E04_GEOMETRY_PREPASS', 1
E03_PACK_DIRNAME = 'e03_feed_pack_v1'
E03_PACK_MANIFEST_SHA256 = 'c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833'
PAIRS_TRAIN_SHA256 = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
RULING = ('configs/amendments/e04_production_resolution.yaml',
          '94c9c37bdb23e6e34673fa65e8e4dfc623b961f9e9c10f5e4f0d241840b5505e')
M6D3A_EVIDENCE = 'outputs/audit/M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION.json'
POPULATION = {'live': 3830, 'spoof': 8838}
SIZE, CROP = 256, 120
MANIFEST = 'prepass_manifest.json'
PAYLOADS = {   # name: (dtype, per-row shape, rows)
    'fitted_params.f32le': ('<f4', (62,), 'all'), 'roi_box.f64le': ('<f8', (4,), 'all'),
    'q140_xy.f32le': ('<f4', (140, 2), 'all'), 'live_m0.f32le': ('<f4', (32, 32), 'live'),
    'live_face_mask.u8': ('|u1', (256, 256), 'live'), 'live_dense_mesh.f32le': ('<f4', (38365, 3), 'live')}
SAMPLE_COLUMNS = ('sample_id', 'role', 'dataset', 'split', 'label_binary', 'face_png_sha256', 'landmarks_le_f4_sha256',
                  'row', 'live_row', 'roi_rounded', 'crop_outside_frame', 'crop_zero_padded_pixels',
                  'fitted_sha256', 'q140_xy_sha256', 'm0_sha256', 'mask_pixels', 'mesh_sha256')
TORCH_SETTINGS = ('torch.set_num_threads(1)', 'torch.manual_seed(0)', 'torch.use_deterministic_algorithms(True)',
                  'cudnn.benchmark=False', 'cudnn.deterministic=True', 'matmul/cudnn allow_tf32=False',
                  'device=cpu')


class PrepassError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise PrepassError('E04 PREPASS STOP: ' + message)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def dumps(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')


def official_crop_functions(root):
    """parse_roi_box_from_landmark and crop_img compiled from the pinned utils/functions.py, verified against the
    113-file pinned-blob hashes recorded by M6D3a (the E04 source verifier hashes only the files it binds)."""
    pinned = json.loads((root / M6D3A_EVIDENCE).read_text())['source_integrity']['source_files_sha256']
    path = root / 'third_party/source_cache/tddfa_v2/utils/functions.py'
    raw = path.read_bytes()
    require(sha256_bytes(raw) == pinned['utils/functions.py'], 'pinned utils/functions.py changed')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and
             n.name in ('parse_roi_box_from_landmark', 'crop_img')]
    scope = {'sqrt': math.sqrt, 'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['parse_roi_box_from_landmark'], scope['crop_img'], pinned['utils/functions.py']


class Firewall:
    """Audit hook: canonical faces only for the listed TRAIN samples; no manifests, runs/ or VAL/TEST face; writes
    only into the output directory. cv2.imread is C-level and unaudited, so it is called only on allowlisted paths."""

    def __init__(self, faces_root, allowed, out_dir, runtime_root):
        self.faces, self.allowed = str(faces_root) + '/', set(allowed)
        self.out, self.rt = str(out_dir) + '/', str(runtime_root) + '/'
        self.reads, self.denied = {}, []

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        s = str(Path(os.fsdecode(args[0])).absolute())
        mode = args[1] if isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        write = any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)
        bad = None
        if s.startswith(self.faces):
            if s not in self.allowed:
                bad = 'face outside the TRAIN pre-pass population'
            else:
                self.reads[s] = self.reads.get(s, 0) + 1
        elif '/manifests/' in s and 'source_cache' not in s:
            bad = 'manifest read'
        elif '/runs/' in s and s.startswith(self.rt):
            bad = 'runs/ access'
        elif write and not s.startswith(self.out):
            bad = 'write outside the pre-pass output directory'
        if bad:
            self.denied.append([bad, s])
            raise PermissionError('E04 pre-pass firewall: %s: %s' % (bad, s))


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def population(pack):
    rows = pack.pair_rows('TRAIN')
    ids = [r['pair_id'] for r in rows]
    require(len(rows) == 8838 and ids == sorted(ids) and len(set(ids)) == len(ids),
            'pairs_train_v1 stored order must be 8838 unique ascending pair_ids')
    live = sorted({r['target_live_id'] for r in rows})
    spoof = sorted({r['source_spoof_id'] for r in rows})
    require((len(live), len(spoof)) == (POPULATION['live'], POPULATION['spoof']) and not set(live) & set(spoof),
            'TRAIN population 3830 live / 8838 spoof, disjoint')
    for sid, role in [(x, 'live') for x in live] + [(x, 'spoof') for x in spoof]:
        s = pack.sample(sid)
        require(s['split'] == 'TRAIN' and s['label_binary'] == (0 if role == 'live' else 1), 'TRAIN label ' + sid)
    return sorted([(x, 'live') for x in live] + [(x, 'spoof') for x in spoof])


def crop_statistics(roi):
    sx, sy, ex, ey = [int(round(v)) for v in roi]           # crop_img's own rounding
    dh, dw = ey - sy, ex - sx
    ox = max(0, min(ex, SIZE) - max(sx, 0))
    oy = max(0, min(ey, SIZE) - max(sy, 0))
    return [sx, sy, ex, ey], [sx < 0, sy < 0, ex > SIZE, ey > SIZE], int(dh * dw - ox * oy)


def run(out_dir, *, root, runtime_root, limit=None):
    """Generate the pre-pass into a fresh out_dir. `limit` (qualification smoke only) truncates the population and
    makes the output non-acceptable (recorded in the manifest)."""
    import cv2
    root, runtime_root, out = Path(root), Path(runtime_root), Path(out_dir)
    require(not out.exists(), '%s exists; the pre-pass writes only into a fresh directory' % out)
    require(sha256_file(root / RULING[0]) == RULING[1], 'owner resolution sha256')
    sys.path.insert(0, str(root))
    from methods.stdn import feed_pack
    pack = feed_pack.load(runtime_root / 'data/processed' / E03_PACK_DIRNAME, E03_PACK_MANIFEST_SHA256)
    require(pack.manifest['frozen_sources']['pairs_train_v1']['sha256'] == PAIRS_TRAIN_SHA256, 'pairs_train_v1 binding')
    pop = population(pack)
    if limit is not None:
        pop = pop[:limit]
    faces = runtime_root / 'data/processed/faces_256'
    paths = {sid: faces / pack.sample(sid)['dataset'] / (sid + '.png') for sid, _ in pop}
    torch = configure_torch()
    from methods.physics_std.geometry import GeometryAdapter
    from methods.physics_std.regressor import FixedGeometryRegressor
    from methods.physics_std.renderer import DepthRenderer
    parse, crop, functions_sha = official_crop_functions(root)
    regressor, geometry, renderer = FixedGeometryRegressor(device='cpu'), GeometryAdapter(), DepthRenderer()
    require(renderer.is_official, 'official Sim3DR renderer')
    out.mkdir(parents=True)
    fw = Firewall(faces, [str(p) for p in paths.values()], out.resolve(), runtime_root)
    sys.addaudithook(fw)
    n, n_live = len(pop), sum(r == 'live' for _, r in pop)
    arrays = {k: np.zeros(((n if rows == 'all' else n_live),) + shape, dtype=dt)
              for k, (dt, shape, rows) in PAYLOADS.items()}
    rows, t0, live_row, roi_dtypes = [], time.monotonic(), 0, set()
    for i, (sid, role) in enumerate(pop):
        s = pack.sample(sid)
        raw = paths[sid].read_bytes()
        require(sha256_bytes(raw) == s['face_png_sha256'], 'canonical face sha256 ' + sid)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        require(img is not None and img.shape == (SIZE, SIZE, 3) and img.dtype == np.uint8, 'BGR uint8 256 ' + sid)
        require(np.array_equal(img, cv2.imread(str(paths[sid]), cv2.IMREAD_COLOR)), 'imdecode == imread ' + sid)
        lm = pack.landmarks_px256(sid)
        require(lm.shape == (68, 2) and lm.dtype == np.float32 and np.isfinite(lm).all(), 'finite landmarks ' + sid)
        pts = np.ascontiguousarray(lm.T)
        roi = parse(pts)
        roi_dtypes.add(','.join(sorted({type(v).__name__ + ':' + str(getattr(v, 'dtype', '')) for v in roi})))
        require(all(math.isfinite(float(v)) for v in roi) and roi[2] > roi[0] and roi[3] > roi[1], 'roi extent ' + sid)
        c = crop(img, roi)
        require(c.ndim == 3 and c.shape[0] > 0 and c.shape[1] > 0, 'empty crop ' + sid)
        fitted = regressor.fit(cv2.resize(c, dsize=(CROP, CROP), interpolation=cv2.INTER_LINEAR))['fitted_parameters']
        mesh = geometry.reconstruct(fitted, roi)
        q = np.ascontiguousarray(mesh['q140_vertices'][:, :2], dtype=np.float32)
        require(np.isfinite(q).all(), 'finite Q140 ' + sid)
        rounded, outside, padded = crop_statistics(roi)
        arrays['fitted_params.f32le'][i] = fitted
        arrays['roi_box.f64le'][i] = [float(v) for v in roi]
        arrays['q140_xy.f32le'][i] = q
        rec = {'sample_id': sid, 'role': role, 'dataset': s['dataset'], 'split': s['split'],
               'label_binary': s['label_binary'], 'face_png_sha256': s['face_png_sha256'],
               'landmarks_le_f4_sha256': s['landmarks_le_f4_sha256'], 'row': i, 'live_row': None,
               'roi_rounded': rounded, 'crop_outside_frame': outside, 'crop_zero_padded_pixels': padded,
               'fitted_sha256': sha256_bytes(fitted.tobytes()), 'q140_xy_sha256': sha256_bytes(q.tobytes()),
               'm0_sha256': None, 'mask_pixels': None, 'mesh_sha256': None}
        if role == 'live':
            r = renderer.render(mesh['dense_vertices'], mesh['triangles'])
            d, m = r['depth'], r['face_mask_256']
            require(d.shape == (32, 32) and d.dtype == np.float32 and np.isfinite(d).all() and
                    0 <= d.min() <= d.max() <= 1 and np.count_nonzero(d) > 0, 'live M0 ' + sid)
            require(m.shape == (SIZE, SIZE) and m.any(), 'live face mask ' + sid)
            dense = np.ascontiguousarray(mesh['dense_vertices'], dtype=np.float32)
            arrays['live_m0.f32le'][live_row] = d
            arrays['live_face_mask.u8'][live_row] = m.astype(np.uint8)
            arrays['live_dense_mesh.f32le'][live_row] = dense
            rec.update(live_row=live_row, m0_sha256=sha256_bytes(d.tobytes()), mask_pixels=int(m.sum()),
                       mesh_sha256=sha256_bytes(dense.tobytes()))
            live_row += 1
        rows.append([rec[k] for k in SAMPLE_COLUMNS])
    seconds = time.monotonic() - t0
    require(not fw.denied and sorted(fw.reads) == sorted(str(p) for p in paths.values()), 'firewall / read set')
    payloads = {}
    for name, a in arrays.items():
        raw = np.ascontiguousarray(a).tobytes(order='C')
        (out / name).write_bytes(raw)
        payloads[name] = {'sha256': sha256_bytes(raw), 'bytes': len(raw), 'dtype': PAYLOADS[name][0],
                          'shape': list(a.shape)}
    samples = dumps({'columns': list(SAMPLE_COLUMNS), 'rows': rows})
    (out / 'samples.json').write_bytes(samples)
    payloads['samples.json'] = {'sha256': sha256_bytes(samples), 'bytes': len(samples)}
    import scipy
    import torchvision
    manifest = {
        'schema': SCHEMA, 'version': VERSION, 'acceptable': limit is None,
        'limit': limit, 'counts': {'samples': n, 'live': n_live, 'spoof': n - n_live, 'VAL': 0, 'TEST': 0},
        'payloads': payloads, 'owner_resolution': {'path': RULING[0], 'sha256': RULING[1]},
        'inputs': {'e03_feed_pack_manifest_sha256': E03_PACK_MANIFEST_SHA256, 'pairs_train_v1_sha256': PAIRS_TRAIN_SHA256,
                   'faces_root': str(faces), 'faces_read': len(fw.reads), 'faces_read_twice_by_cv2_imread_check': True},
        'geometry': {'source_pin': geometry.config['external_assets']['geometry_engine']['pinned_commit'],
                     'utils_functions_sha256': functions_sha,
                     'q140_sha256': geometry.q['vertex_indices_sha256'], 'assets': {k: v['sha256'] for k, v in
                                                                                   regressor.assets.items()},
                     'regressor_checkpoint_keys': regressor.checkpoint_keys,
                     'sim3dr_extension_sha256': sha256_file(renderer._capture.extension.__file__),
                     'crop': 'parse_roi_box_from_landmark(landmarks_px256.T float32) -> crop_img -> '
                             'cv2.resize(120, INTER_LINEAR)', 'roi_value_types': sorted(roi_dtypes)},
        'runtime': {'python': sys.version.split()[0], 'executable': sys.executable, 'numpy': np.__version__,
                    'torch': torch.__version__, 'torchvision': torchvision.__version__, 'opencv': cv2.__version__,
                    'scipy': scipy.__version__, 'platform': platform.platform(), 'torch_settings': list(TORCH_SETTINGS),
                    'torch_num_threads': torch.get_num_threads(), 'device': 'cpu',
                    'GPAT_E04_GEOMETRY_RUNTIME': os.environ.get('GPAT_E04_GEOMETRY_RUNTIME')},
        'firewall': {'denied': fw.denied, 'face_reads': len(fw.reads)},
        'seconds': seconds, 'spoof_depth': 'not rendered; constant exact float32 zeros 32x32 target (not stored)'}
    raw = dumps(manifest)
    (out / MANIFEST).write_bytes(raw)
    return {'manifest_sha256': sha256_bytes(raw), 'seconds': seconds, 'samples': n}


def load(root, expected_manifest_sha256=None):
    """Verify a pre-pass directory (stdlib + numpy; Python 3.8 compatible). Returns (manifest, samples, arrays)."""
    root = Path(root)
    raw = (root / MANIFEST).read_bytes()
    if expected_manifest_sha256 is not None:
        require(sha256_bytes(raw) == expected_manifest_sha256, 'pre-pass manifest sha256')
    m = json.loads(raw.decode('utf-8'))
    require(m['schema'] == SCHEMA and m['version'] == VERSION, 'pre-pass schema')
    require(sorted(p.name for p in root.iterdir()) == sorted(list(PAYLOADS) + ['samples.json', MANIFEST]),
            'pre-pass file set')
    arrays = {}
    for name, rec in m['payloads'].items():
        data = (root / name).read_bytes()
        require(len(data) == rec['bytes'] and sha256_bytes(data) == rec['sha256'], 'payload integrity ' + name)
        if name != 'samples.json':
            arrays[name] = np.frombuffer(data, dtype=rec['dtype']).reshape(rec['shape'])
    s = json.loads((root / 'samples.json').read_bytes().decode('utf-8'))
    require(s['columns'] == list(SAMPLE_COLUMNS), 'samples columns')
    samples = [dict(zip(SAMPLE_COLUMNS, r)) for r in s['rows']]
    require([r['row'] for r in samples] == list(range(len(samples))) and
            [r['sample_id'] for r in samples] == sorted(r['sample_id'] for r in samples), 'sample order')
    require(all(r['split'] == 'TRAIN' for r in samples) and m['counts']['TEST'] == 0 == m['counts']['VAL'],
            'TRAIN only')
    for r in samples:
        require(sha256_bytes(arrays['q140_xy.f32le'][r['row']].tobytes()) == r['q140_xy_sha256'], 'q140 ' + r['sample_id'])
        if r['role'] == 'live':
            require(sha256_bytes(arrays['live_m0.f32le'][r['live_row']].tobytes()) == r['m0_sha256'],
                    'm0 ' + r['sample_id'])
        else:
            require(r['live_row'] is None and r['m0_sha256'] is None, 'spoof has no rendered depth ' + r['sample_id'])
    return m, samples, arrays
