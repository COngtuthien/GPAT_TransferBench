#!/usr/bin/env python3
"""Synthetic-only E04 fixed geometry qualification and read-only evidence checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEAD = '17c46f833a31e0a8a038cd013d12bbd888f2e18f'
BASE = 'outputs/audit/M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError('STOP_AND_REPORT: ' + reason)


class SyntheticAccessAudit:
    """Reject image files and scientific data paths before any runtime imports."""
    def __init__(self):
        self.opens = 0
        self.denied = []

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        self.opens += 1
        image = path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.mp4', '.avi'}
        data = (path.is_relative_to(ROOT / 'data') or path.is_relative_to(ROOT / 'cache') or
                path.is_relative_to(ROOT / 'manifests') or path.suffix == '.parquet' or
                ('GPAT_TransferBench_runtime' in path.parts and
                 any(p in path.parts for p in ('data', 'cache', 'runs', 'banks'))))
        if image or data:
            self.denied.append(str(path))
            raise RuntimeError('STOP_AND_REPORT: scientific data/image access: ' + str(path))

    def report(self):
        return {'python_opens': self.opens, 'denied': self.denied,
                'benchmark_images_decoded': 0, 'TEST_scientific_data_access': False,
                'scope': 'Python open audit; native inference receives generated arrays only'}


def source_integrity():
    source = ROOT / 'third_party/source_cache/tddfa_v2'
    command = ['git', '-C', str(source)]
    head = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip()
    require(head == '1b6c67601abffc1e9f248b291708aef0e43b55ae', 'source pin')
    provenance = json.loads((ROOT / 'outputs/audit/M6A5_TDDFA_ASSET_PROVENANCE.json').read_text())
    absent = {a['path'] for a in provenance['assets']}
    absent.remove('FaceBoxes_weights/FaceBoxesProd.pth')
    absent.add('FaceBoxes/weights/FaceBoxesProd.pth')
    hashes = {}
    for row in subprocess.check_output(command + ['ls-tree', '-r', 'HEAD'], text=True).splitlines():
        meta, relative = row.split('\t')
        mode, kind, oid = meta.split()
        path = source / relative
        if relative in absent:
            require(not path.exists(), 'external asset found in source cache: ' + relative)
            continue
        require(kind == 'blob' and not path.is_symlink(), 'regular source entry')
        raw = path.read_bytes()
        require(hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == oid,
                'source tree bytes changed: ' + relative)
        hashes[relative] = sha(raw)
    require(not list(source.rglob('*.so')) and not list(source.rglob('*.pyc')),
            'generated binary/cache bytes in pinned source')
    # Do not decode any image; upstream examples are source-release bytes only.
    return {'head': head, 'files_sha256': hashes, 'intentionally_absent_assets': sorted(absent)}


def qualify(device):
    import numpy as np
    import cv2
    import torch
    import torchvision
    import importlib.machinery
    from methods.physics_std.geometry import GeometryAdapter
    from methods.physics_std.regressor import FixedGeometryRegressor
    from methods.physics_std.renderer import DepthRenderer, depth_target
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    regressor = FixedGeometryRegressor(device=device)
    geometry = GeometryAdapter()
    renderer = DepthRenderer()
    extension = renderer._capture.extension
    require(isinstance(extension.__loader__, importlib.machinery.ExtensionFileLoader), 'real extension loader')
    require(renderer.is_official, 'production renderer without injected kernel')
    # Synthetic nonconstant input; no decoder, detector, dataset, image file or optimizer.
    image = (np.arange(120*120*3, dtype=np.uint32).reshape(120,120,3) % 256).astype(np.uint8)
    a, b = regressor.fit(image), regressor.fit(image)
    require(all(np.array_equal(a[k], b[k]) for k in a), 'regressor repeat')
    p = np.zeros(62, dtype=np.float32)
    p[:12] = np.eye(3, 4, dtype=np.float32).ravel()
    neutral = geometry.reconstruct(p, [0,0,256,256])
    require(neutral['dense_vertices'].shape == (38365,3), 'dense geometry shape')
    require(neutral['triangles'].shape == (76073,3), 'triangle shape')
    require(len(set(geometry.q['vertex_indices'])) == 140, 'frozen Q140')
    mesh = geometry.reconstruct(a['fitted_parameters'], [0,0,256,256])
    result = depth_target('live', reconstruct=lambda: mesh, renderer=renderer)
    repeated = depth_target('live', reconstruct=lambda: geometry.reconstruct(b['fitted_parameters'], [0,0,256,256]), renderer=renderer)
    for key in ('depth', 'raster_uint8', 'face_mask_256'):
        require(np.array_equal(result[key], repeated[key]), 'end-to-end repeat ' + key)
    depth = result['depth']
    require(depth.shape == (32,32) and depth.dtype == np.float32 and np.isfinite(depth).all(), 'depth shape/type/finite')
    require(0 <= depth.min() <= depth.max() <= 1 and np.count_nonzero(depth) > 0, 'nonempty bounded depth')
    require(result['face_mask_256'].any(), 'mesh visible')
    require(np.all(result['raster_uint8'][~result['face_mask_256']] == 0), 'zero background')
    expected = np.clip(cv2.resize(result['raster_uint8'], (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)/255, 0, 1)
    require(np.array_equal(depth, expected), 'A4 exact order')
    v = np.array([[16,16,1],[240,16,1],[16,240,1],[16,16,3],[240,16,3],[16,240,3]], dtype=np.float32)
    t = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    front, reverse = renderer.render(v,t), renderer.render(v,t[::-1])
    require(front['raster_uint8'][32,32] == 255 and front['face_mask_256'][32,32], 'larger z wins')
    require(all(np.array_equal(front[k],reverse[k]) for k in ('depth','raster_uint8','face_mask_256')), 'triangle order invariance')
    require(front['raster_uint8'].dtype == np.uint8 and front['raster_uint8'].shape == (256,256), 'uint8 intermediate')
    def forbidden(*args, **kwargs):
        raise AssertionError('spoof invoked geometry')
    from unittest.mock import patch
    with patch.object(regressor, 'fit', side_effect=forbidden), patch.object(geometry, 'reconstruct', side_effect=forbidden), patch.object(renderer, 'render', side_effect=forbidden):
        spoof = depth_target('spoof', reconstruct=forbidden, renderer=renderer)['depth']
    require(spoof.shape == (32,32) and spoof.dtype == np.float32 and not spoof.any(), 'spoof short circuit')
    return {'result':'PASS', 'device':device, 'scope':'E04_FIXED_GEOMETRY_DEPTH_ONLY',
            'regressor':{'constructed':True, 'arch':'mobilenet_v1', 'widen_factor':1.0,
                         'input_shape':list(image.shape), 'output_shape':[1,62], 'fitted_shape':[62],
                         'finite':True, 'repeat_byte_identical':True, 'checkpoint':regressor.checkpoint_keys,
                         'normalized_sha256':sha(a['normalized_parameters'].tobytes()),
                         'fitted_sha256':sha(a['fitted_parameters'].tobytes())},
            'geometry':{'dense_shape':[38365,3], 'triangles_shape':[76073,3], 'q140_unique':140,
                        'first_68_frozen_ibug_order_verified':True, 'q140_sha256':geometry.q['vertex_indices_sha256'],
                        'q140_rederived':False, 'source_functions_unchanged':['recon_vers','_parse_param','similar_transform']},
            'renderer':{'extension_path':extension.__file__, 'extension_sha256':sha(Path(extension.__file__).read_bytes()),
                        'loader':type(extension.__loader__).__name__, 'kernel_injected':False,
                        'larger_z_wins':True, 'triangle_order_invariant':True, 'background_zero':True,
                        'intermediate_shape':[256,256], 'intermediate_dtype':'uint8', 'face_mask_captured':True,
                        'resize':'cv2.INTER_AREA', 'order':'uint8 resize -> float32 -> /255 -> clip [0,1]'},
            'end_to_end':{'depth_shape':list(depth.shape), 'depth_dtype':str(depth.dtype),
                          'depth_range':[float(depth.min()),float(depth.max())], 'finite':True,
                          'repeat_byte_identical':True, 'depth_sha256':sha(depth.tobytes()),
                          'mesh_sha256':sha(mesh['dense_vertices'].tobytes()),
                          'face_mask_pixels':int(result['face_mask_256'].sum())},
            'spoof':{'exact_float32_zeros_32x32':True, 'regressor_reconstruction_renderer_calls':0},
            'source':regressor.source,
            'assets':{k:dict(v, deserialized=True) for k,v in regressor.assets.items()},
            'optimizer_steps':0, 'scientific_checkpoints_created':0, 'benchmark_images_decoded':0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.qualify:
        audit = SyntheticAccessAudit()
        sys.addaudithook(audit)
        result = qualify(args.device)
        result['file_access_audit'] = audit.report()
        require(not audit.denied, 'file access firewall')
    else:
        result = verify_evidence()
    raw = json.dumps(result, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.write_text(raw)
    print(raw)


def verify_evidence():
    audit = json.loads((ROOT/(BASE+'.json')).read_text())
    require(audit['decision'] == 'E04_GEOMETRY_RUNTIME_QUALIFIED', 'geometry decision')
    require(audit['fidelity_class'] == 'CONTROLLED_ADAPTATION', 'disclosure')
    require(audit['training_runtime'] == 'NOT_YET_QUALIFIED' and audit['training_graph'] == 'NOT_YET_EXECUTED', 'training scope')
    lock = ROOT/'environments/e04_geometry.lock.json'
    require(sha(lock.read_bytes()) == audit['environment_lock_sha256'], 'lock hash')
    for relative, digest in audit['artifact_sha256'].items():
        require(sha((ROOT/relative).read_bytes()) == digest, 'artifact hash '+relative)
    for probe in audit['probes']:
        require(probe['result'] == 'PASS' and not probe['file_access_audit']['denied'], 'probe/firewall')
        require(probe['renderer']['loader'] == 'ExtensionFileLoader' and not probe['renderer']['kernel_injected'], 'real renderer')
    require(all(audit['focused_tests'][k] == 0 for k in ('failures','errors','skipped')), 'focused tests')
    for relative,digest in audit['frozen_input_sha256'].items():
        raw = (ROOT/relative).read_bytes()
        committed = subprocess.check_output(['git','-C',str(ROOT),'show',HEAD+':'+relative])
        require(raw == committed and sha(raw) == digest, 'frozen input '+relative)
    require(audit['probes'][1]['regressor'] == audit['probes'][2]['regressor'] and
            audit['probes'][1]['end_to_end'] == audit['probes'][2]['end_to_end'], 'fresh-process CUDA repeat')
    ledger = 'outputs/audit/EXECUTION_LEDGER.jsonl'
    prefix = subprocess.check_output(['git','-C',str(ROOT),'show',HEAD+':'+ledger])
    now = (ROOT/ledger).read_bytes()
    require(len(prefix.splitlines()) == 100 and now.startswith(prefix) and len(now.splitlines()) == 101, 'ledger prefix and rows')
    row = json.loads(now[len(prefix):])
    require(row['classification'] == 'M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION', 'ledger classification')
    for relative, digest in row['file_sha256'].items():
        require(sha((ROOT/relative).read_bytes()) == digest, 'ledger file hash '+relative)
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip() == HEAD, 'repository HEAD')
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-list','--left-right','--count','HEAD...origin/m6-baselines'],text=True).split() == ['0','0'], 'repository divergence')
    current_source = source_integrity()
    require(current_source['files_sha256'] == audit['source_integrity']['source_files_sha256'], 'source cache unchanged')
    live_runtime_verified = False
    if os.environ.get('GPAT_E04_GEOMETRY_RUNTIME'):
        from methods.common.config import load_method_config
        from methods.physics_std.assets import validate_assets
        from methods.physics_std.source import validate_source
        from methods.physics_std.runtime import extension_root
        config = load_method_config('E04')
        validate_assets(config)
        extension_root(config, validate_source(config))
        freeze = subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
        require(freeze.strip() == (ROOT/'environments/e04_geometry.pip-freeze.txt').read_text().strip(), 'live package freeze')
        live_runtime_verified = True
    import csv
    import io
    from tools.build_artifact_index import iter_files
    index = (ROOT/'outputs/audit/ARTIFACT_INDEX.csv').read_bytes()
    require(b'\r\n' in index and b'\n' not in index.replace(b'\r\n',b''), 'index CRLF')
    indexed = {r['path']:r for r in csv.DictReader(io.StringIO(index.decode()))}
    actual = {relative:path for relative,path in iter_files()}
    require(set(indexed) == set(actual), 'artifact index coverage')
    for relative,path in actual.items():
        require(indexed[relative]['sha256'] == sha(path.read_bytes()) and
                int(indexed[relative]['size_bytes']) == path.stat().st_size, 'index entry '+relative)
    return {'result':'M6D3A_PREFLIGHT_PASS','decision':audit['decision'], 'ledger_rows':101,
            'environment_lock_sha256':sha(lock.read_bytes()), 'training_runtime':'NOT_YET_QUALIFIED',
            'artifact_index_rows':len(indexed), 'artifact_index_crlf':True,
            'live_runtime_verified':live_runtime_verified}


if __name__ == '__main__':
    main()
