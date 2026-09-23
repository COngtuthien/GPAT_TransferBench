"""Location-only compatibility shim for the separately built E04 geometry runtime.

No scientific value is overridden. Assets retain the frozen size/hash identity;
the external extension and its complete build inputs are verified before import.
"""
import hashlib
import json
import os
from pathlib import Path

from methods.common.config import ROOT
from methods.common.learned import PreparationError


def runtime_manifest(config):
    path = os.environ.get('GPAT_E04_GEOMETRY_RUNTIME')
    if not path:
        return None
    record = json.loads(Path(path).read_text())
    engine = config['external_assets']['geometry_engine']
    if (record['source_pin'] != engine['pinned_commit'] or
            record['qualification_scope'] != 'E04_FIXED_GEOMETRY_DEPTH_ONLY'):
        raise PreparationError('E04 runtime source/scope mismatch')
    expected = {a['path']: {'sha256': a['sha256'], 'bytes': a['bytes']}
                for a in engine['assets']}
    if record['assets'] != expected:
        raise PreparationError('E04 runtime asset identity mismatch')
    return record


def asset_root(config):
    record = runtime_manifest(config)
    return Path(record['asset_root'] if record else
                config['external_assets']['geometry_engine']['external_runtime_root'])


def extension_root(config, source):
    record = runtime_manifest(config)
    if record is None:
        return Path(source['root']) / 'Sim3DR', None
    build = record['sim3dr']
    root = Path(build['build_directory']).resolve()
    if root.is_relative_to(ROOT) or root.is_relative_to(Path(source['root'])):
        raise PreparationError('E04 generated extension must be outside the repository')
    expected = ('setup.py', 'lib/rasterize.pyx', 'lib/rasterize.h', 'lib/rasterize_kernel.cpp')
    if set(build['source_files_sha256']) != set(expected):
        raise PreparationError('E04 extension build input closure mismatch')
    for rel in expected:
        original = Path(source['root']) / 'Sim3DR' / rel
        generated = root / rel
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if (digest != build['source_files_sha256'][rel] or
                hashlib.sha256(generated.read_bytes()).hexdigest() != digest):
            raise PreparationError('E04 extension source changed: ' + rel)
    extension = Path(build['extension_path'])
    if (extension.is_symlink() or extension.resolve().parent != root or
            hashlib.sha256(extension.read_bytes()).hexdigest() != build['sha256']):
        raise PreparationError('E04 compiled extension identity mismatch')
    return root, extension.resolve()
