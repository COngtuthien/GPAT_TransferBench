"""External auxiliary bytes remain outside Git; no weight load in preparation."""
from pathlib import Path
import io
import pickle
import codecs

import numpy as np

from methods.common.learned import authoritative, verify_asset, PreparationError
from .contract import committed_input
from .runtime import asset_root


def validate_assets(config):
    cfg = authoritative(config)
    engine = cfg['external_assets']['geometry_engine']
    committed_input(engine['provenance'])
    root = asset_root(cfg)
    return {a['path']: verify_asset({**a, 'external_runtime_path': str(root / a['path'])})
            for a in engine['assets']}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Only the numeric-array constructors used by the hash-pinned BFM assets."""
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('_codecs', 'encode'): codecs.encode,
        }
        if (module, name) not in allowed:
            raise PreparationError(f'non-array pickle global refused: {module}.{name}')
        return allowed[module, name]


def load_geometry_arrays(config):
    """Explicit geometry-only load; regressor checkpoint is never deserialized.

    Restricted loading is necessary to bind dense reconstruction and check the
    native anchor correspondence. Static preparation only hashes assets.
    """
    import hashlib
    verified = validate_assets(config)
    result = {}
    for relative in ('configs/bfm_noneck_v3.pkl', 'configs/tri.pkl'):
        meta = verified[relative]
        raw = Path(meta['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise PreparationError('geometry asset changed after verification')
        result[relative] = _NumpyOnlyUnpickler(io.BytesIO(raw)).load()
    return result['configs/bfm_noneck_v3.pkl'], result['configs/tri.pkl']
