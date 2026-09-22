"""Dense reconstruction from fitted parameters using pinned upstream functions."""
import json
from types import SimpleNamespace

import numpy as np

from methods.common.config import ROOT, load_method_config
from methods.common.learned import authoritative, PreparationError
from .assets import load_geometry_arrays
from .contract import validate_q140
from .source import validate_source, bind_function


class GeometryAdapter:
    """Consumes fitted, de-normalized 3DDFA parameters; does not infer from images.

    Uses unchanged TDDFA.recon_vers(dense_flag=True) and similar_transform.
    The upstream BFM constructor's source-relative tri.pkl path is avoided by
    assembling its numeric inputs from the verified external assets.
    """
    def __init__(self, config=None):
        self.config = authoritative(config) if config is not None else load_method_config('E04')
        source = validate_source(self.config)
        self.q = validate_q140(self.config)
        engine = self.config['external_assets']['geometry_engine']
        self.inference = engine['inference']
        geometry = json.loads((ROOT / engine['provenance']).read_text())['model_geometry']
        model, triangles = load_geometry_arrays(self.config)
        anchors = np.asarray(model['keypoints']).reshape(self.q['num_anchors'], 3)[:, 0] // 3
        if not np.array_equal(anchors, self.q['anchor_indices']):
            raise PreparationError('BFM iBUG anchor order disagrees with frozen Q140')
        bfm = SimpleNamespace(
            u=model['u'].astype(np.float32),
            w_shp=model['w_shp'].astype(np.float32)[..., :self.inference['shape_dim']],
            w_exp=model['w_exp'].astype(np.float32)[..., :self.inference['exp_dim']],
        )
        n = geometry['num_vertices']
        if (bfm.u.shape != (3*n, 1) or
                bfm.w_shp.shape != (3*n, self.inference['shape_dim']) or
                bfm.w_exp.shape != (3*n, self.inference['exp_dim'])):
            raise PreparationError('BFM dimensions disagree with frozen geometry')
        self.triangles = np.ascontiguousarray(triangles.T, dtype=np.int32)
        if (self.triangles.shape != (geometry['num_triangles'], 3) or
                self.triangles.min() < 0 or self.triangles.max() >= n):
            raise PreparationError('invalid frozen triangulation')
        self._state = SimpleNamespace(bfm=bfm, size=self.inference['input'])
        parse = bind_function(source, 'utils/tddfa_util.py', '_parse_param', {})
        projection = bind_function(source, 'utils/tddfa_util.py', 'similar_transform', {})
        self._reconstruct = bind_function(source, 'TDDFA.py', 'recon_vers',
            {'_parse_param': parse, 'similar_transform': projection}, class_name='TDDFA')

    def reconstruct(self, fitted_parameters, roi_box):
        parameters = np.asarray(fitted_parameters)
        roi = np.asarray(roi_box)
        if (parameters.shape != (self.inference['param_dim'],) or parameters.dtype != np.float32 or
                not np.isfinite(parameters).all()):
            raise PreparationError('expected finite fitted/de-normalized float32 3DDFA parameters')
        if (roi.shape != (4,) or not np.isfinite(roi).all() or
                roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise PreparationError('roi_box must have finite positive extent in canonical coordinates')
        vertices = self._reconstruct(self._state, [parameters], [roi_box], dense_flag=True)[0]
        if not np.isfinite(vertices).all():
            raise PreparationError('nonfinite reconstructed vertices')
        return {'dense_vertices': np.ascontiguousarray(vertices.T),
                'q140_vertices': np.ascontiguousarray(vertices.T[self.q['vertex_indices']]),
                'triangles': self.triangles.copy()}
