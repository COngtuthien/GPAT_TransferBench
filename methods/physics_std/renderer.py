"""A3 official Sim3DR rendering followed by the exact A4 conversion order."""
import importlib.util
import importlib.machinery
from pathlib import Path

import cv2
import numpy as np

from methods.common.config import load_method_config
from methods.common.learned import authoritative, PreparationError
from .contract import validate_overlay
from .source import validate_source, bind_function
from .runtime import extension_root


def validate_mesh(vertices, triangles):
    v = np.asarray(vertices)
    t = np.asarray(triangles)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or
            v.dtype != np.float32 or not np.isfinite(v).all()):
        raise PreparationError('vertices must be finite float32 [N,3]')
    if (t.ndim != 2 or t.shape[1] != 3 or not len(t) or t.dtype != np.int32 or
            t.min() < 0 or t.max() >= len(v)):
        raise PreparationError('triangles must be valid int32 [T,3] indices')
    # Do not invent an epsilon for the upstream division by zero.
    zrange = np.float32(v[:, 2].max() - v[:, 2].min())
    if not np.isfinite(zrange) or zrange <= 0:
        raise PreparationError('degenerate z range: official min-max would divide by zero')
    p = v[t, :2].astype(np.float64)
    area = ((p[:, 1, 0]-p[:, 0, 0])*(p[:, 2, 1]-p[:, 0, 1]) -
            (p[:, 2, 0]-p[:, 0, 0])*(p[:, 1, 1]-p[:, 0, 1]))
    if (area == 0).any():
        raise PreparationError('degenerate projected triangle')
    if np.abs(v[:, :2]).max() >= np.iinfo(np.int32).max:
        raise PreparationError('coordinates exceed official rasterizer integer bounds')
    return np.ascontiguousarray(v), np.ascontiguousarray(t)


class _CaptureBuffer:
    def __init__(self, extension):
        self.extension = extension
        self.mask = None

    def rasterize(self, image, vertices, triangles, colors, buffer, *args, **kwargs):
        initial = buffer.copy()
        self.extension.rasterize(image, vertices, triangles, colors, buffer, *args, **kwargs)
        self.mask = buffer != initial


class DepthRenderer:
    """Default path requires the official Cython extension; never falls back.

    A kernel injection is available ONLY with synthetic_only=True for CPU tests
    of the unchanged official C++ body. Such outputs are not scientific outputs.
    """
    def __init__(self, config=None, *, synthetic_only=False, kernel=None):
        self.config = authoritative(config) if config is not None else load_method_config('E04')
        self.contract = validate_overlay(self.config)
        self.source = validate_source(self.config)
        if kernel is not None and not synthetic_only:
            raise PreparationError('injected rasterizer is restricted to synthetic-only tests')
        self.is_official = not synthetic_only
        if kernel is None:
            root, expected_extension = extension_root(self.config, self.source)
            spec = importlib.machinery.PathFinder.find_spec('Sim3DR_Cython', [str(root)])
            if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                raise PreparationError('official Sim3DR_Cython unavailable; build pinned extension in execution environment')
            if not Path(spec.origin).resolve().is_relative_to(root):
                raise PreparationError('official rasterizer extension escaped source root')
            if expected_extension is not None and Path(spec.origin).resolve() != expected_extension:
                raise PreparationError('official rasterizer extension differs from verified build')
            kernel = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernel)
        self._capture = _CaptureBuffer(kernel)
        rasterize = bind_function(self.source, 'Sim3DR/Sim3DR.py', 'rasterize',
                                 {'Sim3DR_Cython': self._capture})
        contiguous = bind_function(self.source, 'utils/tddfa_util.py', '_to_ctype', {})
        self._depth = bind_function(self.source, 'utils/depth.py', 'depth',
                                   {'rasterize': rasterize, '_to_ctype': contiguous})

    def render(self, vertices, triangles):
        vertices, triangles = validate_mesh(vertices, triangles)
        size = self.config['data']['canonical_face_size']
        self._capture.mask = None
        # Official depth() normalizes over ALL mesh vertices and invokes the
        # unchanged official rasterize() body. One mesh, no display/file output.
        image = self._depth(np.zeros((size, size, 3), dtype=np.uint8),
                             [vertices.T], triangles, with_bg_flag=False)
        if self._capture.mask is None:
            raise PreparationError('official renderer did not expose its written buffer mask')
        return {'depth': convert_live_depth(image, self.config),
                'raster_uint8': image[..., 0].copy(),
                'face_mask_256': self._capture.mask.copy(),
                'scientific_output': self.is_official}


def convert_live_depth(raster_uint8, config=None):
    cfg = authoritative(config) if config is not None else load_method_config('E04')
    overlay = validate_overlay(cfg)['overlay']['depth']
    image = np.asarray(raster_uint8)
    size = cfg['data']['canonical_face_size']
    if image.dtype != np.dtype(overlay['raster_output_dtype']):
        raise PreparationError('A4 requires uint8 input; float-before-resize is forbidden')
    if image.shape == (size, size, 3):
        if not (np.array_equal(image[..., 0], image[..., 1]) and
                np.array_equal(image[..., 0], image[..., 2])):
            raise PreparationError('official depth channels must be identical')
        image = image[..., 0]
    if image.shape != (size, size):
        raise PreparationError('depth raster must match canonical geometry dimensions')
    height, width = overlay['output_shape']
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if resized.dtype != np.dtype(overlay['resize_dtype']):
        raise PreparationError('A4 resize dtype changed')
    numerator, denominator = map(int, overlay['post_resize_scale'].split('/'))
    result = resized.astype(overlay['post_resize_cast'])
    result *= numerator
    result /= float(denominator)
    return np.clip(result, *overlay['final_clip'])


def spoof_depth(config=None):
    cfg = authoritative(config) if config is not None else load_method_config('E04')
    overlay = validate_overlay(cfg)['overlay']['depth']
    return np.zeros(overlay['output_shape'], dtype=overlay['post_resize_cast'])


def depth_target(label, *, reconstruct=None, renderer=None, config=None):
    """Explicit live/spoof names; the spoof branch never invokes geometry."""
    if label == 'spoof':
        return {'depth': spoof_depth(config), 'reconstruction_invoked': False}
    if label != 'live':
        raise PreparationError('depth label must be live or spoof')
    if renderer is None or not renderer.is_official or reconstruct is None:
        raise PreparationError('live scientific target requires official renderer and reconstruction')
    mesh = reconstruct()
    return renderer.render(mesh['dense_vertices'], mesh['triangles'])
