"""Frozen auxiliary MobileNet inference; no training or benchmark image loading."""
import hashlib
import io
from pathlib import Path
from types import ModuleType

import numpy as np

from methods.common.config import load_method_config
from methods.common.learned import authoritative, PreparationError
from .assets import validate_assets, _NumpyOnlyUnpickler
from .source import validate_source, bind_function


class FixedGeometryRegressor:
    """Pinned MobileNet v1 on already prepared 120x120 uint8 BGR inputs.

    Uses the pinned model, load_model, ToTensorGjz and NormalizeGjz unchanged.
    Crop/ROI selection for benchmark images is outside this runtime qualification.
    """
    def __init__(self, config=None, *, device='cpu'):
        import torch
        self.config = authoritative(config) if config is not None else load_method_config('E04')
        self.source = validate_source(self.config)
        self.assets = validate_assets(self.config)
        self.device = torch.device(device)
        inference = self.config['external_assets']['geometry_engine']['inference']
        path = Path(self.source['root']) / 'models/mobilenet_v1.py'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.source['files_sha256']['models/mobilenet_v1.py']:
            raise PreparationError('regressor source changed')
        module = ModuleType('_e04_pinned_mobilenet_v1')
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        model = module.mobilenet(num_classes=inference['param_dim'],
                                 widen_factor=inference['widen_factor'])
        checkpoint = self.assets['weights/mb1_120x120.pth']
        raw = Path(checkpoint['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint['sha256']:
            raise PreparationError('regressor checkpoint changed')
        loaded = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['state_dict']
        mapped = {}
        for key, value in loaded.items():
            key = key.replace('module.', '')
            if key in ('fc_param.bias', 'fc_param.weight'):
                key = key.replace('_param', '')
            if key in mapped:
                raise PreparationError('duplicate mapped checkpoint key')
            mapped[key] = value
        state = model.state_dict()
        missing = sorted(set(state) - set(mapped))
        unexpected = sorted(set(mapped) - set(state))
        mismatched = sorted(k for k in set(mapped) & set(state) if mapped[k].shape != state[k].shape)
        # Older checkpoints can omit non-learned batch counters. Evaluation does
        # not consume them; upstream load_model also retains the zero defaults.
        # The frozen training checkpoint also carries an unused landmark head.
        # Pinned load_model already ignores it; the inference topology is unchanged.
        if (set(unexpected) - {'fc_lm.bias', 'fc_lm.weight'} or mismatched or
                any(not k.endswith('.num_batches_tracked') for k in missing)):
            raise PreparationError(f'checkpoint architecture mismatch: {missing}, {unexpected}, {mismatched}')
        self.checkpoint_keys = {'missing_keys': missing, 'unexpected_keys': unexpected,
                                'shape_mismatches': mismatched, 'checkpoint_key_count': len(mapped),
                                'model_key_count': len(state),
                                'ignored_by_unchanged_upstream_loader': unexpected,
                                'unexpected_key_shapes': {k: list(mapped[k].shape) for k in unexpected}}
        loader = bind_function(self.source, 'utils/tddfa_util.py', 'load_model', {'torch': torch})
        self.model = loader(model, checkpoint['path']).eval().to(self.device)
        self.model.requires_grad_(False)
        for key, value in mapped.items():
            if key not in state:
                continue
            if not torch.equal(self.model.state_dict()[key].cpu(), value.cpu()):
                raise PreparationError('loaded tensor differs from frozen checkpoint: ' + key)
        self.checkpoint_keys['all_inference_tensors_exact'] = True
        util_path = Path(self.source['root']) / 'utils/tddfa_util.py'
        util_raw = util_path.read_bytes()
        if hashlib.sha256(util_raw).hexdigest() != self.source['files_sha256']['utils/tddfa_util.py']:
            raise PreparationError('regressor transform source changed')
        import ast
        tree = ast.parse(util_raw)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
                 n.name in ('ToTensorGjz', 'NormalizeGjz')]
        scope = {'torch': torch, 'np': np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(util_path), 'exec'), scope)
        self.to_tensor = scope['ToTensorGjz']()
        self.normalize = scope['NormalizeGjz'](mean=127.5, std=128)
        meta = self.assets['configs/param_mean_std_62d_120x120.pkl']
        raw = Path(meta['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise PreparationError('parameter normalization asset changed')
        stats = _NumpyOnlyUnpickler(io.BytesIO(raw)).load()
        self.mean, self.std = stats['mean'], stats['std']
        if self.mean.shape != (62,) or self.std.shape != (62,):
            raise PreparationError('parameter normalization dimensions differ')

    def fit(self, image):
        import torch
        image = np.asarray(image)
        if image.shape != (120, 120, 3) or image.dtype != np.uint8:
            raise PreparationError('regressor requires prepared uint8 BGR [120,120,3]')
        with torch.inference_mode():
            tensor = self.normalize(self.to_tensor(image)).unsqueeze(0).to(self.device)
            output = self.model(tensor)
            if tuple(output.shape) != (1, 62) or not torch.isfinite(output).all():
                raise PreparationError('regressor finite 62-d output contract failed')
            param = output.squeeze().cpu().numpy().flatten().astype(np.float32)
            fitted = param * self.std + self.mean
        if fitted.shape != (62,) or fitted.dtype != np.float32 or not np.isfinite(fitted).all():
            raise PreparationError('fitted parameter contract failed')
        return {'normalized_parameters': param, 'fitted_parameters': fitted}
