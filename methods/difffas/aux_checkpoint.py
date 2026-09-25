"""E07c auxiliary-encoder checkpoint seam: frozen WHOLE-nn.Module format, SHA256 before deserialization.

The frozen format (A3 §5.2; configs/methods/e07c_difffas_bin_idfree.yaml
conditioning_encoder.checkpoint_format) is the pinned writer's
torch.save(resnet18, './PADISI.pkl') (models/pretrain_classifier.py:45) consumed by
BeatGANsAutoencModel.encoder(path): torch.load(path) -> .cuda()
(models/unet_autoenc.py:73-77). It is never converted to a state_dict,
safetensors, TorchScript or any other archive.

The one runtime difference is torch.load's modern default (weights_only
resolves to True), which rejects a whole pickled custom module. The explicit
weights_only=False below restores the legacy whole-module semantics; it is a
RUNTIME COMPATIBILITY ADAPTATION justified by M6D6c evidence, not a scientific
adaptation, a new format or a source modification. Because that unpickles
arbitrary objects, it is only ever applied to bytes whose SHA256 equals an
explicitly recorded value, and the hash is taken over the exact in-memory bytes
that are then deserialized (no re-read between check and load).
"""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import sys

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import authoritative, PreparationError
from methods.common.upstream import upstream_modules
from .contract import validate_contract
from .encoder import verify_future_checkpoint
from .source import validate_source, verify_executable_semantics

FROZEN_FORMAT = 'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())'
LOAD_COMPATIBILITY = {
    'weights_only': False,
    'classification': 'RUNTIME_COMPATIBILITY_ADAPTATION_RESTORING_LEGACY_WHOLE_MODULE_LOAD_SEMANTICS',
    'reason': ('qualified torch.load default and weights_only=True reject the frozen whole-module pickle '
               '(custom_rn.ResNet is not an allowed global); explicit weights_only=False restores it'),
    'evidence': 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY.json',
    'scientific_adaptation': False, 'checkpoint_format_changed': False, 'source_modified': False,
    'global_torch_load_patch': False, 'requires_sha256_verified_bytes': True}
# Custom audit events; order proof for SHA-before-deserialize (no effect without an audit hook).
EVENT_VERIFIED = 'gpat.e07c.aux_checkpoint.sha256_verified'
EVENT_REJECTED = 'gpat.e07c.aux_checkpoint.sha256_rejected'


def _config(config):
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    if cfg['conditioning_encoder']['checkpoint_format'] != FROZEN_FORMAT:
        raise PreparationError('frozen auxiliary checkpoint format is not the whole-module pickle')
    return cfg


def _external(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve().is_relative_to(ROOT):
        raise PreparationError('auxiliary checkpoint must remain in external runtime storage')
    return path


def encoder_identity(model, custom_rn, contracts):
    """A3/A6 identity of a constructed or deserialized encoder; no projection, no substitution."""
    import torch
    cls = type(model)
    k = contracts['effective_encoder']['substitute_objective']['K']
    module_file = Path(custom_rn.__file__).resolve()
    if (cls is not custom_rn.ResNet or cls.__module__ != 'custom_rn' or cls.__name__ != 'ResNet' or
            hashlib.sha256(module_file.read_bytes()).hexdigest() != contracts['overlay']['source']['sha256']):
        raise PreparationError('encoder is not the pinned custom_rn.ResNet')
    layers = [len(getattr(model, f'layer{i}')) for i in (1, 2, 3, 4)]
    blocks = {type(b) for i in (1, 2, 3, 4) for b in getattr(model, f'layer{i}')}
    if layers != [3, 4, 6, 3] or blocks != {custom_rn.BasicBlock}:
        raise PreparationError('encoder topology is not BasicBlock [3,4,6,3]')
    fc = model.fc
    if type(fc) is not torch.nn.Linear or (fc.in_features, fc.out_features) != (512, k) or fc.bias is None:
        raise PreparationError('encoder head is not Linear(512, K)')
    return {'module': cls.__module__, 'class': cls.__name__, 'module_file_sha256': contracts['overlay']['source']['sha256'],
            'topology': layers, 'block': 'BasicBlock', 'fc': {'in_features': 512, 'out_features': k, 'bias': True}}


def save_whole_module(model, path, config=None):
    """torch.save(model, path) exactly as pretrain_classifier.py:45; call inside encoder.encoder_model().

    The custom_rn module must still be the live import so the pickle records
    custom_rn.ResNet. Returns the recorded size and SHA256; nothing is converted.
    """
    cfg = _config(config)
    contracts = validate_contract(cfg)
    path = _external(path)
    custom_rn = sys.modules.get(type(model).__module__)
    if custom_rn is None:
        raise PreparationError('pinned custom_rn import is not live; keep encoder_model() open')
    identity = encoder_identity(model, custom_rn, contracts)
    import torch
    torch.save(model, str(path))
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path),
            'serialization_api': 'torch.save(model, path)', 'format': FROZEN_FORMAT, 'identity': identity}


def read_verified(path, expected_sha256):
    """Return the checkpoint bytes only if their SHA256 equals the recorded value; never deserializes."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise PreparationError('record and freeze auxiliary SHA256 before any deserialization')
    path = _external(path)
    if not path.is_file():
        raise PreparationError(f'required external asset unavailable: {path}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        sys.audit(EVENT_REJECTED, str(path), digest)
        raise PreparationError('auxiliary checkpoint SHA256 mismatch; refusing to deserialize')
    sys.audit(EVENT_VERIFIED, str(path), digest)
    return raw


@contextmanager
def load_verified_whole_module(path, expected_sha256, config=None, *, device='cuda'):
    """SHA256-gated legacy whole-module load; yields the encoder while custom_rn stays imported.

    Order: verify SHA256 of the bytes -> verify pinned source -> import custom_rn
    -> torch.load(bytes, weights_only=False) -> identity check -> .cuda()
    (unet_autoenc.py:76). device='cpu' exists only for CUDA-less tests.
    """
    cfg = _config(config)
    raw = read_verified(path, expected_sha256)
    contracts = validate_contract(cfg)
    source = validate_source(cfg)
    verify_executable_semantics(source, contracts)
    import torch
    with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
        model = torch.load(io.BytesIO(raw), weights_only=LOAD_COMPATIBILITY['weights_only'])
        del raw
        encoder_identity(model, modules['custom_rn'], contracts)
        model = model.cuda() if device == 'cuda' else model.to(device)
        yield model


@contextmanager
def load_frozen_aux_encoder(runtime_root, recorded_sha256, config=None):
    """Future main-run entry: the frozen external path and recorded SHA256 only."""
    asset = verify_future_checkpoint(runtime_root, recorded_sha256, config)
    with load_verified_whole_module(asset['path'], recorded_sha256, config) as model:
        yield model
