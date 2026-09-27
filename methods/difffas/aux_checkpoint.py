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
import json
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
# M6D6h owner freeze (OWNER_DECISION / ASSET_FREEZE; not an amendment): the ONE authorized auxiliary encoder.
FREEZE_RECORD = 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml'
FREEZE_RECORD_SHA256 = '6230e0b9531d47664f9eb14a7f87b24f67b0bf231a995adaba1294f944ec7693'
OWNER_FROZEN_SHA256 = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
OWNER_FROZEN_BYTES = 185136819


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


def load_freeze_record(config=None):
    """M6D6h freeze record (JSON subset of YAML): pinned SHA256, bound authorities and frozen identity re-verified."""
    cfg = _config(config)
    raw = (ROOT / FREEZE_RECORD).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FREEZE_RECORD_SHA256:
        raise PreparationError('M6D6h freeze record SHA256 differs from the pinned value')
    record = json.loads(raw)
    doc = record['record_document']
    if sha256_file(ROOT / doc['path']) != doc['sha256']:
        raise PreparationError('M6D6h freeze record document SHA256')
    for rel, digest in record['bound_authority_sha256'].items():
        if sha256_file(ROOT / rel) != digest:
            raise PreparationError('M6D6h-bound authority SHA256 ' + rel)
    asset, owner = record['frozen_asset'], record['owner_freeze']
    if ((record['milestone'], record['method_id'], record['role'], record['classification'], record['fidelity']) !=
            ('M6D6h', 'E07c', 'AUXILIARY_CONDITIONING_ENCODER', 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION',
             {'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021', 'new_deviation': False,
              'new_fidelity_class': False}) or
            (asset['sha256'], asset['bytes'], asset['format']) != (OWNER_FROZEN_SHA256, OWNER_FROZEN_BYTES, FROZEN_FORMAT) or
            asset['path_template'] != cfg['external_assets'][0]['external_runtime_path'] or
            (owner['owner_freeze_performed'], owner['authoritative_for_main_difffas']) != (True, True) or
            owner['frozen_for_main_seeds'] != cfg['seeds']['experiment_seeds'] or
            record['consumption']['secure_loader'] != __name__ + '.load_frozen_aux_encoder' or
            record['consumption']['M8_integration_qualified'] is not False):
        raise PreparationError('M6D6h freeze record semantics')
    return {'sha256': FREEZE_RECORD_SHA256, 'record': record, 'frozen_asset': asset}


def frozen_checkpoint_path(runtime_root, config=None):
    """The configured seed-42 auxiliary path under an absolute, repository-external runtime root; no discovery."""
    cfg = _config(config)
    root = Path(runtime_root)
    if not root.is_absolute() or root.resolve().is_relative_to(ROOT):
        raise PreparationError('auxiliary checkpoint must remain in external runtime storage')
    return _external(cfg['external_assets'][0]['external_runtime_path'].replace('<runtime_root>', str(root)))


@contextmanager
def load_frozen_aux_encoder(runtime_root, recorded_sha256, config=None):
    """Main-run entry: ONLY the M6D6h owner-frozen asset (canonical path, frozen size, frozen SHA256).

    The caller SHA256 must equal the frozen value before the checkpoint file is opened; the size is
    required before hashing; the M6D6c exact-byte SHA256-before-torch.load order is unchanged.
    Any mismatch or missing file is STOP_AND_REPORT: no fallback, re-freeze or retraining.
    """
    frozen = load_freeze_record(config)['frozen_asset']
    if recorded_sha256 != frozen['sha256']:
        raise PreparationError('recorded SHA256 is not the M6D6h owner-frozen auxiliary encoder; STOP_AND_REPORT')
    path = frozen_checkpoint_path(runtime_root, config)
    if not path.is_file():
        raise PreparationError(f'required external asset unavailable: {path}')
    if path.stat().st_size != frozen['bytes']:
        raise PreparationError('auxiliary checkpoint size differs from the frozen size; STOP_AND_REPORT')
    asset = verify_future_checkpoint(runtime_root, recorded_sha256, config)
    if (Path(asset['path']), asset['size_bytes']) != (path, frozen['bytes']):
        raise PreparationError('auxiliary checkpoint is not the frozen canonical asset; STOP_AND_REPORT')
    with load_verified_whole_module(asset['path'], recorded_sha256, config) as model:
        yield model
