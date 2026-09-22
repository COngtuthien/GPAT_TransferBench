"""A3/A6 auxiliary encoder planning and explicit future source-construction seam."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from methods.common.config import ROOT, load_method_config
from methods.common.learned import authoritative, PreparationError, verify_asset
from methods.common.upstream import upstream_modules
from .contract import validate_contract
from .seed_adapter import auxiliary_seed_plan
from .source import validate_source, verify_executable_semantics


def checkpoint_identity(config):
    cfg = authoritative(config)
    asset, = cfg['external_assets']
    return {**deepcopy(asset), 'exists_verified': False,
            'sha256_status': 'RECORDED_AFTER_TRAINING',
            'missing_field_reason': 'One auxiliary encoder has not been trained; no checkpoint/hash exists',
            'serialization': cfg['conditioning_encoder']['checkpoint_format'],
            'same_identity_for_all_main_seeds': True}


def encoder_plan(config, contracts=None):
    cfg = authoritative(config)
    c = contracts or validate_contract(cfg)
    aux = c['effective_encoder']
    obj = aux['substitute_objective']
    if (aux['auxiliary_encoder_training_runs'] != 1 or not aux['trained_exactly_once'] or
            aux['same_frozen_encoder_reused_for_main_seeds'] != cfg['seeds']['experiment_seeds'] or
            aux['removed'] or aux['disabled'] or aux['architecture_substituted'] or
            aux['replaced_with_imagenet_or_torchvision_resnet18'] or
            aux['val_used_for_encoder_selection'] or aux['test_used_for_anything'] or
            sum(obj['train_counts'].values()) != obj['train_total'] or
            len(obj['classes']) != obj['K']):
        raise PreparationError('invalid frozen single-encoder reconstruction contract')
    return {'role': 'aux_encoder', 'mode': 'STATIC_PREPARATION_ONLY',
            'seed': aux['auxiliary_encoder_training_seed'], 'runs': aux['auxiliary_encoder_training_runs'],
            'seed_plan': auxiliary_seed_plan(cfg), 'architecture': aux['architecture'],
            'topology': aux['architecture_topology'],
            'feature_interface_at_256': deepcopy(aux['feature_interface_at_256']),
            'projection_layers_added': False,
            'objective': deepcopy(obj), 'training': deepcopy(aux['training_contract']),
            'run_layout': aux['run_layout'], 'checkpoint': checkpoint_identity(cfg),
            'candidate_encoder_search': aux['candidate_encoder_search'],
            'three_different_encoders_trained': aux['three_different_encoders_trained'],
            'val_selection': False, 'test_access': False, 'trained': False,
            'class_metadata_opened': False,
            'class_metadata_future_hash_check_required': True,
            'config_sha256': cfg['_runtime']['config_sha256'],
            'a6_overlay_sha256': c['overlay_sha256']}


def class_index(label, *, split, config=None):
    """Auxiliary-only label map on synthetic/metadata records; never main targets."""
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    obj = cfg['conditioning_encoder']['substitute_objective']
    if split != 'TRAIN' or obj['split'] != 'TRAIN_ONLY' or label not in obj['classes']:
        raise PreparationError('auxiliary label must be a frozen K7 class in TRAIN')
    return obj['classes'].index(label)


def verify_future_checkpoint(runtime_root, recorded_sha256, config=None):
    """Hash an explicitly frozen future asset; never deserialize a pickle."""
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    if (not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in recorded_sha256)):
        raise PreparationError('record and freeze auxiliary SHA256 before any main run consumes it')
    root = Path(runtime_root)
    if not root.is_absolute() or root.resolve().is_relative_to(ROOT):
        raise PreparationError('auxiliary checkpoint must remain in external runtime storage')
    asset = deepcopy(cfg['external_assets'][0])
    asset['external_runtime_path'] = asset['external_runtime_path'].replace('<runtime_root>', str(root))
    asset['sha256'] = recorded_sha256
    return verify_asset(asset)


@contextmanager
def encoder_model(config=None):
    """Future explicit CPU construction; keep custom_rn import alive for whole-module save.

    No pretraining script, weight download, checkpoint read/write, or CUDA call.
    Caller must keep this context open for future torch.save(model), preserving
    the original custom_rn class identity rather than a state_dict-only format.
    """
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    contracts = validate_contract(cfg)
    source = validate_source(cfg)
    verify_executable_semantics(source, contracts)
    import torch
    with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
        model = modules['custom_rn'].resnet18(pretrained=False)
        # Source/A3 fixes 512 input features; retain the constructed source value.
        k = contracts['effective_encoder']['substitute_objective']['K']
        model.fc = torch.nn.Linear(model.fc.in_features, k)
        yield model
