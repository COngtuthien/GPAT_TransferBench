"""Separate one-off auxiliary seed from each main experiment's complete RNG hook."""
import importlib

from methods.common.config import load_method_config
from methods.common.learned import authoritative, seed_plan, apply_framework_seed, PreparationError


def experiment_seed_plan(config, seed):
    cfg = authoritative(config)
    plan = seed_plan(cfg, seed, 'torch')
    return {**plan, 'scope': 'MAIN_DIFFFAS',
            'replaces_hardcoded_seed': cfg['seed_adapter']['official_hardcoded_seed'],
            'source': 'FAS_train.py:seed_torch',
            'cuda_seed_calls': ['torch.cuda.manual_seed', 'torch.cuda.manual_seed_all'],
            'cudnn_benchmark': cfg['training']['cudnn']['benchmark'],
            'cudnn_deterministic': cfg['training']['cudnn']['deterministic']}


def auxiliary_seed_plan(config):
    cfg = authoritative(config)
    aux = cfg['conditioning_encoder']
    return {**seed_plan(cfg, aux['auxiliary_encoder_training_seed'], 'torch'),
            'scope': 'ONE_OFF_AUXILIARY_ENCODER', 'runs': aux['auxiliary_encoder_training_runs'],
            'reuse_for_main_seeds': aux['same_frozen_encoder_reused_for_main_seeds'],
            'cuda_seed_calls': ['torch.cuda.manual_seed', 'torch.cuda.manual_seed_all']}


def apply_seed(seed, *, cuda, auxiliary=False, config=None):
    """Future explicit hook; preflight never invokes it or probes a CUDA device.

    Common hook verifies PYTHONHASHSEED was set before launching the interpreter.
    No unsupported deterministic-algorithm override is imposed.
    """
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    if auxiliary and seed != cfg['conditioning_encoder']['auxiliary_encoder_training_seed']:
        raise PreparationError('auxiliary encoder has exactly one frozen seed')
    plan = auxiliary_seed_plan(cfg) if auxiliary else experiment_seed_plan(cfg, seed)
    apply_framework_seed(cfg, seed, 'torch', cuda=cuda)
    if cuda:
        importlib.import_module('torch').cuda.manual_seed(seed)
    return plan
