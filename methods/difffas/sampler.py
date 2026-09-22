"""Frozen sampling metadata only; never loads tensors or samples diffusion."""
from copy import deepcopy

from methods.common.learned import authoritative, PreparationError


def sampler_plan(config):
    cfg = authoritative(config)
    sampler = cfg['sampler']
    timesteps = list(range(0, sampler['sample_initial_noise'], sampler['DDIM_skip']))
    if (len(timesteps) != sampler['effective_steps'] or sampler['tensor_set_loaded'] != 'model' or
            sampler['algorithm'].lower() != 'ddim'):
        raise PreparationError('frozen DDIM/model tensor-set contract mismatch')
    return {**deepcopy(sampler), 'timesteps_ascending': timesteps,
            'execution_order': 'reverse, as diffusion.ddim_steps',
            'initial_state': 'Official q_sample of generation target live image; not a training content dependency',
            'use_pair': cfg['training']['use_pair'], 'tensor_set_substitution_allowed': False,
            'sampling_executed': False, 'checkpoint_loaded': False}
