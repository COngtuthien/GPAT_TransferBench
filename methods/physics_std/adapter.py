"""Static E04 preparation and future TensorFlow optimizer/checkpoint hooks."""
from copy import deepcopy

from methods.common.config import load_method_config, load_logging_contract
from methods.common.learned import authoritative, environment_report, seed_plan, PreparationError
from .contract import validate_overlay, validate_q140
from .source import validate_source
from .assets import validate_assets


class PhysicsSTDAdapter:
    def __init__(self, config=None):
        self.config = authoritative(config) if config is not None else load_method_config('E04')
        validate_overlay(self.config)

    def validate_source(self):
        return validate_source(self.config)

    def validate_environment(self):
        return environment_report('tensorflow; fixed 3DDFA auxiliary PyTorch',
                                  ('numpy', 'cv2', 'torch', 'torchvision', 'tensorflow', 'Cython'))

    def prepare(self):
        cfg = authoritative(self.config)
        fixed = cfg['external_assets']
        if (not fixed['fixed_auxiliary_reconstruction_assets'] or
                fixed['retrained_or_rederived_per_experiment_seed'] or
                fixed['reused_for_experiment_seeds'] != cfg['seeds']['experiment_seeds']):
            raise PreparationError('E04 geometry must be fixed across all seeds')
        return {'contracts': validate_overlay(cfg), 'source': self.validate_source(),
                'assets': validate_assets(cfg), 'q140': validate_q140(cfg),
                'environment': self.validate_environment(),
                'logging_contract': load_logging_contract()['version'],
                'fixed_auxiliary_reuse': deepcopy(fixed | {'geometry_engine':
                    {'retrained': False, 'rederived': False}})}

    def checkpoint_policy(self, seed, *, selection_split=None):
        cfg = authoritative(self.config)
        seed_plan(cfg, seed, 'tensorflow')
        p = cfg['checkpoint']
        terminal = cfg['training']['total_iterations']
        if (selection_split is not None or p['selection_uses_val'] or p['selection_uses_test'] or
                p['selection_scope'] != 'WITHIN_SEED' or
                p['rule'] != 'BASELINE_FINAL_STATE_V1' or
                p['terminal_state'] != f'iteration_{terminal}'):
            raise PreparationError('E04 final state cannot be selected by VAL/TEST or another budget')
        return {**deepcopy(p), 'experiment_seed': seed, 'terminal_iteration': terminal,
                'save_terminal_if_cadence_misses': True, 'extra_optimizer_steps': 0,
                'checkpoint_written': False}

    def validate_checkpoint_event(self, seed, *, iteration, terminal=False, selected=False,
                                  extra_optimizer_steps=0, selection_split=None):
        policy = self.checkpoint_policy(seed, selection_split=selection_split)
        end = policy['terminal_iteration']
        if (type(iteration) is not int or not 0 < iteration <= end or
                type(extra_optimizer_steps) is not int or extra_optimizer_steps != 0 or
                ((terminal or selected) and iteration != end)):
            raise PreparationError('checkpoint must respect iteration budget and zero extra optimizer steps')
        return {'iteration': iteration, 'terminal': terminal, 'selected': selected,
                'selection_reason': policy['rule'] if selected else 'Periodic/terminal retention',
                'checkpoint_written': False}

    def build_training_plan(self, seed, *, selection_split=None):
        policy = self.checkpoint_policy(seed, selection_split=selection_split)
        prepared = self.prepare()
        cfg = self.config
        a4 = prepared['contracts']['overlay']
        return {
            'method_id': cfg['method_id'], 'seed': seed,
            'fidelity_class': cfg['fidelity_class'], 'reporting_name': cfg['reporting_label'],
            'base_config_sha256': cfg['_runtime']['config_sha256'],
            'a4_overlay_sha256': prepared['contracts']['overlay_sha256'],
            'contracts': prepared['contracts'], 'source': prepared['source'],
            'external_assets': prepared['assets'],
            'q140': {'hash': prepared['q140']['vertex_indices_sha256'],
                     'indices': prepared['q140']['vertex_indices'],
                     'anchor_indices': prepared['q140']['anchor_indices'],
                     'image_independent': True, 'rederived': False},
            'fixed_auxiliary': prepared['fixed_auxiliary_reuse'],
            'depth_target': {'A3': deepcopy(cfg['controlled_reconstruction']['depth_target_M0']),
                             'A4': deepcopy(a4['depth']),
                             'zero_z_range': 'REJECT_NO_EPSILON',
                             'degenerate_projected_triangles': 'REJECT'},
            'training': deepcopy(cfg['training']),
            'optimizer': {**deepcopy(a4['optimizer']), **deepcopy(cfg['optimizer']),
                          'algorithm_provenance': a4['provenance']['optimizer_resolution'],
                          'lr_schedule_provenance': 'M6B_PAPER'},
            'losses': deepcopy(cfg['losses']),
            'seed_plan': seed_plan(cfg, seed, 'tensorflow'), 'checkpoint': policy,
            'data_firewall': deepcopy(cfg['data']['splits']),
            'logging_contract': prepared['logging_contract'],
            'environment': prepared['environment'], 'mode': 'STATIC_PREPARATION_ONLY',
            'training_launched': False, 'model_inference': False,
            'execution_dependencies': ['compatible TensorFlow execution environment',
                'pinned Sim3DR_Cython build', 'future training runner integration',
                'fixed 3DDFA regressor environment for future live-image fitting'],
        }

    def build_optimizer(self, tf, global_iteration):
        """Future hook: caller supplies TensorFlow; never called by preparation.

        global_iteration denotes PhySTD iterations, not a discriminator substep.
        TensorFlow Adam has no weight-decay term; no regularizer is introduced.
        """
        cfg = authoritative(self.config)
        opt = validate_overlay(cfg)['overlay']['optimizer']
        schedule = cfg['optimizer']['lr_schedule']
        lr = tf.train.exponential_decay(cfg['optimizer']['learning_rate'], global_iteration,
            schedule['every_iterations'], 1 / schedule['factor'], staircase=True)
        return tf.train.AdamOptimizer(lr, beta1=opt['beta1'], beta2=opt['beta2'],
                                      epsilon=opt['epsilon'])
