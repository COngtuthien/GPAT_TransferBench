"""E07c preparation only: one auxiliary plan and three main experiment plans."""
from copy import deepcopy

from methods.common.config import load_method_config, load_logging_contract
from methods.common.learned import (authoritative, environment_report, checkpoint_plan,
                                    PreparationError, mapping)
from .contract import validate_contract, guide_contract
from .source import validate_source, verify_executable_semantics
from .encoder import encoder_plan, checkpoint_identity
from .seed_adapter import experiment_seed_plan
from .sampler import sampler_plan


class DiffFASAdapter:
    def __init__(self, config=None):
        self.config = authoritative(config) if config is not None else load_method_config('E07c')
        validate_contract(self.config)

    def validate_source(self, *, source_root=None):
        return validate_source(self.config, source_root=source_root)

    def validate_environment(self):
        return environment_report('torch', ('numpy', 'torch', 'torchvision', 'scipy', 'tensorfn', 'cv2', 'PIL'))

    def prepare(self):
        cfg = authoritative(self.config)
        c = validate_contract(cfg)
        source = self.validate_source()
        return {'contracts': c, 'source': source,
                'source_evidence': verify_executable_semantics(source, c),
                'guide': guide_contract(cfg, c), 'auxiliary_plan': encoder_plan(cfg, c),
                'environment': self.validate_environment(),
                'logging_contract': load_logging_contract()['version'],
                'logging_runtime': 'methods.common.learned_runlog.LearnedRunContext',
                'execution_ready': False,
                'execution_dependencies': ['one auxiliary checkpoint trained/hashed/frozen in a later authorized run',
                    'compatible pinned-source torch/torchvision/scipy/tensorfn runtime',
                    'future runner integration using frozen manifest rows instead of upstream random guide selection']}

    def build_auxiliary_plan(self):
        return self.prepare()['auxiliary_plan']

    def checkpoint_policy(self, seed, *, selection_split=None):
        return checkpoint_plan(self.config, seed, selection_split=selection_split)

    def validate_checkpoint_event(self, seed, *, completed_epochs, terminal=False,
                                  selected=False, extra_optimizer_steps=0, selection_split=None):
        policy = self.checkpoint_policy(seed, selection_split=selection_split)
        if (type(completed_epochs) is not int or not 0 <= completed_epochs <= policy['final_epoch'] or
                type(extra_optimizer_steps) is not int or extra_optimizer_steps != 0 or
                ((terminal or selected) and completed_epochs != policy['final_epoch'])):
            raise PreparationError('terminal checkpoint requires end of frozen epoch budget and zero extra steps')
        return {'completed_epochs': completed_epochs, 'terminal': terminal, 'selected': selected,
                'checkpoint_written': False, 'extra_optimizer_steps': 0}

    def build_training_plan(self, seed, *, selection_split=None):
        cfg = authoritative(self.config)
        cp = self.checkpoint_policy(seed, selection_split=selection_split)
        prepared = self.prepare()
        return {**prepared, 'method_id': cfg['method_id'], 'seed': seed,
                'reporting_label': cfg['reporting_label'], 'fidelity_class': cfg['fidelity_class'],
                'fidelity_provenance': cfg['fidelity_provenance'], 'deviation': cfg['deviation'],
                'config_sha256': cfg['_runtime']['config_sha256'],
                'adaptation_config_sha256': prepared['contracts']['adaptation_sha256'],
                'a6_overlay_sha256': prepared['contracts']['overlay_sha256'],
                'training': deepcopy(cfg['training']), 'diffusion': deepcopy(cfg['diffusion']),
                'optimizer': deepcopy(cfg['optimizer']), 'sampler': sampler_plan(cfg),
                'seed_adapter': experiment_seed_plan(cfg, seed),
                'auxiliary_checkpoint': checkpoint_identity(cfg), 'checkpoint': cp,
                'data_firewall': deepcopy(cfg['data']['splits']),
                'main_path': deepcopy(prepared['contracts']['adaptation']['official_code_path']),
                'settings_mapping': mapping(cfg, {
                    'args.max_epochs': 'training.max_epochs', 'args.batch_size': 'training.batch_size',
                    'args.use_pair': 'training.use_pair', 'conf.in_channels': 'training.model_in_channels',
                    'args.guidance_prob': 'training.guidance_prob', 'args.means_size': 'training.means_size',
                    'args.var_size': 'training.var_size',
                    'DiffConf.training.optimizer.lr': 'optimizer.learning_rate',
                    'DiffConf.training.scheduler': 'optimizer.scheduler',
                }, 'FAS_train.py; config/diffconfig.py; config/diffusion.conf'),
                'mode': 'STATIC_PREPARATION_ONLY', 'training_launched': False,
                'model_constructed': False, 'checkpoint_created': False,
                'diffusion_sampling': False, 'production_runtime_write': False}
