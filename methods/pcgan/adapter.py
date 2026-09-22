"""E05 static preparation under M6B+A2+A5. No training launcher or model import."""
from copy import deepcopy

from methods.common.config import load_method_config, load_logging_contract
from methods.common.learned import authoritative, environment_report, seed_plan, PreparationError
from .contract import validate_contract
from .source import validate_source, symbol_evidence
from .architecture import architecture_mapping


def loss_mapping(config):
    cfg = authoritative(config)
    # Mathematical identities, not another scientific coefficient registry.
    expressions = {
        'L_rec': {'paper_equation': 2, 'inputs': ['x', 'G(E(x))'],
                  'distance': 'L2 norm; expectation over samples'},
        'L_recblur': {'paper_equation': 3,
                      'inputs': ['blur(x_tgt)', 'blur(G(z_pat_src, z_con_tgt))'],
                      'distance': 'L2 norm; expectation over distinct source/target samples',
                      'blur_authority': 'A5', 'detach_due_to_blur': False},
        'L_advrec': {'paper_equation': 4, 'expression': 'E[-log D(G(E(x_src)))]'},
        'L_advmix': {'paper_equation': 4, 'expression': 'E[-log D(G(z_pat_src, z_con_tgt))]'},
        'L_pat': {'paper_equation': 1,
                  'expression': 'E[-log D_patch(crop(G(z_pat_src,z_con_tgt)), crop(x_src))]'},
    }
    terms = cfg['losses']['total'].split(' + ')
    if set(terms) != set(expressions) or cfg['losses']['weighting'] != 'ALL_UNIT_WEIGHTED':
        raise PreparationError('E05 requires the exact five unit-weighted losses')
    return {'frozen': deepcopy(cfg['losses']),
            'terms': {term: {**expressions[term], 'weight': 1} for term in terms},
            'paper': cfg['source']['paper'],
            'alpha_beta_scope': 'Retained frozen paper parameters. Paper Eq.10 assigns alpha/beta '
                'to PMN; they do not reweight the five unit-weighted PCGAN Eq.5 terms.',
            'source_L1_or_R1_substitution': False, 'loss_computation_executed': False}


class PCGANAdapter:
    def __init__(self, config=None):
        self.config = authoritative(config) if config is not None else load_method_config('E05')
        validate_contract(self.config)

    def validate_source(self, *, source_root=None):
        return validate_source(self.config, source_root=source_root)

    def validate_environment(self):
        return environment_report('torch', ('numpy', 'torch', 'torchvision', 'PIL', 'scipy', 'cv2'))

    def prepare(self):
        cfg = authoritative(self.config)
        contracts = validate_contract(cfg)
        source = self.validate_source()
        return {'contracts': contracts, 'source': source,
                'architecture': architecture_mapping(cfg, source),
                'blur': deepcopy(contracts['overlay']['blur_operator']),
                'optimizer': {'target': 'torch.optim.Adam',
                              'config_fields': {'lr': 'optimizer.learning_rate', 'betas': 'optimizer.betas'},
                              'kwargs': {'lr': cfg['optimizer']['learning_rate'],
                                         'betas': deepcopy(cfg['optimizer']['betas'])},
                              'name': cfg['optimizer']['name'],
                              'upstream_lazy_R1_lr_beta_scaling': False,
                              'source_evidence': symbol_evidence(source,
                                  'optimizers/swapping_autoencoder_optimizer.py',
                                  'SwappingAutoencoderOptimizer', '__init__'),
                              'reference': 'Frozen E05 optimizer values override upstream training defaults'},
                'losses': loss_mapping(cfg), 'environment': self.validate_environment(),
                'logging_contract': load_logging_contract()['version'],
                'logging_runtime': 'methods.common.learned_runlog.LearnedRunContext',
                'external_assets': deepcopy(cfg['external_assets']),
                'training_launched': False, 'model_inference': False}

    def checkpoint_policy(self, seed, *, selection_split=None):
        cfg = authoritative(self.config)
        seed_plan(cfg, seed, 'torch')
        cp = cfg['checkpoint']
        end = cfg['training']['total_iterations']
        if (selection_split is not None or cp['selection_uses_val'] or cp['selection_uses_test']
                or cp['selection_scope'] != 'WITHIN_SEED' or cp['rule'] != 'BASELINE_FINAL_STATE_V1'
                or cp['terminal_state'] != f'iteration_{end}'):
            raise PreparationError('E05 terminal state cannot be selected by VAL/TEST or another budget')
        return {**deepcopy(cp), 'experiment_seed': seed, 'terminal_iteration': end,
                'save_terminal_if_cadence_misses': True, 'extra_optimizer_steps': 0,
                'checkpoint_written': False}

    def validate_checkpoint_event(self, seed, *, iteration, terminal=False, selected=False,
                                  extra_optimizer_steps=0, selection_split=None):
        cp = self.checkpoint_policy(seed, selection_split=selection_split)
        if (type(iteration) is not int or not 0 < iteration <= cp['terminal_iteration'] or
                type(extra_optimizer_steps) is not int or extra_optimizer_steps != 0 or
                ((terminal or selected) and iteration != cp['terminal_iteration'])):
            raise PreparationError('E05 checkpoint requires frozen terminal budget and zero extra steps')
        return {'iteration': iteration, 'terminal': terminal, 'selected': selected,
                'selection_reason': cp['rule'] if selected else 'Periodic/terminal retention',
                'checkpoint_written': False}

    def build_training_plan(self, seed, *, selection_split=None):
        cp = self.checkpoint_policy(seed, selection_split=selection_split)
        prepared = self.prepare()
        cfg = authoritative(self.config)
        return {**prepared, 'method_id': cfg['method_id'], 'seed': seed,
                'fidelity_class': cfg['fidelity_class'], 'fidelity_provenance': cfg['fidelity_provenance'],
                'reporting_label': cfg['reporting_label'], 'forbidden_wording': cfg['forbidden_wording'],
                'config_sha256': cfg['_runtime']['config_sha256'],
                'a5_overlay_sha256': prepared['contracts']['overlay_sha256'],
                'training': deepcopy(cfg['training']), 'seed_plan': seed_plan(cfg, seed, 'torch'),
                'checkpoint': cp, 'data_firewall': deepcopy(cfg['data']['splits']),
                'supervision': cfg['supervision'], 'attack_type_labels': cfg['attack_type_labels'],
                'gpat_identity_or_landmark_losses': cfg['gpat_identity_or_landmark_losses_in_main_row'],
                'mode': 'STATIC_PREPARATION_ONLY', 'checkpoint_created': False,
                'execution_dependencies': ['compatible pinned-source PyTorch/torchvision environment',
                    'upstream custom-op/runtime compatibility verification',
                    'future PCGAN runner integration using explicit source/target inputs and frozen losses; '
                    'do not launch the unmodified even-batch upstream training loop']}
