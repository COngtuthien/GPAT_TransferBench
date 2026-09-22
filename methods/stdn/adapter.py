"""Thin STDN static adapter and lazy binding to the unchanged official graph."""
import ast
from contextlib import contextmanager
from pathlib import Path

from methods.common.config import load_method_config
from methods.common.learned import (authoritative, checkpoint_plan, environment_report,
                                    mapping, PreparationError, seed_plan)
from methods.common.upstream import upstream_modules
from .landmarks import LandmarkAdapter
from .source import validate_source

# Field-name translation only. Values are read from the frozen config on each prepare.
CONFIG_FIELDS = {
    'IMAGE_SIZE': 'training.input_resolution', 'MAP_SIZE': 'training.map_size',
    'BATCH_SIZE': 'training.batch_size', 'G_D_RATIO': 'training.g_d_ratio',
    'MAX_EPOCH': 'training.max_epoch', 'STEPS_PER_EPOCH': 'training.steps_per_epoch',
    'STEPS_PER_EPOCH_VAL': 'training.val_steps', 'LEARNING_RATE': 'optimizer.learning_rate',
    'LEARNING_RATE_DECAY_FACTOR': 'optimizer.lr_decay.rate',
    'MOVING_AVERAGE_DECAY': 'optimizer.weight_ema_decay',
}


class STDNAdapter:
    method_id = 'E03'

    def __init__(self, config=None, *, source_root=None):
        self.config = authoritative(load_method_config(self.method_id) if config is None else config)
        if self.config['method_id'] != self.method_id:
            raise PreparationError('STDN requires E03 config')
        self.source_root = source_root

    def validate_source(self):
        return validate_source(self.config, source_root=self.source_root)

    def validate_environment(self):
        return environment_report('tensorflow', ('tensorflow', 'numpy', 'PIL', 'cv2', 'matplotlib', 'scipy'))

    def prepare(self):
        authoritative(self.config)
        self.source = self.validate_source()
        self.landmarks = LandmarkAdapter(self.config, source_root=self.source_root)
        self.settings = mapping(self.config, CONFIG_FIELDS, 'model/config.py:Config')
        tree = ast.parse((Path(self.source['root']) / 'model/config.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
        constants = {}
        for node in cls.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                constants.update({t.id: value for t in node.targets if isinstance(t, ast.Name)})
        for item in self.settings:
            if constants.get(item['target']) != item['value']:
                raise PreparationError(f"frozen STDN setting contradicts source: {item['target']}")
        decay_epochs = self.config['optimizer']['lr_decay']['every_steps'] / self.config['training']['steps_per_epoch']
        if decay_epochs != constants['NUM_EPOCHS_PER_DECAY']:
            raise PreparationError('STDN decay period contradicts pinned source')
        self.settings.append({'target': 'NUM_EPOCHS_PER_DECAY', 'value': decay_epochs,
                              'config_field': 'optimizer.lr_decay.every_steps / training.steps_per_epoch',
                              'source_file': 'model/model.py:get_train_op'})
        return self

    def build_training_plan(self, seed, *, selection_split=None):
        self.prepare()
        cfg = self.config
        return {'method_id': self.method_id, 'status': 'STATIC_PREPARED_NOT_EXECUTED',
                'experiment_seed': seed, 'config_sha256': cfg['_runtime']['config_sha256'],
                'snapshot_verified': True, 'fidelity_class': cfg['fidelity_class'],
                'source': self.source, 'settings_mapping': self.settings,
                'optimizer_contract': cfg['optimizer'], 'loss_contract': cfg['losses'],
                'seed_plan': seed_plan(cfg, seed, 'tensorflow'),
                'checkpoint': checkpoint_plan(cfg, seed, selection_split=selection_split),
                'environment': self.validate_environment(),
                'upstream_bindings': {'graph': 'train._step', 'generator': 'model.model.Gen',
                                     'discriminators': 'model.model.Disc_s',
                                     'optimizer': 'model.model.get_train_op',
                                     'geometry': 'model.warp.generate_offset_map',
                                     'dataset_reference': 'model.dataset.Dataset'},
                'input_adapter': {'landmarks': 'landmarks_px256 / image width; iBUG68; exact upstream 1-x + permutation',
                                  'image': 'canonical RGB256 float32 /255; upstream brightness retained',
                                  'horizontal_flip': cfg['training']['horizontal_flip'],
                                  'split_manifest': cfg['data']['split_manifest'],
                                  'training_split': 'TRAIN', 'diagnostics_split': 'VAL',
                                  'test_code_path_present': False,
                                  'tf_py_func_parallelism': 'methods.common.learned.serialized_tf_map: num_parallel_calls=1'},
                'execution_notes': ['Use effective model/config.py, never vestigial root config.py.',
                                    'Config.__init__/compile and upstream main are not invoked by preparation.',
                                    'G and D optimizers share upstream global_step; do not equate it to dataset steps.',
                                    'Runner must bind frozen dataset records and logging; no standalone training launcher in M6C2a.',
                                    'TF1 source README range excludes 1.13 while naming 1.13 as tested; '
                                    'no new version pin inferred. Final environment compatibility must be validated.'],
                'training_launched': False, 'checkpoint_created': False}

    @contextmanager
    def official_components(self):
        """Expose official functions lazily; never call the upstream training main."""
        self.prepare()
        env = self.validate_environment()
        if env['missing_modules']:
            raise PreparationError(env['reason'])
        with upstream_modules(self.source['root'], ('train', 'model.warp'), ('train', 'model')) as modules:
            tf = modules['train'].tf
            if not hasattr(tf, 'contrib') or not hasattr(tf, 'set_random_seed'):
                raise PreparationError('pinned STDN needs TensorFlow 1.x tf.contrib and graph APIs')
            yield modules
