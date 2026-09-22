"""Focused source/config/metadata tests. No framework or benchmark data loading."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from methods.common.config import load_method_config, FrozenConfigError
from methods.common import learned
from methods.common.learned_runlog import LearnedRunContext
from methods.common.runlog import environment_metadata
from methods.stdn.source import REQUIRED_FILES
from tools.m6c2a_preflight import StaticAccessAudit


class TestLearnedCommon(unittest.TestCase):
    def setUp(self):
        self.cfg = load_method_config('E03')

    def test_frozen_authority(self):
        changed = copy.deepcopy(self.cfg)
        changed['optimizer']['learning_rate'] *= 2
        with self.assertRaises(FrozenConfigError):
            learned.authoritative(changed)

    def test_seed_plan_all_three(self):
        for seed in self.cfg['seeds']['experiment_seeds']:
            plan = learned.seed_plan(self.cfg, seed, 'tensorflow')
            self.assertEqual(plan['framework_seed'], seed)
            self.assertEqual(plan['launch_environment']['PYTHONHASHSEED'], str(seed))
            self.assertFalse(plan['force_deterministic_algorithms'])

    def test_unknown_seed_rejected(self):
        for seed in (0, True, 42.0):
            with self.assertRaises(learned.PreparationError):
                learned.seed_plan(self.cfg, seed, 'torch')

    def test_seed_hook_requires_launch_hash_seed(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED': 'wrong'}):
            with self.assertRaises(learned.PreparationError):
                learned.apply_framework_seed(self.cfg, 42, 'tensorflow')

    def test_seed_hook_dispatch_and_python_numpy_repeat(self):
        module = Mock()
        samples = []
        with patch.dict(os.environ, {'PYTHONHASHSEED': '42'}), patch.object(
                learned.importlib, 'import_module', return_value=module):
            for _ in range(2):
                learned.apply_framework_seed(self.cfg, 42, 'tensorflow')
                samples.append((random.random(), np.random.rand()))
        self.assertEqual(samples[0], samples[1])
        module.set_random_seed.assert_called_with(42)

    def test_torch_hook_does_not_force_algorithms_or_cuda(self):
        module = Mock()
        with patch.dict(os.environ, {'PYTHONHASHSEED': '42'}), patch.object(
                learned.importlib, 'import_module', return_value=module):
            learned.apply_framework_seed(load_method_config('E06c'), 42, 'torch')
        module.manual_seed.assert_called_once_with(42)
        module.cuda.manual_seed_all.assert_not_called()
        module.use_deterministic_algorithms.assert_not_called()
        self.assertFalse(module.backends.cudnn.benchmark)
        self.assertTrue(module.backends.cudnn.deterministic)

    def test_worker_seed_dispatch(self):
        module = Mock()
        module.initial_seed.return_value = 2**32 + 7
        with patch.object(learned.importlib, 'import_module', return_value=module), patch.object(
                learned.random, 'seed') as py, patch.object(learned.np.random, 'seed') as npseed:
            learned.seed_torch_worker(3)
        py.assert_called_once_with(7)
        npseed.assert_called_once_with(7)

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(learned.PreparationError, 'missing pinned source'):
                learned.verify_source(self.cfg, REQUIRED_FILES, source_root=tmp)

    def test_source_commit_mismatch_rejected(self):
        original = learned.git_read
        def fake(root, *args):
            return '0'*40 if args == ('rev-parse', 'HEAD') else original(root, *args)
        with patch.object(learned, 'git_read', side_effect=fake):
            with self.assertRaisesRegex(learned.PreparationError, 'commit mismatch'):
                learned.verify_source(self.cfg, REQUIRED_FILES)

    def test_repository_mismatch_rejected(self):
        original = learned.git_read
        def fake(root, *args):
            return 'https://invalid.example/repo' if args[:2] == ('remote', 'get-url') else original(root, *args)
        with patch.object(learned, 'git_read', side_effect=fake):
            with self.assertRaisesRegex(learned.PreparationError, 'repository identity'):
                learned.verify_source(self.cfg, REQUIRED_FILES)

    def test_required_source_file_missing_rejected(self):
        with self.assertRaisesRegex(learned.PreparationError, 'missing/unsafe'):
            learned.verify_source(self.cfg, REQUIRED_FILES + ('missing_required.py',))

    def test_dirty_source_blob_rejected(self):
        original = Path.read_bytes
        def changed(path):
            raw = original(path)
            return raw + b'\n# changed' if str(path).endswith('stdn/model/model.py') else raw
        with patch.object(Path, 'read_bytes', changed):
            with self.assertRaisesRegex(learned.PreparationError, 'differs from pinned blob'):
                learned.verify_source(self.cfg, REQUIRED_FILES)

    def test_asset_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'synthetic_metadata.txt'
            p.write_text('metadata only')
            asset = {'external_runtime_path': str(p), 'bytes': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertFalse(learned.verify_asset(asset)['deserialized'])
            for key, value in [('bytes', 1), ('sha256', '0'*64)]:
                with self.subTest(key=key), self.assertRaises(learned.PreparationError):
                    learned.verify_asset(dict(asset, **{key: value}))

    def test_asset_missing(self):
        with self.assertRaises(learned.PreparationError):
            learned.verify_asset({'external_runtime_path': '/tmp/nonexistent-m6c2a-asset'})

    def test_checkpoint_types_and_final_only(self):
        for kind in ('official', 'periodic', 'terminal', 'selected'):
            result = learned.checkpoint_metadata(self.cfg, 42, epoch=50, global_step=123,
                        path='ckpt-50.index', file_size_bytes=10, sha256='a'*64,
                        checkpoint_type=kind, selected_for_final=True)
            self.assertEqual(result['selection_reason'], self.cfg['checkpoint']['rule'])
        with self.assertRaises(learned.PreparationError):
            learned.checkpoint_metadata(self.cfg, 42, epoch=49, global_step=123,
                        path='ckpt-49', file_size_bytes=10, sha256='a'*64,
                        checkpoint_type='selected', selected_for_final=True)

    def test_checkpoint_selection_split_rejected(self):
        for split in ('TRAIN', 'VAL', 'TEST'):
            with self.assertRaises(learned.PreparationError):
                learned.checkpoint_plan(self.cfg, 42, selection_split=split)

    def test_static_audit_denies_data_before_open(self):
        audit = StaticAccessAudit()
        for path in ('/tmp/synthetic.png', str(learned.ROOT/'manifests/fake.json'), '/tmp/fake.parquet'):
            with self.assertRaises(RuntimeError):
                audit('open', (path, 'r', 0))
        self.assertEqual(audit.rejected_data_opens, 3)

    def test_learned_logging_append_resume_and_reasons(self):
        env = environment_metadata()
        env.update(framework='tensorflow', tensorflow_version=None,
                   gpu_note='Static metadata test; hardware not probed')
        reasons = {k: 'Static test: not executed/measured' for k, v in env.items() if v is None}
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(config=self.cfg, method_id='E03', seed=42, runtime_root=tmp,
                          environment=env, missing_environment_reasons=reasons)
            with LearnedRunContext(**kwargs) as run:
                record = {k: None for k in run.contract['trajectory']['required_fields']}
                record['missing_field_reasons'] = {k: 'Synthetic metadata validation only' for k in record}
                run.log_epoch(record)
                before = run.path('metrics').read_bytes()
                directory = run.run_dir
            with LearnedRunContext(**kwargs, resume=True) as run:
                self.assertTrue(run.path('metrics').read_bytes().startswith(before))
                self.assertFalse(json.loads(run.path('checkpoint_index').read_text())['checkpoints'])
            manifest = json.loads((directory/'run_manifest.json').read_text())
            self.assertEqual(manifest['framework'], 'tensorflow')
            self.assertNotIn('non-learned', json.dumps(manifest))
            self.assertNotIn('non-learned', (directory/'run_summary.json').read_text())

    def test_null_environment_requires_reason(self):
        with self.assertRaises(learned.PreparationError):
            LearnedRunContext(config=self.cfg, method_id='E03', seed=42, runtime_root='/tmp/unused',
                              environment={}, missing_environment_reasons={})

    def test_environment_probe_never_imports_framework(self):
        with patch.object(learned.importlib, 'import_module', side_effect=AssertionError('import')):
            report = learned.environment_report('torch', ('torch', 'numpy'))
        self.assertFalse(report['gpu_probe_performed'])
        self.assertFalse(report['final_execution_environment_frozen'])

    def test_serialized_tf_map_preserves_callback(self):
        dataset, callback = Mock(), Mock()
        learned.serialized_tf_map(dataset, callback)
        dataset.map.assert_called_once_with(callback, num_parallel_calls=1)

    def test_torch_loader_rng_hook(self):
        module = Mock()
        with patch.object(learned.importlib, 'import_module', return_value=module):
            options = learned.torch_loader_options(load_method_config('E06c'), 1337)
        module.Generator.return_value.manual_seed.assert_called_once_with(1337)
        self.assertIs(options['worker_init_fn'], learned.seed_torch_worker)

    def test_nested_test_metrics_rejected(self):
        env = environment_metadata()
        env.update(framework='tensorflow', tensorflow_version=None)
        reasons = {k: 'Synthetic metadata only' for k, v in env.items() if v is None}
        with tempfile.TemporaryDirectory() as tmp:
            with LearnedRunContext(config=self.cfg, method_id='E03', seed=42, runtime_root=tmp,
                                   environment=env, missing_environment_reasons=reasons) as run:
                with self.assertRaises(learned.PreparationError):
                    run.log_epoch({'train_metrics': {'test_accuracy': 1}})
                with self.assertRaises(learned.PreparationError):
                    run.close(summary={'checkpoint_selection_rule': 'VAL_BEST'})
