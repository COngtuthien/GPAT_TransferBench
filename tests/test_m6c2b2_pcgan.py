"""E05 source/authority/plan tests. No model, checkpoint or dataset execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import PreparationError, checkpoint_metadata
from methods.pcgan import PCGANAdapter, validate_contract
from methods.pcgan.contract import OVERLAY


class TestPCGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = PCGANAdapter()
        cls.cfg = cls.adapter.config
        cls.plan = cls.adapter.build_training_plan(cls.cfg['seeds']['experiment_seeds'][0])

    def test_source_pin_and_repository(self):
        src = self.plan['source']
        basis = self.cfg['source']['executable_architecture_basis']
        self.assertEqual(src['commit'], basis['pinned_commit'])
        self.assertEqual(src['repository'], basis['repository'])
        self.assertEqual(src['verification'], 'PASS')

    def test_wrong_source_commit_rejected(self):
        from methods.common.learned import git_read
        def altered(root, *args):
            return '0' * 40 if args == ('rev-parse', 'HEAD') else git_read(root, *args)
        with patch('methods.common.learned.git_read', side_effect=altered):
            with self.assertRaisesRegex(PreparationError, 'commit mismatch'):
                self.adapter.validate_source()

    def test_wrong_repository_rejected(self):
        from methods.common.learned import git_read
        def altered(root, *args):
            return 'https://example.invalid/substitute' if args == ('remote', 'get-url', 'origin') else git_read(root, *args)
        with patch('methods.common.learned.git_read', side_effect=altered):
            with self.assertRaisesRegex(PreparationError, 'repository identity'):
                self.adapter.validate_source()

    def test_required_source_missing_rejected(self):
        original = Path.is_file
        def absent(p):
            return False if p.name == 'encoder.py' else original(p)
        with patch.object(Path, 'is_file', absent):
            with self.assertRaisesRegex(PreparationError, 'required source file'):
                self.adapter.validate_source()

    def test_source_checkout_missing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PreparationError, 'missing pinned source'):
                self.adapter.validate_source(source_root=tmp)

    def test_source_byte_change_rejected(self):
        original = Path.read_bytes
        def changed(p):
            raw = original(p)
            return raw + b'\n# changed\n' if p.name == 'encoder.py' else raw
        with patch.object(Path, 'read_bytes', changed):
            with self.assertRaisesRegex(PreparationError, 'pinned blob'):
                self.adapter.validate_source()

    def test_config_hash_and_snapshot(self):
        raw = (ROOT / self.cfg['_runtime']['config_path']).read_bytes()
        self.assertEqual(raw, (ROOT / 'frozen_config_snapshot' / self.cfg['_runtime']['config_path']).read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.plan['config_sha256'])

    def test_all_scientific_overrides_rejected(self):
        for section, field, value in [('training', 'batch_size', 2), ('optimizer', 'learning_rate', .01),
                                      ('losses', 'alpha', .3), ('checkpoint', 'selection_uses_val', True)]:
            cfg = deepcopy(self.cfg)
            cfg[section][field] = value
            with self.subTest(field=field), self.assertRaises(FrozenConfigError):
                PCGANAdapter(cfg)

    def test_wrong_method_rejected(self):
        with self.assertRaisesRegex(PreparationError, 'requires E05'):
            PCGANAdapter(load_method_config('E03'))

    def test_a5_bound_to_exact_base_and_document(self):
        contracts = self.plan['contracts']
        self.assertEqual(contracts['overlay']['base_config_sha256'], self.plan['config_sha256'])
        self.assertEqual(contracts['A2']['sha256'], self.cfg['amendment_a2_sha256'])
        self.assertEqual(contracts['overlay_sha256'], self.plan['a5_overlay_sha256'])
        for name in ('A2', 'A5'):
            self.assertEqual(hashlib.sha256((ROOT / contracts[name]['path']).read_bytes()).hexdigest(),
                             contracts[name]['sha256'])

    def test_changed_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'overlay.yaml'
            p.write_bytes((ROOT / OVERLAY).read_bytes().replace(b'stride: 2', b'stride: 1'))
            with self.assertRaisesRegex(FrozenConfigError, 'changed immutable'):
                validate_contract(self.cfg, overlay_path=p)

    def test_missing_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FrozenConfigError, 'missing immutable'):
                validate_contract(self.cfg, overlay_path=Path(tmp) / 'absent.yaml')

    def test_reporting_fidelity(self):
        self.assertEqual(self.plan['fidelity_class'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(self.plan['reporting_label'], 'PCGAN (controlled architecture resolution)')
        self.assertIn('AMENDMENT_A2_A2_07_ARCHITECTURE_RESOLUTION', self.plan['fidelity_provenance'])
        for forbidden in self.cfg['forbidden_wording']:
            self.assertNotIn(forbidden, self.plan['reporting_label'])

    def test_three_seed_plans(self):
        seeds = self.cfg['seeds']['experiment_seeds']
        self.assertEqual(seeds, [42, 1337, 2026])
        for seed in seeds:
            plan = self.adapter.build_training_plan(seed)
            self.assertEqual(plan['seed_plan']['framework_seed'], seed)
            self.assertEqual(plan['checkpoint']['experiment_seed'], seed)
            self.assertEqual(plan['config_sha256'], self.plan['config_sha256'])
            json.dumps(plan)

    def test_bad_seed_rejected(self):
        for seed in (True, 0, '42', 42.0):
            with self.subTest(seed=seed), self.assertRaises(PreparationError):
                self.adapter.build_training_plan(seed)

    def test_training_budget_and_optimizer(self):
        self.assertEqual(self.plan['training']['batch_size'], 1)
        self.assertEqual(self.plan['training']['total_iterations'], 4000)
        self.assertEqual(self.plan['optimizer']['name'], 'Adam')
        self.assertEqual(self.plan['optimizer']['kwargs'], {'lr': 1e-6, 'betas': [.9, .999]})
        self.assertFalse(self.plan['optimizer']['upstream_lazy_R1_lr_beta_scaling'])

    def test_exact_loss_terms_and_unit_weights(self):
        terms = self.plan['losses']['terms']
        self.assertEqual(list(terms), ['L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat'])
        self.assertTrue(all(term['weight'] == 1 for term in terms.values()))
        self.assertEqual(self.plan['losses']['frozen'], self.cfg['losses'])
        self.assertEqual(self.plan['losses']['frozen']['alpha'], .2)
        self.assertEqual(self.plan['losses']['frozen']['beta'], 1e-6)
        self.assertFalse(self.plan['losses']['source_L1_or_R1_substitution'])

    def test_training_supervision_firewall(self):
        self.assertEqual(self.plan['supervision'], 'pooled_train_live_attack_labels')
        self.assertEqual(self.plan['attack_type_labels'], 'FORBIDDEN')
        self.assertFalse(self.plan['gpat_identity_or_landmark_losses'])
        self.assertFalse(self.plan['data_firewall']['TEST']['allowed'])
        self.assertEqual(self.plan['data_firewall']['VAL']['used_for'], ['diagnostics_only'])

    def test_terminal_policy(self):
        cp = self.plan['checkpoint']
        self.assertEqual(cp['rule'], 'BASELINE_FINAL_STATE_V1')
        self.assertEqual(cp['terminal_iteration'], 4000)
        self.assertTrue(cp['save_terminal_if_cadence_misses'])
        self.assertEqual(cp['extra_optimizer_steps'], 0)
        self.assertFalse(cp['checkpoint_written'])

    def test_no_split_selection(self):
        for split in ('TRAIN', 'VAL', 'TEST', 'best_VAL'):
            with self.subTest(split=split), self.assertRaises(PreparationError):
                self.adapter.build_training_plan(42, selection_split=split)

    def test_terminal_event_and_rejections(self):
        self.assertFalse(self.adapter.validate_checkpoint_event(42, iteration=4000, terminal=True)['checkpoint_written'])
        for kwargs in ({'iteration': 3999, 'selected': True}, {'iteration': 4001},
                       {'iteration': 4000, 'extra_optimizer_steps': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(PreparationError):
                self.adapter.validate_checkpoint_event(42, **kwargs)

    def test_common_learned_checkpoint_metadata_support(self):
        # Metadata-only synthetic record; no checkpoint bytes or files are written.
        args = dict(epoch=0, global_step=4000, path='synthetic_metadata_only',
                    file_size_bytes=1, sha256='a' * 64, checkpoint_type='terminal', selected_for_final=True)
        result = checkpoint_metadata(self.cfg, 42, **args)
        self.assertEqual(result['selection_reason'], 'BASELINE_FINAL_STATE_V1')
        args['global_step'] = 3999
        with self.assertRaises(PreparationError):
            checkpoint_metadata(self.cfg, 42, **args)

    def test_no_execution_in_plan(self):
        self.assertEqual(self.plan['mode'], 'STATIC_PREPARATION_ONLY')
        for name in ('training_launched', 'model_inference', 'checkpoint_created'):
            self.assertFalse(self.plan[name])
        self.assertFalse(self.plan['environment']['gpu_probe_performed'])
        self.assertFalse(self.plan['environment']['final_execution_environment_frozen'])


if __name__ == '__main__':
    unittest.main()
