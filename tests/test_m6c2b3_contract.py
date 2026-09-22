"""E07c source and effective A1/A3/A6 authority; metadata only."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import PreparationError, git_read
from methods.difffas import DiffFASAdapter, validate_contract, training_record
from methods.difffas.contract import OVERLAY


class TestDiffFASContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DiffFASAdapter()
        cls.cfg = cls.adapter.config
        cls.prepared = cls.adapter.prepare()

    def test_source_identity(self):
        src = self.prepared['source']
        self.assertEqual(src['repository'], 'https://github.com/murphytju/DiffFAS')
        self.assertEqual(src['commit'], '23f40519ec25a833ebc06842aa6fbab74fad4d15')
        self.assertEqual(src['verification'], 'PASS')

    def test_wrong_commit_rejected(self):
        def wrong(root, *args):
            return '0' * 40 if args == ('rev-parse', 'HEAD') else git_read(root, *args)
        with patch('methods.common.learned.git_read', side_effect=wrong):
            with self.assertRaisesRegex(PreparationError, 'commit mismatch'):
                self.adapter.validate_source()

    def test_wrong_repository_rejected(self):
        def wrong(root, *args):
            return 'https://invalid.example/repo' if args == ('remote', 'get-url', 'origin') else git_read(root, *args)
        with patch('methods.common.learned.git_read', side_effect=wrong):
            with self.assertRaisesRegex(PreparationError, 'repository identity'):
                self.adapter.validate_source()

    def test_missing_source_checkout(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(PreparationError, 'missing pinned source'):
            self.adapter.validate_source(source_root=tmp)

    def test_required_file_missing(self):
        original = Path.is_file
        with patch.object(Path, 'is_file', lambda p: False if p.name == 'custom_rn.py' else original(p)):
            with self.assertRaisesRegex(PreparationError, 'required source file'):
                self.adapter.validate_source()

    def test_modified_source_bytes_rejected(self):
        original = Path.read_bytes
        with patch.object(Path, 'read_bytes', lambda p: original(p) + (b'\n# changed' if p.name == 'custom_rn.py' else b'')):
            with self.assertRaisesRegex(PreparationError, 'pinned blob'):
                self.adapter.validate_source()

    def test_frozen_base_and_snapshot(self):
        rel = self.cfg['_runtime']['config_path']
        raw = (ROOT / rel).read_bytes()
        self.assertEqual(raw, (ROOT / 'frozen_config_snapshot' / rel).read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.cfg['_runtime']['config_sha256'])

    def test_adaptation_hash_and_snapshot(self):
        c = self.prepared['contracts']
        self.assertEqual(c['adaptation_sha256'], self.cfg['adaptation_semantics']['frozen_config_sha256'])
        self.assertEqual((ROOT / c['adaptation_path']).read_bytes(),
                         (ROOT / 'frozen_config_snapshot' / c['adaptation_path']).read_bytes())

    def test_a6_only_supersedes_feature_shapes(self):
        c = self.prepared['contracts']
        original = deepcopy(self.cfg['conditioning_encoder'])
        effective = deepcopy(c['effective_encoder'])
        self.assertEqual(effective['feature_interface_at_256']['x32x32']['shape'], [32, 32, 256])
        self.assertEqual(effective['feature_interface_at_256']['x16x16']['shape'], [16, 16, 512])
        self.assertEqual(effective['feature_interface_at_256']['x8x8']['shape'], [8, 8, 512])
        self.assertEqual(effective['feature_interface_at_256']['embg']['shape'], ['B', 7])
        effective['feature_interface_at_256'] = original['feature_interface_at_256']
        self.assertEqual(effective, original)
        self.assertEqual(self.cfg['conditioning_encoder']['feature_interface_at_256']['x32x32']['shape'], [32, 32, 128])

    def test_amendment_identity_binding(self):
        c = self.prepared['contracts']
        for name, identity in c['documents'].items():
            self.assertEqual(hashlib.sha256((ROOT / identity['path']).read_bytes()).hexdigest(), identity['sha256'])
        self.assertEqual(c['overlay']['base_config_sha256'], self.cfg['_runtime']['config_sha256'])
        self.assertEqual(c['overlay']['amendment_a3'], c['documents']['A3'])

    def test_changed_a6_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'overlay.yaml'
            p.write_bytes((ROOT / OVERLAY).read_bytes().replace(b'[32, 32, 256]', b'[32, 32, 128]'))
            with self.assertRaisesRegex(FrozenConfigError, 'changed immutable'):
                validate_contract(self.cfg, overlay_path=p)

    def test_missing_adaptation_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(FrozenConfigError, 'missing immutable'):
            validate_contract(self.cfg, adaptation_path=Path(tmp) / 'absent')

    def test_scientific_overrides_refused(self):
        for group, field, value in [('training', 'use_pair', True), ('training', 'batch_size', 8),
                                    ('sampler', 'tensor_set_loaded', 'ema'), ('optimizer', 'learning_rate', .001)]:
            cfg = deepcopy(self.cfg)
            cfg[group][field] = value
            with self.subTest(field=field), self.assertRaises(FrozenConfigError):
                DiffFASAdapter(cfg)

    def test_wrong_method_refused(self):
        with self.assertRaisesRegex(PreparationError, 'requires E07c'):
            DiffFASAdapter(load_method_config('E05'))

    def test_fidelity_and_label(self):
        self.assertEqual(self.cfg['fidelity_class'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(self.cfg['reporting_label'], 'DiffFAS-BIN-IDFREE (controlled encoder reconstruction)')
        self.assertEqual(self.cfg['deviation'], 'DEV-021')
        self.assertEqual(self.cfg['fidelity_provenance'],
                         ['AMENDMENT_A1_IDFREE_ADAPTATION', 'AMENDMENT_A3_CONTROLLED_ENCODER_RECONSTRUCTION'])
        for forbidden in self.cfg['forbidden_wording']:
            self.assertNotIn(forbidden, self.cfg['reporting_label'])

    def test_unpaired_training_semantics(self):
        code = self.prepared['contracts']['adaptation']['official_code_path']
        self.assertIs(code['use_pair'], False)
        self.assertEqual(code['model_in_channels'], 3)
        self.assertIn('torch.cat([x_t], 1)', code['training_input'])
        self.assertEqual(code['conditioning'], 'style_spoof')
        self.assertEqual(code['target'], 'epsilon_of_GT')
        self.assertEqual(code['content_training_role'], 'INERT_API_PLACEHOLDER')
        self.assertEqual(code['vb_term_content_path'], 'NONE')
        self.assertTrue(self.prepared['source_evidence']['training_content_inert'])

    def test_guide_identity_and_no_sampler_replacement(self):
        guide = self.prepared['guide']
        self.assertEqual(guide['policy']['benchmark_policy'], 'DETERMINISTIC_RAW_BYTE_SHA256_RANKING')
        self.assertEqual(guide['policy']['eligibility'], ['m2_complete', 'split_TRAIN', 'label_SPOOF', 'same_dataset'])
        self.assertEqual(guide['manifest']['rows'], 8838)
        self.assertEqual(guide['manifest']['sha256'], '0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6')
        self.assertEqual(guide['selection_mode'], 'CONSUME_FROZEN_MANIFEST_NO_RESAMPLING')
        self.assertTrue(guide['policy']['self_guide']['allowed'])
        self.assertFalse(guide['manifest_opened'])

    def test_identity_and_attack_targets_forbidden(self):
        guide = self.prepared['guide']
        for name, value in guide['identity_firewall'].items():
            if name != 'siw_participates_without_identity':
                self.assertIs(value, False)
        for name in ('attack_raw_as_class_target', 'attack_macro_as_class_target', 'dataset_id_as_semantic_class'):
            self.assertEqual(guide['style_supervision'][name], 'FORBIDDEN')

    def test_record_does_not_consume_identity_or_content(self):
        class PoisonRecord(dict):
            def __getitem__(self, key):
                if key in ('subject_id_global', 'attack_raw', 'attack_macro', 'content_live_id'):
                    raise AssertionError('forbidden input read')
                return super().__getitem__(key)
        record = PoisonRecord(split='TRAIN', use_pair=False, style_id='SPOOF_BINARY',
            selection_policy='DETERMINISTIC_RAW_BYTE_SHA256_RANKING', gt_spoof_id='toy_gt',
            guide_spoof_id='toy_gt', dataset='toy_dataset')
        self.assertEqual(training_record(record)['guide_spoof_id'], 'toy_gt')
        self.assertNotIn('content_live_id', training_record(record))
        record['split'] = 'TEST'
        with self.assertRaises(PreparationError):
            training_record(record)


if __name__ == '__main__':
    unittest.main()
