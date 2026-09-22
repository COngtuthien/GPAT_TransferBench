"""A3/A6 encoder plans; synthetic metadata; optional CPU construction only."""
from copy import deepcopy
import importlib.util
import unittest

from methods.common.config import FrozenConfigError
from methods.common.learned import PreparationError
from methods.difffas import DiffFASAdapter
from methods.difffas.encoder import encoder_plan, class_index, verify_future_checkpoint, encoder_model


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DiffFASAdapter()
        cls.cfg = cls.adapter.config
        cls.prepared = cls.adapter.prepare()
        cls.plan = cls.prepared['auxiliary_plan']

    def test_exact_architecture_and_corrected_features(self):
        self.assertEqual(self.plan['architecture'], 'models/custom_rn.py::resnet18')
        self.assertEqual(self.prepared['source_evidence']['topology'], [3, 4, 6, 3])
        self.assertEqual(self.prepared['source_evidence']['block'], 'BasicBlock')
        self.assertFalse(self.plan['projection_layers_added'])
        self.assertEqual([self.plan['feature_interface_at_256'][k]['shape'] for k in
                          ('x32x32', 'x16x16', 'x8x8')], [[32,32,256], [16,16,512], [8,8,512]])

    def test_head_discarded(self):
        self.assertTrue(self.prepared['source_evidence']['fourth_output_discarded'])
        self.assertFalse(self.plan['feature_interface_at_256']['embg']['consumed'])
        self.assertEqual(self.plan['feature_interface_at_256']['embg']['shape'], ['B', 7])
        self.assertEqual(self.plan['training']['head'], 'fc = nn.Linear(512, 7)')

    def test_k7_order_and_mapping(self):
        classes = ['live', 'makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay']
        self.assertEqual(self.plan['objective']['classes'], classes)
        for i, label in enumerate(classes):
            self.assertEqual(class_index(label, split='TRAIN'), i)

    def test_counts_and_class_source_identity(self):
        obj = self.plan['objective']
        self.assertEqual(obj['train_total'], 14467)
        self.assertEqual(sum(obj['train_counts'].values()), 14467)
        self.assertEqual(obj['train_counts'], dict(live=5629,print=2838,replay=2178,partial=1911,
                                                  mask_3d=1056,makeup=759,mask_2d=96))
        self.assertEqual(obj['class_source_sha256'], 'c7d23e3e3e6a526ae412a50125d0d56379ce98bca99d4133ead3fda7139a3812')
        self.assertFalse(self.plan['class_metadata_opened'])

    def test_train_only_excludes_other_spoof(self):
        for label, split in [('other_spoof','TRAIN'),('live','VAL'),('print','TEST')]:
            with self.subTest(label=label,split=split), self.assertRaises(PreparationError):
                class_index(label, split=split)

    def test_preprocessing_no_augmentation_or_balancing(self):
        c = self.plan['training']
        self.assertEqual(c['input_resolution'], 256)
        self.assertEqual(c['preprocessing'], 'Resize((256,256)) -> ToTensor -> Normalize([0.5]*3, [0.5]*3)')
        self.assertEqual(c['augmentation'], 'NONE')
        self.assertEqual(c['class_balancing'], 'NONE')
        self.assertEqual(c['loss'], 'CrossEntropyLoss_on_fourth_forward_output')

    def test_sgd_and_loader(self):
        c = self.plan['training']
        self.assertEqual((c['optimizer'],c['learning_rate'],c['momentum'],c['weight_decay']), ('SGD',.002,.9,.005))
        self.assertEqual(c['scheduler'], 'NONE')
        self.assertEqual(c['batch_size'], 256)
        self.assertTrue(c['drop_last'])
        self.assertTrue(c['shuffle'])
        self.assertEqual(c['workers'], 6)

    def test_final_whole_module_checkpoint(self):
        self.assertEqual(self.plan['training']['epochs'], 200)
        self.assertEqual(self.plan['training']['checkpoint_rule'], 'FINAL_STATE_AFTER_EPOCH_200')
        self.assertIn('WHOLE nn.Module', self.plan['checkpoint']['serialization'])
        self.assertIn('torch.load(path).cuda()', self.plan['checkpoint']['serialization'])
        self.assertTrue(self.plan['checkpoint']['external_runtime_path'].endswith('aux_encoder/seed_42/checkpoints/encoder_final.pkl'))

    def test_auxiliary_seed_and_single_run(self):
        self.assertEqual((self.plan['seed'],self.plan['runs']), (42,1))
        self.assertEqual(self.plan['three_different_encoders_trained'], 'FORBIDDEN')
        self.assertEqual(self.plan['candidate_encoder_search'], 'FORBIDDEN')
        self.assertFalse(self.plan['val_selection'])
        self.assertFalse(self.plan['test_access'])
        cfg = deepcopy(self.cfg)
        cfg['conditioning_encoder']['auxiliary_encoder_training_runs'] = 3
        with self.assertRaises(FrozenConfigError):
            encoder_plan(cfg)

    def test_future_hash_not_fabricated_and_required(self):
        self.assertEqual(self.plan['checkpoint']['sha256'], 'RECORDED_AFTER_TRAINING')
        self.assertFalse(self.plan['checkpoint']['exists_verified'])
        for digest in (None, 'RECORDED_AFTER_TRAINING', 'z'*64):
            with self.subTest(digest=digest), self.assertRaisesRegex(PreparationError, 'freeze auxiliary SHA256'):
                verify_future_checkpoint('/tmp/future_runtime', digest)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'),
                         'PyTorch/torchvision unavailable: pinned custom_rn CPU construction requires both; AST/interface verified separately')
    def test_real_cpu_encoder_construction(self):
        with encoder_model() as model:
            self.assertEqual([len(model.layer1),len(model.layer2),len(model.layer3),len(model.layer4)], [3,4,6,3])
            self.assertEqual((model.fc.in_features,model.fc.out_features), (512,7))
            self.assertEqual(model.__class__.__module__, 'custom_rn')
            self.assertEqual(next(model.parameters()).device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
