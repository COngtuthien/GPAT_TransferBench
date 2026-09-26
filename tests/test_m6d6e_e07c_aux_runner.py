"""M6D6e E07c auxiliary-encoder production runner: contract, TRAIN population, reader, engine, CLI and evidence.

Static checks run everywhere and never claim CUDA execution. The population tests read the
TRAIN rows of manifests/split_v1.parquet through the production reader (authorized TRAIN
metadata; VAL/TEST rows never reach Python). No test opens a canonical face: reader tests use
synthetic PNGs in a temporary root. The optional live Torch tests (GPU host) run on the CPU
with synthetic tensors only; the one real-TRAIN epoch is validated from the process evidence.
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from methods.common.config import sha256_file
from methods.common.learned import PreparationError
from methods.difffas import aux_runner_io as aio

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '9bfb8dc0fc1e1a4be7ea7f1d4c788b6d2dbbd00f'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
AUTHORITY_SHA = {
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md':
        'b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a',
    'configs/amendments/e07c_a6_feature_interface_source_correction.yaml':
        'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md':
        '759f72d860ccf49927c214bb4ce84e87bcedd98eb8f06ef5fe7361a758d6bc81',
    'configs/amendments/e07c_a7_execution_policy.yaml':
        '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md':
        '0a46c3b06e277ad38b17a6e85fe2924441d8d13b1de887bf54903dda9aa292ba',
    'configs/methods/e07c_difffas_bin_idfree.yaml': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
    'configs/frozen/difffas_bin_idfree_v1.yaml': 'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
    'environments/e07c.lock.json': LOCK_SHA,
    'manifests/split_v1.parquet': 'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d',
    'manifests/artifact_probe_classes_v1.json': 'c7d23e3e3e6a526ae412a50125d0d56379ce98bca99d4133ead3fda7139a3812'}
ENGINE = ROOT / 'methods/difffas/aux_runner.py'
IO = ROOT / 'methods/difffas/aux_runner_io.py'
HARNESS = ROOT / 'methods/difffas/aux_runner_qualification.py'
CLI = ROOT / 'tools/run_e07c_aux.py'
EVIDENCE = {k: ROOT / f'outputs/audit/M6D6E_E07C_AUX_{v}.json'
            for k, v in (('b1', 'B256_PROCESS_1'), ('b2', 'B256_PROCESS_2'), ('epoch', 'ONE_EPOCH'))}
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_PIL = importlib.util.find_spec('PIL') is not None


def fn_source(path, name, cls=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return ast.unparse(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))


def load_cli():
    spec = importlib.util.spec_from_file_location('run_e07c_aux_under_test', CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_id(i):
    return hashlib.sha256(f'm6d6e-test-{i}'.encode()).hexdigest()


class TestM6D6eStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = aio.load_contract()
        cls.ev = {k: json.loads(p.read_text()) for k, p in EVIDENCE.items()}

    # ---------------------------------------------------------------- 1-4 authority / source / amendments / env
    def test_01_current_authority(self):
        head = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', AUTHORITY, 'HEAD'])
        self.assertEqual(head.returncode, 0)
        self.assertEqual(self.contract['authority_commit'], AUTHORITY)

    def test_02_source_pin_unchanged(self):
        src = ROOT / 'third_party/source_cache/difffas'
        run = lambda *a: subprocess.check_output(['git', '-C', str(src), *a], text=True).strip()
        self.assertEqual((run('rev-parse', 'HEAD'), run('rev-parse', 'HEAD^{tree}')), (PIN, TREE))
        self.assertEqual(self.contract['source_pin']['commit'], PIN)
        for rel, key in (('models/pretrain_classifier.py', 'writer_sha256'), ('models/custom_rn.py', 'architecture_sha256')):
            self.assertEqual(sha256_file(src / rel), self.contract['source_pin'][key])

    def test_03_a3_a6_a7_identities_unchanged(self):
        for rel, digest in AUTHORITY_SHA.items():
            self.assertEqual(sha256_file(ROOT / rel), digest, rel)
            if rel in self.contract['bound_inputs_sha256']:
                self.assertEqual(self.contract['bound_inputs_sha256'][rel], digest)

    def test_04_environment_unchanged(self):
        self.assertEqual(sha256_file(ROOT / 'environments/e07c.lock.json'), LOCK_SHA)
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        for k in ('b1', 'b2', 'epoch'):
            env = self.ev[k]['environment_before']
            self.assertEqual({a: b for a, b in env.items() if a != 'launch_environment'},
                             {a: b for a, b in lock['identity'].items() if a != 'launch_environment'})
            self.assertEqual(env, self.ev[k]['environment_after'])

    def test_contract_bound_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / 'contract.yaml'
            c = json.loads((ROOT / aio.CONTRACT_PATH).read_text())
            c['bound_inputs_sha256']['manifests/split_v1.parquet'] = '0' * 64
            bad.write_text(json.dumps(c))
            with mock.patch.object(aio, 'ROOT', ROOT), mock.patch.object(aio, 'CONTRACT_PATH', str(bad)):
                with self.assertRaises(PreparationError):
                    aio.load_contract()

    # ---------------------------------------------------------------- 5-9 source loader / epochs
    def test_05_to_09_source_loader_semantics(self):
        tree = ast.parse((ROOT / 'third_party/source_cache/difffas/models/pretrain_classifier.py').read_text())
        loader = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'train_loader')
        kw = {k.arg: ast.literal_eval(k.value) for k in loader.value.keywords}
        self.assertEqual(kw, {'batch_size': 256, 'shuffle': True, 'num_workers': 6, 'drop_last': True})
        self.assertEqual((aio.BATCH_SIZE, aio.WORKERS, aio.SHUFFLE, aio.DROP_LAST), (256, 6, True, True))
        epochs = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'num_epochs')
        self.assertEqual((ast.literal_eval(epochs.value), aio.EPOCHS), (200, 200))
        loader_src = fn_source(ENGINE, 'build_loader')
        self.assertIn('batch_size=aio.BATCH_SIZE, shuffle=aio.SHUFFLE', loader_src)
        for forbidden in ('generator=', 'worker_init_fn=', 'pin_memory=', 'sampler=', 'persistent_workers=',
                          'prefetch_factor='):
            self.assertNotIn(forbidden, loader_src)
        self.assertEqual((aio.STEPS_PER_EPOCH, aio.CONSUMED_PER_EPOCH, aio.DROPPED_PER_EPOCH), (56, 14336, 131))

    # ---------------------------------------------------------------- 10-16 population, reader
    def test_10_k7_class_order(self):
        cmap = aio.read_class_map()
        self.assertEqual(tuple(cmap['classes']), aio.CLASSES)
        self.assertEqual(cmap['class_index'], {'live': 0, 'makeup': 1, 'mask_2d': 2, 'mask_3d': 3, 'partial': 4,
                                               'print': 5, 'replay': 6})
        self.assertEqual(list(aio.CLASSES), sorted(aio.CLASSES))       # ImageFolder class_to_idx equivalence

    def test_11_12_train_population_and_counts(self):
        records, datasets, ev = aio.read_train_population()
        self.assertEqual(len(records), 14467)
        self.assertEqual(ev['class_counts'], {'live': 5629, 'makeup': 759, 'mask_2d': 96, 'mask_3d': 1056,
                                              'partial': 1911, 'print': 2838, 'replay': 2178})
        self.assertEqual((ev['other_spoof_rows'], ev['val_rows_materialized'], ev['test_rows_materialized']), (0, 0, 0))
        self.assertEqual({r['split'] for r in records}, {'TRAIN'})
        self.assertEqual([(r['label'], r['sample_id']) for r in records],
                         sorted((r['label'], r['sample_id']) for r in records))
        self.assertEqual(ev['population_order_sha256'], self.ev['epoch']['population']['population_order_sha256'])
        self.assertTrue(set(ev['columns_never_read']) >= {'subject_id_global', 'attack_raw', 'video_id'})

    def test_13_main_8838_manifest_never_the_population(self):
        text = IO.read_text()
        self.assertEqual(text.count("MAIN_RELATION = 'manifests/difffas_bin_idfree_train_v1.parquet'"), 1)
        reader = fn_source(IO, 'read_train_population')
        self.assertNotIn('MAIN_RELATION', reader)
        self.assertNotIn('difffas_bin_idfree_train', reader)
        self.assertNotIn(aio.MAIN_RELATION, self.contract['bound_inputs_sha256'])
        calls = self.ev['epoch']['pyarrow_calls']
        self.assertEqual({c['path'] for c in calls}, {'manifests/split_v1.parquet'})

    def _synthetic_split(self, d, extra_rows, train_override=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows, i = [], 0
        for cls, n in aio.CLASS_COUNTS.items():
            for _ in range(n):
                rows.append({'sample_id': fake_id(i), 'dataset': aio.DATASETS[i % 3], 'attack_macro': cls,
                             'split': 'TRAIN', 'm2_status': 'COMPLETE', 'subject_id_global': 'S%d' % i})
                i += 1
        if train_override:
            rows[0].update(train_override)
        for split in extra_rows:
            rows.append({'sample_id': fake_id(i), 'dataset': 'casia_fasd', 'attack_macro': 'print', 'split': split,
                         'm2_status': 'COMPLETE', 'subject_id_global': 'S%d' % i})
            i += 1
        path = Path(d) / 'split.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_14_only_train_rows_become_dataset_items(self):
        cmap = aio.read_class_map()          # the frozen class map stays bound to the frozen split SHA256
        pin = mock.patch.object(aio, 'read_class_map', return_value=cmap)
        with tempfile.TemporaryDirectory() as d, pin:
            path = self._synthetic_split(d, ['VAL'] * 40 + ['TEST'] * 40)
            with mock.patch.object(aio, 'SPLIT_MANIFEST_SHA256', sha256_file(path)):
                records, _, ev = aio.read_train_population(path)
        self.assertEqual((len(records), {r['split'] for r in records}), (14467, {'TRAIN'}))
        self.assertEqual(ev['file_rows_total_from_footer'], 14467 + 80)
        with self.assertRaises(PreparationError):
            aio.AuxTrainDataset([dict(records[0], split='VAL')] + list(records[1:]), None, None)
        with tempfile.TemporaryDirectory() as d, pin:
            for override in ({'attack_macro': 'other_spoof'}, {'m2_status': 'FAILED'}, {'dataset': 'oulu'}):
                path = self._synthetic_split(d, [], override)
                with mock.patch.object(aio, 'SPLIT_MANIFEST_SHA256', sha256_file(path)), self.assertRaises(PreparationError):
                    aio.read_train_population(path)
                path.unlink()
        with tempfile.TemporaryDirectory() as d:
            bogus = Path(d) / 'split.parquet'
            bogus.write_bytes(b'not the frozen manifest')
            with self.assertRaises(PreparationError):
                aio.read_train_population(bogus)          # SHA256 before parse

    @unittest.skipUnless(HAS_PIL, 'Pillow unavailable')
    def test_15_16_canonical_path_and_firewall(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'synthetic_faces'   # not a benchmark root
            (root / 'casia_fasd').mkdir(parents=True)
            good, small, outside = fake_id(1), fake_id(2), fake_id(3)
            buf = io.BytesIO()
            Image.new('RGB', (256, 256), (10, 20, 30)).save(buf, format='PNG')
            (root / 'casia_fasd' / f'{good}.png').write_bytes(buf.getvalue())
            buf = io.BytesIO()
            Image.new('RGB', (128, 128)).save(buf, format='PNG')
            (root / 'casia_fasd' / f'{small}.png').write_bytes(buf.getvalue())
            target = Path(d) / 'elsewhere.png'
            target.write_bytes((root / 'casia_fasd' / f'{good}.png').read_bytes())
            link = fake_id(4)
            (root / 'casia_fasd' / f'{link}.png').symlink_to(target)
            reader = aio.CanonicalFaceReader(root, {good: 'casia_fasd', small: 'casia_fasd', link: 'casia_fasd'})
            self.assertEqual(reader.path(good), root.resolve() / 'casia_fasd' / f'{good}.png')   # <root>/<dataset>/<id>.png
            img = reader(good)
            self.assertEqual((img.mode, img.size), ('RGB', (256, 256)))
            for bad in (outside, '../' + good, good.upper(), 'x' * 64):
                with self.assertRaises(PreparationError):
                    reader(bad)
            with self.assertRaises(PreparationError):
                reader(small)                             # no implicit resize/recrop
            with self.assertRaises((PreparationError, OSError)):
                reader(link)                              # symlink refused
            with self.assertRaises(PreparationError):
                aio.CanonicalFaceReader(root, {good: '../escape'})
        self.assertIn("<faces_256_root>/<dataset>/<sample_id>.png", self.contract['canonical_face']['path'])
        probe = (ROOT / 'gpatbench/probe/data.py').read_text()
        self.assertIn('return self.root / r["dataset"] / f"{r[\'sample_id\']}.png"', probe)   # established M2 path

    # ---------------------------------------------------------------- 17-22 transform, sampler, SGD, CE, A7
    def test_17_18_exact_transform_no_augmentation(self):
        src = fn_source(ENGINE, 'build_transform')
        self.assertIn('transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], '
                      '[0.5, 0.5, 0.5])', src.replace('\n', ' ').replace('  ', ' '))
        text = ENGINE.read_text() + IO.read_text()
        for token in ('RandomHorizontalFlip', 'RandomCrop', 'ColorJitter', 'RandomResizedCrop', 'RandAugment',
                      'artifact_probe.yaml', 'frozen_probe_input'):
            self.assertNotIn(token, text)

    def test_19_no_balancing(self):
        text = ENGINE.read_text() + IO.read_text()
        for token in ('WeightedRandomSampler', 'class_weight', 'replacement=True', 'oversampl'):
            self.assertNotIn(token, text)
        self.assertEqual(self.ev['epoch']['components']['loader']['sampler'], 'RandomSampler(replacement=False, generator=None)')

    def test_20_exact_sgd(self):
        call = next(n for n in ast.walk(ast.parse(fn_source(ENGINE, 'build_optimizer')))
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.optim.SGD')
        self.assertEqual([ast.unparse(a) for a in call.args], ['model.parameters()'])
        self.assertEqual({k.arg: ast.unparse(k.value) for k in call.keywords},
                         {'lr': 'aio.LR', 'momentum': 'aio.MOMENTUM', 'weight_decay': 'aio.WEIGHT_DECAY'})
        self.assertEqual((aio.LR, aio.MOMENTUM, aio.WEIGHT_DECAY), (0.002, 0.9, 5e-3))
        opt = self.ev['epoch']['components']['optimizer']
        self.assertEqual((opt['lr'], opt['momentum'], opt['weight_decay'], opt['scheduler']), (0.002, 0.9, 0.005, 'NONE'))
        self.assertNotIn('lr_scheduler', ENGINE.read_text())

    def test_21_ce_on_fourth_output_and_step_order(self):
        step = fn_source(ENGINE, 'step', 'Trainer')
        order = ['inputs, labels = (inputs_cpu.cuda(), labels_cpu.cuda())', 'self.optimizer.zero_grad()',
                 '_, _, _, outputs = self.model(inputs)', 'loss = self.criterion(outputs, labels)', 'loss.backward()',
                 'self.optimizer.step()', 'value = loss.item()']
        positions = [step.index(s) for s in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('torch.nn.CrossEntropyLoss()', fn_source(ENGINE, 'build_criterion'))
        self.assertEqual(self.ev['epoch']['components']['model']['disconnected_parameters'], ['norm.bias', 'norm.weight'])

    def test_22_a7_precision_mandatory(self):
        self.assertIn('ep.apply_e07c_precision_policy(config)', fn_source(ENGINE, 'configure_precision'))
        text = ENGINE.read_text()
        for token in ('autocast(', 'GradScaler(', 'allow_tf32 = True', '.half()', 'bfloat16', 'use_deterministic_algorithms('):
            self.assertNotIn(token, text)
        for k in ('b1', 'b2', 'epoch'):
            p = self.ev[k]['precision']
            self.assertEqual((p['matmul_tf32'], p['cudnn_tf32'], p['cudnn_benchmark'], p['cudnn_deterministic'],
                              p['default_dtype'], p['deterministic_algorithms_forced'], p['grad_scaler'], p['amp']),
                             (False, False, False, True, 'torch.float32', False, False, False))
            self.assertEqual(self.ev[k]['launch_environment']['NVIDIA_TF32_OVERRIDE'], '0')
        with mock.patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '1'}):
            from methods.difffas import aux_runner
            with self.assertRaises(PreparationError):
                aux_runner.configure_precision(None, None)

    # ---------------------------------------------------------------- 23-25 seeds, CLI
    def test_23_qualification_seed_distinct(self):
        self.assertEqual((aio.QUALIFICATION_SEED, aio.SCIENTIFIC_SEED), (60605, 42))
        self.assertNotIn(aio.QUALIFICATION_SEED, aio.EXPERIMENT_SEEDS)
        with self.assertRaises(PreparationError):
            aio.validate_mode_seed(aio.QUALIFICATION, 42)
        with self.assertRaises(PreparationError):
            aio.validate_mode_seed(aio.SCIENTIFIC, 60605)
        seeding = fn_source(ENGINE, 'seed_process')
        calls = [ast.unparse(n) for n in ast.walk(ast.parse(seeding)) if isinstance(n, ast.Call) and
                 ast.unparse(n.func).endswith('apply_seed')]
        self.assertEqual(calls, ['seed_adapter.apply_seed(seed, cuda=True, auxiliary=True, config=config)'])
        self.assertIn('if mode == aio.SCIENTIFIC:\n        plan = seed_adapter.apply_seed(', seeding)   # scientific branch only
        for k in ('b1', 'b2', 'epoch'):
            s = self.ev[k]['seeding']
            self.assertEqual((s['seed'], s['role'], s['pythonhashseed']), (60605, 'QUALIFICATION_SEED_NOT_AUXILIARY_SEED', '60605'))

    def test_24_cli_accepts_only_seed42(self):
        cli = load_cli()
        env = {'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        for seed in (1337, 2026, 60605, 0):
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', io.StringIO()):
                cli.preconditions(cli.parse(['--seed', str(seed), '--execution-config', aio.EXEC_CONFIG]),
                                  environ=dict(env, PYTHONHASHSEED=str(seed)), dirty=False, branch='m6-baselines')
            self.assertEqual(cm.exception.code, 2)
        args = cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG])
        for kwargs in ({'dirty': True, 'branch': 'm6-baselines', 'environ': env},
                       {'dirty': False, 'branch': 'main', 'environ': env},
                       {'dirty': False, 'branch': 'm6-baselines', 'environ': dict(env, PYTHONHASHSEED='0')},
                       {'dirty': False, 'branch': 'm6-baselines', 'environ': dict(env, NVIDIA_TF32_OVERRIDE='1')},
                       {'dirty': False, 'branch': 'm6-baselines', 'environ': {k: v for k, v in env.items()
                                                                              if k != 'CUBLAS_WORKSPACE_CONFIG'}}):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                cli.preconditions(args, **kwargs)
        torch_before = 'torch' in sys.modules              # the live Torch tests may already have imported it
        plan = cli.preconditions(args, environ=env, dirty=False, branch='m6-baselines', check_interpreter=False)
        self.assertEqual((plan['auxiliary_encoder_training_seed'], plan['epochs']), (42, 200))
        self.assertEqual(plan['torch_imported'], torch_before)          # preconditions never import Torch
        self.assertTrue(plan['run_dir'].endswith('/runs/m6/E07c/aux_encoder/seed_42'))
        self.assertTrue(plan['checkpoint'].endswith('/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl'))

    def test_25_cli_refuses_training_overrides(self):
        cli = load_cli()
        for flag in ('--epochs=1', '--batch-size', '--workers', '--drop-last', '--lr', '--momentum', '--weight-decay',
                     '--scheduler', '--amp', '--fp16', '--bf16', '--tf32', '--microbatch', '--grad-accum',
                     '--activation-checkpointing', '--label-map', '--train-manifest', '--select-checkpoint', '--val',
                     '--test', '--resume', '--qualification-seed', '--weighted-sampler', '--augment'):
            with self.assertRaises(SystemExit) as cm, mock.patch('sys.stderr', io.StringIO()):
                cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG, flag])
            self.assertEqual(cm.exception.code, 2, flag)
        with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
            cli.parse(['--seed', '42', '--execution-config', aio.EXEC_CONFIG, '--unknown'])
        self.assertNotIn('import torch', CLI.read_text().split('def main')[0])

    # ---------------------------------------------------------------- 26-31 B256, repeatability, epoch, loss
    def test_26_b256_outcome_recorded(self):
        for k in ('b1', 'b2'):
            r = self.ev[k]
            self.assertEqual(r['status'], 'PASS')
            self.assertNotIn('oom_stage', r)
            self.assertEqual(r['first_batch']['input']['shape'], [256, 3, 256, 256])
            self.assertEqual(r['first_batch']['target']['shape'], [256])
            self.assertEqual(sum(r['first_batch']['class_histogram']), 256)
            self.assertTrue(r['first_batch']['input']['within_minus1_plus1'])
            for stage in ('after_model_construction', 'after_transfer', 'after_forward', 'after_backward',
                          'after_optimizer_step', 'peak_during_step'):
                self.assertIn(stage, r['memory'])
            self.assertLess(r['memory']['peak_during_step']['max_reserved_bytes'], r['cuda_mem_get_info_before']['total_bytes'])
            s = r['step']
            self.assertEqual((s['batch_size'], s['gradients_all_finite'], s['parameter_tensors_changed'],
                              s['parameter_tensors_unchanged']), (256, True, 110, ['norm.bias', 'norm.weight']))
            self.assertEqual(r['counters']['backward_calls'], 1)
            self.assertTrue(r['transform_check']['resize_identity_bytes'] and
                            r['transform_check']['loader_row_equals_transform'])

    def test_27_first_step_repeatability_bitwise(self):
        cmp = self.ev['b2']['comparison_vs_process_1']
        self.assertTrue(cmp['all_bitwise_equal'])
        self.assertEqual(cmp['tolerance'], 'NONE (bitwise)')
        d = self.ev['b2']['difference_vs_process_1']
        self.assertEqual((d['parameters_after']['bitwise_equal_tensors'], d['parameters_after']['tensors'],
                          d['parameters_after']['max_abs_diff']), (112, 112, 0.0))
        self.assertEqual((d['gradients']['bitwise_equal_tensors'], d['gradients']['tensors']), (110, 110))
        from methods.difffas.aux_runner_qualification import compare
        self.assertTrue(compare(self.ev['b1'], self.ev['b2'])['all_bitwise_equal'])
        first = self.ev['epoch']['first_step']
        self.assertEqual((first['loss_hex'], first['batch_sample_sha256'], first['parameters_after_aggregate_sha256']),
                         (self.ev['b1']['step']['loss_hex'], self.ev['b1']['step']['batch_sample_sha256'],
                          self.ev['b1']['step']['parameters_after_aggregate_sha256']))

    def test_28_29_30_full_epoch_56_steps_14336_131(self):
        e = self.ev['epoch']
        s = e['epoch1']
        self.assertEqual((s['optimizer_steps'], s['batch_sizes_distinct'], s['consumed_examples'], s['unique_consumed'],
                          s['dropped_examples'], s['global_step_end']), (56, [256], 14336, 14336, 131, 56))
        self.assertEqual(sum(s['dropped_class_histogram']), 131)
        c = e['counters']
        self.assertEqual((c['backward_calls'], c['sgd_step_calls'], c['zero_grad_calls'], e['optimizer_applications']),
                         (56, 56, 56, 56))
        self.assertEqual(len(e['epoch1_losses']), 56)
        self.assertTrue(all(float.fromhex(h) == v for h, v in zip(e['epoch1_losses_hex'], e['epoch1_losses'])))
        self.assertEqual(e['metrics_records'].count('epoch'), 56)
        self.assertTrue(e['logged_epoch_summary_equal'])

    def test_31_epoch_loss_denominator_14467(self):
        e = self.ev['epoch']
        s = e['epoch1']
        loss = aio.epoch_loss(e['epoch1_losses'])
        self.assertEqual(loss['denominator'], 14467)
        self.assertEqual(loss['running_loss'], s['running_loss'])
        self.assertEqual(loss['source_epoch_loss'], s['source_epoch_loss'])
        self.assertEqual(s['source_epoch_loss'], s['running_loss'] / 14467)
        self.assertNotEqual(s['source_epoch_loss'], s['running_loss'] / 14336)
        self.assertEqual(s['epoch_loss_denominator'], 14467)
        self.assertIn('running_loss / len(self.dataset)', fn_source(ENGINE, 'run_epoch', 'Trainer'))

    # ---------------------------------------------------------------- 32-35 checkpoint
    def test_32_whole_module_epoch_checkpoint(self):
        src = fn_source(ENGINE, 'checkpoint_event', 'Trainer')
        self.assertIn('save_whole_module(self.model, partial, config)', src)
        self.assertIn('os.replace(partial, final)', src)
        for token in ('state_dict', 'safetensors', 'jit', 'onnx'):
            self.assertNotIn(token, src)
        ck = self.ev['epoch']['checkpoint_event']
        self.assertEqual((ck['path'], ck['epoch'], ck['global_step'], ck['checkpoint_type'], ck['selected_for_final'],
                          ck['serialization_api']), ('checkpoints/encoder_final.pkl', 1, 56, 'periodic', False,
                                                     'torch.save(model, path)'))
        self.assertEqual(ck['identity']['fc'], {'in_features': 512, 'out_features': 7, 'bias': True})
        rl = self.ev['epoch']['checkpoint_reload']
        self.assertTrue(rl['state_dict_equal_in_memory'] and rl['sha256_verified_first'])
        self.assertEqual(self.ev['epoch']['checkpoint_file_before_cleanup']['sha256'], ck['sha256'])

    def test_33_qualification_checkpoint_outside_runs(self):
        e = self.ev['epoch']
        self.assertTrue(e['checkpoint_outside_runs'])
        self.assertIn('/qualification/m6d6e/E07c_aux/q60605-', e['checkpoint_path'])
        self.assertNotIn('/runs/', e['checkpoint_path'])
        with self.assertRaises(PreparationError):
            aio.run_root('/rt', aio.QUALIFICATION, 60605, 'not-hex')

    def test_34_qualification_checkpoint_cleanup(self):
        c = self.ev['epoch']['checkpoint_cleanup']
        self.assertEqual((c['removed'], c['checkpoints_dir_listing'], c['weight_files_in_run_dir']), (True, [], []))
        self.assertTrue(self.ev['b2']['probe_removed'])

    def test_35_scientific_checkpoint_not_created(self):
        for k in ('b1', 'b2', 'epoch'):
            self.assertFalse(self.ev[k]['scientific_checkpoint_created'])
            self.assertFalse(self.ev[k]['scientific_run_root_written'])
            self.assertEqual(self.ev[k]['benchmark_access_audit']['scientific_run_root_accesses'], 0)
        log = (ROOT / 'outputs/audit/M6D6E_E07C_AUX_RUNTIME_LOG.txt').read_text()
        self.assertIn('scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT encoder_final=ABSENT', log)
        self.assertNotIn('encoder_final=PRESENT', log)

    # ---------------------------------------------------------------- 36-40 access, no main training, no bank
    def test_36_37_38_zero_val_test_raw_access(self):
        for k in ('b1', 'b2', 'epoch'):
            a = self.ev[k]['benchmark_access_audit']
            self.assertEqual((a['VAL_image_reads'], a['TEST_image_reads'], a['non_train_face_accesses'],
                              a['raw_or_other_benchmark_data_accesses'], a['unauthorized_manifest_accesses'],
                              a['faces_enumeration_or_write'], a['foreign_weight_accesses'], a['denied_events']),
                             (0, 0, 0, 0, 0, 0, 0, 0))
            self.assertTrue(a['train_faces_all_in_population'])
            s = self.ev[k]['access_summary']
            self.assertEqual((s['TRAIN_rows_exposed_to_dataset'], s['VAL_rows_exposed_to_dataset'],
                              s['TEST_rows_exposed_to_dataset']), (14467, 0, 0))
            self.assertFalse(self.ev[k]['VAL_access'] or self.ev[k]['TEST_access'])
            self.assertTrue(self.ev[k]['authorized_TRAIN_access'])
        a = self.ev['epoch']['benchmark_access_audit']
        self.assertEqual((a['train_face_opens'], a['train_faces_distinct']), (14336, 14336))
        self.assertTrue(a['train_faces_opened_equal_consumed_set'])

    def test_39_no_main_difffas_training(self):
        text = ENGINE.read_text() + IO.read_text() + HARNESS.read_text() + CLI.read_text()
        for token in ('unet_autoenc', 'FAS_train', 'create_gaussian_diffusion', 'get_model_conf', 'main_runner_encoder'):
            self.assertNotIn(token, text)
        for k in ('b1', 'b2', 'epoch'):
            self.assertFalse(self.ev[k]['main_difffas_training'])

    def test_40_no_synthetic_bank(self):
        for k in ('b1', 'b2', 'epoch'):
            self.assertFalse(self.ev[k]['synthetic_bank'])
        self.assertEqual(self.ev['epoch']['run_files']['generation_log.jsonl']['bytes'], 0)

    def test_harness_firewall_denies_val_test_and_main_manifest(self):
        from methods.difffas.aux_runner_qualification import Firewall
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / 'rt'
            faces = rt / 'data/processed/faces_256'
            fd = os.open(Path(d) / 'log.tsv', os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            fw = Firewall(rt, faces, [Path(d) / 'build'], [Path(d) / 'build'], allow_faces=True, access_fd=fd)
            fw.samples = {'a' * 64: 'casia_fasd'}
            fw('open', (str(faces / 'casia_fasd' / ('a' * 64 + '.png')), 'rb', 0))
            fw('open', (str(ROOT / aio.SPLIT_MANIFEST), 'rb', 0))
            for path in (faces / 'casia_fasd' / ('b' * 64 + '.png'), ROOT / aio.MAIN_RELATION,
                         ROOT / 'manifests/pairs_train_v1.parquet', rt / 'runs/m6/E07c/x.pkl', rt / 'data/raw/x.jpg',
                         Path(d) / 'other.pkl'):
                with self.assertRaises(PermissionError):
                    fw('open', (str(path), 'rb', 0))
            with self.assertRaises(PermissionError):
                fw('os.listdir', (str(faces / 'casia_fasd'),))
            with self.assertRaises(PermissionError):
                fw('subprocess.Popen', ('x', ['curl', 'http://example'], None, None))
            fw('subprocess.Popen', (None, ['uname', '-p'], None, None))
            os.close(fd)
            from methods.difffas.aux_runner_qualification import access_audit
            audit = access_audit(Path(d) / 'log.tsv', {'a' * 64: 'casia_fasd'})
        self.assertEqual((audit['train_face_opens'], audit['non_train_face_accesses'], audit['denied_events']), (1, 1, 7))

    def test_resume_not_qualified_and_labels(self):
        self.assertEqual(self.contract['resume']['status'], 'AUX_RESUME_NOT_QUALIFIED')
        for token in ('resume_state', 'load_state_dict', 'torch.load('):
            self.assertNotIn(token, ENGINE.read_text())
        self.assertEqual(list(aio.QUALIFICATION_LABELS),
                         ['QUALIFICATION_ONLY', 'NOT_A_SCIENTIFIC_CHECKPOINT', 'NOT_ELIGIBLE_FOR_BANK',
                          'NOT_ELIGIBLE_FOR_DOWNSTREAM', 'NOT_ELIGIBLE_FOR_REPORTING'])
        self.assertEqual(self.ev['epoch']['labels'], list(aio.QUALIFICATION_LABELS))

    def test_run_id_distinct_from_main(self):
        from methods.common.runlog import compute_run_id
        cfg = self.contract['bound_inputs_sha256']['configs/methods/e07c_difffas_bin_idfree.yaml']
        self.assertNotEqual(aio.run_id(42, cfg, AUTHORITY), compute_run_id('E07c', 42, cfg, AUTHORITY))
        self.assertEqual(aio.run_root('/rt', aio.SCIENTIFIC, 42), Path('/rt/runs/m6/E07c/aux_encoder/seed_42'))

    def test_step_record_null_val_with_reason_no_test_fields(self):
        r = aio.step_record(mode=aio.QUALIFICATION, seed=60605, epoch=1, global_step=1, iteration=0,
                            learning_rate=0.002, loss=1.0, running_loss=256.0, batch_size=256,
                            batch_sample_sha256='0' * 64, batch_class_histogram=[1] * 7, wall_clock_seconds=0.1,
                            gpu_memory_bytes=1)
        self.assertIsNone(r['val_losses'])
        self.assertTrue(r['missing_field_reasons']['val_losses'].startswith('NOT_USED'))
        self.assertFalse(any('test' in k.lower() for k in r))


@unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; the GPU evidence is validated statically')
class TestM6D6eLiveTorch(unittest.TestCase):
    """CPU-only live checks with synthetic tensors/images; no benchmark data, CUDA not required."""

    def setUp(self):
        import torch
        self.torch = torch

    @unittest.skipUnless(HAS_PIL, 'Pillow unavailable')
    def test_live_transform_exact(self):
        import numpy as np
        import torchvision.transforms as transforms
        from PIL import Image
        from methods.difffas import aux_runner
        t = aux_runner.build_transform(transforms)
        aux_runner.transform_evidence(transforms, t)
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), 'RGB')
        self.assertEqual(t.transforms[0](img).tobytes(), img.tobytes())
        x = t(img)
        manual = (self.torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255) - 0.5) / 0.5
        self.assertTrue(self.torch.equal(x, manual))
        self.assertEqual((tuple(x.shape), x.dtype), ((3, 256, 256), self.torch.float32))

    def test_live_loader_plan_56x256_drop_131(self):
        from methods.difffas import aux_runner
        ds = type('DS', (), {'__len__': lambda s: 14467, '__getitem__': lambda s, i: i})()
        loader, ev = aux_runner.build_loader(self.torch, ds)
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(0)
            batches = list(loader.batch_sampler)
        flat = [i for b in batches for i in b]
        self.assertEqual(([len(b) for b in batches], len(set(flat)), 14467 - len(set(flat))), ([256] * 56, 14336, 131))
        self.assertEqual((ev['generator'], ev['worker_init_fn'], ev['pin_memory']), (None, None, False))

    def test_live_model_optimizer_criterion(self):
        from methods.difffas import aux_runner
        from methods.difffas.encoder import encoder_model
        with self.torch.random.fork_rng(devices=[]), encoder_model() as model:
            opt, oev = aux_runner.build_optimizer(self.torch, model)
            crit, cev = aux_runner.build_criterion(self.torch)
            self.assertEqual((model.fc.in_features, model.fc.out_features), (512, 7))
            self.assertEqual(oev['parameter_tensors'], len(list(model.parameters())))
            self.assertEqual(type(opt), self.torch.optim.SGD)

    def test_live_step_refuses_wrong_batch_and_classifies_oom(self):
        from methods.difffas import aux_runner
        torch = self.torch
        records = [{'index': i, 'sample_id': fake_id(i), 'label': i % 7, 'split': 'TRAIN'} for i in range(14467)]
        ds = type('DS', (), {'records': tuple(records), '__len__': lambda s: 14467})()
        model = mock.MagicMock()
        model.named_parameters.return_value = []
        opt = mock.MagicMock()
        tr = aux_runner.Trainer(torch, model, opt, mock.MagicMock(), None, ds, aio.QUALIFICATION, 60605)
        small = [torch.zeros(4, 3, 256, 256), torch.tensor([r['label'] for r in records[:4]]), torch.arange(4)]
        with self.assertRaises(aux_runner.TrainingStop):
            tr.step(small, 1, 0)                          # no microbatch: exact 256 only

        class OOMInputs:
            shape, dtype = (256, 3, 256, 256), torch.float32

            def cuda(self):
                raise torch.OutOfMemoryError('synthetic OOM')
        batch = [OOMInputs(), torch.tensor([r['label'] for r in records[:256]]), torch.arange(256)]
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(aux_runner.AuxMemoryBlocked) as cm:
                tr.step(batch, 1, 0)
        self.assertEqual(cm.exception.stage, 'host_to_device')
        self.assertEqual((tr.global_step, tr.backward_calls, opt.step.call_count, opt.zero_grad.call_count), (0, 0, 0, 0))

    def test_live_whole_module_epoch_checkpoint_roundtrip(self):
        from methods.difffas import aux_runner
        from methods.difffas.aux_checkpoint import load_verified_whole_module
        from methods.difffas.encoder import encoder_model
        torch = self.torch
        events = []

        class Ctx:
            mode = aio.QUALIFICATION

            def record_checkpoint(self, **k):
                events.append(('record', k))

            def log_event(self, name, payload):
                events.append((name, payload))
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            ctx = Ctx()
            ctx.run_dir = Path(d)
            (ctx.run_dir / 'checkpoints').mkdir()
            with torch.random.fork_rng(devices=[]), encoder_model() as model:
                ds = type('DS', (), {'records': (), '__len__': lambda s: 14467})()
                tr = aux_runner.Trainer(torch, model, torch.optim.SGD(model.parameters(), lr=0.002), None, None, ds,
                                        aio.QUALIFICATION, 60605)
                tr.completed_epoch, tr.global_step = 1, 56
                meta = tr.checkpoint_event(1, ctx, None)
                live = {k: v.clone() for k, v in model.state_dict().items()}
            path = ctx.run_dir / 'checkpoints' / 'encoder_final.pkl'
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['encoder_final.pkl'])   # no .partial
            self.assertEqual((meta['sha256'], meta['checkpoint_type'], meta['selected_for_final']),
                             (sha256_file(path), 'periodic', False))
            with load_verified_whole_module(path, meta['sha256'], device='cpu') as loaded:
                self.assertTrue(all(torch.equal(loaded.state_dict()[k], v) for k, v in live.items()))
            with self.assertRaises(PreparationError):
                with load_verified_whole_module(path, '0' * 64, device='cpu'):
                    pass
        self.assertEqual([e[0] for e in events], ['record', 'e07c_aux_checkpoint_event'])


if __name__ == '__main__':
    unittest.main()
