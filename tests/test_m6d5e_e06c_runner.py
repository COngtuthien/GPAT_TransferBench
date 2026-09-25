"""M6D5e E06c production runner: contract, TRAIN relation, reader, loader, logging, checkpoint, resume, CLI.

Static tests need no Torch. Engine tests (skipped where Torch is absent) drive the real
methods/dsdg/runner.py Trainer on CPU with tiny toy networks that expose exactly the
production 17 + 17 + 21 = 55 optimizer-owned tensors, the pinned misc.util and synthetic
in-memory canonical faces. Real-TRAIN CUDA evidence comes only from
methods/dsdg/runner_qualification.py processes: the retained initial M6D5e run (seed 60505,
procedural firewall deviation disclosed) and the M6D5e-r clean requalification (seed 60506).
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from methods.common import runlog
from methods.common.config import load_method_config, sha256_file
from methods.common.learned import PreparationError, checkpoint_metadata, checkpoint_plan, seed_torch_worker
from methods.common.runlog import RunDirectoryError, compute_run_id
from methods.dsdg import DSDGAdapter
from methods.dsdg import microbatch_execution_v2 as v2
from methods.dsdg import runner as engine
from methods.dsdg import runner_io as rio
from methods.dsdg import runner_qualification as rq
from methods.dsdg import training_graph as tg

ROOT = Path(__file__).resolve().parents[1]
DSDG = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'
AUDIT = ROOT / 'outputs/audit'
REPORT = AUDIT / 'M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION.md'
CLEAN_REPORT = AUDIT / 'M6D5E_R_E06C_CLEAN_REQUALIFICATION.md'
EVIDENCE = {k: AUDIT / v for k, v in rq.OUTPUTS.items()}                      # M6D5e-r clean requalification
INITIAL_EVIDENCE = {k: AUDIT / v.replace('M6D5E_R_', 'M6D5E_') for k, v in rq.OUTPUTS.items()}   # retained initial run
QSEED = rio.QUALIFICATION_SEED
NEW_SOURCES = ('methods/dsdg/runner_io.py', 'methods/dsdg/runner.py', 'methods/dsdg/runner_qualification.py',
               'tools/run_e06c.py')
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # laptop: static tests only
    torch = None
try:
    import cv2
except ImportError:
    cv2 = None
needs_torch = unittest.skipIf(torch is None, 'Torch absent; engine tests run in gpat-m6-e06c')
needs_cv2 = unittest.skipIf(cv2 is None, 'OpenCV absent')


def run_cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_e06c', ROOT / 'tools/run_e06c.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)
    return sha256_file(path)


def synthetic_rows(n, dataset='casia_fasd', split='TRAIN'):
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    return [{'pair_id': f'PTR{i:06d}', 'source_spoof_id': h(f's{i}'), 'target_live_id': h(f't{i % 3}'),
             'dataset': dataset, 'split': split, 'seed': 20260814, 'source_sha256': h(f'a{i}'),
             'target_sha256': h(f'b{i}'), 'source_subject': f'subj{i}', 'target_subject': f'subj{i + 1}',
             'attack_macro': 'print'} for i in range(n)]


class TestE06cRunnerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DSDGAdapter()
        cls.contract = rio.load_contract()

    def test_contract_binds_frozen_inputs_and_overlays(self):
        c = self.contract
        self.assertEqual((c['fidelity_class'], c['classification'], c['execution_mode'], c['deviation']),
                         ('CONTROLLED_ADAPTATION', 'CONTROLLED_EXECUTION_ADAPTATION',
                          'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2', 'DEV-020'))
        for w in ('native', 'faithful', 'official reproduction'):
            self.assertIn(w, c['forbidden_wording'])
        ids = rio.identities(c)
        self.assertEqual(ids['m6d5d_v2_overlay_sha256'], sha256_file(ROOT / v2.RESOLUTION_PATH))
        self.assertEqual(ids['config_sha256'], load_method_config('E06c')['_runtime']['config_sha256'])
        self.assertEqual(ids['lightcnn_sha256'], 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964')
        self.assertEqual(c['modes']['QUALIFICATION']['qualification_seed'], 60506)
        self.assertEqual(c['modes']['QUALIFICATION']['initial_qualification_seed_retained_not_reused'], 60505)
        r = c['corrective_requalification']
        self.assertEqual(r['initial_run_status'], 'M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION')
        self.assertTrue(r['initial_run']['val_metadata_read_during_development'] and
                        r['initial_run']['test_metadata_read_during_development'])
        self.assertFalse(r['initial_run']['val_image_bytes_accessed'] or r['initial_run']['test_image_bytes_accessed'])
        self.assertIs(c['execution']['official_visualization_block']['executed_by_runner'], True)
        self.assertEqual(c['modes']['SCIENTIFIC']['experiment_seeds'], [42, 1337, 2026])
        self.assertIs(c['modes']['SCIENTIFIC']['launched_in_m6d5e'], False)

    def test_production_scientific_seeds_exact(self):
        for s in (42, 1337, 2026):
            rio.validate_mode_seed(rio.SCIENTIFIC, s)
        for s in (60505, 60506, 0, 43, 60504, True, '42', 42.0):
            with self.assertRaises(PreparationError):
                rio.validate_mode_seed(rio.SCIENTIFIC, s)
        self.assertEqual(rio.SCIENTIFIC_SEEDS, tuple(load_method_config('E06c')['seeds']['experiment_seeds']))

    def test_qualification_seed_and_root_segregated(self):
        rio.validate_mode_seed(rio.QUALIFICATION, 60506)
        for s in (42, 1337, 2026, 60504, 60505):       # 60505: the retained initial run is never reused
            with self.assertRaises(PreparationError):
                rio.validate_mode_seed(rio.QUALIFICATION, s)
        rt = Path('/rt')
        sci = rio.run_root(rt, rio.SCIENTIFIC, 42)
        q = rio.run_root(rt, rio.QUALIFICATION, 60506, 'a' * 16)
        self.assertEqual(sci, rt / 'runs/m6/E06c/seed_42')
        self.assertEqual(q, rt / 'qualification/m6d5e/E06c' / ('q60506-' + 'a' * 16))
        self.assertFalse(q.is_relative_to(rt / 'runs'))
        with self.assertRaises(PreparationError):
            rio.run_root(rt, rio.QUALIFICATION, 60506, '../../runs')
        self.assertEqual(rq.BUILD_PARTS, ('builds', 'e06c_dsdg_m6d5e_r'))
        self.assertTrue(all(v.startswith('M6D5E_R_E06C_') for v in rq.OUTPUTS.values()))

    def test_deterministic_run_id_formula(self):
        cfg = load_method_config('E06c')['_runtime']['config_sha256']
        expected = hashlib.sha256(f'E06c|60506|{cfg}|{"c" * 40}'.encode()).hexdigest()[:16]
        self.assertEqual(compute_run_id('E06c', 60506, cfg, 'c' * 40), expected)
        self.assertEqual(rio.qualification_run_id(expected), 'q60506-' + expected)
        # same HEAD and config: seed 60505 would regenerate the initial run's root, so it cannot be reused
        self.assertNotEqual(compute_run_id('E06c', 60505, cfg, 'c' * 40), expected)

    def test_manifest_sha_rows_and_identity_firewall(self):
        records, datasets, ev = rio.read_train_relation(self.adapter)
        self.assertEqual((ev['rows'], ev['splits'], ev['sha256']), (8838, ['TRAIN'], rio.TRAIN_RELATION_SHA256))
        self.assertEqual(ev['rows_by_dataset'], {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118})
        self.assertEqual(ev['subject_columns_present_in_file'], ['source_subject', 'target_subject'])
        self.assertEqual(ev['subject_columns_read'], [])
        self.assertTrue(all(set(r) <= set(rio.ALLOWLIST) and 'source_subject' not in r for r in records))
        self.assertEqual(len(datasets), ev['unique_sample_ids'])

    def test_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'x.parquet'
            h = write_parquet(bad, synthetic_rows(5))
            with self.assertRaises(PreparationError):            # SHA differs
                rio.read_train_relation(self.adapter, bad)
            with mock.patch.object(rio, 'TRAIN_RELATION_SHA256', h):
                with self.assertRaises(PreparationError):        # rows != 8838
                    rio.read_train_relation(self.adapter, bad)
                with mock.patch.object(rio, 'TRAIN_ROWS', 5):
                    recs, _, ev = rio.read_train_relation(self.adapter, bad)
                    self.assertEqual(ev['subject_columns_read'], [])
            for rows in (synthetic_rows(5, split='VAL'), synthetic_rows(5, dataset='oulu_npu'),
                         [{k: v for k, v in r.items() if k != 'target_sha256'} for r in synthetic_rows(5)]):
                h = write_parquet(bad, rows)
                with mock.patch.object(rio, 'TRAIN_RELATION_SHA256', h), mock.patch.object(rio, 'TRAIN_ROWS', 5):
                    with self.assertRaises(PreparationError):
                        rio.read_train_relation(self.adapter, bad)

    def test_subject_columns_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.parquet', Path(tmp) / 'b.parquet'
            rows = synthetic_rows(6)
            ha = write_parquet(a, rows)
            hb = write_parquet(b, [dict(r, source_subject=None, target_subject='X') for r in rows])
            out = []
            for p, h in ((a, ha), (b, hb)):
                with mock.patch.object(rio, 'TRAIN_RELATION_SHA256', h), mock.patch.object(rio, 'TRAIN_ROWS', 6):
                    out.append(rio.read_train_relation(self.adapter, p)[:2])
            self.assertEqual(out[0], out[1])

    def test_reader_interface_receives_sample_id_only(self):
        self.assertEqual(list(inspect.signature(rio.CanonicalFaceReader.__call__).parameters), ['self', 'sample_id'])
        code = inspect.getsource(rio.CanonicalFaceReader) + inspect.getsource(rio._cv2_decode)
        for word in ('cv2.resize', '.crop(', 'transforms.', 'ColorJitter', 'Flip', 'normalize(', 'cvtColor'):
            self.assertNotIn(word, code)

    @needs_cv2
    def test_canonical_reader_rgb256_uint8_and_confinement(self):
        sid, other = 'a' * 64, 'b' * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'faces_256'
            (root / 'casia_fasd').mkdir(parents=True)
            rgb = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(root / 'casia_fasd' / f'{sid}.png'), np.ascontiguousarray(rgb[:, :, ::-1]))
            reader = rio.CanonicalFaceReader(root, {sid: 'casia_fasd', other: 'casia_fasd'})
            got = reader(sid)
            self.assertEqual((got.dtype, got.shape), (np.uint8, (256, 256, 3)))
            self.assertTrue(np.array_equal(got, rgb))                      # exact bytes: no crop/resize/jitter
            for bad in ('../' + sid, sid.upper(), 'c' * 64, sid + '.png'):
                with self.assertRaises(PreparationError):
                    reader(bad)
            (root / 'casia_fasd' / f'{other}.png').symlink_to(root / 'casia_fasd' / f'{sid}.png')
            with self.assertRaises(PreparationError):
                reader(other)
            small = 'd' * 64
            cv2.imwrite(str(root / 'casia_fasd' / f'{small}.png'), np.zeros((255, 256, 3), np.uint8))
            gray = 'e' * 64
            cv2.imwrite(str(root / 'casia_fasd' / f'{gray}.png'), np.zeros((256, 256), np.uint8))
            r2 = rio.CanonicalFaceReader(root, {small: 'casia_fasd', gray: 'casia_fasd'})
            for s in (small, gray):
                with self.assertRaises(PreparationError):
                    r2(s)
            (Path(tmp) / 'outside').mkdir()
            (root / 'msu_mfsd').symlink_to(Path(tmp) / 'outside')
            with self.assertRaises(PreparationError):
                rio.CanonicalFaceReader(root, {sid: 'msu_mfsd'})(sid)
            with self.assertRaises(PreparationError):
                rio.CanonicalFaceReader(root, {sid: 'oulu_npu'})

    def test_dataset_adapter_official_keys(self):
        records = [dict(r, **{}) for r in synthetic_rows(2)]
        ds = rio.IndexedPairDataset(self.adapter, records, lambda sid: np.full((256, 256, 3), 255, np.uint8))
        item = ds[1]
        self.assertEqual(sorted(item), ['0', '1', 'index', 'type'])
        self.assertEqual((item['type'], item['index']), (0, 1))
        self.assertEqual((item['0'].dtype, item['0'].shape, float(item['0'].max())), (np.float32, (3, 256, 256), 1.0))
        self.assertIsInstance(ds, rio.IDFreePairDataset)

    def test_loader_contract_constants(self):
        self.assertEqual(rio.LOADER, {'batch_size': 240, 'shuffle': True, 'num_workers': 8, 'pin_memory': True,
                                      'drop_last': False})
        plan = v2.epoch_batch_plan(8838)
        self.assertEqual((plan['batch_sizes'], plan['optimizer_steps_per_epoch']), ([240] * 36 + [198], 37))
        self.assertEqual([s.stop - s.start for s in v2.chunk_slices_v2(198)], [20] * 9 + [18])
        with self.assertRaises(PreparationError):
            v2.epoch_batch_plan(8838, drop_last=True)

    def test_v2_execution_binding_not_generic_adapter_gate(self):
        overlays = engine.verify_overlays()
        self.assertEqual(overlays['v2']['execution_mode'], 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2')
        self.assertEqual(overlays['v1']['physical_batch_240'], 'OOM_RETAINED')
        src = inspect.getsource(engine)
        self.assertIn('v2.run_global_batch_v2(', src)
        self.assertIn('v2.draw_epsilon(', src)
        self.assertNotIn('validate_batch(', src)
        with self.assertRaises(PreparationError):          # the historical generic refusal is untouched
            self.adapter.validate_batch(physical_batch_size=20, gradient_accumulation_steps=12)

    def test_checkpoint_cadence_epoch1_exception_and_selection(self):
        epochs = rio.checkpoint_epochs()
        self.assertEqual(epochs, [1] + list(range(10, 201, 10)))
        self.assertEqual(len(epochs), 21)
        self.assertEqual(rio.official_basename('netG_', 200), 'netG_model_epoch_200_iter_0.pth')
        self.assertEqual(rio.checkpoint_kind('netG', 200), ('selected', True))
        self.assertEqual(rio.checkpoint_kind('netE_nir', 200), ('terminal', False))
        self.assertEqual(rio.checkpoint_kind('netG', 1), ('periodic', False))
        cfg = load_method_config('E06c')
        ok = checkpoint_metadata(cfg, 42, epoch=200, global_step=7400, path='checkpoints/netG_model_epoch_200_iter_0.pth',
                                 file_size_bytes=1, sha256='a' * 64, checkpoint_type='selected', selected_for_final=True)
        self.assertEqual(ok['selection_reason'], 'OFFICIAL_GENERATOR_EPOCH_200')
        with self.assertRaises(PreparationError):
            checkpoint_metadata(cfg, 42, epoch=190, global_step=1, path='netG_model_epoch_190_iter_0.pth',
                                file_size_bytes=1, sha256='a' * 64, checkpoint_type='selected', selected_for_final=True)
        for split in ('VAL', 'TEST', 'TRAIN'):                # no data split selects anything
            with self.assertRaises(PreparationError):
                checkpoint_plan(cfg, 42, selection_split=split)

    def test_no_test_or_val_code_path(self):
        cadence = "VISUALIZATION_CADENCE_KEY, VISUALIZATION_EVERY = 'test_epoch', 10"
        for rel in NEW_SOURCES:
            src = (ROOT / rel).read_text()
            for word in ('val_pairs', 'split_v1', "'TEST'", '"TEST"', 'selection_split='):
                self.assertNotIn(word, src, rel + ': ' + word)
            # the pinned visualization-cadence argparse name is bound exactly once (it is not a data split)
            self.assertEqual(src.count('test_epoch'), src.count(cadence), rel)
        self.assertEqual((ROOT / 'methods/dsdg/runner_io.py').read_text().count(cadence), 1)
        record = rio.step_record(epoch=1, global_step=1, iteration=0, learning_rate=2e-4,
                                 losses=dict.fromkeys(rio.LOSS_KEYS, 0.0), total_loss=0.0, wall_clock_seconds=1.0,
                                 gpu_memory_bytes=1, batch_size=240, chunk_sizes=[20] * 12, batch_pair_sha256='a' * 64,
                                 epsilon_sha256={}, mode=rio.QUALIFICATION, seed=QSEED)
        self.assertNotIn('test', json.dumps(sorted(record)).lower())
        self.assertIsNone(record['val_losses'])
        self.assertTrue(record['missing_field_reasons']['val_metrics'])

    def test_resume_explicit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'checkpoints').mkdir()
            rio.init_resume_index(run, run_id='a' * 16, mode=rio.SCIENTIFIC, seed=42)
            p = run / 'checkpoints' / rio.resume_state_name(10)
            p.write_bytes(b'state')
            rio.record_resume_state(run, path='checkpoints/' + p.name, epoch=10, global_step=370,
                                    resolved_config_sha256='b' * 64)
            path, entry = rio.resume_entry(run, 'checkpoints/' + p.name)
            self.assertEqual((path, entry['completed_epoch'], entry['scientific_checkpoint']), (p, 10, False))
            (run / 'checkpoints' / 'other.pth').write_bytes(b'x')
            for bad in ('checkpoints/other.pth', str(run / 'x.pth'), 'checkpoints'):
                with self.assertRaises(PreparationError):
                    rio.resume_entry(run, bad)
            p.write_bytes(b'tampered')
            with self.assertRaises(PreparationError):
                rio.resume_entry(run, 'checkpoints/' + p.name)
            self.assertNotIn('glob(', inspect.getsource(rio.resume_entry))

    def test_metrics_reconciliation_refuses_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = Path(tmp) / 'metrics.jsonl'
            rows = [{'record_type': 'epoch', 'epoch': 1 + (i >= 37), 'global_step': i + 1} for i in range(40)]
            m.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(RunDirectoryError):
                rio.reconcile_metrics(m, completed_epoch=1, global_step=37)
            m.write_text(''.join(json.dumps(r) + '\n' for r in rows[:37]) + '{"partial"')
            with self.assertRaises(RunDirectoryError):
                rio.reconcile_metrics(m, completed_epoch=1, global_step=37)
            m.write_text(''.join(json.dumps(r) + '\n' for r in rows[:37]))
            self.assertTrue(rio.reconcile_metrics(m, completed_epoch=1, global_step=37)['agreement'])

    def test_production_cli_refuses_overrides_and_seeds(self):
        cli = run_cli()
        base = ['--seed', '42', '--execution-config', rio.EXEC_CONFIG]
        for extra in (['--epochs', '5'], ['--lr=1e-3'], ['--batch-size', '20'], ['--microbatch', '10'],
                      ['--drop-last'], ['--amp'], ['--tf32'], ['--fp16'], ['--checkpoint-selection', 'VAL'],
                      ['--qualification'], ['--resume-latest'], ['--bogus']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                cli.parse(base + extra)
            self.assertEqual(cm.exception.code, 2, extra)
        env = {'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        for seed in ('60505', '7'):
            args = cli.parse(['--seed', seed, '--execution-config', rio.EXEC_CONFIG])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.preconditions(args, environ=dict(env, PYTHONHASHSEED=seed), dirty=False)
        args = cli.parse(base)
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            cli.preconditions(args, environ=env, dirty=True)
        self.assertIn('dirty', err.getvalue())
        for bad in ({'PYTHONHASHSEED': '1'}, {'NVIDIA_TF32_OVERRIDE': '1'}):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.preconditions(args, environ=dict(env, **bad), dirty=False)
        src = (ROOT / 'tools/run_e06c.py').read_text()
        self.assertNotIn("add_argument('--epochs", src)
        self.assertIn('rio.ALL_EPOCHS', inspect.getsource(engine.run_scientific))

    def test_visualization_cadence(self):
        self.assertEqual(rio.visualization_every(load_method_config('E06c')), 10)
        self.assertEqual([e for e in range(1, 201) if rio.is_visualization_epoch(e)], rio.checkpoint_epochs())
        self.assertEqual(rio.visualization_basename(1, 'fake_live'), 'Epoch_001_fake_live.png')
        self.assertEqual(rio.VISUALIZATION_NOISE_SHAPE, (240, 128))
        src = inspect.getsource(engine.run_scientific)
        self.assertLess(src.index('trainer.visualize('), src.index('trainer.checkpoint_event('))

    def test_firewall_access_audit_log_and_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp)
            faces = rt / 'data/processed/faces_256'
            log = rt / 'access.tsv'
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND)
            fw = rq.Firewall(rt, faces, [rt / 'builds/x'], {'a' * 64: 'casia_fasd'}, True, fd)
            ok = str(faces / 'casia_fasd' / ('a' * 64 + '.png'))
            fw('open', (ok, 'rb', 0))
            fw('open', (str(ROOT / rio.TRAIN_RELATION), 'rb', 0))
            fw('open', (str(rt / 'code.py'), 'r', 0))                           # not benchmark: not logged
            for path in (str(ROOT / 'manifests/split_v1.parquet'), str(ROOT / 'manifests/val_pairs_v1.parquet'),
                         str(ROOT / 'manifests/pairs_test_v1.parquet'), str(faces / 'msu_mfsd' / ('b' * 64 + '.png'))):
                with self.assertRaises(PermissionError):
                    fw('open', (path, 'rb', 0))
            os.close(fd)
            audit = rq.access_audit(log, {'a' * 64: 'casia_fasd'})
            c = audit['counts_by_category']
            self.assertEqual((c['TRAIN_FACE'], c['TRAIN_RELATION'], c['SPLIT_METADATA_VAL_TEST'], c['VAL_METADATA'],
                              c['TEST_METADATA'], c['NON_TRAIN_FACE']), (1, 1, 1, 1, 1, 1))
            self.assertEqual((audit['VAL_metadata_accesses'], audit['TEST_metadata_accesses'], audit['denied_events'],
                              audit['VAL_image_accesses'], audit['events_logged']), (2, 2, 4, 1, 6))

    def test_qualification_firewall_denies_scientific_root_val_test_and_enumeration(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp)
            faces = rt / 'data/processed/faces_256'
            fw = rq.Firewall(rt, faces, [rt / 'builds/x', rt / 'qualification/m6d5e'], {'a' * 64: 'casia_fasd'}, True)
            ok = str(faces / 'casia_fasd' / ('a' * 64 + '.png'))
            self.assertIsNone(fw.reason('open', ok, (ok, 'rb', 0)))
            self.assertIsNone(fw.reason('open', str(ROOT / rio.TRAIN_RELATION), ('', 'rb', 0)))
            denied = [('open', str(rt / 'runs/m6/E06c/seed_42/metrics.jsonl'), 'rb'),
                      ('open', str(ROOT / 'manifests/val_pairs_v1.parquet'), 'rb'),
                      ('open', str(ROOT / 'manifests/split_v1.parquet'), 'rb'),
                      ('open', str(faces / 'casia_fasd' / ('b' * 64 + '.png')), 'rb'),
                      ('os.listdir', str(faces / 'casia_fasd'), None),
                      ('open', str(rt / 'elsewhere.txt'), 'w'),
                      ('open', str(rt / 'x.pth'), 'rb')]
            for event, path, mode in denied:
                self.assertIsNotNone(fw.reason(event, path, (path, mode, 0)), path)
            self.assertIsNone(fw.reason('os.mkdir', str(rt / 'qualification/m6d5e'), ('',)))

    def test_resume_comparison_requires_bitwise(self):
        step = {k[-1] if len(k) == 1 else k[0]: None for k in rq.BITWISE_FIELDS}
        step['probe'] = {k[1]: 1 for k in rq.BITWISE_FIELDS if len(k) == 2}
        ref = {'step': step}
        diff = {'gradients': {'tensors': 55, 'bitwise_equal_tensors': 55},
                'parameters_after': {'tensors': 55, 'bitwise_equal_tensors': 55}}
        fresh = {'step': json.loads(json.dumps(step)), 'difference_vs_reference_probe': diff}
        self.assertTrue(rq.compare(ref, fresh)['all_bitwise_equal'])
        fresh['step']['probe']['gradient_sha256'] = 2
        self.assertFalse(rq.compare(ref, fresh)['all_bitwise_equal'])


# ================================================================= engine tests (Torch, CPU toy networks)
if torch is not None:
    class Scaled(torch.nn.Module):
        """Linear + (n - 2) scalar multipliers: exactly n parameter tensors, all on the loss path."""
        def __init__(self, i, o, n, unused=False):
            super().__init__()
            self.lin = torch.nn.Linear(i, o)
            self.scales = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(())) for _ in range(n - 2)])
            self.unused = unused

        def forward(self, x):
            y = self.lin(x)
            for s in (self.scales[:-1] if self.unused else self.scales):
                y = y * s
            return y

    class ToyEncoder(torch.nn.Module):
        def __init__(self, parts, h=128, **kw):
            super().__init__()
            self.parts, self.h, self.net = parts, h, Scaled(48, parts * h, 17, **kw)

        def forward(self, x):
            return tuple(torch.split(self.net(F.adaptive_avg_pool2d(x, 4).flatten(1)), self.h, dim=-1))

    class ToyDecoder(torch.nn.Module):
        def __init__(self, h=128, **kw):
            super().__init__()
            self.net = Scaled(3 * h, 6 * 64, 21, **kw)

        def forward(self, z):
            return torch.sigmoid(F.interpolate(self.net(z).view(z.size(0), 6, 8, 8), size=(256, 256)))

    class ToyIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 256)
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.fc(F.adaptive_avg_pool2d(x, 8).flatten(1))


def toy_models(seed, unused=False):
    with torch.random.fork_rng():   # stands in for the immutable LightCNN asset: identical in every process
        torch.manual_seed(7)
        ip = ToyIP()
    torch.manual_seed(seed)
    return {'netE_nir': ToyEncoder(4), 'netE_vis': ToyEncoder(2), 'netG': ToyDecoder(unused=unused),
            'netCls': torch.nn.Linear(128, 1), 'netIP': ip}


def environment():
    env = runlog.environment_metadata()
    env.update(framework='torch', framework_version='toy', pytorch_version='toy', tensorflow_version=None,
               dependency_fingerprint='toy', environment_lock_path='toy', gpu_note='CPU toy test')
    return env, {k: 'CPU toy test' for k, v in env.items() if v is None}


@needs_torch
class TestE06cRunnerEngine(unittest.TestCase):
    N = 5

    @classmethod
    def setUpClass(cls):
        from methods.common.upstream import upstream_modules
        cls.ctxm = upstream_modules(DSDG, ('misc.util',), ('misc',))
        cls.util = cls.ctxm.__enter__()['misc.util']
        cls.adapter = DSDGAdapter()
        cls.ids = rio.identities(rio.load_contract())
        rng = np.random.default_rng(3)
        cls.records = tuple(cls.adapter.project_pair(r) for r in synthetic_rows(cls.N))
        ids = {r[k] for r in cls.records for k in ('source_spoof_id', 'target_live_id')}
        cls.faces = {s: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for s in sorted(ids)}

    @classmethod
    def tearDownClass(cls):
        cls.ctxm.__exit__(None, None, None)

    def trainer(self, seed, mode=rio.SCIENTIFIC, unused=False):
        models = toy_models(seed, unused)
        opt = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        dataset = rio.IndexedPairDataset(self.adapter, self.records, self.faces.__getitem__)
        cfg = load_method_config('E06c')
        loader, gen, evidence = engine.build_loader(torch, dataset, mode, 42 if mode == rio.SCIENTIFIC else QSEED, cfg)
        t = engine.Trainer(torch, F, self.util, models, opt, loader, gen, self.records, mode,
                           42 if mode == rio.SCIENTIFIC else QSEED, tg.frozen_lambdas(cfg), device='cpu')
        return t, evidence

    def context(self, root, resume=False, mode=rio.SCIENTIFIC):
        env, reasons = environment()
        return engine.E06cRunContext(mode=mode, seed=42 if mode == rio.SCIENTIFIC else QSEED, runtime_root=root,
                                     environment=env, missing_environment_reasons=reasons, identities=self.ids,
                                     resume=resume)

    def test_loader_configuration_determinism_and_worker_seeding(self):
        t, ev = self.trainer(1)
        self.assertEqual((ev['batch_size'], ev['shuffle'], ev['num_workers'], ev['pin_memory'], ev['drop_last']),
                         (240, True, 8, True, False))
        self.assertIs(t.loader.worker_init_fn, seed_torch_worker)
        self.assertIsInstance(t.loader.sampler, torch.utils.data.RandomSampler)
        orders = []
        for _ in range(2):
            loader, _, _ = engine.build_loader(torch, rio.IndexOnlyDataset(8838), rio.QUALIFICATION, QSEED,
                                               load_method_config('E06c'))
            batches = [b.tolist() for b in loader]
            self.assertEqual([len(b) for b in batches], [240] * 36 + [198])
            flat = [i for b in batches for i in b]
            self.assertEqual(sorted(flat), list(range(8838)))
            orders.append(flat)
        self.assertEqual(orders[0], orders[1])
        with self.assertRaises(PreparationError):
            engine.build_loader(torch, rio.IndexOnlyDataset(3), rio.SCIENTIFIC, QSEED, load_method_config('E06c'))

    def test_tail_b198_one_step_and_gradient_gate(self):
        t, _ = self.trainer(2)
        spoof, live = tg.synthetic_pair(torch, 198, 'cpu')
        records = tuple(dict(self.records[0], pair_id=f'P{i}') for i in range(198))
        t.records = records
        batch = {'0': spoof, '1': live, 'type': torch.zeros(198, dtype=torch.long), 'index': torch.arange(198)}
        calls = []
        orig = torch.Tensor.backward
        with mock.patch.object(torch.Tensor, 'backward', lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1]):
            r = t.global_step_run(batch, 1, 0)
        self.assertEqual((r['batch_size'], r['chunk_sizes'], len(calls)), (198, [20] * 9 + [18], 10))
        self.assertEqual((t.optimizer_applications, t.global_step), (1, 1))
        self.assertEqual(r['owned_gradients'], {'tensors': 55, 'non_none': 55, 'finite': 55})
        self.assertEqual(r['netCls_grad_nonzero'], 0)
        self.assertTrue(all(p.grad is None for p in t.models['netIP'].parameters()))
        opt_ids = {id(p) for g in t.optimizer.param_groups for p in g['params']}
        self.assertFalse(opt_ids & {id(p) for n in ('netCls', 'netIP') for p in t.models[n].parameters()})
        self.assertEqual(len(opt_ids), 55)
        bad, _ = self.trainer(2, unused=True)
        bad.records = records
        with self.assertRaises(engine.TrainingStop):
            bad.global_step_run(batch, 1, 0)                    # 54/55 owned gradients -> STOP
        self.assertEqual(bad.optimizer_applications, 0)

    def test_epoch_logging_checkpoint_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            t, _ = self.trainer(3)
            ctx = self.context(tmp).open()
            summary, _ = t.run_epoch(1, ctx)
            self.assertEqual((summary['batch_sizes'], summary['rows']), ([self.N], self.N))
            written, entry = t.checkpoint_event(1, ctx)
            ctx.close(completion_status='interrupted', summary={'failure_reason': 'test stop'})
            run = ctx.run_dir
            # ---- logging contract
            self.assertTrue((run / 'resolved_config.yaml').is_file())
            manifest = json.loads((run / 'run_manifest.json').read_text())
            for f in runlog.load_logging_contract()['run_identity']['required_fields']:
                self.assertIn(f, manifest)
            self.assertEqual(manifest['resolved_config_sha256'], sha256_file(run / 'resolved_config.yaml'))
            self.assertEqual((manifest['execution_mode'], manifest['physical_batch_240'], manifest['drop_last']),
                             ('GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2', 'OOM_RETAINED', False))
            self.assertEqual(manifest['run_id'], compute_run_id('E06c', 42, self.ids['config_sha256'],
                                                                manifest['git_commit']))
            self.assertTrue(manifest['run_uuid'])
            resolved = (run / 'resolved_config.yaml').read_text()
            for s in ('e06c_m6d5c_memory_execution_resolution', 'e06c_m6d5d_tail_batch_execution_resolution',
                      'e06c_m6d5e_production_runner_contract', 'OOM_RETAINED', 'max_microbatch: 20'):
                self.assertIn(s, resolved)
            rows = rio.trajectory(run / 'metrics.jsonl')
            steps = [r for r in rows if r['record_type'] == 'epoch']
            self.assertEqual(len(steps), t.global_step)             # one record per global optimizer step
            r = steps[0]
            for k in runlog.load_logging_contract()['trajectory']['required_fields']:
                self.assertIn(k, r)
            self.assertEqual(sorted(r['train_losses']), sorted(rio.LOSS_KEYS + ('total_loss',)))
            for k in ('train_metrics', 'val_losses', 'val_metrics', 'gpu_memory_bytes'):
                self.assertIsNone(r[k])
                self.assertTrue(r['missing_field_reasons'][k])
            self.assertGreater(r['wall_clock_seconds'], 0)
            self.assertNotIn('test', json.dumps(rows).lower().replace('latest', ''))
            # ---- checkpoint index holds actual file hashes; sidecar indexed separately
            index = json.loads((run / 'checkpoint_index.json').read_text())
            self.assertEqual(len(index['checkpoints']), 3)
            for c in index['checkpoints']:
                self.assertEqual(c['sha256'], sha256_file(run / c['path']))
                self.assertEqual((c['checkpoint_type'], c['selected_for_final'], c['epoch']), ('periodic', False, 1))
            self.assertEqual(sorted(Path(c['path']).name for c in index['checkpoints']),
                             ['netE_live_model_epoch_1_iter_0.pth', 'netE_spoof_model_epoch_1_iter_0.pth',
                              'netG_model_epoch_1_iter_0.pth'])
            blob = torch.load(run / 'checkpoints/netG_model_epoch_1_iter_0.pth', weights_only=False)
            self.assertEqual(sorted(blob), ['epoch', 'model'])
            ridx = json.loads((run / 'resume_state_index.json').read_text())
            self.assertEqual(ridx['entries'][0]['sha256'], sha256_file(run / entry['path']))
            self.assertFalse(ridx['scientific_checkpoint'] or ridx['selection_candidate'])
            state = torch.load(run / entry['path'], weights_only=True)
            self.assertEqual(sorted(state['models']), ['netCls', 'netE_nir', 'netE_vis', 'netG'])
            self.assertEqual(sorted(state['rng']), ['numpy', 'python', 'torch_cpu', 'torch_cuda'])
            self.assertIn('loader_generator', state)
            self.assertNotIn('netIP', state['models'])
            # ---- uninterrupted reference: one epoch-2 step in memory
            t.set_modes()
            ref = t.global_step_run(next(iter(t.loader)), 2, 0)
            ref_params = {n: p.detach().clone() for n, p in t.owned}
            # ---- fresh trainer: explicit resume, one step, identical result; metrics appended
            before = (run / 'metrics.jsonl').read_bytes()
            t2, _ = self.trainer(99)
            ctx2 = self.context(tmp, resume=True).open()
            path, e = rio.resume_entry(run, entry['path'])
            restored = t2.load_resume_state(path, ctx2)
            self.assertEqual((restored['completed_epoch'], restored['global_step']), (1, 1))
            rio.reconcile_metrics(ctx2.path('metrics'), completed_epoch=1, global_step=1)
            t2.set_modes()
            res = t2.global_step_run(next(iter(t2.loader)), 2, 0)
            ctx2.log_epoch(rio.step_record(epoch=2, global_step=res['global_step'], iteration=0, learning_rate=2e-4,
                                           losses=res['losses'], total_loss=res['total_loss'], wall_clock_seconds=1.0,
                                           gpu_memory_bytes=None, batch_size=res['batch_size'],
                                           chunk_sizes=res['chunk_sizes'], batch_pair_sha256=res['batch_pair_sha256'],
                                           epsilon_sha256=res['epsilon_sha256'], mode=rio.SCIENTIFIC, seed=42))
            ctx2.close(completion_status='interrupted', summary={'failure_reason': 'test stop'})
            for k in ('batch_pair_sha256', 'epsilon_sha256', 'losses', 'total_loss', 'chunk_sizes', 'global_step'):
                self.assertEqual(ref[k], res[k], k)
            self.assertTrue(all(torch.equal(ref_params[n], p) for n, p in t2.owned))
            after = (run / 'metrics.jsonl').read_bytes()
            self.assertTrue(after.startswith(before) and len(after) > len(before))
            events = [x.get('event') for x in rio.trajectory(run / 'metrics.jsonl')]
            self.assertIn('resume_reconciliation', events)
            # ---- a fresh open without resume refuses the non-empty directory
            with self.assertRaises(RunDirectoryError):
                self.context(tmp).open()

    def test_upstream_visualization_rng_semantics(self):
        """Epoch-1 block: exactly two CPU normal_ [240,128] draws (noise, noise_s), six grids, no parameter change."""
        with tempfile.TemporaryDirectory() as tmp:
            t, _ = self.trainer(4)
            ctx = self.context(tmp).open()
            t.run_epoch(1, ctx)
            self.assertEqual(sorted(t.visual['grids']), ['img_live', 'img_spoof', 'rec_live', 'rec_spoof'])
            self.assertEqual(tuple(t.visual['grids']['rec_spoof'].shape), (self.N, 3, 128, 128))
            params = {n: p.detach().clone() for n, p in t.owned}
            before = torch.get_rng_state()
            g = torch.Generator()
            g.set_state(before)
            noise = torch.zeros(240, 128).normal_(0, 1, generator=g)
            noise_s = torch.zeros(240, 128).normal_(0, 1, generator=g)
            ev = t.visualize(1, ctx)
            self.assertTrue(torch.equal(g.get_state(), torch.get_rng_state()))      # CPU RNG advanced as the source
            self.assertEqual((ev['noise_sha256'], ev['noise_s_sha256']), (engine.tsha(noise), engine.tsha(noise_s)))
            self.assertTrue(all(torch.equal(params[n], p) for n, p in t.owned))
            self.assertIsNone(t.visual)
            names = [f['path'] for f in ev['files']]
            self.assertEqual(names, [f'diagnostics/visualization/Epoch_001_{n}.png' for n in rio.VISUALIZATION_FILES])
            self.assertEqual([f['images'] for f in ev['files']], [self.N] * 4 + [240, 240])
            for f in ev['files']:
                self.assertEqual(f['sha256'], sha256_file(Path(ctx.run_dir) / f['path']))
            with self.assertRaises(PreparationError):                               # once per epoch; no overwrite
                t.visualize(1, ctx)
            _, entry = t.checkpoint_event(1, ctx)
            state = torch.load(Path(ctx.run_dir) / entry['path'], weights_only=True)
            self.assertTrue(torch.equal(state['rng']['torch_cpu'], g.get_state()))  # sidecar = post-visualization
            ctx.close(completion_status='interrupted', summary={'failure_reason': 'test stop'})
            events = [x.get('event') for x in rio.trajectory(Path(ctx.run_dir) / 'metrics.jsonl')]
            self.assertLess(events.index('e06c_visualization_diagnostic'), events.index('e06c_checkpoint_event'))
        t2, _ = self.trainer(4)
        with self.assertRaises(PreparationError):                                   # not a visualization epoch
            t2.visualize(2, None)

    def test_qualification_context_segregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.context(tmp, mode=rio.QUALIFICATION)
            self.assertTrue(ctx.run_dir.is_relative_to(Path(tmp) / 'qualification/m6d5e/E06c'))
            ctx.open()
            ctx.close(completion_status='interrupted', summary={'failure_reason': 'test stop'})
            m = json.loads((ctx.run_dir / 'run_manifest.json').read_text())
            self.assertIsNone(m['experiment_seed'])
            self.assertEqual((m['qualification_seed'], m['qualification_only'], m['scientific_run']), (QSEED, True, False))
            self.assertEqual(m['labels'], list(rio.QUALIFICATION_LABELS))
            self.assertFalse((Path(tmp) / 'runs').exists())
            with self.assertRaises(RunDirectoryError):
                self.context(tmp, mode=rio.QUALIFICATION).open()      # existing qualification root: STOP
            with self.assertRaises(PreparationError):
                ctx.record_checkpoint(path='checkpoints/netG_model_epoch_200_iter_0.pth', epoch=200, global_step=1,
                                      file_size_bytes=1, sha256='a' * 64, checkpoint_type='selected',
                                      selected_for_final=True, selection_reason='x')


# ================================================================= retained GPU evidence
@unittest.skipUnless(all(p.is_file() for p in INITIAL_EVIDENCE.values()), 'initial M6D5e evidence absent')
class TestE06cInitialRunEvidence(unittest.TestCase):
    """The initial M6D5e run is retained as historical evidence with its procedural deviation disclosed."""
    @classmethod
    def setUpClass(cls):
        cls.ev = {k: json.loads(p.read_text()) for k, p in INITIAL_EVIDENCE.items()}

    def test_initial_run_facts_retained(self):
        e = self.ev['train']
        self.assertEqual((e['status'], e['qualification_seed']), ('PASS', 60505))
        self.assertEqual((e['epoch1']['rows'], e['epoch1']['optimizer_steps']), (8838, 37))
        self.assertIn('/qualification/m6d5e/E06c/q60505-', e['run_dir'])
        self.assertTrue(self.ev['resume']['comparison_vs_reference']['all_bitwise_equal'])

    @unittest.skipUnless(REPORT.is_file(), 'report not yet written')
    def test_initial_deviation_disclosed(self):
        text = REPORT.read_text()
        for w in ('M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION', 'NO VAL/TEST IMAGE ACCESS', 'NO VAL/TEST TRAINING USE',
                  'NO VAL/TEST SELECTION USE', 'NO TEST METRIC USE', 'M6D5E_R_E06C_CLEAN_REQUALIFICATION.md'):
            self.assertIn(w, text)


@unittest.skipUnless(all(p.is_file() for p in EVIDENCE.values()), 'M6D5e-r clean evidence not yet retained')
class TestE06cCleanRequalificationEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = {k: json.loads(p.read_text()) for k, p in EVIDENCE.items()}

    def test_real_train_epoch1(self):
        e = self.ev['train']
        self.assertEqual(e['status'], 'PASS')
        self.assertEqual((e['epoch1']['rows'], e['epoch1']['unique_pair_ids'], e['epoch1']['optimizer_steps']),
                         (8838, 8838, 37))
        self.assertEqual(e['epoch1']['batch_sizes'], [240] * 36 + [198])
        self.assertEqual((e['optimizer_applications_epoch1'], e['backward_calls_epoch1']), (37, 442))
        self.assertIsNone(e['experiment_seed'])
        self.assertEqual(e['qualification_seed'], 60506)
        self.assertFalse(e['VAL_access'] or e['TEST_access'] or e['scientific_training'] or e['synthetic_bank'])
        self.assertEqual(e['firewall']['denied'], [])
        self.assertIn('/qualification/m6d5e/E06c/q60506-', e['run_dir'])

    def test_benchmark_access_audit_train_only(self):
        for k in ('loader', 'train', 'resume'):
            a = self.ev[k]['benchmark_access_audit']
            self.assertEqual((a['VAL_metadata_accesses'], a['TEST_metadata_accesses'], a['VAL_image_accesses'],
                              a['TEST_image_accesses'], a['split_manifest_accesses'], a['denied_events']), (0,) * 6, k)
            self.assertGreater(a['counts_by_category']['TRAIN_RELATION'], 0)
        self.assertEqual(self.ev['train']['benchmark_access_audit']['train_faces_opened_distinct'], 12668)

    def test_visualization_rng_semantics(self):
        v = self.ev['train']['visualization']
        self.assertEqual(v['status'], 'UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED')
        self.assertTrue(all(v['proof'].values()))
        self.assertTrue(v['resume_sidecar_torch_cpu_rng_equals_post_visualization'])

    def test_loader_determinism(self):
        e = self.ev['loader']
        self.assertEqual(e['status'], 'PASS')
        self.assertTrue(all(e['repeat_equal'].values()))
        self.assertEqual(e['epoch1_pair_order_sha256'], self.ev['train']['epoch1']['epoch_pair_order_sha256'])

    def test_resume_bitwise(self):
        f = self.ev['resume']
        self.assertEqual(f['status'], 'PASS')
        self.assertTrue(f['comparison_vs_reference']['all_bitwise_equal'])
        self.assertEqual(f['step']['batch_pair_sha256'], self.ev['loader']['epoch2_first_batch_pair_sha256'])
        self.assertTrue(f['metrics_append_only'])

    @unittest.skipUnless(CLEAN_REPORT.is_file(), 'report not yet written')
    def test_report_wording(self):
        text = CLEAN_REPORT.read_text()
        for w in ('M6D5e PASS_AFTER_CLEAN_REQUALIFICATION', 'E06c_PRODUCTION_RUNNER_QUALIFIED',
                  'E06c_TRAIN_ONLY_FIREWALL_QUALIFIED', 'E06c_UPSTREAM_VISUALIZATION_RNG_SEMANTICS_QUALIFIED',
                  'CONTROLLED_ADAPTATION PRESERVED', 'E06c_SCIENTIFIC_FULL_TRAINING_NOT_YET_EXECUTED', 'QUALIFICATION_ONLY',
                  'NOT_SCIENTIFIC_CHECKPOINT', 'REAL TRAIN DATA ACCESSED FOR QUALIFICATION',
                  'NO VAL ACCESS DURING CLEAN REQUALIFICATION', 'NO TEST ACCESS DURING CLEAN REQUALIFICATION',
                  'M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION'):
            self.assertIn(w, text)


if __name__ == '__main__':
    unittest.main()
