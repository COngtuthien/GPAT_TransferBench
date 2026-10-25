"""E03 STDN production runner: static checks of the owner rulings, feed pack, schedule, cadence and fail-closed CLI.

Stdlib + numpy + pyarrow only (laptop .venv, no TensorFlow). The TF1 graph itself is qualified on the GPU host by
methods/stdn/production_qualification.py (tests/test_e03_production_qualification_evidence.py checks that record).
"""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.stdn import feed_pack as fp  # noqa: E402
from methods.stdn import production_io as pio  # noqa: E402

RUNNER = (ROOT / 'methods/stdn/production_runner.py').read_text()
RUNNER_AST = ast.parse(RUNNER)


def cli():
    spec = importlib.util.spec_from_file_location('run_e03', ROOT / 'tools' / 'run_e03.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def function_source(name):
    node = next(n for n in ast.walk(RUNNER_AST) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(RUNNER, node)


class OwnerRulings(unittest.TestCase):
    def test_01_frozen_identities_and_rulings(self):
        cfg = pio.verify_contract()
        self.assertEqual(cfg['fidelity_class'], 'FAITHFUL_OFFICIAL')          # historical field stays frozen
        self.assertEqual(pio.FIDELITY, 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_'
                                       'RUNTIME_COMPATIBILITY')
        for rel, digest in (pio.RUNTIME_ADDENDUM, pio.RULING_ADDENDUM, pio.RULING_YAML, pio.LOCK):
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)
        self.assertEqual(pio.verify_source()['files'], 25)
        self.assertEqual(pio.verify_source_checkout()['verification'], 'PASS')

    def test_02_val_seed_derivation_d2(self):
        self.assertEqual({s: pio.val_parser_seed(s) for s in (42, 1337, 2026, 60903)},
                         {42: 1355875380, 1337: 1050417158, 2026: 1242808125, 60903: 2050698081})
        for s in (42, 1337, 2026, 60903):
            u32 = int.from_bytes(hashlib.sha256(('GPAT_TRANSFERBENCH|E03|VAL_DATA|%d' % s).encode('utf-8'))
                                 .digest()[:4], 'big')
            self.assertEqual(pio.val_parser_seed(s), 1 + u32 % 2147483647)
            self.assertTrue(1 <= pio.val_parser_seed(s) <= 2 ** 31 - 1)

    def test_03_split_streams_are_independent_objects(self):
        plan = pio.rng_plan(42)
        self.assertEqual((plan['TRAIN']['parser_seed'], plan['TRAIN']['shuffle_seed']), (42, 42))
        self.assertEqual((plan['VAL']['parser_seed'], plan['VAL']['shuffle_seed']), (1355875380, 1355875380))
        parser = function_source('SplitParser')
        self.assertIn('np.random.RandomState(seed)', parser)
        self.assertNotIn('np.random.rand(', parser)             # no process-global NumPy draw
        self.assertNotIn('random.randint', parser)              # no dummy frame-choice draw
        self.assertNotIn('random.random(', parser)
        self.assertIn("parser_seed']", function_source('Controller'))

    def test_04_track_a_relations(self):
        import pyarrow.parquet as pq
        for split, (path, digest, rows) in pio.PAIR_MANIFESTS.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
            self.assertEqual(pq.read_metadata(ROOT / path).num_rows, rows)
        side = function_source('SplitParser')
        self.assertLess(side.index("row['target_live_id']"), side.index("row['source_spoof_id']"))  # live then spoof
        self.assertIn("generate_offset_map(lm_sp, lm_li)", side)

    def test_05_r3_figures_from_the_same_run(self):
        body = function_source('run_plan')
        self.assertEqual(body.count('sess.run('), 2)           # one TRAIN fetch and one VAL fetch per step
        self.assertIn('fig, idx, flips, delta = r[-4], r[-3], r[-2], r[-1]', body)
        self.assertIn('write_figure(ctx.run_dir, pio.train_figure_name(e, step), fig)', body)
        self.assertIn('fig, idx, flips = r[3], r[4], r[5]', body)
        self.assertNotIn('sess.run', function_source('write_figure'))


class Schedule(unittest.TestCase):
    def test_06_official_one_run_fetch(self):
        body = function_source('run_plan')
        self.assertIn('ctl.losses + ([ctl.g_op, ctl.d_op, ctl.fig] if gd else [ctl.g_op, ctl.fig])', body)
        self.assertNotIn('sess.run(ctl.g_op', RUNNER)
        self.assertNotIn('sess.run(ctl.d_op', RUNNER)
        self.assertNotIn('control_dependencies', RUNNER)       # no new G-before-D ordering

    def test_07_shared_global_step(self):
        self.assertEqual([pio.is_gd_step(s) for s in range(4)], [True, False, True, False])
        self.assertEqual(pio.applications_after(0, 0), {'G': 1, 'D': 1, 'global_step': 2})
        self.assertEqual(pio.applications_after(0, 1), {'G': 2, 'D': 1, 'global_step': 3})
        self.assertEqual(pio.applications_after(0, 1999), {'G': 2000, 'D': 1000, 'global_step': 3000})
        self.assertEqual(pio.applications_after(49, 1999), {'G': 100000, 'D': 50000, 'global_step': 150000})
        self.assertEqual((pio.GLOBAL_STEP_PER_EPOCH, pio.FINAL_GLOBAL_STEP), (3000, 150000))

    def test_08_learning_rate_on_shared_step(self):
        self.assertEqual(pio.decay_boundaries(), [20000, 40000, 60000, 80000, 100000, 120000, 140000])
        self.assertEqual(pio.expected_learning_rate(19999), 6e-5)
        self.assertAlmostEqual(pio.expected_learning_rate(20000), 6e-5 * 0.9)
        self.assertAlmostEqual(pio.expected_learning_rate(149999), 6e-5 * 0.9 ** 7)
        self.assertAlmostEqual(20000 / pio.GLOBAL_STEP_PER_EPOCH, 6.6667, places=3)   # not 10 epochs

    def test_09_optimizer_hyperparameters(self):
        self.assertEqual((pio.LEARNING_RATE, pio.DECAY_RATE, pio.DECAY_STEPS, pio.BETA1, pio.BETA2, pio.EPSILON,
                          pio.WEIGHT_EMA, pio.LOSS_EMA), (6e-5, 0.9, 20000, 0.9, 0.999, 1e-8, 0.9999, 0.9))
        self.assertEqual((pio.MAX_EPOCH, pio.STEPS_PER_EPOCH, pio.VAL_STEPS, pio.BATCH_SIZE, pio.G_D_RATIO),
                         (50, 2000, 500, 2, 2))

    def test_10_checkpoint_before_val_and_final_val(self):
        plan = pio.epoch_plan()
        self.assertEqual(len(plan), 50 * 3 + 1)
        for e in range(50):
            self.assertEqual([p[0] for p in plan[3 * e:3 * e + 3]], ['train', 'checkpoint', 'val'])
            self.assertEqual(plan[3 * e + 1][2], 'ckpt-%d' % (e + 1))
        self.assertEqual(plan[-3:], [('checkpoint', 49, 'ckpt-50'), ('val', 49, 500), ('complete', 49, None)])
        body = function_source('run_plan')
        self.assertLess(body.index('write_checkpoint('), body.index('for step in range(n_val)'))
        sci = function_source('run_scientific')
        self.assertLess(sci.index('run_plan('), sci.index("completion_status='completed'"))

    def test_11_ckpt_cadence_and_selection(self):
        self.assertEqual(pio.checkpoint_kind(1), ('official', False, 'Official cadence; not final selection'))
        self.assertEqual(pio.checkpoint_kind(49)[1], False)
        self.assertEqual(pio.checkpoint_kind(50), ('selected', True, 'OFFICIAL_LATEST_FINAL_CKPT_50'))
        from methods.common.config import load_method_config
        from methods.common.learned import PreparationError, checkpoint_metadata
        cfg = load_method_config('E03')
        for suffix in pio.CHECKPOINT_SUFFIXES:
            m = checkpoint_metadata(cfg, 42, epoch=50, global_step=150000, path='checkpoints/ckpt-50' + suffix,
                                    file_size_bytes=1, sha256='0' * 64, checkpoint_type='selected',
                                    selected_for_final=True)
            self.assertEqual(m['selection_reason'], 'OFFICIAL_LATEST_FINAL_CKPT_50')
        with self.assertRaises(PreparationError):
            checkpoint_metadata(cfg, 42, epoch=49, global_step=147000, path='checkpoints/ckpt-49.index',
                                file_size_bytes=1, sha256='0' * 64, checkpoint_type='selected', selected_for_final=True)
        import methods.stdn.production_runner as pr
        self.assertEqual(pr.SCIENTIFIC_PLAN, tuple((2000, 'checkpoint', 500) for _ in range(50)))

    def test_12_figure_cadence(self):
        self.assertEqual([s for s in range(2000) if pio.is_train_figure_step(s)], list(range(0, 2000, 200)))
        self.assertEqual([s for s in range(500) if pio.is_val_figure_step(s)], list(range(0, 500, 50)))
        self.assertEqual(pio.train_figure_name(0, 0), 'Epoch-1-1.png')
        self.assertEqual(pio.val_figure_name(49, 450), 'Epoch-50-Val-451.png')

    def test_13_feed_pipeline_structure(self):
        it = function_source('make_iterator')
        order = [it.index(x) for x in ('from_tensor_slices', '.shuffle(n, seed=shuffle_seed, reshuffle_each_iteration='
                                       'True).repeat(-1)', 'serialized_tf_map(dataset, parse_fn)',
                                       '.batch(batch_size=pio.BATCH_SIZE).prefetch(1)')]
        self.assertEqual(order, sorted(order))
        self.assertIn('tf.image.random_brightness(li, 0.25), tf.image.random_brightness(sp, 0.25)', it)
        self.assertIn('if brightness:', it)
        ctl = function_source('Controller')
        self.assertIn("make_iterator(tf, self.parsers['TRAIN'], n['TRAIN'], plan['TRAIN']['shuffle_seed'], True)", ctl)
        self.assertIn("make_iterator(tf, self.parsers['VAL'], n['VAL'], plan['VAL']['shuffle_seed'], False)", ctl)
        self.assertNotIn('make_initializable_iterator', RUNNER)  # one-shot iterators persist across epochs

    def test_14_landmark_flip_rule(self):
        from methods.common.config import load_method_config
        from methods.stdn.landmarks import LandmarkAdapter
        import methods.stdn.production_runner as pr
        adapter = LandmarkAdapter(load_method_config('E03'))
        self.assertEqual(adapter.permutation.tolist(), pr.LANDMARK_PERMUTATION)
        px = np.random.RandomState(0).uniform(0, 255, (68, 2)).astype(np.float32)
        plain = adapter.from_cache({'landmarks_px256': px})
        np.testing.assert_array_equal(plain, px / 256)
        flipped = adapter.from_cache({'landmarks_px256': px}, flip=True)
        expect = (px / 256).copy()
        expect[:, 0] = 1 - expect[:, 0]
        np.testing.assert_array_equal(flipped, expect[adapter.permutation])
        with self.assertRaises(ValueError):
            adapter.from_cache({'landmarks_norm': px})


class FeedPackChecks(unittest.TestCase):
    def _pack(self, tmp, mutate=None):
        sid = lambda c: c * 64  # noqa: E731
        samples = [[sid('a'), 'TRAIN', 'msu_mfsd', 0, 'f' * 64, 0, 's', 'b' * 64, 'n' * 64, None],
                   [sid('b'), 'TRAIN', 'msu_mfsd', 1, 'f' * 64, 1, 's', 'b' * 64, 'n' * 64, None],
                   [sid('c'), 'VAL', 'msu_mfsd', 0, 'f' * 64, 2, 's', 'b' * 64, 'n' * 64, None],
                   [sid('d'), 'VAL', 'msu_mfsd', 1, 'f' * 64, 3, 's', 'b' * 64, 'n' * 64, None]]
        lm = np.arange(4 * 68 * 2, dtype='<f4').reshape(4, 68, 2)
        for i, r in enumerate(samples):
            r[9] = fp.sha256_bytes(fp.landmark_bytes(lm[i]))
        if mutate:
            mutate(samples)
        rows = {'TRAIN': [['P1', sid('a'), sid('b'), 'msu_mfsd']], 'VAL': [['V1', sid('c'), sid('d'), 'msu_mfsd']]}
        payload = {'pairs_train.json': fp.dumps({'split': 'TRAIN', 'columns': list(fp.PAIR_COLUMNS), 'rows': rows['TRAIN']}),
                   'pairs_val.json': fp.dumps({'split': 'VAL', 'columns': list(fp.PAIR_COLUMNS), 'rows': rows['VAL']}),
                   'samples.json': fp.dumps({'columns': list(fp.SAMPLE_COLUMNS), 'rows': samples}),
                   'landmarks_px256.f32le': lm.tobytes()}
        m = {'schema': fp.SCHEMA, 'version': fp.VERSION, 'test_rows': 0,
             'counts': {'pairs': {'TRAIN': 1, 'VAL': 1}, 'unique_samples': 4, 'landmark_rows': 4,
                        'samples_by_split': {'TRAIN': 2, 'VAL': 2}},
             'pair_id_order_sha256': {s: fp.sha256_bytes('\n'.join(r[0] for r in v).encode()) for s, v in rows.items()},
             'payloads': {n: {'sha256': fp.sha256_bytes(b), 'bytes': len(b)} for n, b in payload.items()}}
        root = Path(tmp) / 'pack'
        root.mkdir()
        for n, b in payload.items():
            (root / n).write_bytes(b)
        (root / fp.MANIFEST).write_bytes(fp.dumps(m))
        return root

    def test_15_pack_verifies_and_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._pack(tmp)
            pack = fp.load(root)
            self.assertEqual(pack.pair_rows('TRAIN')[0]['target_live_id'], 'a' * 64)
            with self.assertRaises(fp.PackError):
                fp.load(root, '0' * 64)                                    # pinned manifest mismatch
            (root / 'landmarks_px256.f32le').write_bytes(b'\0' * 4 * 68 * 2 * 4)
            with self.assertRaises(fp.PackError):
                fp.load(root)                                              # payload integrity
        with tempfile.TemporaryDirectory() as tmp:
            def to_test(rows):
                rows[2][1] = 'TEST'
            with self.assertRaises(fp.PackError):
                fp.load(self._pack(tmp, to_test))                          # TEST sample refused
        with tempfile.TemporaryDirectory() as tmp:
            def swap_role(rows):
                rows[0][3] = 1
            with self.assertRaises(fp.PackError):
                fp.load(self._pack(tmp, swap_role))                        # target_live must be live

    def test_16_pack_membership_excludes_test(self):
        import pyarrow.parquet as pq
        split = {r['sample_id']: r['split'] for r in pq.read_table(ROOT / pio.SPLIT_MANIFEST[0],
                                                                    columns=['sample_id', 'split']).to_pylist()}
        members = {}
        for s, (path, _, _) in pio.PAIR_MANIFESTS.items():
            for r in pq.read_table(ROOT / path, columns=['target_live_id', 'source_spoof_id']).to_pylist():
                for sid in (r['target_live_id'], r['source_spoof_id']):
                    members[sid] = s
        self.assertEqual(sum(v == 'TRAIN' for v in members.values()), 12668)
        self.assertEqual(sum(v == 'VAL' for v in members.values()), 2653)
        self.assertTrue(all(split[k] == v for k, v in members.items()))
        self.assertFalse(any(split[k] == 'TEST' for k in members))


class CLI(unittest.TestCase):
    def setUp(self):
        self.cli = cli()
        self.env = dict(pio.PRECISION_ENV, PYTHONHASHSEED='42')

    def test_17_refusals_before_import(self):
        for argv in (['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--resume'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--epochs', '3'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--val-steps=10'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--best-checkpoint'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--test'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--output-root', '/tmp/x'],
                     ['--seed', '42', '--execution-config', pio.EXEC_CONFIG, '--limit-steps', '3']):
            with self.assertRaises(SystemExit) as cm:
                self.cli.parse(argv)
            self.assertEqual(cm.exception.code, 2)
        self.assertNotIn('tensorflow', sys.modules)

    def _storage(self, tmp, pack_ok=True):
        rt = Path(tmp)
        (rt / 'data/processed/faces_256').mkdir(parents=True)
        pack = rt / 'data/processed' / pio.PACK_DIRNAME
        pack.mkdir()
        manifest = {'payloads': {}}
        raw = fp.dumps(manifest)
        (pack / 'pack_manifest.json').write_bytes(raw)
        return {'runtime_root': str(rt), 'faces_256_root': str(rt / 'data/processed/faces_256'),
                'pack_root': str(pack)}, hashlib.sha256(raw).hexdigest()

    def test_18_preflight_gates(self):
        args = types.SimpleNamespace(seed=42, execution_config=pio.EXEC_CONFIG, preflight_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            storage, digest = self._storage(tmp)
            with self.assertRaises(SystemExit):                       # pack manifest differs from the pin
                self.cli.preconditions(args, environ=self.env, dirty=False, storage=storage)
            original = pio.PACK_MANIFEST_SHA256
            pio.PACK_MANIFEST_SHA256 = digest
            try:
                branch = self.cli.subprocess.run(['git', '-C', str(ROOT), 'branch', '--show-current'],
                                                 capture_output=True, text=True).stdout.strip()
                if branch == 'm6-baselines':
                    plan = self.cli.preconditions(args, environ=self.env, dirty=False, storage=storage)
                    self.assertEqual((plan['run_dir_exists'], plan['tensorflow_imported'], plan['images_decoded']),
                                     (False, False, 0))
                    self.assertFalse(Path(plan['run_dir']).exists())          # preflight writes no run root
                    Path(plan['run_dir']).mkdir(parents=True)
                    with self.assertRaises(SystemExit):                     # fresh only
                        self.cli.preconditions(args, environ=self.env, dirty=False, storage=storage)
                with self.assertRaises(SystemExit):
                    self.cli.preconditions(args, environ=self.env, dirty=True, storage=storage)
                with self.assertRaises(SystemExit):
                    self.cli.preconditions(args, environ=dict(self.env, PYTHONHASHSEED='1'), dirty=False,
                                           storage=storage)
                with self.assertRaises(SystemExit):
                    self.cli.preconditions(types.SimpleNamespace(seed=60903, execution_config=pio.EXEC_CONFIG,
                                                                 preflight_only=True), environ=self.env, dirty=False,
                                           storage=storage)
            finally:
                pio.PACK_MANIFEST_SHA256 = original

    def test_19_roots_and_no_test_path(self):
        rt = Path('/rt')
        self.assertEqual(pio.run_root(rt, pio.SCIENTIFIC, 42), rt / 'runs/m6/E03/seed_42')
        q = pio.run_root(rt, pio.QUALIFICATION, 60903, case='val_a', run_id='x' * 16)
        self.assertEqual(q.relative_to(rt).parts[:3], ('qualification', 'e03_production', 'E03'))
        self.assertNotIn('runs', q.relative_to(rt).parts)
        for bad in ((pio.SCIENTIFIC, 60903), (pio.QUALIFICATION, 42)):
            with self.assertRaises(pio.E03Stop):
                pio.validate_mode_seed(*bad)
        for src in (RUNNER, (ROOT / 'tools/run_e03.py').read_text(), (ROOT / 'methods/stdn/feed_pack.py').read_text()):
            self.assertNotIn("'TEST'", src.replace("!= 'TEST'", ''))
        self.assertNotIn('.restore(', RUNNER)
        self.assertNotIn('get_checkpoint_state', RUNNER)

    def test_20_python38_compat_layer_is_infrastructure_only(self):
        self.assertIn("pathlib.PurePath.is_relative_to = _is_relative_to", RUNNER)
        self.assertIn('landmarks_module.validate_source = original', function_source('make_landmark_adapter'))
        import methods.stdn.production_runner as pr
        self.assertTrue(pr._is_relative_to(Path('/a/b'), '/a') and not pr._is_relative_to(Path('/a/b'), '/c'))


if __name__ == '__main__':
    unittest.main()
