"""E05 PCGAN (controlled architecture resolution) production runner: data path, budget, checkpoints, CLI, lifecycle.

Laptop/CPU. Framework-free cases run in the repository .venv; torch cases (loader order, trainer accounting,
checkpoint writer) run where torch is importable (gpat-m7-cpu, gpat-m6-e05) and are skipped otherwise. No benchmark
image is read: faces are synthetic PNGs in a temp root; the only real data read is the frozen TRAIN pair manifest
through its column allowlist.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.common import runlog  # noqa: E402
from methods.common.config import load_method_config, sha256_file  # noqa: E402
from methods.common.learned import PreparationError, checkpoint_metadata  # noqa: E402
from methods.common.runlog import RunDirectoryError  # noqa: E402
from methods.pcgan import production_io as pio  # noqa: E402
from methods.pcgan import production_runner as pr  # noqa: E402

try:
    import torch
except ImportError:
    torch = None
needs_torch = unittest.skipIf(torch is None, 'torch not installed in this environment')
CFG = 'configs/execution/m5_gpu_3090.yaml'
SID = [hashlib.sha256(f'sample{i}'.encode()).hexdigest() for i in range(12)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    env = runlog.environment_metadata()
    env.update(framework='torch', framework_version='toy', pytorch_version='toy', tensorflow_version=None,
               dependency_fingerprint='toy', environment_lock_path='toy', gpu_note='CPU toy test')
    return env, {k: 'CPU toy test' for k, v in env.items() if v is None}


def write_manifest(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {c: [r[c] for r in rows] for c in pio.ALLOWLIST}
    cols['source_subject'] = ['s'] * len(rows)
    pq.write_table(pa.table(cols), path)
    return sha256_file(path)


def synthetic_rows(n=4, split='TRAIN'):
    return [{'pair_id': f'PTR{i + 1:06d}', 'source_spoof_id': SID[2 * i], 'target_live_id': SID[2 * i + 1],
             'dataset': 'casia_fasd', 'split': split} for i in range(n)]


def identities():
    cfg, contracts = pio.verify_contract()
    return pio.identities(cfg, contracts)


class TestE05DataPath(unittest.TestCase):
    def test_01_real_manifest_train_only_allowlist(self):
        rows, datasets, ev = pio.read_pair_manifest()
        self.assertEqual((ev['rows'], ev['splits'], ev['columns_read'], ev['subject_columns_read']),
                         (8838, ['TRAIN'], list(pio.ALLOWLIST), []))
        self.assertEqual(ev['rows_by_dataset'], {'casia_fasd': 2520, 'msu_mfsd': 1200, 'siwmv2': 5118})
        self.assertTrue(all(r['source_spoof_id'] != r['target_live_id'] for r in rows))
        self.assertEqual(set(datasets.values()), set(pio.DATASETS))

    def test_02_non_train_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (synthetic_rows(3, 'VAL'), synthetic_rows(3, 'TEST'),
                        synthetic_rows(2) + [dict(synthetic_rows(3)[2], split='VAL')]):
                p = Path(d) / 'm.parquet'
                digest = write_manifest(p, bad)
                with self.assertRaisesRegex(PreparationError, 'non-TRAIN'):
                    pio.read_pair_manifest(p, expected_sha256=digest, expected_rows=len(bad))
            rows = synthetic_rows(3)
            rows[1]['target_live_id'] = rows[1]['source_spoof_id']
            digest = write_manifest(p, rows)
            with self.assertRaisesRegex(PreparationError, 'distinct'):
                pio.read_pair_manifest(p, expected_sha256=digest, expected_rows=3)
            good = synthetic_rows(3)
            digest = write_manifest(p, good)
            out, _, ev = pio.read_pair_manifest(p, expected_sha256=digest, expected_rows=3)
            self.assertEqual([r['pair_id'] for r in out], ['PTR000001', 'PTR000002', 'PTR000003'])
            self.assertEqual(ev['subject_columns_read'], [])

    def test_03_manifest_sha_gate_before_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'm.parquet'
            write_manifest(p, synthetic_rows(3))
            with mock.patch('pyarrow.parquet.read_schema', side_effect=AssertionError('parsed before SHA')):
                with self.assertRaisesRegex(PreparationError, 'SHA256'):
                    pio.read_pair_manifest(p)
            p.write_bytes(b'not parquet')
            with self.assertRaisesRegex(PreparationError, 'SHA256'):
                pio.read_pair_manifest(p, expected_rows=3)

    def test_04_canonical_image_contract(self):
        rgb = np.zeros((256, 256, 3), np.uint8)
        rgb[0, 0] = (0, 128, 255)
        x = pio.canonical_input(rgb)
        self.assertEqual((x.shape, x.dtype), ((3, 256, 256), np.float32))
        self.assertEqual((float(x[0, 0, 0]), float(x[2, 0, 0])), (-1.0, 1.0))
        self.assertAlmostEqual(float(x[1, 0, 0]), 128 / 255 * 2 - 1, places=6)
        for bad in (np.zeros((256, 256, 3), np.float32), np.zeros((128, 128, 3), np.uint8),
                    np.zeros((256, 256), np.uint8)):
            with self.assertRaises(PreparationError):
                pio.canonical_input(bad)

    def test_05_dataset_reads_only_the_relation_from_canonical_root(self):
        import cv2
        rows = synthetic_rows(2)
        with tempfile.TemporaryDirectory() as d:
            faces = Path(d) / 'faces_256'
            (faces / 'casia_fasd').mkdir(parents=True)
            rng = np.random.default_rng(0)
            arrays = {}
            for r in rows:
                for k in ('source_spoof_id', 'target_live_id'):
                    arrays[r[k]] = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                    cv2.imwrite(str(faces / 'casia_fasd' / f'{r[k]}.png'), arrays[r[k]][:, :, ::-1])
            reader, decode = pr.face_reader(faces, {r[k]: 'casia_fasd' for r in rows
                                                    for k in ('source_spoof_id', 'target_live_id')})
            item = pr.PairDataset(rows, reader, decode)[1]
            self.assertEqual((item['pair_id'], item['source_spoof_id'], item['target_live_id']),
                             ('PTR000002', rows[1]['source_spoof_id'], rows[1]['target_live_id']))
            np.testing.assert_array_equal(item['x_src'], pio.canonical_input(arrays[rows[1]['source_spoof_id']]))
            np.testing.assert_array_equal(item['x_tgt'], pio.canonical_input(arrays[rows[1]['target_live_id']]))
            png = (faces / 'casia_fasd' / f"{rows[1]['source_spoof_id']}.png").read_bytes()
            self.assertEqual(item['source_png_sha256'], hashlib.sha256(png).hexdigest())
            with self.assertRaises(PreparationError):
                reader(SID[11])              # not in the relation


class TestE05Contract(unittest.TestCase):
    def test_06_budget_and_frozen_values(self):
        cfg, _ = pio.verify_contract()
        self.assertEqual((pio.TOTAL_ITERATIONS, pio.BATCH_SIZE, pio.LEARNING_RATE, pio.BETAS),
                         (4000, 1, 1e-6, (0.9, 0.999)))
        for path, value in ((('training', 'total_iterations'), 3999), (('training', 'batch_size'), 2),
                            (('optimizer', 'learning_rate'), 2e-4), (('losses', 'alpha'), 0.3),
                            (('checkpoint', 'selection_uses_val'), True), (('training', 'input_resolution'), 128)):
            bad = copy.deepcopy(cfg)
            bad[path[0]][path[1]] = value
            with self.assertRaises(PreparationError):
                pio.verify_contract(bad)
        bad = copy.deepcopy(cfg)
        bad['_runtime']['config_sha256'] = '0' * 64
        with self.assertRaises(PreparationError):
            pio.verify_contract(bad)

    def test_07_frozen_inputs_untouched(self):
        self.assertEqual(sha256_file(ROOT / pio.CONFIG_PATH), pio.CONFIG_SHA)
        self.assertEqual((ROOT / pio.CONFIG_PATH).read_bytes(), (ROOT / pio.SNAPSHOT_PATH).read_bytes())
        for rel, digest in ((pio.LOCK_PATH, pio.LOCK_SHA), (pio.PAIR_MANIFEST, pio.PAIR_MANIFEST_SHA),
                            (pio.A5_OVERLAY_PATH, pio.A5_OVERLAY_SHA), (pio.M6D4C_OVERLAY_PATH, pio.M6D4C_OVERLAY_SHA),
                            (pio.COMPAT_MANIFEST_PATH, pio.COMPAT_MANIFEST_SHA), (pio.SPEC_PATH, pio.SPEC_SHA)):
            self.assertEqual(sha256_file(ROOT / rel), digest, rel)
        for rel in ('methods/pcgan/training_runner.py', 'methods/pcgan/training_losses.py',
                    'methods/pcgan/blur.py', 'methods/pcgan/architecture.py'):
            self.assertEqual(subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', '3b0d308', '--', rel]).returncode,
                             0, rel + ' changed since 3b0d308')

    def test_07b_environment_gate_only_tolerates_driver_version(self):
        lock = json.loads((ROOT / pio.LOCK_PATH).read_text())['identity']
        live = json.loads(json.dumps(lock))
        live['launch_environment']['PYTHONHASHSEED'] = '42'
        self.assertEqual(pio.environment_differences(lock, live, 42)[0], [])
        name, version, uuid = lock['gpu_driver'].split(', ')
        live['gpu_driver'] = ', '.join((name, '580.178.04', uuid))
        diff, driver = pio.environment_differences(lock, live, 42)
        self.assertEqual((diff, driver['driver_version_changed'], driver['lock_driver_version'],
                          driver['live_driver_version']), ([], True, version, '580.178.04'))
        for key, value in (('gpu_driver', ', '.join((name, version, 'GPU-other'))), ('torch', '2.13.0'),
                           ('cudnn', 91000), ('python_sha256', '0' * 64)):
            bad = json.loads(json.dumps(live))
            bad[key] = value
            self.assertNotEqual(pio.environment_differences(lock, bad, 42)[0], [], key)
        self.assertNotEqual(pio.environment_differences(lock, live, 1337)[0], [])     # PYTHONHASHSEED = run seed

    def test_08_checkpoint_cadence_terminal_4000_selected(self):
        self.assertEqual(pio.checkpoint_iterations(), [1, 4000])
        self.assertEqual(pio.checkpoint_kind(4000), ('selected', True, 'BASELINE_FINAL_STATE_V1'))
        self.assertEqual(pio.checkpoint_kind(1), ('periodic', False, 'Official cadence; not final selection'))
        for t in (0, 2, 3999, 4001):
            with self.assertRaises(PreparationError):
                pio.checkpoint_kind(t)
        cfg = load_method_config('E05')
        meta = checkpoint_metadata(cfg, 42, epoch=1, global_step=4000, path='checkpoints/iteration_4000.pth',
                                   file_size_bytes=10, sha256='a' * 64, checkpoint_type='selected',
                                   selected_for_final=True)
        self.assertEqual(meta['selection_reason'], 'BASELINE_FINAL_STATE_V1')
        with self.assertRaises(PreparationError):
            checkpoint_metadata(cfg, 42, epoch=1, global_step=1, path='checkpoints/iteration_0001.pth',
                                file_size_bytes=10, sha256='a' * 64, checkpoint_type='selected',
                                selected_for_final=True)

    def test_09_no_val_test_or_best_selection(self):
        from methods.pcgan import PCGANAdapter
        adapter = PCGANAdapter()
        for split in ('VAL', 'TEST'):
            with self.assertRaises(PreparationError):
                adapter.checkpoint_policy(42, selection_split=split)
        self.assertTrue({'--best', '--best-checkpoint', '--selection-split', '--val', '--test'} <=
                        set(pio.REFUSED_OPTIONS))
        for rel in pio.PRODUCTION_FILES + ('methods/pcgan/production_qualification.py',):
            src = (ROOT / rel).read_text()
            self.assertNotIn('best_', src, rel)
            self.assertNotIn('split_v1', src, rel)
            self.assertNotIn('val_pairs', src, rel)

    def test_10_modes_seeds_roots_never_collide(self):
        self.assertEqual(str(pio.run_root('/rt', pio.SCIENTIFIC, 42)), '/rt/runs/m6/E05/seed_42')
        q = pio.run_root('/rt', pio.QUALIFICATION, 60901, case='train', run_id='0123456789abcdef')
        self.assertEqual(str(q), '/rt/qualification/e05_production/E05/train-q60901-0123456789abcdef')
        self.assertFalse(q.is_relative_to('/rt/runs'))
        for mode, seed in ((pio.SCIENTIFIC, 60901), (pio.QUALIFICATION, 42), (pio.SCIENTIFIC, 7),
                           (pio.QUALIFICATION, 1337)):
            with self.assertRaises(PreparationError):
                pio.validate_mode_seed(mode, seed)
        with self.assertRaises(PreparationError):
            pio.run_root('/rt', pio.QUALIFICATION, 60901, case='scientific', run_id='0123456789abcdef')
        with self.assertRaises(PreparationError):
            pio.run_root('/rt', pio.SCIENTIFIC, 42, case='train')


class TestE05CLI(unittest.TestCase):
    def setUp(self):
        self.cli = load('tools/run_e05.py', 'run_e05_under_test')

    def env(self, seed=42, **over):
        return {**pio.LAUNCH_ENVIRONMENT, 'PYTHONHASHSEED': str(seed), **over}

    def storage(self, d, *, build=True):
        rt = Path(d) / 'runtime'
        (rt / 'data/processed/faces_256').mkdir(parents=True)
        if build:
            (rt / 'builds/e05_pcgan').mkdir(parents=True)
            (rt / 'builds/e05_pcgan/compatibility_patch_manifest.json').write_text('{}')
        return {'runtime_root': str(rt), 'faces_256_root': str(rt / 'data/processed/faces_256'),
                'exec_config': CFG, 'exec_config_sha256': 'x'}

    def test_11_override_refusals(self):
        base = ['--seed', '42', '--execution-config', CFG]
        for extra in (['--iterations', '1'], ['--batch-size=2'], ['--lr', '1e-4'], ['--beta1', '0.5'],
                      ['--resolution', '128'], ['--loss-weights', 'x'], ['--alpha', '0'], ['--amp'], ['--tf32'],
                      ['--checkpoint-selection', 'best'], ['--best'], ['--datasets', 'casia_fasd'], ['--val'],
                      ['--test'], ['--pair-manifest', 'x'], ['--output', '/tmp/x'], ['--run-root', '/tmp/x'],
                      ['--qualification-only'], ['--qualification-iterations', '3'], ['--resume'],
                      ['--resume-state', 'x'], ['--auto-resume'], ['--workers', '4'], ['--unknown-flag']):
            with self.assertRaises(SystemExit) as cm:
                self.cli.parse(base + extra)
            self.assertEqual(cm.exception.code, 2, extra)
        self.assertNotIn('torch', sys.modules) if torch is None else None

    def test_12_seed_dirty_env_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            storage = self.storage(d)
            cases = ((['--seed', '60901', '--execution-config', CFG], self.env(60901), False),
                     (['--seed', '7', '--execution-config', CFG], self.env(7), False),
                     (['--seed', '42', '--execution-config', CFG], self.env(), True),
                     (['--seed', '42', '--execution-config', CFG], self.env(PYTHONHASHSEED='1'), False),
                     (['--seed', '42', '--execution-config', CFG], self.env(NVIDIA_TF32_OVERRIDE='1'), False),
                     (['--seed', '42', '--execution-config', CFG], self.env(CUBLAS_WORKSPACE_CONFIG=':16:8'), False),
                     (['--seed', '42', '--execution-config', 'configs/execution/other.yaml'], self.env(), False))
            for argv, env, dirty in cases:
                with self.assertRaises(SystemExit) as cm:
                    self.cli.preconditions(self.cli.parse(argv), environ=env, dirty=dirty, storage=storage)
                self.assertEqual(cm.exception.code, 2, argv)
            with self.assertRaises(SystemExit):
                self.cli.preconditions(self.cli.parse(['--seed', '42', '--execution-config', CFG]), environ=self.env(),
                                       dirty=False, storage=self.storage(Path(d) / 'nobuild', build=False))

    def test_13_preflight_writes_nothing_imports_no_torch(self):
        with tempfile.TemporaryDirectory() as d:
            storage = self.storage(d)
            before = sorted(str(p) for p in Path(d).rglob('*'))
            plan = self.cli.preconditions(self.cli.parse(['--seed', '1337', '--execution-config', CFG,
                                                          '--preflight-only']),
                                          environ=self.env(1337), dirty=False, storage=storage)
            self.assertEqual(sorted(str(p) for p in Path(d).rglob('*')), before)
            self.assertFalse(Path(plan['run_dir']).exists())
            self.assertEqual((plan['experiment_seed'], plan['total_iterations'], plan['checkpoint_iterations'],
                              plan['resume'], plan['images_read'], plan['model_executed']),
                             (1337, 4000, [1, 4000], 'NOT_QUALIFIED_FRESH_ONLY', 0, False))
            if torch is None:
                self.assertFalse(plan['torch_imported'])
                self.assertNotIn('torch', sys.modules)

    def test_14_fresh_only_occupied_root_refused(self):
        with tempfile.TemporaryDirectory() as d:
            storage = self.storage(d)
            (Path(storage['runtime_root']) / 'runs/m6/E05/seed_2026').mkdir(parents=True)
            with self.assertRaises(SystemExit) as cm:
                self.cli.preconditions(self.cli.parse(['--seed', '2026', '--execution-config', CFG]),
                                       environ=self.env(2026), dirty=False, storage=storage)
            self.assertEqual(cm.exception.code, 2)


class TestE05Lifecycle(unittest.TestCase):
    def context(self, root, mode=pio.SCIENTIFIC, case=None):
        env, reasons = environment()
        return pr.E05RunContext(mode=mode, seed=42 if mode == pio.SCIENTIFIC else pio.QUALIFICATION_SEED,
                                runtime_root=root, environment=env, missing_environment_reasons=reasons,
                                identities=identities(), case=case)

    def record(self, t, mode=pio.SCIENTIFIC):
        losses = dict.fromkeys(pio.TRAIN_LOSS_KEYS, 0.5)
        return pio.step_record(iteration=t, learning_rates={'D': 1e-6, 'G': 1e-6}, losses=losses,
                               wall_clock_seconds=1.0, step_seconds=0.5, gpu_memory_bytes=None,
                               pair=synthetic_rows(1)[0], sampler_index=0,
                               applications={'D': t, 'G': t, 'hook_D': t, 'hook_G': t, 'total': 2 * t},
                               gradients={}, mode=mode, seed=42 if mode == pio.SCIENTIFIC else 60901,
                               input_range=[-1.0, 1.0])

    def test_15_scientific_learned_run_context_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(d)
            self.assertEqual(ctx.run_dir, Path(d) / 'runs/m6/E05/seed_42')
            with ctx:
                ctx.log_epoch(self.record(1))
                p = ctx.run_dir / 'checkpoints/iteration_4000.pth'
                p.write_bytes(b'state')
                with self.assertRaises(PreparationError):
                    ctx.record_checkpoint(path='checkpoints/iteration_4000.pth', epoch=1, global_step=4000,
                                          file_size_bytes=5, sha256=sha256_file(p), checkpoint_type='selected',
                                          selected_for_final=True, selection_reason='best')
                ctx.record_checkpoint(path='checkpoints/iteration_4000.pth', epoch=1, global_step=4000,
                                      file_size_bytes=5, sha256=sha256_file(p), checkpoint_type='selected',
                                      selected_for_final=True, selection_reason='BASELINE_FINAL_STATE_V1')
                with self.assertRaises(PreparationError):
                    ctx.log_epoch(dict(self.record(2), test_loss=0.0))
                ctx.close(completion_status='completed', summary={
                    'final_or_selected_checkpoint_path': str(p), 'final_or_selected_checkpoint_sha256': sha256_file(p),
                    'checkpoint_selection_rule': 'BASELINE_FINAL_STATE_V1'})
            files = sorted(x.name for x in ctx.run_dir.iterdir())
            self.assertEqual(files, ['checkpoint_index.json', 'checkpoints', 'generation_log.jsonl', 'metrics.jsonl',
                                     'resolved_config.yaml', 'run_manifest.json', 'run_summary.json', 'stderr.log',
                                     'stdout.log'])
            index = json.loads((ctx.run_dir / 'checkpoint_index.json').read_text())
            self.assertEqual((index['selected_final'], index['checkpoint_rule'], len(index['checkpoints'])),
                             ('checkpoints/iteration_4000.pth', 'BASELINE_FINAL_STATE_V1', 1))
            self.assertEqual(index['checkpoints'][0]['selection_reason'], 'BASELINE_FINAL_STATE_V1')
            summary = json.loads((ctx.run_dir / 'run_summary.json').read_text())
            self.assertEqual((summary['experiment_seed'], summary['completion_status'],
                              summary['checkpoint_selection_rule'], summary['test_split_accessed']),
                             (42, 'completed', 'BASELINE_FINAL_STATE_V1', False))
            manifest = json.loads((ctx.run_dir / 'run_manifest.json').read_text())
            self.assertEqual((manifest['resume'], manifest['fidelity_class'], manifest['val_split_accessed']),
                             ('NOT_QUALIFIED_FRESH_ONLY', 'CONTROLLED_ADAPTATION', False))
            rows = pio.trajectory(ctx.run_dir / 'metrics.jsonl')
            self.assertEqual([r['global_step'] for r in rows if r['record_type'] == 'epoch'], [1])
            with self.assertRaises(RunDirectoryError):       # fresh only: the root now exists
                self.context(d).open()

    def test_16_qualification_context_labels_and_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(d, pio.QUALIFICATION, 'train')
            self.assertTrue(ctx.run_dir.is_relative_to(Path(d) / 'qualification/e05_production/E05'))
            with ctx:
                ctx.log_epoch(self.record(1, pio.QUALIFICATION))
                p = ctx.run_dir / 'checkpoints/iteration_0001.pth'
                p.write_bytes(b'q')
                kw = dict(path='checkpoints/iteration_0001.pth', epoch=1, global_step=1, file_size_bytes=1,
                          sha256=sha256_file(p))
                with self.assertRaises(PreparationError):
                    ctx.record_checkpoint(**kw, checkpoint_type='selected', selected_for_final=True,
                                          selection_reason='BASELINE_FINAL_STATE_V1')
                ctx.record_checkpoint(**kw, checkpoint_type='periodic', selected_for_final=False,
                                      selection_reason=pio.PERIODIC_REASON)
                with self.assertRaises(PreparationError):     # only the terminal probe records iteration 4000
                    ctx.record_checkpoint(path='checkpoints/iteration_4000.pth', epoch=1, global_step=4000,
                                          file_size_bytes=1, sha256='a' * 64, checkpoint_type='selected',
                                          selected_for_final=True, selection_reason='BASELINE_FINAL_STATE_V1')
                ctx.close(completion_status='interrupted', summary={'failure_reason': 'planned'})
            m = json.loads((ctx.run_dir / 'run_manifest.json').read_text())
            self.assertEqual((m['experiment_seed'], m['qualification_seed'], m['scientific_run'], m['labels'][0]),
                             (None, 60901, False, 'QUALIFICATION_ONLY'))
            s = json.loads((ctx.run_dir / 'run_summary.json').read_text())
            self.assertNotEqual(s['completion_status'], 'completed')
            probe = self.context(d, pio.QUALIFICATION, 'terminal_probe')
            self.assertIn(pio.TERMINAL_PROBE_LABEL, probe.labels())
            self.assertNotEqual(probe.run_dir, ctx.run_dir)


class FakeRunner:
    """TrainingRunner-shaped stand-in on tiny CPU tensors: emits the real stage sequence, D step then G step."""

    def __init__(self, double_d=False):
        self.models = {'Encoder': torch.nn.Linear(2, 2), 'Generator': torch.nn.Linear(2, 2),
                       'ImageD': torch.nn.Linear(2, 1), 'PatchD': torch.nn.Linear(2, 1)}
        self.groups = {'G': [(f'{c}.{n}', p) for c in ('Encoder', 'Generator')
                             for n, p in self.models[c].named_parameters()],
                       'D': [(f'{c}.{n}', p) for c in ('ImageD', 'PatchD')
                             for n, p in self.models[c].named_parameters()]}
        self.optimizers = {k: torch.optim.Adam([p for _, p in v], lr=1e-6, betas=(0.9, 0.999))
                           for k, v in self.groups.items()}
        self.benchmark_iteration, self.applications, self.failed, self.double_d = 0, {'D': 0, 'G': 0}, False, double_d

    def _set(self, group):
        for key, entries in self.groups.items():
            for _, p in entries:
                p.requires_grad_(key == group)

    def cycle(self, src, tgt, observer=None):
        x = src.reshape(-1)[:2]
        for group, terms in (('D', pio.D_TERMS + ('L_D_image', 'L_D_patch', 'L_D_total')),
                             ('G', pio.G_TERMS + ('L_G_total',))):
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            self._set(group)
            names = ('ImageD', 'PatchD') if group == 'D' else ('Encoder', 'Generator')
            total = sum(self.models[n](x).square().sum() + 1.0 for n in names)
            observer(group + '_forward', self, {'losses': {k: total for k in terms}})
            total.backward()
            observer(group + '_backward', self, {})
            for _ in range(2 if (self.double_d and group == 'D') else 1):
                self.optimizers[group].step()
            self.applications[group] += 1
        self.benchmark_iteration += 1
        observer('after_G', self, {})
        for m in self.models.values():
            m.requires_grad_(True)


@needs_torch
class TestE05TorchPaths(unittest.TestCase):
    def patched(self):
        return mock.patch.multiple(pr, G_PARAMETERS=12, D_PARAMETERS=6), \
            mock.patch.object(pio, 'GROUP_TENSORS', {'D': 4, 'G': 4})

    def batch(self, i=0):
        return {'x_src': torch.rand(1, 3, 256, 256) * 2 - 1, 'x_tgt': torch.rand(1, 3, 256, 256) * 2 - 1,
                'index': torch.tensor([i]), 'pair_id': [f'PTR{i:06d}'], 'source_spoof_id': [SID[0]],
                'target_live_id': [SID[1]], 'dataset': ['casia_fasd'], 'source_png_sha256': ['a' * 64],
                'target_png_sha256': ['b' * 64]}

    def test_17_deterministic_order_by_seed(self):
        class Index:
            def __len__(self):
                return 8838

            def __getitem__(self, i):
                return i
        cfg = load_method_config('E05')
        orders = {}
        for seed in (42, 1337, 2026):
            for rep in range(2):
                loader, gen, ev = pr.build_loader(torch, Index(), pio.SCIENTIFIC, seed, cfg)
                got = [int(b[0]) for _, b in zip(range(pio.TOTAL_ITERATIONS), loader)]
                orders.setdefault(seed, []).append(got)
            self.assertEqual(orders[seed][0], orders[seed][1])
            self.assertEqual(len(set(orders[seed][0])), 4000)
            g = torch.Generator()
            g.manual_seed(seed)
            torch.empty((), dtype=torch.int64).random_(generator=g)
            self.assertEqual(orders[seed][0], torch.randperm(8838, generator=g).tolist()[:4000])
            self.assertEqual((ev['batch_size'], ev['shuffle'], ev['num_workers'], ev['drop_last']), (1, True, 0, False))
        self.assertNotEqual(orders[42][0], orders[1337][0])
        with self.assertRaises(PreparationError):
            pr.build_loader(torch, Index(), pio.SCIENTIFIC, 60901, cfg)

    def test_18_one_d_one_g_per_iteration(self):
        a, b = self.patched()
        with a, b:
            runner = FakeRunner()
            t = pr.E05Trainer(torch, runner, pio.SCIENTIFIC, 42, device='cpu')
            for it in (1, 2, 3):
                r = t.step(self.batch(it), it)
                self.assertEqual(t.applications(), {'D': it, 'G': it, 'hook_D': it, 'hook_G': it, 'total': 2 * it})
                self.assertEqual(r['gradients']['D']['inactive_group_non_none'], 0)
            with self.assertRaises(pr.TrainingStop):
                t.step(self.batch(9), 9)                    # iteration boundary
            bad = pr.E05Trainer(torch, FakeRunner(double_d=True), pio.SCIENTIFIC, 42, device='cpu')
            with self.assertRaisesRegex(pr.TrainingStop, 'one D and one G'):
                bad.step(self.batch(), 1)

    def test_19_budget_enforced_by_the_loop(self):
        class Ctx:
            mode = pio.SCIENTIFIC
        for n in (1, 3, 3999, 4001):
            with self.assertRaisesRegex(PreparationError, '4000'):
                pr.run_iterations(None, None, Ctx(), iterations=n, t_start=0.0)
        Ctx.mode = pio.QUALIFICATION
        with self.assertRaises(PreparationError):
            pr.run_iterations(None, None, Ctx(), iterations=4000, t_start=0.0)

    def test_20_checkpoint_writer_roundtrip_and_refusals(self):
        a, b = self.patched()
        with a, b, tempfile.TemporaryDirectory() as d:
            runner = FakeRunner()
            t = pr.E05Trainer(torch, runner, pio.SCIENTIFIC, 42, device='cpu')
            env, reasons = environment()
            ctx = pr.E05RunContext(mode=pio.SCIENTIFIC, seed=42, runtime_root=d, environment=env,
                                   missing_environment_reasons=reasons, identities=identities())
            with ctx:
                t.step(self.batch(), 1)
                w = pr.write_checkpoint(torch, ctx, t, 1)
                self.assertEqual((w['path'], w['checkpoint_type'], w['selected_for_final']),
                                 ('checkpoints/iteration_0001.pth', 'periodic', False))
                with self.assertRaises(PreparationError):
                    pr.write_checkpoint(torch, ctx, t, 4000)                     # not iteration 4000
                with self.assertRaises(PreparationError):
                    pr.write_checkpoint(torch, ctx, t, 4000, terminal_semantics_probe=True)  # scientific: no probe
                self.assertEqual(t.applications()['total'], 2)
                state = torch.load(ctx.run_dir / w['path'], map_location='cpu', weights_only=True)
                self.assertEqual((state['format'], state['benchmark_iteration'], state['provenance']['checkpoint_type'],
                                  sorted(state['models'])),
                                 ('E05_BASELINE_STATE_V1', 1, 'periodic', ['Encoder', 'Generator', 'ImageD', 'PatchD']))
                runner.benchmark_iteration = 4000                                # simulated terminal boundary
                runner.applications = {'D': 4000, 'G': 4000}
                t.hook_applications = {'D': 4000, 'G': 4000}
                w = pr.write_checkpoint(torch, ctx, t, 4000)
                self.assertEqual((w['path'], w['checkpoint_type'], w['selected_for_final'], w['selection_reason']),
                                 ('checkpoints/iteration_4000.pth', 'selected', True, 'BASELINE_FINAL_STATE_V1'))
                ctx.close(completion_status='interrupted', summary={'failure_reason': 'unit test'})
            index = json.loads((ctx.run_dir / 'checkpoint_index.json').read_text())
            self.assertEqual([(c['global_step'], c['checkpoint_type'], c['selected_for_final']) for c in
                              index['checkpoints']], [(1, 'periodic', False), (4000, 'selected', True)])
            self.assertEqual(sum(c['selected_for_final'] for c in index['checkpoints']), 1)

    def test_21_qualification_loop_integration(self):
        import cv2
        rows = synthetic_rows(6)
        a, b = self.patched()
        with a, b, tempfile.TemporaryDirectory() as d:
            faces = Path(d) / 'faces_256'
            (faces / 'casia_fasd').mkdir(parents=True)
            rng = np.random.default_rng(1)
            for r in rows:
                for k in ('source_spoof_id', 'target_live_id'):
                    cv2.imwrite(str(faces / 'casia_fasd' / f'{r[k]}.png'),
                                rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            reader, decode = pr.face_reader(faces, {r[k]: 'casia_fasd' for r in rows
                                                    for k in ('source_spoof_id', 'target_live_id')})
            loader, _, _ = pr.build_loader(torch, pr.PairDataset(rows, reader, decode), pio.QUALIFICATION,
                                           pio.QUALIFICATION_SEED, load_method_config('E05'))
            t = pr.E05Trainer(torch, FakeRunner(), pio.QUALIFICATION, pio.QUALIFICATION_SEED, device='cpu')
            env, reasons = environment()
            ctx = pr.E05RunContext(mode=pio.QUALIFICATION, seed=pio.QUALIFICATION_SEED, runtime_root=d,
                                   environment=env, missing_environment_reasons=reasons, identities=identities(),
                                   case='train')
            g = torch.Generator()
            g.manual_seed(pio.QUALIFICATION_SEED)
            torch.empty((), dtype=torch.int64).random_(generator=g)
            expected = torch.randperm(6, generator=g).tolist()[:3]
            with ctx:
                written = pr.run_iterations(t, loader, ctx, iterations=3, t_start=0.0)
                ctx.close(completion_status='interrupted', summary={'failure_reason': 'planned'})
            self.assertEqual([(w['global_step'], w['checkpoint_type']) for w in written], [(1, 'periodic')])
            recs = [r for r in pio.trajectory(ctx.run_dir / 'metrics.jsonl') if r['record_type'] == 'epoch']
            self.assertEqual([r['sampler_index'] for r in recs], expected)
            self.assertEqual([r['pair_id'] for r in recs], [rows[i]['pair_id'] for i in expected])
            self.assertEqual([r['optimizer_applications']['total'] for r in recs], [2, 4, 6])
            self.assertEqual({(tuple(r['input_shape']), r['input_dtype'], r['qualification_seed']) for r in recs},
                             {((1, 3, 256, 256), 'torch.float32', 60901)})
            self.assertTrue(all('experiment_seed' not in r for r in recs))


if __name__ == '__main__':
    unittest.main()
