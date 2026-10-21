"""E06b checkpoint-writer context gate (BASELINE-EXEC-E06B-S42-R1 regression).

E06b seed42 attempt 1 (run d3c0ddd5dc82a4a1, authority 24f7049) trained epoch 1 cleanly and then crashed in
E06bTrainer.checkpoint_event with AttributeError: 'LearnedRunContext' object has no attribute 'mode'. No earlier
test called the E06b checkpoint writer with the context class that run_scientific actually opens.

Every case here builds that REAL context (methods.common.learned_runlog.LearnedRunContext, constructed with the same
keywords as native_runner.run_scientific) in a temporary runtime root and calls the real
E06bTrainer.checkpoint_event. The static class needs no Torch: the trainer is built without __init__ and its
`util.save_checkpoint` writes bytes under the pinned misc/util.py file name. The live class (Torch present) uses the
real E06bTrainer.__init__, toy models and the pinned upstream misc/util.py save_checkpoint itself.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.common import runlog  # noqa: E402
from methods.common.learned import PreparationError  # noqa: E402
from methods.common.learned_runlog import LearnedRunContext  # noqa: E402
from methods.dsdg import native  # noqa: E402
from methods.dsdg import native_runner as nr  # noqa: E402

DSDG = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'
EPOCH1_FILES = ['netE_live_model_epoch_1_iter_0.pth', 'netE_spoof_model_epoch_1_iter_0.pth',
                'netG_model_epoch_1_iter_0.pth']
needs_torch = unittest.skipIf(importlib.util.find_spec('torch') is None, 'Torch absent; live case needs Torch')


def environment():
    env = runlog.environment_metadata()
    env.update(framework='torch', framework_version='toy', pytorch_version='toy', tensorflow_version=None,
               dependency_fingerprint='toy', environment_lock_path='toy', gpu_note='CPU checkpoint-gate test')
    return env, {k: 'CPU checkpoint-gate test' for k, v in env.items() if v is None}


def context(runtime_root, seed=42):
    """The run_scientific construction (methods/dsdg/native_runner.py), in a temporary runtime root."""
    env, reasons = environment()
    return LearnedRunContext(method_id=nr.METHOD_ID, seed=seed, config=nr.verify_contract(native.DSDGNativeAdapter().config),
                             runtime_root=runtime_root, command_line='test', resume=False, environment=env,
                             missing_environment_reasons=reasons)


class FakeUtil:
    """misc/util.py::save_checkpoint file naming (model_path + name + 'model_epoch_{}_iter_{}.pth'); no Torch."""

    def __init__(self):
        self.calls = []

    def save_checkpoint(self, model_path, model, epoch, iteration, name):
        out = model_path + name + 'model_epoch_{}_iter_{}.pth'.format(epoch, iteration)
        os.makedirs(model_path, exist_ok=True)
        Path(out).write_bytes(('checkpoint-gate test ' + repr((model, epoch, iteration))).encode())
        self.calls.append((model_path, model, epoch, iteration, name))


def static_trainer(mode, global_step=16):
    trainer = object.__new__(nr.E06bTrainer)
    trainer.util, trainer.mode, trainer.global_step = FakeUtil(), mode, global_step
    trainer.models = {m: m for m in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')}
    return trainer


class CheckpointGateAssertions:
    def assert_epoch1_written(self, ctx, written):
        ckdir = ctx.run_dir / 'checkpoints'
        self.assertEqual(sorted(p.name for p in ckdir.iterdir()), EPOCH1_FILES)    # no runner_state sidecar
        self.assertFalse(any('runner_state' in p.name for p in ctx.run_dir.rglob('*')))
        index = json.loads(ctx.path('checkpoint_index').read_text())['checkpoints']
        self.assertEqual(len(index), 3)
        self.assertEqual(sorted(e['path'] for e in index), ['checkpoints/' + f for f in EPOCH1_FILES])
        for entry in index:
            p = ctx.run_dir / entry['path']
            self.assertEqual((entry['epoch'], entry['global_step']), (1, 16))
            self.assertEqual((entry['checkpoint_type'], entry['selected_for_final']), ('periodic', False))
            self.assertEqual(entry['selection_reason'], 'Official cadence; not final selection')
            self.assertEqual((entry['file_size_bytes'], entry['sha256']), (p.stat().st_size, nr.sha256_file(p)))
        self.assertEqual([w['model'] for w in written], ['netE_nir', 'netE_vis', 'netG'])
        events = [json.loads(line) for line in ctx.path('metrics').read_text().splitlines()]
        event = [e for e in events if e.get('event') == 'e06b_checkpoint_event']
        self.assertEqual(len(event), 1)
        self.assertIsNone(event[0]['payload']['resume_sidecar'])

    def assert_refused_and_untouched(self, ctx, trainer, epoch):
        with self.assertRaises(PreparationError):
            trainer.checkpoint_event(epoch, ctx)
        self.assertEqual(list((ctx.run_dir / 'checkpoints').iterdir()), [])
        self.assertEqual(json.loads(ctx.path('checkpoint_index').read_text())['checkpoints'], [])


class TestE06bCheckpointGateStatic(CheckpointGateAssertions, unittest.TestCase):
    def test_01_real_context_has_no_mode_attribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                self.assertIsInstance(ctx, LearnedRunContext)
                self.assertFalse(hasattr(ctx, 'mode'))
                ctx.close(completion_status='failed', failure_reason='test only')

    def test_02_scientific_trainer_writes_epoch1_through_real_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                trainer = static_trainer(nr.SCIENTIFIC)
                written = trainer.checkpoint_event(1, ctx)
                self.assert_epoch1_written(ctx, written)
                self.assertEqual([(c[1], c[2], c[3], c[4]) for c in trainer.util.calls],
                                 [('netE_nir', 1, 0, 'netE_spoof_'), ('netE_vis', 1, 0, 'netE_live_'),
                                  ('netG', 1, 0, 'netG_')])
                ctx.close(completion_status='failed', failure_reason='test only')

    def test_03_qualification_trainer_cannot_write_scientific_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                trainer = static_trainer(nr.QUALIFICATION)
                self.assert_refused_and_untouched(ctx, trainer, 1)
                self.assertEqual(trainer.util.calls, [])
                ctx.close(completion_status='failed', failure_reason='test only')

    def test_04_non_checkpoint_epoch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                trainer = static_trainer(nr.SCIENTIFIC)
                self.assert_refused_and_untouched(ctx, trainer, 2)
                self.assertEqual(trainer.util.calls, [])
                ctx.close(completion_status='failed', failure_reason='test only')

    def test_05b_epoch200_selected_terminal_and_completed_close(self):
        """The final checkpoint event and the run_scientific completion summary through the same real context."""
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                trainer = static_trainer(nr.SCIENTIFIC, global_step=3200)
                trainer.checkpoint_event(200, ctx)
                index = {Path(e['path']).name: e for e in
                         json.loads(ctx.path('checkpoint_index').read_text())['checkpoints']}
                self.assertEqual({k: (v['checkpoint_type'], v['selected_for_final']) for k, v in index.items()},
                                 {'netE_spoof_model_epoch_200_iter_0.pth': ('terminal', False),
                                  'netE_live_model_epoch_200_iter_0.pth': ('terminal', False),
                                  'netG_model_epoch_200_iter_0.pth': ('selected', True)})
                self.assertEqual(index['netG_model_epoch_200_iter_0.pth']['selection_reason'],
                                 'OFFICIAL_GENERATOR_EPOCH_200')
                final = ctx.run_dir / 'checkpoints' / 'netG_model_epoch_200_iter_0.pth'
                ctx.close(completion_status='completed', summary={
                    'final_or_selected_checkpoint_path': str(final),
                    'final_or_selected_checkpoint_sha256': nr.sha256_file(final),
                    'checkpoint_selection_rule': ctx.config['checkpoint']['rule'],
                    'training_duration_seconds': 1.0, 'peak_vram_bytes': 0, 'optimizer_applications': 3200,
                    'missing_field_reasons': {
                        'seed_level_evaluation_metrics': 'Downstream evaluation is a later milestone'}})
            summary = json.loads(ctx.path('run_summary').read_text())
            self.assertEqual((summary['completion_status'], summary['final_or_selected_checkpoint_sha256']),
                             ('completed', nr.sha256_file(final)))

    def test_05_gate_reads_trainer_mode_not_context(self):
        src = (ROOT / 'methods/dsdg/native_runner.py').read_text()
        self.assertIn('rio.is_checkpoint_epoch(epoch) and self.mode == SCIENTIFIC', src)
        self.assertNotIn('ctx.mode', src)


@needs_torch
class TestE06bCheckpointGateLive(CheckpointGateAssertions, unittest.TestCase):
    """Real E06bTrainer.__init__ and the pinned upstream misc/util.py::save_checkpoint (torch.save)."""

    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn.functional as F
        from methods.common.upstream import upstream_modules
        from methods.dsdg import training_graph as tg
        cls.torch, cls.F, cls.tg = torch, F, tg
        cls.ctxm = upstream_modules(DSDG, ('misc.util',), ('misc',))
        cls.util = cls.ctxm.__enter__()['misc.util']

    @classmethod
    def tearDownClass(cls):
        cls.ctxm.__exit__(None, None, None)

    def trainer(self, mode, seed):
        torch = self.torch
        torch.manual_seed(0)
        models = {m: torch.nn.Linear(2, 2) for m in ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')}
        opt = self.tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'], 2e-4)
        trainer = nr.E06bTrainer(torch, self.F, self.util, models, opt, None, None, None, mode, seed,
                                 dict(nr.LAMBDAS), device='cpu')
        trainer.global_step = 16
        return trainer

    def test_06_live_scientific_epoch1_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                trainer = self.trainer(nr.SCIENTIFIC, 42)
                self.assertEqual(trainer.mode, nr.SCIENTIFIC)
                written = trainer.checkpoint_event(1, ctx)
                self.assert_epoch1_written(ctx, written)
                ctx.close(completion_status='failed', failure_reason='test only')

    def test_07_live_qualification_trainer_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with context(tmp) as ctx:
                self.assert_refused_and_untouched(ctx, self.trainer(nr.QUALIFICATION, nr.QUALIFICATION_SEED), 1)
                ctx.close(completion_status='failed', failure_reason='test only')


if __name__ == '__main__':
    unittest.main()
