"""M7D1-N1: ATOMIC_AMP_BACKOFF_RETRY (additive runtime numerical-stability resolution of the update-1971 overflow).

Tiers:
  StaticM7D1 (stdlib + yaml; any host): resolution record, policy wiring (Trainer = retry, step default = M7C4
      fail-closed), GradScaler construction and every scientific value unchanged, qualification evidence gates.
  LiveCPULoopM7D1 (torch; gpat-m7-cpu): Trainer loop around a retrying step -- one update/schedule advance, access
      counters logged once per group, amp_retry events, no checkpoint while a group is in flight.
  LiveGPUM7D1 (CUDA; gpat-m7-gpu): the real GeneratorStep / WarmupStep with stub teachers on synthetic tensors and
      injected non-finite gradients: backoff topology, final stop at scale 1, restore-retry == single attempt at the
      accepted scale (parameters, BN buffers, optimizer/scaler states, RNG, record), EMA once, resume keeps the scale.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '6cf271ebd0d658dcc5384da00265326e7bac6476'
RECORD = 'configs/amendments/gpat_m7d1_amp_retry_resolution.yaml'
EVIDENCE = 'outputs/audit/M7D1_GPAT_AMP_RETRY_QUALIFICATION.json'
THIS = 'tests/test_m7d1_gpat_amp_retry.py'


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7d1_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7d1(rel):
    commit = m7d1_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def has(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def cuda_available():
    if not has('torch', 'ptwt'):
        return False
    import torch
    return torch.cuda.is_available()


# ============================================================================= static
class StaticM7D1(unittest.TestCase):
    def test_01_resolution_record(self):
        rec = json.loads(at_m7d1(RECORD))
        self.assertEqual(rec['classification'], 'ADDITIVE_RUNTIME_NUMERICAL_STABILITY_RESOLUTION')
        self.assertEqual(rec['parent_commit'], AUTHORITY)
        for flag in ('NOT_A_FROZEN_SPEC_REWRITE', 'NOT_AN_ARCHITECTURE_CHANGE', 'NOT_AN_OPTIMIZER_LR_LOSS_CHANGE'):
            self.assertIn(flag, rec['status'])
        c = rec['invariants']
        self.assertTrue(c['amp_fp16_enabled'] and c['grad_scaler_enabled'])
        self.assertEqual(c['successful_scientific_updates'], 66300)
        self.assertFalse(c['overflowed_attempts_count_as_updates'])
        self.assertFalse(c['scientific_group_skipped'])
        self.assertEqual(rec['policy']['name'], 'ATOMIC_AMP_BACKOFF_RETRY')
        self.assertEqual(rec['policy']['backoff_factor'], 0.5)
        self.assertEqual(rec['policy']['final_failure'], 'FAIL_CLOSED_AMP_OVERFLOW_FINAL')
        self.assertEqual(rec['failed_attempt_evidence']['last_safe_recovery_sha256'],
                         'a34e73d1d8e8f6571ee214aba221a7d0858893cc25208df6be24531239e66706')
        self.assertEqual(rec['failed_attempt_evidence']['attempted_global_update'], 1971)
        self.assertEqual(rec['diagnosis']['classification'], 'LOSS_SCALE_OVERFLOW_CONFIRMED')

    def test_02_policy_wiring(self):
        import ast
        src = at_m7d1('methods/gpat/runner.py').decode()
        tree = ast.parse(src)
        consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                  if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual((consts['AMP_FAIL_CLOSED'], consts['AMP_ATOMIC_RETRY'], consts['AMP_BACKOFF'],
                          consts['AMP_MIN_SCALE']), ('FAIL_CLOSED_AMP_OVERFLOW', 'ATOMIC_AMP_BACKOFF_RETRY', 0.5, 1.0))
        self.assertIn('PRODUCTION_AMP_POLICY = AMP_ATOMIC_RETRY', src)
        # the Trainer builds every step with the production policy; the step classes default to M7C4 fail-closed
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        built = [c for c in ast.walk(trainer) if isinstance(c, ast.Call) and ast.unparse(c.func) in
                 ('GeneratorStep', 'WarmupStep')]
        self.assertEqual(len(built), 3)
        for c in built:
            self.assertEqual({k.arg: ast.unparse(k.value) for k in c.keywords}.get('amp_policy'),
                             'PRODUCTION_AMP_POLICY')
        for cls in ('GeneratorStep', 'WarmupStep'):
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            init = next(f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == '__init__')
            self.assertEqual(ast.unparse(init.args.defaults[-1]), 'AMP_FAIL_CLOSED')

    def test_03_grad_scaler_and_scientific_values_unchanged(self):
        import ast
        src = at_m7d1('methods/gpat/runner.py').decode()
        scalers = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and
                   ast.unparse(n.func) == 'torch.amp.GradScaler']
        self.assertEqual(len(scalers), 4)                         # G, D, WARMUP, handoff G: defaults only
        self.assertTrue(all(ast.unparse(n) == "torch.amp.GradScaler('cuda')" for n in scalers))
        for line in ('CLIP = 1.0', 'BETAS = (0.5, 0.999)', 'WARMUP_WD = 1e-4'):
            self.assertIn(line, src)
        for rel in ('methods/gpat/' + n for n in ('artifact_encoder.py', 'composition.py', 'config.py',
                                                  'discriminator.py', 'ema.py', 'generator.py', 'losses.py', 'model.py',
                                                  'spectral.py', 'teacher_preprocess.py', 'wavelet.py',
                                                  'runtime_contract.py', 'runner_io.py', 'runner_data.py',
                                                  'runner_checkpoint.py', 'runner_cli.py')):
            self.assertEqual(at_m7d1(rel), at_authority(rel), rel)
        for rel in ('configs/methods/gpat_b0.yaml', 'configs/methods/gpat_b1.yaml', 'configs/methods/gpat_b2.yaml',
                    'configs/methods/gpat_b3.yaml'):
            self.assertEqual(at_m7d1(rel), at_authority(rel), rel)

    def test_04_qualification_evidence(self):
        ev = json.loads(at_m7d1(EVIDENCE))
        self.assertEqual(ev['status'], 'PASS')
        self.assertTrue(all(ev['gates'].values()), ev['gates'])
        self.assertEqual(ev['classification'], 'LOSS_SCALE_OVERFLOW_CONFIRMED')
        fw = ev['firewall_qualification']
        self.assertEqual((fw['val_images'], fw['test_images'], fw['non_train_images'], fw['val_metadata'],
                          fw['test_metadata']), (0, 0, 0, 0, 0))
        sc = ev['scenarios']
        self.assertEqual(sc['preserve_before']['tree_sha256'], sc['preserve_after']['tree_sha256'])
        self.assertEqual(sc['continue_']['updates'], list(range(1971, 1992)))
        self.assertTrue(all('qualification/m7/M7D1_N1/' in s['run_dir'] for n, s in sc.items()
                            if not n.startswith('preserve')))


# ============================================================================= CPU: Trainer loop around a retrying step
class _FakeCtx:
    def __init__(self, index_path):
        self.index_path, self.logs = index_path, []

    def path(self, key):
        return self.index_path

    def log(self, kind, record):
        self.logs.append((kind, record))


@unittest.skipUnless(has('torch', 'pyarrow'), 'loop tier needs torch (gpat-m7-cpu)')
class LiveCPULoopM7D1(unittest.TestCase):
    def fake_trainer(self, d, retries_at):
        import numpy as np
        from methods.gpat import runner as R, runner_io as rio
        k = 3
        index = Path(d) / 'checkpoint_index.json'
        index.write_text(json.dumps({'checkpoints': []}))

        class RetryingStep:
            """Stands in for GeneratorStep under ATOMIC_AMP_BACKOFF_RETRY: retries inside one call."""
            amp_policy = R.AMP_ATOMIC_RETRY

            def __init__(self):
                self.on_retry, self.active, self.calls, self.during_retry = None, False, [], None

            def __call__(self, group, u):
                self.active = True
                try:
                    self.calls.append(u)
                    for r in range(retries_at.get(u, 0)):
                        self.on_retry({'global_update_attempted': u, 'retry_number': r + 1,
                                       'offending_optimizers': ['D_OPT'], 'old_scale': {'D_OPT': 65536.0 / 2 ** r},
                                       'new_scale': {'D_OPT': 32768.0 / 2 ** r}, 'optimizer_update': False})
                        if self.during_retry is not None:
                            self.during_retry()
                    return {'global_update': u, 'learning_rate': __import__(
                        'methods.gpat.runtime_contract', fromlist=['x']).main_lr(u), 'amp_attempts': 1 + retries_at.get(u, 0)}
                finally:
                    self.active = False

        class Fake(R.Trainer):
            def __init__(self):
                self.position = {'stage': 'generator', 'epoch': 1, 'next_group': k + 1, 'global_update': k}
                self.ctx, self.warm = _FakeCtx(index), None
                self.plan = [[[2 * g], [2 * g + 1]] for g in range(rio.GROUPS_PER_EPOCH)]
                self.gen_sampler = type('S', (), {'set_plan': lambda s, p: setattr(s, 'p', p)})()
                sampler = self.gen_sampler
                self.gen_loader = type('L', (), {'__iter__': lambda s: iter(
                    [{'index': np.array(mb)} for mb in sampler.p])})()
                self.accessed = []
                self.step = RetryingStep()

            def generator_plan(self, epoch):
                return self.plan

            def log_access(self, batch, roles):
                self.accessed.append(batch['index'].tolist())

        return Fake(), R, k

    def test_10_11_13_update_schedule_and_access_once_per_group(self):
        from methods.gpat import runtime_contract as rc
        with tempfile.TemporaryDirectory() as d:
            tr, R, k = self.fake_trainer(d, {5: 1, 6: 3})
            tr.run_generator_groups(R.StopFlag(), limit=4)
            self.assertEqual(tr.step.calls, [4, 5, 6, 7])                 # each update index called exactly once
            self.assertEqual(tr.position['global_update'], 7)              # advanced once per group
            self.assertEqual(tr.position['next_group'], 8)
            groups = [r for kind, r in tr.ctx.logs if kind == 'optimizer_group']
            self.assertEqual([r['global_update'] for r in groups], [4, 5, 6, 7])   # one COMPLETE record per update
            self.assertEqual([r['learning_rate'] for r in groups], [rc.main_lr(u) for u in (4, 5, 6, 7)])
            retries = [r for kind, r in tr.ctx.logs if kind == 'amp_retry']
            self.assertEqual([(r['global_update_attempted'], r['retry_number'], r['epoch'], r['group'])
                              for r in retries], [(5, 1, 1, 5), (6, 1, 1, 6), (6, 2, 1, 6), (6, 3, 1, 6)])
            self.assertTrue(all(r['optimizer_update'] is False for r in retries))
            # every planned microbatch was admitted (logged) exactly once, retries included
            self.assertEqual(tr.accessed, [[i] for g in range(3, 7) for i in (2 * g, 2 * g + 1)])

    def test_14_no_checkpoint_while_a_group_is_in_flight(self):
        with tempfile.TemporaryDirectory() as d:
            tr, R, k = self.fake_trainer(d, {4: 1})
            attempts = []

            def during():
                try:
                    tr.save_recovery()                  # the Fake has no model: only the in-flight gate can stop it
                except R.TrainingStop as exc:
                    attempts.append(str(exc))
            tr.step.during_retry = during
            tr.run_generator_groups(R.StopFlag(), limit=1)
            self.assertEqual(len(attempts), 1)
            self.assertIn('no recovery checkpoint while an optimizer group or retry is in flight', attempts[0])
            self.assertFalse(tr.in_flight())
            self.assertEqual(tr.position['global_update'], 4)
            tr.warm = type('Wm', (), {'active': True})()
            with self.assertRaises(R.TrainingStop):
                tr.save_warmup_recovery()


# ============================================================================= GPU: real steps, injected overflow
class _StubTeachers:
    """M7C4 stub teachers plus a deliberate RNG draw that enters the loss (so an un-restored RNG would show)."""

    def __init__(self, torch, rng=False):
        self.torch, self.rng = torch, rng
        g = torch.Generator().manual_seed(5)
        self.w = torch.randn(3, 512, generator=g).cuda()

    def modules(self):
        return {}

    def facexformer(self, x):
        import torch.nn.functional as F
        y = F.interpolate(x.float(), size=(224, 224), mode='bilinear', align_corners=False)
        seg = y.mean(1, keepdim=True).repeat(1, 11, 1, 1) * self.torch.arange(11., device=x.device).view(1, 11, 1, 1)
        lm = self.torch.tanh(y[:, :2, :68, 0].permute(0, 2, 1))
        return seg, lm

    def adaface(self, x):
        import torch.nn.functional as F
        return F.normalize(x.float().mean((-2, -1)) @ self.w, dim=1)

    def art(self, x):
        out = x.float().mean((-2, -1)) @ self.w
        if self.rng:
            out = out + 1e-2 * self.torch.randn_like(out) + 1e-2 * float(self.torch.rand(1))
        return out


@unittest.skipUnless(cuda_available(), 'GPU tier: needs CUDA (gpat-m7-gpu)')
class LiveGPUM7D1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        from methods.gpat import runner as R, runtime_contract as rc
        from methods.gpat.config import load_config
        from methods.gpat.model import GPATCore
        rc.apply_qualification_determinism(70404, gpu=True)
        cls.torch, cls.R, cls.rc = torch, R, rc
        cls.cfg = load_config('B3')
        torch.manual_seed(1)
        cls.core = GPATCore(cls.cfg, pretrained_state=torchvision.models.resnet18(weights=None).state_dict(),
                            with_discriminator=True).cuda().train()

    def batch(self, n, seed, valid):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        return {'index': torch.arange(n), 'x_source': (torch.rand(n, 3, 256, 256, generator=g) * 2 - 1).cuda(),
                'x_target': (torch.rand(n, 3, 256, 256, generator=g) * 2 - 1).cuda(),
                'attack': torch.randint(0, 6, (n,), generator=g).cuda(),
                'identity': torch.where(torch.tensor(valid), torch.randint(0, 60, (n,), generator=g), -1).cuda(),
                'identity_valid': torch.tensor(valid).cuda()}

    def group(self):
        return [self.batch(4, 21, [True, False, True, False]), self.batch(4, 22, [False] * 4)]

    def make(self, *, policy=None, d_scale=65536.0, g_scale=65536.0, rng=False, ema=False):
        torch, R = self.torch, self.R
        core = copy.deepcopy(self.core)
        g_opt, d_opt = R.generator_optimizer(core), R.discriminator_optimizer(core)
        step = R.GeneratorStep(core, _StubTeachers(torch, rng), self.cfg, g_opt, d_opt,
                               torch.amp.GradScaler('cuda', init_scale=g_scale),
                               torch.amp.GradScaler('cuda', init_scale=d_scale),
                               amp_policy=policy or R.AMP_ATOMIC_RETRY)
        if ema:
            from methods.gpat.ema import ModelEMA
            step.ema = {'e_art': ModelEMA(core.e_art), 'g_res': ModelEMA(core.g_res)}
        counts = {'D': 0, 'G': 0}
        for name, opt in (('D', d_opt), ('G', g_opt)):
            orig = opt.step
            opt.step = (lambda *a, _n=name, _f=orig, **k: (counts.__setitem__(_n, counts[_n] + 1), _f(*a, **k))[1])
        step.counts = counts
        step.events = []
        step.on_retry = lambda ev: step.events.append((ev, dict(counts)))
        return core, step

    def inject(self, step, *, d=0, g=0, d_always=False):
        """Non-finite gradient after unscale on the first `d` (D) / `g` (G) attempts."""
        core = step.core
        seen = {'D': 0, 'G': 0}
        p_d = next(core.discriminator.parameters())
        p_g = core.g_res.ending.weight

        def inspect(stage, s):
            seen[stage] += 1
            if stage == 'D' and (d_always or seen['D'] <= d):
                p_d.grad.view(-1)[0].fill_(float('inf'))
            if stage == 'G' and seen['G'] <= g:
                p_g.grad.view(-1)[0].fill_(float('nan'))
        step.inspect = inspect
        return seen

    def seed(self):
        self.torch.manual_seed(1234)

    def state(self, core, step):
        torch = self.torch
        return {'core': {k: v.detach().clone() for k, v in core.state_dict().items()},
                'g_opt': copy.deepcopy(step.g_opt.state_dict()), 'd_opt': copy.deepcopy(step.d_opt.state_dict()),
                'g_scaler': step.g_scaler.state_dict(), 'd_scaler': step.d_scaler.state_dict(),
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}

    def assert_same(self, a, b):
        torch = self.torch

        def eq(x, y, path):
            if torch.is_tensor(x):
                self.assertTrue(torch.equal(x, y), path)
            elif isinstance(x, dict):
                self.assertEqual(set(x), set(y), path)
                for k in x:
                    eq(x[k], y[k], f'{path}/{k}')
            elif isinstance(x, (list, tuple)):
                self.assertEqual(len(x), len(y), path)
                for i, (u, v) in enumerate(zip(x, y)):
                    eq(u, v, f'{path}[{i}]')
            else:
                self.assertEqual(x, y, path)
        eq(a, b, '')

    @staticmethod
    def strip(rec):
        return {k: v for k, v in rec.items() if k not in ('wall_seconds', 'peak_allocated_bytes', 'amp_policy',
                                                          'amp_attempts')}

    # ------------------------------------------------------------------ 1-5: backoff topology
    def test_01_d_overflow_then_success_after_one_backoff(self):
        core, step = self.make()
        self.inject(step, d=1)
        rec = step(self.group(), 6000)
        self.assertEqual(rec['amp_attempts'], 2)
        self.assertEqual((rec['D_scale_before'], rec['G_scale_before']), (32768.0, 65536.0))
        (ev, counts_at_event), = step.events
        self.assertEqual((ev['global_update_attempted'], ev['retry_number'], ev['offending_optimizers']),
                         (6000, 1, ['D_OPT']))
        self.assertEqual((ev['old_scale'], ev['new_scale']), ({'D_OPT': 65536.0}, {'D_OPT': 32768.0}))
        self.assertEqual(ev['offending_parameter_count'], {'D_OPT': 1})
        self.assertFalse(ev['optimizer_update'])
        self.assertEqual(step.counts, {'D': 1, 'G': 1})
        self.assertEqual(step.g_scaler.state_dict()['_growth_tracker'], 1)        # G untouched by the retry
        self.assertEqual((step.d_scaler.get_scale(), step.d_scaler.state_dict()['_growth_tracker']), (32768.0, 1))

    def test_02_g_overflow_then_success_after_one_backoff(self):
        core, step = self.make()
        self.inject(step, g=1)
        rec = step(self.group(), 6000)
        (ev, _), = step.events
        self.assertEqual(ev['offending_optimizers'], ['G_OPT'])
        self.assertIn('g_res.ending.weight', ev['offending_parameters']['G_OPT'])
        self.assertEqual((rec['D_scale_before'], rec['G_scale_before']), (65536.0, 32768.0))
        self.assertEqual(step.counts, {'D': 1, 'G': 1})

    def test_03_both_overflow(self):
        core, step = self.make()
        self.inject(step, d=1, g=1)
        rec = step(self.group(), 6000)
        (ev, _), = step.events
        self.assertEqual(ev['offending_optimizers'], ['D_OPT', 'G_OPT'])
        self.assertEqual(ev['new_scale'], {'D_OPT': 32768.0, 'G_OPT': 32768.0})
        self.assertEqual((rec['D_scale_before'], rec['G_scale_before']), (32768.0, 32768.0))
        self.assertEqual(step.counts, {'D': 1, 'G': 1})

    def test_04_multiple_backoffs_then_success(self):
        core, step = self.make()
        self.inject(step, d=3, g=1)
        rec = step(self.group(), 6000)
        self.assertEqual([e['retry_number'] for e, _ in step.events], [1, 2, 3])
        self.assertEqual([e['new_scale'] for e, _ in step.events],
                         [{'D_OPT': 32768.0, 'G_OPT': 32768.0}, {'D_OPT': 16384.0}, {'D_OPT': 8192.0}])
        # a scaler that is not offending in a later retry keeps its already-reduced scale
        self.assertEqual([e['scales_after_restore'] for e, _ in step.events][-1], {'D_OPT': 8192.0, 'G_OPT': 32768.0})
        self.assertEqual((rec['D_scale_before'], rec['G_scale_before'], rec['amp_attempts']), (8192.0, 32768.0, 4))
        self.assertEqual(step.counts, {'D': 1, 'G': 1})

    def test_05_overflow_persisting_at_scale_one_fails_closed_final(self):
        torch, R = self.torch, self.R
        core, step = self.make(ema=True)
        params = {k: v.detach().clone() for k, v in core.named_parameters()}
        self.inject(step, d_always=True)
        with self.assertRaises(R.AmpOverflowFinalStop) as cm:
            step(self.group(), 6000)
        r = cm.exception.record
        self.assertEqual(r['failure_type'], 'FAIL_CLOSED_AMP_OVERFLOW_FINAL')
        self.assertEqual(r['scale_at_final_attempt'], {'D_OPT': 1.0})
        self.assertEqual(r['amp_retries'], 16)                                    # 65536 -> 1 in 16 backoffs
        self.assertEqual([e['new_scale']['D_OPT'] for e, _ in step.events], [2.0 ** (15 - i) for i in range(16)])
        self.assertEqual(step.counts, {'D': 0, 'G': 0})
        self.assertTrue(all(torch.equal(p, params[k]) for k, p in core.named_parameters()))
        self.assertEqual((step.ema['e_art'].updates, step.ema['g_res'].updates), (0, 0))
        self.assertFalse(step.active)

    # ------------------------------------------------------------------ 6-12: exact restore semantics
    def run_pair(self, *, d=0, g=0, accepted=(32768.0, 65536.0), rng=False, ema=False, u=6000):
        """A: retry policy with injected overflow(s); B: one attempt from the same pre-group state at the accepted
        scales (growth tracker 0, as after a standard backoff)."""
        core_a, a = self.make(rng=rng, ema=ema)
        self.inject(a, d=d, g=g)
        self.seed()
        rec_a = a(self.group(), u)
        st_a = self.state(core_a, a)
        core_b, b = self.make(policy=self.R.AMP_FAIL_CLOSED, d_scale=accepted[0], g_scale=accepted[1], rng=rng,
                              ema=ema)
        self.seed()
        rec_b = b(self.group(), u)
        st_b = self.state(core_b, b)
        return (core_a, a, rec_a, st_a), (core_b, b, rec_b, st_b)

    def test_06_bn_buffers_restored_before_retry(self):
        torch = self.torch
        core, step = self.make()
        pre = {n: b.clone() for n, b in core.e_art.named_buffers()}
        mutated = []
        seen = self.inject(step, d=1)
        orig = step.inspect

        def inspect(stage, s):
            if stage == 'D' and seen['D'] == 0:                                   # first (failing) attempt
                mutated.append(any(not torch.equal(b, pre[n]) for n, b in core.e_art.named_buffers()))
            orig(stage, s)
        step.inspect = inspect
        self.seed()
        step(self.group(), 6000)
        self.assertEqual(mutated, [True])               # the failed forward really moved BN running statistics
        (ca, a, ra, sa), (cb, b, rb, sb) = self.run_pair(d=1)
        bn = lambda st: {k: v for k, v in st['core'].items() if 'running' in k or 'num_batches' in k}  # noqa: E731
        self.assertTrue(bn(sa))
        self.assert_same(bn(sa), bn(sb))
        self.assert_same(sa, sb)                       # params, buffers, optimizer + scaler states, RNG
        self.assertEqual(self.strip(ra), self.strip(rb))

    def test_07_rng_restored_before_retry(self):
        (ca, a, ra, sa), (cb, b, rb, sb) = self.run_pair(d=1, rng=True)
        self.assertTrue(self.torch.equal(sa['rng_cpu'], sb['rng_cpu']))
        self.assertTrue(self.torch.equal(sa['rng_cuda'], sb['rng_cuda']))
        self.assertEqual(ra['train_losses'], rb['train_losses'])                # the RNG-dependent loss is identical
        self.assert_same(sa, sb)
        # control: without the restore the RNG would have advanced twice
        core, step = self.make(policy=self.R.AMP_FAIL_CLOSED, rng=True)
        self.seed()
        step(self.group(), 6000)
        once = self.torch.get_rng_state()
        self.assertTrue(self.torch.equal(once, sa['rng_cpu']))

    def test_08_09_no_step_on_failed_attempt_and_one_step_each_on_success(self):
        core, step = self.make()
        self.inject(step, d=2, g=1)
        step(self.group(), 6000)
        self.assertEqual([c for _, c in step.events], [{'D': 0, 'G': 0}, {'D': 0, 'G': 0}])
        self.assertEqual(step.counts, {'D': 1, 'G': 1})

    def test_10_11_update_and_schedule_advance_once(self):
        (ca, a, ra, sa), (cb, b, rb, sb) = self.run_pair(d=2, accepted=(16384.0, 65536.0))
        self.assertEqual(ra['global_update'], 6000)
        self.assertEqual(ra['learning_rate'], self.rc.main_lr(6000))
        self.assertEqual({g['lr'] for g in a.g_opt.param_groups} | {g['lr'] for g in a.d_opt.param_groups},
                         {self.rc.main_lr(6000)})
        steps = {float(s['step']) for s in a.g_opt.state.values()} | {float(s['step']) for s in a.d_opt.state.values()}
        self.assertEqual(steps, {1.0})                                            # Adam advanced exactly once
        self.assertEqual(self.strip(ra), self.strip(rb))
        self.assert_same(sa, sb)

    def test_12_ema_updates_exactly_once(self):
        (ca, a, ra, sa), (cb, b, rb, sb) = self.run_pair(d=1, ema=True, u=6000)
        self.assertTrue(ra['ema_updated'])
        self.assertEqual((a.ema['e_art'].updates, a.ema['g_res'].updates), (1, 1))
        self.assert_same(a.ema['g_res'].state_dict(), b.ema['g_res'].state_dict())
        self.assert_same(a.ema['e_art'].state_dict(), b.ema['e_art'].state_dict())

    def test_16_hard_stops_are_not_retried(self):
        torch, R = self.torch, self.R
        core, step = self.make()
        orig = step.t.art
        step.t.art = lambda x: orig(x) * float('nan')
        with self.assertRaises(R.NonFiniteLossStop):
            step(self.group(), 6000)
        self.assertEqual(step.events, [])
        core, step = self.make()
        orig_step = step.g_opt.step

        def corrupting(*a, **k):
            out = orig_step(*a, **k)
            with torch.no_grad():
                core.g_res.intro.weight.view(-1)[0] = float('inf')
            return out
        step.g_opt.step = corrupting
        with self.assertRaises(R.NumericalPostStepStop):
            step(self.group(), 6000)
        self.assertEqual(step.events, [])

    def test_17_default_step_policy_is_m7c4_fail_closed(self):
        torch, R = self.torch, self.R
        core = copy.deepcopy(self.core)
        step = R.GeneratorStep(core, _StubTeachers(torch), self.cfg, R.generator_optimizer(core),
                               R.discriminator_optimizer(core), torch.amp.GradScaler('cuda'),
                               torch.amp.GradScaler('cuda'))
        self.assertEqual(step.amp_policy, R.AMP_FAIL_CLOSED)
        p = next(core.discriminator.parameters())
        step.inspect = lambda stage, s: p.grad.view(-1)[0].fill_(float('inf')) if stage == 'D' else None
        with self.assertRaises(R.AmpOverflowStop) as cm:
            step(self.group(), 6000)
        self.assertNotIsInstance(cm.exception, R.AmpOverflowFinalStop)
        self.assertEqual(step.d_scaler.get_scale(), 65536.0)

    # ------------------------------------------------------------------ 15: resume keeps the reduced scale
    def test_15_recovery_round_trip_keeps_reduced_scale(self):
        torch, R = self.torch, self.R
        from methods.gpat import runner_checkpoint as ck
        core, step = self.make()
        self.inject(step, d=2)
        step(self.group(), 6000)
        self.assertEqual(step.d_scaler.get_scale(), 16384.0)
        with tempfile.TemporaryDirectory() as d:
            payload = ck.recovery_payload(
                modules={'e_art': core.e_art, 'g_res': core.g_res, 'discriminator': core.discriminator},
                optimizers={'G_OPT': step.g_opt, 'D_OPT': step.d_opt},
                scalers={'G_SCALER': step.g_scaler, 'D_SCALER': step.d_scaler}, ema=None,
                position={'epoch': 6, 'next_group': 476, 'global_update': 6000}, provenance={'x': 1},
                identity_map_sha256=None, order={})
            info = ck.atomic_save(payload, Path(d) / 'latest.pt')
            p = ck.load(info['path'])
        fresh_d, fresh_g = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
        fresh_d.load_state_dict(p['scalers']['D_SCALER'])
        fresh_g.load_state_dict(p['scalers']['G_SCALER'])
        self.assertEqual(fresh_d.state_dict(), step.d_scaler.state_dict())
        self.assertEqual(fresh_g.state_dict(), step.g_scaler.state_dict())
        self.assertEqual((fresh_d.get_scale(), fresh_d.state_dict()['_growth_tracker']), (16384.0, 1))

    # ------------------------------------------------------------------ warmup: same semantics on WARMUP_SCALER
    def warm(self, policy=None, scale=65536.0):
        torch, R = self.torch, self.R
        core = copy.deepcopy(self.core)
        opt = R.warmup_optimizer(core)
        w = R.WarmupStep(core, opt, torch.amp.GradScaler('cuda', init_scale=scale),
                         amp_policy=policy or R.AMP_ATOMIC_RETRY)
        w.counts = []
        orig = opt.step
        opt.step = lambda *a, **k: (w.counts.append(1), orig(*a, **k))[1]
        w.events = []
        w.on_retry = w.events.append
        return core, w

    def test_18_warmup_atomic_retry(self):
        torch, R = self.torch, self.R
        b = self.batch(6, 30, [False] * 6)
        core_a, a = self.warm()
        n = {'k': 0}

        def inspect(s):
            n['k'] += 1
            if n['k'] == 1:
                core_a.attack_head.fc.weight.grad.view(-1)[0].fill_(float('inf'))
        a.inspect = inspect
        self.seed()
        rec = a(b, 5)
        (ev,) = a.events
        self.assertEqual((ev['warmup_step_attempted'], ev['offending_optimizers'], ev['new_scale']),
                         (5, ['WARMUP_OPT'], {'WARMUP_OPT': 32768.0}))
        self.assertEqual((rec['scale_before'], rec['amp_attempts'], len(a.counts)), (32768.0, 2, 1))
        core_b, bb = self.warm(policy=R.AMP_FAIL_CLOSED, scale=32768.0)
        self.seed()
        rec_b = bb(b, 5)
        self.assertEqual(self.strip(rec), self.strip(rec_b))
        self.assert_same({k: v for k, v in core_a.state_dict().items()}, {k: v for k, v in core_b.state_dict().items()})
        self.assert_same(a.opt.state_dict(), bb.opt.state_dict())
        self.assertEqual(a.scaler.state_dict(), bb.scaler.state_dict())

    def test_19_warmup_final_stop_at_scale_one(self):
        R = self.R
        core, w = self.warm()
        w.inspect = lambda s: core.attack_head.fc.weight.grad.view(-1)[0].fill_(float('inf'))
        with self.assertRaises(R.AmpOverflowFinalStop) as cm:
            w(self.batch(6, 31, [False] * 6), 7)
        self.assertEqual(cm.exception.record['scale_at_final_attempt'], {'WARMUP_OPT': 1.0})
        self.assertEqual((len(w.events), w.counts), (16, []))


if __name__ == '__main__':
    unittest.main()
