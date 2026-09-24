"""Focused numerical contracts and retained real-GPU evidence.

Unit orchestration uses step spies, never additional Adam applications. CUDA
optimizer qualification is exclusively the two separately launched processes.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from methods.pcgan.training_runner import training_contract, OVERLAY, OVERLAY_SHA, ROOT, parameter_groups

HAS_TORCH = importlib.util.find_spec('torch') is not None


class TestBinding(unittest.TestCase):
    def test_committed_overlay_binding(self):
        self.assertEqual(hashlib.sha256((ROOT / OVERLAY).read_bytes()).hexdigest(), OVERLAY_SHA)
        self.assertEqual(training_contract()['fidelity_class'], 'CONTROLLED_ADAPTATION')

    def test_overlay_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / OVERLAY).parent.mkdir(parents=True)
            (root / OVERLAY).write_bytes((ROOT / OVERLAY).read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'immutable'):
                training_contract(root)

    def test_terminal_policy_and_test_forbidden(self):
        contract = training_contract()
        cp = contract['checkpoint']
        self.assertEqual(cp['rule'], 'BASELINE_FINAL_STATE_V1')
        self.assertEqual(cp['terminal_benchmark_iteration'], 4000)
        self.assertEqual(cp['extra_optimizer_applications'], 0)
        for key in ('selection_uses_VAL', 'selection_uses_TEST', 'best_seed'):
            self.assertFalse(cp[key])
        self.assertFalse(contract['data_policy']['TEST_allowed'])

    def test_no_extra_regularizers(self):
        d = training_contract()['discriminator']
        for name in ('R1', 'patch_R1', 'lazy_R1_scaling', 'gradient_penalty', 'additional_patch_regularizer'):
            self.assertFalse(d[name])

    def test_implementation_has_no_data_checkpoint_or_swap_api(self):
        import ast
        for file in ('training_losses.py', 'training_runner.py'):
            tree = ast.parse((Path(__file__).parents[1] / 'methods/pcgan' / file).read_text())
            calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)]
            self.assertFalse(set(calls) & {'save', 'load', 'swap', 'read_parquet', 'imread', 'fix_noise'})


@unittest.skipUnless(HAS_TORCH, 'Torch numerical tests run in unchanged GPU environment')
class TestLosses(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.src = torch.full((1, 3, 256, 256), .1, requires_grad=True)
        self.tgt = torch.full_like(self.src, .8, requires_grad=True)
        self.rec = torch.full_like(self.src, -.2, requires_grad=True)
        self.mix = torch.full_like(self.src, .4, requires_grad=True)
        self.calls = []
        class ImageD(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.7))
            def forward(self, x):
                return x.mean((1, 2, 3))[:, None] * self.weight
        class PatchD(ImageD):
            def __init__(self):
                super().__init__()
                self.aggregations = []
            def extract_features(self, x, aggregate=False):
                self.aggregations.append(aggregate)
                return x.mean((2, 3, 4)).reshape(-1, 1) * self.weight
            def discriminate_features(self, ref, candidate):
                return ref + candidate
            def forward(self, *args):
                raise AssertionError('obsolete patch forward forbidden')
        self.D, self.P = ImageD(), PatchD()
        def crop(x):
            self.calls.append(x)
            return x[:, None, :, :128, :128].expand(-1, 8, -1, -1, -1)
        self.crop = crop

    def g(self):
        from methods.pcgan.training_losses import generator_losses
        return generator_losses(self.src, self.tgt, self.rec, self.mix, self.D, self.P, self.crop)

    def d(self):
        from methods.pcgan.training_losses import discriminator_losses
        return discriminator_losses(self.src, self.tgt, self.rec, self.mix, self.D, self.P, self.crop)

    def test_unsquared_per_sample_norm_no_normalization(self):
        from methods.pcgan.training_losses import norm2_distance
        t = self.torch
        a = t.tensor([3., 4., 6., 8.]).reshape(2, 1, 1, 2)
        self.assertEqual(norm2_distance(a, t.zeros_like(a)).item(), 7.5)
        self.assertEqual(norm2_distance(a, a).item(), 0.)

    def test_source_only_reconstruction(self):
        losses, _ = self.g()
        expected = (self.src - self.rec).square().sum().sqrt()
        self.assertTrue(self.torch.equal(losses['L_rec'], expected))
        self.assertIsNone(self.torch.autograd.grad(losses['L_rec'], self.tgt, allow_unused=True)[0])

    def test_exact_a5_blur_and_connectivity(self):
        losses, tensors = self.g()
        t = self.torch
        bt = self.tgt.reshape(1, 3, 128, 2, 128, 2).mean((3, 5))
        bm = self.mix.reshape(1, 3, 128, 2, 128, 2).mean((3, 5))
        self.assertTrue(t.equal(tensors['blur_target'], bt))
        self.assertTrue(t.equal(tensors['blur_mixed'], bm))
        self.assertTrue(t.equal(losses['L_recblur'], (bt - bm).square().sum().sqrt()))
        self.assertTrue(t.autograd.grad(losses['L_recblur'], self.mix)[0].abs().sum() > 0)

    def test_raw_logit_softplus(self):
        losses, values = self.g()
        f = self.torch.nn.functional.softplus
        for name, logit in [('L_advrec', 'd_rec'), ('L_advmix', 'd_mix'), ('L_pat', 'p_mix')]:
            self.assertTrue(self.torch.equal(losses[name], f(-values[logit]).mean()))

    def test_patch_active_interface_and_independent_calls(self):
        losses, _ = self.g()
        self.assertEqual(self.P.aggregations, [True, False])
        self.assertIs(self.calls[0], self.src)
        self.assertIs(self.calls[1], self.mix)
        self.assertTrue(self.torch.autograd.grad(losses['L_pat'], self.mix)[0].abs().sum() > 0)

    def test_unit_weights_and_external_G_total(self):
        from methods.pcgan.training_diagnostics import external_totals
        losses, _ = self.g()
        self.assertTrue(external_totals(losses)['external']['L_G_total']['exact'])
        c = training_contract()['generator']
        self.assertEqual(list(c['weights'].values()), [1] * 5)
        self.assertEqual((c['alpha'], c['beta']), (.2, 1e-6))

    def test_image_D_averaging_weights_and_external_total(self):
        from methods.pcgan.training_diagnostics import external_totals
        losses, values = self.d()
        f = self.torch.nn.functional.softplus
        real = .5 * (f(-values['d_src']).mean() + f(-values['d_tgt']).mean())
        image = real + .5 * f(values['d_rec']).mean() + .5 * f(values['d_mix']).mean()
        self.assertTrue(self.torch.equal(losses['L_D_real'], real))
        self.assertTrue(self.torch.equal(losses['L_D_image'], image))
        self.assertTrue(all(v['exact'] for v in external_totals(losses)['external'].values()))

    def test_D_patch_three_draws_and_fake_detach(self):
        losses, values = self.d()
        self.assertEqual(self.P.aggregations, [True, False, False])
        self.assertEqual(len(self.calls), 3)
        self.assertIs(self.calls[0], self.src)
        self.assertIs(self.calls[1], self.src)
        self.assertFalse(self.calls[2].requires_grad)
        self.assertTrue(self.torch.equal(self.calls[2], self.mix))
        for key in ('rec_detached', 'mix_detached'):
            self.assertFalse(values[key].requires_grad)
        gradients = self.torch.autograd.grad(losses['L_D_total'], [self.rec, self.mix], allow_unused=True)
        self.assertEqual(gradients, (None, None))

    def test_D_patch_positive_and_fake_formula(self):
        losses, values = self.d()
        f = self.torch.nn.functional.softplus
        self.assertTrue(self.torch.equal(losses['L_D_patch_real'], f(-values['p_real']).mean()))
        self.assertTrue(self.torch.equal(losses['L_D_patch_fake'], f(values['p_fake']).mean()))

    def test_exact_Adam_configuration_without_step(self):
        from methods.pcgan.training_runner import make_optimizers
        models = dict(Encoder=self.D, Generator=self.P, ImageD=type(self.D)(), PatchD=type(self.P)())
        groups = parameter_groups(models)
        optimizers = make_optimizers(groups)
        self.assertEqual(set(optimizers), {'G', 'D'})
        for name, opt in optimizers.items():
            self.assertEqual(type(opt), self.torch.optim.Adam)
            self.assertEqual(opt.param_groups[0]['lr'], 1e-6)
            self.assertEqual(opt.param_groups[0]['betas'], (.9, .999))
            self.assertEqual(opt.param_groups[0]['weight_decay'], 0)
            self.assertEqual(opt.state, {})
            self.assertEqual([id(p) for p in opt.param_groups[0]['params']], [id(p) for _, p in groups[name]])

    def test_duplicate_and_overlap_rejection(self):
        models = dict(Encoder=self.D, Generator=self.D, ImageD=self.P, PatchD=type(self.P)())
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            parameter_groups(models)
        models['Generator'] = type(self.D)()
        models['PatchD'] = self.D
        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            parameter_groups(models)

    def test_explicit_batch_one_and_distinct_pair_rejections(self):
        from methods.pcgan.training_runner import explicit_pair
        for src, tgt in [(self.src, self.src), (self.src.expand(2, -1, -1, -1), self.tgt),
                         (self.src.double(), self.tgt)]:
            with self.assertRaises(RuntimeError):
                explicit_pair(Mock(), Mock(), src, tgt)

    def spy_cycle(self, *, fail_G=False, terminal=False):
        from methods.pcgan import training_runner as module
        t = self.torch
        t.backends.cuda.matmul.allow_tf32 = False
        t.backends.cudnn.allow_tf32 = False
        t.backends.cudnn.benchmark = False
        models = dict(Encoder=type(self.D)(), Generator=type(self.D)(), ImageD=self.D, PatchD=self.P)
        events, forwards, steps = [], [], []
        def make_spies(groups):
            opts = {}
            for key, entries in groups.items():
                def zero(entries=entries, **kwargs):
                    for _, p in entries:
                        p.grad = None
                def step(key=key):
                    steps.append(key)
                    if key == 'G' and fail_G:
                        raise RuntimeError('injected G failure')
                opts[key] = Mock(zero_grad=zero, step=step)
            return opts
        def forward(E, G, src, tgt):
            forwards.append((len(steps), id(src), id(tgt)))
            return dict(r_src=src * E.weight * G.weight, m=tgt * E.weight * G.weight)
        def observe(stage, runner, state):
            events.append((stage, runner.benchmark_iteration))
            if stage == 'D_backward':
                self.assertTrue(all(p.grad is None for _, p in runner.groups['G']))
            if stage == 'G_backward':
                self.assertTrue(all(p.grad is None for _, p in runner.groups['D']))
        with patch.dict(module.COUNTS, {k: 1 for k in models}, clear=True), \
                patch.object(module, 'make_optimizers', side_effect=make_spies), \
                patch.object(module, 'explicit_pair', side_effect=forward):
            runner = module.TrainingRunner(models, self.crop)
            if terminal:
                runner.benchmark_iteration = 4000
                with self.assertRaisesRegex(RuntimeError, 'iteration boundary'):
                    runner.cycle(self.src, self.tgt, observe)
                self.assertEqual(steps, [])
            elif fail_G:
                with self.assertRaisesRegex(RuntimeError, 'injected G failure'):
                    runner.cycle(self.src, self.tgt, observe)
                self.assertEqual(runner.benchmark_iteration, 0)
                self.assertTrue(runner.failed)
                with self.assertRaisesRegex(RuntimeError, 'iteration boundary'):
                    runner.cycle(self.src, self.tgt, observe)
            else:
                runner.cycle(self.src, self.tgt, observe)
                self.assertEqual(steps, ['D', 'G'])
                self.assertEqual(runner.applications, dict(D=1, G=1))
                self.assertEqual(runner.benchmark_iteration, 1)
                self.assertEqual([event[1] for event in events], [0]*7 + [1])
                self.assertEqual(forwards, [(0, id(self.src), id(self.tgt)), (1, id(self.src), id(self.tgt))])

    def test_cycle_order_recomputation_isolation_and_counter_with_step_spies(self):
        self.spy_cycle()

    def test_failed_G_does_not_increment_and_cannot_retry_partial_cycle(self):
        self.spy_cycle(fail_G=True)

    def test_terminal_counter_prevents_extra_application(self):
        self.spy_cycle(terminal=True)


class TestRetainedQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [ROOT / f'outputs/audit/M6D4D_E05_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2)]
        if not all(p.exists() for p in paths):
            raise unittest.SkipTest('qualification evidence not yet generated')
        cls.runs = [json.loads(p.read_text()) for p in paths]

    def test_both_processes_pass(self):
        for run in self.runs:
            self.assertEqual(run['status'], 'PASS')

    def test_connectivity(self):
        from methods.pcgan.training_diagnostics import validate_connectivity
        for run in self.runs:
            validate_connectivity(run['G_connectivity'], generator=True)
            validate_connectivity(run['D_connectivity'], generator=False)

    def test_D_G_order_counter_and_exact_applications(self):
        for run in self.runs:
            self.assertEqual(run['optimizer_applications'], dict(D=1, G=1, total=2))
            self.assertEqual(run['benchmark_iteration'], 1)
            self.assertEqual([v['stage'] for v in run['events']],
                ['before_D', 'D_forward', 'D_backward', 'after_D', 'before_G', 'G_forward', 'G_backward', 'after_G'])
            self.assertEqual([v['benchmark_iteration'] for v in run['events']], [0] * 7 + [1])

    def test_isolation_and_recomputation(self):
        for run in self.runs:
            self.assertTrue(run['D_fakes_detached'] and run['G_recomputation_verified'])
            for name in ('Encoder', 'Generator'):
                self.assertEqual(run['parameters_initial'][name], run['parameters_post_D'][name])
            for name in ('ImageD', 'PatchD'):
                self.assertEqual(run['parameters_post_D'][name], run['parameters_post_G'][name])

    def test_finite_owned_adam_states_and_gradients(self):
        for run in self.runs:
            for group in ('D', 'G'):
                self.assertTrue(run[group + '_adam']['ownership_verified'])
                inv = run[group + '_step_gradients'][group]
                self.assertEqual(inv['total'], inv['non_none'])
                self.assertEqual(inv['finite'], inv['non_none'])
                for state in run[group + '_adam']['entries'].values():
                    self.assertEqual(state['step']['mean'], 1.)
                    self.assertTrue(all(v['finite'] for v in state.values()))

    def test_environment_build_and_data_firewall(self):
        for run in self.runs:
            self.assertEqual(run['environment_before'], run['environment_after'])
            self.assertEqual(run['source_before'], run['source_after'])
            self.assertTrue(run['build_closure_unchanged'])
            self.assertFalse(run['build_closure']['jit_build_invoked'])
            self.assertEqual(run['firewall']['denied'], [])
            for flag in ('benchmark_data_access', 'benchmark_training', 'synthetic_bank', 'checkpoints', 'pretrained_loads'):
                self.assertFalse(run[flag])


class TestPreflightRejections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import m6d4d_e05_training_preflight as pre
        cls.pre = pre
        path = ROOT / pre.PROCESSES[0]
        if not path.exists():
            raise unittest.SkipTest('qualification evidence not yet generated')
        cls.record = json.loads(path.read_text())

    def reject(self, mutate):
        r = deepcopy(self.record)
        mutate(r)
        with self.assertRaises(ValueError):
            self.pre.validate_process(r, 1)

    def test_extra_optimizer_application_rejected(self):
        self.reject(lambda r: r['optimizer_applications'].update(D=2, total=3))

    def test_premature_counter_rejected(self):
        self.reject(lambda r: r['events'][3].update(benchmark_iteration=1))

    def test_loss_weight_drift_rejected(self):
        self.reject(lambda r: r['initial_G']['scalars'].update(L_G_total=1.))

    def test_group_alias_rejected(self):
        self.reject(lambda r: r['optimizer_membership']['G']['entries'][0].update(
            identity=r['optimizer_membership']['D']['entries'][0]['identity']))

    def test_adam_step_drift_rejected(self):
        self.reject(lambda r: next(iter(r['D_adam']['entries'].values()))['step'].update(mean=2.))

    def test_disconnected_G_adversarial_path_rejected(self):
        self.reject(lambda r: r['G_connectivity']['L_advmix']['Generator'].update(nonzero=0))

    def test_D_fake_gradient_leak_rejected(self):
        self.reject(lambda r: r['D_connectivity']['L_D_mix']['Encoder'].update(non_none=1, finite=1))

    def test_environment_mutation_rejected(self):
        self.reject(lambda r: r['environment_after'].update(torch='changed'))

    def test_checkpoint_write_rejected(self):
        self.reject(lambda r: r.update(checkpoints=1))

    def test_data_paths_rejected_before_open(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('must not open')):
            for p in ('manifests/split_v1.parquet', 'manifests/pairs_train_v1.parquet',
                      'outputs/data/a.json', 'outputs/faces_256/a.jpg', '/home/forbidden', '../forbidden'):
                with self.subTest(path=p), self.assertRaises(ValueError):
                    self.pre.read(p)


if __name__ == '__main__':
    unittest.main()
