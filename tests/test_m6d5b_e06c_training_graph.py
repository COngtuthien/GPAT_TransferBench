"""M6D5b E06c: exact training-graph semantics and the retained B=240 STOP_OOM evidence.

Formula tests run the unchanged methods/dsdg/training_graph.py transcription with the
pinned misc.util functions on tiny CPU toy networks (no optimizer, no backward, no
CUDA). They are skipped where Torch is absent. Retained-evidence tests are static.
The CUDA B=240 evidence comes only from the separately launched
methods/dsdg/training_qualification.py process.
"""
import ast
import inspect
import json
import math
from pathlib import Path
import unittest

from methods.common.learned import PreparationError
from methods.dsdg import DSDGAdapter
from methods.dsdg import training_graph as tg

ROOT = Path(__file__).resolve().parents[1]
DSDG = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'
AUDIT = ROOT / 'outputs/audit'
PROCESS_1 = AUDIT / 'M6D5B_E06C_SYNTHETIC_PROCESS_1.json'
REPORT = AUDIT / 'M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.md'
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # laptop: static tests only
    torch = None
needs_torch = unittest.skipIf(torch is None, 'Torch absent; formula tests run in gpat-m6-e06c')


def toy_graph(batch=6, h=8):
    """Tiny CPU stand-ins with the pinned interface; netIP is frozen like the real one."""
    from methods.common.upstream import upstream_modules
    g = torch.Generator().manual_seed(7)
    pool = lambda x: F.adaptive_avg_pool2d(x, 4).flatten(1)  # noqa: E731
    lin = lambda i, o: torch.nn.Linear(i, o)  # noqa: E731
    e_nir, e_vis, cls, dec, ip = lin(48, 4 * h), lin(48, 2 * h), lin(h, 1), lin(3 * h, 6 * 64), lin(64, 256)
    for p in ip.parameters():
        p.requires_grad = False
    seen = []

    def netIP(x):
        seen.append(x)
        return ip(F.adaptive_avg_pool2d(x, 8).flatten(1))
    nets = [lambda x: tuple(torch.split(e_nir(pool(x)), h, dim=-1)),
            lambda x: tuple(torch.split(e_vis(pool(x)), h, dim=-1)),
            lambda z: torch.sigmoid(F.interpolate(dec(z).view(z.size(0), 6, 8, 8), size=(256, 256))),
            cls, netIP]
    eps = torch.randn(3, batch, h, generator=g)
    with upstream_modules(DSDG, ('misc.util',), ('misc',)) as modules:
        util = modules['misc.util']
        draws = iter(eps)

        class U:  # pinned util except the CUDA-only reparameterize (qualified on GPU in M6D5a)
            kl_loss, reconstruction_loss, rgb2gray = util.kl_loss, util.reconstruction_loss, util.rgb2gray
            reparameterize = staticmethod(lambda mu, logvar: next(draws) * logvar.mul(0.5).exp() + mu)
        spoof, live = tg.synthetic_pair(torch, batch, 'cpu')
        label = torch.zeros(batch, dtype=torch.long)
        out = tg.training_forward(torch, F, U, nets, spoof, live, label, dict(tg.FROZEN_LAMBDAS),
                                  torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())
        return out, spoof, live, seen, util


class TestE06cTrainingGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DSDGAdapter()
        cls.p1 = json.loads(PROCESS_1.read_text())
        if torch is not None:
            cls.out, cls.spoof, cls.live, cls.seen, cls.util = toy_graph()

    # ---------------------------------------------------------------- frozen contract
    def test_exact_frozen_coefficients(self):
        self.assertEqual(tg.frozen_lambdas(self.adapter.config), tg.FROZEN_LAMBDAS)
        self.assertEqual(tg.FROZEN_LAMBDAS, {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10,
                                             'lambda_ort': 1, 'lambda_pair': 0.0})
        self.assertEqual(self.p1['lambdas'], tg.FROZEN_LAMBDAS)

    def test_pinned_statements_verbatim(self):
        self.assertEqual(tg.verify_pinned_statements((DSDG / 'train_generator.py').read_text()), [])
        self.assertEqual(self.p1['pinned_statements_verified'], len(tg.PINNED_STATEMENTS))
        self.assertEqual(self.p1['pinned_statement_mismatches'], [])

    def test_lambda_pair_zero_not_upstream_default(self):
        cfg = self.adapter.config
        self.assertEqual(cfg['losses']['lambda_pair'], 0.0)
        self.assertEqual(cfg['losses']['lambda_pair_official_default'], 5)
        bad = json.loads(json.dumps({k: cfg[k] for k in ('losses', 'training', 'optimizer')}))
        bad['losses']['lambda_pair'] = 5
        with self.assertRaises(PreparationError):
            tg.frozen_lambdas(bad)

    def test_epoch1_warmup_formula(self):
        t = dict(zip(tg.LOSS_TERMS, (100.0, 2.0, 3.0, 5.0, 0.0, 0.0, 7.0)))
        self.assertAlmostEqual(tg.total_loss(t, 1), 100.0 + 0.01 * (2 + 3 + 5 + 0 + 0 + 7))
        self.assertEqual(tg.WARMUP_BELOW_EPOCH, 2)

    def test_postwarmup_formula(self):
        t = dict(zip(tg.LOSS_TERMS, (100.0, 2.0, 3.0, 5.0, 0.0, 0.0, 7.0)))
        for epoch in (2, 3, 200):
            self.assertEqual(tg.total_loss(t, epoch), 117.0)

    # ---------------------------------------------------------------- optimizer
    def test_optimizer_call_is_pinned(self):
        tree = ast.parse(inspect.getsource(tg.build_optimizer))
        call, = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'Adam']
        self.assertEqual([k.arg for k in call.keywords], ['lr'])
        self.assertEqual(ast.unparse(call.args[0]),
                         'list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters())')

    def test_optimizer_group_exact_membership(self):
        o = self.p1['optimizer']
        self.assertEqual((o['parameters'], o['parameter_tensors'], o['param_groups'], o['duplicates']),
                         (45_370_182, 55, 1, 0))
        self.assertEqual(o['explicit_arguments_beyond_lr'], [])

    def test_netcls_and_netip_excluded(self):
        self.assertEqual(self.p1['optimizer']['overlap_netCls_netIP'], 0)
        self.assertEqual(tg.OPTIMIZER_OWNED, ('netE_nir', 'netE_vis', 'netG'))
        self.assertFalse(self.adapter.config['optimizer']['netCls_in_optimizer'])
        self.assertEqual(self.p1['parameters_initial']['netIP']['requires_grad_parameters'], 0)

    def test_adam_lr_and_no_weight_decay_override(self):
        d = self.p1['optimizer']['resolved_group_defaults']
        self.assertEqual((d['lr'], d['weight_decay'], d['amsgrad'], d['betas'], d['eps']),
                         (2e-4, 0, False, [0.9, 0.999], 1e-8))

    def test_modern_zero_grad_default_recorded(self):
        z = self.p1['zero_grad']
        self.assertIs(z['runtime_signature_default_set_to_none'], True)
        self.assertIn('no argument', z['pinned_call'])

    # ---------------------------------------------------------------- execution guard
    def test_physical_batch_exactly_240(self):
        b = tg.execution_guard(self.adapter)
        self.assertEqual((b['physical_batch_size'], b['gradient_accumulation_steps'], b['replica_factor']), (240, 1, 1))
        self.assertEqual((self.p1['physical_batch_size'], self.p1['gradient_accumulation_steps'],
                          self.p1['replica_factor']), (240, 1, 1))
        self.assertEqual(self.p1['inputs']['x_spoof']['shape'], [240, 3, 256, 256])

    def test_accumulation_forbidden(self):
        for phys, acc in ((120, 2), (60, 4), (1, 240)):
            with self.assertRaisesRegex(PreparationError, 'E06C_REQUIRES_PHYSICAL_BATCH_240'):
                tg.execution_guard(self.adapter, physical_batch_size=phys, gradient_accumulation_steps=acc)

    def test_reduced_batch_forbidden(self):
        for phys in (120, 60, 239):
            with self.assertRaises(PreparationError):
                tg.execution_guard(self.adapter, physical_batch_size=phys)

    def test_amp_and_other_workarounds_forbidden(self):
        for flag in tg.FORBIDDEN_EXECUTION:
            with self.assertRaisesRegex(PreparationError, 'forbids OOM workaround'):
                tg.execution_guard(self.adapter, **{flag: True})
        with self.assertRaises(PreparationError):
            tg.execution_guard(self.adapter, expandable_segments=True)
        self.assertFalse(self.p1['environment_before']['precision']['autocast_cuda'])
        self.assertFalse(self.p1['environment_before']['precision']['matmul_tf32'])

    # ---------------------------------------------------------------- firewall / scope
    def test_no_benchmark_data(self):
        src = inspect.getsource(tg)
        for word in ('open(', 'read_text', 'read_bytes', 'parquet', 'faces_256', 'DataLoader(', 'listdir'):
            self.assertNotIn(word, src.split('"""', 2)[2])
        self.assertEqual(self.p1['firewall']['denied'], [])
        self.assertFalse(self.p1['benchmark_data_access'] or self.p1['benchmark_training'])

    def test_test_split_forbidden(self):
        test = self.adapter.config['data']['splits']['TEST']
        self.assertFalse(test['allowed'] or test['code_path_present'])
        self.assertFalse(self.p1['TEST_access'])

    def test_no_scientific_checkpoint(self):
        self.assertNotIn('save', inspect.getsource(tg).split('"""', 2)[2])
        self.assertEqual(self.p1['counters']['checkpoint_saves'], 0)
        self.assertFalse(self.p1['checkpoint_created'] or self.p1['synthetic_bank'])

    def test_controlled_adaptation_wording(self):
        self.assertEqual(self.p1['fidelity'], 'CONTROLLED_ADAPTATION')
        report = REPORT.read_text().lower()
        self.assertIn('controlled_adaptation', report)
        for word in ('trained e06c', 'faithful dsdg', 'native dsdg', 'official reproduction of'):
            self.assertNotIn(word, report)

    # ---------------------------------------------------------------- retained STOP_OOM evidence
    def test_b240_stop_oom_evidence(self):
        p = self.p1
        self.assertEqual((p['status'], p['label'], p['diagnostic_seed']), ('STOP_OOM', 'SYNTHETIC_PHYSICAL_BATCH_240', 60502))
        self.assertNotIn(p['diagnostic_seed'], (42, 1337, 2026))
        o = p['oom']
        self.assertEqual((o['error_type'], o['phase']), ('OutOfMemoryError', 'forward_classifier_generator'))
        self.assertIn('CUDA out of memory', o['error'])
        self.assertFalse(o['optimizer_step_entered'] or o['optimizer_application_occurred'] or o['any_parameter_changed'])
        self.assertEqual((p['optimizer_applications'], p['backward_passes']), (0, 0))
        self.assertFalse(p['physical_batch_240_feasible'] or p['training_graph_executed'])

    def test_clean_resource_state(self):
        g = self.p1['gpu_before']
        self.assertTrue(self.p1['resource_clean'])
        self.assertEqual(g['compute_processes'], [])
        self.assertLessEqual(g['used_mib'], 1024)

    def test_no_process_2_fabricated(self):
        self.assertFalse((AUDIT / 'M6D5B_E06C_SYNTHETIC_PROCESS_2.json').exists())

    def test_synthetic_rows_distinct(self):
        i = self.p1['inputs']
        self.assertEqual((i['distinct_rows'], i['rows']), (480, 480))
        for k in ('x_spoof', 'x_live'):
            self.assertGreaterEqual(i[k]['min'], 0.0)
            self.assertLessEqual(i[k]['max'], 1.0)
            self.assertEqual(i[k]['dtype'], 'torch.float32')

    # ---------------------------------------------------------------- formulas (Torch, CPU toy graph)
    @needs_torch
    def test_loss_rec_formula(self):
        o = self.out
        B = o['rec'].shape[0]
        manual = ((o['rec'] - o['img']) ** 2).view(B, -1).sum(-1).mean() / 2.0
        self.assertTrue(torch.equal(o['loss_rec'], manual))
        self.assertGreater(abs(o['loss_rec'].item() - F.mse_loss(o['rec'], o['img']).item()), 1.0)

    @needs_torch
    def test_kl_assembly(self):
        o = self.out
        kl = lambda mu, lv: (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(-1)).mean()  # noqa: E731
        expected = (kl(o['mu_nir'], o['logvar_nir']) + kl(o['mu_vis'], o['logvar_vis']) + kl(o['mu_a'], o['logvar_a'])) / 3
        self.assertTrue(torch.allclose(o['loss_kl'], expected, rtol=1e-6, atol=1e-7))

    @needs_torch
    def test_mmd_full_batch_formula(self):
        o = self.out
        full = 50 * torch.abs(o['z_nir'].mean(0) - o['z_vis'].mean(0)).mean()
        self.assertTrue(torch.equal(o['loss_mmd'], full))
        halves = sum(50 * torch.abs(o['z_nir'][s].mean(0) - o['z_vis'][s].mean(0)).mean()
                     for s in (slice(0, 3), slice(3, 6))) / 2
        self.assertNotAlmostEqual(o['loss_mmd'].item(), halves.item(), places=5)

    @needs_torch
    def test_one_logit_ce_zero(self):
        o = self.out
        self.assertEqual(list(o['pre_spoof'].shape), [6, 1])
        self.assertEqual(o['loss_cls'].item(), 0.0)

    @needs_torch
    def test_orthogonality_formula(self):
        o = self.out
        self.assertTrue(torch.equal(o['loss_ort'], 1 * torch.abs((o['z_cls'] * o['z_nir']).sum(dim=1).mean())))

    @needs_torch
    def test_lightcnn_preprocessing(self):
        o, util = self.out, self.util
        self.assertEqual([list(x.shape) for x in self.seen], [[6, 1, 128, 128]] * 4)
        ref = util.rgb2gray(F.interpolate(self.spoof, size=(128, 128), mode='bilinear', align_corners=False))
        self.assertTrue(torch.equal(self.seen[0], ref))
        rec_ref = util.rgb2gray(F.interpolate(o['rec'][:, 3:6], size=(128, 128), mode='bilinear', align_corners=False))
        self.assertTrue(torch.equal(self.seen[3], rec_ref))
        self.assertAlmostEqual(float(o['nir_fc'].norm(dim=1).mean()), 1.0, places=5)

    @needs_torch
    def test_lightcnn_target_detach_semantics(self):
        o = self.out
        self.assertFalse(o['nir_fc'].requires_grad or o['vis_fc'].requires_grad)
        a, b = o['rec_nir_fc'].detach(), o['nir_fc']
        c, d = o['rec_vis_fc'].detach(), o['vis_fc']
        self.assertAlmostEqual(o['loss_ip'].item(), (1000 * (F.mse_loss(a, b) + F.mse_loss(c, d)) / 2).item(), places=3)

    @needs_torch
    def test_reconstructed_feature_gradient_path(self):
        o = self.out
        self.assertTrue(o['rec_nir_fc'].requires_grad and o['rec_vis_fc'].requires_grad)
        self.assertIsNotNone(o['rec_nir_fc'].grad_fn)
        self.assertTrue(o['loss_ip'].requires_grad)

    @needs_torch
    def test_loss_pair_zero(self):
        o = self.out
        self.assertEqual(o['loss_pair'].item(), 0.0)
        self.assertGreater(F.mse_loss(o['rec_nir_fc'], o['rec_vis_fc']).item(), 0.0)

    @needs_torch
    def test_synthetic_pair_analytic(self):
        a, b = tg.synthetic_pair(torch, 240, 'cpu')
        a2, _ = tg.synthetic_pair(torch, 240, 'cpu')
        self.assertTrue(torch.equal(a, a2))
        self.assertEqual((list(a.shape), a.dtype), ([240, 3, 256, 256], torch.float32))
        self.assertTrue(0 <= float(a.min()) and float(b.max()) <= 1)
        self.assertEqual(len({hash(r.numpy().tobytes()) for x in (a, b) for r in x}), 480)
        self.assertTrue(math.isfinite(float(a.sum())))


if __name__ == '__main__':
    unittest.main()
