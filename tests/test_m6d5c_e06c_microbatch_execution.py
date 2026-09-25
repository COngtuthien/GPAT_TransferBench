"""M6D5c E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1: contract, formulas and retained evidence.

Formula tests run methods/dsdg/microbatch_execution.py against the unchanged M6D5b
full-batch transcription on tiny CPU float64 toy networks with the pinned misc.util
(skipped where Torch is absent). They prove the algebra, not B=240 feasibility; the
CUDA evidence comes only from methods/dsdg/microbatch_qualification.py processes.
"""
import ast
import inspect
import json
from pathlib import Path
import unittest

from methods.common.learned import PreparationError
from methods.dsdg import DSDGAdapter
from methods.dsdg import microbatch_execution as mb
from methods.dsdg import training_graph as tg

ROOT = Path(__file__).resolve().parents[1]
DSDG = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'
AUDIT = ROOT / 'outputs/audit'
OVERLAY = ROOT / mb.RESOLUTION_PATH
REFERENCE = AUDIT / 'M6D5C_E06C_REFERENCE_CHECK.json'
PROCESSES = (AUDIT / 'M6D5C_E06C_PROCESS_1.json', AUDIT / 'M6D5C_E06C_PROCESS_2.json')
REPORT = AUDIT / 'M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION.md'
M6D5B_JSON = AUDIT / 'M6D5B_E06C_TRAINING_GRAPH_QUALIFICATION.json'
M6D5B_P1 = AUDIT / 'M6D5B_E06C_SYNTHETIC_PROCESS_1.json'
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # laptop: static tests only
    torch = None
needs_torch = unittest.skipIf(torch is None, 'Torch absent; formula tests run in gpat-m6-e06c')


def body(fn):
    return inspect.getsource(fn).split('"""', 2)[-1]


def toy(batch=6, h=8, seed=11):
    """Tiny float64 CPU stand-ins with the pinned interface; netIP frozen like LightCNN."""
    torch.manual_seed(seed)  # identical initial bytes for the full-batch and microbatch paths
    g = torch.Generator().manual_seed(seed)
    pool = lambda x: F.adaptive_avg_pool2d(x, 4).flatten(1)  # noqa: E731
    lin = lambda i, o: torch.nn.Linear(i, o).double()  # noqa: E731
    e_nir, e_vis, cls, dec, ip = lin(48, 4 * h), lin(48, 2 * h), lin(h, 1), lin(3 * h, 6 * 64), lin(64, 256)
    for p in ip.parameters():
        p.requires_grad = False
    nets = [lambda x: tuple(torch.split(e_nir(pool(x)), h, dim=-1)),
            lambda x: tuple(torch.split(e_vis(pool(x)), h, dim=-1)),
            lambda z: torch.sigmoid(F.interpolate(dec(z).view(z.size(0), 6, 8, 8), size=(256, 256))),
            cls, lambda x: ip(F.adaptive_avg_pool2d(x, 8).flatten(1))]
    owned = list(e_nir.parameters()) + list(e_vis.parameters()) + list(dec.parameters())
    spoof, live = (x.double() for x in tg.synthetic_pair(torch, batch, 'cpu'))
    eps = {k: torch.randn(batch, h, generator=g, dtype=torch.float64) for k in mb.EPS_ORDER}
    return nets, owned, cls, spoof, live, torch.zeros(batch, dtype=torch.long), eps


class Spy:
    """Optimizer stand-in recording zero_grad/step calls; never changes parameters."""
    def __init__(self, params):
        self.params, self.calls = list(params), []

    def zero_grad(self, set_to_none=True):
        self.calls.append(('zero_grad', set_to_none))
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.calls.append(('step', None))


class TestE06cMicrobatchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DSDGAdapter()
        cls.overlay = json.loads(OVERLAY.read_text())

    def test_global_batch_240_contract(self):
        o = self.overlay
        self.assertEqual((o['global_batch_size'], mb.GLOBAL_BATCH), (240, 240))
        self.assertEqual(self.adapter.config['training']['effective_batch_size'], 240)
        self.assertEqual(mb.execution_guard(o)['global_batch_size'], 240)

    def test_microbatch_20_contract_and_no_tuning(self):
        self.assertEqual((self.overlay['microbatch_size'], mb.MICROBATCH), (20, 20))
        for micro in (10, 5, 40, 240):
            with self.assertRaisesRegex(PreparationError, 'microbatch tuning is forbidden'):
                mb.execution_guard(self.overlay, microbatch_size=micro)
        with self.assertRaises(PreparationError):
            mb.execution_guard(self.overlay, global_batch_size=120)

    def test_twelve_chunks(self):
        s = mb.chunk_slices()
        self.assertEqual((len(s), self.overlay['microbatches_per_step'], mb.MICROBATCHES), (12, 12, 12))
        self.assertEqual([(c.start, c.stop) for c in s], [(20 * i, 20 * i + 20) for i in range(12)])
        with self.assertRaises(PreparationError):
            mb.chunk_slices(240, 7)

    def test_one_optimizer_step_per_global_batch(self):
        self.assertEqual(self.overlay['optimizer_steps_per_global_batch'], 1)
        src = body(mb.run_global_batch)
        self.assertEqual(src.count('optimizer.step()'), 1)
        loop = src.index('for i, s in enumerate(slices)')
        self.assertGreater(src.index('optimizer.step()'), src.index('after_backward(i)'))
        self.assertEqual(src[loop:src.index('optimizer.step()')].count('.backward()'), 1)
        self.assertFalse(self.overlay['optimizer']['state_update_between_microbatches'])

    def test_epsilon_order_and_replay_policy(self):
        self.assertEqual(mb.EPS_ORDER, ('cls', 'nir', 'vis'))
        e = self.overlay['stochastic_latents']
        self.assertEqual((e['draw_order'], e['policy'], e['redraw_in_pass_2']),
                         (['cls', 'nir', 'vis'], 'STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION', False))
        self.assertFalse(e['bitwise_equivalence_with_physical_b240_claimed'])
        self.assertEqual(e['tensors'], {'eps_cls': [240, 128], 'eps_nir': [240, 128], 'eps_vis': [240, 128]})
        self.assertNotIn('normal_', body(mb.run_global_batch) + body(mb.chunk_forward) + body(mb.pass1_statistics))
        # pinned train_generator.py:125-127 order
        self.assertEqual([tg.PINNED_STATEMENTS[n] for n in (125, 126, 127)],
                         ['z_cls = reparameterize(mu_a, logvar_a)', 'z_nir = reparameterize(mu_nir, logvar_nir)',
                          'z_vis = reparameterize(mu_vis, logvar_vis)'])

    def test_pass1_no_grad_encoders_only(self):
        src = body(mb.pass1_statistics)
        self.assertIn('with torch.no_grad():', src)
        for word in ('netG', 'netCls', 'netIP', 'backward', 'step', 'zero_grad'):
            self.assertNotIn(word, src)

    def test_zero_grad_set_to_none_false_once_before_loop(self):
        src = body(mb.run_global_batch)
        self.assertEqual(src.count('zero_grad('), 1)
        self.assertIn('optimizer.zero_grad(set_to_none=False)', src)
        self.assertLess(src.index('zero_grad('), src.index('for i, s in enumerate(slices)'))

    def test_lightcnn_target_no_grad_only(self):
        tree = ast.parse(inspect.getsource(mb.chunk_forward))
        blocks = [n for n in ast.walk(tree) if isinstance(n, ast.With)]
        self.assertEqual(len(blocks), 1)
        assigned = [t.id for s in blocks[0].body for t in s.targets]
        self.assertEqual(assigned, ['nir_fc', 'vis_fc'])
        self.assertNotIn('rec_nir_fc.detach', inspect.getsource(mb.chunk_forward))
        l = self.overlay['lightcnn']
        self.assertEqual(l['target_features_no_grad'], ['nir_fc', 'vis_fc'])
        self.assertFalse(l['reconstruction_features_no_grad'] or l['reconstruction_features_detached'])

    def test_netcls_and_netip_excluded(self):
        self.assertEqual(tg.OPTIMIZER_OWNED, ('netE_nir', 'netE_vis', 'netG'))
        o = self.overlay['optimizer']
        self.assertTrue(o['netCls_excluded'] and o['netIP_excluded'])
        self.assertEqual(o['constructor'], 'torch.optim.Adam(list(netE_nir.parameters()) + '
                                           'list(netE_vis.parameters()) + list(netG.parameters()), lr=2e-4)')

    def test_fp32_no_amp_no_tf32(self):
        o = self.overlay
        self.assertEqual((o['precision'], o['amp'], o['tf32'], o['activation_checkpointing'],
                          o['cpu_parameter_offload'], o['optimizer_offload']), ('FP32', False, False, False, False, False))
        for flag in mb.FORBIDDEN_EXECUTION:
            with self.assertRaisesRegex(PreparationError, 'forbids execution workaround'):
                mb.execution_guard(o, **{flag: True})
        with self.assertRaises(PreparationError):
            mb.execution_guard(o, expandable_segments=True)
        self.assertIn('torch.float32', body(mb.draw_epsilon))
        from methods.dsdg import microbatch_qualification as mq
        src = inspect.getsource(mq.setup)
        for line in ('torch.backends.cuda.matmul.allow_tf32 = False', 'torch.backends.cudnn.allow_tf32 = False',
                     "torch.set_float32_matmul_precision('highest')"):
            self.assertIn(line, src)
        self.assertNotIn('autocast(', inspect.getsource(mq) + inspect.getsource(mb))

    def test_overlay_drift_refused(self):
        bad = dict(self.overlay, naive_microbatch_loss_averaging='allowed')
        with self.assertRaises(PreparationError):
            mb.execution_guard(bad)
        with self.assertRaises(PreparationError):
            mb.execution_guard(dict(self.overlay, physical_batch_240='QUALIFIED'))

    def test_generic_adapter_accumulation_still_refused(self):
        with self.assertRaisesRegex(PreparationError, 'E06C_REQUIRES_PHYSICAL_BATCH_240'):
            self.adapter.validate_batch(physical_batch_size=20, gradient_accumulation_steps=12)

    def test_lambda_pair_zero(self):
        self.assertEqual(tg.frozen_lambdas(self.adapter.config)['lambda_pair'], 0.0)

    def test_no_benchmark_data(self):
        from methods.dsdg import microbatch_qualification as mq
        for mod in (mb, mq):
            src = inspect.getsource(mod).split('"""', 2)[2]
            for word in ('parquet', 'faces_256', 'DataLoader(', 'listdir', 'split_v1', 'pairs_train'):
                self.assertNotIn(word, src)
        self.assertNotIn('torch.save(', inspect.getsource(mb))
        self.assertIn('benchmark data', self.overlay['not_authorized'])

    def test_physical_b240_oom_history_retained(self):
        self.assertEqual(self.overlay['physical_batch_240'], 'OOM_RETAINED')
        b = json.loads(M6D5B_JSON.read_text())
        self.assertEqual((b['classification'], b['failure_classification']),
                         ('M6D5B_E06C_TRAINING_GRAPH_STOPPED_OOM', 'CUDA_OOM_PHYSICAL_BATCH_240'))
        self.assertEqual(json.loads(M6D5B_P1.read_text())['status'], 'STOP_OOM')
        self.assertFalse((AUDIT / 'M6D5B_E06C_SYNTHETIC_PROCESS_2.json').exists())
        from methods.common.config import sha256_file
        for path, h in self.overlay['immutable_inputs_sha256'].items():
            self.assertEqual(sha256_file(ROOT / path), h, path)

    def test_controlled_adaptation_wording(self):
        self.assertEqual((self.overlay['fidelity_class'], self.overlay['classification']),
                         ('CONTROLLED_ADAPTATION', 'CONTROLLED_EXECUTION_ADAPTATION'))
        report = REPORT.read_text()
        self.assertIn('CONTROLLED_ADAPTATION PRESERVED', report)
        self.assertIn('E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED', report)
        low = report.lower()
        for word in ('trained e06c', 'faithful dsdg', 'native dsdg', 'official reproduction of',
                     'e06c_physical_batch_240_qualified'):
            self.assertNotIn(word, low)


@needs_torch
class TestE06cMicrobatchFormulas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.common.upstream import upstream_modules
        cls.ctx = upstream_modules(DSDG, ('misc.util',), ('misc',))
        cls.util = cls.ctx.__enter__()['misc.util']
        cls.lam = dict(tg.FROZEN_LAMBDAS)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def shim(self, eps):
        draws, util = iter(mb.EPS_ORDER), self.util

        class U:
            kl_loss, reconstruction_loss, rgb2gray = util.kl_loss, util.reconstruction_loss, util.rgb2gray
            reparameterize = staticmethod(lambda mu, lv: mb.replay_latent(eps[next(draws)], mu, lv))
        return U

    def full_batch(self, epoch=1, seed=11):
        nets, owned, cls, spoof, live, label, eps = toy(seed=seed)
        out = tg.training_forward(torch, F, self.shim(eps), nets, spoof, live, label, self.lam,
                                  torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())
        loss = tg.total_loss(out, epoch)
        loss.backward()
        return out, loss, [p.grad.clone() for p in owned]

    def microbatch(self, micro=2, epoch=1, seed=11, **kw):
        nets, owned, cls, spoof, live, label, eps = toy(seed=seed)
        spy, grads = Spy(owned), []
        run = mb.run_global_batch(torch, F, self.util, nets, spy, spoof, live, label, eps, self.lam,
                                  torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), epoch, microbatch=micro,
                                  before_step=lambda: grads.extend(p.grad.clone() for p in owned), **kw)
        return run, grads, spy, cls

    def test_two_pass_equals_full_batch_losses_and_gradients(self):
        for epoch in (1, 2):
            out, loss, ga = self.full_batch(epoch)
            for micro in (1, 2, 3):
                run, gb, _, _ = self.microbatch(micro, epoch)
                for k in tg.LOSS_TERMS:
                    self.assertAlmostEqual(run['global_losses'][k], out[k].item(), delta=1e-9 * max(1, abs(out[k].item())))
                total = run['epoch1_total'] if epoch == 1 else run['postwarmup_total_algebraic_only']
                self.assertAlmostEqual(total, loss.item(), delta=1e-9 * abs(loss.item()))
                self.assertAlmostEqual(run['sum_of_chunk_objectives'], loss.item(), delta=1e-9 * abs(loss.item()))
                for a, b in zip(ga, gb):
                    torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-12)

    def test_naive_microbatch_average_differs(self):
        out, _, ga = self.full_batch()
        z_nir, z_vis, z_cls = (out[k].detach() for k in ('z_nir', 'z_vis', 'z_cls'))
        naive = sum(50 * (z_nir[s] - z_vis[s]).mean(0).abs().mean() for s in mb.chunk_slices(6, 2)) / 3
        self.assertGreater(abs(naive.item() - out['loss_mmd'].item()), 1e-6)
        # explicit sign change across chunks: per-row dots [3,-1,-1,-1] -> chunk means 1, -1; global mean 0
        zc, zn = torch.ones(4, 1, dtype=torch.float64), torch.tensor([[3.], [-1.], [-1.], [-1.]], dtype=torch.float64)
        naive_ort = sum((zc[s] * zn[s]).sum(1).mean().abs() for s in mb.chunk_slices(4, 2)) / 2
        self.assertEqual((naive_ort.item(), (zc * zn).sum(1).mean().abs().item()), (1.0, 0.0))

    def test_global_delta_and_ort_statistics(self):
        nets, owned, cls, spoof, live, label, eps = toy()
        st = mb.pass1_statistics(torch, nets, spoof, live, eps, mb.chunk_slices(6, 2), self.lam, keep_latents=True)
        z = {k: torch.cat([c[k] for c in st['latents']]) for k in ('z_cls', 'z_nir', 'z_vis')}
        torch.testing.assert_close(st['delta'], z['z_nir'].mean(0) - z['z_vis'].mean(0))
        torch.testing.assert_close(st['ort_mean'], (z['z_cls'] * z['z_nir']).sum(1).mean())
        self.assertAlmostEqual(st['loss_mmd_global'].item(), 50 * st['delta'].abs().mean().item())
        self.assertAlmostEqual(st['loss_ort_global'].item(), abs(st['ort_mean'].item()))
        self.assertFalse(st['grad_enabled_inside'] or st['delta'].requires_grad)

    def test_mmd_surrogate_value_and_gradient(self):
        g = torch.Generator().manual_seed(3)
        zn, zv = (torch.randn(12, 5, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        zv.data[:, 2] = zn.data[:, 2]  # delta exactly 0 on one coordinate: sign(0) = 0
        ref = 50 * (zn.mean(0) - zv.mean(0)).abs().mean()
        ref.backward()
        gn, gv = zn.grad.clone(), zv.grad.clone()
        zn.grad = zv.grad = None
        sign = torch.sign((zn.detach().mean(0) - zv.detach().mean(0)))
        self.assertEqual(sign[2].item(), 0.0)
        sur = sum(mb.mmd_surrogate(zn[s], zv[s], sign, self.lam, 12) for s in mb.chunk_slices(12, 4))
        sur.backward()
        self.assertAlmostEqual(sur.item(), ref.item(), delta=1e-10 * max(1.0, abs(ref.item())))
        torch.testing.assert_close(zn.grad, gn)
        torch.testing.assert_close(zv.grad, gv)
        self.assertTrue(bool((gn[:, 2] == 0).all()))

    def test_ort_surrogate_value_and_gradient(self):
        g = torch.Generator().manual_seed(5)
        zc, zn = (torch.randn(12, 5, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        ref = (zc * zn).sum(1).mean().abs()
        ref.backward()
        gc, gn = zc.grad.clone(), zn.grad.clone()
        zc.grad = zn.grad = None
        sign = torch.sign((zc.detach() * zn.detach()).sum(1).mean())
        sur = sum(mb.ort_surrogate(zc[s], zn[s], sign, self.lam, 12) for s in mb.chunk_slices(12, 3))
        sur.backward()
        self.assertAlmostEqual(sur.item(), ref.item(), delta=1e-10 * max(1.0, abs(ref.item())))
        torch.testing.assert_close(zc.grad, gc)
        torch.testing.assert_close(zn.grad, gn)

    def test_weighted_local_mean_losses(self):
        out, _, _ = self.full_batch()
        run, _, _, _ = self.microbatch(2)
        for k in mb.LOCAL_TERMS:
            self.assertAlmostEqual(run['global_losses'][k], out[k].item(), delta=1e-9 * max(1, abs(out[k].item())))
        self.assertIn('w = chunk_size / global_batch', body(mb.chunk_objective))

    def test_epoch1_warmup_chunk_objective(self):
        o = {k: 1.0 for k in mb.LOCAL_TERMS}
        o.update(loss_rec=100.0, loss_kl=2.0, loss_ip=5.0, loss_mmd_surrogate=0.25, loss_ort_surrogate=0.5)
        self.assertAlmostEqual(mb.chunk_objective(o, 20, 1), 100 / 12 + 0.01 * ((2 + 5 + 1 + 1) / 12 + 0.25 + 0.5))
        self.assertAlmostEqual(mb.chunk_objective(o, 20, 2), 100 / 12 + (2 + 5 + 1 + 1) / 12 + 0.25 + 0.5)

    def test_epsilon_replay_and_draw_shape(self):
        run, _, _, _ = self.microbatch(2, keep_latents=True)
        for a, b in zip(run['stats']['latents'], run['pass2_latents']):
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]))
        state = torch.get_rng_state()
        eps = mb.draw_epsilon(torch, 240, device='cpu')
        torch.set_rng_state(state)
        again = [torch.empty((240, 128)).normal_() for _ in mb.EPS_ORDER]
        self.assertEqual(list(eps), ['cls', 'nir', 'vis'])
        for k, a in zip(mb.EPS_ORDER, again):
            self.assertEqual((list(eps[k].shape), eps[k].dtype), ([240, 128], torch.float32))
            self.assertTrue(torch.equal(eps[k], a))

    def test_set_to_none_false_once_and_single_step(self):
        run, _, spy, _ = self.microbatch(2)
        self.assertEqual(spy.calls, [('zero_grad', False), ('step', None)])
        self.assertEqual(run['chunks'], 3)

    def test_lightcnn_target_no_grad_and_reconstruction_gradient(self):
        nets, owned, cls, spoof, live, label, eps = toy()
        st = mb.pass1_statistics(torch, nets, spoof, live, eps, mb.chunk_slices(6, 2), self.lam)
        out = mb.chunk_forward(torch, F, self.util, nets, spoof[:2], live[:2], label[:2], eps, slice(0, 2), self.lam,
                               torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), st, 6)
        self.assertFalse(out['nir_fc'].requires_grad or out['vis_fc'].requires_grad)
        self.assertTrue(out['rec_nir_fc'].requires_grad and out['rec_vis_fc'].requires_grad)
        seen = {}
        out['rec_nir'].register_hook(lambda g: seen.__setitem__('rec_nir', g))
        out['loss_ip'].backward()
        self.assertGreater(int(torch.count_nonzero(seen['rec_nir'])), 0)
        self.assertGreater(float(owned[-1].grad.abs().sum()), 0)

    def test_pair_zero_and_one_logit_ce_degeneracy(self):
        run, _, _, cls = self.microbatch(2)
        self.assertEqual((run['global_losses']['loss_pair'], run['global_losses']['loss_cls']), (0.0, 0.0))
        self.assertEqual(cls.out_features, 1)
        self.assertEqual(float(cls.weight.grad.abs().sum()), 0.0)


class TestE06cMicrobatchEvidence(unittest.TestCase):
    """Retained GPU evidence (reference B=4, then two fresh B=240 processes)."""
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads(REFERENCE.read_text())
        cls.p = [json.loads(p.read_text()) for p in PROCESSES]

    def test_reference_gates(self):
        r = self.ref
        self.assertEqual((r['status'], r['mode'], r['reference_global_batch'], r['reference_microbatch']),
                         ('PASS', 'reference', 4, 2))
        self.assertTrue(all(r['gates'].values()))
        c = r['comparison']
        self.assertLessEqual(c['total_rel_error'], 1e-4)
        self.assertGreaterEqual(c['gradient']['cosine'], 0.999)
        self.assertLessEqual(c['gradient']['relative_l2'], 1e-2)
        self.assertGreaterEqual(c['update']['cosine'], 0.99)
        self.assertTrue(all(r['epsilon']['bitwise_equal_to_pinned_reparameterize_draws'].values()))

    def test_b240_processes_pass(self):
        for p in self.p:
            self.assertEqual((p['status'], p['mode'], p['diagnostic_seed']), ('PASS', 'b240', 60503))
            self.assertNotIn(p['diagnostic_seed'], (42, 1337, 2026))
            self.assertEqual((p['global_batch_size'], p['microbatch_size'], p['microbatches_per_step']), (240, 20, 12))
            self.assertEqual((p['optimizer_constructions'], p['optimizer_applications'], p['backward_calls']),
                             (1, 1, 12))
            self.assertEqual(p['physical_batch_240'], 'OOM_RETAINED')
            self.assertEqual(len(p['chunk_evidence']), 12)
            self.assertEqual(p['zero_grad']['calls'], [{'set_to_none': False, 'backward_calls_before': 0}])
            self.assertTrue(p['rng_no_redraw'])
            self.assertTrue(all(p['epsilon']['bitwise_equal_to_pinned_reparameterize_draws'].values()))
            self.assertEqual(p['optimizer_state']['step_values'], [1.0])
            self.assertEqual(p['optimizer_state']['entries'], 55)
            self.assertEqual(p['parameter_change']['netCls']['changed_elements'], 0)
            self.assertEqual(p['parameter_change']['netIP']['changed_elements'], 0)
            self.assertFalse(p['benchmark_data_access'] or p['TEST_access'] or p['checkpoint_created']
                             or p['synthetic_bank'] or p['benchmark_training'])
            self.assertEqual(p['firewall']['denied'], [])
        self.assertEqual(sum(p['backward_calls'] for p in self.p), 24)
        self.assertEqual(sum(p['optimizer_applications'] for p in self.p), 2)

    def test_b240_global_statistics_and_lightcnn_paths(self):
        for p in self.p:
            g = p['global_statistics']
            self.assertTrue(all(g['pass2_latents_bitwise_equal_pass1'].values()))
            self.assertEqual(g['pass2_recomputed_sign_mismatches'], 0)
            self.assertAlmostEqual(g['surrogate_sums_pass2']['loss_mmd_surrogate'], g['loss_mmd_global_fp32'], places=4)
            self.assertAlmostEqual(g['surrogate_sums_pass2']['loss_ort_surrogate'], g['loss_ort_global_fp32'], places=4)
            for h in p['chunk_evidence']:
                self.assertFalse(h['nir_fc_requires_grad'] or h['vis_fc_requires_grad'])
                self.assertTrue(h['rec_nir_fc_requires_grad'] and h['rec_vis_fc_requires_grad'])
                self.assertGreater(h['grads']['rec_nir']['nonzero'], 0)
                self.assertEqual(h['grads']['pre_spoof']['nonzero'], 0)

    def test_fp32_environment(self):
        for p in [self.ref] + self.p:
            prec = p['environment_before']['precision']
            self.assertEqual((prec['default_dtype'], prec['matmul_tf32'], prec['cudnn_tf32'], prec['autocast_cuda']),
                             ('torch.float32', False, False, False))
