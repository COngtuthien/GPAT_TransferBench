"""M6D5d E06c GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2: tail-batch contract, formulas and evidence.

Formula tests run methods/dsdg/microbatch_execution_v2.py against the unchanged M6D5b
full-batch transcription on tiny CPU float64 toy networks (the M6D5c toy, imported)
with the pinned misc.util; skipped where Torch is absent. CUDA evidence comes only
from methods/dsdg/microbatch_qualification_v2.py processes.
"""
import inspect
import json
from pathlib import Path
import unittest

from methods.common.config import sha256_file
from methods.common.learned import PreparationError
from methods.dsdg import DSDGAdapter
from methods.dsdg import microbatch_execution as v1
from methods.dsdg import microbatch_execution_v2 as v2
from methods.dsdg import training_graph as tg

ROOT = Path(__file__).resolve().parents[1]
DSDG = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'
AUDIT = ROOT / 'outputs/audit'
OVERLAY = ROOT / v2.RESOLUTION_PATH
REFERENCE = AUDIT / 'M6D5D_E06C_REFERENCE_CHECK.json'
V1COMPAT = AUDIT / 'M6D5D_E06C_V1_COMPATIBILITY.json'
PROCESSES = (AUDIT / 'M6D5D_E06C_B198_PROCESS_1.json', AUDIT / 'M6D5D_E06C_B198_PROCESS_2.json')
REPORT = AUDIT / 'M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION.md'
try:
    import torch
    import torch.nn.functional as F
    from tests.test_m6d5c_e06c_microbatch_execution import Spy, toy
except ImportError:  # laptop: static tests only
    torch = None
needs_torch = unittest.skipIf(torch is None, 'Torch absent; formula tests run in gpat-m6-e06c')


def body(fn):
    return inspect.getsource(fn).split('"""', 2)[-1]


def sizes(B, m=20):
    return [s.stop - s.start for s in v2.chunk_slices_v2(B, m)]


class TestE06cTailBatchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DSDGAdapter()
        cls.overlay = json.loads(OVERLAY.read_text())

    def test_v1_remains_immutable(self):
        for path, h in self.overlay['immutable_inputs_sha256'].items():
            self.assertEqual(sha256_file(ROOT / path), h, path)
        self.assertEqual(v1.chunk_slices(240, 20), v2.chunk_slices_v2(240))
        with self.assertRaises(PreparationError):
            v1.chunk_slices(198, 20)   # V1 unchanged: still refuses the tail batch
        self.assertIs(v2.v1, v1)

    def test_chunk_examples(self):
        self.assertEqual(sizes(240), [20] * 12)
        self.assertEqual(sizes(198), [20] * 9 + [18])
        self.assertEqual(self.overlay['final_batch_chunks'], [20] * 9 + [18])
        self.assertEqual(sizes(21), [20, 1])
        self.assertEqual(sizes(20), [20])
        self.assertEqual(sizes(19), [19])
        self.assertEqual(sizes(7, 3), [3, 3, 1])

    def test_slices_exhaustive_disjoint_ordered_no_padding_no_duplicate(self):
        for B in range(1, 241):
            s = v2.chunk_slices_v2(B)
            idx = [i for c in s for i in range(c.start, c.stop)]
            self.assertEqual(idx, list(range(B)))                      # exhaustive, ordered, no duplicate
            self.assertTrue(all(0 < c.stop - c.start <= 20 for c in s))  # no empty chunk, max 20
            self.assertEqual(s[-1].stop, B)                             # no padding beyond B
            self.assertTrue(all(a.stop == b.start for a, b in zip(s, s[1:])))  # contiguous, disjoint
        for bad in (0, -1, 241, 480, 19.5, True):
            with self.assertRaises(PreparationError):
                v2.chunk_slices_v2(bad)

    def test_epoch_batching_8838(self):
        p = v2.epoch_batch_plan()
        self.assertEqual(8838, 36 * 240 + 198)
        self.assertEqual((p['rows'], p['full_batches'], p['final_batch'], p['optimizer_steps_per_epoch'],
                          p['rows_covered'], p['drop_last']), (8838, 36, 198, 37, 8838, False))
        self.assertEqual(p['batch_sizes'], [240] * 36 + [198])
        o = self.overlay
        self.assertEqual((o['expected_train_rows'], o['full_batches_per_epoch'], o['final_batch_size'],
                          o['optimizer_steps_per_epoch']), (8838, 36, 198, 37))
        self.assertEqual(self.adapter.semantics['training_relation']['rows'], 8838)
        sidecar = json.loads((AUDIT / 'pairs_train_v1.sha256').read_text())
        self.assertEqual((sidecar['rows'], sidecar['sha256']),
                         (8838, self.adapter.config['data']['training_relation_sha256']))

    def test_drop_last_false(self):
        self.assertFalse(self.overlay['drop_last'])
        with self.assertRaises(PreparationError):
            v2.epoch_batch_plan(drop_last=True)
        src = (DSDG / 'train_generator.py').read_text()
        self.assertEqual(v2.verify_pinned_dataloader(src), [])
        self.assertNotIn('drop_last', src)
        self.assertTrue(v2.verify_pinned_dataloader(src.replace('pin_memory=True)', 'pin_memory=True, drop_last=True)')))
        self.assertEqual(self.overlay['final_batch_policy'], 'EXACT_REMAINDER_NO_PADDING_NO_DUPLICATION')
        with self.assertRaises(PreparationError):
            v2.execution_guard(self.overlay, drop_last=True)

    def test_guard_and_fixed_microbatch(self):
        self.assertEqual(v2.execution_guard(self.overlay)['max_microbatch'], 20)
        for micro in (10, 5, 18):
            with self.assertRaisesRegex(PreparationError, 'tuning is forbidden'):
                v2.execution_guard(self.overlay, max_microbatch=micro)
        for flag in v1.FORBIDDEN_EXECUTION + ('padding', 'duplication'):
            with self.assertRaises(PreparationError):
                v2.execution_guard(self.overlay, **{flag: True})
        with self.assertRaises(PreparationError):
            v2.execution_guard(dict(self.overlay, drop_last=True))
        o = self.overlay
        self.assertEqual((o['precision'], o['amp'], o['tf32'], o['activation_checkpointing'],
                          o['cpu_parameter_offload'], o['optimizer_offload']), ('FP32', False, False, False, False, False))

    def test_one_step_per_global_batch_and_zero_grad(self):
        src = body(v2.run_global_batch_v2)
        self.assertEqual(src.count('optimizer.step()'), 1)
        self.assertEqual(src.count('zero_grad('), 1)
        self.assertIn('optimizer.zero_grad(set_to_none=False)', src)
        loop = src.index('for i, s in enumerate(slices)')
        self.assertLess(src.index('zero_grad('), loop)
        self.assertEqual(src[loop:src.index('optimizer.step()')].count('.backward()'), 1)
        self.assertIn('chunk_slices_v2(B, max_microbatch)', src)

    def test_dynamic_epsilon_contract(self):
        e = self.overlay['stochastic_latents']
        self.assertEqual(e['draw_order'], ['cls', 'nir', 'vis'])
        self.assertIn('[B,128]', e['shape'])
        self.assertEqual(v2.EPS_ORDER, ('cls', 'nir', 'vis'))
        self.assertIn('v1.draw_epsilon(torch, B, HDIM, device)', body(v2.draw_epsilon))
        self.assertNotIn('normal_', body(v2.run_global_batch_v2))

    def test_no_val_test_or_benchmark_training(self):
        from methods.dsdg import microbatch_qualification_v2 as q2
        for mod in (v2, q2):
            src = inspect.getsource(mod).split('"""', 2)[2]
            for word in ('parquet', 'faces_256', 'DataLoader(G', 'listdir', 'split_v1', "'VAL'", "'TEST'", 'torch.save('):
                self.assertNotIn(word, src)
        self.assertNotIn('DataLoader(', body(v2.run_global_batch_v2) + inspect.getsource(q2))
        self.assertFalse(self.overlay['train_rows_provenance']['parquet_parsed_in_m6d5d'])
        for item in ('benchmark training', 'VAL access', 'TEST access', 'scientific checkpoint'):
            self.assertIn(item, self.overlay['not_authorized'])
        test = self.adapter.config['data']['splits']['TEST']
        self.assertFalse(test['allowed'] or test['code_path_present'])

    def test_controlled_adaptation_wording(self):
        o = self.overlay
        self.assertEqual((o['fidelity_class'], o['classification'], o['execution_mode']),
                         ('CONTROLLED_ADAPTATION', 'CONTROLLED_EXECUTION_ADAPTATION', 'GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'))
        report = REPORT.read_text()
        for word in ('CONTROLLED_ADAPTATION PRESERVED', 'E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED',
                     'E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED', 'E06c_PRODUCTION_RUNNER_NOT_YET_QUALIFIED'):
            self.assertIn(word, report)
        low = report.lower()
        for word in ('trained e06c', 'e06c was trained', 'faithful dsdg', 'native dsdg', 'official reproduction of',
                     'e06c_physical_batch_240_qualified'):
            self.assertNotIn(word, low)


@needs_torch
class TestE06cTailBatchFormulas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.common.upstream import upstream_modules
        cls.ctx = upstream_modules(DSDG, ('misc.util',), ('misc',))
        cls.util = cls.ctx.__enter__()['misc.util']
        cls.lam = dict(tg.FROZEN_LAMBDAS)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def full_batch(self, B, epoch=1):
        nets, owned, cls, spoof, live, label, eps = toy(batch=B)
        draws, util = iter(v2.EPS_ORDER), self.util

        class U:
            kl_loss, reconstruction_loss, rgb2gray = util.kl_loss, util.reconstruction_loss, util.rgb2gray
            reparameterize = staticmethod(lambda mu, lv: v1.replay_latent(eps[next(draws)], mu, lv))
        out = tg.training_forward(torch, F, U, nets, spoof, live, label, self.lam,
                                  torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())
        loss = tg.total_loss(out, epoch)
        loss.backward()
        return out, loss, [p.grad.clone() for p in owned]

    def v2run(self, B, micro, epoch=1, **kw):
        nets, owned, cls, spoof, live, label, eps = toy(batch=B)
        spy, grads = Spy(owned), []
        run = v2.run_global_batch_v2(torch, F, self.util, nets, spy, spoof, live, label, eps, self.lam,
                                     torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), epoch, max_microbatch=micro,
                                     before_step=lambda: grads.extend(p.grad.clone() for p in owned), **kw)
        return run, grads, spy

    def test_nondivisible_two_pass_equals_full_batch(self):
        for B, micro, expect in ((7, 3, [3, 3, 1]), (11, 4, [4, 4, 3]), (5, 20, [5])):
            for epoch in (1, 2):
                out, loss, ga = self.full_batch(B, epoch)
                run, gb, spy = self.v2run(B, micro, epoch)
                self.assertEqual(run['chunk_sizes'], expect)
                for k in tg.LOSS_TERMS:
                    self.assertAlmostEqual(run['global_losses'][k], out[k].item(), delta=1e-9 * max(1, abs(out[k].item())))
                total = run['epoch1_total'] if epoch == 1 else run['postwarmup_total_algebraic_only']
                self.assertAlmostEqual(total, loss.item(), delta=1e-9 * abs(loss.item()))
                self.assertAlmostEqual(run['sum_of_chunk_objectives'], loss.item(), delta=1e-9 * abs(loss.item()))
                for a, b in zip(ga, gb):
                    torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-12)
                self.assertEqual(spy.calls, [('zero_grad', False), ('step', None)])

    def test_local_weighting_m_over_B(self):
        o = {k: 1.0 for k in v1.LOCAL_TERMS}
        o.update(loss_mmd_surrogate=0.0, loss_ort_surrogate=0.0)
        self.assertAlmostEqual(v1.chunk_objective(o, 18, 2, 198), 18 / 198 * 5)
        self.assertAlmostEqual(v1.chunk_objective(o, 20, 2, 198), 20 / 198 * 5)
        self.assertAlmostEqual(sum(v1.chunk_objective(o, m, 2, 198) for m in sizes(198)), 5.0)

    def test_dynamic_B_mmd_surrogate(self):
        g = torch.Generator().manual_seed(9)
        zn, zv = (torch.randn(198, 128, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        ref = 50 * (zn.mean(0) - zv.mean(0)).abs().mean()
        ref.backward()
        gn = zn.grad.clone()
        zn.grad = zv.grad = None
        sign = torch.sign(zn.detach().mean(0) - zv.detach().mean(0))
        sur = sum(v1.mmd_surrogate(zn[s], zv[s], sign, self.lam, 198) for s in v2.chunk_slices_v2(198))
        sur.backward()
        self.assertAlmostEqual(sur.item(), ref.item(), delta=1e-10 * max(1.0, ref.item()))
        torch.testing.assert_close(zn.grad, gn)
        self.assertIn('lambda_mmd / (B * 128)', self.overlay_text())

    def test_dynamic_B_ort_surrogate(self):
        g = torch.Generator().manual_seed(10)
        zc, zn = (torch.randn(198, 128, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        ref = (zc * zn).sum(1).mean().abs()
        ref.backward()
        gc = zc.grad.clone()
        zc.grad = zn.grad = None
        sign = torch.sign((zc.detach() * zn.detach()).sum(1).mean())
        sur = sum(v1.ort_surrogate(zc[s], zn[s], sign, self.lam, 198) for s in v2.chunk_slices_v2(198))
        sur.backward()
        self.assertAlmostEqual(sur.item(), ref.item(), delta=1e-10 * max(1.0, ref.item()))
        torch.testing.assert_close(zc.grad, gc)

    def overlay_text(self):
        return OVERLAY.read_text()

    def test_dynamic_epsilon_shape_and_order(self):
        state = torch.get_rng_state()
        eps = v2.draw_epsilon(torch, 198, device='cpu')
        torch.set_rng_state(state)
        again = [torch.empty((198, 128)).normal_() for _ in v2.EPS_ORDER]
        self.assertEqual(list(eps), ['cls', 'nir', 'vis'])
        for k, a in zip(v2.EPS_ORDER, again):
            self.assertEqual((list(eps[k].shape), eps[k].dtype), ([198, 128], torch.float32))
            self.assertTrue(torch.equal(eps[k], a))
        with self.assertRaises(PreparationError):   # epsilon must match the actual B, never a truncated 240 draw
            nets, owned, cls, spoof, live, label, eps7 = toy(batch=7)
            eps7 = {k: torch.cat([v, v[:1]]) for k, v in eps7.items()}
            v2.run_global_batch_v2(torch, F, self.util, nets, Spy(owned), spoof, live, label, eps7, self.lam,
                                   torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), 1, max_microbatch=3)

    def test_backward_count_per_chunk(self):
        original, calls = torch.Tensor.backward, []

        def counted(self_, *a, **k):
            calls.append(tuple(self_.shape))
            return original(self_, *a, **k)
        torch.Tensor.backward = counted
        try:
            run, _, spy = self.v2run(7, 3)
        finally:
            torch.Tensor.backward = original
        self.assertEqual((run['chunks'], len(calls)), (3, 3))   # B=198 analogue: 10 chunks -> 10 backward calls
        self.assertEqual(len(v2.chunk_slices_v2(198)), 10)
        self.assertEqual([c[0] for c in spy.calls].count('step'), 1)


class TestE06cTailBatchEvidence(unittest.TestCase):
    """Retained GPU evidence: B=7 reference, V1/V2 B=240 compatibility, two fresh B=198 processes."""
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads(REFERENCE.read_text())
        cls.compat = json.loads(V1COMPAT.read_text())
        cls.p = [json.loads(p.read_text()) for p in PROCESSES]

    def test_reference7(self):
        r = self.ref
        self.assertEqual((r['status'], r['reference_global_batch'], r['reference_max_microbatch']), ('PASS', 7, 3))
        self.assertEqual(r['inputs']['chunk_sizes'], [3, 3, 1])
        self.assertTrue(all(r['gates'].values()))
        self.assertLessEqual(r['comparison']['total_rel_error'], 1e-4)
        self.assertGreaterEqual(r['comparison']['gradient']['cosine'], 0.999)
        self.assertLessEqual(r['comparison']['gradient']['relative_l2'], 1e-2)

    def test_v1_compatibility_b240(self):
        c = self.compat
        self.assertEqual(c['status'], 'PASS')
        self.assertTrue(all(c['gates'].values()))
        self.assertEqual(c['comparison']['chunk_sizes'], {'v1': [20] * 12, 'v2': [20] * 12})
        self.assertEqual(c['comparison']['backward_calls'], {'v1': 12, 'v2': 12})

    def test_b198_processes(self):
        for p in self.p:
            self.assertEqual((p['status'], p['mode'], p['diagnostic_seed']), ('PASS', 'b198', 60504))
            self.assertEqual(p['chunk_sizes'], [20] * 9 + [18])
            self.assertEqual((p['global_batch_size'], p['final_chunk'], p['drop_last']), (198, 18, False))
            self.assertEqual((p['optimizer_constructions'], p['optimizer_applications'], p['backward_calls']), (1, 1, 10))
            self.assertEqual(p['epsilon']['shape'], [198, 128])
            self.assertEqual(p['inputs']['distinct_rows'], 396)
            self.assertEqual(p['zero_grad']['calls'], [{'set_to_none': False, 'backward_calls_before': 0}])
            self.assertEqual(sum(p['gradient_inventory'][n]['non_none'] for n in tg.OPTIMIZER_OWNED), 55)
            self.assertEqual((p['optimizer_state']['entries'], p['optimizer_state']['step_values']), (55, [1.0]))
            self.assertEqual(p['parameter_change']['netCls']['changed_elements'], 0)
            self.assertEqual(p['parameter_change']['netIP']['changed_elements'], 0)
            self.assertAlmostEqual(sum(p['losses']['local_weights']), 1.0)
            self.assertFalse(p['benchmark_data_access'] or p['TEST_access'] or p['VAL_access'] or
                             p['checkpoint_created'] or p['synthetic_bank'] or p['benchmark_training'])
            self.assertEqual(p['firewall']['denied'], [])
        self.assertEqual(sum(p['backward_calls'] for p in self.p), 20)
