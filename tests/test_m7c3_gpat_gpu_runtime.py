"""M7C3: GPAT GPU runtime qualification + R-04 Level-2 teacher parity.

Two clearly separated tiers:
  StaticM7C3 (laptop / any host, stdlib + json only): environment lock, evidence gates, R-04 attempt 1 (FAIL, kept)
      and attempt 2 (exact forward), floors, TRAIN-only access, repeatability, protected authority, no runner.
      Pinned to the commit that ADDED this test.
  LiveCPUExactForwardM7C3 (gpat-m7-cpu; torch + PIL): exact-forward FaceXFormer adapter semantics against the PIL
      oracle and the surrogate backward.
  LiveGPUM7C3 (GPU host, gpat-m7-gpu, CUDA required; skips visibly elsewhere): re-runs fast synthetic GPU checks.
No dataset image is read by these tests.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '33955b05ff70289c42386c3ac1623bae4a5fae5e'
THIS = 'tests/test_m7c3_gpat_gpu_runtime.py'
EV_GPU = 'outputs/audit/M7C3_GPAT_GPU_QUALIFICATION.json'
EV_REP = 'outputs/audit/M7C3_GPAT_GPU_REPEATABILITY.json'
EV_R04 = 'outputs/audit/M7C3_R04_LEVEL2_TEACHER_PARITY.json'
EV_R04_A1 = 'outputs/audit/M7C3_R04_LEVEL2_ATTEMPT1.json'
SELECTION_SHA = 'd02a8a96eedd18b067fd4c7bf809368ced68b7aff14c88daac053b999baa855f'
LOCK = 'environments/gpat_m7_gpu.lock.json'
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7c3_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7c3(rel):
    commit = m7c3_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def load_tool(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cuda_available():
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('ptwt') is None:
        return False
    import torch
    return torch.cuda.is_available()


class StaticM7C3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = load_tool('m7c3_preflight', 'tools/m7c3_gpat_gpu_runtime_preflight.py')
        cls.gpu = json.loads(at_m7c3(EV_GPU))
        cls.r04 = json.loads(at_m7c3(EV_R04))
        cls.a1 = json.loads(at_m7c3(EV_R04_A1))
        cls.rep = json.loads(at_m7c3(EV_REP))

    def test_01_environment_exact_delta(self):
        lock = self.pf.check_environment(at_m7c3)
        self.assertEqual(lock['freeze_diff_vs_base'], {'added': ['PyWavelets==1.9.0', 'ptwt==1.0.1'], 'removed': [],
                                                      'changed': []})
        self.assertIs(lock['existing_environments_mutated'], False)
        self.assertIs(lock['protected_environment_fingerprints_identical'], True)
        self.assertEqual(lock['versions']['pywt___version__'], '1.8.0')                 # M7C2A-OBS-02 reproduces
        self.assertEqual(at_m7c3('environments/gpat_m7_gpu.pip-freeze.txt'),
                         at_authority('environments/gpat_m7_cpu.pip-freeze.txt'))      # same pip layer as CPU env
        self.assertEqual(at_m7c3('environments/gpat_m7_cpu.lock.json'), at_authority('environments/gpat_m7_cpu.lock.json'))

    def test_02_protected_authority_unchanged(self):
        self.pf.check_protected(at_m7c3, at_authority)

    def test_03_naf_and_determinism(self):
        naf = self.gpu['naf']
        self.assertEqual(naf['commit'], '2b4af71ebe098a92a75910c233a3965a3e93ede4')
        self.assertTrue(naf['saved_variables_works'] and naf['pass'])
        self.assertFalse(naf['basicsr_imported'] or naf['lmdb_imported'])
        for mode in ('fp32', 'fp16_autocast'):
            b = naf['nafblock'][mode]
            self.assertTrue(b['finite_forward'] and b['finite_backward'] and b['shape_preserved'], mode)
        det = self.gpu['determinism']['observed']
        self.assertEqual((det['CUBLAS_WORKSPACE_CONFIG'], det['cudnn_benchmark'], det['cudnn_deterministic'],
                          det['tf32_matmul'], det['tf32_cudnn'], det['use_deterministic_algorithms'],
                          det['deterministic_warn_only']), (':4096:8', False, True, False, False, True, False))
        self.assertTrue(self.gpu['determinism_after']['use_deterministic_algorithms'])
        bil = self.gpu['bilinear_matrix_parity']
        self.assertLessEqual(bil['float32']['forward_max_abs'], 1e-6)
        self.assertIs(bil['adopted'], False)

    def test_04_gpu_gates(self):
        self.pf.check_gpu_evidence(at_m7c3)
        g = self.gpu
        self.assertTrue(all(g['gates'].values()))
        self.assertLess(g['d11_d12']['D11_max_abs'], 1e-5)
        for v in g['forward_n4'].values():
            self.assertEqual(v['shapes']['x_hat'], [4, 3, 256, 256])
            self.assertEqual(v['D_logits_shape'], [4, 1, 30, 30])
            self.assertEqual(v['dtypes']['x_hat'], 'float32')
        self.assertEqual(g['forward_n4']['B3']['attack_logits']['shape'], [4, 6])
        self.assertEqual(g['forward_n4']['B3']['identity_logits']['shape'], [4, 60])
        self.assertIsNone(g['forward_n4']['B0']['attack_logits'])

    def test_05_amp_boundaries(self):
        s = self.gpu['optimizer_group_smoke']
        self.assertEqual(set(s['amp_module_output_dtypes']), {'e_art', 'g_res', 'discriminator', 'attack_head',
                                                              'identity_head'})
        self.assertTrue(all(v == ['float16'] for v in s['amp_module_output_dtypes'].values()))
        fb = s['amp_function_boundaries']
        for name in ('wavelet.dwt', 'wavelet.idwt', 'composition.activate', 'composition.compose',
                     'composition.artifact_map', 'spectral.s_radial', 'spectral.s_orient', 'losses.l_low',
                     'losses.l_tv', 'losses.l_budget', 'losses.l_type', 'losses.idadv_microbatch', 'losses.l_parse',
                     'losses.l_gadv', 'losses.l_d', 'teacher.facexformer_input', 'teacher.adaface_input'):
            self.assertEqual(fb[name]['outputs'], ['float32'], name)
        self.assertEqual(fb['composition.activate']['inputs'], ['float16'])               # crosses the boundary
        self.assertTrue(self.gpu['teacher_adapter_autocast']['pass'])

    def test_06_optimizer_group_smoke(self):
        s = self.gpu['optimizer_group_smoke']
        self.assertEqual(len(s['microbatches']), 2)
        self.assertEqual(s['weights'], [0.5, 0.5])
        for m in s['microbatches']:
            self.assertTrue(m['finite'] and m['D_grads_unchanged_by_G_backward'] and m['G_grad_norm_unchanged_by_D_backward'])
        self.assertTrue(s['D']['clip_after_unscale'] and s['G']['clip_after_unscale'])
        self.assertTrue(s['params_finite_after_step'] and all(s['params_changed'].values()))
        self.assertEqual(s['optimizer_steps'], {'G': 1, 'D': 1})
        self.assertEqual(s['checkpoint_writes'], 0)
        demo = self.gpu['scaler_clip_demo']
        self.assertTrue(demo['clip_binds_after_unscale'] and demo['scalers_independent'])
        self.assertAlmostEqual(demo['G_norm_after_clip'], 1.0, places=3)
        self.assertEqual(demo['D_scale'][1], demo['D_scale'][0] / 2)
        self.assertEqual(demo['G_scale'][1], demo['G_scale'][0])

    def test_07_dev022_and_grl(self):
        d = self.gpu['dev022']
        self.assertTrue(d['pass'])
        self.assertLessEqual(d['abs_diff'], 1e-5)
        self.assertTrue(d['differs_from_sample_weighted'])
        self.assertEqual(d['zero_labelled_value'], 0.0)
        self.assertTrue(self.gpu['grl']['pass'])

    def test_08_vram_batch4(self):
        v = self.gpu['vram']
        self.assertEqual(v['physical_batch'], 4)
        self.assertIs(v['oom'], False)
        for k in ('A_B0_forward_N4', 'B_B3_forward_N4', 'C_B3_one_group_backward_and_step'):
            self.assertLess(v[k]['max_memory_reserved_GiB'], 24.0, k)
        m = self.r04['teacher_gradient']['memory']
        self.assertEqual(m['physical_batch'], 4)
        self.assertIs(m['oom'], False)

    def test_09_repeatability(self):
        self.assertIs(self.rep['bitwise_identical'], True)
        self.assertEqual(self.rep['process_1_sha256'], self.rep['process_2_sha256'])
        self.assertEqual(set(self.rep['digests']['core']), {'fp32', 'amp_fp16'})
        self.assertGreaterEqual(len(self.rep['digests']['ops']), 12)

    def test_10_teacher_hashes_and_train_only(self):
        r = self.pf.check_r04(at_m7c3)
        self.assertTrue(r['inventory']['all_match'])
        sel = self.a1['selection']
        self.assertEqual(sel['selected_list_sha256'], SELECTION_SHA)
        self.assertEqual(sel['manifest_sha256']['manifests/split_v1.parquet'],
                         'fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d')
        ids = [s for v in sel['selected_ids'].values() for s in v]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({k: len(v) for k, v in sel['selected_ids'].items()}, {'casia_fasd': 64, 'msu_mfsd': 64, 'siwmv2': 64})
        for ds, v in sel['selected_ids'].items():                                  # smallest SHA256 digests, ascending
            digests = [hashlib.sha256(x.encode('utf-8')).hexdigest() for x in v]
            self.assertEqual(digests, sorted(digests), ds)
        for ev in (self.a1, r):
            self.assertEqual(ev['access']['images_opened'], 192)
            self.assertTrue(ev['access']['all_train'] and ev['access']['opened_equals_selection'])
            self.assertEqual((ev['access']['val_images_opened'], ev['access']['test_images_opened']), (0, 0))

    def test_11_attempt1_preserved_and_floors_immutable(self):
        self.assertEqual(hashlib.sha256(at_m7c3(EV_R04_A1)).hexdigest(), self.pf.EV_R04_A1_SHA)
        a1, s1 = self.a1['level2'], self.r04['attempt_1']
        self.assertEqual(self.a1['status'], 'FAIL')
        self.assertEqual((s1['result'], s1['failed_gate'], s1['adapter']),
                         ('FAIL', 'landmark_abs_max', 'CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY'))
        self.assertGreater(a1['candidate']['landmark_abs'], a1['thresholds_noise_floor']['landmark_abs'])
        self.assertEqual(s1['floor'], a1['thresholds_noise_floor'])
        self.assertEqual(s1['candidate'], a1['candidate'])
        a2 = self.r04['attempt_2']
        self.assertEqual(a2['floors_from_attempt_1'], a1['thresholds_noise_floor'])    # never recomputed
        self.assertIs(a2['floors_recomputed'], False)
        rows = a1['per_sample']                                                      # floors recomputable from attempt 1
        self.assertEqual(a1['thresholds_noise_floor']['landmark_abs'], max(x['lm_max'] for r in rows for x in r['fx_replicas']))
        self.assertEqual(a1['thresholds_noise_floor']['adaface_cos'], min(c for r in rows for c in r['ada_cos_replicas']))

    def test_12_same_selection_and_replica_seeds(self):
        from methods.gpat import teacher_preprocess as tp
        a2 = self.r04['attempt_2']
        self.assertEqual(a2['selection_sha256'], SELECTION_SHA)
        seeds = [tp.level2_seed(r['dataset'], r['sample_id'], k) for r in self.a1['selection']['records'] for k in range(8)]
        canon = json.dumps(seeds, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        self.assertEqual(hashlib.sha256(canon).hexdigest(), a2['replica_seed_list_sha256'])
        self.assertEqual(self.a1['level2']['replicas_per_image'], 8)

    def test_15_attempt2_exact_forward_and_gates(self):
        r, a2 = self.r04, self.r04['attempt_2']
        self.assertEqual(a2['adapter'], 'EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY')
        self.assertEqual((r['level1_exact_gpu']['images'], r['level1_exact_gpu']['uint8_bitwise_equal']), (44, 44))
        self.assertEqual((a2['input_parity']['images'], a2['input_parity']['uint8_bitwise_equal']), (192, 192))
        f, c = a2['floors_from_attempt_1'], a2['candidate']
        self.assertGreaterEqual(c['adaface_cos'], f['adaface_cos'])
        self.assertLessEqual(c['landmark_mean'], f['landmark_mean'])
        self.assertLessEqual(c['landmark_abs'], f['landmark_abs'])
        self.assertGreaterEqual(c['parsing_argmax'], f['parsing_argmax'])
        self.assertGreaterEqual(c['parsing_dice'], f['parsing_dice'])
        self.assertTrue(a2['pass'] and all(a2['gates'].values()))
        self.assertEqual(r['status'], 'PASS')
        sb = r['surrogate_backward']
        self.assertTrue(sb['bitwise_equal'] and sb['finite'] and sb['nonzero'])
        self.assertTrue(r['backward_repeatability']['bitwise_identical'])
        self.assertTrue(r['laptop_data_partition']['unmounted'])
        self.pf.check_teacher_preprocess(at_m7c3, at_authority)

    def test_12b_teacher_gradient_structure(self):
        t = self.r04['teacher_gradient']
        for name in ('facexformer', 'adaface'):
            self.assertTrue(t[name]['x_grad_finite'] and t[name]['x_grad_abs_sum'] > 0, name)
            self.assertTrue(t[name]['teacher_param_grads_none'] and not t[name]['teacher_requires_grad_any'], name)
        self.assertIs(t['x_t_targets_requires_grad'], False)
        self.assertTrue(t['integrated_teacher_param_grads_none'] and t['integrated_loss_finite'])

    def test_16_runtime_resolution_record(self):
        rec = self.pf.check_record(at_m7c3)
        r4 = rec['resolutions']['R04_LEVEL2']
        a1f = self.a1['level2']['thresholds_noise_floor']
        for key in ('adaface_cos', 'landmark_mean', 'landmark_abs', 'parsing_argmax', 'parsing_dice'):   # full precision
            self.assertEqual(r4['floors_from_attempt_1'][key], a1f[key], key)
            self.assertEqual(self.r04['attempt_2']['floors_from_attempt_1'][key], a1f[key], key)
        self.assertEqual(a1f['landmark_abs'], 0.013857558369636536)
        self.assertEqual(self.a1['level2']['candidate']['landmark_abs'], 0.01423092931509018)
        self.assertEqual(r4['selected_list_sha256'], SELECTION_SHA)
        self.assertEqual(rec['resolutions']['FACEXFORMER_TEACHER_ADAPTER']['name'],
                         'EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY')
        self.assertIsNone(rec['new_deviation'])
        self.assertEqual(rec['record_kind'], 'ADDITIVE_RUNTIME_QUALIFICATION_RECORD')

    def test_13_no_runner_no_checkpoint_no_bank(self):
        self.pf.check_no_runner(at_m7c3)
        for rel in ('tools/m7c3_gpat_gpu_qualification.py', 'tools/m7c3_r04_level2_parity.py'):
            src = at_m7c3(rel).decode()
            self.assertNotIn('torch.save', src)
            self.assertNotIn('val_pairs', src)
        self.assertEqual((self.gpu['checkpoint_writes'], self.gpu['bank_writes'], self.gpu['training_runs']), (0, 0, 0))

    def test_14_preflight_rejections(self):
        pf = self.pf
        for mutate in (lambda e: e['gates'].__setitem__('D11', False),
                       lambda e: e['vram'].__setitem__('physical_batch', 2),
                       lambda e: e['determinism']['observed'].__setitem__('deterministic_warn_only', True)):
            bad = copy.deepcopy(self.gpu)
            mutate(bad)
            raw = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                pf.check_gpu_evidence(lambda rel, raw=raw: raw if rel == EV_GPU else at_m7c3(rel))
        for mutate in (lambda e: e['attempt_2']['candidate'].__setitem__('landmark_abs', 0.02),
                       lambda e: e['attempt_2']['floors_from_attempt_1'].__setitem__('landmark_abs', 0.02),
                       lambda e: e['attempt_2']['input_parity'].__setitem__('uint8_bitwise_equal', 191),
                       lambda e: e['access'].__setitem__('val_images_opened', 1),
                       lambda e: e['inventory']['adaface_weight'].__setitem__('sha256', '0' * 64)):
            bad = copy.deepcopy(self.r04)
            mutate(bad)
            raw = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                pf.check_r04(lambda rel, raw=raw: raw if rel == EV_R04 else at_m7c3(rel))


def cpu_exact_stack():
    return all(importlib.util.find_spec(m) is not None for m in ('torch', 'torchvision', 'PIL'))


@unittest.skipUnless(cpu_exact_stack(), 'exact-forward tier needs torch + torchvision + PIL (gpat-m7-cpu)')
class LiveCPUExactForwardM7C3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from methods.gpat import teacher_preprocess as tp
        cls.np, cls.torch, cls.Image, cls.tp = np, torch, Image, tp
        cls.tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                            T.Normalize(tp.IMAGENET_MEAN, tp.IMAGENET_STD)])
        q = load_tool('m7c2a_q', 'tools/m7c2a_gpat_cpu_qualification.py')
        cls.corpus = [np.ascontiguousarray(v) for v in q.corpus().values()]

    def x_of(self, rgb):
        return self.torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 127.5 - 1.0

    def pil_u8(self, rgb):
        return self.np.array(self.Image.fromarray(rgb).resize((224, 224), self.Image.BICUBIC))

    def exact_u8(self, rgb):
        return self.tp.facexformer_uint8_exact(self.x_of(rgb))[0].permute(1, 2, 0).numpy().astype(self.np.uint8)

    def test_30_quantization_semantics(self):
        torch, tp = self.torch, self.tp
        u = torch.arange(256, dtype=torch.float32)
        self.assertTrue(torch.equal(tp.teacher_uint8((u / 127.5 - 1.0).view(1, 1, 1, 256)).flatten(), u.double()))
        x = torch.tensor([-1.5, -1.0, 1.0, 1.7]).view(1, 1, 1, 4)
        self.assertEqual(tp.teacher_uint8(x).flatten().tolist(), [0.0, 0.0, 255.0, 255.0])          # clip
        self.assertEqual(torch.round(torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)).tolist(), [0.0, 2.0, 2.0])
        xs = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(9)) * 2.4 - 1.2
        v = (xs.double() + 1.0) * 0.5
        self.assertTrue(torch.equal(tp.teacher_uint8(xs), torch.round(v * 255.0).clamp(0.0, 255.0)))  # exact rule

    def test_31_fixed_point_coefficients(self):
        np, tp = self.np, self.tp
        k = tp.pil_bicubic_fixed_point()
        self.assertEqual(k.shape, (224, 256))
        self.assertTrue(set(k.sum(1).tolist()) <= {4194304, 4194305})            # 2^22 up to coefficient rounding
        self.assertTrue((k < 0).any())                                           # negative bicubic lobes kept
        w = tp.pil_bicubic_matrix()                                              # M7C2a float64 coefficients
        q = np.where(w < 0, np.trunc(-0.5 + w * 2 ** 22), np.trunc(0.5 + w * 2 ** 22)).astype(np.int64)
        self.assertLessEqual(int(np.abs(q - k).max()), 1)                         # same coefficients up to 1 LSB

    def test_32_exact_bytes_corpus_and_random(self):
        np = self.np
        self.assertEqual(sum(np.array_equal(self.exact_u8(r), self.pil_u8(r)) for r in self.corpus), 44)
        rng = np.random.default_rng(7)
        rand = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(200)]
        self.assertEqual(sum(np.array_equal(self.exact_u8(r), self.pil_u8(r)) for r in rand), 200)

    def test_33_pass_order_and_intermediate_saturation_matter(self):
        """Non-vacuity: vertical-first or no intermediate clip8 would NOT reproduce PIL on saturating content."""
        np, torch, tp = self.np, self.torch, self.tp
        k = torch.as_tensor(tp.pil_bicubic_fixed_point(), dtype=torch.float64)
        half = float(1 << 21)
        bad_order = bad_clip = 0
        for rgb in self.corpus:
            u8 = tp.teacher_uint8(self.x_of(rgb))
            v_first = tp._clip8_pass(half + torch.einsum('nchp,wp->nchw', tp._clip8_pass(half + torch.einsum('oh,nchw->ncow', k, u8)), k))
            h_float = (half + torch.einsum('nchw,pw->nchp', u8, k)) / 2 ** 22             # no rounding/clip8 in between
            no_mid = tp._clip8_pass(half + torch.einsum('oh,nchp->ncop', k, h_float))
            ref = torch.from_numpy(self.pil_u8(rgb)).permute(2, 0, 1)[None].double()
            bad_order += not torch.equal(v_first, ref)
            bad_clip += not torch.equal(no_mid, ref)
        self.assertGreater(bad_order, 0)
        self.assertGreater(bad_clip, 0)

    def test_34_normalized_tensor_parity(self):
        torch = self.torch
        for rgb in self.corpus[:12]:
            a = self.tp.facexformer_input_exact(self.x_of(rgb))[0]
            b = self.tf(self.Image.fromarray(rgb))
            self.assertTrue(torch.equal(a, b))

    def test_35_surrogate_backward_equals_approved_vjp(self):
        torch, tp = self.torch, self.tp
        g = torch.Generator().manual_seed(3)
        x = torch.rand(2, 3, 256, 256, generator=g) * 1.6 - 0.8
        up = torch.randn(2, 3, 224, 224, generator=g)
        a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        (tp.facexformer_input_exact(a) * up).sum().backward()
        (tp.facexformer_input(b) * up).sum().backward()
        self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertTrue(torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0)
        teacher = torch.nn.Conv2d(3, 4, 3).requires_grad_(False)                 # frozen stand-in teacher
        c = x.clone().requires_grad_(True)
        teacher(tp.facexformer_input_exact(c)).square().mean().backward()
        self.assertIsNone(teacher.weight.grad)
        self.assertTrue(torch.isfinite(c.grad).all() and c.grad.abs().sum() > 0)
        out = tp.facexformer_input_exact(c)
        self.assertTrue(out.requires_grad and out.dtype == torch.float32)        # x_hat never detached

    def test_36_forward_ignores_old_approximation(self):
        torch, tp = self.torch, self.tp
        rgb = self.corpus[-1]
        x = self.x_of(rgb)
        exact, approx = tp.facexformer_input_exact(x), tp.facexformer_input(x)
        self.assertFalse(torch.equal(exact, approx))                             # no 1.13-LSB approximation in forward
        self.assertTrue(torch.equal(exact, self.tf(self.Image.fromarray(rgb))[None]))


@unittest.skipUnless(cuda_available(), 'GPU tier: needs CUDA + gpat-m7-gpu (runs on the RTX 3090 host only)')
class LiveGPUM7C3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = load_tool('m7c3_harness', 'tools/m7c3_gpat_gpu_qualification.py')
        cls.q.setup()

    def test_20_counts_and_naf(self):
        self.assertTrue(self.q.static_authority()['counts_match'])
        self.assertTrue(self.q.naf_gpu()['pass'])

    def test_21_forward_b0_b3(self):
        b0, b3 = self.q.forward_n4('B0'), self.q.forward_n4('B3')
        self.assertTrue(b0['finite'] and b3['finite'])
        self.assertEqual(b0['shapes'], b3['shapes'])

    def test_22_d11_d12(self):
        r = self.q.d11_d12()
        self.assertTrue(r['D11_pass'] and r['D12_pass'])

    def test_23_dev022_grl_scalers(self):
        self.assertTrue(self.q.dev022_gpu()['pass'])
        self.assertTrue(self.q.grl_gpu()['pass'])
        self.assertTrue(self.q.scaler_clip_demo()['pass'])

    def test_24_repeatable_within_process(self):
        a, b = self.q.op_digests(), self.q.op_digests()
        self.assertEqual(a, b)

    def test_25_determinism_still_enabled(self):
        import torch
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


if __name__ == '__main__':
    unittest.main()
