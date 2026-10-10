"""M7C2b: GPAT static core implementation + synthetic CPU qualification.

Static tests need only the stdlib, PyYAML and NumPy. Live tests need torch + ptwt (run them in gpat-m7-cpu); they
skip, visibly, anywhere else. History assertions read the state at the commit that ADDED this test (candidate: the
worktree), so later milestones never break them. Synthetic tensors only: no GPU, no dataset sample, no teacher
weights, no training, no checkpoint.
"""
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tokenize
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = 'c9a12e10eff959a31aaa361cbff98469aae1e7c2'
THIS = 'tests/test_m7c2b_gpat_static_core.py'
EVIDENCE = 'outputs/audit/M7C2B_GPAT_STATIC_CORE.json'
RESOLUTION = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
M7C2A_CODE = ('methods/gpat/naf_source.py', 'methods/gpat/runtime_contract.py', 'methods/gpat/teacher_preprocess.py')
PACKAGE_INIT = 'methods/gpat/__init__.py'          # M7C2b: docstring-only current-state update (owner review)
STATIC_CORE = ('artifact_encoder.py', 'batching.py', 'composition.py', 'config.py', 'discriminator.py', 'ema.py',
               'generator.py', 'grl.py', 'heads.py', 'highpass.py', 'identity_labels.py', 'losses.py', 'model.py',
               'schedule.py', 'spectral.py', 'wavelet.py')
GPAT_DIR = sorted(['.gitkeep', '__init__.py', 'naf_source.py', 'runtime_contract.py', 'teacher_preprocess.py',
                   *STATIC_CORE])
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7c2b_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7c2b(rel):
    commit = m7c2b_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def listdir_at_m7c2b(rel):
    commit = m7c2b_commit()
    if commit:
        return sorted(Path(p).name for p in git('ls-tree', '--name-only', f'{commit}:{rel}/').stdout.decode().split())
    return sorted(p.name for p in (ROOT / rel).iterdir() if p.name != '__pycache__')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


LIVE = all(importlib.util.find_spec(m) is not None for m in ('torch', 'torchvision', 'ptwt', 'pywt'))


def load_tool(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_tokens(raw: bytes):
    """(names, attribute names, strings) of a Python source, comments dropped (docstrings are strings)."""
    names, attrs, strings, prev = set(), set(), [], None
    for tok in tokenize.tokenize(io.BytesIO(raw).readline):
        if tok.type == tokenize.NAME:
            (attrs if prev == '.' else names).add(tok.string)
        elif tok.type == tokenize.STRING:
            strings.append(tok.string)
        prev = tok.string if tok.type == tokenize.OP else None
    return names, attrs, strings


class StaticM7C2B(unittest.TestCase):
    # ------------------------------------------------------------------ 1. config hash verification
    def test_01_config_loader_hash_verification(self):
        from methods.gpat import config as gc
        expected = {'B0': (False, False, 0.0, 0.0), 'B1': (True, False, 0.2, 0.0), 'B2': (False, True, 0.0, 0.1),
                    'B3': (True, True, 0.2, 0.1)}
        reader = lambda rel: at_m7c2b(rel)                                        # noqa: E731
        for v, (atk, ida, lt, li) in expected.items():
            for name in (v, gc.VARIANTS[v][0], gc.VARIANTS[v][1]):
                cfg = gc.load_config(name, reader=reader)
                self.assertEqual((cfg.variant, cfg.attack_type_head, cfg.identity_adversary, cfg.lambda_type,
                                  cfg.lambda_idadv), (v, atk, ida, lt, li))
                self.assertEqual(cfg.sha256, CONFIG_SHA[cfg.path])
                self.assertEqual((cfg.gamma, cfg.delta_scale_hf, cfg.delta_scale_ll), (0.0, 0.15, 0.05))
        cfg = gc.load_config('B3', reader=reader)
        with self.assertRaises(TypeError):
            cfg.raw['loss']['lambda_id'] = 2.0                                    # immutable mapping
        with self.assertRaises(Exception):
            cfg.lambda_type = 0.0                                                 # frozen dataclass
        self.assertIsInstance(cfg.raw['residual_generator']['enc_blocks'], tuple)
        for bad in ('B4', 'E12', 'GPAT-B5', 'b0', ''):
            with self.assertRaises(gc.GPATConfigError):
                gc.load_config(bad, reader=reader)
        b1 = 'configs/methods/gpat_b1.yaml'
        tampered = at_m7c2b(b1).replace(b'lambda_type: 0.2', b'lambda_type: 0.3')
        with self.assertRaises(gc.GPATConfigError):
            gc.load_config('B1', reader=lambda rel: tampered if rel == b1 else at_m7c2b(rel))
        with self.assertRaises(gc.GPATConfigError):                               # snapshot byte equality
            gc.load_config('B1', reader=lambda rel: b'x' if rel.startswith('frozen_config_snapshot/') else at_m7c2b(rel))
        gc.load_config('B1', verify_snapshot=False,
                       reader=lambda rel: b'x' if rel.startswith('frozen_config_snapshot/') else at_m7c2b(rel))
        with self.assertRaises(gc.GPATConfigError):                               # authority record changed
            gc.load_config('B0', reader=lambda rel: at_m7c2b(rel) + b' ' if rel == RESOLUTION else at_m7c2b(rel))
        self.assertEqual(gc.AUTHORITY_SHA256[RESOLUTION],
                         'b61555a2ca15954bdd8929d92c28d0d957561c595ea38eeef496d06c871e8e70')
        self.assertEqual({p: d for _, _, p, d in gc.VARIANTS.values()}, CONFIG_SHA)

    # ------------------------------------------------------------------ 25. B0-B3 / authority unchanged
    def test_02_authority_unchanged(self):
        protected = ('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
                     'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
                     'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
                     'configs/amendments/gpat_m7b_owner_clarifications.yaml', RESOLUTION,
                     'environments/gpat_m7_cpu.lock.json', 'outputs/audit/method_status.csv',
                     'outputs/audit/deviation_report.md', 'outputs/audit/STAGE_STATE.json',
                     'outputs/audit/M7B_GPAT_CONFIG_FREEZE.json', 'third_party/source_pins.json', 'models/registry.yaml',
                     'gpatbench/preprocess/aux_models.py', 'gpatbench/probe/preprocess.py', *M7C2A_CODE)
        for rel in (*protected, *CONFIG_SHA, *('frozen_config_snapshot/' + p for p in CONFIG_SHA)):
            self.assertEqual(at_m7c2b(rel), at_authority(rel), rel)
        for rel, digest in CONFIG_SHA.items():
            self.assertEqual(sha(at_m7c2b(rel)), digest, rel)
            self.assertEqual(sha(at_m7c2b('frozen_config_snapshot/' + rel)), digest, rel)
        self.assertEqual(sha(at_m7c2b(RESOLUTION)), 'b61555a2ca15954bdd8929d92c28d0d957561c595ea38eeef496d06c871e8e70')
        self.assertEqual(sha(at_m7c2b('environments/gpat_m7_cpu.lock.json')),
                         'd52b5e7ccc8ee0f6c289db9f31d4521e4b241a5e22664912a6185e073be9c23a')
        for i in range(1, 11):
            for p in (ROOT / 'docs/spec/amendments').glob(f'*Amendment_A{i}_*'):
                rel = str(p.relative_to(ROOT))
                self.assertEqual(at_m7c2b(rel), at_authority(rel), rel)
        rec = json.loads(at_m7c2b(RESOLUTION))
        self.assertEqual(list(rec['remaining_deferred']), ['R-04_LEVEL2'])           # R-04 Level 2 still deferred

    # ------------------------------------------------------------------ 38/39/40 scope, teacher adapters, no VAL/TEST/GPU
    def test_03_static_core_scope(self):
        names = listdir_at_m7c2b('methods/gpat')
        self.assertEqual(names, GPAT_DIR)
        for forbidden in ('train.py', 'training.py', 'runner.py', 'trainer.py', 'checkpoint.py', 'bank.py'):
            self.assertNotIn(forbidden, names)

    def test_04_no_training_val_test_or_gpu_path(self):
        forbidden_attrs = {'backward', 'zero_grad', 'step', 'save', 'cuda', 'read_parquet', 'load_state_dict_from_url',
                           'download', 'optim', 'data'}
        forbidden_names = {'GradScaler', 'DataLoader', 'Adam', 'AdamW', 'SGD', 'optim', 'cuda', 'pandas', 'pyarrow',
                           'PIL', 'cv2', 'open', 'requests', 'urllib', 'download'}
        forbidden_strings = ('parquet', 'manifests/', 'faces_256', 'val_pairs', 'test_pairs', 'runs/', 'cuda:', "'cuda'")
        for name in STATIC_CORE:
            raw = at_m7c2b('methods/gpat/' + name)
            names, attrs, strings = code_tokens(raw)
            self.assertFalse(names & forbidden_names, f'{name}: {sorted(names & forbidden_names)}')
            self.assertFalse(attrs & forbidden_attrs, f'{name}: {sorted(attrs & forbidden_attrs)}')
            for s in strings:
                for f in forbidden_strings:
                    self.assertNotIn(f, s, name)
        # the only non-config file read is the SHA-256 verified local IMAGENET1K_V1 weight; configs are read by loader
        enc = at_m7c2b('methods/gpat/artifact_encoder.py').decode()
        self.assertIn('weights_only=True', enc)
        self.assertIn('torchvision.models.resnet18(weights=None)', enc)

    def test_07_package_init_docstring_only(self):
        import ast
        old, new = ast.parse(at_authority(PACKAGE_INIT)), ast.parse(at_m7c2b(PACKAGE_INIT))
        for tree in (old, new):                                                 # no code, only the module docstring
            self.assertEqual(len(tree.body), 1)
            self.assertIsInstance(tree.body[0].value, ast.Constant)
        self.assertIn('not implemented here', ast.get_docstring(old))           # M7C2a history stays verifiable
        doc = ast.get_docstring(new)
        for phrase in ('static GPAT core is implemented', 'synthetic CPU', 'GPU runtime qualification',
                       'no production training runner'):
            self.assertIn(phrase, doc)

    def test_08_l_low_and_lferr_are_distinct_contracts(self):
        import ast
        tree = ast.parse(at_m7c2b('methods/gpat/losses.py'))
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def calls(fn):
            return {ast.unparse(c.func) for c in ast.walk(funcs[fn]) if isinstance(c, ast.Call)}

        def names(fn):
            return {n.id for n in ast.walk(funcs[fn]) if isinstance(n, ast.Name)} | \
                   {n.attr for n in ast.walk(funcs[fn]) if isinstance(n, ast.Attribute)}

        self.assertEqual([a.arg for a in funcs['l_low'].args.args], ['x_hat', 'll_target'])
        self.assertIn('wavelet.dwt', calls('l_low'))                            # literal re-DWT of x_hat
        self.assertFalse({'lferr_internal', 'lferr_selection', 'rc', 'LL_syn'} & names('l_low'))
        self.assertIn('rc.lferr_internal', calls('lferr_selection'))            # N-08 internal LL, selection only
        self.assertFalse(any('dwt' in c for c in calls('lferr_selection')))
        self.assertEqual([a.arg for a in funcs['lferr_selection'].args.args], ['ll_syn_internal', 'll_target'])
        self.assertIn('IDADV_RUNNER_CONTRACT', at_m7c2b('methods/gpat/losses.py').decode())

    def test_05_evidence_record(self):
        ev = json.loads(at_m7c2b(EVIDENCE))
        self.assertEqual((ev['milestone'], ev['status'], ev['authority_commit']), ('M7C2b', 'PASS', AUTHORITY))
        self.assertTrue(all(ev['gates'].values()))
        for k in ('dataset_images_read', 'training_steps', 'checkpoint_writes'):
            self.assertEqual(ev[k], 0, k)
        for k in ('gpu_used', 'teacher_weights_loaded', 'imagenet_weight_file_read'):
            self.assertIs(ev[k], False, k)
        self.assertIs(ev['environment']['cuda_available'], False)
        self.assertEqual(ev['environment']['PyWavelets_distribution'], '1.9.0')
        self.assertEqual(ev['environment']['ptwt'], '1.0.1')
        self.assertEqual({k: v['total'] for k, v in ev['parameter_counts'].items()}, EXPECTED_COUNTS)
        self.assertEqual(ev['config_sha256'], {v: CONFIG_SHA[f'configs/methods/gpat_{v.lower()}.yaml']
                                               for v in ('B0', 'B1', 'B2', 'B3')})
        self.assertLess(ev['wavelet']['D11_max_abs'], 1e-5)
        self.assertLess(ev['d12']['zero_residual_x_hat_minus_idwt_dwt_x_t_max_abs'], 1e-5)
        self.assertLess(ev['d12']['source_independence_max_abs'], 1e-5)
        self.assertLessEqual(ev['highpass']['max_abs_vs_cv2_reflect101'], 1e-6)
        t = ev['shape_traces']
        self.assertEqual((t['B0']['attack_logits'], t['B0']['identity_logits']), (None, None))
        self.assertEqual((t['B1']['attack_logits'], t['B1']['identity_logits']), ([1, 6], None))
        self.assertEqual((t['B2']['attack_logits'], t['B2']['identity_logits']), (None, [1, 60]))
        self.assertEqual((t['B3']['attack_logits'], t['B3']['identity_logits']), ([1, 6], [1, 60]))
        self.assertEqual(ev['batching']['optimizer_groups'], 1105)
        oc = ev['owner_clarifications']
        self.assertEqual(oc['L_low_training']['definition'], 'mean |DWT(x_hat).LL - LL_t|')
        self.assertIs(oc['L_low_training']['uses_internal_LL_syn'], False)
        self.assertIs(oc['LFErr_selection']['used_as_training_loss'], False)
        self.assertEqual(oc['patchgan']['class'], 'IMPLEMENTATION_CLARIFICATION')
        self.assertIsNone(oc['patchgan']['new_deviation'])
        self.assertEqual(oc['dev022_group_normalization']['status'], 'RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED')
        self.assertIs(oc['dev022_group_normalization']['microbatch_sample_weight_applied_to_idadv'], False)
        ll = ev['l_low']
        self.assertLess(ll['zero_residual'], 1e-5)
        self.assertGreater(ll['perturbed_x_hat'], 1e-3)
        self.assertGreater(ll['grad_to_x_hat_abs_sum'], 0.0)
        self.assertEqual(ll['selection_lferr_gamma0_max'], 0.0)

    def test_06_preflight_checks_and_rejections(self):
        pf = load_tool('m7c2b_preflight', 'tools/m7c2b_gpat_static_core_preflight.py')
        self.assertEqual(pf.AUTHORITY, AUTHORITY)
        pf.check_evidence(at_m7c2b)
        pf.check_protected(at_m7c2b, at_authority)
        pf.check_scope(listdir_at_m7c2b)
        pf.check_config_status(at_m7c2b, at_authority)
        ev = json.loads(at_m7c2b(EVIDENCE))
        for mutate in (lambda e: e.__setitem__('gpu_used', True),
                       lambda e: e.__setitem__('dataset_images_read', 1),
                       lambda e: e['parameter_counts']['g_res'].__setitem__('total', 1),
                       lambda e: e['gates'].__setitem__('source_independence', False),
                       lambda e: e['d12'].__setitem__('source_independence_max_abs', 1e-3)):
            bad = copy.deepcopy(ev)
            mutate(bad)
            raw = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                pf.check_evidence(lambda rel, raw=raw: raw if rel == EVIDENCE else at_m7c2b(rel))
        with self.assertRaises(ValueError):
            pf.check_scope(lambda rel: [*GPAT_DIR, 'training.py'])
        with self.assertRaises(ValueError):
            pf.check_protected(lambda rel: b'x' if rel == RESOLUTION else at_m7c2b(rel), at_authority)


@unittest.skipUnless(LIVE, 'live CPU tier needs torch + torchvision + ptwt (run in gpat-m7-cpu)')
class LiveM7C2B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn.functional as F
        cls.torch, cls.F = torch, F
        from methods.gpat import naf_source, runtime_contract, teacher_preprocess
        cls.naf, cls.rc, cls.tp = naf_source, runtime_contract, teacher_preprocess
        try:
            naf_source.verify_source()
        except naf_source.NAFSourceError as exc:
            raise unittest.SkipTest(f'pinned NAFNet source unavailable: {exc}')
        cls.tool = load_tool('m7c2b_evidence', 'tools/m7c2b_gpat_static_core_evidence.py')
        torch.use_deterministic_algorithms(True)
        cls.b0 = cls.tool.build('B0')
        cls.b3 = cls.tool.build('B3')
        cls.x_s, cls.x_s2, cls.x_t = cls.tool.images(1, 4242)

    def assertMaxAbs(self, a, b, tol):
        self.assertLessEqual(float((a - b).abs().max()), tol)

    # ------------------------------------------------------------------ 2/3 wavelet
    def test_10_dwt_idwt_roundtrip(self):
        from methods.gpat import wavelet
        torch = self.torch
        g = torch.Generator().manual_seed(42)
        x = torch.rand(100, 3, 256, 256, generator=g) * 2 - 1                   # A10 D11 corpus
        bands = wavelet.dwt(x)
        self.assertEqual([tuple(b.shape) for b in bands], [(100, 3, 128, 128)] * 4)
        self.assertTrue(all(b.dtype == torch.float32 for b in bands))
        self.assertLess(float((wavelet.idwt(*bands) - x).abs().max()), 1e-5)
        fixed = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4) / 10
        ll, lh, hl, hh = wavelet.dwt(fixed)
        a, b, c, d = fixed[..., 0::2, 0::2], fixed[..., 0::2, 1::2], fixed[..., 1::2, 0::2], fixed[..., 1::2, 1::2]
        self.assertMaxAbs(ll, (a + b + c + d) / 2, 1e-5)                         # orthonormal Haar, deterministic
        self.assertMaxAbs(lh.abs(), ((a + b - c - d) / 2).abs(), 1e-5)            # row difference -> cH
        self.assertMaxAbs(hl.abs(), ((a - b + c - d) / 2).abs(), 1e-5)            # column difference -> cV
        self.assertMaxAbs(hh.abs(), ((a - b - c + d) / 2).abs(), 1e-5)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(bands, wavelet.dwt(x))))
        self.assertEqual(tuple(wavelet.concat_bands(*wavelet.dwt(x[:1])).shape), (1, 12, 128, 128))

    def test_11_directional_lh_hl_mapping(self):
        from methods.gpat import wavelet
        torch = self.torch
        stripes = torch.zeros(1, 3, 256, 256)
        stripes[..., 1::2, :] = 1.0                                              # horizontal stripes: vary along rows
        _, lh, hl, hh = wavelet.dwt(stripes)
        self.assertGreater(float(lh.abs().max()), 0.9)
        self.assertLess(max(float(hl.abs().max()), float(hh.abs().max())), 1e-6)
        _, lh, hl, hh = wavelet.dwt(stripes.transpose(-1, -2).contiguous())      # vertical stripes
        self.assertGreater(float(hl.abs().max()), 0.9)
        self.assertLess(max(float(lh.abs().max()), float(hh.abs().max())), 1e-6)
        checker = (torch.arange(256)[:, None] + torch.arange(256)[None, :]).remainder(2).float().expand(1, 3, 256, 256)
        _, lh, hl, hh = wavelet.dwt(checker.contiguous())
        self.assertGreater(float(hh.abs().max()), 0.9)
        self.assertLess(max(float(lh.abs().max()), float(hl.abs().max())), 1e-6)

    # ------------------------------------------------------------------ 4 high-pass
    def test_12_highpass_parity(self):
        from methods.gpat.highpass import highpass
        torch = self.torch
        x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(7)) * 2 - 1
        hp = highpass(x)
        self.assertTrue(torch.equal(hp, self.tp.highpass(x)))                    # the M7C2a reference, unchanged
        if importlib.util.find_spec('cv2') is not None:
            import cv2
            ref = np.stack([np.stack([c - cv2.GaussianBlur(c, (9, 9), 1.5, borderType=cv2.BORDER_REFLECT_101)
                                      for c in img]) for img in x.numpy()])
            self.assertLessEqual(float(np.abs(hp.numpy() - ref).max()), 1e-6)
        self.assertLessEqual(float(highpass(torch.full((1, 3, 32, 32), -0.6)).abs().max()), 1e-6)
        self.assertMaxAbs(highpass(x), 2 * highpass((x + 1) / 2), 1e-6)          # no [0, 1] remap, linear

    # ------------------------------------------------------------------ 5/6/7 E_art
    def test_13_e_art_shapes_norm_and_input(self):
        from methods.gpat.artifact_encoder import encoder_input
        from methods.gpat import wavelet
        torch, F = self.torch, self.F
        e = self.b0.e_art
        x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(3)) * 2 - 1
        bands = wavelet.dwt(x)
        inp = encoder_input(x, *bands[1:])
        self.assertEqual(tuple(inp.shape), (2, 12, 256, 256))
        self.assertTrue(torch.equal(inp[:, :3], self.tp.highpass(x)))            # HP RGB, no ImageNet normalization
        for i, band in enumerate(bands[1:]):
            up = F.interpolate(band, scale_factor=2, mode='bilinear', align_corners=False)
            self.assertTrue(torch.equal(inp[:, 3 + 3 * i:6 + 3 * i], up))
        with torch.no_grad():
            spatial, l4, z = e(inp)
        self.assertEqual((tuple(spatial.shape), tuple(l4.shape), tuple(z.shape)),
                         ((2, 256, 16, 16), (2, 512, 8, 8), (2, 512)))
        self.assertMaxAbs(z.norm(dim=1), torch.ones(2), 1e-5)
        self.assertMaxAbs(z, F.normalize(l4.mean(dim=(-2, -1)), dim=1), 1e-6)
        self.assertFalse(hasattr(e, 'fc') or hasattr(e, 'avgpool'))
        self.assertFalse(any(k.startswith(('fc.', 'avgpool')) for k in e.state_dict()))
        self.assertEqual(tuple(e.conv1.weight.shape), (64, 12, 7, 7))
        with self.assertRaises(ValueError):
            e(torch.zeros(1, 3, 256, 256))

    def test_14_e_art_conv1_init_rule(self):
        import torchvision
        from methods.gpat.artifact_encoder import (ArtifactEncoder, PretrainedWeightError, apply_pretrained_init,
                                                   imagenet_state_dict, widen_conv1)
        torch = self.torch
        state = torchvision.models.resnet18(weights=None).state_dict()
        w3 = torch.arange(64 * 3 * 49, dtype=torch.float32).view(64, 3, 7, 7) / 1000.0
        state['conv1.weight'] = w3
        enc = ArtifactEncoder()
        info = apply_pretrained_init(enc, state)
        self.assertEqual(info['ignored'], ['fc.bias', 'fc.weight'])
        w = enc.conv1.weight.detach()
        self.assertTrue(torch.equal(w[:, :3], w3))
        extra = w3.mean(dim=1) / 3.0
        for c in range(3, 12):
            self.assertTrue(torch.equal(w[:, c], extra), c)
        self.assertMaxAbs(w[:, 3], (w3[:, 0] + w3[:, 1] + w3[:, 2]) / 9.0, 1e-6)
        for k, v in state.items():
            if not k.startswith('fc.') and k != 'conv1.weight':
                self.assertTrue(torch.equal(enc.state_dict()[k], v), k)
        self.assertTrue(torch.equal(widen_conv1(w3), w))
        bad = dict(state)
        del bad['layer1.0.conv1.weight']
        with self.assertRaises(KeyError):
            apply_pretrained_init(ArtifactEncoder(), bad)
        legacy = {k: v for k, v in state.items() if not k.endswith('num_batches_tracked')}   # official file layout
        info = apply_pretrained_init(ArtifactEncoder(), legacy)
        self.assertEqual(info['num_batches_tracked_not_in_source'], 20)
        with self.assertRaises(PretrainedWeightError):                           # never downloads
            imagenet_state_dict('/nonexistent/resnet18-f37072fd.pth')

    def test_15_e_art_optional_local_imagenet_asset(self):
        """If the official IMAGENET1K_V1 file is already cached locally (SHA-256 verified), the rule holds on it."""
        from methods.gpat.artifact_encoder import IMAGENET_WEIGHT_FILE, ArtifactEncoder, apply_pretrained_init, imagenet_state_dict
        torch = self.torch
        path = Path(torch.hub.get_dir()) / 'checkpoints' / IMAGENET_WEIGHT_FILE
        if not path.is_file():
            self.skipTest('IMAGENET1K_V1 file not cached locally (never downloaded)')
        state = imagenet_state_dict(path)
        enc = ArtifactEncoder()
        apply_pretrained_init(enc, state)
        w3 = state['conv1.weight']
        self.assertTrue(torch.equal(enc.conv1.weight[:, :3].detach(), w3))
        self.assertTrue(torch.equal(enc.conv1.weight[:, 11].detach(), w3.mean(dim=1) / 3.0))

    # ------------------------------------------------------------------ 8/9 NAF
    def test_16_naf_source_and_blocks(self):
        torch = self.torch
        verified = self.naf.verify_source()
        self.assertEqual({k: sha(v) for k, v in verified['files'].items()}, self.naf.FILES_SHA256)
        self.assertEqual(verified['commit'], '2b4af71ebe098a92a75910c233a3965a3e93ede4')
        from methods.gpat.generator import naf_symbols
        syms = naf_symbols()
        self.assertEqual(sorted(syms), ['LayerNorm2d', 'LayerNormFunction', 'NAFBlock', 'SimpleGate'])
        blocks = [m for m in self.b0.g_res.modules() if type(m).__name__ == 'NAFBlock']
        self.assertEqual(len(blocks), 2 + 2 + 4 + 8 + 12 + 2 + 2 + 2 + 2)
        self.assertTrue(all(type(b) is syms['NAFBlock'] and type(b).__module__ == 'gpat_pinned_nafnet' for b in blocks))
        self.assertNotIn('basicsr', sys.modules)
        blk = syms['NAFBlock'](32)
        with torch.no_grad():
            blk.beta.normal_()
            blk.gamma.normal_()
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            y = blk(x)
            y.square().mean().backward()
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in blk.parameters()))
        self.assertEqual(sum(p.numel() for p in blk.parameters()), 8224)

    def test_17_naf_layernorm_native_parity(self):
        torch = self.torch
        from methods.gpat.generator import naf_symbols
        lnf = naf_symbols()['LayerNormFunction']

        def native(x, w, b, eps):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            return w.view(1, -1, 1, 1) * ((x - mu) / (var + eps).sqrt()) + b.view(1, -1, 1, 1)

        for dt, tol in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
            x = torch.randn(2, 16, 8, 8, dtype=dt) * 2 + 0.5
            w, b, g = torch.randn(16, dtype=dt), torch.randn(16, dtype=dt), torch.randn(2, 16, 8, 8, dtype=dt)
            a = [t.clone().requires_grad_(True) for t in (x, w, b)]
            r = [t.clone().requires_grad_(True) for t in (x, w, b)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                y1 = lnf.apply(*a, 1e-6)
                y1.backward(g)
            y2 = native(*r, 1e-6)
            y2.backward(g)
            self.assertLessEqual(float((y1 - y2).detach().abs().max()), tol)
            for p, q in zip(a, r):
                self.assertLessEqual(float((p.grad - q.grad).abs().max()), tol * max(1.0, float(q.grad.abs().max())))

    # ------------------------------------------------------------------ 10/11 FiLM, G_res
    def test_18_film_identity_at_init(self):
        torch = self.torch
        from methods.gpat.generator import FiLM
        film = FiLM(64)
        h, z = torch.randn(2, 64, 8, 8), torch.randn(2, 512)
        self.assertTrue(torch.equal(film(h, z), h))
        self.assertEqual(tuple(film.affine.weight.shape), (128, 512))
        g = self.b0.g_res
        films = [s.film for s in (*g.encoders, *g.decoders)]
        self.assertEqual([f.channels for f in films], [32, 64, 128, 256, 256, 128, 64, 32])
        for f in films:
            self.assertEqual(float(f.affine.weight.detach().abs().sum()) + float(f.affine.bias.detach().abs().sum()), 0.0)
        self.assertFalse(any(isinstance(m, FiLM) for m in g.middle.modules()))  # no bottleneck FiLM
        with torch.no_grad():
            film.affine.bias.copy_(torch.cat([torch.full((64,), 0.5), torch.full((64,), -1.0)]))
        self.assertMaxAbs(film(h, z), 1.5 * h - 1.0, 1e-6)                        # (1 + gamma) * h + beta

    def test_19_g_res_shape_trace(self):
        trace = self.tool.shape_trace(self.b0, self.x_s, self.x_t)
        expected = {'g_res.intro': [1, 32, 128, 128], 'g_res.enc1': [1, 32, 128, 128], 'g_res.down1': [1, 64, 64, 64],
                    'g_res.enc2': [1, 64, 64, 64], 'g_res.down2': [1, 128, 32, 32], 'g_res.enc3': [1, 128, 32, 32],
                    'g_res.down3': [1, 256, 16, 16], 'g_res.enc4': [1, 256, 16, 16], 'g_res.down4': [1, 512, 8, 8],
                    'g_res.code_proj': [1, 512, 16, 16], 'g_res.code_pool': [1, 512, 8, 8],
                    'g_res.fuse': [1, 512, 8, 8], 'g_res.middle': [1, 512, 8, 8],
                    'g_res.up_dec4': [1, 256, 16, 16], 'g_res.dec4': [1, 256, 16, 16],
                    'g_res.up_dec3': [1, 128, 32, 32], 'g_res.dec3': [1, 128, 32, 32],
                    'g_res.up_dec2': [1, 64, 64, 64], 'g_res.dec2': [1, 64, 64, 64],
                    'g_res.up_dec1': [1, 32, 128, 128], 'g_res.dec1': [1, 32, 128, 128],
                    'g_res.ending': [1, 13, 128, 128], 'e_art.layer3': [1, 256, 16, 16], 'e_art.layer4': [1, 512, 8, 8],
                    'e_art_input': [1, 12, 256, 256], 'z_a': [1, 512], 'g_res_raw': [1, 13, 128, 128],
                    'x_hat': [1, 3, 256, 256], 'A': [1, 1, 256, 256], 'M': [1, 1, 128, 128],
                    'D_logits': [1, 1, 30, 30]}
        for k, v in expected.items():
            self.assertEqual(trace[k], v, k)
        g = self.b0.g_res
        self.assertEqual([len(s.blocks) for s in g.encoders], [2, 2, 4, 8])
        self.assertEqual([len(s.blocks) for s in g.decoders], [2, 2, 2, 2])
        self.assertEqual(len(g.middle), 12)
        self.assertEqual([(u.conv.in_channels, u.conv.out_channels, u.conv.kernel_size, u.conv.bias is not None)
                          for u in g.ups], [(c * 2, c, (3, 3), True) for c in (256, 128, 64, 32)])
        self.assertEqual([(d.in_channels, d.out_channels, d.kernel_size, d.stride) for d in g.downs],
                         [(c, 2 * c, (2, 2), (2, 2)) for c in (32, 64, 128, 256)])
        torch = self.torch
        code = torch.randn(2, 512, 16, 16, requires_grad=True)
        pooled = g.code_pool(code)                                              # exact adaptive-pool equivalent
        ref = torch.nn.functional.adaptive_avg_pool2d(code, (8, 8))
        self.assertTrue(torch.equal(pooled, ref))
        gp, = torch.autograd.grad(pooled.sum(), code)
        gr, = torch.autograd.grad(ref.sum(), code)
        self.assertTrue(torch.equal(gp, gr))
        with self.assertRaises(ValueError):
            g(torch.zeros(1, 12, 64, 64), torch.zeros(1, 512), torch.zeros(1, 256, 16, 16))

    def test_20_skip_after_film_and_decoder_order(self):
        """Encoder skip is the post-FiLM tensor; decoder adds the skip before its blocks and applies FiLM after."""
        torch = self.torch
        g = copy.deepcopy(self.b0.g_res)
        with torch.no_grad():
            g.encoders[0].film.affine.bias[:32] = 1.0                           # scale enc1 output by 2
        captured = {}
        h1 = g.encoders[0].register_forward_hook(lambda m, i, o: captured.__setitem__('enc1', o.detach().clone()))
        h2 = g.decoders[3].blocks.register_forward_pre_hook(lambda m, i: captured.__setitem__('dec1_in', i[0].detach().clone()))
        h3 = g.ups[3].register_forward_hook(lambda m, i, o: captured.__setitem__('up1', o.detach().clone()))
        try:
            with torch.no_grad():
                out = self.b0.forward_generator(self.x_s, self.x_t, scale_hf=0.15)
                g(torch.cat(out.target_bands, 1), out.z_a, out.spatial_code)
        finally:
            for h in (h1, h2, h3):
                h.remove()
        self.assertTrue(torch.equal(captured['dec1_in'], captured['up1'] + captured['enc1']))

    # ------------------------------------------------------------------ 12/13/14 channel mapping, mask, gamma
    def test_21_exact_channel_mapping_and_activations(self):
        torch = self.torch
        from methods.gpat import composition
        from methods.gpat.generator import OUTPUT_SLICES, split_output
        self.assertEqual(OUTPUT_SLICES, {'delta_LL': (0, 3), 'delta_LH': (3, 6), 'delta_HL': (6, 9),
                                         'delta_HH': (9, 12), 'mask_logit': (12, 13)})
        raw = torch.arange(13, dtype=torch.float32).view(1, 13, 1, 1).expand(2, 13, 128, 128) / 10 - 0.6
        parts = split_output(raw)
        for k, (a, b) in OUTPUT_SLICES.items():
            self.assertTrue(torch.equal(parts[k], raw[:, a:b]), k)
        d = composition.activate(raw, 0.1)
        self.assertTrue(torch.equal(d['delta_LL'], torch.tanh(raw[:, 0:3]) * 0.05))
        for band, (a, b) in (('LH', (3, 6)), ('HL', (6, 9)), ('HH', (9, 12))):
            self.assertTrue(torch.equal(d['delta_' + band], torch.tanh(raw[:, a:b]) * 0.1), band)
        self.assertTrue(torch.equal(d['M'], torch.sigmoid(raw[:, 12:13])))
        model = copy.deepcopy(self.b0)
        bias = torch.linspace(-0.6, 0.6, 13)
        with torch.no_grad():
            model.g_res.ending.weight.zero_()
            model.g_res.ending.bias.copy_(bias)
            out = model(self.x_s, self.x_t, scale_hf=0.15)
        for k, (a, b) in OUTPUT_SLICES.items():
            self.assertTrue(torch.equal(out.raw[:, a:b], bias[a:b].view(1, -1, 1, 1).expand(1, b - a, 128, 128)), k)
        self.assertMaxAbs(out.delta_HH, torch.tanh(bias[9:12]).view(1, 3, 1, 1) * 0.15, 0)
        self.assertMaxAbs(out.M, torch.sigmoid(bias[12]).expand(1, 1, 128, 128), 0)

    def test_22_mask_range_and_gamma_preserves_ll(self):
        torch = self.torch
        from methods.gpat import composition, wavelet
        raw = torch.randn(2, 13, 128, 128, generator=torch.Generator().manual_seed(5)) * 50
        d = composition.activate(raw, 0.15)
        self.assertEqual(tuple(d['M'].shape), (2, 1, 128, 128))
        self.assertTrue(bool((d['M'] >= 0).all() and (d['M'] <= 1).all()))
        self.assertLessEqual(float(d['delta_LH'].abs().max()), 0.15 + 1e-7)
        self.assertLessEqual(float(d['delta_LL'].abs().max()), 0.05 + 1e-7)
        x_t = torch.rand(2, 3, 256, 256) * 2 - 1
        bands = wavelet.dwt(x_t)
        self.assertGreater(float(d['delta_LL'].abs().max()), 0.01)
        syn = composition.compose(bands, d, 0.0)
        self.assertTrue(torch.equal(syn['LL_syn'], bands[0]))                    # gamma = 0: LL exactly preserved
        syn5 = composition.compose(bands, d, 0.05)
        self.assertTrue(torch.equal(syn5['LL_syn'], bands[0] + 0.05 * d['delta_LL']))
        self.assertTrue(torch.equal(syn['LH_syn'], bands[1] + d['M'] * d['delta_LH']))
        self.assertTrue(torch.equal(syn['x_hat'], wavelet.idwt(syn['LL_syn'], syn['LH_syn'], syn['HL_syn'], syn['HH_syn'])))
        with torch.no_grad():
            out = self.b0(self.x_s, self.x_t, scale_hf=0.15)
        self.assertTrue(torch.equal(out.LL_syn, out.target_bands[0]))
        self.assertEqual(out.gamma, 0.0)
        for bad in ({'gamma': 0.03}, {'scale_hf': 0.2}, {'artifact_scale': 0.5}):
            kw = {'scale_hf': 0.15, **bad}
            with self.assertRaises(ValueError):
                composition.generate(raw, bands, x_t, **kw)
        hi = wavelet.dwt(torch.full((1, 3, 256, 256), 0.99))
        sat = {'delta_LL': torch.zeros(1, 3, 128, 128), 'delta_LH': torch.full((1, 3, 128, 128), 0.15),
               'delta_HL': torch.zeros(1, 3, 128, 128), 'delta_HH': torch.zeros(1, 3, 128, 128),
               'M': torch.ones(1, 1, 128, 128)}
        self.assertGreater(float(composition.compose(hi, sat, 0.0)['x_hat'].max()), 1.05)   # x_hat never clamped

    # ------------------------------------------------------------------ 15/16 D12
    def test_23_zero_residual_and_source_independence(self):
        torch = self.torch
        from methods.gpat import wavelet
        model = copy.deepcopy(self.b0)
        with torch.no_grad():                                                   # make the live residual large
            model.g_res.ending.weight.normal_(0, 0.05)
            model.g_res.ending.bias.normal_(0, 1.0)
            a = model(self.x_s, self.x_t, scale_hf=0.15, artifact_scale=0.0)
            b = model(self.x_s2, self.x_t, scale_hf=0.15, artifact_scale=0.0)
            live_a = model(self.x_s, self.x_t, scale_hf=0.15)
            live_b = model(self.x_s2, self.x_t, scale_hf=0.15)
        recon = wavelet.idwt(*wavelet.dwt(self.x_t))
        self.assertLess(float((a.x_hat - recon).abs().max()), 1e-5)
        self.assertLess(float((a.x_hat - self.x_t).abs().max()), 1e-5)
        self.assertLess(float((a.x_hat - b.x_hat).abs().max()), 1e-5)
        for k in ('delta_LL', 'delta_LH', 'delta_HL', 'delta_HH'):
            self.assertEqual(float(getattr(a, k).abs().max()), 0.0, k)
        self.assertFalse(torch.equal(a.z_a, b.z_a))                              # E_art and G_res really ran
        self.assertFalse(torch.equal(a.raw, b.raw))
        self.assertGreater(float((live_a.x_hat - live_b.x_hat).abs().max()), 1e-5)   # non-vacuous
        self.assertGreater(float((live_a.x_hat - self.x_t).abs().max()), 1e-3)

    # ------------------------------------------------------------------ 17 artifact map
    def test_24_artifact_map_exact_formula(self):
        torch, F = self.torch, self.F
        from methods.gpat import composition
        g = torch.Generator().manual_seed(11)
        n = 2
        d = {k: (torch.rand(n, 3, 128, 128, generator=g) - 0.5) * 0.3 for k in ('delta_LL', 'delta_LH', 'delta_HL', 'delta_HH')}
        d['M'] = torch.rand(n, 1, 128, 128, generator=g)
        x_t = torch.rand(n, 3, 256, 256, generator=g) * 2 - 1
        x_hat = x_t + (torch.rand(n, 3, 256, 256, generator=g) - 0.5) * 1.5
        for gamma in (0.0, 0.05):
            a, comp = composition.artifact_map(x_hat, x_t, d, gamma, components=True)
            self.assertEqual(tuple(a.shape), (n, 1, 256, 256))
            a_rgb = (x_hat - x_t).abs().mean(1, keepdim=True)
            a_f = (d['delta_LH'].abs() + d['delta_HL'].abs() + d['delta_HH'].abs() + gamma * d['delta_LL'].abs()).mean(1, keepdim=True)
            up = lambda t: F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)   # noqa: E731
            u = up(d['M']) * (0.5 * a_rgb + 0.5 * up(a_f))
            ref = (u / 2.0).clamp(0, 1)
            self.assertMaxAbs(a, ref, 1e-7)
            self.assertTrue(bool((a >= 0).all() and (a <= 1).all()))
            forbidden = {'A_rgb_only': (a_rgb / 2).clamp(0, 1), 'no_A_freq': (up(d['M']) * 0.5 * a_rgb / 2).clamp(0, 1),
                         'no_M': ((0.5 * a_rgb + 0.5 * up(a_f)) / 2).clamp(0, 1),
                         'divide_only_A_rgb': (up(d['M']) * (0.5 * a_rgb / 2 + 0.5 * up(a_f))).clamp(0, 1)}
            for name, wrong in forbidden.items():
                self.assertGreater(float((a - wrong).abs().max()), 1e-3, name)
        self.assertTrue(torch.equal(comp['M_256'], F.interpolate(d['M'], scale_factor=2, mode='bilinear', align_corners=False)))
        big = {k: torch.full((1, 3, 128, 128), 5.0) for k in ('delta_LL', 'delta_LH', 'delta_HL', 'delta_HH')}
        big['M'] = torch.ones(1, 1, 128, 128)
        self.assertEqual(float(composition.artifact_map(x_t[:1] + 9, x_t[:1], big, 0.0).min()), 1.0)   # clip at 1

    # ------------------------------------------------------------------ 18 PatchGAN
    def test_25_patchgan(self):
        torch = self.torch
        from methods.gpat.discriminator import PatchGANDiscriminator, discriminator_input
        d = self.b0.discriminator
        convs = [m for m in d.model if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual([(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding) for c in convs],
                         [(6, 64, (4, 4), (2, 2), (1, 1)), (64, 128, (4, 4), (2, 2), (1, 1)),
                          (128, 256, (4, 4), (2, 2), (1, 1)), (256, 512, (4, 4), (1, 1), (1, 1)),
                          (512, 1, (4, 4), (1, 1), (1, 1))])
        kinds = [type(m).__name__ for m in d.model]
        self.assertEqual(kinds, ['Conv2d', 'LeakyReLU', 'Conv2d', 'InstanceNorm2d', 'LeakyReLU', 'Conv2d',
                                 'InstanceNorm2d', 'LeakyReLU', 'Conv2d', 'InstanceNorm2d', 'LeakyReLU', 'Conv2d'])
        self.assertTrue(all(m.negative_slope == 0.2 for m in d.model if isinstance(m, torch.nn.LeakyReLU)))
        self.assertTrue(all(c.bias is not None for c in convs))                  # every conv keeps its bias
        norms = [m for m in d.model if isinstance(m, torch.nn.InstanceNorm2d)]
        self.assertEqual([m.num_features for m in norms], [128, 256, 512])       # layers 2..4 only
        self.assertTrue(all(m.affine is False and m.track_running_stats is False for m in norms))
        self.assertFalse(any(isinstance(m, (torch.nn.Sigmoid, torch.nn.Tanh)) for m in d.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in d.modules()))
        self.assertIsInstance(d.model[-1], torch.nn.Conv2d)                     # no sigmoid
        x = torch.rand(2, 3, 256, 256) * 2 - 1
        inp = discriminator_input(x)
        self.assertEqual(tuple(inp.shape), (2, 6, 256, 256))
        self.assertTrue(torch.equal(inp[:, :3], x) and torch.equal(inp[:, 3:], self.tp.highpass(x)))
        with torch.no_grad():
            out = d(x)
        self.assertEqual(tuple(out.shape), (2, 1, 30, 30))
        self.assertEqual(sum(p.numel() for p in PatchGANDiscriminator().parameters()), EXPECTED_COUNTS['discriminator'])

    # ------------------------------------------------------------------ 19/20/21 heads and GRL
    def test_26_grl_gradient_sign(self):
        torch, F = self.torch, self.F
        from methods.gpat.heads import IdentityAdversaryHead
        from methods.gpat.grl import GRL_ALPHA, grad_reverse
        self.assertEqual(GRL_ALPHA, 1.0)
        head = IdentityAdversaryHead()
        ref = torch.nn.Linear(512, 60)
        ref.load_state_dict(head.fc.state_dict())
        z = F.normalize(torch.randn(4, 512), dim=1)
        y = torch.tensor([0, 5, 17, 59])
        z1, z2 = z.clone().requires_grad_(True), z.clone().requires_grad_(True)
        out1 = head(z1)
        self.assertTrue(torch.equal(out1, ref(z2)))                              # forward identity
        F.cross_entropy(out1, y).backward()
        F.cross_entropy(ref(z2), y).backward()
        self.assertMaxAbs(z1.grad, -z2.grad, 1e-7)                               # reversed into z_a
        self.assertGreater(float(z2.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(head.fc.weight.grad, ref.weight.grad))       # head minimizes CE (same sign)
        self.assertTrue(torch.equal(head.fc.bias.grad, ref.bias.grad))
        w = torch.randn(3, requires_grad=True)
        grad_reverse(w, 2.0).sum().backward()
        self.assertTrue(torch.equal(w.grad, torch.full((3,), -2.0)))

    def test_27_attack_and_identity_heads(self):
        torch = self.torch
        from methods.gpat.heads import ATTACK_CLASSES, ATTACK_CLASS_INDEX, AttackTypeHead, IdentityAdversaryHead
        self.assertEqual(ATTACK_CLASSES, ('makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay'))
        self.assertEqual(ATTACK_CLASS_INDEX['print'], 4)
        self.assertNotIn('live', ATTACK_CLASSES)
        a, i = AttackTypeHead(), IdentityAdversaryHead()
        self.assertEqual((a.fc.in_features, a.fc.out_features, a.fc.bias is not None), (512, 6, True))
        self.assertEqual((i.fc.in_features, i.fc.out_features, i.fc.bias is not None), (512, 60, True))
        self.assertEqual(i.grl.alpha, 1.0)
        z = torch.randn(3, 512)
        self.assertEqual((tuple(a(z).shape), tuple(i(z).shape)), ((3, 6), (3, 60)))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 3078)
        self.assertEqual(sum(p.numel() for p in i.parameters()), 30780)

    # ------------------------------------------------------------------ 22/23 DEV-022
    def test_28_dev022_labelled_normalization(self):
        torch, F = self.torch, self.F
        from methods.gpat import losses
        g = torch.Generator().manual_seed(2)
        l1, l2 = torch.randn(4, 60, generator=g, requires_grad=True), torch.randn(2, 60, generator=g, requires_grad=True)
        y1, y2 = torch.tensor([3, -1, 7, 59]), torch.tensor([-1, 12])
        v1, v2 = torch.tensor([True, False, True, True]), torch.tensor([False, True])
        s1, c1 = losses.idadv_microbatch(l1, y1, v1)
        s2, c2 = losses.idadv_microbatch(l2, y2, v2)
        self.assertEqual((c1, c2), (3, 1))
        group = losses.idadv_group([s1, s2], [c1, c2])
        ref = F.cross_entropy(torch.cat([l1[v1], l2[v2]]), torch.cat([y1[v1], y2[v2]]), reduction='mean')
        self.assertMaxAbs(group, ref, 1e-6)
        self.assertGreater(abs(float(group) - 0.5 * (float(s1) / c1 + float(s2) / c2)), 1e-4)   # not mean of means
        shares = losses.idadv_share(s1, c1 + c2) + losses.idadv_share(s2, c1 + c2)
        self.assertMaxAbs(shares, group, 1e-6)
        group.backward()
        self.assertEqual(float(l1.grad[1].abs().sum()), 0.0)                     # masked row gets no gradient
        self.assertGreater(float(l1.grad[0].abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            losses.idadv_microbatch(torch.randn(2, 6), torch.tensor([0, 1]), torch.tensor([True, True]))

    def test_28b_dev022_owner_group_example(self):
        torch, F = self.torch, self.F
        from methods.gpat import batching, losses
        g = torch.Generator().manual_seed(22)
        la, lb = torch.randn(4, 60, generator=g) * 3, torch.randn(2, 60, generator=g) * 3
        ya, yb = torch.tensor([-1, 9, -1, -1]), torch.tensor([30, 44])
        va, vb = torch.tensor([False, True, False, False]), torch.tensor([True, True])
        (sa, ca), (sb, cb) = losses.idadv_microbatch(la, ya, va), losses.idadv_microbatch(lb, yb, vb)
        self.assertEqual((ca, cb), (1, 2))
        group = losses.idadv_group([sa, sb], [ca, cb])
        ref = F.cross_entropy(torch.cat([la[va], lb[vb]]), torch.cat([ya[va], yb[vb]]), reduction='sum') / 3
        self.assertMaxAbs(group, ref, 1e-6)
        wa, wb = batching.group_weights([4, 2])
        self.assertEqual((wa, wb), (4 / 6, 2 / 6))
        sample_weighted = wa * (sa / ca) + wb * (sb / cb)
        mean_of_means = 0.5 * (sa / ca + sb / cb)
        self.assertGreater(abs(float(group - sample_weighted)), 1e-3)
        self.assertGreater(abs(float(group - mean_of_means)), 1e-3)
        self.assertMaxAbs(losses.idadv_share(sa, 3) + losses.idadv_share(sb, 3), group, 1e-6)
        self.assertEqual(losses.IDADV_RUNNER_CONTRACT, 'RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED')

    def test_29_dev022_zero_labelled_exact_zero(self):
        torch = self.torch
        from methods.gpat import losses
        logits = torch.randn(4, 60, requires_grad=True)
        s, c = losses.idadv_microbatch(logits, torch.full((4,), -1), torch.zeros(4, dtype=torch.bool))
        self.assertEqual((float(s), c), (0.0, 0))
        group = losses.idadv_group([s, s], [0, 0])
        self.assertEqual(float(group), 0.0)
        self.assertTrue(group.requires_grad)
        group.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros(4, 60)))
        self.assertEqual(float(losses.idadv_share(s, 0)), 0.0)

    # ------------------------------------------------------------------ 24/25/26 spectra
    def test_30_spectral_summaries(self):
        torch = self.torch
        from methods.gpat import spectral
        g = torch.Generator().manual_seed(9)
        x = torch.rand(2, 3, 256, 256, generator=g) * 2 - 1
        r, o = spectral.s_radial(x), spectral.s_orient(x)
        self.assertEqual((tuple(r.shape), tuple(o.shape)), ((2, 3, 128), (2, 3, 8)))
        self.assertEqual((r.dtype, o.dtype), (torch.float32, torch.float32))
        self.assertMaxAbs(r.sum(-1), torch.ones(2, 3), 1e-5)
        self.assertMaxAbs(o.sum(-1), torch.ones(2, 3), 1e-5)
        self.assertTrue(torch.equal(r, spectral.s_radial(x)) and torch.equal(o, spectral.s_orient(x)))   # deterministic
        # independent NumPy reference (float64)
        xn = x.double().numpy()
        p = np.abs(np.fft.fftshift(np.fft.fft2(xn), axes=(-2, -1))) ** 2
        dy, dx = np.meshgrid(np.arange(256) - 128, np.arange(256) - 128, indexing='ij')
        rad = np.floor(np.sqrt(dx ** 2 + dy ** 2)).astype(int)
        keep = rad < 128
        rr = np.zeros((2, 3, 128))
        for b in range(2):
            for c in range(3):
                rr[b, c] = np.bincount(rad[keep], weights=np.log1p(p[b, c][keep]), minlength=128)
        rr /= rr.sum(-1, keepdims=True) + 1e-12
        self.assertLessEqual(float(np.abs(r.numpy() - rr).max()), 1e-6)
        fy, fx = dy / 256, dx / 256
        support = (np.hypot(fx, fy) > 0) & (np.hypot(fx, fy) <= 0.5)
        th = np.mod(np.arctan2(fy, fx), np.pi)
        ob = np.minimum(np.floor(th / (np.pi / 8)).astype(int), 7)
        oo = np.zeros((2, 3, 8))
        for b in range(2):
            for c in range(3):
                oo[b, c] = np.bincount(ob[support], weights=p[b, c][support], minlength=8)
        oo /= oo.sum(-1, keepdims=True) + 1e-12
        self.assertLessEqual(float(np.abs(o.numpy() - oo).max()), 1e-5)
        const = torch.full((1, 3, 256, 256), 0.3)
        self.assertMaxAbs(spectral.s_radial(const)[0, :, 0], torch.ones(3), 1e-6)      # DC bin included
        yy, xx = torch.meshgrid(torch.arange(256.), torch.arange(256.), indexing='ij')
        rows = torch.cos(2 * math.pi * 32 * yy / 256).expand(1, 3, 256, 256).contiguous()
        cols = torch.cos(2 * math.pi * 32 * xx / 256).expand(1, 3, 256, 256).contiguous()
        self.assertEqual(int(spectral.s_orient(rows)[0, 0].argmax()), 4)        # theta = pi/2
        self.assertEqual(int(spectral.s_orient(cols)[0, 0].argmax()), 0)        # theta = 0
        self.assertGreater(float(spectral.s_orient(rows)[0, 0, 4]), 0.99)
        oblique = torch.cos(2 * math.pi * (16 * xx + 32 * yy) / 256).expand(1, 3, 256, 256).contiguous()
        self.assertEqual(int(spectral.s_orient(oblique)[0, 0].argmax()), 2)     # theta = atan2(32, 16) ~ 1.107
        ls = spectral.spec_loss(x[:1], x[1:])
        ref = (r[:1] - r[1:]).abs().mean() + 0.5 * (o[:1] - o[1:]).abs().mean()
        self.assertMaxAbs(ls, ref, 1e-7)
        with self.assertRaises(ValueError):
            spectral.s_radial(torch.zeros(1, 3, 128, 128))

    # ------------------------------------------------------------------ 27 parsing
    def test_31_parsing_loss_known_examples(self):
        torch = self.torch
        from methods.gpat import losses
        cls = torch.randint(0, 11, (2, 224, 224), generator=torch.Generator().manual_seed(1))
        onehot = torch.nn.functional.one_hot(cls, 11).permute(0, 3, 1, 2).float() * 1e4
        total, comp = losses.l_parse(onehot, onehot.clone(), components=True)
        self.assertLessEqual(float(comp['dice']), 1e-6)
        self.assertLessEqual(float(comp['kl']), 1e-6)
        g = torch.Generator().manual_seed(4)
        lt, lh = torch.randn(1, 11, 2, 3, generator=g), torch.randn(1, 11, 2, 3, generator=g)
        pt = np.exp(lt.double().numpy())
        pt /= pt.sum(1, keepdims=True)
        ph = np.exp(lh.double().numpy())
        ph /= ph.sum(1, keepdims=True)
        dice = (2 * (pt * ph)[:, 1:].sum((2, 3)) + 1e-6) / (pt[:, 1:].sum((2, 3)) + ph[:, 1:].sum((2, 3)) + 1e-6)
        kl = (pt * (np.log(np.maximum(pt, 1e-8)) - np.log(np.maximum(ph, 1e-8)))).sum(1).mean()
        total, comp = losses.l_parse(lt, lh, components=True)
        self.assertAlmostEqual(float(comp['dice']), 1 - dice.mean(), places=6)
        self.assertAlmostEqual(float(comp['kl']), kl, places=6)
        self.assertAlmostEqual(float(total), 1 - dice.mean() + 0.1 * kl, places=6)
        self.assertEqual(total.dtype, torch.float32)
        with self.assertRaises(ValueError):
            losses.l_parse(torch.zeros(1, 10, 4, 4), torch.zeros(1, 10, 4, 4))

    # ------------------------------------------------------------------ 28 BCE
    def test_32_bce_definitions(self):
        torch, F = self.torch, self.F
        from methods.gpat import losses
        real, fake = torch.randn(3, 1, 30, 30, dtype=torch.float64), torch.randn(3, 1, 30, 30, dtype=torch.float64)
        ld = losses.l_d(real, fake)
        r, f = losses.l_d_terms(real, fake)
        self.assertEqual(float(ld), float(0.5 * (r + f)))
        cat = F.binary_cross_entropy_with_logits(torch.cat([real, fake]), torch.cat([torch.ones_like(real),
                                                                                   torch.zeros_like(fake)]))
        self.assertLessEqual(abs(float(ld) - float(cat)), 1e-12)
        self.assertEqual(ld.dtype, torch.float64)                               # fp64 never forced down
        self.assertAlmostEqual(float(r), float(F.binary_cross_entropy_with_logits(real, torch.ones_like(real))), places=12)
        self.assertEqual(losses.l_d(real.half(), fake.half()).dtype, torch.float32)   # promoted to fp32
        fl = fake.float().requires_grad_(True)
        lg = losses.l_gadv(fl)
        self.assertAlmostEqual(float(lg), float(F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))), places=6)
        lg.backward()
        self.assertGreater(float(fl.grad.abs().sum()), 0.0)                     # no detach on the G path
        self.assertTrue(bool((fl.grad < 0).all()))                               # pushes logits towards 'real'

    # ------------------------------------------------------------------ 29 TV / budget / bg / misc
    def test_33_regularization_losses(self):
        torch = self.torch
        from methods.gpat import losses
        ramp = torch.arange(8.).view(1, 1, 1, 8).expand(2, 1, 8, 8)
        self.assertAlmostEqual(float(losses.tv(ramp)), 1.0, places=6)             # |dx| = 1, |dy| = 0
        self.assertAlmostEqual(float(losses.tv(ramp.transpose(-1, -2))), 1.0, places=6)
        m = torch.rand(2, 1, 128, 128)
        d = [torch.rand(2, 3, 128, 128) for _ in range(3)]
        self.assertMaxAbs(losses.l_tv(m, *d), losses.tv(m) + 0.25 * (losses.tv(d[0]) + losses.tv(d[1]) + losses.tv(d[2])), 1e-6)
        a = torch.stack([torch.full((1, 256, 256), 0.005), torch.full((1, 256, 256), 0.3)])
        self.assertAlmostEqual(float(losses.l_budget(a)), (0.005 + 0.05) / 2, places=6)   # per-image hinge, batch mean
        self.assertEqual(float(losses.l_budget(torch.full((2, 1, 256, 256), 0.1))), 0.0)
        x_t, x_hat = torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)
        self.assertEqual(float(losses.l_bg(x_hat, x_t, torch.ones(2, 1, 256, 256))), 0.0)
        self.assertAlmostEqual(float(losses.l_bg(x_hat, x_t, torch.zeros(2, 1, 256, 256))),
                               float((x_hat - x_t).abs().mean()), places=6)
        with self.assertRaises(ValueError):
            losses.l_low(torch.rand(2, 3, 128, 128), torch.rand(2, 3, 64, 64))  # takes x_hat, not LL coefficients
        e1, e2 = torch.randn(3, 512), torch.randn(3, 512)
        self.assertAlmostEqual(float(losses.l_id(e1, e1)), 0.0, places=6)
        self.assertAlmostEqual(float(losses.l_id(e1, -e1)), 2.0, places=6)
        self.assertAlmostEqual(float(losses.l_id(e1, e2)), float((1 - self.F.cosine_similarity(e1, e2)).mean()), places=6)
        lm = torch.rand(2, 68, 2)
        self.assertAlmostEqual(float(losses.l_lm(lm, lm + 0.1)), 0.1, places=6)
        with self.assertRaises(ValueError):
            losses.l_lm(torch.rand(2, 68, 3), torch.rand(2, 68, 3))
        f_s, f_t = torch.randn(4, 512), torch.randn(4, 512)
        f_h = f_s + 0.1 * torch.randn(4, 512)
        cs, ct = self.F.cosine_similarity(f_h, f_s), self.F.cosine_similarity(f_h, f_t)
        self.assertMaxAbs(losses.l_artcon(f_h, f_s, f_t), (1 - cs + (ct - cs + 0.2).clamp_min(0)).mean(), 1e-6)
        logits = torch.randn(5, 6)
        lab = torch.tensor([0, 1, 2, 3, 5])
        self.assertMaxAbs(losses.l_type(logits, lab), self.F.cross_entropy(logits, lab), 1e-7)

    # ------------------------------------------------------------------ 30 generator assembler
    def test_34_generator_assembler_idadv_sign(self):
        torch = self.torch
        from methods.gpat import losses, schedule
        keys = ('id', 'lm', 'parse', 'low', 'artcon', 'spec', 'gadv', 'budget', 'tv', 'bg')
        zero = {k: torch.tensor(0.0) for k in keys}
        cur = schedule.curriculum(20000)
        total, rec = losses.assemble_generator_loss({**zero, 'idadv': torch.tensor(1.0)}, cur, lambda_type=0.0,
                                                    lambda_idadv=0.1)
        self.assertAlmostEqual(float(total), 0.1, places=7)                      # + lambda_idadv * L_idadv
        self.assertGreater(float(total), 0.0)
        vals = {k: torch.tensor(float(i + 1)) for i, k in enumerate(keys)}
        vals.update(type=torch.tensor(11.0), idadv=torch.tensor(12.0))
        total, rec = losses.assemble_generator_loss(vals, schedule.curriculum(1), lambda_type=0.2, lambda_idadv=0.1)
        w = {'id': 1.0, 'lm': 1.0, 'parse': 0.5, 'low': 2.0, 'artcon': 0.5, 'spec': 0.25, 'gadv': 0.0, 'budget': 0.5,
             'tv': 0.05, 'bg': 1.0, 'type': 0.2, 'idadv': 0.1}
        self.assertEqual(rec['weights'], w)
        self.assertAlmostEqual(float(total), sum(w[k] * float(vals[k]) for k in w), places=5)
        self.assertFalse(any(v.requires_grad for v in rec['components'].values()))
        with self.assertRaises(ValueError):                                     # B0 must not carry L_type
            losses.assemble_generator_loss({**zero, 'type': torch.tensor(1.0)}, cur, lambda_type=0.0, lambda_idadv=0.0)
        with self.assertRaises(ValueError):                                     # B1 needs L_type
            losses.assemble_generator_loss(zero, cur, lambda_type=0.2, lambda_idadv=0.0)
        x = torch.tensor(2.0, requires_grad=True)
        t, _ = losses.assemble_generator_loss({**zero, 'idadv': x}, cur, lambda_type=0.0, lambda_idadv=0.1)
        self.assertIsNone(x.grad)                                               # no backward inside
        t.backward()
        self.assertAlmostEqual(float(x.grad), 0.1, places=7)

    # ------------------------------------------------------------------ 31 face mask
    def test_35_face_mask_nearest_exact_dilation(self):
        torch = self.torch
        from methods.gpat import losses
        self.assertEqual(float(losses.face_mask_dilated(torch.zeros(2, 224, 224, dtype=torch.long)).sum()), 0.0)
        m = torch.zeros(1, 224, 224, dtype=torch.long)
        m[0, 100, 50] = 7
        m[0, 150, 150] = 10
        m[0, 20, 200] = 1
        out = losses.face_mask_dilated(m)
        self.assertEqual(tuple(out.shape), (1, 1, 256, 256))
        self.assertTrue(bool(((out == 0) | (out == 1)).all()))
        src = np.floor((np.arange(256) + 0.5) * 224 / 256).astype(int)           # nearest-exact index map
        fg = ((m[0].numpy() >= 1) & (m[0].numpy() <= 10))[src][:, src]
        pad = np.pad(fg, 7)
        ref = np.zeros_like(fg)
        for dy in range(15):
            for dx in range(15):
                ref |= pad[dy:dy + 256, dx:dx + 256]
        self.assertTrue(np.array_equal(out[0, 0].numpy().astype(bool), ref))
        with self.assertRaises(ValueError):
            losses.face_mask_dilated(torch.zeros(1, 256, 256, dtype=torch.long))

    # ------------------------------------------------------------------ 32 EMA
    def test_36_ema_parameters_buffers(self):
        torch = self.torch
        from methods.gpat.ema import ModelEMA
        live = copy.deepcopy(self.b0.e_art).train()
        ema = ModelEMA(live)
        self.assertFalse(ema.module.training)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))
        ema.initialize_from(live)
        before = {k: v.clone() for k, v in ema.module.state_dict().items()}
        with torch.no_grad():
            live(torch.randn(2, 12, 256, 256))                                  # BN running stats + num_batches_tracked
            for p in live.parameters():
                p.add_(1.0)
        ema.update_from(live)
        now, cur = ema.module.state_dict(), live.state_dict()
        self.assertTrue(torch.equal(now['bn1.num_batches_tracked'], cur['bn1.num_batches_tracked']))
        self.assertEqual(int(now['bn1.num_batches_tracked']), int(before['bn1.num_batches_tracked']) + 1)
        for k in ('conv1.weight', 'bn1.running_mean', 'bn1.running_var', 'layer3.0.bn2.weight'):
            self.assertMaxAbs(now[k], 0.999 * before[k] + 0.001 * cur[k], 1e-6)
        self.assertFalse(torch.equal(now['bn1.running_mean'], before['bn1.running_mean']))
        state = ema.state_dict()
        other = ModelEMA(live)
        other.load_state_dict(state)
        self.assertTrue(all(torch.equal(other.module.state_dict()[k], v) for k, v in now.items()))
        self.assertEqual(other.updates, 1)
        ev = ema.evaluation_copy()
        self.assertFalse(ev.training)
        self.assertIsNot(ev, ema.module)
        ModelEMA(self.b0.g_res)
        for bad in (self.b3.attack_head, self.b3.identity_head, self.b3.discriminator):
            with self.assertRaises(TypeError):
                ModelEMA(bad)
        with self.assertRaises(ValueError):
            ModelEMA(live, decay=0.99)

    # ------------------------------------------------------------------ 33/34/35 schedule, curriculum, batching
    def test_37_lr_schedule_endpoints(self):
        from methods.gpat import schedule
        self.assertEqual(schedule.main_lr(1), 0.0)
        self.assertAlmostEqual(schedule.main_lr(5525), 2e-4, places=15)
        self.assertAlmostEqual(schedule.main_lr(5526), 2e-4, places=15)
        self.assertAlmostEqual(schedule.main_lr(66300), 2e-6, places=15)
        self.assertAlmostEqual(schedule.main_lr(2), 2e-4 / 5524, places=15)
        mid = 5526 + (66300 - 5526) // 2
        self.assertAlmostEqual(schedule.main_lr(mid), 2e-6 + 0.5 * (2e-4 - 2e-6) * (1 + math.cos(math.pi * (mid - 5526) / 60774)),
                               places=15)
        self.assertAlmostEqual(schedule.attack_warmup_lr(1), 1e-4, places=15)
        self.assertAlmostEqual(schedule.attack_warmup_lr(1390), 0.0, places=15)
        for bad in (0, 66301, 1.0, True):
            with self.assertRaises(ValueError):
                schedule.main_lr(bad)
        with self.assertRaises(ValueError):
            schedule.attack_warmup_lr(1391)
        self.assertEqual((schedule.update_index(1, 1), schedule.update_index(5, 1105), schedule.update_index(6, 1),
                          schedule.update_index(60, 1105)), (1, 5525, 5526, 66300))
        with self.assertRaises(ValueError):
            schedule.update_index(61, 1)

    def test_38_curriculum_transitions(self):
        # M7D1-A1 (N6A): the pins below are the ORIGINAL v1.0 curriculum (historical `curriculum_v1_0`, lambda_adv
        # jump at u5526); production schedule.curriculum is the A1 one (lambda_adv ramp u5526..u6630, all else equal).
        from methods.gpat import runtime_contract as rc, schedule
        self.assertIs(schedule.curriculum, rc.curriculum_a1)
        self.assertEqual(schedule.curriculum(5526)['lambda_adv'], 0.0)
        c = rc.curriculum_v1_0
        self.assertAlmostEqual(c(1)['s_hf'], 0.02, places=15)
        self.assertAlmostEqual(c(5525)['s_hf'], 0.05, places=15)
        self.assertEqual({k: c(5525)[k] for k in ('lambda_adv', 'lambda_con', 'lambda_spec')}, {'lambda_adv': 0.0, 'lambda_con': 0.5, 'lambda_spec': 0.25})
        for u in (5526, 16575):
            self.assertEqual(c(u), {'stage': 2, 's_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0, 'lambda_spec': 0.5})
        for u in (16576, 66300):
            self.assertEqual(c(u), {'stage': 3, 's_hf': 0.15, 'lambda_adv': 0.10, 'lambda_con': 1.0, 'lambda_spec': 0.5})
        s = [c(u)['s_hf'] for u in range(1, 5526)]
        self.assertTrue(all(b > a for a, b in zip(s, s[1:])))

    def test_39_batch_accounting(self):
        from methods.gpat import batching
        acc = batching.accounting()
        self.assertEqual((acc['samples'], acc['microbatches'], acc['optimizer_groups']), (8838, 2210, 1105))
        self.assertEqual((acc['regular_group'], acc['last_group']), ([4, 4], [4, 2]))
        self.assertEqual(acc['regular_weights'], [0.5, 0.5])
        self.assertEqual(acc['last_weights'], [4 / 6, 2 / 6])
        groups = batching.optimizer_groups()
        self.assertEqual(sum(map(sum, groups)), 8838)
        self.assertTrue(all(g == [4, 4] for g in groups[:-1]))

    # ------------------------------------------------------------------ 36 identity map
    def test_40_identity_map_determinism(self):
        import random
        from methods.gpat import identity_labels as il
        casia = [{'dataset': 'casia_fasd', 'source_subject': str(i)} for i in range(1, 36)]
        msu = [{'dataset': 'msu_mfsd', 'source_subject': s} for s in
               [f'client{i:03d}' for i in range(1, 24)] + ['Zoë', 'ü-subject']]
        siw = [{'dataset': 'siwmv2', 'source_subject': None}] * 5
        recs = casia + msu + siw + casia[:4]                                    # duplicates and masked rows
        a = il.build_identity_map(recs, 'f' * 64)
        shuffled = list(recs)
        random.Random(1).shuffle(shuffled)
        b = il.build_identity_map(shuffled, 'f' * 64)
        self.assertEqual((a['json'], a['mapping_sha256'], a['json_sha256']), (b['json'], b['mapping_sha256'], b['json_sha256']))
        self.assertEqual(len(a['classes']), 60)
        order = [(c['dataset'], c['source_subject']) for c in a['classes']]
        self.assertEqual(order, sorted(order, key=lambda k: (k[0].encode('utf-8'), k[1].encode('utf-8'))))
        self.assertEqual(order[:3], [('casia_fasd', '1'), ('casia_fasd', '10'), ('casia_fasd', '11')])   # not numeric
        self.assertEqual(order[35], ('msu_mfsd', 'Zoë'))                        # UTF-8 bytes: 'Z' < 'c' < 0xC3
        self.assertEqual(order[-2:], [('msu_mfsd', 'client023'), ('msu_mfsd', 'ü-subject')])
        self.assertEqual(a['classes'][35]['dataset'], 'msu_mfsd')
        self.assertEqual(a['mapping_sha256'], self.rc.identity_class_map(il.identity_keys(recs), 'f' * 64)['mapping_sha256'])
        self.assertEqual(json.loads(a['json'])['mapping_sha256'], a['mapping_sha256'])
        labels, valid = il.labels_for([casia[0], siw[0], msu[-1]], a)
        self.assertEqual((labels, valid), ([a['index'][('casia_fasd', '1')], -1, 59], [True, False, True]))
        fewer = [r for r in recs if r is not casia[0]]
        with self.assertRaises(self.rc.IdentityMapError):
            il.build_identity_map(fewer, 'f' * 64)                              # K = 59 != 60
        self.assertEqual(len(il.build_identity_map(fewer, 'f' * 64, strict=False)['classes']), 59)
        with self.assertRaises(self.rc.IdentityMapError):
            il.build_identity_map([{'dataset': 'casia_fasd', 'source_subject': 3}], 'f' * 64, strict=False)

    # ------------------------------------------------------------------ 37 variants
    def test_41_variant_wiring(self):
        torch = self.torch
        outs, models = {}, {}
        for v in ('B0', 'B1', 'B2', 'B3'):
            models[v] = self.b0 if v == 'B0' else (self.b3 if v == 'B3' else self.tool.build(v, with_discriminator=False))
            with torch.no_grad():
                outs[v] = models[v](self.x_s, self.x_t, scale_hf=0.15)
        self.assertEqual([(outs[v].attack_logits is None, outs[v].identity_logits is None) for v in outs],
                         [(True, True), (False, True), (True, False), (False, False)])
        self.assertEqual(tuple(outs['B1'].attack_logits.shape), (1, 6))
        self.assertEqual(tuple(outs['B2'].identity_logits.shape), (1, 60))
        ref = {k: tuple(v.shape) for k, v in models['B0'].g_res.state_dict().items()}
        ref_e = {k: tuple(v.shape) for k, v in models['B0'].e_art.state_dict().items()}
        for v, m in models.items():
            self.assertEqual({k: tuple(t.shape) for k, t in m.g_res.state_dict().items()}, ref, v)
            self.assertEqual({k: tuple(t.shape) for k, t in m.e_art.state_dict().items()}, ref_e, v)
            self.assertEqual(tuple(outs[v].raw.shape), (1, 13, 128, 128))
        self.assertEqual(sorted(models['B0'].generator_modules()), ['e_art', 'g_res'])
        self.assertEqual(sorted(models['B3'].generator_modules()), ['attack_head', 'e_art', 'g_res', 'identity_head'])
        self.assertNotIn('discriminator', models['B3'].generator_modules())
        self.assertTrue(torch.equal(outs['B0'].x_hat, outs['B3'].x_hat))      # same seed -> same shared core
        counts = self.tool.counts(self.b3)
        self.assertEqual({k: v['total'] for k, v in counts.items()}, EXPECTED_COUNTS)
        self.assertEqual({k: v['trainable'] for k, v in counts.items()}, EXPECTED_COUNTS)
        with self.assertRaises(ValueError):
            self.b0(self.x_s, self.x_t[:, :, :128, :128], scale_hf=0.15)

    def test_42_forward_does_not_run_d_or_teachers_and_is_cpu(self):
        torch = self.torch
        calls = []
        h = self.b0.discriminator.register_forward_hook(lambda m, i, o: calls.append(1))
        try:
            with torch.no_grad():
                self.b0(self.x_s, self.x_t, scale_hf=0.15)
        finally:
            h.remove()
        self.assertEqual(calls, [])
        self.assertFalse(torch.cuda.is_available())
        self.assertTrue(all(p.device.type == 'cpu' for p in self.b3.parameters()))
        for mod in ('methods.gpat.model', 'methods.gpat.losses'):
            self.assertIn(mod, sys.modules)
        for name in ('gpatbench.preprocess.aux_models', 'basicsr', 'torch.utils.tensorboard'):
            self.assertNotIn(name, sys.modules)

    def test_44_l_low_literal_frozen_definition(self):
        torch = self.torch
        from unittest import mock
        from methods.gpat import losses, wavelet
        with torch.no_grad():
            out = self.b0(self.x_s, self.x_t, scale_hf=0.15)
        ll_t = out.target_bands[0]
        self.assertEqual(out.gamma, 0.0)
        self.assertTrue(torch.equal(out.LL_syn, ll_t))                          # internal LL == LL_t under gamma 0
        # A. literal definition
        self.assertEqual(float(losses.l_low(out.x_hat, ll_t)), float((wavelet.dwt(out.x_hat)[0] - ll_t).abs().mean()))
        # 6. gamma = 0 does not bypass the DWT
        with mock.patch.object(wavelet, 'dwt', wraps=wavelet.dwt) as spy:
            losses.l_low(out.x_hat, ll_t)
        self.assertEqual(spy.call_count, 1)
        self.assertIs(spy.call_args.args[0], out.x_hat)
        # B. gradient reaches x_hat, and through it G_res / E_art
        x = out.x_hat.clone().requires_grad_(True)
        losses.l_low(x, ll_t).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        model = copy.deepcopy(self.b0)
        with torch.no_grad():
            model.g_res.ending.weight.normal_(0, 0.05)
        live = model(self.x_s, self.x_t, scale_hf=0.15, gamma=0.05)
        losses.l_low(live.x_hat, live.target_bands[0]).backward()
        self.assertGreater(float(model.g_res.ending.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.e_art.conv1.weight.grad.abs().sum()), 0.0)
        # C/D. a modified x_hat changes L_low although the internal LL is unchanged (not mean|LL_syn - LL_t|)
        yy, xx = torch.meshgrid(torch.arange(256.), torch.arange(256.), indexing='ij')
        perturbed = out.x_hat + 0.1 * torch.sin(2 * math.pi * (xx + 2 * yy) / 64)
        self.assertGreater(float(losses.l_low(perturbed, ll_t)), 1e-3)
        self.assertEqual(float((out.LL_syn - ll_t).abs().mean()), 0.0)
        self.assertGreater(float(losses.l_low(out.x_hat + 0.2, ll_t)), 0.39)     # LL of a +0.2 shift = +0.4
        # 7. N-08 selection LFErr stays exact zero from the internal LL
        self.assertTrue(torch.equal(losses.lferr_selection(out.LL_syn, ll_t), torch.zeros(1)))
        self.assertTrue(torch.equal(losses.lferr_selection(out.LL_syn, ll_t), self.rc.lferr_internal(out.LL_syn, ll_t)))

    def test_43_end_to_end_gradient_structure(self):
        """One backward through the whole B3 core (N=1): E_art, G_res, heads receive finite gradients."""
        torch, F = self.torch, self.F
        from methods.gpat import losses, spectral
        model = copy.deepcopy(self.b3).train()
        with torch.no_grad():
            for blk in (m for m in model.g_res.modules() if type(m).__name__ == 'NAFBlock'):
                blk.beta.fill_(0.1)
                blk.gamma.fill_(0.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            out = model(self.x_s, self.x_t, scale_hf=0.15)
            loss = (losses.l_tv(out.M, out.delta_LH, out.delta_HL, out.delta_HH) + losses.l_budget(out.A)
                    + spectral.spec_loss(out.x_hat, self.x_s) + losses.l_type(out.attack_logits, torch.tensor([4]))
                    + losses.idadv_group(*[[v] for v in losses.idadv_microbatch(out.identity_logits, torch.tensor([7]),
                                                                                torch.tensor([True]))])
                    + F.mse_loss(out.x_hat, self.x_t))
            loss.backward()
        for name in ('e_art', 'g_res', 'attack_head', 'identity_head'):
            grads = [p.grad for p in getattr(model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), name)
        self.assertGreater(float(model.g_res.ending.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.e_art.conv1.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.discriminator.model[0].weight.grad)


if __name__ == '__main__':
    unittest.main()
