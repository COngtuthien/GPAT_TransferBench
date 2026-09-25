"""M6D6d E07c: A7 execution-policy authority, RNG-compatibility helper and process evidence.

CPU/static checks never claim CUDA execution. The two separately launched
methods/difffas/execution_policy_qualification.py processes produced the evidence
validated here. The optional live Torch tests (GPU host, CUDA hidden) replay the
throwaway constructor on the CPU inside torch.random.fork_rng and restore every
backend flag they touch; they perform no torch.save and no torch.load.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from methods.common.config import FrozenConfigError, sha256_file
from methods.difffas import DiffFASAdapter
from methods.difffas import execution_policy as ep
from methods.difffas.contract import validate_contract
from methods.difffas.execution_policy_qualification import (BUILD_PARTS, ENCODER_SHAPES, LABEL, MAIN_OUTPUT_SHAPE,
                                                            PATHS, ROOT, SEED)
from methods.difffas.runtime_qualification import ALLOWED_EXACT_ARGV, Firewall

AUTHORITY = 'e70c221464b4a2db4ea3d45176a0cee52a7747cf'
PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
A7_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md'
A7_DOC_SHA = '0a46c3b06e277ad38b17a6e85fe2924441d8d13b1de887bf54903dda9aa292ba'
A7_OVERLAY_SHA = '3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
HISTORICAL = {
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md':
        '03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md':
        'b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md':
        'b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a',
    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md':
        '759f72d860ccf49927c214bb4ce84e87bcedd98eb8f06ef5fe7361a758d6bc81',
    'configs/amendments/e07c_a6_feature_interface_source_correction.yaml':
        'dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b',
    'configs/methods/e07c_difffas_bin_idfree.yaml': 'dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c',
    'configs/frozen/difffas_bin_idfree_v1.yaml': 'aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb',
}
HELPER = ROOT / 'methods/difffas/execution_policy.py'
HARNESS = ROOT / 'methods/difffas/execution_policy_qualification.py'
PROCESSES = [ROOT / f'outputs/audit/M6D6D_E07C_RNG_PROCESS_{i}.json' for i in (1, 2)]
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')
REPLAY = ('helper', 'direct', 'call_site', 'helper_ctx')
HAS_TORCH = importlib.util.find_spec('torch') is not None


def function(name):
    tree = ast.parse(HELPER.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node):
    return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


class TestE07cExecutionPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = [p.read_bytes() for p in PROCESSES]
        cls.runs = [json.loads(r) for r in cls.raw]
        cls.adapter = DiffFASAdapter()
        cls.contracts = validate_contract(cls.adapter.config)
        cls.policy = yaml.safe_load((ROOT / ep.A7_OVERLAY).read_text())

    # 1 authority -------------------------------------------------------------------------
    def test_authority_commit_is_ancestor(self):
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', AUTHORITY, 'HEAD'], check=True)

    # 2 source pin ------------------------------------------------------------------------
    def test_source_pin_unchanged(self):
        local = self.adapter.validate_source()
        self.assertEqual((local['commit'], local['tree']), (PIN, TREE))
        pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']['difffas']
        self.assertEqual((pins['pinned_commit'], pins['commit_tree']), (PIN, TREE))
        for rel, digest in self.policy['source']['files_sha256'].items():
            self.assertEqual(local['files_sha256'][rel], digest, rel)
        for r in self.runs:
            self.assertEqual(r['source_before'], r['source_after'])
            self.assertEqual(r['source_before']['files_sha256'], local['files_sha256'])

    # 3 A7 authority exists and hashes ----------------------------------------------------
    def test_a7_authority_hashes(self):
        self.assertEqual(ep.A7_OVERLAY_SHA256, A7_OVERLAY_SHA)
        self.assertEqual(sha256_file(ROOT / ep.A7_OVERLAY), A7_OVERLAY_SHA)
        self.assertEqual(sha256_file(ROOT / A7_DOC), A7_DOC_SHA)
        self.assertEqual(self.policy['amendment_document'], {'path': A7_DOC, 'sha256': A7_DOC_SHA})
        self.assertEqual((self.policy['classification'], self.policy['provenance']),
                         ('DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'OWNER_FROZEN_BENCHMARK_EXECUTION_POLICY'))
        self.assertFalse(self.policy['upstream_author_choice_claimed'])
        loaded = ep.load_policy(self.adapter.config)
        self.assertEqual(loaded['sha256'], A7_OVERLAY_SHA)
        for r in self.runs:
            self.assertEqual((r['a7']['overlay_sha256'], r['a7']['document']['sha256']), (A7_OVERLAY_SHA, A7_DOC_SHA))

    def test_tampered_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'a7.yaml'
            shutil.copyfile(ROOT / ep.A7_OVERLAY, copy)
            copy.write_bytes(copy.read_bytes().replace(b'cudnn_tf32: false', b'cudnn_tf32: true '))
            with self.assertRaises(FrozenConfigError):
                ep.load_policy(self.adapter.config, overlay_path=copy)

    # 4 historical amendments unchanged ----------------------------------------------------
    def test_historical_authority_byte_identical(self):
        for rel, digest in HISTORICAL.items():
            raw = (ROOT / rel).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest, rel)
            committed = subprocess.run(['git', '-C', str(ROOT), 'show', f'{AUTHORITY}:{rel}'],
                                       check=True, capture_output=True).stdout
            self.assertEqual(raw, committed, rel)
        self.assertEqual((self.policy['a6_overlay']['sha256'], self.policy['amendment_a3']['sha256'],
                          self.policy['base_config_sha256']),
                         (HISTORICAL['configs/amendments/e07c_a6_feature_interface_source_correction.yaml'],
                          HISTORICAL['docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md'],
                          HISTORICAL['configs/methods/e07c_difffas_bin_idfree.yaml']))

    # 5, 6 fidelity / deviation ----------------------------------------------------------------
    def test_fidelity_and_deviation_unchanged(self):
        cfg = self.adapter.config
        self.assertEqual((cfg['fidelity_class'], cfg['deviation']), ('CONTROLLED_ADAPTATION', 'DEV-021'))
        f = self.policy['fidelity']
        self.assertEqual((f['fidelity_class'], f['deviation']), ('CONTROLLED_ADAPTATION', 'DEV-021'))
        self.assertFalse(f['new_fidelity_class'] or f['new_deviation'])
        for r in self.runs:
            self.assertEqual((r['fidelity'], r['contract']['fidelity_class'], r['contract']['deviation']),
                             ('CONTROLLED_ADAPTATION', 'CONTROLLED_ADAPTATION', 'DEV-021'))

    # 7-13 frozen precision / cuDNN policy -------------------------------------------------------
    def test_precision_policy_frozen(self):
        p, c = self.policy['precision'], self.policy['cudnn']
        self.assertEqual({k: p[k] for k in ep.PRECISION}, {'dtype': 'float32', 'autocast': False, 'grad_scaler': False,
                                                            'matmul_tf32': False, 'cudnn_tf32': False})
        self.assertEqual(ep.PRECISION, {k: p[k] for k in ep.PRECISION})
        self.assertEqual(p['forbidden'], ['AMP', 'FP16', 'BF16', 'mixed_precision'])
        self.assertEqual({k: c[k] for k in ep.CUDNN}, {'benchmark': False, 'deterministic': True})
        self.assertEqual(ep.CUDNN, self.adapter.config['training']['cudnn'])
        self.assertEqual(c['applies_to'], ['auxiliary_encoder_production', 'main_difffas_production'])
        self.assertEqual(self.policy['use_deterministic_algorithms'], 'NOT_SET')
        self.assertFalse(self.policy['launch_environment_changed'])

    def test_precision_policy_observed_in_processes(self):
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        for r in self.runs:
            p = r['precision']
            self.assertEqual(p['applied_state'], ep.EXPECTED_STATE)
            self.assertEqual(p['state_after_all_sections'], ep.EXPECTED_STATE)
            self.assertEqual(p['expected_state'], ep.EXPECTED_STATE)
            self.assertEqual(p['library_defaults_before_policy']['cudnn_tf32'], True)  # why TF32 must be explicit
            self.assertFalse(p['grad_scaler_constructed'] or p['autocast_entered'])
            self.assertEqual((r['counters']['autocast_entries'], r['counters']['grad_scaler_constructions']), (0, 0))
            self.assertEqual(r['runtime_forward']['precision_state_inside_forwards'], ['encoder', 'main_model'])
            self.assertEqual(r['upstream_precision_requests'], [])
            env = r['environment_before']
            self.assertEqual(env, r['environment_after'])
            self.assertEqual({k: v for k, v in env.items() if k != 'launch_environment'},
                             {k: v for k, v in lock['identity'].items() if k != 'launch_environment'})
            strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
            self.assertEqual(strip(env['launch_environment']), strip(lock['identity']['launch_environment']))
            self.assertEqual(env['launch_environment']['PYTHONHASHSEED'], str(SEED))
        self.assertEqual(sha256_file(ROOT / 'environments/e07c.lock.json'), LOCK_SHA)

    def test_apply_sets_only_the_frozen_flags(self):
        fn = function('apply_e07c_precision_policy')
        assigned = sorted(ast.unparse(t) for n in ast.walk(fn) if isinstance(n, ast.Assign) for t in n.targets
                          if ast.unparse(t).startswith('torch.'))
        self.assertEqual(assigned, ['torch.backends.cuda.matmul.allow_tf32', 'torch.backends.cudnn.allow_tf32',
                                    'torch.backends.cudnn.benchmark', 'torch.backends.cudnn.deterministic'])
        names = set(calls(ast.parse(HELPER.read_text())))
        self.assertIn('torch.set_default_dtype', names)
        for forbidden in ('torch.use_deterministic_algorithms', 'torch.autocast', 'torch.amp.GradScaler',
                          'torch.cuda.amp.GradScaler', 'torch.set_float32_matmul_precision'):
            self.assertNotIn(forbidden, names)

    # 14-17 RNG-compatibility decision and exact constructor --------------------------------------
    def test_throwaway_rng_preservation_frozen(self):
        rng = self.policy['rng_compatibility']
        self.assertIs(rng['preserve_upstream_throwaway_encoder_constructor_rng'], True)
        self.assertEqual((rng['constructor'], rng['device'], rng['count'], rng['role']),
                         ('custom_rn.resnet18(pretrained=False)', 'cpu', 1, 'RNG_COMPATIBILITY_CONSUMPTION'))
        self.assertFalse(rng['conditioning_role'] or rng['object_used'] or rng['object_moved_to_gpu'] or
                         rng['object_saved'] or rng['object_retained'] or rng['object_returned'])
        self.assertEqual(rng['sampler_FAS_sample_py_48'], 'NOT_DECIDED_BY_A7')
        sem = ep.upstream_loader_semantics(self.adapter.validate_source())
        self.assertEqual(sem['consumer']['body'][0], ['model_autoencoder = resnet18()', 74])
        self.assertEqual(sem['constructor']['defaults'], {'pretrained': False, 'progress': True})
        for r in self.runs:
            self.assertEqual(r['upstream_loader_semantics'], sem)

    def test_helper_uses_exact_pinned_constructor_only(self):
        fn = function('consume_upstream_encoder_loader_rng')
        text = ast.unparse(fn)
        ctor = [c for c in calls(fn) if c.endswith('resnet18')]
        self.assertEqual(ctor, ["modules['custom_rn'].resnet18"])
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith('resnet18'))
        self.assertEqual(ast.unparse(call), "modules['custom_rn'].resnet18(pretrained=False)")
        self.assertIn("upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'})", text)
        stmt = next(n for n in ast.walk(fn) if isinstance(n, ast.Expr) and n.value is call)
        self.assertIsInstance(stmt, ast.Expr)  # bare expression statement: result never bound
        for token in ('torchvision', '.cuda', '.to(', 'torch.load', 'torch.save', 'open(', 'read_bytes',
                      'fc =', 'Linear', 'state_dict'):
            self.assertNotIn(token, text, token)

    def test_no_manual_random_draw_substitute(self):
        for name in ('consume_upstream_encoder_loader_rng', 'main_runner_encoder', 'apply_e07c_precision_policy'):
            names = calls(function(name))
            for forbidden in ('torch.rand', 'torch.randn', 'torch.randint', 'torch.randperm', 'torch.manual_seed',
                              'torch.set_rng_state', 'torch.get_rng_state', 'random.random', 'np.random.seed',
                              'torch.random.fork_rng', 'torch.Generator'):
                self.assertNotIn(forbidden, names, f'{name}: {forbidden}')

    # 18-20 RNG equivalence evidence ---------------------------------------------------------------
    def test_rng_state_transition_equal(self):
        for r in self.runs:
            q = r['rng_equivalence']
            self.assertEqual(q['order_of_paths'], list(PATHS))
            self.assertEqual(q['initial_seeding'], f'torch.manual_seed({SEED})')
            paths = q['paths']
            for name in PATHS:
                self.assertEqual(paths[name]['before']['cpu_sha256'], q['initial_cpu_rng_sha256'], name)
                for k in ('cuda_sha256', 'python_random_sha256', 'numpy_sha256'):
                    self.assertEqual(paths[name]['before'][k], paths[name]['after'][k], f'{name} {k}')
            for name in REPLAY:
                self.assertTrue(paths[name]['cpu_state_changed'], name)
                self.assertEqual(paths[name]['after']['cpu_sha256'], q['reference_after_cpu_rng_sha256'], name)
            self.assertEqual(paths['direct']['after']['cpu_sha256'], q['reference_after_cpu_rng_sha256'])
            self.assertFalse(paths['bypass']['cpu_state_changed'])
            ci = q['call_site_interception']
            self.assertEqual(ci['caller'], {'file': 'unet_autoenc.py', 'function': 'encoder', 'line': 75})
            self.assertEqual(ci['rng_at_torch_load'], paths['call_site']['after'])
            self.assertFalse(ci['io_performed'] or ci['deserialization'])
            obj = paths['direct']['object']
            self.assertEqual((obj['type'], obj['fc'], obj['devices']), ('custom_rn.ResNet', [512, 17], ['cpu']))
            self.assertEqual((ci['object']['type'], ci['object']['aggregate_parameter_sha256']),
                             ('models.custom_rn.ResNet', obj['aggregate_parameter_sha256']))

    def test_next_random_tensor_equal(self):
        for r in self.runs:
            q = r['rng_equivalence']
            ref = q['reference_next_rand_sha256']
            for name in REPLAY:
                self.assertEqual(q['paths'][name]['next_rand']['sha256'], ref, name)
                self.assertEqual(q['paths'][name]['next_rand']['first4'], q['paths']['direct']['next_rand']['first4'])
            self.assertNotEqual(q['bypass_next_rand_sha256'], ref)  # bypassing is not trajectory-neutral

    def test_throwaway_not_returned_or_retained(self):
        fn = function('consume_upstream_encoder_loader_rng')
        ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
        self.assertIsInstance(ret.value, ast.Dict)
        # literals, or the module-level string constant naming the constructor; never the constructed object
        self.assertTrue(all(isinstance(v, ast.Constant) or (isinstance(v, ast.Name) and v.id == 'THROWAWAY_CONSTRUCTOR')
                            for v in ret.value.values))
        self.assertIsInstance(ep.THROWAWAY_CONSTRUCTOR, str)
        for r in self.runs:
            for name in PATHS:
                p = r['rng_equivalence']['paths'][name]
                self.assertEqual((p['cuda_memory_allocated_delta_bytes'], p['throwaway_instances_after']), (0, 0), name)
            for name in ('helper', 'helper_ctx'):
                p = r['rng_equivalence']['paths'][name]
                self.assertTrue(p['return_value_plain_data'])
                self.assertEqual(p['return_value']['role'], 'RNG_COMPATIBILITY_ONLY')
                self.assertIs(p['return_value']['object_returned'], False)

    def test_two_processes_byte_identical(self):
        self.assertEqual(self.raw[0], self.raw[1])
        for r in self.runs:
            self.assertEqual((r['status'], r['label'], r['milestone']), ('PASS', LABEL, 'M6D6d'))

    # runtime forward under the policy ----------------------------------------------------------
    def test_runtime_forward_under_policy(self):
        for r in self.runs:
            f = r['runtime_forward']
            self.assertTrue(f['finite_all'])
            for name, shape in ENCODER_SHAPES.items():
                self.assertEqual(f['outputs']['encoder_' + name]['shape'], shape)
                self.assertEqual(f['outputs']['encoder_' + name]['dtype'], 'torch.float32')
            self.assertEqual(f['outputs']['main_model_output']['shape'], MAIN_OUTPUT_SHAPE)
            self.assertEqual(f['a6_feature_shapes'], [[4, 256, 32, 32], [4, 512, 16, 16], [4, 512, 8, 8]])
            self.assertFalse(f['architecture_changed'] or f['backward'] or f['optimizer_step'] or f['checkpoint'])

    # main-runner order ------------------------------------------------------------------------------
    def test_main_runner_order_source_derived(self):
        steps = ep.upstream_main_order(self.adapter.validate_source(), self.policy)
        roles = [s['role'] for s in steps]
        order = ['scientific_seed_initialization', 'dataloader_construction', 'main_model_construction',
                 'ema_construction', 'optimizer_construction', 'scheduler_construction', 'diffusion_construction',
                 'throwaway_constructor_rng_consumption', 'secure_frozen_auxiliary_checkpoint_load', 'encoder_eval',
                 'training_iterator_cpu_rng', 'timestep_sampling_cuda_rng']
        self.assertEqual([roles.index(x) for x in order], sorted(roles.index(x) for x in order))
        for r in self.runs:
            self.assertEqual(r['upstream_main_order'], steps)

    # 21 benchmark access ---------------------------------------------------------------------------
    def test_zero_benchmark_access(self):
        for r in self.runs:
            fw = r['firewall']
            self.assertEqual(fw['denied'], [])
            self.assertEqual(fw['attempts'], {k: 0 for k in Firewall.CLASSES})
            self.assertFalse(r['benchmark_data_access'] or r['TRAIN_access'] or r['VAL_access'] or r['TEST_access'])
            self.assertTrue(all(Path(a[0]).name in ('git', 'nvidia-smi') or a in ALLOWED_EXACT_ARGV
                                for a in fw['subprocesses']))
        for path in (HELPER, HARNESS):
            text = path.read_text()
            for token in ('manifests/', '.parquet', 'faces_256', 'ImageFolder', 'DataLoader(', 'artifact_probe_classes'):
                self.assertNotIn(token, text)

    # 22-24 no checkpoint, backward or optimizer ----------------------------------------------------------
    def test_no_checkpoint_backward_or_optimizer(self):
        zero = ('optimizer_constructions', 'optimizer_step_calls', 'backward_calls', 'autograd_grad_calls',
                'torch_save_calls', 'torch_load_calls')
        for r in self.runs:
            self.assertEqual({k: r['counters'][k] for k in zero}, {k: 0 for k in zero})
            self.assertEqual(r['counters']['torch_load_intercepted_before_io'], 1)
            self.assertEqual((r['backward_calls'], r['optimizer_step_calls'], r['epochs'], r['torch_save_calls'],
                              r['torch_load_calls']), (0, 0, 0, 0, 0))
            self.assertFalse(r['scientific_checkpoint_created'] or r['scientific_checkpoint_loaded'] or
                             r['auxiliary_encoder_trained'] or r['scientific_training'] or r['main_difffas_training'])
        self.assertEqual(BUILD_PARTS, ('builds', 'e07c_difffas', 'm6d6d'))
        for text in (HELPER.read_text(), HARNESS.read_text()):
            self.assertNotIn('runs/m6', text)

    # 25, 26 scientific seeds unused --------------------------------------------------------------------
    def test_scientific_seeds_unused(self):
        self.assertEqual(SEED, 60604)
        self.assertNotIn(SEED, (42, 1337, 2026))
        self.assertEqual(self.contracts['effective_encoder']['auxiliary_encoder_training_seed'], 42)
        self.assertEqual(self.adapter.config['seeds']['experiment_seeds'], [42, 1337, 2026])
        q = self.policy['qualification']
        self.assertEqual((q['qualification_seed'], q['scientific_auxiliary_runs'], q['scientific_main_runs']), (SEED, 0, 0))
        for r in self.runs:
            self.assertEqual(r['qualification_seed'], SEED)
            self.assertIsNone(r['auxiliary_encoder_training_seed'])
            self.assertIsNone(r['experiment_seed'])
            self.assertFalse(r['scientific_seed_consumed'])

    # 27 M6D6c secure loader still mandatory --------------------------------------------------------------
    def test_secure_loader_mandatory_and_ordered(self):
        s = self.policy['secure_loader']
        self.assertTrue(s['mandatory'])
        self.assertEqual(s['entry'], 'methods/difffas/aux_checkpoint.py::load_frozen_aux_encoder')
        order = []
        fake_encoder = object()

        class Load:
            def __init__(self, *args):
                order.append(('load', args))

            def __enter__(self):
                return fake_encoder

            def __exit__(self, *exc):
                return False
        with mock.patch.object(ep, 'consume_upstream_encoder_loader_rng', side_effect=lambda c: order.append(('rng', c))), \
                mock.patch.object(ep, 'load_frozen_aux_encoder', Load):
            with ep.main_runner_encoder('/abs/runtime', 'f' * 64, 'CFG') as encoder:
                self.assertIs(encoder, fake_encoder)
        self.assertEqual(order, [('rng', 'CFG'), ('load', ('/abs/runtime', 'f' * 64, 'CFG'))])
        fn = function('main_runner_encoder')
        self.assertEqual([c for c in calls(fn)], ['consume_upstream_encoder_loader_rng', 'load_frozen_aux_encoder'])

    # 28 no source patch ---------------------------------------------------------------------------------
    def test_no_source_patch(self):
        for r in self.runs:
            self.assertEqual(r['source_patch'], 'NONE')
            self.assertEqual(r['source_before']['worktree_status'], '')
        text = HELPER.read_text()
        self.assertNotIn('write_text', text)
        self.assertNotIn('write_bytes', text)

    # live Torch (GPU host, CUDA hidden): CPU RNG replay + flag application; state restored ----------------
    @unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; CUDA evidence validated from process JSON')
    def test_live_cpu_rng_equivalence(self):
        import torch
        from methods.common.upstream import upstream_modules
        source = self.adapter.validate_source()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            initial = torch.get_rng_state()
            ep.consume_upstream_encoder_loader_rng(self.adapter.config)
            helper = (torch.get_rng_state().clone(), torch.rand(8))
            torch.set_rng_state(initial)
            with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
                modules['custom_rn'].resnet18(pretrained=False)
            direct = (torch.get_rng_state().clone(), torch.rand(8))
            torch.set_rng_state(initial)
            bypass = torch.rand(8)
        self.assertTrue(torch.equal(helper[0], direct[0]))
        self.assertTrue(torch.equal(helper[1], direct[1]))
        self.assertFalse(torch.equal(bypass, direct[1]))
        self.assertNotIn('custom_rn', sys.modules)

    @unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host')
    def test_live_apply_policy_and_restore(self):
        import torch
        saved = (torch.get_default_dtype(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.assertEqual(ep.apply_e07c_precision_policy(self.adapter.config), ep.EXPECTED_STATE)
        finally:
            torch.set_default_dtype(saved[0])
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = saved[1:]


if __name__ == '__main__':
    unittest.main()
