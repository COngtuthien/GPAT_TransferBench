"""M6D6c E07c: whole-module auxiliary checkpoint serialization / loader-compatibility evidence and guards.

CPU/static checks never claim CUDA execution. The three separately launched
methods/difffas/aux_checkpoint_qualification.py processes (1 writer, 2 loaders)
produced the evidence validated here. Guard tests replace torch with a sentinel
so any torch.load/torch.save reached is an immediate failure; the optional live
Torch test constructs the encoder but performs NO torch.save and NO torch.load,
so the qualification checkpoint counts (1 save, 2 x 5 loads) stay exact.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import yaml

from methods.common.config import sha256_file
from methods.common.learned import PreparationError
from methods.difffas import DiffFASAdapter
from methods.difffas import aux_checkpoint as seam
from methods.difffas.aux_checkpoint_qualification import (BATCH, BUILD_PARTS, CHECKPOINT_RELATIVE, ELIGIBILITY, K,
                                                          LABEL, OUTPUT_SHAPES, OUTPUTS, PRECISION_POLICY, PROBES, ROOT,
                                                          SEED, upstream_serialization_semantics)
from methods.difffas.contract import validate_contract
from methods.difffas.runtime_qualification import ALLOWED_EXACT_ARGV, Firewall

PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
PRETRAIN_SHA = '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'
CONSUMER_SHA = '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
CHECKPOINT_SHA = 'cf89680a56cdb257e9739fb0ded55522631a9f8291719d972a05c1168460522b'
CHECKPOINT_BYTES = 185141005
HARNESS = ROOT / 'methods/difffas/aux_checkpoint_qualification.py'
SEAM = ROOT / 'methods/difffas/aux_checkpoint.py'
WRITER = ROOT / 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_WRITER.json'
LOADERS = [ROOT / f'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_LOADER_PROCESS_{i}.json' for i in (1, 2)]
LOG = ROOT / 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_RUNTIME_LOG.txt'
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')
WEIGHTS_ONLY = 'WEIGHTS_ONLY_UNPICKLER_REJECTS_CUSTOM_MODULE_GLOBAL'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def sentinel_torch():
    """A stand-in 'torch' whose serialization entry points fail the test if ever reached."""
    fake = types.ModuleType('torch')

    def reached(*args, **kwargs):
        raise AssertionError('torch serialization reached before the SHA256/path gate')
    fake.load = fake.save = reached
    return fake


class TestE07cAuxCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.writer = json.loads(WRITER.read_bytes())
        cls.raw = [p.read_bytes() for p in LOADERS]
        cls.loaders = [json.loads(r) for r in cls.raw]
        cls.runs = [cls.writer, *cls.loaders]
        cls.adapter = DiffFASAdapter()
        cls.contracts = validate_contract(cls.adapter.config)

    # 1 source pin ----------------------------------------------------------------------
    def test_source_pin_unchanged(self):
        local = self.adapter.validate_source()
        self.assertEqual((local['commit'], local['tree']), (PIN, TREE))
        self.assertEqual(local['files_sha256']['models/custom_rn.py'], CUSTOM_RN_SHA)
        self.assertEqual(local['files_sha256']['models/pretrain_classifier.py'], PRETRAIN_SHA)
        self.assertEqual(local['files_sha256']['models/unet_autoenc.py'], CONSUMER_SHA)
        pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']['difffas']
        self.assertEqual((pins['pinned_commit'], pins['commit_tree']), (PIN, TREE))
        for r in self.runs:
            self.assertEqual(r['source_before'], r['source_after'])
            self.assertEqual(r['source_before']['files_sha256'], local['files_sha256'])
            self.assertEqual(r['source_before']['worktree_status'], '')
            self.assertEqual(r['source_patch'], 'NONE')

    # 2 environment lock ----------------------------------------------------------------
    def test_environment_lock_unchanged(self):
        self.assertEqual(sha256_file(ROOT / 'environments/e07c.lock.json'), LOCK_SHA)
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
        for r in self.runs:
            env = r['environment_before']
            self.assertEqual(env, r['environment_after'])
            self.assertEqual({k: v for k, v in env.items() if k != 'launch_environment'},
                             {k: v for k, v in lock['identity'].items() if k != 'launch_environment'})
            self.assertEqual(strip(env['launch_environment']), strip(lock['identity']['launch_environment']))
            self.assertEqual(env['launch_environment']['PYTHONHASHSEED'], str(SEED))
            self.assertIn('/gpat-m6-e07c/', env['executable'])

    # 3, 4 frozen whole-module format; no state_dict substitution -----------------------
    def test_frozen_checkpoint_format_is_whole_module(self):
        fmt = self.adapter.config['conditioning_encoder']['checkpoint_format']
        self.assertEqual(fmt, 'torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())')
        self.assertEqual(seam.FROZEN_FORMAT, fmt)
        a3 = (ROOT / 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md')
        self.assertIn('Checkpoint format is preserved: `torch.save(model)` of the **whole `nn.Module`**', a3.read_text())
        for r in self.runs:
            self.assertEqual(r['checkpoint_format_authority']['value'], fmt)
        self.assertEqual(self.writer['checkpoint']['format'], fmt)
        self.assertEqual(self.writer['checkpoint']['serialization_api'], 'torch.save(model, path)')

    def test_state_dict_substitution_forbidden(self):
        for path in (SEAM, HARNESS):
            tree = ast.parse(path.read_text())
            names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
                     if isinstance(n, (ast.Attribute, ast.Name))}
            self.assertFalse({'state_dict', 'load_state_dict', 'safetensors', 'jit', 'onnx', 'save_file',
                              'add_safe_globals', 'safe_globals'} & names, path.name)
        tree = ast.parse(SEAM.read_text())
        saves = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save']
        self.assertEqual(saves, ['torch.save(model, str(path))'])
        archive = self.writer['checkpoint']['archive']
        self.assertIn('custom_rn.ResNet', archive['globals'])
        self.assertIn('custom_rn.BasicBlock', archive['globals'])
        self.assertEqual(archive['format'], 'torch zip archive')

    # 5, 6 qualification seed only --------------------------------------------------------
    def test_qualification_seed_only(self):
        self.assertEqual(SEED, 60603)
        self.assertNotIn(SEED, (42, 1337, 2026))
        self.assertEqual(self.contracts['effective_encoder']['auxiliary_encoder_training_seed'], 42)
        for r in self.runs:
            self.assertEqual((r['qualification_seed'], r['label']), (SEED, LABEL))
            self.assertIsNone(r['auxiliary_encoder_training_seed'])
            self.assertIsNone(r['experiment_seed'])
            self.assertFalse(r['scientific_seed_consumed'])

    # 7 writer semantics ------------------------------------------------------------------
    def test_writer_uses_whole_module_torch_save(self):
        w = self.writer
        self.assertEqual(w['counters']['torch_save_calls'], 1)
        self.assertEqual(w['counters']['torch_load_calls'], 0)
        save = [e for e in w['writer']['save_trace'] if e[0] == 'torch.save']
        self.assertEqual(save, [['torch.save', {'extra_arguments': False, 'object': 'custom_rn.ResNet'}]])
        self.assertEqual(w['writer']['training_flags_at_save'], [True])
        self.assertEqual(w['writer']['state_before_save']['devices'], ['cuda:0'])
        up = w['upstream_serialization_semantics']
        self.assertEqual((up['writer']['statement'], up['writer']['line'], up['writer']['cuda_before_save_line']),
                         ("torch.save(resnet18, './PADISI.pkl')", 45, 17))

    def test_upstream_semantics_recomputed_from_pinned_source(self):
        up = upstream_serialization_semantics(self.adapter.validate_source())
        self.assertEqual(up, self.writer['upstream_serialization_semantics'])
        self.assertEqual(up['consumer']['torch_load_arguments'], 'path only (no weights_only, no map_location)')
        self.assertEqual(up['caller']['statements']['encoder.eval()'], 35)
        self.assertFalse(up['imported_or_executed'] or up['state_dict_anywhere_in_writer_or_consumer_encoder'])

    # 8, 9 path outside runs/, SHA computed ------------------------------------------------
    def test_qualification_checkpoint_outside_runs_with_sha(self):
        self.assertEqual(BUILD_PARTS, ('builds', 'e07c_difffas', 'm6d6c'))
        path = self.writer['checkpoint']['path']
        self.assertEqual(path, '<runtime_root>/builds/e07c_difffas/m6d6c/' + '/'.join(CHECKPOINT_RELATIVE))
        self.assertNotIn('/runs/', path)
        self.assertNotIn('seed_42', path)
        self.assertNotIn('encoder_final', path)
        self.assertEqual((self.writer['checkpoint']['sha256'], self.writer['checkpoint']['size_bytes']),
                         (CHECKPOINT_SHA, CHECKPOINT_BYTES))
        for name, flag in ELIGIBILITY.items():
            self.assertTrue(flag)
            for r in self.runs:
                self.assertIs(r[name], True, name)
            self.assertIs(self.writer['checkpoint'][name], True)
        for r in self.loaders:
            self.assertEqual(r['checkpoint_consumed']['sha256'], CHECKPOINT_SHA)
            self.assertEqual(r['firewall']['checkpoint_opens']['write'], 0)

    # 10 wrong SHA rejected before torch.load --------------------------------------------
    def test_wrong_sha_rejected_before_torch_load_evidence(self):
        for r in self.loaders:
            rej = r['rejection_before_deserialization']
            wrong = rej['wrong_sha256_one_hex_digit']
            self.assertEqual(wrong['events'], ['open_checkpoint', 'sha256_rejected'])
            self.assertEqual((wrong['torch_load_calls'], wrong['find_class_events']), (0, 0))
            self.assertEqual(wrong['message'], 'auxiliary checkpoint SHA256 mismatch; refusing to deserialize')
            self.assertEqual(rej['malformed_sha256']['events'], [])
            order = r['seam_load']['event_order']
            self.assertEqual(order[:3], ['open_checkpoint', 'sha256_verified', 'torch.load'])
            self.assertTrue(order[3].startswith('find_class x'))
            self.assertEqual(r['seam_load']['torch_load_call'], {'kwargs': {'weights_only': 'False'}, 'source': 'bytes'})

    def test_wrong_sha_guard_never_reaches_torch(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': sentinel_torch()}):
            path = Path(tmp) / 'not_a_checkpoint.pkl'
            path.write_bytes(b'GPAT M6D6c guard bytes; never unpickled')
            good = hashlib.sha256(path.read_bytes()).hexdigest()
            wrong = good[:-1] + ('1' if good[-1] == '0' else '0')
            for bad in (wrong, 'not-a-sha', good.upper(), None):
                with self.assertRaises(PreparationError):
                    with seam.load_verified_whole_module(path, bad):
                        self.fail('must not yield')
            with self.assertRaises(PreparationError):  # missing file
                seam.read_verified(Path(tmp) / 'absent.pkl', good)
            with self.assertRaises(PreparationError):  # repository-internal path refused
                seam.read_verified(SEAM, sha256_file(SEAM))
            self.assertEqual(seam.read_verified(path, good), path.read_bytes())  # bytes only; nothing deserialized
            with self.assertRaises(PreparationError):  # frozen scientific path absent -> hash gate, no load
                with seam.load_frozen_aux_encoder(tmp, good):
                    self.fail('must not yield')
            with self.assertRaises(PreparationError):  # writer refuses repository paths before torch.save
                seam.save_whole_module(object(), ROOT / 'outputs/never.pkl')

    # 11 custom_rn identity during unpickle ------------------------------------------------
    def test_custom_rn_identity_during_unpickle(self):
        for r in self.loaders:
            m = r['probes']['weights_only_false_without_pinned_module']
            self.assertEqual((m['outcome'], m['category'], m['exception_class']),
                             ('REJECTED', 'PINNED_CUSTOM_RN_MODULE_NOT_IMPORTABLE', 'builtins.ModuleNotFoundError'))
            s = r['seam_load']
            self.assertIn('custom_rn.ResNet', s['find_class_globals'])
            self.assertLessEqual(set(s['find_class_globals']), set(self.writer['checkpoint']['archive']['globals']))
            self.assertEqual(s['custom_rn_live_module_file'], 'third_party/source_cache/difffas/models/custom_rn.py')
            self.assertTrue(s['class_is_pinned_module_class'])
        text = SEAM.read_text()
        self.assertIn("upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'})", text)

    # 12, 13, 14 observed torch.load behaviours -------------------------------------------
    def test_torch_load_behaviour_recorded_not_assumed(self):
        self.assertEqual([p[0] for p in PROBES], ['default_no_argument', 'weights_only_true', 'weights_only_false',
                                                  'weights_only_false_without_pinned_module'])
        for r in self.runs:
            self.assertIn('weights_only: bool | None = None', r['torch_load_signature'])
            self.assertEqual(r['torch_load_weights_only_default'], 'None')
            self.assertEqual(r['torch_load_env_overrides'], {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': None,
                                                            'TORCH_FORCE_WEIGHTS_ONLY_LOAD': None})
        for r in self.loaders:
            p = r['probes']
            for name in ('default_no_argument', 'weights_only_true'):
                self.assertEqual((p[name]['outcome'], p[name]['category'], p[name]['exception_class']),
                                 ('REJECTED', WEIGHTS_ONLY, '_pickle.UnpicklingError'), name)
                self.assertEqual(p[name]['find_class_globals'], [])
            self.assertEqual(p['default_no_argument']['call'], 'torch.load(path)')
            f = p['weights_only_false']
            self.assertEqual((f['outcome'], f['type'], f['devices']), ('LOADED', 'custom_rn.ResNet', ['cuda:0']))
            self.assertTrue(f['is_pinned_custom_rn_ResNet'])
            st = self.writer['writer']['state_before_save']
            self.assertEqual((f['aggregate_parameter_sha256'], f['aggregate_buffer_sha256']),
                             (st['aggregate_parameter_sha256'], st['aggregate_buffer_sha256']))
            self.assertTrue(all(v['sha256_verified_before_call'] for v in p.values()))

    # 15 compatibility branch only because the evidence requires it -----------------------
    def test_compatibility_branch_justified_and_minimal(self):
        self.assertIs(seam.LOAD_COMPATIBILITY['weights_only'], False)
        self.assertEqual(seam.LOAD_COMPATIBILITY['classification'],
                         'RUNTIME_COMPATIBILITY_ADAPTATION_RESTORING_LEGACY_WHOLE_MODULE_LOAD_SEMANTICS')
        self.assertFalse(seam.LOAD_COMPATIBILITY['scientific_adaptation'] or
                         seam.LOAD_COMPATIBILITY['checkpoint_format_changed'] or
                         seam.LOAD_COMPATIBILITY['source_modified'] or seam.LOAD_COMPATIBILITY['global_torch_load_patch'])
        for r in self.loaders:
            d = r['compatibility_decision']
            self.assertEqual((d['observed'], d['selected']), ('DEFAULT_AND_WEIGHTS_ONLY_TRUE_REJECT_WEIGHTS_ONLY_FALSE_RESTORES',
                                                              'EXPLICIT_WEIGHTS_ONLY_FALSE_AFTER_SHA256_VERIFICATION'))
        tree = ast.parse(SEAM.read_text())
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
        self.assertEqual([ast.unparse(n) for n in loads],
                         ["torch.load(io.BytesIO(raw), weights_only=LOAD_COMPATIBILITY['weights_only'])"])
        patched = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets]
        self.assertFalse([t for t in patched if t.startswith('torch.')])
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_verified_whole_module')
        order = [ast.unparse(s) for s in fn.body]
        self.assertLess(order.index('raw = read_verified(path, expected_sha256)'), order.index('import torch'))

    # 16, 17, 18, 19 loaded identity, parameters, buffers -----------------------------------
    def test_loaded_identity_parameters_and_buffers(self):
        ref = self.writer['writer']
        st = ref['state_before_save']
        for r in self.loaders:
            s = r['seam_load']
            self.assertEqual(s['identity'], ref['identity'])
            self.assertEqual((s['identity']['module'], s['identity']['class'], s['identity']['topology']),
                             ('custom_rn', 'ResNet', [3, 4, 6, 3]))
            self.assertEqual(s['fc'], {'in_features': 512, 'out_features': 7})
            self.assertEqual((s['parameters'], s['parameter_tensors'], s['buffers']),
                             (st['parameters'], st['parameter_tensors'], st['buffers']))
            self.assertEqual(s['parameters'], 46233707)
            self.assertEqual(s['parameter_tensors_bitwise_equal'], len(st['parameter_sha256']))
            self.assertEqual(s['buffer_tensors_bitwise_equal'], len(st['buffer_sha256']))
            self.assertEqual((s['aggregate_parameter_sha256'], s['aggregate_buffer_sha256']),
                             (st['aggregate_parameter_sha256'], st['aggregate_buffer_sha256']))
            self.assertEqual((s['parameter_devices'], s['buffer_devices']), (['cuda:0'], ['cuda:0']))
            self.assertEqual((s['parameter_dtypes'], s['buffer_dtypes']), (['torch.float32'], ['torch.float32', 'torch.int64']))
            self.assertTrue(s['module_tree_equal'])
            self.assertFalse(s['projection_or_adapter_modules_added'])
            self.assertEqual(s['training_flags_after_load'], ref['training_flags_at_save'])

    # 20, 21 A6 features and bitwise forward equivalence -------------------------------------
    def test_round_trip_forward_bitwise(self):
        ref = self.writer['reference_forward']
        for r in self.loaders:
            f = r['round_trip_forward']
            self.assertEqual(f['bitwise_equal'], {n: True for n in OUTPUTS})
            self.assertEqual(f['tolerance_used'], 'NONE (bitwise)')
            self.assertEqual(f['input'], ref['input'])
            for n in OUTPUTS:
                self.assertEqual(f['outputs'][n]['sha256'], ref['outputs'][n]['sha256'])
                self.assertEqual(f['outputs'][n]['shape'], OUTPUT_SHAPES[n])
            self.assertEqual(f['a6_feature_shapes'], [[BATCH, 256, 32, 32], [BATCH, 512, 16, 16], [BATCH, 512, 8, 8]])
            self.assertEqual(f['fourth_output_shape'], [BATCH, K])
        self.assertEqual({k: v['shape'] for k, v in self.contracts['effective_encoder']['feature_interface_at_256'].items()},
                         {'x32x32': [32, 32, 256], 'x16x16': [16, 16, 512], 'x8x8': [8, 8, 512], 'embg': ['B', 7]})

    def test_two_loader_processes_bitwise_identical(self):
        self.assertEqual(self.raw[0], self.raw[1])

    # 22, 23 no training --------------------------------------------------------------------
    def test_no_backward_no_optimizer_step(self):
        expected = {'optimizer_constructions': 0, 'optimizer_step_calls': 0, 'backward_calls': 0,
                    'autograd_grad_calls': 0}
        for r in self.runs:
            self.assertEqual({k: r['counters'][k] for k in expected}, expected)
            self.assertEqual((r['backward_calls'], r['optimizer_step_calls'], r['epochs']), (0, 0, 0))
            self.assertFalse(r['auxiliary_encoder_trained'] or r['scientific_training'] or
                             r['main_difffas_training'] or r['main_difffas_model_constructed'])
        for r in self.loaders:
            self.assertEqual((r['counters']['torch_load_calls'], r['counters']['torch_save_calls']), (len(PROBES) + 1, 0))

    # 24 zero benchmark access -----------------------------------------------------------------
    def test_zero_benchmark_access(self):
        for r in self.runs:
            fw = r['firewall']
            self.assertEqual(fw['denied'], [])
            self.assertEqual(fw['attempts'], {k: 0 for k in Firewall.CLASSES})
            self.assertFalse(r['benchmark_data_access'] or r['TRAIN_access'] or r['VAL_access'] or r['TEST_access'])
            self.assertTrue(all(Path(a[0]).name in ('git', 'nvidia-smi') or a in ALLOWED_EXACT_ARGV
                                for a in fw['subprocesses']))
        for path in (HARNESS, SEAM):
            text = path.read_text()
            for token in ('manifests/', '.parquet', 'faces_256', 'ImageFolder', 'DataLoader(', 'artifact_probe_classes'):
                self.assertNotIn(token, text)

    # 25, 26 no scientific checkpoint; qualification checkpoint removed ------------------------
    def test_no_scientific_checkpoint_and_cleanup(self):
        for r in self.runs:
            self.assertFalse(r['scientific_checkpoint_created'] or r['scientific_checkpoint_loaded'])
            self.assertEqual(r['firewall']['attempts']['scientific_run_root'], 0)
        for text in (HARNESS.read_text(), SEAM.read_text()):
            self.assertNotIn('runs/m6', text)
        log = LOG.read_text()
        after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
        self.assertEqual(after, ['scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT '
                                 'encoder_final=ABSENT'] * 4)  # disclosed writer attempt 1 + writer + 2 loaders
        for token in ('QUALIFICATION_PKL_ABSENT', 'ENCODER_FINAL_ABSENT', 'RUNS_ABSENT'):
            self.assertIn(token, log)

    # 27 precision policy not invented -------------------------------------------------------
    def test_precision_policy_not_silently_invented(self):
        self.assertEqual(PRECISION_POLICY, 'PRECISION_POLICY_DECISION_REQUIRED')
        for r in self.runs:
            p = r['precision']
            self.assertEqual((p['label'], p['production_policy']), ('ENGINEERING_QUALIFICATION_CONTROLS', PRECISION_POLICY))
            self.assertFalse(p['freezes_scientific_precision_policy'])
        for rel in ('configs/methods/e07c_difffas_bin_idfree.yaml', 'configs/frozen/difffas_bin_idfree_v1.yaml'):
            cfg = yaml.safe_load((ROOT / rel).read_text())
            keys = set()
            stack = [cfg]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    keys |= set(node)
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
            tokens = {'precision', 'tf32', 'amp', 'autocast', 'fp16', 'bf16', 'gradscaler'}  # whole '_' tokens
            self.assertFalse({k for k in keys if set(str(k).lower().split('_')) & tokens}, rel)

    # live Torch (GPU host, CUDA hidden): construction + guards; NO torch.save / NO torch.load
    @unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; CUDA evidence validated from process JSON')
    def test_live_identity_and_guards_without_serialization(self):
        import torch
        from methods.difffas.encoder import encoder_model
        calls = []
        with mock.patch.object(torch, 'save', side_effect=lambda *a, **k: calls.append('save')), \
                mock.patch.object(torch, 'load', side_effect=lambda *a, **k: calls.append('load')), \
                torch.random.fork_rng(devices=[]), encoder_model(self.adapter.config) as enc:
            torch.manual_seed(SEED)
            custom_rn = sys.modules[type(enc).__module__]
            identity = seam.encoder_identity(enc, custom_rn, self.contracts)
            self.assertEqual(identity, self.writer['writer']['identity'])
            with self.assertRaises(PreparationError):
                seam.save_whole_module(enc, ROOT / 'outputs/never.pkl')
            enc.fc = torch.nn.Linear(512, 17)  # the pinned 17-way head is NOT the A3 K=7 encoder
            with self.assertRaises(PreparationError):
                seam.encoder_identity(enc, custom_rn, self.contracts)
        self.assertEqual(calls, [])
        self.assertNotIn('custom_rn', sys.modules)


if __name__ == '__main__':
    unittest.main()
