"""M6D6h E07c auxiliary conditioning-encoder OWNER FREEZE: freeze record, secure-loader binding, fail-closed order.

Bounded and static: no CUDA, no TRAIN/VAL/TEST access, no training, and the real 185-MB scientific checkpoint is
never opened or deserialized (no synthetic 185-MB file is generated either). Fail-closed and ordering tests use
tiny temporary files outside the repository, a sentinel 'torch' whose load/save fail if ever reached, and mocks.
Historical claims ("M6D6h left X unchanged") are checked at the M6D6h state (the commit that added this file, or
the worktree while it is still a candidate), never by locking a future HEAD.
"""
from contextlib import contextmanager
import ast
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from methods.common.config import load_method_config
from methods.common.learned import PreparationError
from methods.difffas import aux_checkpoint as seam

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'ee0a9b9166577cac606c142af5bc2e338d75eea1'
M6D6F_COMMIT = '7642b23e60a1d89fcd481b03d4c9cb361f4c6e20'
FROZEN_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
FROZEN_BYTES = 185136819
MAIN_SEEDS = [42, 1337, 2026]
RELATIVE = 'runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl'
OVERLAY = ROOT / 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml'
EVIDENCE = ROOT / 'outputs/audit/M6D6H_E07C_AUX_ENCODER_FREEZE.json'
THIS = 'tests/test_m6d6h_e07c_aux_encoder_freeze.py'
B1_TEST = 'tests/test_m6d6g_e07c_aux_scientific_run.py'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6d6h_preflight_under_test',
                                                  ROOT / 'tools/m6d6h_e07c_aux_encoder_freeze_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6h_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6h(rel):
    """File bytes at the M6D6h state: the commit that added this test, else the (candidate) worktree."""
    commit = m6d6h_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def sentinel_torch():
    fake = types.ModuleType('torch')

    def reached(*args, **kwargs):
        raise AssertionError('torch serialization reached before the owner-freeze / SHA256 / size gate')
    fake.load = fake.save = reached
    return fake


def forbidden(name):
    def reached(*args, **kwargs):
        raise AssertionError(name + ' reached before the gate that must reject first')
    return reached


def small_asset(tmp, payload=b'GPAT M6D6h guard bytes; never unpickled'):
    path = Path(tmp) / RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def fake_freeze(sha256, size):
    return lambda config=None: {'sha256': 'patched', 'record': {}, 'frozen_asset': {'sha256': sha256, 'bytes': size}}


class TestM6D6hFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.record = json.loads(OVERLAY.read_bytes())
        cls.cfg = load_method_config('E07c')

    # 1-5 freeze record ------------------------------------------------------------------------
    def test_01_freeze_overlay_sha_matches_pinned_constant(self):
        self.assertEqual(hashlib.sha256(OVERLAY.read_bytes()).hexdigest(), seam.FREEZE_RECORD_SHA256)
        self.assertEqual(seam.FREEZE_RECORD, 'configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml')
        self.assertEqual(seam.load_freeze_record()['sha256'], seam.FREEZE_RECORD_SHA256)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(seam, 'ROOT', Path(tmp)):
            (Path(tmp) / seam.FREEZE_RECORD).parent.mkdir(parents=True)
            (Path(tmp) / seam.FREEZE_RECORD).write_bytes(OVERLAY.read_bytes().replace(b'1337', b'1338'))
            with self.assertRaisesRegex(PreparationError, 'pinned'):
                seam.load_freeze_record()

    def test_02_bound_identities_config_a3_a6_a7_a8_m6d6g(self):
        self.assertEqual(self.record['bound_authority_sha256'], self.pf.BINDINGS)
        for rel, digest in self.record['bound_authority_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)
        keys = ' '.join(self.record['bound_authority_sha256'])
        for token in ('e07c_difffas_bin_idfree.yaml', 'Amendment_A3_', 'Amendment_A6_', 'e07c_a6_', 'Amendment_A7_',
                      'e07c_a7_', 'Amendment_A8_', 'e07c_a8_', 'M6D6G_E07C_AUX_FINAL_CHECKPOINT.json'):
            self.assertIn(token, keys)
        self.assertEqual(self.record['source'], {'commit': self.pf.PIN, 'tree': self.pf.TREE})
        self.assertEqual(self.record['producer']['producer_commit'], AUTHORITY)
        self.pf.freeze_record()

    def test_03_frozen_sha(self):
        self.assertEqual(self.record['frozen_asset']['sha256'], FROZEN_SHA)
        self.assertEqual(seam.OWNER_FROZEN_SHA256, FROZEN_SHA)
        final = json.loads(at_authority('outputs/audit/M6D6G_E07C_AUX_FINAL_CHECKPOINT.json'))
        self.assertEqual(final['sha256'], FROZEN_SHA)

    def test_04_frozen_size(self):
        self.assertEqual((self.record['frozen_asset']['bytes'], seam.OWNER_FROZEN_BYTES), (FROZEN_BYTES, FROZEN_BYTES))
        final = json.loads(at_authority('outputs/audit/M6D6G_E07C_AUX_FINAL_CHECKPOINT.json'))
        self.assertEqual(final['file_size_bytes'], FROZEN_BYTES)

    def test_05_frozen_main_seeds(self):
        self.assertEqual(self.record['frozen_for_main_seeds'], MAIN_SEEDS)
        self.assertEqual(self.record['owner_freeze']['frozen_for_main_seeds'], MAIN_SEEDS)
        self.assertEqual(self.cfg['seeds']['experiment_seeds'], MAIN_SEEDS)

    # 6-12 fail-closed gates ----------------------------------------------------------------------
    def test_06_other_valid_sha_rejected_before_checkpoint_read(self):
        other = FROZEN_SHA[:-1] + ('0' if FROZEN_SHA[-1] != '0' else '1')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': sentinel_torch()}), \
                mock.patch.object(seam, 'frozen_checkpoint_path', forbidden('frozen_checkpoint_path')), \
                mock.patch.object(seam, 'verify_future_checkpoint', forbidden('verify_future_checkpoint')), \
                mock.patch.object(seam, 'read_verified', forbidden('read_verified')), \
                mock.patch.object(seam, 'load_verified_whole_module', forbidden('load_verified_whole_module')):
            small_asset(tmp)
            for bad in (other, hashlib.sha256(b'x').hexdigest(), '0' * 64):
                with self.assertRaisesRegex(PreparationError, 'owner-frozen'):
                    with seam.load_frozen_aux_encoder(tmp, bad):
                        self.fail('must not yield')

    def test_07_malformed_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': sentinel_torch()}), \
                mock.patch.object(seam, 'verify_future_checkpoint', forbidden('verify_future_checkpoint')), \
                mock.patch.object(seam, 'read_verified', forbidden('read_verified')):
            small_asset(tmp)
            for bad in (FROZEN_SHA.upper(), FROZEN_SHA[:-1], 'not-a-sha', None, 42, FROZEN_SHA + '0'):
                with self.assertRaises(PreparationError):
                    with seam.load_frozen_aux_encoder(tmp, bad):
                        self.fail('must not yield')

    def test_08_relative_runtime_root_rejected(self):
        with mock.patch.dict(sys.modules, {'torch': sentinel_torch()}):
            with self.assertRaisesRegex(PreparationError, 'external runtime storage'):
                seam.frozen_checkpoint_path('relative/runtime')
            with self.assertRaisesRegex(PreparationError, 'external runtime storage'):
                with seam.load_frozen_aux_encoder('relative/runtime', FROZEN_SHA):
                    self.fail('must not yield')

    def test_09_runtime_root_inside_repository_rejected(self):
        with mock.patch.dict(sys.modules, {'torch': sentinel_torch()}):
            for inside in (ROOT, ROOT / 'outputs', ROOT / 'runs'):
                with self.assertRaisesRegex(PreparationError, 'external runtime storage'):
                    with seam.load_frozen_aux_encoder(str(inside), FROZEN_SHA):
                        self.fail('must not yield')

    def test_10_canonical_path_is_configured_seed_42_aux_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = seam.frozen_checkpoint_path(tmp)
            self.assertEqual(path, Path(tmp) / RELATIVE)
            template = self.cfg['external_assets'][0]['external_runtime_path']
            self.assertEqual(template, '<runtime_root>/' + RELATIVE)
            self.assertEqual(self.record['frozen_asset']['path_template'], template)
            self.assertEqual(str(path), template.replace('<runtime_root>', tmp))
        self.assertTrue(self.record['frozen_asset']['observed_gpu_path'].endswith('/' + RELATIVE))
        self.assertTrue(self.record['frozen_asset']['observed_gpu_path_role'].startswith('PROVENANCE_ONLY'))

    def test_11_size_mismatch_or_missing_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': sentinel_torch()}), \
                mock.patch.object(seam, 'verify_future_checkpoint', forbidden('verify_future_checkpoint')), \
                mock.patch.object(seam, 'read_verified', forbidden('read_verified')):
            with self.assertRaisesRegex(PreparationError, 'unavailable'):
                with seam.load_frozen_aux_encoder(tmp, FROZEN_SHA):
                    self.fail('must not yield')
            small_asset(tmp)
            with self.assertRaisesRegex(PreparationError, 'frozen size'):
                with seam.load_frozen_aux_encoder(tmp, FROZEN_SHA):
                    self.fail('must not yield')

    def test_12_wrong_bytes_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'torch': sentinel_torch()}):
            path = small_asset(tmp)
            size = path.stat().st_size
            # size simulated to match the freeze; bytes are not the frozen bytes -> external-asset SHA256 gate
            with mock.patch.object(seam, 'load_freeze_record', fake_freeze(FROZEN_SHA, size)), \
                    mock.patch.object(seam, 'read_verified', forbidden('read_verified')):
                with self.assertRaisesRegex(PreparationError, 'SHA256 mismatch'):
                    with seam.load_frozen_aux_encoder(tmp, FROZEN_SHA):
                        self.fail('must not yield')
            # second, independent layer: exact in-memory bytes are re-hashed before any torch.load
            passthrough = lambda root, sha, config=None: {'path': str(path), 'size_bytes': size, 'sha256': sha}
            with mock.patch.object(seam, 'load_freeze_record', fake_freeze(FROZEN_SHA, size)), \
                    mock.patch.object(seam, 'verify_future_checkpoint', passthrough):
                with self.assertRaisesRegex(PreparationError, 'refusing to deserialize'):
                    with seam.load_frozen_aux_encoder(tmp, FROZEN_SHA):
                        self.fail('must not yield')

    # 13-17 exact-byte ordering and the one unsafe-load seam ----------------------------------------
    def test_13_read_verified_before_torch_import_and_load(self):
        events = []

        class Stop(Exception):
            pass
        fake = types.ModuleType('torch')

        def load(buf, **kwargs):
            events.append(('torch.load', type(buf).__name__, buf.getvalue(), kwargs))
            raise Stop
        fake.load = load
        real_read = seam.read_verified

        def read_verified(path, expected):
            events.append(('read_verified', Path(path).name))
            return real_read(path, expected)

        @contextmanager
        def modules(*args, **kwargs):
            events.append(('custom_rn_import',))
            yield {'custom_rn': types.ModuleType('custom_rn')}
        with tempfile.TemporaryDirectory() as tmp:
            path = small_asset(tmp)
            raw = path.read_bytes()
            with mock.patch.dict(sys.modules, {'torch': fake}), \
                    mock.patch.object(seam, 'load_freeze_record', fake_freeze(hashlib.sha256(raw).hexdigest(), len(raw))), \
                    mock.patch.object(seam, 'read_verified', read_verified), \
                    mock.patch.object(seam, 'validate_source', lambda cfg: events.append(('validate_source',)) or {'root': tmp}), \
                    mock.patch.object(seam, 'verify_executable_semantics', lambda *a: events.append(('semantics',))), \
                    mock.patch.object(seam, 'upstream_modules', modules):
                with self.assertRaises(Stop):
                    with seam.load_frozen_aux_encoder(tmp, hashlib.sha256(raw).hexdigest()):
                        self.fail('must not yield')
        self.assertEqual([e[0] for e in events], ['read_verified', 'validate_source', 'semantics', 'custom_rn_import',
                                                  'torch.load'])
        self.assertEqual(events[-1], ('torch.load', 'BytesIO', raw, {'weights_only': False}))
        tree = ast.parse(inspect.getsource(seam))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_verified_whole_module')
        order = [ast.unparse(s) for s in fn.body]
        self.assertLess(order.index('raw = read_verified(path, expected_sha256)'), order.index('import torch'))

    def test_14_exactly_one_whole_module_unsafe_load(self):
        tree = ast.parse(inspect.getsource(seam))
        loads = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
        self.assertEqual(loads, ["torch.load(io.BytesIO(raw), weights_only=LOAD_COMPATIBILITY['weights_only'])"])
        self.assertIs(seam.LOAD_COMPATIBILITY['weights_only'], False)
        self.assertEqual(self.pf.seam()['torch_load'], loads)

    def test_15_no_state_dict_or_format_conversion(self):
        tree = ast.parse(inspect.getsource(seam))
        names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
                 if isinstance(n, (ast.Attribute, ast.Name))}
        self.assertFalse({'state_dict', 'load_state_dict', 'safetensors', 'jit', 'onnx', 'add_safe_globals',
                          'safe_globals', 'environ', 'getenv', 'argv', 'glob', 'rglob', 'iterdir', 'listdir'} & names)
        self.assertEqual(seam.FROZEN_FORMAT, self.cfg['conditioning_encoder']['checkpoint_format'])
        self.assertEqual(self.record['frozen_asset']['format'], seam.FROZEN_FORMAT)

    def test_16_no_tracked_weight_file(self):
        out = git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors')
        self.assertEqual((out.returncode, out.stdout.decode().strip()), (0, ''))
        self.assertFalse(self.record['frozen_asset']['weight_bytes_in_git'])

    def test_17_load_frozen_aux_encoder_is_the_authorized_seam(self):
        self.assertEqual(str(inspect.signature(inspect.unwrap(seam.load_frozen_aux_encoder))),
                         '(runtime_root, recorded_sha256, config=None)')
        self.assertEqual(self.record['consumption']['secure_loader'],
                         'methods.difffas.aux_checkpoint.load_frozen_aux_encoder')
        self.assertEqual(self.record['consumption']['currently_qualified_consumer_scope'], 'MAIN_DIFFFAS')
        from methods.difffas import execution_policy
        self.assertIs(execution_policy.load_frozen_aux_encoder, seam.load_frozen_aux_encoder)
        self.assertEqual(at_m6d6h('methods/difffas/execution_policy.py'),
                         at_authority('methods/difffas/execution_policy.py'))

    # 18-24 scope and zero-science ------------------------------------------------------------------
    def test_18_main_seeds_share_one_sha(self):
        o = self.record['owner_freeze']
        self.assertTrue(o['all_main_seeds_share_one_checkpoint'])
        self.assertEqual(o['alternative_checkpoints_authorized'], 0)
        self.assertEqual(self.cfg['conditioning_encoder']['same_frozen_encoder_reused_for_main_seeds'], MAIN_SEEDS)
        self.assertTrue(self.cfg['external_assets'][0]['fixed_across_experiment_seeds'])
        self.assertIsInstance(self.record['frozen_asset']['sha256'], str)   # one value, not a per-seed map

    def test_19_train_val_test_not_accessed(self):
        f = self.record['freeze_operation']
        self.assertEqual((f['TRAIN_access'], f['VAL_access'], f['TEST_access']), (False, False, False))
        text = inspect.getsource(seam)
        for token in ('manifests/', '.parquet', 'faces_256', 'ImageFolder', 'DataLoader(', 'runs/m6'):
            self.assertNotIn(token, text)

    def test_20_zero_optimizer_and_backward(self):
        f = self.record['freeze_operation']
        self.assertEqual((f['optimizer_steps'], f['backward_calls'], f['checkpoint_deserializations']), (0, 0, 0))

    def test_21_zero_scientific_runs(self):
        self.assertEqual(self.record['freeze_operation']['training_runs'], 0)

    def test_22_zero_main_difffas_runs(self):
        self.assertEqual(self.record['freeze_operation']['main_difffas_scientific_runs'], 0)

    def test_23_no_m8_bank(self):
        self.assertEqual(self.record['freeze_operation']['M8_outputs'], 0)
        self.assertEqual(self.record['consumption']['M8_BANK'], 'UNQUALIFIED')

    def test_24_m8_integration_unqualified(self):
        c = self.record['consumption']
        self.assertEqual((c['M8_integration_qualified'], c['A7_M8_rng_policy_question']), (False, 'UNRESOLVED'))

    # 25-27 history ------------------------------------------------------------------------------------
    def test_25_m6d6g_evidence_byte_identical(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').stdout.decode().splitlines()
                 if Path(p).name.startswith(('M6D6C_', 'M6D6D_', 'M6D6E_', 'M6D6F_', 'M6D6G_'))]
        self.assertEqual(len([p for p in paths if 'M6D6G_' in p]), 4)
        for rel in paths:
            self.assertEqual(at_m6d6h(rel), at_authority(rel), rel)
        final = json.loads(at_m6d6h('outputs/audit/M6D6G_E07C_AUX_FINAL_CHECKPOINT.json'))
        self.assertEqual((final['final_checkpoint_status'], final['authoritative_for_main_difffas'],
                          final['owner_freeze_performed']), ('SHA256_RECORDED_PENDING_OWNER_FREEZE', False, False))
        ledger = at_m6d6h('outputs/audit/EXECUTION_LEDGER.jsonl')
        self.assertTrue(ledger.startswith(at_authority('outputs/audit/EXECUTION_LEDGER.jsonl')))

    def test_26_b1_verifies_historical_range_not_current_head(self):
        new, old = at_m6d6h(B1_TEST), at_authority(B1_TEST)
        self.assertEqual((hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()),
                         (self.pf.B1_OLD_SHA, self.pf.B1_NEW_SHA))
        fns = lambda raw: {n.name: ast.unparse(n) for c in ast.parse(raw.decode()).body if isinstance(c, ast.ClassDef)
                           for n in c.body if isinstance(n, ast.FunctionDef)}
        f_old, f_new = fns(old), fns(new)
        self.assertEqual(sorted(k for k in f_old if f_old[k] != f_new[k]),
                         ['test_17_scientific_files_byte_identical_to_authority'])
        t17 = f_new['test_17_scientific_files_byte_identical_to_authority']
        self.assertIn("git('diff', '--name-only', AUTHORITY, M6D6G_COMMIT", t17)
        self.assertNotIn('scientific_files_unchanged', t17)
        self.assertIn(f"M6D6G_COMMIT = '{AUTHORITY}'", new.decode())
        self.assertEqual(git('rev-parse', AUTHORITY + '^').stdout.decode().strip(), M6D6F_COMMIT)
        self.assertEqual(at_m6d6h('tools/m6d6g_e07c_aux_scientific_preflight.py'),
                         at_authority('tools/m6d6g_e07c_aux_scientific_preflight.py'))

    def test_27_unrelated_scientific_code_and_frozen_inputs_unchanged(self):
        for rel in self.pf.PRESERVED:
            self.assertEqual(at_m6d6h(rel), at_authority(rel), rel)
        amendments = git('ls-tree', '-r', '--name-only', AUTHORITY, 'docs/spec/amendments', 'configs/amendments')
        for rel in amendments.stdout.decode().splitlines():
            self.assertEqual(at_m6d6h(rel), at_authority(rel), rel)
        now = {n.name: ast.unparse(n) for n in ast.parse(at_m6d6h(self.pf.SEAM).decode()).body
               if isinstance(n, ast.FunctionDef)}
        old = {n.name: ast.unparse(n) for n in ast.parse(at_authority(self.pf.SEAM).decode()).body
               if isinstance(n, ast.FunctionDef)}
        for name in self.pf.UNCHANGED_SEAM_FUNCTIONS:
            self.assertEqual(now[name], old[name], name)
        self.assertEqual(set(now) - set(old), {'load_freeze_record', 'frozen_checkpoint_path'})

    def test_28_preflight_is_static(self):
        tree = ast.parse((ROOT / 'tools/m6d6h_e07c_aux_encoder_freeze_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        with self.assertRaises(ValueError):
            self.pf.read('outputs/x/encoder_final.pkl')
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6D6h evidence not yet recorded')
class TestM6D6hEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = load_preflight()
        cls.ev = json.loads(EVIDENCE.read_bytes())

    def test_evidence_validates(self):
        self.pf.check_evidence()

    def test_historical_vs_new_status_distinguished(self):
        h, n = self.ev['historical_m6d6g'], self.ev['m6d6h_owner_freeze']
        self.assertEqual((h['final_checkpoint_status'], h['authoritative_for_main_difffas']),
                         ('SHA256_RECORDED_PENDING_OWNER_FREEZE', False))
        self.assertEqual((n['owner_freeze_performed'], n['authoritative_for_main_difffas'], n['sha256'], n['bytes']),
                         (True, True, h['sha256'], FROZEN_BYTES))

    def test_zero_science_and_firewall(self):
        self.assertEqual(self.ev['counters'], self.pf.ZERO)
        self.assertFalse(self.ev['real_checkpoint_deserialized'] or self.ev['training'] or self.ev['TRAIN_access'] or
                         self.ev['VAL_access'] or self.ev['TEST_access'] or self.ev['M8_bank'] or
                         self.ev['M8_integration_qualified'] or self.ev['chmod_performed'])

    def test_gpu_revalidation(self):
        g = self.ev['gpu_revalidation']
        self.assertEqual((g['sha256'], g['size_bytes'], g['checkpoint_deserialized'], g['read_only']),
                         (FROZEN_SHA, FROZEN_BYTES, False, True))

    def test_statuses(self):
        self.assertEqual(self.ev['qualified_statuses'], self.pf.QUALIFIED)
        self.assertEqual(self.ev['not_qualified'], self.pf.NOT_QUALIFIED)
        self.assertEqual(self.ev['method_status'], 'IMPLEMENTED_NOT_EXECUTED')


if __name__ == '__main__':
    unittest.main()
