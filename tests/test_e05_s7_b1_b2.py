"""E05 owner ruling B1/B2 (configs/amendments/e05_s7_b1_b2_ruling.yaml): tensor2im verified at its defining module, and
`s7 --seed S` process isolation (P2_ONE_PROCESS_PER_SEED). Needs numpy (learned_bank), not torch: the tensor2im
function here is compiled from the real pinned source file, so its code object, module and file are genuine; the real
CUDA runtime is qualified on the GPU host (E05_S7_B1_B2_QUALIFICATION)."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.common import learned_bank as lb  # noqa: E402
from methods.pcgan import bank_adapter as ba  # noqa: E402

RULING = json.loads((ROOT / 'configs/amendments/e05_s7_b1_b2_ruling.yaml').read_text())
PINNED_SRC = ROOT / 'third_party/source_cache/swapping_autoencoder/util/util.py'
BASE = 'ef96fcbafb609c95ba930ec0754931c6e35be257'          # last commit before B1/B2


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_show(commit, rel):
    return subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (commit, rel)], capture_output=True).stdout


def function_sources(text):
    tree = ast.parse(text)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(text, node)
    return out


def pinned_segment_sha():
    text = PINNED_SRC.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2im')
    return sha(ast.get_source_segment(text, node).encode())


@unittest.skipUnless(PINNED_SRC.is_file(), 'pinned PCGAN source cache not present')
class B1RuntimeTensor2im(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.gen = self.tmp / 'source'
        (self.gen / 'util').mkdir(parents=True)
        (self.gen / 'util' / '__init__.py').write_text('from .util import *\n')
        self.util_py = self.gen / 'util' / 'util.py'
        shutil.copy2(PINNED_SRC, self.util_py)
        self.closure = {'util/util.py': sha(self.util_py.read_bytes()),
                        'util/__init__.py': sha((self.gen / 'util/__init__.py').read_bytes())}
        self.pinned = pinned_segment_sha()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def modules(self, path=None, module_name='util.util', share=True):
        path = path or self.util_py
        text = path.read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2im')
        import numpy as np
        ns = {'__name__': module_name, 'np': np, 'torch': None, 'tile_images': None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        fn = ns['tensor2im']
        mod = types.ModuleType('util.util')
        mod.__file__ = str(self.util_py)
        mod.tensor2im = fn
        pkg = types.ModuleType('util')
        pkg.__file__ = str(self.gen / 'util' / '__init__.py')
        pkg.__path__ = [str(self.gen / 'util')]
        if share:
            pkg.tensor2im = fn
        else:                                      # a second, separately compiled copy (same module name and file)
            ns2 = dict(ns)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns2)
            pkg.tensor2im = ns2['tensor2im']
        return {'util': pkg, 'util.util': mod}

    def check(self, modules, closure=None, pinned=None, gen=None):
        return ba.runtime_tensor2im(modules, gen or self.gen, closure or self.closure, pinned or self.pinned)

    def test_01_pinned_segment_is_the_ruling_value(self):
        self.assertEqual(self.pinned, RULING['tensor2im']['pinned_segment_sha256'])
        manifest = json.loads((ROOT / 'environments/e05.compatibility_patch_manifest.json').read_text())
        self.assertEqual(manifest['generated_files_sha256']['util/util.py'], RULING['tensor2im']['qualified_file_sha256'])

    def test_02_defining_module_accepted(self):
        mods = self.modules()
        fn, rec = self.check(mods)
        self.assertIs(fn, mods['util'].tensor2im)
        self.assertEqual(rec, {'module': 'util.util', 'file': 'util/util.py', 'file_sha256': self.closure['util/util.py'],
                               'segment_sha256': self.pinned})

    def test_03_package_init_or_other_module_rejected(self):
        with self.assertRaisesRegex(lb.LearnedBankError, "defined in 'util', not util.util"):
            self.check(self.modules(module_name='util'))
        with self.assertRaisesRegex(lb.LearnedBankError, 'is not util.util.tensor2im'):
            self.check(self.modules(share=False))

    def test_04_wrong_file_hash_or_location_rejected(self):
        with self.assertRaisesRegex(lb.LearnedBankError, 'not the qualified closure file'):
            self.check(self.modules(), closure=dict(self.closure, **{'util/util.py': '0' * 64}))
        other = self.tmp / 'elsewhere'
        (other / 'util').mkdir(parents=True)
        with self.assertRaisesRegex(lb.LearnedBankError, 'not the generated-closure'):
            self.check(self.modules(), gen=other)

    def test_05_code_from_another_file_rejected(self):
        shadow = self.tmp / 'shadow_util.py'
        shutil.copy2(PINNED_SRC, shadow)
        with self.assertRaisesRegex(lb.LearnedBankError, 'code does not come from'):
            self.check(self.modules(path=shadow))

    def test_06_changed_implementation_rejected(self):
        text = self.util_py.read_text().replace('255.0', '254.0', 1)
        self.assertNotEqual(text, self.util_py.read_text())
        self.util_py.write_text(text)
        closure = dict(self.closure, **{'util/util.py': sha(self.util_py.read_bytes())})
        with self.assertRaisesRegex(lb.LearnedBankError, 'differs from pinned'):
            self.check(self.modules(), closure=closure)


class B1NarrowChange(unittest.TestCase):
    def test_01_only_the_check_changed(self):
        old = git_show(BASE, 'methods/pcgan/bank_adapter.py').decode()
        new = (ROOT / 'methods/pcgan/bank_adapter.py').read_text()
        self.assertIn("live_segment = ast.get_source_segment(Path(util.__file__).read_text(), next(", old)
        self.assertNotIn('Path(util.__file__)', new)
        self.assertIn("runtime_tensor2im(sys.modules, generated_root, runtime['build_closure']['generated_files_sha256'],",
                      new)
        fo, fn = function_sources(old), function_sources(new)
        for name in ('PCGANBankAdapter', 'mix', 'generate', 'finalize', '_tensor', 'canonical_input', 'decode_face',
                     'pinned_tensor2im', 'configure_precision', 'load_checkpoint_state', 'checkpoint_structure',
                     'build_fixture'):
            self.assertEqual(fo[name], fn[name], name)
        self.assertIn('util.tensor2im, torch.device(device)', fn['load_real'])
        self.assertEqual(set(fn) - set(fo), {'runtime_tensor2im'})

    def test_02_scientific_modules_unchanged(self):
        for rel in ('methods/common/learned_bank.py', 'methods/pcgan/production_io.py',
                    'methods/pcgan/production_runner.py', 'methods/pcgan/training_qualification.py',
                    'methods/pcgan/source_relocation.py', 'methods/pcgan/runtime_qualification.py', 'tools/run_e05.py',
                    'environments/e05.compatibility_patch_manifest.json', 'environments/e05.lock.json',
                    'configs/methods/e05_pcgan.yaml', 'configs/amendments/learned_bank_generation_contract.yaml',
                    'configs/amendments/learned_bank_s7_launch_authority.yaml',
                    'configs/amendments/e05_source_relocation_ruling.yaml', lb.TOY_MANIFEST_PATH):
            self.assertEqual(git_show(BASE, rel), (ROOT / rel).read_bytes(), rel)

    def test_03_rng_and_conversion_unchanged(self):
        self.assertEqual(lb.CONVERSION_RULES['E05'],
                         'util.tensor2im(mix[0]): clip((x+1)/2*255, 0, 255).astype(uint8) [swapping_autoencoder '
                         'util/util.py:125-131]')
        s = hashlib.sha256(b'GPAT_TRANSFERBENCH|E05|BANK|1337|PTR000028').digest()
        self.assertEqual(lb.e05_pair_seed(1337, 'PTR000028'), int.from_bytes(s[:8], 'big') % (2 ** 31 - 1))


def cli_module():
    spec = importlib.util.spec_from_file_location('learned_bank_cli', ROOT / 'tools' / 'learned_bank.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class B2SeedProcess(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()).resolve()
        self.cli = cli_module()
        self.calls = []

        def fake_generate(auth, method, seed, rt, run_dir, kind):
            self.calls.append((auth.mode, method, seed, str(run_dir), kind))
            return {'status': 'completed', 'rows': 186, 'outputs': 186, 'failure_count': 0, 'manifest_sha256': 'x',
                    'checkpoint_unchanged_after': True, 'peak_cuda_memory_allocated_bytes': 1}
        self.cli.generate_real = fake_generate

    def tearDown(self):
        shutil.rmtree(self.out)

    def run_cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.cli.main(argv)
        return rc, json.loads(buf.getvalue().strip().splitlines()[-1])

    def test_01_exactly_one_seed_per_process(self):
        root = self.out / 'seed_1337'
        root.mkdir()
        rc, rec = self.run_cli(['s7', '--method', 'E05', '--label', 'A', '--out-root', str(root),
                                '--runtime-root', '/nonexistent-runtime', '--seed', '1337'])
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [('S7_TOY', 'E05', 1337, str(root / 'run_A/E05/seed_1337'), 'S7_TOY_POPULATION')])
        self.assertEqual((rec['seed'], rec['process_seeds'], sorted(rec['results'])), (1337, [1337], ['1337']))

    def test_02_without_seed_unchanged(self):
        rc, rec = self.run_cli(['s7', '--method', 'E05', '--label', 'B', '--out-root', str(self.out),
                                '--runtime-root', '/nonexistent-runtime'])
        self.assertEqual([c[2] for c in self.calls], [42, 1337, 2026])
        self.assertNotIn('seed', rec)
        self.assertNotIn('process_seeds', rec)

    def test_03_seed_refused_for_other_methods_and_unapproved_seeds(self):
        for argv in (['s7', '--method', 'E06c', '--label', 'A', '--out-root', str(self.out), '--runtime-root', '/x',
                      '--seed', '42'],
                     ['s7-compare', '--method', 'E03', '--out-root', str(self.out), '--seed', '42'],
                     ['s7', '--method', 'E05', '--label', 'A', '--out-root', str(self.out), '--runtime-root', '/x',
                      '--seed', '7']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                self.cli.main(argv)
            self.assertEqual(cm.exception.code, 2)
        self.assertEqual((self.calls, list(self.out.iterdir())), ([], []))

    def test_04_existing_run_root_refused(self):
        (self.out / 'run_A').mkdir()
        with self.assertRaises(FileExistsError):
            self.cli.main(['s7', '--method', 'E05', '--label', 'A', '--out-root', str(self.out),
                           '--runtime-root', '/nonexistent-runtime', '--seed', '42'])
        self.assertEqual(self.calls, [])

    def test_05_other_seeds_checkpoint_refused(self):
        with self.assertRaisesRegex(lb.LearnedBankError, 'is not runs/m6/E05/seed_1337'):
            lb.verify_checkpoint('E05', 1337, Path('/w/runs/m6/E05/seed_42'))
        adapter = types.SimpleNamespace(method_id='E05', bank_seed=1337, generator_seed=1337, is_fixture=False,
                                        contract_sha256='c', checkpoint_identity_sha256=lb.checkpoint_identity_sha256('E05', 42))
        pop = types.SimpleNamespace(method_id='E05', seed=1337, kind='S7_TOY_POPULATION')
        auth = lb.Authorization(mode='S7_TOY', methods=('E05',), seeds=lb.SEEDS, contract_sha256='c', label='s7_toy')
        with self.assertRaisesRegex(lb.LearnedBankError, 'not the approved checkpoint of E05 seed 1337'):
            lb._check_mode(adapter, pop, auth, self.out / 'x')


class HistoryPreserved(unittest.TestCase):
    def test_01_records_unchanged(self):
        for commit, rel in (('2d430e9', 'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.json'),
                            ('ef96fcb', 'outputs/audit/E05_SOURCE_RELOCATION_QUALIFICATION.json'),
                            ('8fb5bac', 'outputs/audit/LEARNED_BANK_S7_E06c.json'),
                            ('cd9c294', 'outputs/audit/LEARNED_BANK_S7_E03.json'),
                            ('f84f05f', 'outputs/audit/LEARNED_BANK_S7_E06b.json'),
                            ('eeff32d', 'outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json'),
                            ('0e3f1d1', 'outputs/audit/E04_SEED2026_COMPLETION.json'),
                            ('7677c3a', 'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.json'),
                            ('ed05c0b', 'outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.json'),
                            ('35ad1fd', 'outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.json')):
            self.assertEqual(git_show(commit, rel), (ROOT / rel).read_bytes(), rel)

    def test_02_ruling(self):
        self.assertEqual(RULING['decisions'], {'B1': 'TENSOR2IM_CHECK_AT_DEFINING_MODULE', 'B2': 'P2_ONE_PROCESS_PER_SEED'})
        self.assertEqual(RULING['resolves']['qualification_record_sha256'],
                         sha((ROOT / 'outputs/audit/E05_SOURCE_RELOCATION_QUALIFICATION.json').read_bytes()))
        a = RULING['authorizes']
        self.assertFalse(any(a[k] for k in ('production_bank_generation', 'generator_retraining', 'downstream_training',
                                            'test_evaluation', 'dinov3', 'gpat')))


if __name__ == '__main__':
    unittest.main()
