"""E05 owner ruling O2: path-independent identity of the pinned PCGAN upstream source (methods/pcgan/source_relocation.py)
and the corrected existing_runtime gate. Stdlib + git; uses the local pinned source cache (copied into a temporary
checkout). The CUDA load itself is qualified on the GPU host (E05_SOURCE_RELOCATION_QUALIFICATION)."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'third_party/source_cache/swapping_autoencoder'
MANIFEST = json.loads((ROOT / 'environments/e05.compatibility_patch_manifest.json').read_text())
RULING = 'configs/amendments/e05_source_relocation_ruling.yaml'


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = load('methods/pcgan/source_relocation.py', 'e05_source_relocation')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_show(commit, rel):
    return subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (commit, rel)], capture_output=True).stdout


@unittest.skipUnless((SRC / '.git').exists(), 'pinned PCGAN source cache not present')
class RelocatedSourceIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp()).resolve()
        cls.base = cls.tmp / 'checkout'
        shutil.copytree(SRC, cls.base / sr.SOURCE_REL, symlinks=True)
        (cls.base / 'configs/amendments').mkdir(parents=True)
        shutil.copy2(ROOT / RULING, cls.base / RULING)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def fresh(self, name):
        """A private checkout copy that a test may damage."""
        d = self.tmp / name
        shutil.copytree(self.base, d, symlinks=True)
        return d

    def live(self, checkout):
        return dict(MANIFEST['source'], root=str(checkout / sr.SOURCE_REL))

    def check(self, checkout, live=None, manifest=None):
        return sr.verify_source_identity(manifest or MANIFEST['source'], live or self.live(checkout), checkout)

    def test_01_relocated_identical_source_accepted(self):
        out = self.check(self.base)
        self.assertTrue(out['relocated'])
        self.assertEqual(out['differing_fields'], ['root'])
        self.assertEqual((out['files_verified'], out['code_files_verified']), (23, 67))
        self.assertEqual(out['ruling'], {'path': RULING, 'sha256': sha((ROOT / RULING).read_bytes())})

    def test_02_original_path_accepted_without_relocation(self):
        orig = sr.HISTORICAL_ROOT
        try:
            sr.HISTORICAL_ROOT = str(self.base / sr.SOURCE_REL)
            manifest = dict(MANIFEST['source'], root=sr.HISTORICAL_ROOT)
            out = self.check(self.base, live=dict(manifest), manifest=manifest)
        finally:
            sr.HISTORICAL_ROOT = orig
        self.assertFalse(out['relocated'])
        self.assertNotIn('ruling', out)

    def test_03_relocation_requires_owner_ruling(self):
        d = self.fresh('no_ruling')
        (d / RULING).unlink()
        with self.assertRaisesRegex(sr.SourceIdentityError, 'owner ruling'):
            self.check(d)
        r = json.loads((ROOT / RULING).read_text())
        r['relocatable_fields'] = ['root', 'commit']
        (d / RULING).write_text(json.dumps(r))
        with self.assertRaisesRegex(sr.SourceIdentityError, 'not the approved O2'):
            self.check(d)

    def test_04_changed_bytes_rejected(self):
        d = self.fresh('changed')
        p = d / sr.SOURCE_REL / 'models/networks/generator.py'
        p.write_bytes(p.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'bytes differ'):
            self.check(d)

    def test_05_missing_file_rejected(self):
        d = self.fresh('missing')
        (d / sr.SOURCE_REL / 'models/networks/encoder.py').unlink()
        with self.assertRaisesRegex(sr.SourceIdentityError, 'missing/unsafe'):
            self.check(d)

    def test_06_unexpected_executable_rejected(self):
        d = self.fresh('extra')
        (d / sr.SOURCE_REL / 'models/networks/extra_layer.py').write_text('x = 1\n')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'unexpected executable'):
            self.check(d)

    def test_07_symlink_and_traversal_rejected(self):
        d = self.fresh('symlink_in_tree')
        os.symlink('/etc/hostname', d / sr.SOURCE_REL / 'data/escape.txt')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'symlink'):
            self.check(d)
        d = self.fresh('symlinked_source')
        real = self.tmp / 'elsewhere_src'
        shutil.move(str(d / sr.SOURCE_REL), str(real))
        os.symlink(real, d / sr.SOURCE_REL)
        with self.assertRaisesRegex(sr.SourceIdentityError, 'no symlink or traversal'):
            self.check(d)
        live = dict(MANIFEST['source'], root=str(self.base / 'third_party/../third_party/source_cache/swapping_autoencoder'))
        with self.assertRaisesRegex(sr.SourceIdentityError, 'no symlink or traversal'):
            self.check(self.base, live=live)
        other = dict(MANIFEST['source'], root=str(SRC))       # a valid source, but not inside this checkout
        with self.assertRaisesRegex(sr.SourceIdentityError, 'no symlink or traversal'):
            self.check(self.base, live=other)

    def test_08_wrong_commit_tree_or_inventory_rejected(self):
        for key, value in (('commit', '0' * 40), ('tree', '1' * 40), ('worktree_status', ' M x.py'),
                           ('repository', 'https://example.invalid/fork')):
            with self.subTest(key=key), self.assertRaisesRegex(sr.SourceIdentityError, 'content identity differs'):
                self.check(self.base, live=dict(self.live(self.base), **{key: value}))
        files = dict(MANIFEST['source']['files_sha256'])
        files.pop(sorted(files)[0])
        with self.assertRaisesRegex(sr.SourceIdentityError, 'content identity differs'):
            self.check(self.base, live=dict(self.live(self.base), files_sha256=files))
        with self.assertRaisesRegex(sr.SourceIdentityError, 'field set differs'):
            self.check(self.base, live=dict(self.live(self.base), extra_field=1))

    def test_09_wrong_historical_root_in_manifest_rejected(self):
        manifest = dict(MANIFEST['source'], root='/somewhere/else')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'historical root'):
            self.check(self.base, manifest=manifest)


class ExtensionAndImportProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.build = self.tmp / 'build'
        self.generated = self.build / 'source'
        (self.generated / 'models/networks/stylegan2_op').mkdir(parents=True)
        (self.build / 'ext').mkdir()
        self.src = self.generated / 'models/networks/stylegan2_op/kernel.cu'
        self.src.write_bytes(b'__global__ void k() {}\n')
        self.binary = self.build / 'ext' / 'fused.so'
        self.binary.write_bytes(b'\x7fELF qualified binary')
        (self.build / 'ext' / 'build.ninja').write_text('rule cxx\n')
        self.closure = {'models/networks/stylegan2_op/kernel.cu': sha(self.src.read_bytes())}
        self.expected = {'extension': {'path': str(self.binary), 'sha256': sha(self.binary.read_bytes())},
                         'build_ninja': 'rule cxx\n'}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def verify(self, path=None, expected=None, sources=None):
        return sr.verify_extension(path or self.binary, expected or self.expected, self.build, self.generated,
                                   self.closure, sources or [str(self.src)])

    def test_01_qualified_extension_accepted(self):
        self.assertEqual(self.verify()['sha256'], self.expected['extension']['sha256'])

    def test_02_wrong_binary_ninja_location_or_source_rejected(self):
        with self.assertRaisesRegex(sr.SourceIdentityError, 'binary hash'):
            self.verify(expected=dict(self.expected, extension=dict(self.expected['extension'], sha256='0' * 64)))
        with self.assertRaisesRegex(sr.SourceIdentityError, 'Ninja'):
            self.verify(expected=dict(self.expected, build_ninja='rule other\n'))
        outside = self.tmp / 'fused.so'
        outside.write_bytes(self.binary.read_bytes())
        with self.assertRaisesRegex(sr.SourceIdentityError, 'binary location'):
            self.verify(path=outside)
        link = self.build / 'ext' / 'link.so'
        os.symlink(outside, link)
        with self.assertRaisesRegex(sr.SourceIdentityError, 'binary location'):
            self.verify(path=link)
        self.src.write_bytes(b'__global__ void k() { /* unapproved patch */ }\n')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'extension source input'):
            self.verify()

    def test_03_import_resolution(self):
        mods, closure = {}, {}
        for name in sr.IMPORTED_MODULES:
            rel = name.replace('.', '/') + '.py'
            p = self.generated / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# %s\n' % name)
            closure[rel] = sha(p.read_bytes())
            mods[name] = types.SimpleNamespace(__file__=str(p))
        self.assertEqual(len(sr.verify_import_resolution(mods, self.generated, closure)), 7)
        shadow = self.tmp / 'util.py'
        shadow.write_text('# util\n')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'outside the generated closure'):
            sr.verify_import_resolution(dict(mods, util=types.SimpleNamespace(__file__=str(shadow))),
                                        self.generated, closure)
        p = Path(mods['util'].__file__)
        p.write_text('# patched util\n')
        with self.assertRaisesRegex(sr.SourceIdentityError, 'not the qualified closure file'):
            sr.verify_import_resolution(mods, self.generated, closure)


class RulingCorrectionAndHistory(unittest.TestCase):
    def test_01_ruling_binds_failure_and_unchanged_identities(self):
        r = json.loads((ROOT / RULING).read_text())
        self.assertEqual((r['status'], r['approved'], r['decision'], r['relocatable_fields']),
                         ('OWNER_APPROVED', True, 'O2_PATH_INDEPENDENT_SOURCE_IDENTITY', ['root']))
        self.assertEqual(r['historical_root'], sr.HISTORICAL_ROOT)
        self.assertEqual(r['historical_root'], MANIFEST['source']['root'])
        fail = 'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.json'
        self.assertEqual(r['resolves']['failure_record_sha256'], sha((ROOT / fail).read_bytes()))
        for key in ('compatibility_patch_manifest', 'environment_lock', 'frozen_config'):
            u = r['unchanged'][key]
            self.assertEqual(sha((ROOT / u['path']).read_bytes()), u['sha256'], key)
        a = r['authorizes']
        self.assertTrue(a['path_independent_source_identity'] and a['e05_s7_attempt_2'])
        self.assertFalse(any(a[k] for k in ('production_bank_generation', 'generator_retraining', 'downstream_training',
                                            'test_evaluation', 'dinov3', 'gpat')))

    def test_02_compatibility_manifest_and_production_code_unchanged(self):
        for rel in ('environments/e05.compatibility_patch_manifest.json', 'environments/e05.lock.json',
                    'methods/pcgan/production_io.py', 'methods/pcgan/production_runner.py', 'tools/run_e05.py',
                    'methods/pcgan/production_qualification.py', 'methods/pcgan/bank_adapter.py',
                    'methods/pcgan/runtime_qualification.py', 'configs/methods/e05_pcgan.yaml'):
            self.assertEqual(git_show('f291b34', rel), (ROOT / rel).read_bytes(), rel)

    def test_03_existing_runtime_correction_is_narrow(self):
        old = git_show('f291b34', 'methods/pcgan/training_qualification.py').decode()
        new = (ROOT / 'methods/pcgan/training_qualification.py').read_text()
        self.assertIn("require(manifest['source'] == source, 'source identity')", old)
        self.assertNotIn("require(manifest['source'] == source, 'source identity')", new)
        self.assertIn("relocation.verify_source_identity(manifest['source'], source, ROOT)", new)
        for kept in ("require(manifest_raw == (ROOT / 'environments/e05.compatibility_patch_manifest.json').read_bytes(),",
                     "require(closure == manifest['generated_files_sha256'], 'generated source closure')",
                     "require(name in binaries, 'only qualified extension')",
                     "require(util.is_custom_kernel_supported(), 'custom CUDA branch')",
                     "'both real CUDA extensions loaded, no fallback/rebuild')"):
            self.assertIn(kept, new)
        removed = [x for x in old.splitlines() if x not in new.splitlines()]
        self.assertEqual(len(removed), 8, removed)       # the equality line + the 7 inlined extension-check lines

    def test_04_historical_records_unchanged(self):
        for commit, rels in (('2d430e9', ['outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.json',
                                         'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.md',
                                         'outputs/audit/LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE_SHA256.txt']),
                             ('8fb5bac', ['outputs/audit/LEARNED_BANK_S7_E06c.json']),
                             ('cd9c294', ['outputs/audit/LEARNED_BANK_S7_E03.json']),
                             ('f84f05f', ['outputs/audit/LEARNED_BANK_S7_E06b.json']),
                             ('ed05c0b', ['outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.json'] +
                              ['outputs/audit/E01_PRODUCTION_BANK_SEED%d.json' % s for s in (42, 1337, 2026)]),
                             ('35ad1fd', ['outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.json'] +
                              ['outputs/audit/E02_PRODUCTION_BANK_SEED%d.json' % s for s in (42, 1337, 2026)])):
            for rel in rels:
                self.assertEqual(git_show(commit, rel), (ROOT / rel).read_bytes(), rel)


if __name__ == '__main__':
    unittest.main()
