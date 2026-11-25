"""M8 learned-bank PRODUCTION governance and audit records (framework-free; laptop .venv with numpy + pyarrow).

- the host launch scripts and the audit tool agree on the exact launch environments (E03 = the qualified
  methods.stdn.production_io.PRECISION_ENV; PyTorch methods = the S7 environment);
- every committed per-method authority is exactly what tools/learned_bank_production_authority.py derives from the
  committed records, and lb.load_production_authority accepts it (storage checked live only on the GPU host);
- every committed LEARNED_BANK_PRODUCTION_<M>_SEED<S>.json re-derives byte-identically from its committed evidence,
  its SHA list matches, and its gates fail closed on tampered evidence;
- every committed family closure re-derives byte-identically.
"""
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

AUDIT = ROOT / 'outputs' / 'audit'
SCRIPTS = ROOT / 'tools' / 'learned_bank_production_scripts'


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed_records():
    out = {}
    for m in ('E06c', 'E03', 'E06b', 'E05'):
        for s in lb.SEEDS:
            p = AUDIT / ('LEARNED_BANK_PRODUCTION_%s_SEED%d.json' % (m, s))
            if p.exists():
                out[(m, s)] = json.loads(p.read_text())
    return out


class ScriptsAndTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load('lbprod', 'tools/learned_bank_production_audit.py')

    def test_e03_environment_is_the_qualified_precision_env(self):
        tree = ast.parse((ROOT / 'methods/stdn/production_io.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'PRECISION_ENV')
        env = ast.literal_eval(node.value)
        want = 'PATH=/usr/bin:/bin LANG=C.UTF-8 ' + ' '.join('%s=%s' % kv for kv in sorted(env.items()))
        self.assertEqual(self.tool.E03_ENV, want)

    def test_scripts_use_the_audited_environments(self):
        text = (SCRIPTS / 'm8prod_launch.sh').read_text()
        self.assertEqual(text.count('ENVS="%s"' % self.tool.E03_ENV), 1)
        self.assertEqual(text.count('ENVS="%s"' % self.tool.PY_ENV), 2)
        for m in ('E03', 'E05', 'E06b', 'E06c'):
            self.assertIn(lb.ENVIRONMENTS[m]['python'], text)
        self.assertIn('tools/learned_bank.py production --method $M --seed $S --runtime-root $RT', text)
        self.assertIn('test ! -e $BANK', text)
        for f in ('m8prod_launch.sh', 'm8prod_wait.sh', 'm8prod_post.sh'):
            self.assertEqual(subprocess.run(['bash', '-n', str(SCRIPTS / f)]).returncode, 0, f)
        self.assertNotIn('kill', ''.join((SCRIPTS / f).read_text() for f in ('m8prod_wait.sh', 'm8prod_post.sh')))

    def test_run_id_is_deterministic(self):
        a = self.tool.run_id('E05', 42, 'c' * 40, 'a' * 64)
        self.assertEqual(a, self.tool.run_id('E05', 42, 'c' * 40, 'a' * 64))
        self.assertNotEqual(a, self.tool.run_id('E05', 1337, 'c' * 40, 'a' * 64))
        self.assertEqual(len(a), 16)

    def test_seed31_matches_contract_vectors(self):
        c = json.loads((ROOT / lb.CONTRACT_PATH).read_text())['byte_level_interpretation']['test_vectors']
        for k, v in c['E05'].items():
            s, pid = k.split('|')
            self.assertEqual(self.tool.seed31('GPAT_TRANSFERBENCH|E05|BANK|%s|%s' % (s, pid)), v)
        for m in ('E06b', 'E06c'):
            for s, v in c[m].items():
                self.assertEqual(self.tool.seed31('GPAT_TRANSFERBENCH|%s|BANK|%s' % (m, s)), v)


class Authorities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load('lbauth', 'tools/learned_bank_production_authority.py')

    def _root(self, M, auth):
        d = Path(tempfile.mkdtemp(prefix='lbprod_'))
        rels = [lb.CONTRACT_PATH, lb.DRAFT_PATH, lb.STORAGE_AUTHORITY_PATH, lb.E04_COMPLETION_RECORD,
                lb.E04_FAMILY_CLOSURE_RECORD, 'outputs/audit/LEARNED_BANK_S7_%s.json' % M]
        for rel in rels:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, d / rel)
        p = d / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=M)
        p.write_text(json.dumps(auth, indent=1) + '\n')
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def test_generated_authority_loads_for_every_method(self):
        for M in ('E06c', 'E03', 'E06b', 'E05'):
            with self.subTest(method=M):
                auth = self.gen.authority(M, '0' * 40, '2026-11-25')
                root = self._root(M, auth)
                for s in lb.SEEDS:
                    a = lb.load_production_authority(M, s, root, live_storage=False)
                    self.assertEqual((a.mode, a.methods, a.seeds), ('PRODUCTION', (M,), (s,)))
                    self.assertEqual(a.record()['records']['outputs/audit/LEARNED_BANK_S7_%s.json' % M],
                                     auth['s7_acceptance_record']['sha256'])
                self.assertEqual(auth['authorizes']['production_bank_generation'], True)
                self.assertFalse(any(v for k, v in auth['authorizes'].items()
                                     if k not in ('production_bank_generation', 'methods')))

    def test_tampered_authority_refused(self):
        auth = self.gen.authority('E06c', '0' * 40, '2026-11-25')
        for patch in (lambda a: a['s7_acceptance_record'].update(sha256='0' * 64),
                      lambda a: a.update(method='E03'), lambda a: a.update(status='DRAFT'),
                      lambda a: a.update(seeds=[42]), lambda a: a.update(contract_sha256='1' * 64)):
            bad = copy.deepcopy(auth)
            patch(bad)
            root = self._root('E06c', bad)
            with self.assertRaises(lb.LearnedBankBlocked):
                lb.load_production_authority('E06c', 1337, root, live_storage=False)

    def test_committed_authorities_are_derived(self):
        for M in ('E06c', 'E03', 'E06b', 'E05'):
            p = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=M)
            if not p.exists():
                continue
            with self.subTest(method=M):
                rec = json.loads(p.read_text())
                self.assertEqual(rec, self.gen.authority(M, rec['code_authority']['source_commit'], rec['date']))
                self.assertEqual(p.read_text(), json.dumps(rec, indent=1) + '\n')


class ProductionRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load('lbprod2', 'tools/learned_bank_production_audit.py')
        cls.docs = committed_records()

    def evidence(self, m, s):
        return json.loads((ROOT / self.tool.EVIDENCE.format(M=m, S=s) / 'collected.json').read_text())

    def test_rebuild_and_sha(self):
        for (m, s), d in self.docs.items():
            with self.subTest(method=m, seed=s):
                doc = self.tool.build(self.evidence(m, s), ROOT)
                self.assertEqual(json.loads(json.dumps(doc)), d)
                out = self.tool.RECORD.format(M=m, S=s)
                self.assertEqual(self.tool.report(doc), (ROOT / (out + '.md')).read_text())
                for line in (ROOT / (out + '_SHA256.txt')).read_text().splitlines():
                    digest, name = line.split('  ')
                    self.assertEqual(sha(ROOT / name), digest)
                body = gzip.decompress((ROOT / d['bank']['manifest_evidence']).read_bytes())
                self.assertEqual(hashlib.sha256(body).hexdigest(), d['bank']['manifest_sha256'])
                self.assertEqual(len(body.splitlines()), lb.N_SYN)

    def test_passed_shape(self):
        for (m, s), d in self.docs.items():
            with self.subTest(method=m, seed=s):
                self.assertEqual(d['production_audit_passed'], d['failed_gates'] == [])
                self.assertEqual((d['method'], d['seed'], d['bank']['rows']), (m, s, lb.N_SYN))

    def tampered(self, m, s, fn):
        c = copy.deepcopy(self.evidence(m, s))
        fn(c)
        return self.tool.build(c, ROOT)['failed_gates']

    def test_fail_closed(self):
        for (m, s) in self.docs:
            with self.subTest(method=m, seed=s):
                def s7_bytes(c):
                    c['s7_equivalence']['runs']['B']['byte_mismatch_count'] = 1
                self.assertIn('S.s7_production_equivalence', self.tampered(m, s, s7_bytes))

                def test_atime(c):
                    c['atime']['by_split']['TEST']['atime_changed'] = 1
                self.assertIn('F.atime_ledger', self.tampered(m, s, test_atime))

                def ckpt(c):
                    k = '%s|%d' % (m, s)
                    f = c['checkpoints_after'][k]['files']
                    next(iter(f.values()))['sha256'] = '0' * 64
                self.assertIn('S.checkpoint_after', self.tampered(m, s, ckpt))

                def exit1(c):
                    c['markers']['exit_raw'] = '1\n'
                self.assertIn('R.exit_zero', self.tampered(m, s, exit1))

                def missing_output(c):
                    c['independent']['files'] = lb.N_SYN - 1
                    c['independent']['problem_count'] = 1
                self.assertIn('S.independent_verification', self.tampered(m, s, missing_output))

                def dirty(c):
                    c['git']['status_porcelain'] = ' M tools/learned_bank.py'
                self.assertIn('A.execution_checkout', self.tampered(m, s, dirty))

                def wrong_commit(c):
                    c['git']['head'] = '18ba00e94813f005876c8d6235b0b848226b2c1d'
                self.assertIn('A.execution_checkout', self.tampered(m, s, wrong_commit))

                def other_bank_changed(c):
                    if c['preflight']['registry']:
                        k = sorted(c['preflight']['registry'])[0]
                        c['registry_after'][k] = dict(c['registry_after'][k], manifest_sha256='0' * 64)
                    else:
                        c['registry_after']['E03/seed_42'] = {'manifest_sha256': None, 'provenance': False,
                                                              'aborted': True, 'mode': '0o755'}
                self.assertIn('F.other_banks_unchanged', self.tampered(m, s, other_bank_changed))

                def storage_uuid(c):
                    c['storage_after']['filesystem']['filesystem_uuid'] = 'x'
                self.assertIn('F.storage_after', self.tampered(m, s, storage_uuid))

                def e01_changed(c):
                    c['e01_e02_manifests_after']['E01|42'] = '0' * 64
                self.assertIn('P.e01_e02_banks_unchanged', self.tampered(m, s, e01_changed))

                def busy_gpu(c):
                    c['preflight']['processes'] = ['123 python tools/learned_bank.py s7']
                self.assertIn('P.resources', self.tampered(m, s, busy_gpu))

                def bank_existed(c):
                    c['preflight']['bank_dir_exists'] = True
                self.assertIn('P.fresh_bank_dir', self.tampered(m, s, bank_existed))


class FamilyClosures(unittest.TestCase):
    def test_rebuild(self):
        tool = load('lbprod3', 'tools/learned_bank_production_audit.py')
        for M in ('E06c', 'E03', 'E06b', 'E05'):
            p = ROOT / (tool.CLOSURE.format(M=M) + '.json')
            if not p.exists():
                continue
            with self.subTest(method=M):
                doc = tool.build_closure(ROOT, M)
                self.assertEqual(json.loads(json.dumps(doc)), json.loads(p.read_text()))
                self.assertEqual(tool.closure_report(doc), (ROOT / (tool.CLOSURE.format(M=M) + '.md')).read_text())
                for line in (ROOT / (tool.CLOSURE.format(M=M) + '_SHA256.txt')).read_text().splitlines():
                    digest, name = line.split('  ')
                    self.assertEqual(sha(ROOT / name), digest)


if __name__ == '__main__':
    unittest.main()
