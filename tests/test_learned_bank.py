"""Framework-free tests of the M8 learned-bank infrastructure (methods/common/learned_bank.py, tools/learned_bank.py).

Runs in the laptop .venv (no torch / TensorFlow). Framework-specific adapter qualification is
tools/learned_bank_fixture_qualification.py (GPU host CPU, CUDA hidden); real S7 acceptance is GPU work after E04.
"""
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402

CONTRACT = ROOT / lb.CONTRACT_PATH


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_root(extra=None, contract_patch=None):
    """A temp repo root with the governance files the gates read (optionally patched / extended)."""
    d = Path(tempfile.mkdtemp(prefix='lbroot_'))
    for rel in (lb.CONTRACT_PATH, lb.DRAFT_PATH):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, d / rel)
    if contract_patch:
        c = json.loads((d / lb.CONTRACT_PATH).read_text())
        contract_patch(c)
        (d / lb.CONTRACT_PATH).write_text(json.dumps(c))
    for rel, obj in (extra or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return d


# ------------------------------------------------------------------ numpy fixture adapters (writer/reader tests)
class FakeDSDG:
    def __init__(self, method, seed, contract_sha, *, nan_at=None, fixture=True):
        self.method_id, self.bank_seed, self.generator_seed = method, seed, seed
        self.is_fixture, self.contract_sha256 = fixture, contract_sha
        self.config_sha256 = lb.METHOD_CONFIGS[method][1]
        self.checkpoint_identity_sha256 = 'f' * 64 if fixture else lb.checkpoint_identity_sha256(method, seed)
        self.nan_at, self.discarded = nan_at, 0

    def identity(self):
        return {'fake': True}

    def generate(self, population, reader=None):
        n = len(population.rows)
        plan = lb.dsdg_stream_plan(n)
        rng = np.random.default_rng(lb.dsdg_bank_seed(self.method_id, self.bank_seed))
        i = 0
        for _ in range(plan['batches']):
            batch = rng.random((100, 3, 256, 256), dtype=np.float32)
            for j in range(100):
                if i >= n:
                    self.discarded += 1
                    continue
                raw = batch[j]
                if i == self.nan_at:
                    raw = raw.copy()
                    raw[0, 0, 0] = np.nan
                ok = bool(np.isfinite(raw).all())
                yield lb.GenResult(population.rows[i], lb.reference_dsdg_uint8(raw) if ok else None, ok,
                                   float(raw.min()) if ok else None, float(raw.max()) if ok else None,
                                   None if ok else 'NONFINITE_GENERATOR_OUTPUT',
                                   noise_seed=lb.dsdg_bank_seed(self.method_id, self.bank_seed))
                i += 1

    def finalize(self):
        return {'discarded_final_batch_samples': self.discarded}


class FakeE05:
    def __init__(self, seed, contract_sha):
        self.method_id, self.bank_seed, self.generator_seed = 'E05', seed, seed
        self.is_fixture, self.contract_sha256 = True, contract_sha
        self.config_sha256, self.checkpoint_identity_sha256 = lb.METHOD_CONFIGS['E05'][1], 'e' * 64

    def identity(self):
        return {'fake': True}

    def generate(self, population, reader):
        from PIL import Image
        for row in population.rows:
            rs, ss = reader.read(row.source_spoof_id)
            rt, st = reader.read(row.target_live_id)
            src = np.asarray(Image.open(io.BytesIO(rs)), np.float32) / 127.5 - 1
            tgt = np.asarray(Image.open(io.BytesIO(rt)), np.float32) / 127.5 - 1
            ps = lb.e05_pair_seed(self.bank_seed, row.pair_id)
            noise = np.random.default_rng(ps).standard_normal(src.shape[:2], dtype=np.float32)[..., None]
            raw = np.transpose(0.6 * src + 0.6 * tgt + 0.3 * noise, (2, 0, 1)).astype(np.float32)
            yield lb.GenResult(row, lb.reference_e05_uint8(raw), True, float(raw.min()), float(raw.max()),
                               noise_seed=ps, target_face_sha256=st, source_face_sha256=ss)

    def finalize(self):
        return {}


def write_faces(root, n_pairs):
    """Synthetic PNG faces (not benchmark data) + a TRAIN allowlist + a pair relation."""
    from PIL import Image
    rng = np.random.default_rng(7)
    samples, relation = {}, []
    for i in range(n_pairs):
        ids = []
        for role in ('live', 'spoof'):
            sid = hashlib.sha256(('%s%d' % (role, i)).encode()).hexdigest()
            d = Path(root) / 'casia_fasd'
            d.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(buf, format='PNG')
            (d / (sid + '.png')).write_bytes(buf.getvalue())
            samples[sid] = {'split': 'TRAIN', 'dataset': 'casia_fasd',
                            'face_png_sha256': hashlib.sha256(buf.getvalue()).hexdigest()}
            ids.append(sid)
        relation.append(('PTR%06d' % (i + 1), ids[0], ids[1], 'casia_fasd'))
    return samples, relation


class Governance(unittest.TestCase):
    def test_contract_approved_equals_draft_recommendations(self):
        rec, digest = lb.load_generation_contract()
        draft = json.loads((ROOT / lb.DRAFT_PATH).read_text())
        self.assertEqual(sha(ROOT / lb.DRAFT_PATH), lb.DRAFT_SHA256)
        self.assertEqual(draft['status'], 'DRAFT_PENDING_OWNER_DECISION')
        self.assertEqual(set(rec['decisions']), set(draft['questions']))
        for q, opt in rec['decisions'].items():
            self.assertIn(opt, draft['questions'][q]['options'])
            self.assertEqual(opt, draft['questions'][q]['recommended'])
        self.assertEqual(digest, sha(CONTRACT))
        a = rec['authorizes']
        for k in ('s7_gpu_execution', 'production_bank_generation', 'generator_retraining', 'checkpoint_change',
                  'val_or_test_access', 'downstream_scientific_training', 'gpat', 'method_status_update'):
            self.assertIs(a[k], False, k)
        self.assertEqual(rec['approved_draft']['assessment_sha256'],
                         sha(ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.md'))

    def test_contract_tamper_blocks(self):
        for patch in (lambda c: c['decisions'].__setitem__('E05-Q1_stylegan_noise', 'B_SEQUENTIAL_STREAM'),
                      lambda c: c.__setitem__('approved', False),
                      lambda c: c['approved_draft'].__setitem__('sha256', '0' * 64),
                      lambda c: c['authorizes'].__setitem__('production_bank_generation', True)):
            root = make_root(contract_patch=patch)
            with self.assertRaises(lb.LearnedBankBlocked) as cm:
                lb.load_generation_contract(root)
            self.assertEqual(cm.exception.status, lb.STATUS_SCIENTIFIC)
        root = Path(tempfile.mkdtemp())
        with self.assertRaises(lb.LearnedBankBlocked):
            lb.load_generation_contract(root)

    def test_seed_vectors_match_contract(self):
        tv = json.loads(CONTRACT.read_text())['byte_level_interpretation']['test_vectors']
        for key, want in tv['E05'].items():
            seed, pid = key.split('|')
            self.assertEqual(lb.e05_pair_seed(int(seed), pid), want)
            pre = b'GPAT_TRANSFERBENCH|E05|BANK|' + seed.encode() + b'|' + pid.encode()
            self.assertEqual(int.from_bytes(hashlib.sha256(pre).digest()[:8], 'big') % (2 ** 31 - 1), want)
        for m in ('E06b', 'E06c'):
            for seed, want in tv[m].items():
                self.assertEqual(lb.dsdg_bank_seed(m, int(seed)), want)
        self.assertNotEqual(lb.dsdg_bank_seed('E06b', 42), lb.dsdg_bank_seed('E06c', 42))
        with self.assertRaises(lb.LearnedBankError):
            lb.e05_pair_seed(42, 'PVA000001')
        with self.assertRaises(lb.LearnedBankError):
            lb.dsdg_bank_seed('E05', 42)

    def test_stream_plans(self):
        p = lb.dsdg_stream_plan(8838)
        self.assertEqual((p['batches'], p['kept_in_final_batch'], p['discarded_final_batch_samples']), (89, 38, 62))
        p = lb.dsdg_stream_plan(186)
        self.assertEqual((p['batches'], p['kept_in_final_batch'], p['discarded_final_batch_samples']), (2, 86, 14))
        self.assertEqual(lb.synthetic_id('E06b', 42, 0), 'E06b_S42_000000')
        self.assertEqual(lb.synthetic_id('E06c', 2026, 8837), 'E06c_S2026_008837')
        self.assertEqual(lb.dsdg_location(185), (1, 85))

    def test_checkpoint_table_equals_assessment(self):
        a = json.loads((ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.json').read_text())
        self.assertEqual(len(lb.CHECKPOINTS), 12)
        for r in a['checkpoint_authority']:
            c = lb.CHECKPOINTS[(r['method'], r['seed'])]
            want = r['sha256'] if isinstance(r['sha256'], dict) else {'pth': r['sha256']}
            self.assertEqual(c['sha256'], want)
            self.assertTrue(all(len(v) == 64 for v in c['sha256'].values()))
            self.assertEqual((c['run_id'], c['seed_closure_commit'], c['family_closure_commit']),
                             (r['run_id'], r['seed_closure_commit'], r['family_closure_commit']))
            self.assertEqual(set(c['sha256']), set(lb.CHECKPOINT_FILES[r['method']]))
            self.assertEqual(r['config_sha256'], lb.METHOD_CONFIGS[r['method']][1])
            self.assertEqual(lb.CHECKPOINT_RULES[r['method']], r['checkpoint_rule'])

    def test_three_seed_closure_csvs_agree(self):
        for m, name in (('E03', 'E03'), ('E05', 'E05'), ('E06b', 'E06B'), ('E06c', 'E06C')):
            text = (ROOT / 'outputs/audit' / ('%s_THREE_SEED_FINAL_CHECKPOINTS.csv' % name)).read_text()
            for seed in lb.SEEDS:
                for digest in lb.CHECKPOINTS[(m, seed)]['sha256'].values():
                    if m != 'E03' or digest == lb.CHECKPOINTS[(m, seed)]['sha256']['data-00000-of-00001']:
                        self.assertIn(digest, text, (m, seed))

    def test_configs_locks_and_sources(self):
        for m, (rel, digest) in lb.METHOD_CONFIGS.items():
            self.assertEqual(lb.verify_config_identity(m)['sha256'], digest)
            lrel, ldigest = lb.ENVIRONMENTS[m]['lock']
            self.assertEqual(sha(ROOT / lrel), ldigest)
        import yaml
        from methods.dsdg.bank_adapter import ATTACK_TYPE
        for m in lb.DSDG_METHODS:
            cfg = yaml.safe_load((ROOT / lb.METHOD_CONFIGS[m][0]).read_text())
            self.assertEqual(cfg['training']['attack_type'], ATTACK_TYPE[m])
            self.assertEqual(cfg['training']['hdim'], lb.DSDG_HDIM)
        table = json.loads((ROOT / lb.SOURCES_PATH).read_text())['families']
        pins = {'STDN': 'c79f1f8c615d2b8471b3df29da881bb18dd54c90', 'PCGAN': '6baa180f1184ee79a6b967f9d80ee0e02a979ac7',
                'DSDG': '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'}
        for fam, t in table.items():
            self.assertEqual(t['commit'], pins[fam])
            lb.verify_pinned_sources(fam)
            repo = ROOT / t['root']
            top = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
            if top.returncode == 0:
                self.assertEqual(top.stdout.strip(), t['commit'])
                for rel in t['files']:
                    blob = subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--', rel])
                    self.assertEqual(blob.returncode, 0, rel)

    def test_toy_manifest(self):
        rows = lb.load_toy_manifest()
        pos = [r['position'] for r in rows]
        self.assertEqual(len(rows), 186)
        self.assertTrue(all(a < b for a, b in zip(pos, pos[1:])))
        self.assertTrue(all(lb.PAIR_ID_RE.match(r['pair_id']) for r in rows))
        self.assertEqual(rows[0]['pair_id'], 'PTR%06d' % (pos[0] + 1))


class Gates(unittest.TestCase):
    def test_s7_and_production_refuse_now(self):
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E06c')
        self.assertEqual(cm.exception.status, lb.STATUS_E04)
        for m in lb.METHODS:
            for s in lb.SEEDS:
                with self.assertRaises(lb.LearnedBankBlocked) as cm:
                    lb.load_production_authority(m, s)
                self.assertIn(cm.exception.status, (lb.STATUS_E04, lb.STATUS_PRODUCTION))
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_storage_authority()
        self.assertEqual(cm.exception.status, lb.STATUS_STORAGE)
        with self.assertRaises(lb.LearnedBankBlocked):
            lb.resolve_registered_bank('E05', 42)
        self.assertFalse((ROOT / lb.S7_AUTHORITY_PATH).exists())
        self.assertFalse((ROOT / lb.STORAGE_AUTHORITY_PATH).exists())
        self.assertFalse((ROOT / lb.E04_COMPLETION_RECORD).exists())

    def _e04(self, family=True):
        e = {lb.E04_COMPLETION_RECORD: {'status': lb.E04_COMPLETION_STATUS, 'failed_gates': [], 'seed': 2026,
                                        'run_id': lb.E04_RUN_ID}}
        if family:
            e[lb.E04_FAMILY_CLOSURE_RECORD] = {'status': lb.E04_FAMILY_CLOSURE_STATUS, 'failed_gates': []}
        return e

    def test_gate_chain_order(self):
        contract_sha = sha(CONTRACT)
        root = make_root({lb.E04_COMPLETION_RECORD: {'status': 'RUNNING'}})
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E03', root)
        self.assertEqual(cm.exception.status, lb.STATUS_E04)
        root = make_root(self._e04(family=False))
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E03', root)
        self.assertEqual(cm.exception.status, lb.STATUS_E04)     # final checkpoint alone / no closure is not enough
        root = make_root(self._e04())
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E03', root)
        self.assertEqual(cm.exception.status, lb.STATUS_S7)
        e04_sha = hashlib.sha256((root / lb.E04_COMPLETION_RECORD).read_bytes()).hexdigest()
        s7 = {'schema_version': lb.S7_AUTHORITY_SCHEMA, 'status': 'OWNER_APPROVED', 'approved': True,
              'methods': ['E05'], 'contract_sha256': contract_sha, 'e04_seed_2026_completion_sha256': e04_sha,
              'resource_preflight': {'required': True}}
        root = make_root(dict(self._e04(), **{lb.S7_AUTHORITY_PATH: s7}))
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E03', root)                    # E03 not covered
        self.assertEqual(cm.exception.status, lb.STATUS_S7)
        auth = lb.load_s7_authority('E05', root)
        self.assertEqual((auth.mode, auth.methods, auth.seeds), ('S7_TOY', ('E05',), lb.SEEDS))
        prod = {'schema_version': lb.PRODUCTION_AUTHORITY_SCHEMA, 'status': 'OWNER_APPROVED', 'approved': True,
                'method': 'E05', 'seeds': [42], 'contract_sha256': contract_sha,
                's7_acceptance_record': {'path': 'outputs/audit/S7.json', 'sha256': '0' * 64}}
        root = make_root(dict(self._e04(), **{lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E05'): prod}))
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_production_authority('E05', 42, root)        # S7 record not bound
        self.assertEqual(cm.exception.status, lb.STATUS_PRODUCTION)
        s7rec = {'method': 'E05', 's7_acceptance_passed': True, 'failed_gates': [], 'seeds': [42, 1337, 2026]}
        prod['s7_acceptance_record']['sha256'] = hashlib.sha256(json.dumps(s7rec).encode()).hexdigest()
        root = make_root(dict(self._e04(), **{lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method='E05'): prod,
                                              'outputs/audit/S7.json': s7rec}))
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_production_authority('E05', 42, root)        # storage not activated
        self.assertEqual(cm.exception.status, lb.STATUS_STORAGE)

    def test_pending_storage_record_is_not_an_authority(self):
        pending = {'schema_version': lb.STORAGE_AUTHORITY_SCHEMA, 'status': 'OWNER_SELECTED_PENDING_ACTIVATION',
                   'approved': False, 'decisions': {'L1_bank_root': 'B_EXTERNAL_HDD_ADDITIVE_ROOT'}}
        root = make_root({lb.STORAGE_PENDING_PATH: pending})
        with self.assertRaises(lb.LearnedBankBlocked):
            lb.load_storage_authority(root)
        root = make_root({lb.STORAGE_AUTHORITY_PATH: dict(pending)})
        with self.assertRaises(lb.LearnedBankBlocked):
            lb.load_storage_authority(root)

    def test_storage_preflight_readonly(self):
        missing = lb.storage_preflight('/nonexistent/student20261_gpat_m8_banks')
        self.assertFalse(missing['ok'])
        self.assertFalse(missing['exists'])
        d = Path(tempfile.mkdtemp())
        os.chmod(d, 0o700)
        pf = lb.storage_preflight(str(d))
        self.assertTrue(pf['checks']['mode_0700_no_setgid'])
        self.assertFalse(pf['checks']['filesystem_uuid'])
        self.assertFalse(pf['ok'])
        os.chmod(d, 0o2700)
        self.assertFalse(lb.storage_preflight(str(d))['checks']['mode_0700_no_setgid'])
        link = Path(tempfile.mkdtemp()) / 'link'
        os.symlink(d, link)
        self.assertFalse(lb.storage_preflight(str(link))['checks']['not_symlink'])
        self.assertFalse(lb.storage_preflight('/media/student20261/x/y')['ok'])


class WriterReader(unittest.TestCase):
    def setUp(self):
        self.auth = lb.fixture_authorization('unit')
        self.tmp = Path(tempfile.mkdtemp(prefix='lbbank_'))

    def tearDown(self):
        for p in self.tmp.rglob('*'):
            if p.is_dir() and not p.is_symlink():
                os.chmod(p, 0o755)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dsdg(self, n=186, method='E06c', seed=42, **kw):
        return FakeDSDG(method, seed, self.auth.contract_sha256, **kw), lb.dsdg_population(method, seed, n,
                                                                                         'SYNTHETIC_FIXTURE')

    def test_dsdg_bank_roundtrip_and_replay(self):
        a, pop = self.dsdg()
        prov = lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        self.assertEqual((prov['status'], prov['rows'], prov['outputs']), ('completed', 186, 186))
        self.assertEqual(prov['adapter_finalize']['discarded_final_batch_samples'], 14)
        r = lb.read_bank(self.tmp / 'A', expect_method='E06c', expect_seed=42, expect_mode='FIXTURE', expect_rows=186)
        self.assertTrue(r['ok'], r['problems'])
        row = r['rows'][185]
        self.assertEqual((row['row_id'], row['pair_id'], row['target_live_id'], row['source_spoof_id']),
                         ('E06c_S42_000185', None, None, None))
        self.assertEqual((row['dsdg_batch_index'], row['dsdg_batch_offset']), (1, 85))
        a2, pop2 = self.dsdg()
        lb.write_bank(a2, pop2, self.auth, self.tmp / 'B')
        cmp = lb.compare_runs(self.tmp / 'A', self.tmp / 'B')
        self.assertTrue(cmp['bitwise_identical'], cmp)
        # toy prefix equals the corresponding rows of a longer stream (toy == production rows rule)
        a3, pop3 = self.dsdg(n=250)
        lb.write_bank(a3, pop3, self.auth, self.tmp / 'C')
        for i in (0, 99, 100, 185):
            name = 'E06c_S42_%06d.png' % i
            self.assertEqual((self.tmp / 'A/bank' / name).read_bytes(), (self.tmp / 'C/bank' / name).read_bytes())
        # immutability
        self.assertEqual(stat.S_IMODE(os.stat(self.tmp / 'A/bank/E06c_S42_000000.png').st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(os.stat(self.tmp / 'A').st_mode), 0o555)

    def test_create_only(self):
        a, pop = self.dsdg(n=5)
        lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        a, pop = self.dsdg(n=5)
        with self.assertRaises(lb.LearnedBankError):
            lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        with self.assertRaises(lb.LearnedBankError):
            lb.write_bank(a, pop, self.auth, self.tmp / 'missing_parent' / 'A')

    def test_nonfinite_is_counted_failure(self):
        a, pop = self.dsdg(n=7, nan_at=3)
        prov = lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        self.assertEqual((prov['status'], prov['failure_count'], prov['outputs']), ('completed_with_failures', 1, 6))
        r = lb.read_bank(self.tmp / 'A')
        self.assertTrue(r['ok'], r['problems'])
        self.assertEqual(r['rows'][3]['generation_status'], 'failure')
        self.assertFalse((self.tmp / 'A/bank/E06c_S42_000003.png').exists())

    def test_reader_detects_tampering(self):
        a, pop = self.dsdg(n=4)
        lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        os.chmod(self.tmp / 'A', 0o755)
        os.chmod(self.tmp / 'A/bank', 0o755)
        p = self.tmp / 'A/bank/E06c_S42_000001.png'
        os.chmod(p, 0o644)
        raw = bytearray(p.read_bytes())
        raw[-20] ^= 1
        p.write_bytes(bytes(raw))
        r = lb.read_bank(self.tmp / 'A')
        self.assertFalse(r['ok'])
        (self.tmp / 'A/bank/extra.png').write_bytes(b'x')
        self.assertTrue(any('file set' in x for x in lb.read_bank(self.tmp / 'A')['problems']))

    def test_reader_rejects_false_pair_identity_for_dsdg(self):
        a, pop = self.dsdg(n=3)
        lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        os.chmod(self.tmp / 'A', 0o755)
        lines = (self.tmp / 'A' / lb.MANIFEST).read_text().splitlines()
        r = json.loads(lines[0])
        r['pair_id'] = 'PTR000001'
        lines[0] = lb.canonical_json(r)
        body = ('\n'.join(lines) + '\n').encode()
        os.chmod(self.tmp / 'A' / lb.MANIFEST, 0o644)
        os.chmod(self.tmp / 'A' / lb.MANIFEST_SHA, 0o644)
        (self.tmp / 'A' / lb.MANIFEST).write_bytes(body)
        (self.tmp / 'A' / lb.MANIFEST_SHA).write_text('%s  %s\n' % (hashlib.sha256(body).hexdigest(), lb.MANIFEST))
        res = lb.read_bank(self.tmp / 'A')
        self.assertTrue(any('DSDG synthetic identity' in x for x in res['problems']), res['problems'])

    def test_mode_separation(self):
        a, pop = self.dsdg(n=3, fixture=False)
        with self.assertRaises(lb.LearnedBankError):            # fixture mode refuses a real-checkpoint adapter
            lb.write_bank(a, pop, self.auth, self.tmp / 'A')
        a, _ = self.dsdg(n=3)
        toy = lb.dsdg_population('E06c', 42, 186, 'S7_TOY_POPULATION')
        with self.assertRaises(lb.LearnedBankError):            # a fixture authorization cannot write an S7 bank
            lb.write_bank(a, toy, self.auth, self.tmp / 'B')
        fake_s7 = lb.Authorization(mode='S7_TOY', methods=('E06c',), seeds=lb.SEEDS,
                                   contract_sha256=self.auth.contract_sha256, label='s7_toy')
        with self.assertRaises(lb.LearnedBankError):            # an S7 authorization refuses fixture weights
            lb.write_bank(a, toy, fake_s7, self.tmp / 'C')
        a, pop = self.dsdg(n=3)
        (self.tmp / 'data/processed/faces_256').mkdir(parents=True)
        with self.assertRaises(lb.LearnedBankError):
            lb.write_bank(a, pop, self.auth, self.tmp / 'data/processed/faces_256/x')
        with self.assertRaises(lb.LearnedBankError):            # production population sizes are exact
            lb.dsdg_population('E06b', 42, 8837, 'PRODUCTION_FROZEN_TRAIN_POPULATION')

    def test_pair_bank_with_noatime_reader(self):
        faces = self.tmp / 'faces'
        samples, relation = write_faces(faces, 6)
        pop = lb.pair_population('E05', 42, 'SYNTHETIC_FIXTURE', relation, positions=[0, 2, 5])
        reader = lb.TrainFaceReader(faces, samples)
        prov = lb.write_bank(FakeE05(42, self.auth.contract_sha256), pop, self.auth, self.tmp / 'A', reader=reader)
        self.assertEqual(prov['reader']['atime_checked_reads'], 6)
        r = lb.read_bank(self.tmp / 'A', expect_method='E05')
        self.assertTrue(r['ok'], r['problems'])
        self.assertEqual([x['row_index'] for x in r['rows']], [0, 2, 5])
        self.assertEqual(r['rows'][1]['deterministic_noise_seed'], lb.e05_pair_seed(42, 'PTR000003'))
        self.assertEqual(r['rows'][1]['source_face_sha256'], samples[relation[2][2]]['face_png_sha256'])
        # order independence of the per-pair seed: a single-pair bank reproduces row PTR000003 exactly
        pop1 = lb.pair_population('E05', 42, 'SYNTHETIC_FIXTURE', relation, positions=[2])
        lb.write_bank(FakeE05(42, self.auth.contract_sha256), pop1, self.auth, self.tmp / 'B',
                      reader=lb.TrainFaceReader(faces, samples))
        self.assertEqual((self.tmp / 'A/bank/PTR000003.png').read_bytes(),
                         (self.tmp / 'B/bank/PTR000003.png').read_bytes())


class FaceFirewall(unittest.TestCase):
    def test_reader_refusals(self):
        tmp = Path(tempfile.mkdtemp())
        samples, relation = write_faces(tmp, 2)
        reader = lb.TrainFaceReader(tmp, samples)
        sid = relation[0][1]
        st0 = os.stat(tmp / 'casia_fasd' / (sid + '.png')).st_atime_ns
        reader.read(sid)
        self.assertEqual(os.stat(tmp / 'casia_fasd' / (sid + '.png')).st_atime_ns, st0)
        with self.assertRaises(lb.LearnedBankError):
            reader.read('0' * 64)                               # outside the allowlist: no path exists
        bad = dict(samples)
        bad[sid] = dict(bad[sid], split='VAL')
        with self.assertRaises(lb.LearnedBankError):
            lb.TrainFaceReader(tmp, bad)
        bad = dict(samples)
        bad[sid] = dict(bad[sid], face_png_sha256='0' * 64)
        with self.assertRaises(lb.LearnedBankError):
            lb.TrainFaceReader(tmp, bad).read(sid)


class Conversions(unittest.TestCase):
    def test_reference_conversions(self):
        x = np.array([[[-0.5, 0.0, 0.5], [0.999, 1.0, 1.7]]], np.float32)
        self.assertEqual(lb.reference_e03_uint8(x).tolist(), [[[0, 0, 127], [254, 255, 255]]])
        chw = np.array([[[-1.2, -1.0, 0.0, 0.999, 1.0, 3.0]]], np.float32).repeat(3, 0)
        self.assertEqual(lb.reference_e05_uint8(chw)[0, :, 0].tolist(), [0, 0, 127, 254, 255, 255])
        s = np.array([[[0.0, 0.5, 0.99999, 1.0]]], np.float32).repeat(3, 0)
        self.assertEqual(lb.reference_dsdg_uint8(s)[0, :, 0].tolist(), [0, 127, 254, 255])


class NonRegression(unittest.TestCase):
    def test_e01_e02_authority_files_unchanged(self):
        auth = json.loads((ROOT / 'configs/amendments/e01_e02_production_authority.yaml').read_text())
        for rel, digest in auth['authority_files'].items():
            self.assertEqual(sha(ROOT / rel), digest, rel)
        from methods.common import bank_writer as bw
        self.assertEqual(lb.PNG_COMPRESS_LEVEL, bw.PNG_COMPRESS_LEVEL)
        arr = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.assertEqual(lb.encode_png(arr), bw.encode_png(arr, 'RGB'))
        self.assertEqual(lb.pixel_sha256(arr), bw.pixel_sha256(arr))

    def test_protected_paths_unchanged_since_assessment(self):
        paths = ['methods/common/bank_writer.py', 'outputs/audit/method_status.csv', 'configs/methods',
                 'frozen_config_snapshot', 'outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.json',
                 'outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.json', lb.DRAFT_PATH,
                 'configs/amendments/learned_bank_storage_location.DRAFT.yaml',
                 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.json']
        r = subprocess.run(['git', '-C', str(ROOT), 'diff', '--name-only', '3433ad040df88af5d0a61f6d1ce9198d087b35c9',
                            '--'] + paths, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), '')


class CLI(unittest.TestCase):
    def test_s7_and_production_refuse_before_any_work(self):
        out = Path(tempfile.mkdtemp())
        for argv in (['s7', '--method', 'E06c', '--label', 'A', '--out-root', str(out), '--runtime-root', '/nonexistent'],
                     ['production', '--method', 'E03', '--seed', '42', '--runtime-root', '/nonexistent']):
            p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/learned_bank.py')] + argv,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, p.stderr)
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertEqual(rec['status'], lb.STATUS_E04)
        self.assertEqual(list(out.iterdir()), [])
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/learned_bank.py'), 'gates', '--method', 'E05'],
                           capture_output=True, text=True)
        rep = json.loads(p.stdout)
        self.assertTrue(rep['contract']['ok'])
        self.assertFalse(rep['s7']['ok'] or rep['production']['ok'] or rep['storage_authority']['ok'])


if __name__ == '__main__':
    unittest.main()
