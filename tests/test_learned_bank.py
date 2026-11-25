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


def live_s7_expectation():
    """The refusal the live repository state must give for S7 (None = S7 authorised). The state only moves forward:
    E04 open -> BLOCKED_BY_E04; E04 seed 2026 + family closed (0e3f1d1 + family closure) but no owner S7 launch
    authority -> BLOCKED_BY_S7; S7 authority committed -> S7 loads."""
    if not lb.e04_completion_state()['ok']:
        return lb.STATUS_E04
    if not (ROOT / lb.S7_AUTHORITY_PATH).exists():
        return lb.STATUS_S7
    return None


def live_production_expectation(method):
    """Production refusal for `method` under the live state (None = authorised). Without the owner's per-method
    learned_bank_production_<M>.yaml -> BLOCKED_BY_PRODUCTION; with it, the live HDD preflight decides: off the GPU host
    (no bank root) -> BLOCKED_BY_STORAGE; on the GPU host with the verified HDD -> authorised."""
    if not lb.e04_completion_state()['ok']:
        return lb.STATUS_E04
    if not (ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=method)).exists():
        return lb.STATUS_PRODUCTION
    return None if lb.storage_preflight()['ok'] else lb.STATUS_STORAGE


class Gates(unittest.TestCase):
    def test_s7_and_production_refuse_now(self):
        want = live_s7_expectation()
        if want is None:
            auth = lb.load_s7_authority('E06c')
            self.assertEqual((auth.mode, auth.methods, auth.seeds), ('S7_TOY', ('E06c',), lb.SEEDS))
        else:
            with self.assertRaises(lb.LearnedBankBlocked) as cm:
                lb.load_s7_authority('E06c')
            self.assertEqual(cm.exception.status, want)
        for m in lb.METHODS:
            pwant = live_production_expectation(m)
            for s in lb.SEEDS:
                if pwant is None:
                    auth = lb.load_production_authority(m, s)
                    self.assertEqual((auth.mode, auth.methods, auth.seeds), ('PRODUCTION', (m,), (s,)))
                    continue
                with self.assertRaises(lb.LearnedBankBlocked) as cm:
                    lb.load_production_authority(m, s)
                self.assertEqual(cm.exception.status, pwant)
        # storage authority is activated (L1 B) but is never itself a production authorization
        storage = lb.load_storage_authority()
        self.assertEqual(storage['bank_root'], lb.HDD['bank_root'])
        self.assertEqual(lb.resolve_registered_bank('E05', 42),
                         Path('/media/ext_data/student20261_gpat_m8_banks/E05/seed_42'))
        self.assertEqual((ROOT / lb.S7_AUTHORITY_PATH).exists(), want is None)
        self.assertEqual((ROOT / lb.E04_COMPLETION_RECORD).exists(), want != lb.STATUS_E04)
        for m in lb.METHODS:            # any committed production authority is an owner approval binding a passed S7
            p = ROOT / lb.PRODUCTION_AUTHORITY_TEMPLATE.format(method=m)
            if p.exists():
                rec = json.loads(p.read_text())
                self.assertEqual((rec['schema_version'], rec['status'], rec['approved'], rec['method']),
                                 (lb.PRODUCTION_AUTHORITY_SCHEMA, 'OWNER_APPROVED', True, m))
                s7 = ROOT / rec['s7_acceptance_record']['path']
                self.assertEqual(sha(s7), rec['s7_acceptance_record']['sha256'])
                self.assertTrue(json.loads(s7.read_text())['s7_acceptance_passed'])

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


def activated_storage(**patch):
    reg = {'bank_root': lb.HDD['bank_root'], 'mount_point': lb.HDD['mount_point'],
           'filesystem_uuid': lb.HDD['filesystem_uuid'], 'fstype': 'ext4', 'owner': 'student20261', 'mode': '0700',
           'layout': lb.BANK_ROOT_LAYOUT, 'methods': list(lb.METHODS)}
    rec = {'schema_version': lb.STORAGE_AUTHORITY_SCHEMA, 'status': 'OWNER_APPROVED', 'approved': True,
           'decisions': {'L1_bank_root': 'B_EXTERNAL_HDD_ADDITIVE_ROOT'}, 'bank_root_registry': reg,
           'activation_verification': {'ok': True}}
    for k, v in patch.items():
        if k in reg:
            reg[k] = v
        else:
            rec[k] = v
    return rec


class StoragePreflight(unittest.TestCase):
    """Live-check semantics on a local temp directory (expected identity patched to the temp filesystem)."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix='lbhdd_')) / 'student20261_gpat_m8_banks'
        os.mkdir(self.d, 0o700)
        os.chmod(self.d, 0o700)
        self.fs = lb.filesystem_identity(str(self.d))
        import pwd
        self.exp = {'mount_point': self.fs['mount_point'], 'filesystem_uuid': self.fs['filesystem_uuid'],
                    'fstype': self.fs['fstype'], 'owner': pwd.getpwuid(os.getuid()).pw_name}

    def pf(self, root=None, **exp):
        return lb.storage_preflight(str(root or self.d), dict(self.exp, **exp))

    def test_correct_identity_passes(self):
        if self.fs['filesystem_uuid'] is None:
            self.skipTest('temp filesystem has no /dev/disk/by-uuid entry')
        r = self.pf()
        self.assertTrue(r['ok'], r['checks'])
        self.assertEqual(r['unexpected_entries'], [])

    def test_wrong_uuid_refused(self):
        r = self.pf(filesystem_uuid='00000000-0000-0000-0000-000000000000')
        self.assertFalse(r['checks']['filesystem_uuid'])
        self.assertFalse(r['ok'])
        # the real approved UUID is not the temp filesystem's
        self.assertFalse(lb.storage_preflight(str(self.d), {k: v for k, v in self.exp.items()
                                                            if k != 'filesystem_uuid'})['checks']['filesystem_uuid'])

    def test_missing_mount_refused(self):
        r = self.pf(mount_point='/media/ext_data')         # directory exists but not under the approved mount
        self.assertFalse(r['checks']['mount_point'])
        self.assertFalse(r['ok'])
        r = lb.storage_preflight(lb.HDD['bank_root'] + '_absent_mount_probe')
        self.assertEqual((r['exists'], r['ok']), (False, False))

    def test_read_only_mount_refused(self):
        real = lb.filesystem_identity
        for key, val in (('mount_options', 'ro,relatime'), ('super_options', 'ro,errors=remount-ro')):
            def fake(path, key=key, val=val):
                out = real(path)
                out[key] = val
                return out
            lb.filesystem_identity = fake
            try:
                r = self.pf()
            finally:
                lb.filesystem_identity = real
            self.assertFalse(r['checks']['mount_read_write'], key)
            self.assertFalse(r['ok'])

    def test_symlink_refused(self):
        link = self.d.parent / 'link_root'
        os.symlink(self.d, link)
        r = self.pf(link)
        self.assertFalse(r['checks']['not_symlink'])
        self.assertFalse(r['checks']['realpath_equals_path'])
        self.assertFalse(r['ok'])

    def test_incorrect_owner_refused(self):
        r = self.pf(owner='root' if os.getuid() != 0 else 'nobody')
        self.assertFalse(r['checks']['owner'])
        self.assertFalse(r['ok'])

    def test_incorrect_mode_refused(self):
        for mode in (0o2700, 0o750, 0o755, 0o770, 0o1700):
            os.chmod(self.d, mode)
            r = self.pf()
            self.assertFalse(r['checks']['mode_0700_no_setgid'], oct(mode))
            self.assertFalse(r['ok'])
        os.chmod(self.d, 0o700)

    def test_unexpected_contents_refused(self):
        (self.d / 'stray.txt').write_bytes(b'x')
        r = self.pf()
        self.assertEqual(r['unexpected_entries'], ['stray.txt'])
        self.assertFalse(r['ok'])
        os.unlink(self.d / 'stray.txt')
        os.symlink('/tmp', self.d / 'E05')                  # a method name that is a symlink
        self.assertFalse(self.pf()['checks']['no_unexpected_contents'])
        os.unlink(self.d / 'E05')
        os.mkdir(self.d / 'E05')
        os.mkdir(self.d / 'E05' / 'seed_42')
        self.assertTrue(self.pf()['checks']['no_unexpected_contents'])     # the registered layout is expected
        os.mkdir(self.d / 'E05' / 'seed_7')
        self.assertEqual(self.pf()['unexpected_entries'], ['E05/seed_7'])
        os.rmdir(self.d / 'E05' / 'seed_7')
        os.mkdir(self.d / 'E99')
        self.assertEqual(self.pf()['unexpected_entries'], ['E99'])


class BankRootRegistry(unittest.TestCase):
    def test_registry_correctness(self):
        root = make_root({lb.STORAGE_AUTHORITY_PATH: activated_storage()})
        st = lb.load_storage_authority(root)
        for m in lb.METHODS:
            for s in lb.SEEDS:
                p = lb.resolve_registered_bank(m, s, root)
                self.assertEqual(p, Path(lb.HDD['bank_root']) / m / ('seed_%d' % s))
                self.assertEqual(lb.production_bank_dir(m, s, st), p)
                self.assertFalse(str(p).startswith('/media/student20261/'))
                self.assertTrue(str(p).startswith('/media/ext_data/student20261_gpat_m8_banks/'))

    def test_registry_mismatch_refused(self):
        for patch in ({'bank_root': '/media/student20261/608647d2-9cb0-45b0-b999-73c080ba909b/x'},
                      {'bank_root': '/home/student20261/workdir/GPAT_TransferBench_runtime/banks'},
                      {'filesystem_uuid': '227ce2f6-373e-4f13-9cdb-84b911775706'}, {'mount_point': '/'},
                      {'fstype': 'ntfs'}, {'mode': '2700'}, {'owner': 'alex'}, {'methods': ['E05']},
                      {'layout': '<bank_root>/<seed>/<method>/'}, {'approved': False}, {'status': 'DRAFT'},
                      {'activation_verification': {'ok': False}},
                      {'decisions': {'L1_bank_root': 'A_ROOT_SSD_ONE_FAMILY_AT_A_TIME'}}):
            root = make_root({lb.STORAGE_AUTHORITY_PATH: activated_storage(**patch)})
            with self.assertRaises(lb.LearnedBankBlocked, msg=str(patch)) as cm:
                lb.load_storage_authority(root)
            self.assertEqual(cm.exception.status, lb.STATUS_STORAGE)

    def test_path_traversal_refused(self):
        st = {'bank_root': lb.HDD['bank_root']}
        for m, s in (('..', 42), ('../E05', 42), ('E05/../../x', 42), ('/etc', 42), ('E05', -1), ('E05', '42'),
                     ('E05', 43), ('e05', 42), ('E01', 42), ('E02', 42)):
            with self.assertRaises(lb.LearnedBankError, msg=(m, s)):
                lb.production_bank_dir(m, s, st)
        for bad_root in (lb.HDD['bank_root'] + '/../x', lb.HDD['bank_root'] + '/', 'media/ext_data/x',
                         '/media/student20261/608647d2-9cb0-45b0-b999-73c080ba909b/student20261_gpat_m8_banks',
                         '/home/student20261/workdir/GPAT_TransferBench_runtime'):
            with self.assertRaises(lb.LearnedBankError, msg=bad_root):
                lb.production_bank_dir('E05', 42, {'bank_root': bad_root})

    def test_production_still_refused_with_storage_activated(self):
        root = make_root({lb.STORAGE_AUTHORITY_PATH: activated_storage()})
        for m in lb.METHODS:
            with self.assertRaises(lb.LearnedBankBlocked) as cm:
                lb.load_production_authority(m, 42, root)
            self.assertEqual(cm.exception.status, lb.STATUS_E04)


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
        want = live_s7_expectation()
        # production is only ever invoked here for a refusal: a method without authority, or (off the GPU host) a
        # method whose authority exists but whose HDD bank root is absent. Never on a host where it could write.
        cases = []
        for m in lb.METHODS:
            pwant = live_production_expectation(m)
            if pwant is not None:
                cases.append((['production', '--method', m, '--seed', '42', '--runtime-root', '/nonexistent'], pwant))
                break
        if want is not None:            # once S7 is authorised the s7 command proceeds; production still refuses
            cases.insert(0, (['s7', '--method', 'E06c', '--label', 'A', '--out-root', str(out),
                              '--runtime-root', '/nonexistent'], want))
        for argv, status in cases:
            p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/learned_bank.py')] + argv,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, p.stderr)
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertEqual(rec['status'], status)
        self.assertEqual(list(out.iterdir()), [])
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/learned_bank.py'), 'gates', '--method', 'E05'],
                           capture_output=True, text=True)
        rep = json.loads(p.stdout)
        self.assertTrue(rep['contract']['ok'])
        self.assertEqual(rep['s7']['ok'], want is None)
        self.assertEqual(rep['production']['ok'], live_production_expectation('E05') is None)
        self.assertTrue(rep['storage_authority']['ok'])         # activated L1 B record; never a production grant


if __name__ == '__main__':
    unittest.main()


class FixtureQualificationEvidence(unittest.TestCase):
    """Pins the committed CPU fixture-qualification evidence (b09a1cc); never a substitute for S7."""
    EXPECTED = {'DSDG': 60, 'E05': 33, 'E03': 32}

    def test_evidence(self):
        sums = (ROOT / 'outputs/audit/LEARNED_BANK_FIXTURE_QUALIFICATION_SHA256.txt').read_text().split('\n')
        listed = dict(reversed(line.split('  ')) for line in sums if line.strip())
        for fam, n in self.EXPECTED.items():
            rel = 'outputs/audit/LEARNED_BANK_FIXTURE_QUALIFICATION_%s.json' % fam
            self.assertEqual(sha(ROOT / rel), listed[rel])
            d = json.loads((ROOT / rel).read_text())
            self.assertEqual((d['gates_passed'], d['gates_total'], d['failed_gates']), (n, n, []))
            self.assertEqual(d['label'], 'SYNTHETIC_FIXTURE_QUALIFICATION_NOT_S7_NOT_BENCHMARK')
            self.assertIs(d['gpu_bitwise_reproducibility_proven'], False)
            self.assertTrue(d['git_head'].startswith('b09a1cc'))
            self.assertEqual(d['environment']['cuda_visible_devices'], '')
            self.assertEqual(d['gates']['zero_opens_under_runtime_data']['detail']['runtime_data'], 0)
            self.assertIsNone(d['error'])


class ReadinessRecords(unittest.TestCase):
    def test_storage_pending_record_is_not_usable(self):
        rec = json.loads((ROOT / lb.STORAGE_PENDING_PATH).read_text())
        self.assertEqual((rec['status'], rec['approved'], rec['usable_for_banks']),
                         ('OWNER_SELECTED_PENDING_ACTIVATION', False, False))
        self.assertEqual(rec['owner_selection']['L1_bank_root'], 'B_EXTERNAL_HDD_ADDITIVE_ROOT')
        draft = json.loads((ROOT / rec['draft']['path']).read_text())
        self.assertIn('B_EXTERNAL_HDD_ADDITIVE_ROOT', draft['questions']['L1_bank_root']['options'])
        self.assertEqual(sha(ROOT / rec['draft']['path']), rec['draft']['sha256'])
        self.assertEqual(rec['target']['bank_root'], lb.HDD['bank_root'])
        self.assertEqual(rec['target']['filesystem_uuid'], lb.HDD['filesystem_uuid'])
        self.assertIn('chmod 00700 /media/ext_data/student20261_gpat_m8_banks',
                      rec['required_owner_action']['commands_run_by_the_owner_on_the_gpu_host'])
        # pinned to history (not HEAD): at 2892533 no activated authority existed, and both earlier records are
        # byte-unchanged since then
        r = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', '2892533:' + lb.STORAGE_AUTHORITY_PATH],
                           capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        r = subprocess.run(['git', '-C', str(ROOT), 'diff', '--name-only', '28925330e1481b6c6ebdef8d6eaad6f40b0bed4c',
                            '--', lb.STORAGE_PENDING_PATH, rec['draft']['path']], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, ''))

    def test_activated_storage_authority_binds_verification(self):
        rec = json.loads((ROOT / lb.STORAGE_AUTHORITY_PATH).read_text())
        self.assertEqual((rec['status'], rec['approved'], rec['decisions']['L1_bank_root']),
                         ('OWNER_APPROVED', True, 'B_EXTERNAL_HDD_ADDITIVE_ROOT'))
        reg = rec['bank_root_registry']
        self.assertEqual((reg['bank_root'], reg['mount_point'], reg['filesystem_uuid'], reg['fstype'], reg['mode']),
                         ('/media/ext_data/student20261_gpat_m8_banks', '/media/ext_data',
                          '608647d2-9cb0-45b0-b999-73c080ba909b', 'ext4', '0700'))
        for k in ('draft', 'pending_activation'):
            self.assertEqual(sha(ROOT / rec['supersedes_nothing'][k]['path']), rec['supersedes_nothing'][k]['sha256'])
        act = rec['activation_verification']
        self.assertTrue(act['ok'])
        self.assertTrue(all(act['preflight_checks'].values()))
        self.assertEqual(len(act['preflight_checks']), 13)
        ev_path = ROOT / act['evidence']['path']
        self.assertEqual(sha(ev_path), act['evidence']['sha256'])
        ev = json.loads(ev_path.read_text())
        self.assertEqual((ev['status'], ev['ok']), ('PASSED', True))
        self.assertTrue(ev['storage_preflight']['result']['ok'])
        self.assertEqual(ev['storage_preflight']['result']['checks'], act['preflight_checks'])
        self.assertEqual(ev['physical_device']['filesystem_uuid'], lb.HDD['filesystem_uuid'])
        pr = ev['permission_probe']
        self.assertTrue(pr['ok'] and pr['hash_match'] and pr['probe_removed'] and pr['dir_unchanged'])
        self.assertEqual((pr['contents_after'], ev['final_directory_state']['contents']), ([], []))
        self.assertEqual(ev['directory_creation']['final']['mode'], '0o700')
        self.assertEqual(ev['git_source_authority']['verifier_commit'], act['verifier_commit'])
        r = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', act['verifier_commit'], 'HEAD'])
        self.assertEqual(r.returncode, 0)
        for m in lb.METHODS:
            self.assertEqual(ev['registry_preview'][m],
                             [str(lb.resolve_registered_bank(m, s)) for s in lb.SEEDS])

    def test_s7_plan_and_snapshot(self):
        plan = json.loads((ROOT / 'outputs/audit/LEARNED_BANK_S7_ACCEPTANCE_PLAN.json').read_text())
        self.assertEqual(plan['status'], 'PREPARED_NOT_EXECUTED')
        self.assertEqual(sorted(plan['methods']), sorted(lb.METHODS))
        for m, p in plan['methods'].items():
            for s in lb.SEEDS:
                c = p['checkpoints'][str(s)]
                self.assertEqual(c['checkpoint_identity_sha256'], lb.checkpoint_identity_sha256(m, s))
                self.assertEqual(sorted(c['files'].values()), sorted(lb.CHECKPOINTS[(m, s)]['sha256'].values()))
        snap = json.loads((ROOT / 'outputs/audit/LEARNED_BANK_READINESS_SNAPSHOT.json').read_text())
        self.assertTrue(snap['checkpoints_all_equal_approved'])
        self.assertEqual(snap['collected']['sha256'], sha(ROOT / snap['collected']['path']))
        self.assertEqual(snap['e04_seed2026']['status'], 'RUNNING_NOT_COMPLETE')
        self.assertFalse(snap['learned_gpu_inference_run'])
