"""E04 Physics-STD bank adapter and additive learned-bank extension (laptop; no TensorFlow, no SciPy, no data).

TF-dependent gates (graph integrity, restore, frozen-graph comparison, BN-mode isolation, numpy references) run on the
GPU host CPU in tools/e04_bank_fixture_qualification.py; its committed evidence is checked by
tests/test_e04_bank_fixture_qualification_record.py.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import learned_bank as lb  # noqa: E402
from methods.physics_std import bank_adapter as ba  # noqa: E402

CONTRACT = ROOT / lb.E04_CONTRACT_PATH


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy_into(tmp, rels):
    for rel in rels:
        Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(ROOT / rel), str(Path(tmp, rel)))


class Contract(unittest.TestCase):
    def test_approved_and_binds_unchanged_draft(self):
        rec, digest = lb.load_e04_generation_contract()
        self.assertEqual(digest, sha(CONTRACT))
        self.assertEqual(sha(ROOT / lb.E04_DRAFT_PATH), lb.E04_DRAFT_SHA256)
        draft = json.loads((ROOT / lb.E04_DRAFT_PATH).read_text())
        self.assertEqual(rec['decisions'], {q['id']: q['recommended'] for q in draft['owner_questions']})
        self.assertEqual(rec['frozen_by_existing_authority'], draft['frozen_by_existing_authority'])
        for k, v in rec['frozen_inputs_sha256'].items():
            self.assertEqual(sha(ROOT / k.split(' (')[0]), v, k)
        self.assertEqual(sha(ROOT / rec['scientific_closure']['path']), rec['scientific_closure']['sha256'])
        a = rec['authorizes']
        self.assertEqual((a['s7_gpu_inference'], a['production_bank_generation'], a['storage_directory_creation'],
                          a['generator_retraining']), (False, False, False, False))
        self.assertTrue(any('batch statistics over 8-image training batches' in d for d in rec['scientific_disclosures']))

    def test_tampered_contract_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            copy_into(t, [lb.E04_CONTRACT_PATH, lb.E04_DRAFT_PATH])
            rec = json.loads(CONTRACT.read_text())
            for mutate in (lambda r: r['decisions'].update({'E04-Q1_normalization_statistics':
                                                            'B_TRAINING_BATCH_STATISTICS_FROZEN_BATCHES'}),
                           lambda r: r['authorizes'].update({'production_bank_generation': True}),
                           lambda r: r['checkpoints']['per_seed']['42']['sha256'].update({'index': '0' * 64}),
                           lambda r: r.update({'status': 'DRAFT'})):
                bad = json.loads(json.dumps(rec))
                mutate(bad)
                Path(t, lb.E04_CONTRACT_PATH).write_text(json.dumps(bad))
                with self.assertRaises(lb.LearnedBankBlocked) as cm:
                    lb.load_e04_generation_contract(t)
                self.assertEqual(cm.exception.status, lb.STATUS_SCIENTIFIC)
            Path(t, lb.E04_CONTRACT_PATH).write_text(json.dumps(rec))
            Path(t, lb.E04_DRAFT_PATH).write_text('{}')
            with self.assertRaises(lb.LearnedBankBlocked):
                lb.load_e04_generation_contract(t)


class Checkpoints(unittest.TestCase):
    def test_table_equals_closure_inventory_and_contract(self):
        with open(str(ROOT / 'outputs/audit/E04_THREE_SEED_CHECKPOINT_INVENTORY.csv')) as f:
            inv = [r for r in csv.DictReader(f) if r['iteration'] == '150000']
        contract = json.loads(CONTRACT.read_text())['checkpoints']['per_seed']
        for seed in lb.SEEDS:
            want = {r['file'].split('iteration_150000.')[1]: r['sha256'] for r in inv if r['seed'] == str(seed)}
            self.assertEqual(lb.E04_CHECKPOINTS[('E04', seed)]['sha256'], want)
            self.assertEqual(contract[str(seed)]['checkpoint_identity_sha256'], lb.checkpoint_identity_sha256('E04', seed))
            self.assertEqual(contract[str(seed)]['run_id'], lb.checkpoint_authority('E04', seed)['run_id'])

    def test_original_tables_unchanged(self):
        self.assertEqual(lb.METHODS, ('E03', 'E05', 'E06b', 'E06c'))
        self.assertEqual(lb.PAIR_METHODS, ('E03', 'E05'))
        self.assertEqual(len(lb.CHECKPOINTS), 12)
        self.assertNotIn('E04', {m for m, _ in lb.CHECKPOINTS})

    def test_verify_checkpoint_refuses_wrong_root(self):
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / 'runs' / 'm6' / 'E04' / 'seed_42'
            (bad / 'checkpoints').mkdir(parents=True)
            for rel in lb.CHECKPOINT_FILES['E04'].values():
                (bad / rel).write_bytes(b'x')
            with self.assertRaises(lb.LearnedBankBlocked) as cm:
                lb.verify_checkpoint('E04', 42, bad)
            self.assertEqual(cm.exception.status, lb.STATUS_CHECKPOINT)


class Governance(unittest.TestCase):
    def test_s7_production_storage_refuse_e04(self):
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_s7_authority('E04')
        self.assertEqual(cm.exception.status, lb.STATUS_S7)
        for s in lb.SEEDS:
            with self.assertRaises(lb.LearnedBankBlocked) as cm:
                lb.load_production_authority('E04', s, live_storage=False)
            self.assertEqual(cm.exception.status, lb.STATUS_PRODUCTION)
        with self.assertRaises(lb.LearnedBankBlocked) as cm:
            lb.load_storage_authority(method_id='E04')
        self.assertEqual(cm.exception.status, lb.STATUS_STORAGE)
        for p in (lb.E04_S7_AUTHORITY_PATH, lb.E04_STORAGE_EXTENSION_PATH, 'configs/amendments/learned_bank_production_E04.yaml'):
            self.assertFalse((ROOT / p).exists(), p)

    def test_existing_methods_unaffected(self):
        for m in lb.METHODS:
            self.assertEqual(lb.load_method_contract(m), lb.load_generation_contract())
            self.assertEqual(lb.load_s7_authority(m).methods, (m,))
            for s in lb.SEEDS:
                a = lb.load_production_authority(m, s, live_storage=False)
                self.assertIn(lb.CONTRACT_PATH, dict(a.records))
                self.assertNotIn(lb.E04_CONTRACT_PATH, dict(a.records))
        self.assertEqual(sorted(lb.load_storage_authority()['methods']), sorted(lb.METHODS))

    def test_fixture_authorization_scope(self):
        a = lb.e04_fixture_authorization('e04_t')
        self.assertEqual((a.mode, a.methods, a.contract_sha256), ('FIXTURE', ('E04',), sha(CONTRACT)))
        self.assertNotEqual(lb.fixture_authorization('x').contract_sha256, a.contract_sha256)

    def _ext(self, t, **over):
        base_sha = sha(ROOT / lb.STORAGE_AUTHORITY_PATH)
        rec = {'schema_version': lb.E04_STORAGE_EXTENSION_SCHEMA, 'status': 'OWNER_APPROVED', 'approved': True,
               'methods': ['E04'], 'base_storage_authority': {'path': lb.STORAGE_AUTHORITY_PATH, 'sha256': base_sha},
               'registry_extension': {'bank_root': lb.HDD['bank_root'], 'filesystem_uuid': lb.HDD['filesystem_uuid'],
                                      'layout': lb.E04_BANK_LAYOUT, 'seeds': list(lb.SEEDS)},
               'activation_verification': {'ok': True}}
        rec.update(over)
        copy_into(t, [lb.STORAGE_AUTHORITY_PATH])
        Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_text(json.dumps(rec))

    def test_registry_accepts_e04_only_when_extension_activated(self):
        self.assertEqual(lb.registered_bank_methods(), lb.METHODS)
        with tempfile.TemporaryDirectory() as t:
            copy_into(t, [lb.STORAGE_AUTHORITY_PATH])
            self.assertEqual(lb.registered_bank_methods(t), lb.METHODS)
            Path(t, lb.E04_STORAGE_EXTENSION_PATH.replace('.yaml', '.DRAFT.yaml')).write_text('{}')
            self.assertEqual(lb.registered_bank_methods(t), lb.METHODS)
            self._ext(t)
            self.assertEqual(lb.registered_bank_methods(t), lb.BANK_METHODS)
            st = lb.load_storage_authority(t, method_id='E04')
            self.assertIn('E04', st['methods'])
            self.assertEqual(lb.production_bank_dir('E04', 42, st), Path(lb.HDD['bank_root']) / 'E04' / 'seed_42')
            for bad in ({'status': 'DRAFT'}, {'methods': ['E04', 'E07']}, {'activation_verification': {'ok': False}},
                        {'base_storage_authority': {'path': lb.STORAGE_AUTHORITY_PATH, 'sha256': '0' * 64}}):
                self._ext(t, **bad)
                self.assertEqual(lb.registered_bank_methods(t), lb.METHODS, bad)
        with self.assertRaises(lb.LearnedBankError):
            lb.production_bank_dir('E04', 42, lb.load_storage_authority())

    def test_root_content_checker(self):
        with tempfile.TemporaryDirectory() as t:
            for m in ('E03', 'E04'):
                Path(t, m, 'seed_42').mkdir(parents=True)
            uid = os.getuid()
            self.assertEqual(lb.bank_root_contents_ok(t, uid), ['E04'])
            self.assertEqual(lb.bank_root_contents_ok(t, uid, lb.BANK_METHODS), [])


class Conversion(unittest.TestCase):
    def test_cases(self):
        f = np.float32
        x = np.array([-1, -0.0, 0, 0.5 / 255, 1 / 255, 0.5, 127.5 / 255, 254.5 / 255, np.nextafter(f(1), f(0)), 1, 1.5,
                      3], f).reshape(1, 4, 3)
        got = ba.convert_uint8(x).ravel().tolist()
        self.assertEqual(got, [int(f(min(max(v, f(0)), f(1))) * f(255)) for v in x.ravel()])
        self.assertEqual(got[5], 127)                         # 127.5 truncates (rint would give 128)
        self.assertEqual((got[0], got[9], got[10], got[11]), (0, 255, 255, 255))

    def test_equals_reference_and_rejects_nonfinite_and_float64(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-0.3, 1.3, (256, 256, 3)).astype(np.float32)
        self.assertTrue(np.array_equal(ba.convert_uint8(x), lb.reference_e04_uint8(x)))
        for bad in (np.array([[[np.nan, 0, 0]]], np.float32), np.array([[[np.inf, 0, 0]]], np.float32),
                    np.zeros((2, 2, 3), np.float64)):
            with self.assertRaises(lb.LearnedBankError):
                ba.convert_uint8(bad)

    def test_rule_string_and_d8_decode(self):
        self.assertIn('A_PREDECESSOR_CLIP_TRUNCATE', lb.CONVERSION_RULES['E04'])
        from PIL import Image
        img = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        out = ba.decode_face(buf.getvalue())
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.array_equal(out, img.astype(np.float32) / 255))


class Source(unittest.TestCase):
    def test_frozen_sources_and_inference_layers(self):
        src = ba.verify_frozen_sources()
        self.assertEqual(src['files'], ba.FROZEN_FILES)
        self.assertTrue(ba.inference_layers_differ_only_in_bn_mode())
        for rel in ('methods/physics_std/network.py', 'methods/physics_std/trace_transfer.py',
                    'methods/physics_std/training_graph.py', 'methods/physics_std/production_runner.py',
                    'methods/physics_std/production_io.py', 'tools/run_e04.py'):
            r = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', '18ba00e94813f005876c8d6235b0b848226b2c1d',
                                '--', rel])
            self.assertEqual(r.returncode, 0, rel + ' changed since the E04 training authority')

    def test_graph_class_refuses_training_layers(self):
        from methods.physics_std import network
        with self.assertRaises(lb.LearnedBankError):
            ba.E04InferenceGraph(None, layers_cls=network.Layers)


class StubAdapter(ba.E04BankAdapter):
    """Replaces only the TF forward by a deterministic numpy function; reader, decoding, conversion, failure
    accounting and the common writer/reader are the real code."""
    nan_rows = ()

    def synthesize(self, live, spoof, q_live, q_spoof):
        self.forwards += 1
        out = (0.7 * live + 0.5 * spoof - 0.1).astype(np.float32)
        if self.forwards in self.nan_rows:
            out[3, 4, 1] = np.nan
        return out


def fixture(tmp, n=4):
    from PIL import Image
    rng = np.random.default_rng(5)
    samples, relation = {}, []
    for i in range(n):
        ids = []
        for role in ('live', 'spoof'):
            sid = hashlib.sha256(('t|%s|%d' % (role, i)).encode()).hexdigest()
            d = Path(tmp, 'faces', 'msu_mfsd')
            d.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(buf, format='PNG')
            (d / (sid + '.png')).write_bytes(buf.getvalue())
            samples[sid] = {'split': 'TRAIN', 'dataset': 'msu_mfsd', 'face_png_sha256': sha256b(buf.getvalue())}
            ids.append(sid)
        relation.append(('PTR%06d' % (i + 1), ids[0], ids[1], 'msu_mfsd'))
    return Path(tmp, 'faces'), samples, relation


def sha256b(b):
    return hashlib.sha256(b).hexdigest()


class WriterReader(unittest.TestCase):
    def make(self, nan_rows=()):
        ad = StubAdapter(None, None, None, lambda sid: np.zeros((140, 2), np.float32), 7, is_fixture=True,
                         checkpoint_identity_sha256='f' * 64, contract_sha256=sha(CONTRACT),
                         config_sha256=lb.METHOD_CONFIGS['E04'][1], identity_extra={'stub': True})
        ad.nan_rows = nan_rows
        return ad

    def test_banks_replay_failures_and_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            faces, samples, relation = fixture(t)
            auth = lb.e04_fixture_authorization('e04_stub')
            pop = lambda: lb.pair_population('E04', 7, 'SYNTHETIC_FIXTURE', relation, positions=[0, 1, 3])  # noqa
            pa = lb.write_bank(self.make(), pop(), auth, Path(t, 'A'), reader=lb.TrainFaceReader(faces, samples))
            lb.write_bank(self.make(), pop(), auth, Path(t, 'B'), reader=lb.TrainFaceReader(faces, samples))
            ra = lb.read_bank(Path(t, 'A'), expect_method='E04', expect_mode='FIXTURE', expect_rows=3)
            self.assertTrue(ra['ok'], ra['problems'])
            self.assertTrue(lb.compare_runs(Path(t, 'A'), Path(t, 'B'))['bitwise_identical'])
            self.assertEqual(pa['reader']['atime_checked_reads'], 6)
            r = ra['rows'][1]
            self.assertEqual((r['pair_id'], r['target_live_id'], r['source_spoof_id']), relation[1][:3])
            self.assertIsNone(r['deterministic_noise_seed'])
            self.assertEqual(pa['adapter']['decisions'], lb.E04_APPROVED_DECISIONS)
            with self.assertRaises(lb.LearnedBankError):
                lb.write_bank(self.make(), pop(), auth, Path(t, 'A'), reader=lb.TrainFaceReader(faces, samples))
            pn = lb.write_bank(self.make(nan_rows=(2,)), pop(), auth, Path(t, 'N'),
                               reader=lb.TrainFaceReader(faces, samples))
            rn = lb.read_bank(Path(t, 'N'), expect_method='E04', expect_mode='FIXTURE', expect_rows=3)
            self.assertTrue(rn['ok'], rn['problems'])
            self.assertEqual((pn['failure_count'], rn['rows'][1]['generation_status'], rn['rows'][1]['output_path']),
                             (1, 'failure', None))
            self.assertEqual(len(list(Path(t, 'N', 'bank').iterdir())), 2)

    def test_reader_detects_e04_noise_seed_and_wrong_contract_refused(self):
        with tempfile.TemporaryDirectory() as t:
            faces, samples, relation = fixture(t, 2)
            auth = lb.e04_fixture_authorization('e04_stub')
            pop = lb.pair_population('E04', 7, 'SYNTHETIC_FIXTURE', relation, positions=[0])
            lb.write_bank(self.make(), pop, auth, Path(t, 'A'), reader=lb.TrainFaceReader(faces, samples))
            man = Path(t, 'A', lb.MANIFEST)
            os.chmod(str(Path(t, 'A')), 0o755)
            row = json.loads(man.read_text())
            row['deterministic_noise_seed'] = 1
            os.chmod(str(man), 0o644)
            man.write_text(lb.canonical_json(row) + '\n')
            self.assertTrue(any('not stochastic' in p for p in lb.read_bank(Path(t, 'A'))['problems']))
            with self.assertRaises(lb.LearnedBankError):     # base-contract fixture authorization cannot write E04
                lb.write_bank(self.make(), pop, lb.fixture_authorization('x'), Path(t, 'C'),
                              reader=lb.TrainFaceReader(faces, samples))


class CLI(unittest.TestCase):
    def run_cli(self, *argv):
        return subprocess.run([sys.executable, str(ROOT / 'tools/learned_bank.py')] + list(argv), cwd=str(ROOT),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_gates_and_refusals(self):
        r = self.run_cli('gates', '--method', 'E04')
        self.assertEqual(r.returncode, 0)
        g = json.loads(r.stdout)
        self.assertEqual((g['contract']['ok'], g['s7']['status'], g['production']['status'],
                          g['storage_authority']['status']),
                         (True, lb.STATUS_S7, lb.STATUS_PRODUCTION, lb.STATUS_STORAGE))
        with tempfile.TemporaryDirectory() as t:
            r = self.run_cli('s7', '--method', 'E04', '--label', 'A', '--out-root', t, '--runtime-root', '/nonexistent')
            self.assertEqual((r.returncode, json.loads(r.stdout)['status']), (3, lb.STATUS_S7))
            self.assertEqual(os.listdir(t), [])
        r = self.run_cli('production', '--method', 'E04', '--seed', '42', '--runtime-root', '/nonexistent')
        self.assertEqual((r.returncode, json.loads(r.stdout)['status']), (3, lb.STATUS_PRODUCTION))
        r = self.run_cli('s7', '--method', 'E04', '--label', 'A', '--out-root', '/x', '--runtime-root', '/y',
                         '--seed', '42')
        self.assertNotEqual(r.returncode, 0)                  # per-seed selection stays E05-only (owner ruling B2)


if __name__ == '__main__':
    unittest.main()
