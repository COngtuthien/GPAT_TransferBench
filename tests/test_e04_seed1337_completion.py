"""E04 seed1337 closure: evidence checks for the completed E04 Physics-STD seed 1337 scientific run (run_id
a42fdd2a2accaf0a, launch authority 18ba00e, production qualification b0f0ffd under D12 c60481b, CONTROLLED_ADAPTATION,
fresh only).

Stdlib only, any host. The run root lives on the GPU host; these tests validate the committed evidence (completion
JSON with both embedded read-only collected documents, checkpoint manifest, report, SHA-256 manifest) and re-derive
every evidence file from the collected documents to prove the build is deterministic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
SEED = 1337
RUN_ID = 'a42fdd2a2accaf0a'
PAIR_SEED, HARD_SEED = 1349475396, 35554372
FINAL_DATA_SHA = 'fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5'
LISTING_DIGEST = '6a8f471effd41d0e1eb93bbfa455c86cd64518a30fc36278bd3cd9e6fec7585c'
METRICS_SHA = 'eda6e3940136bd3a9ec8b092ef3559e5c77ad0e5dc690edc53e02dd5047748a9'
AUTHORITY = '18ba00e94813f005876c8d6235b0b848226b2c1d'
CONFIG_SHA = '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a'
D12_CRITERIA = '9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d'
PREFIX = 'E04_SEED%d_' % SEED


def tool():
    spec = importlib.util.spec_from_file_location('e04_seed_audit', ROOT / 'tools' / 'e04_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE04Seed1337(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((AUDIT / (PREFIX + 'COMPLETION.json')).read_text())
        cls.c, cls.t = cls.doc['collected'], cls.doc['collected_tf']
        cls.m = cls.c['metrics']

    def test_01_status_seed_run_id_authority(self):
        d = self.doc
        self.assertEqual(d['status'], 'E04_SEED%d_COMPLETED_CLEANLY' % SEED)
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual(d['gates_passed'], d['gates_total'])
        self.assertEqual((d['seed'], d['run_id'], d['scientific_authority'], d['config_sha256']),
                         (SEED, RUN_ID, AUTHORITY, CONFIG_SHA))
        self.assertEqual(hashlib.sha256(('E04|%d|%s|%s' % (SEED, CONFIG_SHA, AUTHORITY)).encode()).hexdigest()[:16],
                         RUN_ID)
        man = self.c['run_manifest']
        self.assertEqual((man['run_id'], man['git_commit'], man['git_dirty'], man['experiment_seed']),
                         (RUN_ID, AUTHORITY, False, SEED))
        self.assertEqual((d['method']['fidelity'], d['resume']), ('CONTROLLED_ADAPTATION', 'NOT_QUALIFIED_FRESH_ONLY'))
        self.assertEqual(self.c['event_counts'], {'e04_checkpoint_event': 10, 'e04_run_start': 1})
        self.assertEqual(d['d12']['acceptance_criteria_sha256'], D12_CRITERIA)

    def test_02_exit_zero_completed_no_reboot_in_window(self):
        h, s = self.c['host'], self.c['run_summary']
        self.assertEqual((h['exit_file_raw'], h['runner_processes']), ('0\n', []))
        self.assertEqual((s['completion_status'], s['failure_reason']), ('completed', None))
        g = {x['gate']: x for x in self.doc['gates']}
        self.assertTrue(g['B.no_reboot_during_run']['pass'])
        self.assertEqual(g['B.no_reboot_during_run']['detail']['records_inside_window'], [])

    def test_03_counts(self):
        self.assertEqual(self.doc['counts'], {
            'iterations': 150000, 'final_global_iteration': 150000, 'g_applications': 300000,
            'd_applications': 150000, 'checkpoints': 10, 'checkpoint_files': 30, 'metrics_records': 150011,
            'face_reads': 1200000, 'pair_rows_consumed': 600000})

    def test_04_lifecycle_order(self):
        rle = self.m['rle']
        self.assertEqual(rle[0], ['EVENT:e04_run_start', 1])
        for k in range(10):
            self.assertEqual(rle[1 + 2 * k:3 + 2 * k], [['ITERATION', 15000], ['EVENT:e04_checkpoint_event', 1]])
        self.assertEqual(len(rle), 21)
        self.assertTrue(all(v == 150000 for v in self.m['checks'].values()))

    def test_05_rng_streams(self):
        r = self.doc['rng']
        mod = tool()
        self.assertEqual((r['pair_order_seed'], r['hard_sample_seed']), (PAIR_SEED, HARD_SEED))
        self.assertEqual((mod.derive_seed('PAIR_ORDER', SEED), mod.derive_seed('HARD_SAMPLE', SEED)),
                         (PAIR_SEED, HARD_SEED))
        v = json.loads((AUDIT / 'E04_RNG_TEST_VECTORS.json').read_text())['seeds'][str(SEED)]
        self.assertEqual(r['pair_sha256_int32_le'], v['pair_stream']['sha256_int32_le'])
        self.assertEqual(r['hard_sha256'], v['hard_stream']['sha256_float32_scales_then_keep_per_iteration'])
        self.assertTrue(all(x == 150000 for x in r['record_by_record_replay'].values()))

    def test_06_optimizer_counters(self):
        mod = tool()
        seq = mod.beta_power_sequence(0.999, 60000)
        ck = {r['iteration']: r for r in self.t['checkpoints']}
        self.assertEqual(ck[15000]['beta_powers']['beta2_power'], seq[30000])
        self.assertEqual(ck[15000]['beta_powers']['beta2_power_1'], seq[15000])
        self.assertEqual(ck[30000]['beta_powers']['beta2_power'], seq[60000])
        for it, r in ck.items():
            self.assertEqual((r['global_iteration'], r['minibatch_phase'], r['nonfinite_variables']), (it, 0, 0))

    def test_07_checkpoints(self):
        ck = self.t['checkpoints']
        self.assertEqual([r['iteration'] for r in ck], list(range(15000, 150001, 15000)))
        final = self.doc['checkpoints']['final_selected']
        self.assertEqual((final['prefix'], final['rule'], final['sha256']['.data-00000-of-00001']),
                         ('checkpoints/iteration_150000', 'BASELINE_FINAL_STATE_V1', FINAL_DATA_SHA))
        self.assertEqual(self.c['run_summary']['final_or_selected_checkpoint_sha256'], FINAL_DATA_SHA)
        self.assertEqual((self.c['partial_files'], self.c['resume_files'], self.c['tmp_files']), ([], [], []))

    def test_08_test_isolation(self):
        f = self.doc['firewall']
        self.assertEqual((f['test_access'], f['test_changed'], f['test_files']), (0, 0, 3027))
        self.assertEqual(f['by_split']['TEST']['sensitive_whole_window'], 3027)
        self.assertEqual(f['by_split']['VAL']['atime_changed'], 0)
        self.assertTrue(all(v['changed_not_consumed'] == 0 for v in f['by_split'].values()))
        self.assertEqual((self.c['pack']['test_rows'], self.c['pack']['val_rows']), (0, 0))
        self.assertFalse(self.c['run_manifest']['test_split_accessed'])

    def test_09_driver_ruling(self):
        r = self.doc['driver_ruling']
        self.assertEqual((r['lock_driver_version'], r['observed_driver_version']), ('595.84', '580.178.04'))
        self.assertEqual(self.t['environment_recheck']['differences_other_than_driver'], [])

    def test_10_run_root_unchanged(self):
        r = self.doc['run_root_integrity']
        self.assertTrue(r['unchanged_whole_audit'])
        self.assertEqual((r['sha_listing_digest'], r['metrics_jsonl_sha256']), (LISTING_DIGEST, METRICS_SHA))
        lines = ''.join(x + '\n' for x in self.c['run_root_sha_listing']).encode()
        self.assertEqual(hashlib.sha256(lines).hexdigest(), LISTING_DIGEST)

    def test_11_deterministic_rebuild(self):
        mod = tool()
        doc, text = mod.build(self.c, self.t, ROOT)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, (AUDIT / (PREFIX + 'CHECKPOINT_MANIFEST.txt')).read_text())
        self.assertEqual(mod.report(doc), (AUDIT / (PREFIX + 'COMPLETION.md')).read_text())

    def test_12_evidence_sha_manifest(self):
        names = []
        for line in (AUDIT / (PREFIX + 'EVIDENCE_SHA256.txt')).read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/%s%s' % (PREFIX, x) for x in
                                         ('CHECKPOINT_MANIFEST.txt', 'COMPLETION.json', 'COMPLETION.md')])


class BootContinuityPredicate(unittest.TestCase):
    """B.no_reboot_during_run: the uninterrupted-boot branch (seed 1337) passes; any reboot/shutdown or journal boot
    inside the window still fails; the seed-42 rebooted-after branch is unchanged."""
    @classmethod
    def setUpClass(cls):
        cls.mod = tool()
        cls.c1337 = json.loads((AUDIT / 'E04_SEED1337_COMPLETION.json').read_text())['collected']
        cls.c42 = json.loads((AUDIT / 'E04_SEED42_COMPLETION.json').read_text())['collected']

    def variant(self, c, wtmp=None, journal=None, btime=None):
        c = json.loads(json.dumps(c))
        b = c['host']['boot']
        if wtmp is not None:
            b['wtmp'] = wtmp
        if journal is not None:
            b['journal_boots'] = journal
        if btime is not None:
            b['btime'] = btime
        return c

    def test_uninterrupted_passes(self):
        ok, detail = self.mod.boot_continuity(self.c1337)
        self.assertTrue(ok)
        self.assertIsNone(detail['next_boot_wtmp'])
        self.assertIn('still running', detail['run_boot_wtmp'])

    def test_seed42_rebooted_after_still_passes(self):
        self.assertTrue(self.mod.boot_continuity(self.c42)[0])

    def test_reboot_inside_window_fails(self):
        w = list(self.c1337['host']['boot']['wtmp'])
        inside = 'reboot   system boot  7.0.0-43-generic 2026-11-19T12:00:00+07:00   still running'
        w[0] = w[0].replace('   still running', ' - 2026-11-19T11:59:00+07:00 (1+20:42)')
        c = self.variant(self.c1337, wtmp=[inside, 'shutdown system down  7.0.0-43-generic 2026-11-19T11:59:00+07:00 '
                                                   '- 2026-11-19T12:00:00+07:00  (00:01)'] + w)
        self.assertFalse(self.mod.boot_continuity(c)[0])

    def test_journal_boot_inside_window_fails(self):
        j = json.loads(json.dumps(self.c1337['host']['boot']['journal_boots']))
        j[-1]['first_entry'] = int(self.mod.epoch_seconds('2026-11-19T00:00:00Z') * 1e6)
        self.assertFalse(self.mod.boot_continuity(self.variant(self.c1337, journal=j))[0])

    def test_btime_after_launch_fails(self):
        c = self.variant(self.c1337, btime=int(self.mod.epoch_seconds('2026-11-19T00:00:00Z')))
        self.assertFalse(self.mod.boot_continuity(c)[0])


if __name__ == '__main__':
    unittest.main()
