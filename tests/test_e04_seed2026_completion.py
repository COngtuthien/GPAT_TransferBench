"""E04 seed2026 closure: evidence checks for the completed E04 Physics-STD seed 2026 scientific run (run_id
e7100eb9a5ab0537, launch authority 18ba00e, production qualification b0f0ffd under D12 c60481b, CONTROLLED_ADAPTATION,
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
SEED = 2026
RUN_ID = 'e7100eb9a5ab0537'
PAIR_SEED, HARD_SEED = 692890966, 1053992267
FINAL_DATA_SHA = '089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2'
LISTING_DIGEST = '3503f2d15e21a7c20b837aabe20c5b85908ab697db94f9b3bc5ec4e851eb7b10'
METRICS_SHA = '831e5a8e1d3063b141725ec0586b234849a700eedb39dd5f5a8601881d6e21cd'
AUTHORITY = '18ba00e94813f005876c8d6235b0b848226b2c1d'
CONFIG_SHA = '418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a'
D12_CRITERIA = '9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d'
PREFIX = 'E04_SEED%d_' % SEED


def tool():
    spec = importlib.util.spec_from_file_location('e04_seed_audit', ROOT / 'tools' / 'e04_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE04Seed2026(unittest.TestCase):
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


    def test_13_val_zero_whole_window(self):
        f = self.doc['firewall']['by_split']
        self.assertEqual((f['VAL']['files'], f['VAL']['sensitive_whole_window'], f['VAL']['atime_changed']),
                         (3121, 3121, 0))
        g = {x['gate']: x for x in self.doc['gates']}
        self.assertTrue(g['H.val_zero']['pass'] and g['H.test_zero']['pass'])

    def test_14_uninterrupted_boot(self):
        ok, detail = tool().boot_continuity(self.c)
        self.assertTrue(ok)
        self.assertIsNone(detail['next_boot_wtmp'])
        self.assertIn('still running', detail['run_boot_wtmp'])


class SiblingSessionPredicate(unittest.TestCase):
    """J.no_other_e04_runner: a leftover e04_* tmux session passes only when it is another registered seed's session
    with one dead pane (status 0), a durable exit marker '0' and an end marker before this seed's launch. Seed 42 and
    seed 1337 (no sibling session) keep the original rule and re-derive byte-identically."""
    @classmethod
    def setUpClass(cls):
        cls.mod = tool()
        cls.host = json.loads((AUDIT / (PREFIX + 'COMPLETION.json')).read_text())['collected']['host']

    def variant(self, **kw):
        h = json.loads(json.dumps(self.host))
        h['sibling_e04_sessions']['e04_physics_std_seed1337'].update(kw)
        return h

    def test_finished_earlier_sibling_passes(self):
        ok, detail = self.mod.finished_siblings(self.host, SEED)
        self.assertTrue(ok)
        self.assertEqual(detail['e04_physics_std_seed1337']['end_time'], '2026-11-21T08:09:40Z')
        self.assertEqual(self.host['other_e04_tmux'], ['e04_physics_std_seed1337', 'e04_physics_std_seed2026'])

    def test_live_pane_fails(self):
        self.assertFalse(self.mod.finished_siblings(self.variant(panes=[['0', '']]), SEED)[0])
        self.assertFalse(self.mod.finished_siblings(self.variant(panes=[['1', '1']]), SEED)[0])

    def test_nonzero_or_missing_exit_fails(self):
        self.assertFalse(self.mod.finished_siblings(self.variant(exit_file_raw='1\n'), SEED)[0])
        self.assertFalse(self.mod.finished_siblings(self.variant(exit_file_raw=None), SEED)[0])

    def test_overlapping_end_fails(self):
        self.assertFalse(self.mod.finished_siblings(self.variant(end_time='2026-11-22T00:00:00Z'), SEED)[0])
        self.assertFalse(self.mod.finished_siblings(self.variant(end_time=None), SEED)[0])

    def test_unknown_session_fails(self):
        h = json.loads(json.dumps(self.host))
        h['other_e04_tmux'].append('e04_scratch')
        self.assertFalse(self.mod.finished_siblings(h, SEED)[0])
        h = json.loads(json.dumps(self.host))
        del h['sibling_e04_sessions']
        self.assertFalse(self.mod.finished_siblings(h, SEED)[0])

    def test_runner_process_still_fails_gate(self):
        c = json.loads((AUDIT / (PREFIX + 'COMPLETION.json')).read_text())
        cc, tt = c['collected'], c['collected_tf']
        cc['host']['runner_processes'] = ['123 python tools/run_e04.py --seed 1337']
        doc, _ = self.mod.build(cc, tt, ROOT)
        self.assertIn('J.no_other_e04_runner', doc['failed_gates'])

    def test_historical_seeds_unaffected(self):
        for s in (42, 1337):
            h = json.loads((AUDIT / ('E04_SEED%d_COMPLETION.json' % s)).read_text())['collected']['host']
            self.assertNotIn('sibling_e04_sessions', h)
            self.assertEqual(self.mod.finished_siblings(h, s), (True, {}))


if __name__ == '__main__':
    unittest.main()
