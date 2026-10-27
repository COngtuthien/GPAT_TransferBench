"""E03 seed42 closure: evidence checks for the completed E03 STDN seed 42 scientific run (run_id a565b76bc7349532,
training authority d547fc9, FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY,
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
SEED = 42
RUN_ID = 'a565b76bc7349532'
VAL_SEED = 1355875380
FINAL_DATA_SHA = '5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753'
LISTING_DIGEST = 'a7bed081022c09ab753ae359536e71a964966a484e45f435d51a5809b509b163'
AUTHORITY = 'd547fc9f8932a273d719ae589aade85a9d989b85'
CONFIG_SHA = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
PREFIX = 'E03_SEED%d_' % SEED


def tool():
    spec = importlib.util.spec_from_file_location('e03_seed_audit', ROOT / 'tools' / 'e03_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE03Seed42(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((AUDIT / (PREFIX + 'COMPLETION.json')).read_text())
        cls.c, cls.t = cls.doc['collected'], cls.doc['collected_tf']
        cls.m = cls.c['metrics']

    def test_01_status_seed_run_id_authority(self):
        d = self.doc
        self.assertEqual(d['status'], 'E03_SEED%d_COMPLETED_CLEANLY' % SEED)
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual(d['gates_passed'], d['gates_total'])
        self.assertEqual((d['seed'], d['run_id'], d['scientific_authority'], d['config_sha256']),
                         (SEED, RUN_ID, AUTHORITY, CONFIG_SHA))
        self.assertEqual(hashlib.sha256(('E03|%d|%s|%s' % (SEED, CONFIG_SHA, AUTHORITY)).encode()).hexdigest()[:16],
                         RUN_ID)
        man = self.c['run_manifest']
        self.assertEqual((man['run_id'], man['git_commit'], man['git_dirty'], man['experiment_seed']),
                         (RUN_ID, AUTHORITY, False, SEED))
        self.assertEqual(d['method']['fidelity'],
                         'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY')
        self.assertEqual(d['resume'], 'NOT_QUALIFIED_FRESH_ONLY')
        self.assertEqual(self.c['event_counts'], {'e03_checkpoint_event': 50, 'e03_run_start': 1})

    def test_02_exit_zero_completed(self):
        h, s = self.c['host'], self.c['run_summary']
        self.assertEqual((h['exit_file'], h['tmux_pane_dead'], h['tmux_pane_dead_status'], h['runner_processes']),
                         ('0', True, 0, []))
        self.assertEqual((s['completion_status'], s['failure_reason']), ('completed', None))

    def test_03_counts(self):
        self.assertEqual(self.doc['counts'], {
            'epochs': 50, 'train_steps': 100000, 'val_steps': 25000, 'val_blocks': 50, 'g_applications': 100000,
            'd_applications': 50000, 'final_global_step': 150000, 'checkpoints': 50, 'checkpoint_files': 150})

    def test_04_lifecycle_order(self):
        rle = self.m['rle']
        self.assertEqual(rle[0], ['EVENT:e03_run_start', None, 1])
        for e in range(1, 51):
            self.assertEqual(rle[3 * e - 2:3 * e + 1], [['TRAIN_STEP', e, 2000], ['EVENT:e03_checkpoint_event', e, 1],
                                                        ['VAL_STEP', e, 500]])
        self.assertEqual(len(rle), 151)

    def test_05_independent_rng_streams(self):
        r = self.doc['rng']
        self.assertEqual(r['VAL']['parser_seed'], VAL_SEED)
        self.assertEqual(tool().val_seed(SEED), VAL_SEED)
        for sp, n in (('TRAIN', 200000), ('VAL', 50000)):
            self.assertEqual((r['flips'][sp]['match'], r['flips'][sp]['pairs']), (n, n))
            self.assertEqual(self.m['streams'][sp]['sha256'], self.t['predicted_streams'][sp]['sha256'])

    def test_06_checkpoints(self):
        ck = self.t['checkpoints']
        self.assertEqual([r['epoch'] for r in ck], list(range(1, 51)))
        for r in ck:
            self.assertEqual((r['variables'], r['nonfinite_variables'], r['global_step']), (747, 0, 3000 * r['epoch']))
        final = self.doc['checkpoints']['final_selected']
        self.assertEqual((final['prefix'], final['rule'], final['sha256']['.data-00000-of-00001']),
                         ('checkpoints/ckpt-50', 'OFFICIAL_LATEST_FINAL_CKPT_50', FINAL_DATA_SHA))
        self.assertEqual(self.c['run_summary']['final_or_selected_checkpoint_sha256'], FINAL_DATA_SHA)
        self.assertEqual((self.c['partial_files'], self.c['resume_files'], self.c['runner_state_files']), ([], [], []))

    def test_07_test_isolation(self):
        f = self.doc['firewall']
        self.assertEqual((f['test_access'], f['test_changed'], f['test_files']), (0, 0, 3027))
        self.assertEqual(f['by_split']['TEST']['sensitive_whole_window'], 3027)
        self.assertTrue(all(v['changed_not_consumed'] == 0 for v in f['by_split'].values()))
        self.assertEqual(self.c['pack']['test_rows'], 0)
        self.assertFalse(self.c['run_manifest']['test_split_accessed'])

    def test_08_driver_ruling(self):
        r = self.doc['driver_ruling']
        self.assertEqual((r['lock_driver_version'], r['observed_driver_version'], r['driver_policy']),
                         ('595.84', '580.178.04', 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'))
        self.assertEqual(self.t['environment_recheck']['differences_other_than_driver'], [])

    def test_09_run_root_unchanged(self):
        r = self.doc['run_root_integrity']
        self.assertTrue(r['unchanged_whole_audit'])
        self.assertEqual(r['sha_listing_digest'], LISTING_DIGEST)
        lines = ''.join(x + '\n' for x in self.c['run_root_sha_listing']).encode()
        self.assertEqual(hashlib.sha256(lines).hexdigest(), LISTING_DIGEST)

    def test_10_deterministic_rebuild(self):
        mod = tool()
        doc, text = mod.build(self.c, self.t, ROOT)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, (AUDIT / (PREFIX + 'CHECKPOINT_MANIFEST.txt')).read_text())
        self.assertEqual(mod.report(doc), (AUDIT / (PREFIX + 'COMPLETION.md')).read_text())

    def test_11_evidence_sha_manifest(self):
        names = []
        for line in (AUDIT / (PREFIX + 'EVIDENCE_SHA256.txt')).read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/%s%s' % (PREFIX, x) for x in
                                         ('CHECKPOINT_MANIFEST.txt', 'COMPLETION.json', 'COMPLETION.md')])


if __name__ == '__main__':
    unittest.main()
