"""E03 seed2026 closure: evidence checks for the completed E03 STDN seed 2026 scientific run (run_id 237887eebab18637,
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
SEED = 2026
RUN_ID = '237887eebab18637'
VAL_SEED = 1242808125
FINAL_DATA_SHA = '9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3'
LISTING_DIGEST = 'f42708341b91e6b6d11b601b363e2dd5309dd7b5df33b65c6ed79ca032e25136'
AUTHORITY = 'd547fc9f8932a273d719ae589aade85a9d989b85'
CONFIG_SHA = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
PREFIX = 'E03_SEED%d_' % SEED


def tool():
    spec = importlib.util.spec_from_file_location('e03_seed_audit', ROOT / 'tools' / 'e03_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE03Seed2026(unittest.TestCase):
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
        # the host rebooted after completion: no live pane; exit 0 is proven by the durable reboot evidence (test_12)
        h, s = self.c['host'], self.c['run_summary']
        self.assertEqual((h['exit_file'], h['exit_file_raw'], h['tmux_pane_dead'], h['tmux_pane_dead_status'],
                          h['runner_processes'], h['other_e03_tmux']), ('0', '0\n', None, None, [], []))
        self.assertEqual((s['completion_status'], s['failure_reason']), ('completed', None))
        self.assertEqual(h['pane_log_last'], ['{"status": "completed", "run_dir": "%s"}' % self.doc['run_root']])
        self.assertEqual(set(h['pane_log_fail_hits'].values()), {0})

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

    def test_12_post_completion_reboot_disclosed(self):
        d = self.doc
        gates = ['B.process_exited_zero', 'J.single_e03_session']
        self.assertEqual(d['gates_passed_by_approved_alternative'], gates)
        alt = [g for g in d['gates'] if isinstance(g.get('detail'), dict) and 'result' in g['detail']]
        self.assertEqual([g['gate'] for g in alt], gates)
        for g in alt:
            self.assertTrue(g['pass'])
            self.assertEqual(g['detail']['result'], 'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE')
            self.assertTrue(all(v['pass'] for v in g['detail']['checks'].values()))
        r = d['disclosures']['post_completion_reboot']
        self.assertEqual(sorted(d['disclosures']), ['post_completion_reboot'])       # no keyboard echo this seed
        self.assertEqual((r['run_end_utc'], r['end_time_marker'], r['exit_marker'], r['pane_scope_stopped_utc']),
                         ('2026-10-28T18:52:21Z', '2026-10-28T18:52:21Z', '0', '2026-10-28T18:52:21Z'))
        self.assertEqual((r['run_boot_id'], r['run_boot_last_entry_utc'], r['current_boot_id'], r['kernel_btime_utc']),
                         ('0b3cbc4d97704f239a11c81d3469ba67', '2026-10-29T02:44:03Z',
                          'a74f71de47614d8381b53a3f56741765', '2026-10-29T02:44:20Z'))
        self.assertTrue(r['wtmp_shutdown'].startswith('shutdown system down'))
        self.assertIn('2026-10-29T09:44:06+07:00', r['wtmp_shutdown'])
        self.assertEqual(r['ruling']['ruling'], 'OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION')
        report = (AUDIT / (PREFIX + 'COMPLETION.md')).read_text()
        self.assertNotIn('pane dead with status 0', report)
        self.assertIn('owner-approved post-completion-reboot evidence', report)

    def test_13_prior_needs_review_build_preserved(self):
        hist = AUDIT / 'history' / 'E03_SEED2026_AUDIT1_NEEDS_REVIEW'
        want = {'E03_SEED2026_COMPLETION.json': '62bbc549c838f96be97cfc093fa0b6be66a3a52d0925cbb12858c1583b079ceb',
                'E03_SEED2026_COMPLETION.md': 'f84abc0d17858e11f60d27d35d0afcf116d50807e5d60bd6d4e94578a35e84aa',
                'E03_SEED2026_CHECKPOINT_MANIFEST.txt':
                    '74f4dea62ad74d7fc60f486799261efe58c993ee63a14bccb611240a7b9df7cb',
                'E03_SEED2026_EVIDENCE_SHA256.txt': '900401ce85f3c3c4837da3475a8cf89172de7e33ff2dea8bb885cbff777a6aa1'}
        for name, digest in want.items():
            self.assertEqual(hashlib.sha256((hist / name).read_bytes()).hexdigest(), digest, name)
        old = json.loads((hist / 'E03_SEED2026_COMPLETION.json').read_text())
        self.assertEqual((old['status'], old['gates_passed'], old['failed_gates']),
                         ('E03_SEED2026_NEEDS_REVIEW', 86, ['B.process_exited_zero', 'J.single_e03_session']))
        self.assertEqual(old['collected'], json.loads((hist / 'E03_SEED2026_AUDIT1_collected.json').read_text()))
        self.assertEqual(old['collected_tf'], json.loads((hist / 'E03_SEED2026_AUDIT1_collected_tf.json').read_text()))
        self.assertEqual(self.doc['disclosures']['post_completion_reboot']['prior_diagnostic']['sha256'],
                         want['E03_SEED2026_COMPLETION.json'])
        # the first build still re-derives to 86/88: its collection carries no reboot provenance, so it fails closed
        doc, _ = tool().build(old['collected'], old['collected_tf'], ROOT)
        self.assertEqual((doc['gates_passed'], doc['failed_gates']), (86, old['failed_gates']))


if __name__ == '__main__':
    unittest.main()
