"""E03 three-seed family closure: evidence checks for the E03 STDN scientific training family (seeds 42, 1337, 2026;
training authority d547fc9; FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY).

Stdlib + git only, any host. Validates the committed family evidence (closure JSON with the embedded host recheck,
final-checkpoint CSV, SHA-256 manifest) against the three committed seed closures and re-derives the family JSON and
CSV to prove the build is deterministic.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
CLOSURE = AUDIT / 'E03_THREE_SEED_TRAINING_CLOSURE.json'
CSV_FILE = AUDIT / 'E03_THREE_SEED_FINAL_CHECKPOINTS.csv'
SHA_FILE = AUDIT / 'E03_THREE_SEED_EVIDENCE_SHA256.txt'
AUTHORITY = 'd547fc9f8932a273d719ae589aade85a9d989b85'
CONFIG_SHA = '08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0'
SEEDS = (42, 1337, 2026)
RUN_IDS = {42: 'a565b76bc7349532', 1337: 'e8c85cc378e778f5', 2026: '237887eebab18637'}
FINAL_DATA_SHA = {42: '5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753',
                  1337: '316c94c066c32314c6819bc105d0043e1ddc7b27329afd48ddcfb8bc8d181551',
                  2026: '9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3'}
CLOSURE_COMMITS = {42: 'b95c127e394d9b2ea644573d317b44405652039e', 1337: '5cdb65db0a75fcdcddb64b1b5af040364a057723',
                   2026: '15bfc84dd4cf1d68427d85d867188b702a847b65'}
REBOOT_GATES = ['B.process_exited_zero', 'J.single_e03_session']


def tool():
    spec = importlib.util.spec_from_file_location('e03_family', ROOT / 'tools' / 'e03_three_seed_training_closure.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE03ThreeSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(CLOSURE.read_text())
        cls.seeds = {s: json.loads((AUDIT / f'E03_SEED{s}_COMPLETION.json').read_text()) for s in SEEDS}
        cls.rows = {int(r['seed']): r for r in csv.DictReader(CSV_FILE.read_text().splitlines())}
        cls.gates = {g['gate']: g for g in cls.doc['gates']}

    def test_01_exact_seed_set(self):
        self.assertEqual(self.doc['family']['seeds'], list(SEEDS))
        self.assertEqual(sorted(self.rows), sorted(SEEDS))
        self.assertEqual(sorted(p.name for p in AUDIT.glob('E03_SEED*_COMPLETION.json')),
                         sorted(f'E03_SEED{s}_COMPLETION.json' for s in SEEDS))

    def test_02_three_successful_runs(self):
        self.assertEqual(self.doc['status'], 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertEqual((self.doc['family']['completed_seeds'], self.doc['family']['scientific_runs']), (3, 3))
        for s, d in self.seeds.items():
            self.assertEqual((d['status'], d['gates_passed'], d['gates_total']),
                             (f'E03_SEED{s}_COMPLETED_CLEANLY', 88, 88))

    def test_03_authority_and_run_ids(self):
        self.assertEqual(self.doc['scientific_training_authority'], AUTHORITY)
        self.assertEqual(self.doc['host_recheck']['host_state']['git_head'], AUTHORITY)
        for s, d in self.seeds.items():
            rid = hashlib.sha256(f'E03|{s}|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16]
            self.assertEqual((rid, d['run_id'], self.rows[s]['run_id']), (RUN_IDS[s],) * 3)
            self.assertEqual((d['scientific_authority'], d['collected']['run_manifest']['git_commit'],
                              d['collected']['git']['head'], self.rows[s]['scientific_authority']), (AUTHORITY,) * 4)
            self.assertEqual(self.rows[s]['seed_closure_commit'], CLOSURE_COMMITS[s])
        self.assertEqual(self.doc['family']['run_ids'], {str(s): r for s, r in RUN_IDS.items()})

    def test_04_counts_and_final_checkpoints(self):
        for s, d in self.seeds.items():
            self.assertEqual(d['counts'], {'epochs': 50, 'train_steps': 100000, 'val_steps': 25000, 'val_blocks': 50,
                                           'g_applications': 100000, 'd_applications': 50000,
                                           'final_global_step': 150000, 'checkpoints': 50, 'checkpoint_files': 150})
            self.assertEqual(d['checkpoints']['final_selected']['sha256']['.data-00000-of-00001'], FINAL_DATA_SHA[s])
            self.assertEqual(self.rows[s]['final_data_sha256'], FINAL_DATA_SHA[s])
            h = self.doc['host_recheck']['seeds'][str(s)]
            self.assertEqual(h['final_checkpoint']['checkpoints/ckpt-50.data-00000-of-00001']['sha256'],
                             FINAL_DATA_SHA[s])
            self.assertEqual(h['sha_listing_digest'], d['run_root_integrity']['sha_listing_digest'])

    def test_05_no_best_seed_test_zero(self):
        f = self.doc['family']
        self.assertEqual((f['best_seed'], f['preferred_seed'], f['dropped_seeds'], f['replacement_seeds']),
                         (None, None, [], []))
        self.assertFalse(any(self.doc['confirmations'].values()))
        self.assertEqual(self.doc['firewall']['test_access'], 0)
        self.assertEqual({r['TEST_access'] for r in self.rows.values()}, {'0'})
        self.assertEqual(self.doc['not_completed'], {'synthetic_bank': 'NOT_STARTED',
                                                     'downstream_fas_evaluation': 'NOT_STARTED',
                                                     'whole_benchmark': 'NOT_COMPLETED'})

    def test_06_both_tool_amendments_disclosed_separately(self):
        a = self.doc['seed_tool_amendments']
        self.assertEqual([(x['id'], x['from'], x['to']) for x in a], [
            ('seed1337_pane_log_keyboard_echo', CLOSURE_COMMITS[42], CLOSURE_COMMITS[1337]),
            ('seed2026_post_completion_reboot_evidence', CLOSURE_COMMITS[1337], CLOSURE_COMMITS[2026])])
        self.assertTrue(all(x['diff_excluding_run_ids'] for x in a))
        self.assertEqual(self.doc['approved_alternative_gates'], {'42': [], '1337': [], '2026': REBOOT_GATES})
        self.assertEqual(sorted(self.doc['per_seed_disclosures']['1337']), ['pane_log_keyboard_echo'])
        self.assertEqual(sorted(self.doc['per_seed_disclosures']['2026']), ['post_completion_reboot'])
        self.assertNotIn('42', self.doc['per_seed_disclosures'])
        for s in SEEDS:
            self.assertTrue(self.gates[f'A.seed{s}.rederives_current_tool']['pass'])
            self.assertTrue(self.gates[f'A.seed{s}.rederives_closure_commit_tool']['pass'])
        self.assertTrue(self.gates['B.seed2026.prior_needs_review_build_preserved']['pass'])

    def test_07_family_tmux_state_by_verified_reboot_only(self):
        st = self.doc['host_recheck']['host_state']
        self.assertEqual(set(st['e03_tmux_panes'].values()), {''})
        self.assertEqual(self.gates['H.host_final_state']['detail']['e03_tmux'],
                         'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE')
        self.assertEqual(st['boot_id'], self.seeds[2026]['disclosures']['post_completion_reboot']['current_boot_id'])
        mod = tool()
        for change in ({'boot_id': '0' * 32}, {'btime': 1793213000}, {'tmux_sessions': ['e03_stdn_seed42']},
                       {'method_processes': ['1 python tools/run_e03.py --seed 2026']}):
            host = copy.deepcopy(self.doc['host_recheck'])
            host['host_state'].update(change)
            doc, _ = mod.build(ROOT, host)
            self.assertEqual(doc['failed_gates'], ['H.host_final_state'], change)

    def test_08_deterministic_rederivation(self):
        doc, text = tool().build(ROOT, self.doc['host_recheck'])
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, CSV_FILE.read_text())
        self.assertEqual(tool().report(doc), (AUDIT / 'E03_THREE_SEED_TRAINING_CLOSURE.md').read_text())

    def test_09_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/E03_THREE_SEED_FINAL_CHECKPOINTS.csv',
                                         'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.json',
                                         'outputs/audit/E03_THREE_SEED_TRAINING_CLOSURE.md'])


if __name__ == '__main__':
    unittest.main()
