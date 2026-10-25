"""E05 three-seed family closure: evidence checks for the E05 PCGAN (controlled architecture resolution) scientific
training family (seeds 42, 1337, 2026; training authority efdf753; CONTROLLED_ADAPTATION).

Stdlib + git only, any host. Validates the committed family evidence (closure JSON with the embedded host recheck,
final-checkpoint CSV, SHA-256 manifest) against the three committed seed closures and re-derives the family JSON and
CSV to prove the build is deterministic.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
CLOSURE = AUDIT / 'E05_THREE_SEED_TRAINING_CLOSURE.json'
CSV_FILE = AUDIT / 'E05_THREE_SEED_FINAL_CHECKPOINTS.csv'
SHA_FILE = AUDIT / 'E05_THREE_SEED_EVIDENCE_SHA256.txt'
AUTHORITY = 'efdf753156045dd2bed52229d9eac869b18d72f8'
CONFIG_SHA = '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'
SEEDS = (42, 1337, 2026)
RUN_IDS = {42: '18463bf5cd471e55', 1337: '036c806bb2f339b7', 2026: 'b739351daf9779bc'}
FINAL_SHA = {42: '4ad80b35e759154f741d249db8c6e43622cab916e13dc785ad9273bd65113cb1',
             1337: 'b6a4cb2c61b1b582f1acd1eb3514f6a15ea49e544e1dac48415004c6ce4f7b34',
             2026: 'b7f0575f1e061936175ac04d2ac333e1008577bdd3654566c009a2f72bf0b166'}
N = 4000


def tool():
    spec = importlib.util.spec_from_file_location('e05_family', ROOT / 'tools' / 'e05_three_seed_training_closure.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE05ThreeSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(CLOSURE.read_text())
        cls.seeds = {s: json.loads((AUDIT / f'E05_SEED{s}_COMPLETION.json').read_text()) for s in SEEDS}
        cls.rows = {int(r['seed']): r for r in csv.DictReader(CSV_FILE.read_text().splitlines())}

    def test_01_exact_seed_set(self):
        self.assertEqual(self.doc['family']['seeds'], list(SEEDS))
        self.assertEqual(sorted(self.rows), sorted(SEEDS))
        self.assertEqual(sorted(p.name for p in AUDIT.glob('E05_SEED*_COMPLETION.json')),
                         sorted(f'E05_SEED{s}_COMPLETION.json' for s in SEEDS))

    def test_02_three_successful_runs(self):
        self.assertEqual(self.doc['status'], 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertEqual((self.doc['family']['completed_seeds'], self.doc['family']['scientific_runs']), (3, 3))
        for s, d in self.seeds.items():
            self.assertEqual(d['status'], f'E05_SEED{s}_COMPLETED_CLEANLY')

    def test_03_authority(self):
        self.assertEqual(self.doc['scientific_training_authority'], AUTHORITY)
        for s, d in self.seeds.items():
            self.assertEqual((d['scientific_authority'], d['collected']['run_manifest']['git_commit'],
                              d['collected']['git']['head']), (AUTHORITY,) * 3)
            self.assertEqual(self.rows[s]['scientific_authority'], AUTHORITY)
        self.assertEqual(self.doc['host_recheck']['host_state']['git_head'], AUTHORITY)

    def test_04_run_ids(self):
        for s in SEEDS:
            rid = hashlib.sha256(f'E05|{s}|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16]
            self.assertEqual((rid, self.seeds[s]['run_id'], self.rows[s]['run_id']), (RUN_IDS[s],) * 3)
        self.assertEqual(self.doc['family']['run_ids'], {str(s): r for s, r in RUN_IDS.items()})

    def test_05_same_protocol_identity(self):
        ids = {json.dumps(d['collected']['run_start']['identities'], sort_keys=True) for d in self.seeds.values()}
        self.assertEqual(len(ids), 1)
        self.assertEqual({d['config_sha256'] for d in self.seeds.values()}, {CONFIG_SHA})
        files = {json.dumps(d['collected']['files_sha256'], sort_keys=True) for d in self.seeds.values()}
        self.assertEqual(len(files), 1)
        g = {x['gate']: x for x in self.doc['gates']}
        for name in ('C.manifest_common_fields_equal', 'C.run_start_equal_except_seed',
                     'C.resolved_config_differs_only_by_seed', 'C.environment_equal', 'C.extensions_equal_no_rebuild'):
            self.assertTrue(g[name]['pass'], name)

    def test_06_08_iterations_and_applications(self):
        final = {'D': N, 'G': N, 'hook_D': N, 'hook_G': N, 'total': 2 * N}
        for d in self.seeds.values():
            self.assertEqual(d['iterations'], N)
            self.assertEqual(d['optimizer_applications'], final)
            self.assertEqual(d['collected']['metrics']['applications_exact'], N)

    def test_09_distinct_train_pairs(self):
        for s, d in self.seeds.items():
            self.assertEqual(d['distinct_train_pairs'], N)
            self.assertEqual(d['collected']['metrics']['order']['match'], N)
            self.assertEqual(self.rows[s]['distinct_train_pairs'], str(N))

    def test_10_13_checkpoints(self):
        for s, d in self.seeds.items():
            self.assertEqual(len(d['collected']['checkpoint_events']), 2)
            sel = [x for x in d['checkpoints']['files'] if x['selected_for_final']]
            self.assertEqual([(x['path'], x['global_step'], x['selection_reason']) for x in sel],
                             [('checkpoints/iteration_4000.pth', N, 'BASELINE_FINAL_STATE_V1')])
            self.assertEqual(self.rows[s]['final_checkpoint_path'], 'checkpoints/iteration_4000.pth')

    def test_14_no_resume_sidecar(self):
        for d in self.seeds.values():
            c = d['collected']
            self.assertEqual((c['runner_state_files'], c['resume_files'], c['partial_files']), ([], [], []))
            self.assertEqual(d['resume'], 'NOT_QUALIFIED_FRESH_ONLY')

    def test_15_no_best_seed(self):
        f = self.doc['family']
        self.assertEqual((f['best_seed'], f['preferred_seed'], f['dropped_seeds'], f['replacement_seeds']),
                         (None, None, [], []))
        self.assertFalse(any(self.doc['confirmations'].values()))
        self.assertEqual(self.doc['aggregation']['primary_result_rule'], 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS')
        self.assertEqual(self.doc['not_completed'], {'synthetic_bank': 'NOT_STARTED',
                                                     'downstream_fas_evaluation': 'NOT_STARTED',
                                                     'whole_benchmark': 'NOT_COMPLETED'})

    def test_16_17_val_test_zero(self):
        self.assertEqual((self.doc['firewall']['val_access'], self.doc['firewall']['test_access']), (0, 0))
        for s, d in self.seeds.items():
            self.assertEqual((d['firewall']['val_access'], d['firewall']['test_access']), (0, 0))
            self.assertEqual((self.rows[s]['VAL_access'], self.rows[s]['TEST_access']), ('0', '0'))

    def test_18_driver_ruling(self):
        r = self.doc['driver_ruling']
        self.assertEqual((r['lock_driver_version'], r['observed_driver_version'], r['driver_policy']),
                         ('595.84', '580.178.04', 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'))
        for s, d in self.seeds.items():
            self.assertEqual(d['driver_ruling']['driver_policy'], r['driver_policy'])
            self.assertEqual(self.rows[s]['observed_driver'], '580.178.04')

    def test_19_final_sha_matches_seed_closures(self):
        for s, d in self.seeds.items():
            self.assertEqual(d['checkpoints']['final_selected']['sha256'], FINAL_SHA[s])
            self.assertEqual(self.rows[s]['final_checkpoint_sha256'], FINAL_SHA[s])
            self.assertEqual(self.rows[s]['metrics_sha256'], d['run_root_integrity']['metrics_jsonl_sha256'])
            self.assertEqual(self.rows[s]['run_root_listing_digest'], d['run_root_integrity']['sha_listing_digest'])
            h = self.doc['host_recheck']['seeds'][str(s)]
            self.assertEqual(h['checkpoints']['checkpoints/iteration_4000.pth']['sha256'], FINAL_SHA[s])

    def test_20_deterministic_rederivation(self):
        doc, text = tool().build(ROOT, self.doc['host_recheck'])
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, CSV_FILE.read_text())

    def test_21_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/E05_THREE_SEED_FINAL_CHECKPOINTS.csv',
                                         'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.json',
                                         'outputs/audit/E05_THREE_SEED_TRAINING_CLOSURE.md'])


if __name__ == '__main__':
    unittest.main()
