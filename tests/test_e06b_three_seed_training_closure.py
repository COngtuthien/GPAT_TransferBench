"""E06b three-seed closure: evidence checks for the completed E06b DSDG-NATIVE scientific training family
(successful seeds 42, 1337, 2026; training authority d9ecbef, FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY).

Stdlib only (plus git for the closure-commit byte check). The run roots live on the GPU host; these tests validate
the committed family evidence and re-derive it from the committed per-seed evidence and the embedded host recheck.
"""
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
CLOSURE = AUDIT / 'E06B_THREE_SEED_TRAINING_CLOSURE.json'
CSV_FILE = AUDIT / 'E06B_THREE_SEED_FINAL_CHECKPOINTS.csv'
SHA_FILE = AUDIT / 'E06B_THREE_SEED_EVIDENCE_SHA256.txt'
AUTHORITY = 'd9ecbef7295fc6b41ced6a4966b49810d36a8c87'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
FINAL = {42: 'd56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa',
         1337: '645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53',
         2026: 'e8d0d01cfdb59b92d0f24318ab4fec042188bf0a72f9c59054ea10204a326cad'}
RUN_IDS = {42: 'aa96129289d080ba', 1337: '9ffda3ea3f6cc767', 2026: 'ec71778b28479de5'}
ATTEMPT1_RUN_ID = 'd3c0ddd5dc82a4a1'
FINAL_NETG = 'checkpoints/netG_model_epoch_200_iter_0.pth'


def tool():
    spec = importlib.util.spec_from_file_location('e06b_family', ROOT / 'tools' / 'e06b_three_seed_training_closure.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_keys(v)


class StaticE06bThreeSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(CLOSURE.read_text())
        cls.rows = list(csv.DictReader(io.StringIO(CSV_FILE.read_text())))
        cls.seed_docs = {s: json.loads((AUDIT / f'E06B_SEED{s}_COMPLETION.json').read_text()) for s in FINAL}
        cls.t = tool()

    def test_01_exact_seed_set(self):
        self.assertEqual(self.doc['status'], 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertEqual(self.doc['family']['seeds'], [42, 1337, 2026])
        self.assertEqual([int(r['seed']) for r in self.rows], [42, 1337, 2026])
        self.assertEqual((self.doc['family']['dropped_seeds'], self.doc['family']['replacement_seeds']), ([], []))
        self.assertEqual(sorted(p.name for p in AUDIT.glob('E06B_SEED*_COMPLETION.json')),
                         sorted(f'E06B_SEED{s}_COMPLETION.json' for s in FINAL))

    def test_02_three_completed_scientific_runs(self):
        self.assertEqual((self.doc['family']['completed_seeds'], self.doc['family']['scientific_runs']), (3, 3))
        self.assertEqual(len(self.rows), 3)
        for s, d in self.seed_docs.items():
            self.assertEqual(d['status'], f'E06B_SEED{s}_COMPLETED_CLEANLY')
            self.assertEqual(d['collected']['run_manifest']['completion_status'], 'completed')
            self.assertEqual(d['collected']['metrics']['step_records'], 3200)
        self.assertEqual({int(r['seed']): r['run_id'] for r in self.rows}, RUN_IDS)
        self.assertEqual(self.doc['family']['run_ids'], {str(s): v for s, v in RUN_IDS.items()})

    def test_03_seed42_attempt1_excluded(self):
        self.assertNotIn(ATTEMPT1_RUN_ID, {r['run_id'] for r in self.rows})
        self.assertNotIn(ATTEMPT1_RUN_ID, CSV_FILE.read_text())
        x = self.doc['excluded_history']
        self.assertEqual((x['run_id'], x['classification'], x['member'], x['counted'], x['selectable']),
                         (ATTEMPT1_RUN_ID, 'FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG', False, False, False))
        self.assertEqual(self.doc['family']['successful_attempt_per_seed'], {'42': 2, '1337': 1, '2026': 1})
        a1 = self.doc['host_recheck']['attempt1_archive']
        self.assertTrue(a1['exists'])
        self.assertFalse(a1['under_runs'])
        self.assertEqual(a1['sha_listing_digest'], '8ce8d5a8be3ec6179ef4299a51ab7903383bb963307e809d50c7e2dd9d91f6ca')
        self.assertFalse(self.doc['confirmations']['attempt1_counted'])

    def test_04_authority_and_protocol_identity(self):
        self.assertEqual(self.doc['scientific_training_authority'], AUTHORITY)
        for r in self.rows:
            self.assertEqual(r['authority'], AUTHORITY)
        ids = self.seed_docs[42]['collected']['run_start_event']['identities']
        for s, d in self.seed_docs.items():
            m = d['collected']['run_manifest']
            self.assertEqual((m['git_commit'], m['git_dirty'], d['collected']['git']['head']),
                             (AUTHORITY, False, AUTHORITY))
            self.assertEqual(m['config_sha256'], CONFIG_SHA)
            self.assertEqual(d['collected']['run_start_event']['identities'], ids)
            self.assertEqual(d['method']['fidelity'], 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY')
            self.assertEqual(hashlib.sha256(f'E06b|{s}|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16], RUN_IDS[s])
            ld = d['collected']['run_start_event']['loader']
            self.assertEqual((ld['batch_size'], ld['num_workers'], ld['persistent_workers'], ld['shuffle'],
                              ld['drop_last']), (240, 8, False, True, False))
        self.assertEqual(self.doc['frozen']['config_sha256'], CONFIG_SHA)
        self.assertEqual(self.doc['method']['fidelity'], 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY')
        self.assertEqual(self.doc['different_by_design']['manifest_fields'],
                         ['command_line', 'end_utc', 'experiment_seed', 'run_id', 'run_uuid', 'start_utc'])

    def test_05_epochs_steps_checkpoints(self):
        for r in self.rows:
            self.assertEqual((r['completion_status'], r['completed_epoch'], r['global_step']),
                             ('completed', '200', '3200'))
        for d in self.seed_docs.values():
            t, k = d['training'], d['checkpoints']
            self.assertEqual((t['completed_epoch'], t['global_step'], t['step_records'], t['optimizer_applications']),
                             (200, 3200, 3200, 3200))
            self.assertEqual((k['events'], k['official_files'], k['resume_sidecars']), (21, 63, 0))
            self.assertEqual(d['collected']['runner_state_files'], [])

    def test_06_one_selected_epoch200_netG_per_seed(self):
        for s, d in self.seed_docs.items():
            sel = [r for r in d['collected']['official_checkpoints'] if r['selected_for_final']]
            self.assertEqual([(r['path'], r['epoch'], r['checkpoint_type']) for r in sel],
                             [(FINAL_NETG, 200, 'selected')], s)
            self.assertEqual(sel[0]['selection_reason'], 'OFFICIAL_GENERATOR_EPOCH_200')

    def test_07_final_hashes_agree_with_seed_evidence(self):
        for r in self.rows:
            s = int(r['seed'])
            d = self.seed_docs[s]
            self.assertEqual(r['final_netG_path'], FINAL_NETG)
            self.assertEqual(r['final_netG_sha256'], FINAL[s])
            self.assertEqual(r['final_netG_sha256'], d['checkpoints']['final_netG']['sha256'])
            self.assertEqual(r['final_netG_sha256'],
                             d['collected']['run_summary']['final_or_selected_checkpoint_sha256'])
            self.assertEqual(r['run_root_sha_listing_digest'], d['run_root_integrity']['sha_listing_digest'])
            self.assertEqual(r['metrics_sha256'], d['run_root_integrity']['metrics_jsonl_sha256'])
            h = self.doc['host_recheck']['seeds'][str(s)]
            self.assertEqual((h['final_netG_sha256'], h['sha_listing_digest']),
                             (r['final_netG_sha256'], r['run_root_sha_listing_digest']))

    def test_08_no_best_seed(self):
        keys = {k.lower() for k in walk_keys({k: v for k, v in self.doc.items() if k != 'host_recheck'})}
        self.assertFalse({k for k in keys if 'best' in k} - {'best_seed_selection'})
        self.assertIs(self.doc['confirmations']['best_seed_selection'], False)
        self.assertNotIn('best', CSV_FILE.read_text().lower())
        self.assertEqual(self.doc['aggregation']['primary_result_rule'], 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS')

    def test_09_firewall_val_test_siw_zero(self):
        f = self.doc['firewall']
        self.assertEqual((f['val_access'], f['test_access'], f['siw_access']), (0, 0, 0))
        for r in self.rows:
            self.assertEqual((r['val_access'], r['test_access'], r['siw_access']), ('0', '0', '0'))
        for d in self.seed_docs.values():
            self.assertFalse(d['firewall']['test_split_accessed'])
            for cls, v in d['firewall']['atime']['by_class'].items():
                if cls not in ('casia_fasd|TRAIN', 'msu_mfsd|TRAIN'):
                    self.assertEqual(v['atime_changed'], 0, cls)
        self.assertFalse(any(self.doc['confirmations'].values()))
        self.assertEqual(set(self.doc['not_completed'].values()), {'NOT_STARTED', 'NOT_COMPLETED'})

    def test_10_rebuild_reproduces_family_evidence(self):
        doc, text = self.t.build(ROOT, self.doc['host_recheck'])
        self.assertEqual(text, CSV_FILE.read_text())
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)

    def test_11_gates_reject_tampering(self):
        host = copy.deepcopy(self.doc['host_recheck'])
        host['seeds']['2026']['final_netG_sha256'] = '0' * 64
        host['seeds']['1337']['resolved_config_yaml'] += '  learning_rate: 0.001\n'   # a non-seed difference
        host['parent_listing'].append('seed_7')
        host['attempt1_archive']['under_runs'] = True
        doc, _ = self.t.build(ROOT, host)
        self.assertEqual(doc['status'], 'E06B_FAMILY_NEEDS_REVIEW')
        self.assertTrue({'I.seed2026.host_matches_evidence', 'C.resolved_config_differs_only_by_seed',
                         'I.host_parent_exactly_three_seeds', 'H.attempt1_archived_untouched'}
                        <= set(doc['failed_gates']), doc['failed_gates'])

    def test_12_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), sorted(f'outputs/audit/E06B_THREE_SEED_{n}' for n in (
            'TRAINING_CLOSURE.json', 'FINAL_CHECKPOINTS.csv', 'TRAINING_CLOSURE.md')))


if __name__ == '__main__':
    unittest.main()
