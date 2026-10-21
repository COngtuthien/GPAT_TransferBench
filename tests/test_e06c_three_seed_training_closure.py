"""E06c three-seed closure: evidence checks for the completed E06c DSDG-BIN-IDFREE scientific training family
(seeds 42, 1337, 2026; launch authority 84809868, CONTROLLED_ADAPTATION, DEV-020).

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
CLOSURE = AUDIT / 'E06C_THREE_SEED_TRAINING_CLOSURE.json'
CSV_FILE = AUDIT / 'E06C_THREE_SEED_FINAL_CHECKPOINTS.csv'
SHA_FILE = AUDIT / 'E06C_THREE_SEED_EVIDENCE_SHA256.txt'
AUTHORITY = '84809868acf472af5ba8cf6ab47165ae038e1012'
CONFIG_SHA = '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002'
TRAIN_REL_SHA = 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243'
FINAL = {42: '665684fddf11ec28b95e2a98122c85bfc5352bb8ea69ecb60caac9418c0871d9',
         1337: '43ffa30dc7eaa568bc0b573a34978f897f46156232ee1a7e71d7f994beb8f336',
         2026: 'dc6c54b4357a51546c3abab3b957dd4b19044ae0188dde417df03446fc951878'}
RUN_IDS = {42: '27dcaf23aac18d7a', 1337: '49a179a4dca77f41', 2026: '86b631403e3c4ee1'}


def tool():
    spec = importlib.util.spec_from_file_location('e06c_family', ROOT / 'tools' / 'e06c_three_seed_training_closure.py')
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


class StaticE06cThreeSeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(CLOSURE.read_text())
        cls.rows = list(csv.DictReader(io.StringIO(CSV_FILE.read_text())))
        cls.seed_docs = {s: json.loads((AUDIT / f'E06C_SEED{s}_COMPLETION.json').read_text()) for s in FINAL}
        cls.t = tool()

    def test_01_exact_seed_set(self):
        self.assertEqual(self.doc['status'], 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertEqual(self.doc['family']['seeds'], [42, 1337, 2026])
        self.assertEqual([int(r['seed']) for r in self.rows], [42, 1337, 2026])
        self.assertEqual((self.doc['family']['dropped_seeds'], self.doc['family']['replacement_seeds']), ([], []))
        self.assertEqual(sorted(p.name for p in AUDIT.glob('E06C_SEED*_COMPLETION.json')),
                         sorted(f'E06C_SEED{s}_COMPLETION.json' for s in FINAL))

    def test_02_three_completed_scientific_runs(self):
        self.assertEqual((self.doc['family']['completed_seeds'], self.doc['family']['scientific_runs']), (3, 3))
        for s, d in self.seed_docs.items():
            self.assertEqual(d['status'], f'E06C_SEED{s}_COMPLETED_CLEANLY')
            m = d['collected']['run_manifest']
            self.assertEqual((m['runner_mode'], m['scientific_run'], m['qualification_only'], m['completion_status']),
                             ('SCIENTIFIC', True, False, 'completed'))
        self.assertEqual({r['run_id'] for r in self.rows}, set(RUN_IDS.values()))

    def test_03_authority_config_relation(self):
        self.assertEqual(self.doc['scientific_training_authority'], AUTHORITY)
        for r in self.rows:
            self.assertEqual(r['launch_authority'], AUTHORITY)
        for s, d in self.seed_docs.items():
            m = d['collected']['run_manifest']
            self.assertEqual((m['git_commit'], m['git_dirty'], d['collected']['git']['head']),
                             (AUTHORITY, False, AUTHORITY))
            self.assertEqual(m['config_sha256'], CONFIG_SHA)
            self.assertEqual(m['identities']['train_relation_sha256'], TRAIN_REL_SHA)
            self.assertEqual(hashlib.sha256(f'E06c|{s}|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16], RUN_IDS[s])
        self.assertEqual((self.doc['frozen']['config_sha256'], self.doc['frozen']['train_relation_sha256']),
                         (CONFIG_SHA, TRAIN_REL_SHA))
        self.assertEqual((self.doc['method']['fidelity_class'], self.doc['method']['deviation']),
                         ('CONTROLLED_ADAPTATION', 'DEV-020'))

    def test_04_epochs_and_steps(self):
        for r in self.rows:
            self.assertEqual((r['completion_status'], r['completed_epoch'], r['global_step']),
                             ('completed', '200', '7400'))
        for d in self.seed_docs.values():
            t = d['training']
            self.assertEqual((t['completed_epoch'], t['global_step'], t['step_records'], t['optimizer_applications']),
                             (200, 7400, 7400, 7400))

    def test_05_one_selected_epoch200_netG_per_seed(self):
        for s, d in self.seed_docs.items():
            sel = [r for r in d['collected']['official_checkpoints'] if r['selected_for_final']]
            self.assertEqual([(r['path'], r['epoch'], r['checkpoint_type']) for r in sel],
                             [('checkpoints/netG_model_epoch_200_iter_0.pth', 200, 'selected')], s)
            self.assertEqual(sel[0]['selection_reason'], 'OFFICIAL_GENERATOR_EPOCH_200')

    def test_06_final_hashes_agree_with_seed_evidence(self):
        for r in self.rows:
            s = int(r['seed'])
            d = self.seed_docs[s]
            self.assertEqual(r['final_checkpoint_sha256'], FINAL[s])
            self.assertEqual(r['final_checkpoint_sha256'], d['checkpoints']['final_netG']['sha256'])
            self.assertEqual(r['final_checkpoint_sha256'],
                             d['collected']['run_summary']['final_or_selected_checkpoint_sha256'])
            self.assertEqual(r['run_root_listing_digest'], d['run_root_integrity']['listing_digest'])
            self.assertEqual(r['metrics_sha256'], d['run_root_integrity']['metrics_jsonl_sha256'])
            h = self.doc['host_recheck']['seeds'][str(s)]
            self.assertEqual((h['final_netG_sha256'], h['listing_digest']),
                             (r['final_checkpoint_sha256'], r['run_root_listing_digest']))

    def test_07_no_best_seed(self):
        keys = {k.lower() for k in walk_keys({k: v for k, v in self.doc.items() if k != 'host_recheck'})}
        self.assertFalse({k for k in keys if 'best' in k} - {'best_seed_selection'})
        self.assertIs(self.doc['confirmations']['best_seed_selection'], False)
        self.assertNotIn('best', CSV_FILE.read_text().lower())
        self.assertEqual(self.doc['aggregation']['primary_result_rule'], 'MEAN_AND_STD_OVER_ALL_THREE_SEEDS')

    def test_08_firewall(self):
        self.assertEqual((self.doc['firewall']['test_benchmark_access'],
                          self.doc['firewall']['val_checkpoint_selection_access']), (0, 0))
        for r in self.rows:
            self.assertEqual((r['test_access'], r['val_selection']), ('0', '0'))
        for d in self.seed_docs.values():
            self.assertFalse(d['firewall']['val_split_accessed'] or d['firewall']['test_split_accessed'])
        self.assertFalse(any(self.doc['confirmations'].values()))
        self.assertEqual(set(self.doc['not_completed'].values()), {'NOT_STARTED', 'NOT_COMPLETED'})

    def test_09_rebuild_reproduces_family_evidence(self):
        doc, text = self.t.build(ROOT, self.doc['host_recheck'])
        self.assertEqual(text, CSV_FILE.read_text())
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)

    def test_10_gates_reject_tampering(self):
        host = copy.deepcopy(self.doc['host_recheck'])
        host['seeds']['2026']['final_netG_sha256'] = '0' * 64
        host['seeds']['1337']['resolved_config_yaml'] += '  learning_rate: 0.001\n'   # a non-seed difference
        host['parent_listing'].append('seed_7')
        doc, _ = self.t.build(ROOT, host)
        self.assertEqual(doc['status'], 'E06C_FAMILY_NEEDS_REVIEW')
        self.assertTrue({'H.seed2026.host_matches_evidence', 'C.resolved_config_differs_only_by_seed',
                         'H.host_parent_exactly_three_seeds'} <= set(doc['failed_gates']), doc['failed_gates'])

    def test_11_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), sorted(f'outputs/audit/E06C_THREE_SEED_{n}' for n in (
            'TRAINING_CLOSURE.json', 'FINAL_CHECKPOINTS.csv', 'TRAINING_CLOSURE.md')))


if __name__ == '__main__':
    unittest.main()
