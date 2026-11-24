"""E04 three-seed family closure: the committed closure re-derives byte-identically from the committed per-seed evidence
and the embedded host recheck, and the gates fail closed on tampered inputs. Stdlib only; needs the git history."""
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


def tool():
    spec = importlib.util.spec_from_file_location('e04_family', ROOT / 'tools' / 'e04_three_seed_training_closure.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class E04ThreeSeedClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = tool()
        cls.doc = json.loads((AUDIT / 'E04_THREE_SEED_TRAINING_CLOSURE.json').read_text())
        cls.host = cls.doc['host_recheck']

    def test_01_status_and_totals(self):
        d = self.doc
        self.assertEqual((d['status'], d['failed_gates']), ('SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS', []))
        self.assertEqual(d['gates_passed'], d['gates_total'])
        self.assertEqual(d['totals'], {'iterations': 450000, 'g_applications': 900000, 'd_applications': 450000,
                                       'checkpoint_events': 30, 'checkpoint_files': 90})
        self.assertEqual(d['family']['run_ids'], {'42': '40befa6004aee85c', '1337': 'a42fdd2a2accaf0a',
                                                  '2026': 'e7100eb9a5ab0537'})
        self.assertEqual(d['scientific_training_authority'], '18ba00e94813f005876c8d6235b0b848226b2c1d')

    def test_02_deterministic_rebuild(self):
        doc, text, inv, repro = self.mod.build(ROOT, self.host)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, (AUDIT / 'E04_THREE_SEED_FINAL_CHECKPOINTS.csv').read_text())
        self.assertEqual(inv, (AUDIT / 'E04_THREE_SEED_CHECKPOINT_INVENTORY.csv').read_text())
        self.assertEqual(json.dumps(repro, indent=2, sort_keys=True) + '\n',
                         (AUDIT / 'E04_THREE_SEED_REPRODUCTION_MANIFEST.json').read_text())
        self.assertEqual(self.mod.report(doc), (AUDIT / 'E04_THREE_SEED_TRAINING_CLOSURE.md').read_text())

    def test_03_evidence_sha(self):
        for line in (AUDIT / 'E04_THREE_SEED_EVIDENCE_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_04_inventory(self):
        rows = list(csv.DictReader(io.StringIO((AUDIT / 'E04_THREE_SEED_CHECKPOINT_INVENTORY.csv').read_text())))
        self.assertEqual(len(rows), 90)
        finals = {r['seed']: r['sha256'] for r in rows if r['final_selected'] == 'True' and r['file'].endswith('.data-00000-of-00001')}
        self.assertEqual(finals, {'42': '55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6',
                                  '1337': 'fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5',
                                  '2026': '089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2'})

    def failed(self, host):
        return self.mod.build(ROOT, host)[0]['failed_gates']

    def test_05_tampered_host_checkpoint_fails(self):
        h = copy.deepcopy(self.host)
        f = h['seeds']['1337']['checkpoint_files']['checkpoints/iteration_150000.data-00000-of-00001']
        f['sha256'] = '0' * 64
        self.assertIn('H.seed1337.host_matches_evidence', self.failed(h))

    def test_06_running_method_process_fails(self):
        h = copy.deepcopy(self.host)
        h['host_state']['method_processes'] = ['1 python tools/run_e04.py --seed 2026']
        self.assertIn('H.host_final_state', self.failed(h))

    def test_07_live_pane_or_moved_checkout_fails(self):
        h = copy.deepcopy(self.host)
        h['host_state']['e04_tmux_panes']['e04_physics_std_seed2026'] = '0'
        self.assertIn('H.host_final_state', self.failed(h))
        h = copy.deepcopy(self.host)
        h['host_state']['git_head'] = '0' * 40
        self.assertIn('H.gpu_checkout_at_authority_clean', self.failed(h))

    def test_08_extra_seed_dir_or_resolved_config_change_fails(self):
        h = copy.deepcopy(self.host)
        h['parent_listing'].append('seed_7')
        self.assertIn('H.host_parent_exactly_three_seeds', self.failed(h))
        h = copy.deepcopy(self.host)
        y = h['seeds']['2026']['resolved_config_yaml'].replace('150000', '150001', 1)
        h['seeds']['2026'].update(resolved_config_yaml=y, resolved_config_sha256=hashlib.sha256(y.encode()).hexdigest())
        self.assertIn('C.resolved_config_differs_only_by_seed', self.failed(h))

    def test_09_nonzero_exit_marker_fails(self):
        h = copy.deepcopy(self.host)
        h['seeds']['42']['exit_marker'] = '1\n'
        self.assertIn('H.seed42.host_matches_evidence', self.failed(h))


if __name__ == '__main__':
    unittest.main()
