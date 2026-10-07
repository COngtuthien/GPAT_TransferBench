"""M7D1-N2: evidence checks for the GPAT-B0 / E08 / seed 42 saturation / generator-drift diagnosis (DIAGNOSTIC_ONLY).

Stdlib only, any host: validates the committed collected document, the diagnosis JSON and the per-epoch CSV, and
re-derives both outputs from the collected document.
"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
DIAG = AUDIT / 'M7D1_E08_SEED42_SATURATION_DIAGNOSIS.json'
CSV = AUDIT / 'M7D1_E08_SEED42_TRAIN_TRENDS.csv'
COLLECTED = AUDIT / 'M7D1_E08_SEED42_SATURATION_COLLECTED.json'
CLOSE = AUDIT / 'M7D1_E08_SEED42_COMPLETION.json'


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n2', ROOT / 'tools' / 'm7d1_n2_saturation_diagnosis.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticM7D1N2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(DIAG.read_text())
        cls.c = json.loads(COLLECTED.read_text())
        cls.t = tool()

    def test_01_identity_and_classification(self):
        d = self.doc
        self.assertEqual((d['method'], d['experiment'], d['seed'], d['run_id']), ('GPAT-B0', 'E08', 42, '7b799fbd6d0426da'))
        self.assertEqual(d['classification_kind'], 'DESCRIPTIVE_DIAGNOSIS_NOT_A_GATE')
        self.assertIn(d['status'], self.t.CLASSES)
        self.assertEqual(d['status'], self.t.classify(self.c)[0])

    def test_02_trends_cover_60_epochs_and_csv(self):
        tr = self.doc['B_metric_trends']['per_epoch']
        self.assertEqual(sorted(int(e) for e in tr), list(range(1, 61)))
        for e, v in tr.items():
            self.assertEqual(set(v), set(self.t.METRIC_KEYS))
            for k, s in v.items():
                self.assertLessEqual(s['p10'], s['median'])
                self.assertLessEqual(s['median'], s['p90'])
            self.assertEqual(v['D_total']['n'], 1105)
        rows = list(csv.reader(io.StringIO(CSV.read_text())))
        self.assertEqual(len(rows), 61)
        self.assertEqual(len(rows[0]), 1 + 3 * len(self.t.METRIC_KEYS))

    def test_03_candidate_inputs_match_closure_manifest(self):
        close = {r['epoch']: r['sha256'] for r in json.loads(CLOSE.read_text())['ema_candidates']['candidates']}
        got = {r['epoch']: r['sha256'] for r in self.doc['input_candidates']}
        self.assertEqual(got, close)
        self.assertEqual(sorted(got), list(range(10, 61)))

    def test_04_parameter_drift_complete(self):
        cons = self.doc['C_parameter_drift']['consecutive']
        self.assertEqual(sorted(cons), sorted(f'{e - 1}->{e}' for e in range(11, 61)))
        dec = self.doc['C_parameter_drift']['decade_and_long_range']
        self.assertEqual(sorted(dec), sorted(['10->20', '20->30', '30->40', '40->50', '50->60', '30->60', '10->60']))
        for comp in list(cons.values()) + list(dec.values()):
            for mod in ('g_res', 'e_art', 'combined'):
                m = comp[mod]['all_floating']
                for k in ('abs_l2', 'rel_l2', 'max_abs', 'frac_bitwise_changed', 'frac_abs_gt_1e-08',
                          'frac_abs_gt_1e-07', 'frac_abs_gt_1e-06', 'state_equal'):
                    self.assertIn(k, m)
                self.assertGreaterEqual(m['frac_bitwise_changed'], m['frac_abs_gt_1e-08'])

    def test_05_output_drift_train_only(self):
        o = self.doc['D_output_drift']
        self.assertEqual(len(o['subset']), 32)
        self.assertEqual(sorted(o['per_epoch'], key=int), ['10', '20', '30', '40', '50', '60'])
        self.assertIn('30->60', o['drift'])
        f = self.doc['firewall']
        acc = f['access_log']
        for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata', 'test_metadata'):
            self.assertEqual(acc[k], 0, k)
        self.assertEqual(acc['train_images'], 64)
        self.assertEqual(f['faces_files_opened'], 64)
        self.assertTrue(f['faces_files_opened_all_in_subset'])
        self.assertEqual(f['suspicious_val_test_paths_opened'], [])
        self.assertEqual(f['manifests_opened'], ['manifests/pairs_train_v1.parquet'])

    def test_06_no_mutation_no_selection(self):
        self.assertTrue(self.doc['run_root_unchanged'])
        self.assertFalse(any(self.doc['confirmations'].values()))

    def test_08_identity_collapse_finding(self):
        f = self.doc['primary_finding']
        self.assertEqual(f['name'], 'GENERATOR_IDENTITY_COLLAPSE')
        ev = f['training_metric_evidence']
        self.assertEqual((ev['onset_update'], ev['onset_epoch']), (5781, 6))
        self.assertEqual(ev['updates_after_onset'], 66300 - 5781 + 1)
        self.assertLess(ev['after_onset_max']['bg'], 1e-6)
        for e, v in f['candidate_output_evidence'].items():
            self.assertEqual(v['u8_x_hat_equals_u8_target_fraction'], 1.0, e)
        self.assertTrue(f['owner_decision_required'])
        self.assertEqual(self.doc['classification_basis']['rule_v1_result'], 'INCONCLUSIVE_SATURATION_DIAGNOSIS')

    def test_07_rebuild_reproduces(self):
        doc, text = self.t.build(self.c, self.doc['authority'])
        self.assertEqual(text, CSV.read_text())
        doc.pop('audit_timestamp_utc')
        mine = dict(self.doc)
        mine.pop('audit_timestamp_utc')
        self.assertEqual(json.loads(json.dumps(doc)), mine)


if __name__ == '__main__':
    unittest.main()
