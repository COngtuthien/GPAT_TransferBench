"""E04 production qualification evidence (qualification seed 60904): pins the committed record. The qualification is
INCOMPLETE: exactly one gate (real-data adversarial-gradient repeatability) fails closed under D9 condition 6, no
scientific seed was launched and no E04 qualification / scientific authority exists. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
DOC = json.loads((AUDIT / 'E04_PRODUCTION_QUALIFICATION.json').read_text())


class E04ProductionQualificationEvidence(unittest.TestCase):
    def test_01_status_fail_closed(self):
        self.assertEqual(DOC['status'], 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE')
        self.assertEqual(DOC['failed_gates'], ['K.real_data_gradient_repeatability'])
        self.assertEqual((DOC['gates_passed'], DOC['gates_total']), (41, 42))
        self.assertIsNone(DOC['e04_scientific_authority'])
        self.assertEqual(DOC['scientific_seeds_launched'], [])
        self.assertFalse(any(DOC['confirmations'].values()))

    def test_02_gradient_finding(self):
        g = DOC['gradient_determinism']
        self.assertTrue(g['iteration0_identical_state_and_inputs'])
        self.assertEqual(g['non_repeatable_losses'], ['L_G'])
        for k, v in g['per_loss'].items():
            if k != 'L_G':
                self.assertEqual((v['in_process_repeat_differs'], v['cross_process_differs']), (0, 0), k)
        self.assertEqual(g['known_nondeterministic_gpu_ops'], {'ResizeBilinearGrad': 8, 'ScatterNd': 16})
        self.assertLess(g['L_G_rel_l2_median'], 1e-5)

    def test_03_passing_gates(self):
        gates = {x['gate']: x['pass'] for x in DOC['gates']}
        for name in ('B.prepass_repeat_byte_identical', 'C.feed_pack_verified_gpu_and_laptop',
                     'D.det_a_rows_and_hard_draws_equal_replay', 'D.seed42_full_streams_equal_approved_vectors',
                     'E.construction_equals_qualified_m6d3e', 'F.det_a_step_transitions_equal_m6d3e',
                     'H.terminal_probe_selected_semantics', 'H.failure_probe', 'J.atime_test_unchanged',
                     'J.atime_test_unchanged_after_diagnostics', 'J.no_scientific_run_root',
                     'K.d9_compatibility_reprobe_matches_accepted_variant'):
            self.assertTrue(gates[name], name)

    def test_04_identities(self):
        from sys import path
        path.insert(0, str(ROOT))
        from methods.physics_std import production_io as pio
        self.assertEqual(DOC['prepass']['manifest_sha256'], pio.PREPASS_MANIFEST_SHA256)
        self.assertEqual(DOC['feed_pack']['manifest_sha256'], pio.PACK_MANIFEST_SHA256)
        self.assertEqual(DOC['prepass']['counts'], {'samples': 12668, 'live': 3830, 'spoof': 8838, 'VAL': 0, 'TEST': 0})
        self.assertEqual(DOC['graph']['construction'][1], 11315)

    def test_05_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_PRODUCTION_QUALIFICATION_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(names), 3)


if __name__ == '__main__':
    unittest.main()
