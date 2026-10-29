"""E04 D10 numerical-gradient investigation (protocol fixed at a6bb459): pins the committed result. The pre-registered
decision is STOP_AND_REPORT (C1 executed-op-set identity failed; C5 / C6 trajectory criteria failed), D10 is not
recorded, no requalification ran, the original 41/42 record is unchanged and E04 stays INCOMPLETE. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
DOC = json.loads((AUDIT / 'E04_D10_INVESTIGATION.json').read_text())
ANALYSIS = json.loads((AUDIT / 'E04_D10_ANALYSIS.json').read_text())
ORIGINAL_SHA256 = 'd158a1ca773cfae2b099a5942b126581efa620a683a10c8494214d4f3da38843'


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


class E04D10Investigation(unittest.TestCase):
    def test_01_decision_follows_protocol_rule(self):
        c = ANALYSIS['criteria']
        self.assertEqual(ANALYSIS['decision'], 'STOP_AND_REPORT')
        self.assertFalse(c['C1_contract_identity'] and c['C2_finite'] and c['C3_localization'])
        self.assertEqual(c, {'C1_contract_identity': False, 'C2_finite': True, 'C3_localization': True,
                             'C4_fixed_state_magnitude': True, 'C5_trajectory_parameters': False,
                             'C6_trajectory_losses': False})
        self.assertEqual((DOC['decision'], DOC['criteria']), (ANALYSIS['decision'], c))

    def test_02_no_authority_no_d10(self):
        self.assertEqual(DOC['status'], 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE')
        self.assertFalse(DOC['d10_recorded'])
        self.assertIsNone(DOC['d10_amendment_file'])
        self.assertFalse(DOC['requalification_run'])
        self.assertIsNone(DOC['e04_scientific_authority'])
        self.assertFalse(DOC['creates_scientific_authority'])
        self.assertEqual(DOC['scientific_seeds_launched'], [])
        self.assertFalse((ROOT / 'configs' / 'amendments' / 'e04_d10_numerical_reproducibility_resolution.yaml').exists())
        self.assertFalse((AUDIT / 'E04_PRODUCTION_QUALIFICATION_D10.json').exists())

    def test_03_original_failed_qualification_preserved(self):
        self.assertEqual(sha('outputs/audit/E04_PRODUCTION_QUALIFICATION.json'), ORIGINAL_SHA256)
        o = DOC['authority_preserved']['original_failed_qualification']
        self.assertEqual((o['record_sha256'], o['status']), (ORIGINAL_SHA256, 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE'))

    def test_04_protocol_and_execution_identity(self):
        p = DOC['protocol']
        self.assertEqual(p['sha256'], sha('outputs/audit/E04_D10_INVESTIGATION_PROTOCOL.json'))
        self.assertEqual(p['analysis_protocol_sha256'], p['sha256'])
        e = DOC['execution']
        self.assertEqual(e['gpu_checkout_commit'], p['fixed_at_commit'])
        self.assertTrue(e['gpu_checkout_clean'])
        self.assertEqual(e['process_exit'], ['P1 rc=0', 'P2 rc=0', 'P3 rc=0', 'analysis rc=0'])
        for v in e['processes'].values():
            self.assertEqual(v['firewall']['denied'], [])
            self.assertEqual(v['firewall']['runtime_writes'], [])
        self.assertEqual(len({v['initial_state_sha256'] for v in e['processes'].values()}), 1)

    def test_05_c1_failure_is_executed_op_set_only(self):
        f = DOC['findings']['C1_contract_identity']
        self.assertEqual(f['failed_check'], ['executed_signature_set_equal'])
        self.assertTrue(all(v == [] for v in f['per_iteration_first_mismatches'].values()))
        self.assertEqual(sorted(f['executed_op_type_device_counts_differing']),
                         ['Const|GPU', 'Identity|GPU', 'RealDiv|GPU', 'Reshape|GPU', 'Tile|GPU'])

    def test_06_localization(self):
        f = DOC['findings']['C3_localization']
        self.assertEqual(f['origin_types'], ['ScatterNd'])
        self.assertEqual((f['origin_count'], f['undetermined']), (32, []))
        self.assertEqual(f['in_process_repeat_differences_outside_L_G'], [])
        self.assertTrue(all(f['iteration0_non_L_G_gradients_equal_across_processes'].values()))
        self.assertTrue(f['forward_warp_and_resize_on_cpu'])

    def test_07_test_isolation(self):
        t = DOC['test_isolation']
        self.assertEqual((t['test_files_accessed'], t['val_files_accessed']), (0, 0))
        self.assertTrue(t['manifests_mtime_unchanged'])
        self.assertEqual(t['manifests_atime_changed'], 0)
        self.assertTrue(t['atime']['same_paths'])

    def test_08_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_D10_INVESTIGATION_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(sha(name), digest, name)
        self.assertEqual(len(names), 8)
        self.assertEqual(DOC['analysis']['sha256'], sha('outputs/audit/E04_D10_ANALYSIS.json'))
        self.assertEqual(DOC['raw_outputs']['sha256sums_sha256'], sha('outputs/audit/E04_D10_RAW_SHA256SUMS.txt'))


if __name__ == '__main__':
    unittest.main()
