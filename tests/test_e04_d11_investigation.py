"""E04 D11 targeted CPU gradient-placement investigation (protocol fixed at ba625b8): pins the committed result. The
pre-registered decision is E04_D11_UNSUPPORTED (S4 forward identity and S5 CPU bitwise reproducibility failed), the
bounded stage did not run, no D11 amendment exists, the original 41/42 record and the D10 record are unchanged and E04
stays INCOMPLETE. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
DOC = json.loads((AUDIT / 'E04_D11_INVESTIGATION.json').read_text())
SHORT = json.loads((AUDIT / 'E04_D11_SHORT_ANALYSIS.json').read_text())
TRACE = json.loads((AUDIT / 'E04_D11_TRACE_ANALYSIS.json').read_text())
ORIGINAL_SHA256 = 'd158a1ca773cfae2b099a5942b126581efa620a683a10c8494214d4f3da38843'
D10_RECORD_SHA256 = '2f0c0c00a1757711ab02ede4f857fe984928aa92e079a87d4f15910c4ebf76d9'


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


class E04D11Investigation(unittest.TestCase):
    def test_01_decision_follows_protocol_rule(self):
        c = SHORT['criteria']
        self.assertEqual(c, {'S1_placement_scope': True, 'S2_gradients_present': True, 'S3_contract_identity': True,
                             'S4_forward_identity': False, 'S5_cpu_bitwise_reproducibility': False, 'S6_finite': True,
                             'S7_deviation_from_control': True, 'S8_completion': True})
        self.assertEqual(SHORT['decision'], 'E04_D11_UNSUPPORTED')
        self.assertEqual((DOC['decision'], DOC['d11_status'], DOC['criteria']), (SHORT['decision'], SHORT['decision'], c))
        self.assertEqual(DOC['failed_criteria'], ['S4_forward_identity', 'S5_cpu_bitwise_reproducibility'])
        self.assertFalse(DOC['l_bounded_run'])
        self.assertIsNone(DOC['analyses']['bounded'])
        self.assertIn('L_bounded NOT_RUN (S criteria)', DOC['execution']['host_log'])

    def test_02_no_authority_no_d11(self):
        self.assertEqual(DOC['status'], 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE')
        self.assertFalse(DOC['d11_approved'])
        self.assertIsNone(DOC['d11_amendment_file'])
        self.assertFalse(DOC['requalification_run'])
        self.assertIsNone(DOC['e04_scientific_authority'])
        self.assertFalse(DOC['creates_scientific_authority'])
        self.assertEqual(DOC['scientific_seeds_launched'], [])
        self.assertFalse(list((ROOT / 'configs').glob('**/*d11*')))
        self.assertFalse((AUDIT / 'E04_PRODUCTION_QUALIFICATION_D11.json').exists())

    def test_03_prior_evidence_preserved(self):
        self.assertEqual(sha('outputs/audit/E04_PRODUCTION_QUALIFICATION.json'), ORIGINAL_SHA256)
        self.assertEqual(sha('outputs/audit/E04_D10_INVESTIGATION.json'), D10_RECORD_SHA256)
        a = DOC['authority_preserved']
        self.assertEqual((a['original_failed_qualification']['record_sha256'], a['d10']['record_sha256']),
                         (ORIGINAL_SHA256, D10_RECORD_SHA256))

    def test_04_protocol_and_execution_identity(self):
        p = DOC['protocol']
        self.assertEqual(p['sha256'], sha('outputs/audit/E04_D11_INVESTIGATION_PROTOCOL.json'))
        self.assertEqual(p['analysis_protocol_sha256'], p['sha256'])
        self.assertEqual(SHORT['protocol_sha256'], p['sha256'])
        e = DOC['execution']
        self.assertEqual(e['gpu_checkout_commit'], p['fixed_at_commit'])
        self.assertTrue(e['gpu_checkout_clean'])
        self.assertTrue(e['tool_sha256_recorded_equals_commit'])
        self.assertEqual(len(e['processes']), 14)
        self.assertEqual(set(e['process_exit'].values()), {0})
        for v in e['processes'].values():
            self.assertEqual(v['firewall']['denied'], [])
            self.assertEqual(v['firewall']['runtime_writes'], [])
            self.assertEqual(v['protocol_sha256'], p['sha256'])
            self.assertEqual(v['design_sha256'], p['design_file_sha256'])
            self.assertEqual(v['bn_cancelled_biases'], 31)
        self.assertEqual(len({v['initial_state_sha256'] for v in e['processes'].values()}), 1)

    def test_05_placement_scope(self):
        f = DOC['findings']['cpu_placement_implementation']
        self.assertEqual(f['graphdef_node_diff_counts'], {'added': 0, 'changed': 34, 'removed': 0})
        self.assertEqual(f['released_colocation'], ['gradients_1/AddN_5', 'gradients_7/AddN_5'])
        self.assertTrue(f['targets_executed_on_cpu_in_both_traces_every_process'])
        self.assertEqual(f['non_target_device_type_differences_against_control'], 0)
        self.assertFalse(f['source_changed'])
        g = DOC['findings']['original_graph_and_placement']
        self.assertEqual(g['targets'], 32)
        self.assertTrue(all(n.endswith('_grad/ScatterNd') and n.split('/')[0] in ('gradients_1', 'gradients_7')
                            for n in g['target_names']))
        self.assertEqual(g['probe_sets_depending_on_targets'], ['L_G', 'step1'])

    def test_06_targets_removed_but_residual_variation(self):
        loc = DOC['findings']['short_probe']['localization_by_process']
        for k, v in loc.items():
            if '/cpu_scatternd/' in k:
                self.assertEqual(v['origins'], 0, k)
                self.assertGreater(v['varying_tensors'], 0, k)
            else:
                self.assertEqual((v['origins'], v['origin_types']), (32, ['ScatterNd']), k)
        rep = DOC['findings']['short_probe']['in_process_repeat_differences_by_probe_iteration']
        for k, v in rep.items():
            if '/cpu_scatternd/' in k:
                self.assertTrue(all('step1' not in s for s in v.values()), k)
                self.assertTrue(any('L_G' in s for s in v.values()), k)

    def test_07_signature_difference_is_duplicate_multiplicity_only(self):
        f = DOC['findings']['signature_difference_explanation']
        self.assertTrue(f['executed_set_equal_between_two_traces_in_every_process'])
        self.assertEqual(TRACE['canonical_set_differences'], [])
        for arm in ('original', 'cpu_scatternd'):
            self.assertEqual(len(f['partition_canonical_set_sha256_by_arm'][arm]), 1)
            self.assertEqual(len(f['retval_canonical_sha256_by_arm'][arm]), 1)
            d = f['duplicate_multiplicity'][arm]
            self.assertTrue(d['min_multiplicity_always_ge_1'])
            self.assertLessEqual({k.split('|')[0] for k in d['by_op_device']},
                                 {'Const', 'Identity', 'RealDiv', 'Reshape', 'Tile'})
        self.assertEqual(len(TRACE['processes']), 14)

    def test_08_test_isolation(self):
        t = DOC['test_isolation']
        self.assertEqual((t['test_files_accessed'], t['val_files_accessed']), (0, 0))
        self.assertTrue(t['manifests_mtime_unchanged'])
        self.assertTrue(t['atime']['same_paths'])
        self.assertEqual(t['atime']['pre_sha256'], t['atime']['post_sha256'])

    def test_09_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_D11_INVESTIGATION_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(sha(name), digest, name)
        self.assertEqual(len(names), 9)
        self.assertEqual(DOC['analyses']['short']['sha256'], sha('outputs/audit/E04_D11_SHORT_ANALYSIS.json'))
        self.assertEqual(DOC['analyses']['trace']['sha256'], sha('outputs/audit/E04_D11_TRACE_ANALYSIS.json'))
        self.assertEqual(DOC['raw_outputs']['sha256sums_sha256'], sha('outputs/audit/E04_D11_RAW_SHA256SUMS.txt'))


if __name__ == '__main__':
    unittest.main()
