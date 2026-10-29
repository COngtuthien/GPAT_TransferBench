"""E04 D12 procedural-reproducibility semantic investigation (protocol fixed at d1af9b6): pins the committed result. The
pre-registered decision is NO_SEMANTIC_DEFECT_FOUND (every criterion passed); the D/D3 forward variation occurs on the
uninstrumented production path; D12 is a draft ready for owner approval, nothing is approved or active, the original
41/42 record and the D10 / D11 records are unchanged and E04 stays INCOMPLETE. The "nothing approved" facts are pinned
to the result commit 57bd1f9 (the owner approved D12 in a later commit). Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
DOC = json.loads((AUDIT / 'E04_D12_INVESTIGATION.json').read_text())
AN = json.loads((AUDIT / 'E04_D12_ANALYSIS.json').read_text())
PROTOCOL = json.loads((AUDIT / 'E04_D12_INVESTIGATION_PROTOCOL.json').read_text())
DRAFT_PATH = 'configs/amendments/e04_d12_procedural_reproducibility.DRAFT.yaml'
RESULT_COMMIT = '57bd1f931010f8e0470fb0beac8209000f12e0f7'
ORIGINAL_SHA256 = 'd158a1ca773cfae2b099a5942b126581efa620a683a10c8494214d4f3da38843'
D10_RECORD_SHA256 = '2f0c0c00a1757711ab02ede4f857fe984928aa92e079a87d4f15910c4ebf76d9'
D11_RECORD_SHA256 = '9305fb825d56db75e940a44254c43171969dd06365df23f895ee83a78fdc0578'


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


class E04D12Investigation(unittest.TestCase):
    def test_01_decision_follows_protocol_rule(self):
        c = AN['criteria']
        self.assertEqual(set(c), {'A_authority', 'completion', 'B_restore_exact', 'B_production_semantics',
                                  'C1_production_forward_integrity', 'C2_production_gradient_integrity',
                                  'C3_confound_forward_integrity', 'D1_confound_reproduced', 'E_graph_and_placement'})
        self.assertTrue(all(c.values()))
        self.assertEqual(AN['decision'], 'NO_SEMANTIC_DEFECT_FOUND')
        self.assertIn('NO_SEMANTIC_DEFECT_FOUND', PROTOCOL['decision_rule'])
        self.assertEqual((DOC['decision'], DOC['criteria'], DOC['failed_criteria']), (AN['decision'], c, []))
        self.assertFalse(DOC['findings']['semantic_defect_found'])

    def test_02_ready_for_approval_but_nothing_active(self):
        self.assertEqual(DOC['d12_status'], 'E04_D12_READY_FOR_OWNER_APPROVAL')
        self.assertEqual(DOC['status'], 'E04_PRODUCTION_QUALIFICATION_INCOMPLETE')
        self.assertFalse(DOC['d12_approved'])
        self.assertFalse(DOC['d12_gate_active'])
        self.assertFalse(DOC['requalification_run'])
        self.assertIsNone(DOC['e04_scientific_authority'])
        self.assertFalse(DOC['creates_scientific_authority'])
        self.assertEqual(DOC['scientific_seeds_launched'], [])
        self.assertEqual(DOC['d12_draft_file'], DRAFT_PATH)
        self.assertEqual(DOC['d12_draft_sha256'], sha(DRAFT_PATH))
        tree = subprocess.run(['git', 'ls-tree', '-r', '--name-only', RESULT_COMMIT], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.split()
        self.assertEqual([p for p in tree if p.startswith('configs/') and 'd12' in p], [DRAFT_PATH])
        self.assertNotIn('outputs/audit/E04_PRODUCTION_QUALIFICATION_D12.json', tree)
        self.assertNotIn('docs/spec/amendments/GPAT_TransferBench_v1_0_E04_D12_Procedural_Reproducibility_Addendum.md',
                         tree)

    def test_03_prior_evidence_preserved(self):
        self.assertEqual(sha('outputs/audit/E04_PRODUCTION_QUALIFICATION.json'), ORIGINAL_SHA256)
        self.assertEqual(sha('outputs/audit/E04_D10_INVESTIGATION.json'), D10_RECORD_SHA256)
        self.assertEqual(sha('outputs/audit/E04_D11_INVESTIGATION.json'), D11_RECORD_SHA256)
        a = DOC['authority_preserved']
        self.assertEqual((a['original_failed_qualification']['record_sha256'], a['d10']['record_sha256'],
                          a['d11']['record_sha256']), (ORIGINAL_SHA256, D10_RECORD_SHA256, D11_RECORD_SHA256))
        self.assertTrue(all(a[k]['raw_reverified'] for k in ('original_failed_qualification', 'd10', 'd11')))
        q = json.loads((AUDIT / 'E04_PRODUCTION_QUALIFICATION.json').read_text())
        self.assertEqual((q['status'], q['failed_gates']), ('E04_PRODUCTION_QUALIFICATION_INCOMPLETE',
                                                            ['K.real_data_gradient_repeatability']))
        d = a['d10_gate_draft']
        self.assertEqual((d['committed'], d['activated'], d['used']), (False, False, False))

    def test_04_protocol_and_execution_identity(self):
        p = DOC['protocol']
        self.assertEqual(p['sha256'], sha('outputs/audit/E04_D12_INVESTIGATION_PROTOCOL.json'))
        self.assertEqual((p['analysis_protocol_sha256'], AN['protocol_sha256']), (p['sha256'], p['sha256']))
        e = DOC['execution']
        self.assertEqual(e['gpu_checkout_commit'], p['fixed_at_commit'])
        self.assertTrue(e['gpu_checkout_clean'])
        self.assertTrue(e['tool_sha256_recorded_equals_commit'])
        self.assertEqual(len(e['processes']), 19)
        self.assertEqual(set(e['process_exit'].values()), {0})
        self.assertEqual(len(e['process_exit']), 21)
        for v in e['processes'].values():
            self.assertEqual(v['firewall'], {'denied': [], 'runtime_writes': []})
            self.assertEqual(v['protocol_sha256'], p['sha256'])
        self.assertEqual({v['initial_state_sha256'] for v in e['processes'].values()},
                         {'951dad42fd2e2c6d09622bbc12dd50fa7094d4c611db45467fe996beeb107df0'})
        auth = json.loads((AUDIT / 'E04_D12_AUTHORITY.json').read_text())
        self.assertEqual(auth['identities_sha256'], 'cc77c431600f126af9c0877a0f6590dee816cc1dc09e5c175e40ec0298428291')
        self.assertTrue(auth['construction_equals_reference'] and auth['prepare_identities_equal'])

    def test_05_confound_resolved_on_plain_path(self):
        f = DOC['findings']['instrumentation_confound']
        by = f['processes_with_forward_variation_by_factor']
        self.assertEqual(by, AN['detail']['confound_processes_with_forward_variation'])
        self.assertEqual(by['plain'], 4)
        for k, v in f['per_process'].items():
            if k.startswith('plain/'):
                self.assertEqual(v['traced_calls_before_localization'], 0, k)
            self.assertLessEqual(v['forward_max_relative_l2'], 1e-3, k)
        self.assertTrue(all(e['plain_place_is_requested_device'] for k, e in DOC['execution']['processes'].items()
                            if k.startswith('plain/')))
        self.assertEqual(f['forward_origin']['tensor'], 'T_A:0')
        self.assertFalse(f['forward_origin']['kernel_mechanism_claimed'])
        self.assertEqual(f['scope']['subnetworks_with_varying_forward_tensors'], ['D/D3'])
        self.assertEqual((f['scope']['violations_of_integrity_bound'], f['scope']['varying_fed_inputs'],
                          f['scope']['varying_data_movement_roots']), (0, 0, 0))
        self.assertTrue(any(AN['detail']['production_forward_varying'].values()))

    def test_06_production_semantics(self):
        s = DOC['findings']['production_path_semantics']
        for k in ('adam', 'beta_powers', 'connectivity', 'counters', 'diagnostics_non_mutating', 'finite',
                  'graph_unchanged_by_session', 'no_random_ops_on_step_paths', 'optimizer_application',
                  'proxy_count_equals_d10_algorithm', 'restore_exact', 'scope'):
            self.assertIs(s[k], True, k)
        self.assertEqual(s['semantic_replays'], 54)
        g = DOC['findings']['production_gradient_integrity']
        self.assertTrue(all(r['ratio'] <= g['bound'] for r in g['rows']))
        self.assertEqual(len(g['rows']), 27)
        self.assertEqual(g['max_functional_ratio_by_step']['step3'], 0.0)
        self.assertTrue(all(x['variables_differing'] == 0 for x in DOC['findings']['production_step_replay']['step3']))
        self.assertTrue(DOC['findings']['cross_process']['inputs_and_pair_rows_identical'])
        e = DOC['findings']['graph_and_placement']
        self.assertEqual(e['checks'], {'canonical_partitions_equal_d11_original': True, 'placement_consistent': True})
        self.assertEqual(len(e['canonical']), 8)

    def test_07_draft_gate_validity_check_is_not_qualification(self):
        v = DOC['draft_gate_validity_check']
        self.assertTrue(v['pass'])
        self.assertIn('not qualification evidence', v['note'])
        self.assertEqual(v['superseded_gate'], {'gate': 'K.real_data_gradient_repeatability', 'recorded_pass': False,
                                                'counted_as_passed': False})
        draft = json.loads((ROOT / DRAFT_PATH).read_text())
        self.assertEqual(v['criteria_sha256'], draft['acceptance_criteria_sha256'])
        self.assertEqual(DOC['acceptance_criteria_sha256'], draft['acceptance_criteria_sha256'])

    def test_08_test_isolation(self):
        t = DOC['test_isolation']
        self.assertEqual((t['test_files_accessed'], t['val_files_accessed']), (0, 0))
        self.assertTrue(t['manifests_mtime_unchanged'])
        self.assertFalse(t['e04_run_root_exists'])
        self.assertTrue(t['atime']['same_paths'])
        self.assertEqual(t['atime']['pre_sha256'], t['atime']['post_sha256'])
        for s in ('TEST', 'VAL'):
            self.assertEqual((t['atime']['by_split'][s]['atime_changed'], t['atime']['by_split'][s]['mtime_changed']),
                             (0, 0))

    def test_09_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_D12_INVESTIGATION_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            at_result = subprocess.run(['git', 'show', '%s:%s' % (RESULT_COMMIT, name)], cwd=ROOT,
                                       capture_output=True, check=True).stdout
            self.assertEqual(hashlib.sha256(at_result).hexdigest(), digest, name)
            if name != 'tools/e04_d12_procedural_gate.py':      # activated after the owner approval (c60481b)
                self.assertEqual(sha(name), digest, name)
        self.assertEqual(len(names), 13)
        self.assertEqual(DOC['analyses']['analysis']['sha256'], sha('outputs/audit/E04_D12_ANALYSIS.json'))
        self.assertEqual(DOC['raw_outputs']['sha256sums_sha256'], sha('outputs/audit/E04_D12_RAW_SHA256SUMS.txt'))
        raw = (AUDIT / 'E04_D12_RAW_SHA256SUMS.txt').read_text().splitlines()
        self.assertEqual(len(raw), DOC['raw_outputs']['files'])
        self.assertIn('%s  ./analysis.json' % sha('outputs/audit/E04_D12_ANALYSIS.json'), raw)
        self.assertIn('%s  ./authority.json' % sha('outputs/audit/E04_D12_AUTHORITY.json'), raw)


if __name__ == '__main__':
    unittest.main()
