"""E04 owner-resolution investigation (DRAFT D1-D9): checks that the package is marked DRAFT / not approved, pins
the resolved non-scientific facts (RNG reference reproduces the committed vectors and the qualified M6D3e hard-sample
fixture; CPU geometry repeatability; D9 BN findings), verifies the SHA-256 list, and re-derives the investigation record
from the committed evidence alone.

Stdlib + numpy; no TensorFlow, no benchmark data.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
RECORD = AUDIT / 'E04_OWNER_RESOLUTION_INVESTIGATION.json'
DRAFT = ROOT / 'configs' / 'amendments' / 'e04_production_resolution.DRAFT.yaml'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class E04OwnerResolutionInvestigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(RECORD.read_text())
        cls.draft = json.loads(DRAFT.read_text())
        cls.vectors = json.loads((AUDIT / 'E04_RNG_TEST_VECTORS.json').read_text())

    def test_01_draft_not_approved(self):
        self.assertEqual(self.doc['status'], 'E04_OWNER_RULING_READY_FOR_FINAL_APPROVAL')
        self.assertTrue(self.doc['draft'])
        self.assertFalse(self.doc['approved'])
        self.assertIsNone(self.doc['e04_scientific_authority'])
        self.assertFalse(self.doc['scientific_execution_authorized'])
        d = self.draft
        self.assertEqual((d['status'], d['approved'], d['creates_scientific_authority'], d['execution_authorized']),
                         ('DRAFT_FOR_OWNER_FINAL_APPROVAL', False, False, False))
        self.assertEqual(sorted(d['decisions']), ['D%d' % i for i in range(1, 10)])
        self.assertTrue(all(v['final_approval_required'] for v in d['decisions'].values()))
        self.assertFalse(any('APPROVED' == v['owner_status'] for v in d['decisions'].values()))
        self.assertTrue(all(self.doc['immutable_inputs_verified'].values()))
        self.assertIn('DRAFT, NOT APPROVED', (ROOT / self.doc['draft_addendum']).read_text().splitlines()[0])

    def test_02_rng_reference_reproduces_vectors(self):
        mod = load('tools/e04_rng_test_vectors.py', 'e04_rng')
        now = mod.vectors()
        strip = lambda d: {k: v for k, v in d.items() if k not in ('numpy', 'python')}  # noqa: E731
        self.assertEqual(strip(now), strip(self.vectors))
        self.assertTrue(now['qualification_realization_check']['equals_m6d3e_fixture_sha256'])
        hist = json.loads((AUDIT / 'M6D3E_E04_SYNTHETIC_PROCESS_1.json').read_text())['fixture_sha256'][3:5]
        q = now['qualification_realization_check']
        self.assertEqual([q['hard_scales_sha256'], q['hard_keep_sha256']], hist)

    def test_03_rng_properties(self):
        for s, v in self.vectors['seeds'].items():
            p = v['pair_stream']
            self.assertEqual((p['rows_drawn'], p['full_passes'], p['tail_rows']), (600000, 67, 7854))
            self.assertTrue(p['every_full_pass_is_a_permutation'])
            self.assertNotEqual(v['pair_seed'], v['hard_seed'])
            self.assertAlmostEqual(v['hard_stream']['keep_rate'], .5, delta=.002)
        self.assertTrue(self.vectors['derived_seeds_distinct'])
        self.assertTrue(self.doc['D4_D7_rng']['identical_across_interpreters'])
        self.assertEqual({v['numpy'] for v in self.doc['D4_D7_rng']['interpreters'].values()}, {'1.23.5', '2.5.3'})

    def test_04_d2_d3_facts(self):
        d2 = self.doc['D2_population']
        self.assertEqual((d2['rows'], d2['unique_target_live'], d2['unique_source_spoof'], d2['live_spoof_overlap']),
                         (8838, 3830, 8838, 0))
        self.assertEqual(d2['feed_pack_landmarks_cover'], {'live': 3830, 'spoof': 8838})
        self.assertEqual(d2['pairs_train_v1_sha256'],
                         'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243')
        d3 = self.doc['D3_cpu_repeatability']
        self.assertTrue(d3['cpu_outputs_byte_identical'])
        self.assertTrue(all(v['cpu_in_committed_m6d3a'] for v in d3['values'].values()))

    def test_05_d9_findings(self):
        d9 = self.doc['D9_bn_investigation']
        for k in ('graph_def_equal_history', 'fixtures_equal_history', 'native_runtime_equal_history',
                  'bn_update_sets_equal_history', 'changed_trainable_sets_equal_history',
                  'losses_before_each_step_equal_history', 'final_losses_equal_history'):
            self.assertTrue(d9[k], k)
        self.assertFalse(d9['fused_batch_norm_reads_moving_statistics'])
        s1 = d9['steps']['step1']
        self.assertEqual(s1['varying_among_current_processes'], [])
        self.assertEqual(s1['current_vs_historical_p1_differing'],
                         ['D/D3/conv%d/BN/moving_mean:0' % i for i in range(1, 5)])
        for r in s1['tensors_vs_history'].values():
            self.assertLessEqual(r['summary_delta_ulps_at_abs_max'], 2)
            self.assertLessEqual(abs(r['relative_delta_l2']), 1e-8)
        s2 = d9['steps']['step2']
        self.assertEqual(s2['current_equal_historical_p2_on_historically_varying'], [47, 47, 47, 47])
        self.assertTrue(all('moving_mean' in n for n in s2['varying_among_current_processes']))

    def test_06_evidence_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_OWNER_RESOLUTION_INVESTIGATION_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(names), 8)
        for f, h in self.doc['draft_files_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / f).read_bytes()).hexdigest(), h, f)

    def test_07_rederives_from_committed_evidence(self):
        mod = load('tools/e04_owner_resolution_investigation.py', 'e04_inv')
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for name, v in self.doc['committed_probe_variants'].items():
                for k in v['probes']:
                    shutil.copyfile(AUDIT / ('E04_D9_BN_PROBE_%s.json' % name), t / ('probe_%d.json' % k))
            for k, iv in self.doc['D4_D7_rng']['interpreters'].items():
                (t / ('rng_%s.json' % k)).write_text(json.dumps(dict(self.vectors, **iv), sort_keys=True))
            for k in (2, 3):
                shutil.copyfile(AUDIT / 'E04_READINESS_REQUAL_M6D3A_CPU.json', t / ('m6d3a_cpu_%d.json' % k))
            doc, _, sha = mod.build(ROOT, json.loads((AUDIT / 'E04_D2_PREPASS_POPULATION.json').read_text()), t)
        doc['committed_probe_variants'] = self.doc['committed_probe_variants']
        self.assertEqual(json.loads(json.dumps(sha)), self.doc['probe_files_sha256'])
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(mod.report(doc), (AUDIT / 'E04_OWNER_RESOLUTION_INVESTIGATION.md').read_text())


if __name__ == '__main__':
    unittest.main()
