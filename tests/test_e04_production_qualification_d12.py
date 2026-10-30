"""E04 production requalification under the approved D12 (qualification seed 60904): pins the committed NEW record.
52/52 gates (41 retained + R1-R10 + evidence) pass; K.real_data_gradient_repeatability was recomputed, FAILS, and is
recorded as superseded, non-gating and never counted; the original 41/42 record is unchanged; the runs are fresh; no
scientific seed was launched or authorized. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
sys.path.insert(0, str(ROOT / 'tools'))

import e04_d12_procedural_gate as g  # noqa: E402

DOC = json.loads((AUDIT / 'E04_PRODUCTION_QUALIFICATION_D12.json').read_text())
ORIGINAL = json.loads((AUDIT / 'E04_PRODUCTION_QUALIFICATION.json').read_text())


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


class E04ProductionQualificationD12(unittest.TestCase):
    def test_01_status(self):
        self.assertEqual(DOC['status'], 'E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION')
        self.assertEqual((DOC['gates_passed'], DOC['gates_total'], DOC['failed_gates']), (52, 52, []))
        self.assertTrue(all(x['pass'] for x in DOC['gates']))
        self.assertTrue(DOC['production_qualification_authority'])
        self.assertFalse(DOC['scientific_seed_execution_authorized'])
        self.assertEqual(DOC['scientific_seeds_launched'], [])
        self.assertIsNone(DOC['e04_scientific_authority'])
        self.assertFalse(any(DOC['confirmations'].values()))

    def test_02_gate_set(self):
        names = [x['gate'] for x in DOC['gates']]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(n for n in names if not n.startswith('D12.')), sorted(g.RETAINED_GATES))
        self.assertEqual(sorted(n for n in names if n.startswith('D12.R')), sorted('D12.' + k for k in g.CRITERIA))
        self.assertIn('D12.evidence_complete_and_hashed', names)
        self.assertNotIn(g.SUPERSEDED_GATE, names)
        d = DOC['d12']
        self.assertEqual(d['acceptance_criteria_sha256'], g.CRITERIA_SHA256)
        self.assertEqual(d['amendment_status'], 'OWNER_APPROVED')
        self.assertTrue(all(d['criteria'].values()))
        self.assertEqual(d['failures'], {})

    def test_03_superseded_gate_recorded_not_counted(self):
        self.assertEqual([(k['gate'], k['pass'], k['gating'], k['counted_as_passed']) for k in DOC['superseded_gates']],
                         [(g.SUPERSEDED_GATE, False, False, False)])
        self.assertEqual(DOC['gradient_determinism']['non_repeatable_losses'], ['L_G'])
        self.assertEqual(DOC['d12']['superseded_gate']['counted_as_passed'], False)
        o = DOC['original_record']
        self.assertEqual(o['sha256'], sha('outputs/audit/E04_PRODUCTION_QUALIFICATION.json'))
        self.assertEqual(o['sha256'], 'd158a1ca773cfae2b099a5942b126581efa620a683a10c8494214d4f3da38843')
        self.assertEqual((ORIGINAL['status'], ORIGINAL['failed_gates'], ORIGINAL['gates_passed']),
                         ('E04_PRODUCTION_QUALIFICATION_INCOMPLETE', [g.SUPERSEDED_GATE], 41))

    def test_04_fresh_evidence(self):
        e = DOC['evidence']
        self.assertTrue(e['pass'])
        self.assertEqual(e['problems'], [])
        self.assertEqual(set(e['run_ids'].values()), {'0f86725250f57541'})
        self.assertNotIn('caf16974d17fe5b7', {c['run_id'] for c in DOC['cases'].values()})
        self.assertEqual(e['gpu_checkout_commit'], 'c676459a3d4b1ee64f5348bf84d44b72a36fb685')
        self.assertEqual(e['raw_files'], 82)
        raw = (AUDIT / 'E04_PRODUCTION_QUALIFICATION_D12_RAW_SHA256SUMS.txt').read_text().splitlines()
        self.assertEqual(len(raw), 82)
        self.assertEqual(sha('outputs/audit/E04_PRODUCTION_QUALIFICATION_D12_RAW_SHA256SUMS.txt'),
                         '3742793b1b9d5718f7ea6996752ab06535325b4ad7396babc5a4f5f60c415ba5')

    def test_05_identities_and_isolation(self):
        sys.path.insert(0, str(ROOT))
        from methods.physics_std import production_io as pio
        self.assertEqual(DOC['prepass']['manifest_sha256'], pio.PREPASS_MANIFEST_SHA256)
        self.assertEqual(DOC['feed_pack']['manifest_sha256'], pio.PACK_MANIFEST_SHA256)
        self.assertEqual(DOC['graph']['construction'], ['61f1ed5868d6774abb809915c2e9dfd0a50c5313064672389033ca46dd951e1d',
                                                        11315])
        self.assertTrue(DOC['prepass']['compare']['identical'])
        for a in (DOC['atime'], DOC['atime_after_diagnostics']):
            self.assertEqual(a['prefix'], 'e04_d12q')
            self.assertEqual(a['by_split']['TEST']['atime_changed'], 0)
            self.assertTrue(all(v['mtime_changed'] == 0 for v in a['by_split'].values()))

    def test_06_sha_manifest(self):
        lines = (AUDIT / 'E04_PRODUCTION_QUALIFICATION_D12_SHA256.txt').read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            digest, name = line.split('  ')
            self.assertEqual(sha(name), digest, name)


if __name__ == '__main__':
    unittest.main()
