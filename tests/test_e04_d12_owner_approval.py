"""E04 D12 owner approval: the approved amendment is the DRAFT committed at 57bd1f9 with only governance fields changed
(status OWNER_APPROVED, the ruling, rationale, scope and affected gates); the R1-R10 criteria, their hash, the
integrity bound, the retained gates and the superseded-gate rule are unchanged; the approved addendum keeps the
draft's sections 1-6 verbatim; no scientific seed or authority is created and the original 41/42, D10 and D11
records are unchanged. Stdlib only.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import e04_d12_procedural_gate as g  # noqa: E402

DRAFT_COMMIT = '57bd1f931010f8e0470fb0beac8209000f12e0f7'
CRITERIA_SHA256 = '9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d'
APPROVED = 'configs/amendments/e04_d12_procedural_reproducibility.yaml'
DRAFT = 'configs/amendments/e04_d12_procedural_reproducibility.DRAFT.yaml'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_D12_Procedural_Reproducibility_Addendum.md'
DRAFT_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E04_D12_Procedural_Reproducibility_Addendum_DRAFT.md'
GOVERNANCE = {'schema_version', 'status', 'approved', 'approval_date', 'document', 'approval_scope', 'activates_gate',
              'gate_activation_authorized', 'requalification_authorized', 'scientific_seed_execution_authorized',
              'creates_scientific_authority', 'approved_draft', 'owner_ruling', 'rationale',
              'affected_qualification_gates', 'execution_authorized'}
RECORDS = {'outputs/audit/E04_PRODUCTION_QUALIFICATION.json':
           'd158a1ca773cfae2b099a5942b126581efa620a683a10c8494214d4f3da38843',
           'outputs/audit/E04_D10_INVESTIGATION.json': '2f0c0c00a1757711ab02ede4f857fe984928aa92e079a87d4f15910c4ebf76d9',
           'outputs/audit/E04_D11_INVESTIGATION.json': '9305fb825d56db75e940a44254c43171969dd06365df23f895ee83a78fdc0578'}


def at_draft(path):
    return subprocess.run(['git', 'show', '%s:%s' % (DRAFT_COMMIT, path)], cwd=ROOT, capture_output=True,
                          check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


A = json.loads((ROOT / APPROVED).read_text())
D = json.loads(at_draft(DRAFT))


class E04D12OwnerApproval(unittest.TestCase):
    def test_01_approved_governance(self):
        self.assertEqual((A['status'], A['approved']), ('OWNER_APPROVED', True))
        self.assertEqual(A['acceptance_criteria_sha256'], CRITERIA_SHA256)
        self.assertEqual(g.require_approval(ROOT)['acceptance_criteria_sha256'], CRITERIA_SHA256)
        self.assertIs(A['scientific_seed_execution_authorized'], False)
        self.assertIs(A['creates_scientific_authority'], False)
        self.assertIs(A['requalification_authorized'], True)
        self.assertIn('not authorization to revise those criteria', A['owner_ruling'])
        self.assertEqual(A['document'], DOC)

    def test_02_only_governance_fields_changed(self):
        for k in set(A) | set(D):
            if k not in GOVERNANCE:
                self.assertEqual(A.get(k), D.get(k), k)
        self.assertEqual(A['replacement_acceptance_criteria'], g.CRITERIA)
        self.assertEqual(sha(json.dumps(A['replacement_acceptance_criteria'], sort_keys=True).encode()),
                         CRITERIA_SHA256)
        self.assertEqual(A['integrity_bound']['relative_l2'], 1e-3)
        self.assertEqual(A['retained_gates'], list(g.RETAINED_GATES))
        self.assertEqual(A['performance_criteria'], [])
        self.assertIs(A['superseded_gate']['retroactively_passed'], False)
        self.assertIs(A['superseded_gate']['original_record_modified'], False)

    def test_03_approved_draft_pins(self):
        p = A['approved_draft']
        self.assertEqual(p['commit'], DRAFT_COMMIT)
        self.assertEqual(p['companion_sha256'], sha(at_draft(DRAFT)))
        self.assertEqual(p['document_sha256'], sha(at_draft(DRAFT_DOC)))
        self.assertEqual(p['evidence_sha256'], sha(at_draft(p['evidence'])))
        self.assertEqual(p['gate_sha256_at_approved_draft'], sha(at_draft('tools/e04_d12_procedural_gate.py')))
        self.assertEqual((ROOT / DRAFT).read_bytes(), at_draft(DRAFT))
        self.assertEqual((ROOT / DRAFT_DOC).read_bytes(), at_draft(DRAFT_DOC))

    def test_04_addendum_keeps_draft_sections(self):
        draft = at_draft(DRAFT_DOC).decode()
        body = draft[draft.index('## 1. Why'):draft.index('## 7. ')]
        doc = (ROOT / DOC).read_text()
        self.assertIn(body, doc)
        self.assertIn('`OWNER_APPROVED`', doc)
        self.assertIn(CRITERIA_SHA256, doc)

    def test_05_prior_records_unchanged(self):
        for p, h in RECORDS.items():
            self.assertEqual(sha((ROOT / p).read_bytes()), h, p)
        q = json.loads((ROOT / 'outputs/audit/E04_PRODUCTION_QUALIFICATION.json').read_text())
        self.assertEqual((q['status'], q['failed_gates'], q['gates_passed']),
                         ('E04_PRODUCTION_QUALIFICATION_INCOMPLETE', [g.SUPERSEDED_GATE], 41))


if __name__ == '__main__':
    unittest.main()
