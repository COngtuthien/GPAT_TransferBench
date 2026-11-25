"""E04 bank-contract preparation: evidence, DRAFT ruling, readiness record, E05 reporting decision.

Laptop-only, no TensorFlow, no data: the host probe output is a committed JSON; these tests rebuild every committed
record from committed inputs, check fail-closed behaviour and confirm that nothing authorizes E04 bank inference yet.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import e04_bank_contract_probe as probe  # noqa: E402
import e04_bank_readiness as readiness  # noqa: E402

DRAFT = ROOT / readiness.DRAFT
PROBE_COMMIT = '0728cb1b2be09312b38e3a3e978b641c4ea3852f'


def git_show(rev, rel):
    return subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (rev, rel)], stdout=subprocess.PIPE,
                          check=True).stdout.decode()


class EvidenceRecord(unittest.TestCase):
    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as t:
            doc = probe.build(ROOT, ROOT / probe.COLLECTED, out_root=t)
            self.assertEqual(doc['status'], 'E04_BANK_CONTRACT_EVIDENCE_COLLECTED')
            for ext in ('.json', '.md', '_SHA256.txt'):
                self.assertEqual(Path(t, probe.OUT + ext).read_bytes(), (ROOT / (probe.OUT + ext)).read_bytes(), ext)

    def test_host_collector_is_the_committed_one(self):
        """The host JSON came from clone 0728cb1; the host-side code is unchanged since."""
        c = json.loads((ROOT / probe.COLLECTED).read_text())
        self.assertEqual(c['head'], PROBE_COMMIT)
        self.assertTrue(c['clean'] and c['cpu_only'])

        def host_part(text):
            return text[text.index('# ====================================================================== host'):
                        text.index('# ====================================================================== laptop build')]
        self.assertEqual(host_part(git_show(PROBE_COMMIT, 'tools/e04_bank_contract_probe.py')),
                         host_part((ROOT / 'tools/e04_bank_contract_probe.py').read_text()))

    def test_final_checkpoints_equal_closure_inventory(self):
        ev = json.loads((ROOT / probe.OUT).with_suffix('.json').read_text())
        import csv
        with open(str(ROOT / probe.INVENTORY)) as f:
            rows = [r for r in csv.DictReader(f) if r['iteration'] == '150000']
        self.assertEqual(len(rows), 9)
        for r in rows:
            f = ev['final_checkpoints'][r['seed']]['files'][r['file'].split('/')[-1]]
            self.assertEqual((f['bytes'], f['sha256']), (int(r['bytes']), r['sha256']))

    def _tampered(self, mutate):
        c = json.loads((ROOT / probe.COLLECTED).read_text())
        mutate(c)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'c.json'
            p.write_text(json.dumps(c))
            return probe.build(ROOT, p, out_root=t)

    def test_fail_closed_checkpoint_hash(self):
        def m(c):
            c['checkpoints']['1337']['files']['iteration_150000.index']['sha256'] = '0' * 64
        d = self._tampered(m)
        self.assertIn('C.final_checkpoint_bytes_seed1337', d['failed_gates'])
        self.assertEqual(d['status'], 'E04_BANK_CONTRACT_EVIDENCE_FAILED')

    def test_fail_closed_graph_facts(self):
        def m(c):
            c['graph']['T_A_is_ancestor'] = True
            c['graph']['ancestor_variables_outside_G'] = ['D/D4/conv1/kernel']
            c['checkpoint_variables']['42']['global_iteration'] = 149999
        d = self._tampered(m)
        for g in ('G.synthesized_spoof_independent_of_T_A_AddN', 'G.synthesized_spoof_generator_only',
                  'C.final_state_counters_seed42'):
            self.assertIn(g, d['failed_gates'])


class DraftRuling(unittest.TestCase):
    def setUp(self):
        self.d = json.loads(DRAFT.read_text())

    def test_draft_unchanged_and_approval_binds_it(self):
        import hashlib
        self.assertEqual(self.d['status'], 'DRAFT_PENDING_OWNER_DECISION')
        self.assertIs(self.d['approved'], False)
        p = ROOT / 'configs/amendments/e04_bank_generation_contract.yaml'
        if p.exists():                                      # owner approval (2026-11-25) binds the unchanged DRAFT
            rec = json.loads(p.read_text())
            self.assertEqual(rec['approved_draft']['sha256'], hashlib.sha256(DRAFT.read_bytes()).hexdigest())
            self.assertEqual(rec['decisions'], {q['id']: q['recommended'] for q in self.d['owner_questions']})

    def test_questions_complete(self):
        ids = [q['id'] for q in self.d['owner_questions']]
        self.assertEqual(ids, ['E04-Q1_normalization_statistics', 'E04-Q2_synthesis_variant', 'E04-Q3_float_to_uint8',
                               'E04-Q4_auxiliary_outputs', 'E04-Q5_device_and_reproducibility',
                               'E04-Q6_governance_inclusion'])
        self.assertEqual([q['id'] for q in self.d['owner_questions'] if q['pixel_affecting']],
                         [ids[0], ids[1], ids[2], ids[4]])
        for q in self.d['owner_questions']:
            opts = [o['id'] for o in q['options']]
            self.assertGreaterEqual(len(opts), 2, q['id'])
            self.assertIn(q['recommended'], opts)
            for o in q['options']:
                self.assertTrue(o['rule'] and o['provenance'] and o['expected_effect'], o['id'])

    def test_frozen_inputs_bound(self):
        import hashlib
        for k, v in self.d['frozen_inputs_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / k.split(' (')[0]).read_bytes()).hexdigest(), v, k)

    def test_no_e03_e05_dsdg_ruling_used_as_reason(self):
        for q in self.d['owner_questions']:
            text = json.dumps(q)
            for token in ('E03-Q', 'E05-Q', 'DSDG-Q', 'A_OFFICIAL_TEST_MODE', 'A_SOURCE_IMAGE_WRITER'):
                self.assertNotIn(token, text, q['id'])


class Readiness(unittest.TestCase):
    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as t:
            doc = readiness.build(ROOT, out_root=t)
            self.assertEqual(doc['status'], readiness.STATUS_BLOCKED)
            self.assertEqual(doc['failed_gates'], [])
            for rel in (readiness.DRAFT_DOC, readiness.OUT + '.json', readiness.OUT + '.md',
                        readiness.OUT + '_SHA256.txt'):
                self.assertEqual(Path(t, rel).read_bytes(), (ROOT / rel).read_bytes(), rel)

    def test_fail_closed_when_draft_approved(self):
        with tempfile.TemporaryDirectory() as t:
            for rel in [readiness.DRAFT, readiness.EVIDENCE, readiness.INTEGRITY, readiness.PRIOR_VERIFY,
                        readiness.E05_DECISION, readiness.STORAGE_AUTHORITY, probe.COLLECTED] + \
                    list(readiness.FAMILY_CLOSURES.values()) + \
                    list(json.loads(DRAFT.read_text())['frozen_inputs_sha256']) + \
                    list(json.loads((ROOT / readiness.E05_DECISION).read_text())['evidence_sha256']):
                rel = rel.split(' (')[0]
                Path(t, rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(ROOT / rel), str(Path(t, rel)))
            d = json.loads(DRAFT.read_text())
            d['status'], d['approved'] = 'OWNER_APPROVED', True
            Path(t, readiness.DRAFT).write_text(json.dumps(d))
            doc = readiness.build(t)
            self.assertIn('draft_not_approved', doc['failed_gates'])
            self.assertNotEqual(doc['status'], readiness.STATUS_BLOCKED)

    def test_storage_not_silently_extended(self):
        doc = json.loads((ROOT / (readiness.OUT + '.json')).read_text())
        self.assertFalse(doc['storage']['e04_covered_by_existing_authority'])
        self.assertNotIn('E04', doc['storage']['bank_root_entries'])

    def test_existing_banks_and_closures(self):
        doc = json.loads((ROOT / (readiness.OUT + '.json')).read_text())
        self.assertEqual(len(doc['existing_banks']['learned_banks']), 12)
        self.assertTrue(all(v['png_files'] == 8838 for v in doc['existing_banks']['learned_banks'].values()))
        self.assertEqual(sorted(doc['existing_banks']['family_closures']), ['E01', 'E02', 'E03', 'E05', 'E06b', 'E06c'])
        self.assertEqual(sum(v['outputs'] for v in doc['existing_banks']['family_closures'].values()), 6 * 26514)


class LearnedBankRefusesE04(unittest.TestCase):
    def test_library_and_cli_exclude_e04(self):
        from methods.common import learned_bank as lb
        self.assertNotIn('E04', lb.METHODS)
        storage = json.loads((ROOT / readiness.STORAGE_AUTHORITY).read_text())
        self.assertNotIn('E04', storage['bank_root_registry']['methods'])
        r = subprocess.run([sys.executable, str(ROOT / 'tools/learned_bank.py'), 'gates', '--method', 'E04'],
                           cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(b"invalid choice: 'E04'", r.stderr)


class E05ReportingDecision(unittest.TestCase):
    def test_record(self):
        import hashlib
        d = json.loads((ROOT / readiness.E05_DECISION).read_text())
        self.assertEqual((d['status'], d['kind']), ('OWNER_DECIDED', 'REPORTING_DECISION_ONLY'))
        self.assertIs(d['changes_any_generation_contract'], False)
        self.assertIs(d['changes_any_bank_or_record'], False)
        self.assertIn('Do not retrain, filter, enhance, replace or cherry-pick its outputs.', d['decision'])
        for k, v in d['evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / k).read_bytes()).hexdigest(), v, k)


if __name__ == '__main__':
    unittest.main()
