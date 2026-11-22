"""M8 learned-method bank authority assessment (E03/E05/E06b/E06c): evidence re-derivation and invariants.

The assessment authorizes nothing. These tests prove that the committed evidence re-derives from the committed host
collection, that the decision package is still an unapproved DRAFT, that the E01/E02 production authority files
(bank_writer.py, ruling, closures) were not touched, and that every bank-contract citation still matches the pinned
official source (skipped when the git-ignored source cache is absent).
"""
import hashlib
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/learned_bank_authority_assessment.py'
OUT = ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.json'
COLLECTED = ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_COLLECTED.json'
SHA_LIST = ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT_SHA256.txt'
DRAFTS = ('configs/amendments/learned_bank_generation_contract.DRAFT.yaml',
          'configs/amendments/learned_bank_storage_location.DRAFT.yaml')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_tool():
    spec = importlib.util.spec_from_file_location('learned_bank_authority_assessment', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLearnedBankAuthorityAssessment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()
        cls.doc = json.loads(OUT.read_text())

    def test_01_sha_list_matches_files(self):
        for line in SHA_LIST.read_text().splitlines():
            digest, rel = line.split('  ', 1)
            self.assertEqual(sha(ROOT / rel), digest, rel)

    def test_02_status_and_gates(self):
        d = self.doc
        self.assertEqual(d['status'], 'LEARNED_BANK_AUTHORITY_ASSESSED_OWNER_DECISIONS_REQUIRED')
        self.assertEqual(d['failed_gates'], [])
        self.assertEqual(d['gates_passed'], d['gates_total'])
        self.assertEqual(d['authorizes'], {'inference': False, 'bank_generation': False, 'gpu_use': False,
                                           'hdd_writes': False})

    def test_03_rederives_from_committed_collection(self):
        cites_available = (ROOT / 'third_party/source_cache/stdn/train.py').is_file()
        if not cites_available:
            self.skipTest('git-ignored third_party/source_cache absent')
        rebuilt = json.loads(json.dumps(self.tool.build(COLLECTED)))
        self.assertEqual(rebuilt, self.doc)
        self.assertEqual(self.tool.render_md(rebuilt), (ROOT / 'outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.md').read_text())

    def test_04_twelve_runs_verified_against_closure_csvs(self):
        table = self.doc['checkpoint_authority']
        self.assertEqual(sorted((r['method'], r['seed']) for r in table),
                         sorted((m, s) for m in ('E03', 'E05', 'E06b', 'E06c') for s in (42, 1337, 2026)))
        self.assertTrue(all(r['verified'] for r in table))
        collected = json.loads(COLLECTED.read_text())
        for r in table:
            got = {k: v['sha256'] for k, v in collected['runs'][f"{r['method']}.seed_{r['seed']}"]['final'].items()}
            want = r['sha256'] if isinstance(r['sha256'], dict) else {'pth': r['sha256']}
            self.assertEqual(got, want)
        self.assertEqual({r['test_access'] for r in table}, {'0'})

    def test_05_every_method_blocked_by_scientific_authority(self):
        self.assertEqual(self.doc['statuses'], {m: 'BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY'
                                                for m in ('E03', 'E05', 'E06b', 'E06c')})

    def test_06_drafts_unapproved_and_no_approved_learned_contract(self):
        for p in DRAFTS:
            d = json.loads((ROOT / p).read_text())
            self.assertIs(d['approved'], False)
            self.assertTrue(d['status'].startswith('DRAFT'))
        q = json.loads((ROOT / DRAFTS[0]).read_text())['questions']
        for key in ('E03-Q1_inference_mode', 'E05-Q1_stylegan_noise', 'DSDG-Q1_latent_stream', 'E06b-Q0_n_syn',
                    'Q4_float_to_uint8', 'C-Q2_device_and_reproducibility', 'C-Q3_s7_bank_stage_acceptance'):
            self.assertIn(key, q)
            self.assertIn(q[key]['recommended'], q[key]['options'])

    def test_07_e01_e02_production_authority_files_unchanged(self):
        auth = json.loads((ROOT / 'configs/amendments/e01_e02_production_authority.yaml').read_text())
        for rel, digest in auth['authority_files'].items():
            self.assertEqual(sha(ROOT / rel), digest, rel)

    def test_08_e01_e02_closure_records_unchanged_since_family_closure(self):
        for commit, files in (('ed05c0b', ('outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.json',
                                           'outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE.md')),
                              ('35ad1fd', ('outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.json',
                                           'outputs/audit/E02_PRODUCTION_FAMILY_CLOSURE.md'))):
            for rel in files:
                blob = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'], capture_output=True, check=True).stdout
                self.assertEqual((ROOT / rel).read_bytes(), blob, rel)

    def test_09_citations_match_pinned_sources(self):
        if not (ROOT / 'third_party/source_cache/stdn/train.py').is_file():
            self.skipTest('git-ignored third_party/source_cache absent')
        cites = self.tool.check_citations()
        self.assertTrue(all(c['ok'] for c in cites), [c['id'] for c in cites if not c['ok']])
        self.assertFalse([c['id'] for c in cites if c['matches_pin'] is False])

    def test_10_e04_snapshot_running_not_audited(self):
        e = self.doc['e04_seed2026']
        self.assertEqual(e['scientific_completion'], 'RUNNING_NOT_COMPLETE')
        self.assertFalse(e['exit_marker_present'])
        self.assertEqual(e['gpu_checkout_head'], '18ba00e94813f005876c8d6235b0b848226b2c1d')
        self.assertLess(e['metrics']['last_global_step'], 150000)

    def test_11_storage_conclusion(self):
        s = self.doc['storage']
        self.assertTrue(s['ext_mounted'])
        self.assertFalse(s['ext_in_runtime_containment_contract'])
        self.assertFalse(s['all_four_families_fit_root_worst_case'])
        self.assertTrue(s['one_family_fits_root_worst_case'])

    def test_12_tool_refuses_overwrite(self):
        with self.assertRaises(SystemExit):
            self.tool.write_outputs(self.doc, COLLECTED)

    def test_13_collect_phase_is_stdlib_and_read_only(self):
        src = TOOL.read_text()
        collect_src = src[src.index('def collect('):src.index('# ---------------------------------------------------------------------------------------------'
                                                               '-- build (laptop)')]
        for forbidden in ('import torch', 'import tensorflow', "open(", 'write_text', 'mkdir', 'unlink', 'rename'):
            if forbidden == "open(":
                self.assertNotIn("'w'", collect_src)
                continue
            self.assertNotIn(forbidden, collect_src)
        self.assertNotIn('faces_256', collect_src)


if __name__ == '__main__':
    unittest.main()
