"""E03 family host-state owner ruling (2026-10-29, H.host_final_state): checks the committed ruling record against the
committed E03 evidence, re-derives it, and pins the condition-7 behaviour of the closure tool (the reboot path fails
when any other family gate fails; the live path and the committed closure stay unchanged).

Stdlib + git only, any host.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
RECORD = AUDIT / 'E03_FAMILY_HOST_STATE_OWNER_RULING.json'
SHA_FILE = AUDIT / 'E03_FAMILY_HOST_STATE_OWNER_RULING_SHA256.txt'
RULING = ROOT / 'configs' / 'amendments' / 'e03_family_host_state_reboot_resolution.yaml'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class E03FamilyHostStateRuling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(RECORD.read_text())
        cls.ruling = json.loads(RULING.read_text())
        cls.closure = json.loads((AUDIT / 'E03_THREE_SEED_TRAINING_CLOSURE.json').read_text())

    def test_01_ruling_scope(self):
        r = self.ruling
        self.assertEqual((r['gate'], r['result'], r['date']),
                         ('H.host_final_state', 'PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE', '2026-10-29'))
        self.assertEqual(len(r['conditions']), 7)
        self.assertFalse(r['changes_trained_model_or_procedure'])
        self.assertEqual(r['scientific_training_authority'], 'd547fc9f8932a273d719ae589aade85a9d989b85')
        self.assertEqual([x['id'] for x in r['related_rulings']],
                         ['seed1337_pane_log_keyboard_echo', 'seed2026_post_completion_reboot_evidence'])

    def test_02_all_conditions_verified(self):
        self.assertEqual(self.doc['status'], 'OWNER_RULING_RECORDED_CONDITIONS_VERIFIED')
        self.assertEqual(self.doc['failed_checks'], [])
        self.assertEqual(sorted(k for k in self.doc['checks'] if k[0].isdigit()),
                         sorted(self.ruling['conditions']))
        self.assertTrue(all(v['pass'] for v in self.doc['checks'].values()))
        for p in self.doc['checks']['gate_probes_fail_closed_and_live_path_unchanged']['probes'].values():
            self.assertIs(p['pass'], p['expected_pass'])

    def test_03_record_rederives(self):
        mod = load('tools/e03_family_host_state_ruling_record.py', 'ruling_record')
        doc = mod.build(ROOT, self.doc['fresh_host_recheck'])
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(mod.report(doc), (AUDIT / 'E03_FAMILY_HOST_STATE_OWNER_RULING.md').read_text())

    def test_04_condition_7_blocks_the_reboot_path(self):
        mod = load('tools/e03_three_seed_training_closure.py', 'e03_family')
        host = copy.deepcopy(self.closure['host_recheck'])
        host['seeds']['2026']['file_count'] = 1158
        doc, _ = mod.build(ROOT, host)
        g = next(x for x in doc['gates'] if x['gate'] == 'H.host_final_state')
        self.assertFalse(g['pass'])
        self.assertEqual(g['detail']['e03_tmux'], 'FAIL_POST_COMPLETION_REBOOT_EVIDENCE')
        self.assertEqual(g['detail']['blocked_by'], ['H.seed2026.host_matches_evidence'])
        # the live path does not depend on the other gates (unchanged behaviour)
        host['host_state'].update(e03_tmux_panes={f'e03_stdn_seed{s}': '1 0' for s in (42, 1337, 2026)})
        g = next(x for x in mod.build(ROOT, host)[0]['gates'] if x['gate'] == 'H.host_final_state')
        self.assertEqual((g['pass'], g['detail']['e03_tmux']), (True, 'LIVE_PANES_DEAD_STATUS_0'))

    def test_05_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), sorted([
            'outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING.json', 'outputs/audit/E03_FAMILY_HOST_STATE_OWNER_RULING.md',
            'configs/amendments/e03_family_host_state_reboot_resolution.yaml',
            'docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Family_Host_State_Reboot_Owner_Ruling.md']))


if __name__ == '__main__':
    unittest.main()
