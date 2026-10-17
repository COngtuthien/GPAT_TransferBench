"""M7D1-N6B: evidence checks for the completed M7D1-A1 GPAT-B0 / E08 / seed 42 scientific run (run_id 92e91ddb78f1e242).

Stdlib + the production runtime contract, any host. The run root lives on the GPU host; these tests validate the
committed evidence (read-only collected document, completion JSON, EMA manifest, SHA-256 manifest), re-check the A1
lambda_adv checkpoints against the canonical production implementation, and re-derive the completion JSON from the
collected document to prove the build is deterministic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
COMPLETION = AUDIT / 'M7D1_A1_E08_SEED42_COMPLETION.json'
MANIFEST = AUDIT / 'M7D1_A1_E08_SEED42_EMA_MANIFEST.txt'
COLLECTED = AUDIT / 'M7D1_A1_E08_SEED42_COLLECTED.json'
SHA_FILE = AUDIT / 'M7D1_A1_E08_SEED42_EVIDENCE_SHA256.txt'
AUTHORITY = '4466e1258c24e4a5d448dd5b30e8d8768dfe7174'
RUN_ID = '92e91ddb78f1e242'
CURRICULUM_ID = 'GPAT-TransferBench-v1.0+M7D1-A1'
LAMBDA_ADV = {5525: 0.0, 5526: 0.0, 5527: 4.528985507246377e-05, 6629: 0.04995471014492754, 6630: 0.05,
              6631: 0.05, 16575: 0.05, 16576: 0.10, 16577: 0.10, 66300: 0.10}


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n6b', ROOT / 'tools' / 'm7d1_n6b_a1_seed42_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticM7D1N6B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(COMPLETION.read_text())
        cls.collected = json.loads(COLLECTED.read_text())
        cls.manifest = MANIFEST.read_text()
        cls.t = tool()

    def test_01_status_identity(self):
        d = self.doc
        self.assertEqual(d['status'], 'M7D1_N6B_A1_SEED42_COMPLETED')
        self.assertEqual(d['milestone'], 'M7D1-N6B')
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['method'], d['experiment'], d['seed'], d['run_id']), ('GPAT-B0', 'E08', 42, RUN_ID))
        self.assertEqual(d['launch_authority_commit'], AUTHORITY)
        self.assertEqual(d['provenance']['code_commit'], AUTHORITY)
        self.assertEqual((d['curriculum_id'], d['amendment_id']), (CURRICULUM_ID, 'M7D1-A1'))
        self.assertEqual(d['provenance']['config_sha256'],
                         '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a')
        self.assertEqual(d['frozen_specification']['sha256'],
                         'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e')
        self.assertTrue(d['run_root'].endswith('/runs/m7_a1/E08/seed_42'))

    def test_02_completion_exact(self):
        d = self.doc
        self.assertEqual((d['optimizer_group_count'], d['epoch_count'], d['global_update'], d['final_epoch']),
                         (66300, 60, 66300, 60))
        self.assertEqual(d['update_sequence'], {'first': 1, 'last': 66300, 'contiguous': True, 'missing': [],
                                                'duplicates': []})
        self.assertEqual(self.collected['metrics']['non_complete_group_records'], [])
        self.assertEqual(d['non_finite_record_count'], 0)
        self.assertEqual(d['terminal_marker_counts'], {})

    def test_03_full_a1_schedule(self):
        a = self.doc['a1_schedule_audit']
        self.assertEqual(a['updates_checked'], 66300)
        self.assertEqual(a['mismatch_total'], 0)
        self.assertEqual(a['mismatches'], {})
        self.assertEqual(set(a['fields']), {'lambda_adv', 'scale_hf', 'lambda_con', 'lambda_spec', 'learning_rate',
                                            'epoch', 'group'})

    def test_04_lambda_adv_checkpoints_vs_production(self):
        from methods.gpat import runtime_contract as rc
        cps = self.doc['a1_schedule_audit']['lambda_adv_checkpoints']
        self.assertEqual(sorted(int(k) for k in cps), sorted(LAMBDA_ADV))
        for u, v in LAMBDA_ADV.items():
            self.assertEqual(cps[str(u)]['logged'], v, u)
            self.assertEqual(rc.curriculum(u)['lambda_adv'], v, u)
            self.assertEqual(cps[str(u)]['logged_hex'], float.hex(rc.lambda_adv_a1(u)), u)

    def test_05_amp_retries_recovered(self):
        a = self.doc['amp_retry_summary']
        self.assertEqual((a['total_retry_events'], a['affected_update_count']), (49, 46))
        self.assertEqual((a['min_D_scale'], a['min_G_scale']), (1024.0, 4096.0))
        self.assertEqual((self.doc['final_scalers']['D_scale'], self.doc['final_scalers']['G_scale']),
                         (131072.0, 65536.0))
        for r in a['events']:
            self.assertTrue(r['all_optimizer_steps_taken_zero'] and r['all_optimizer_update_false']
                            and r['all_pre_group_state_restored'] and r['update_not_advanced_on_retry'])
            self.assertEqual(r['complete_record_count'], 1)
            self.assertEqual(r['complete_amp_attempts'], len(r['retry_numbers']) + 1)

    def test_06_ema_candidates_and_manifest(self):
        e = self.doc['ema_candidates']
        self.assertEqual((e['count'], e['epoch_range']), (51, [10, 60]))
        self.assertIsNone(e['best_selected'])
        rows = e['candidates']
        self.assertEqual([r['epoch'] for r in rows], list(range(10, 61)))
        self.assertEqual(len({r['sha256'] for r in rows}), 51)
        for r in rows:
            self.assertEqual((r['code_commit'], r['selected'], r['index_selected_for_final']), (AUTHORITY, False, False))
        for cd in self.collected['candidates']:
            self.assertEqual(cd['metadata']['curriculum_id'], CURRICULUM_ID)
        self.assertEqual(self.t.ema_manifest_text(list(reversed(rows))), self.manifest)
        self.assertEqual(len(self.manifest.splitlines()), 52)
        self.assertEqual(hashlib.sha256(self.manifest.encode()).hexdigest(), e['sha256_manifest_digest'])

    def test_07_recovery_not_selection(self):
        r = self.doc['recovery_checkpoint']
        self.assertEqual((r['position']['global_update'], r['position']['epoch'], r['position']['next_group']),
                         (66300, 61, 1))
        self.assertEqual(r['provenance']['curriculum_id'], CURRICULUM_ID)
        self.assertEqual(r['provenance']['code_commit'], AUTHORITY)
        self.assertIn('NOT_ELIGIBLE_FOR_VAL_SELECTION', r['labels'])
        self.assertNotIn(r['sha256'], {c['sha256'] for c in self.doc['ema_candidates']['candidates']})

    def test_08_firewall_and_no_downstream(self):
        f = self.doc['firewall']
        for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata', 'test_metadata'):
            self.assertEqual(f[k], 0, k)
        self.assertFalse(any(self.doc['confirmations'].values()))
        self.assertTrue(all(p == 'E08' or p.startswith('E08/seed_42') for p in self.doc['a1_root_listing']))

    def test_09_historical_root_unchanged(self):
        h = self.doc['historical_root']
        self.assertEqual(h['listing_digest_before'], self.t.HISTORICAL_LISTING_DIGEST)
        self.assertEqual(h['listing_digest_after'], self.t.HISTORICAL_LISTING_DIGEST)
        self.assertEqual(h['file_count'], 59)

    def test_10_no_collapse_signature(self):
        s = self.doc['scientific_health']['collapse_signature']
        self.assertEqual(s['matching_updates'], 0)
        self.assertEqual(len(self.doc['scientific_health']['per_epoch']), 60)

    def test_11_run_root_integrity(self):
        i = self.doc['run_root_integrity']
        self.assertEqual(i['metrics_jsonl_sha256'], self.collected['metrics_sha256'])
        self.assertEqual(i['file_count'], 59)
        self.assertTrue(self.collected['run_root_unchanged'])
        self.assertTrue(self.collected['run_root_unchanged_whole_audit'])

    def test_12_rebuild_reproduces_completion(self):
        doc, text = self.t.build(self.collected, ROOT, self.doc['authority'])
        self.assertEqual(text, self.manifest)
        mine = dict(self.doc)
        for k in ('audit_timestamp_utc', 'observations'):
            doc.pop(k)
            mine.pop(k)
        self.assertEqual(json.loads(json.dumps(doc)), mine)

    def test_13_evidence_sha_manifest(self):
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)


if __name__ == '__main__':
    unittest.main()
