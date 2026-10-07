"""M7D1-CLOSE-SEED42: evidence checks for the completed GPAT-B0 / E08 / seed 42 scientific run (run_id 7b799fbd6d0426da).

Stdlib only, any host. The run root lives on the GPU host; these tests validate the committed evidence
(the read-only collected document, the completion JSON and the EMA SHA-256 manifest) and re-derive the completion
JSON from the collected document to prove the build is deterministic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
COMPLETION = AUDIT / 'M7D1_E08_SEED42_COMPLETION.json'
MANIFEST = AUDIT / 'M7D1_E08_SEED42_EMA_SHA256.txt'
COLLECTED = AUDIT / 'M7D1_E08_SEED42_COLLECTED.json'
AUTHORITY = '058e976e5a538c4d18e1177f0eb27727cfb735cd'


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_close', ROOT / 'tools' / 'm7d1_close_seed42_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticM7D1Close(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(COMPLETION.read_text())
        cls.collected = json.loads(COLLECTED.read_text())
        cls.manifest = MANIFEST.read_text()
        cls.t = tool()

    def test_01_schema_and_status(self):
        d = self.doc
        for key in ('status', 'method', 'experiment', 'seed', 'run_id', 'authority_commit', 'completion_status',
                    'final_epoch', 'global_update', 'optimizer_group_count', 'epoch_count', 'amp_retry_summary',
                    'final_scalers', 'firewall', 'recovery_checkpoint', 'ema_candidates', 'provenance',
                    'confirmations', 'audit_timestamp_utc', 'classification', 'gates'):
            self.assertIn(key, d)
        self.assertEqual(d['classification'], 'SCIENTIFIC_TRAINING_COMPLETION_AUDIT')
        self.assertEqual(d['status'], 'M7D1_E08_SEED42_TRAINING_AUDITED_COMPLETE')
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['method'], d['experiment'], d['seed'], d['run_id']),
                         ('GPAT-B0', 'E08', 42, '7b799fbd6d0426da'))
        self.assertEqual(d['authority_commit'], AUTHORITY)
        self.assertEqual(d['completion_status'], 'completed')

    def test_02_update_sequence_exact(self):
        d, m = self.doc, self.collected['metrics']
        self.assertEqual((d['optimizer_group_count'], d['epoch_count'], d['global_update'], d['final_epoch']),
                         (66300, 60, 66300, 60))
        self.assertEqual(d['update_sequence'], {'first': 1, 'last': 66300, 'contiguous': True, 'missing': [],
                                                'duplicates': []})
        self.assertEqual(m['duplicate_epoch_group_pairs'], [])
        self.assertEqual(m['non_complete_group_records'], [])
        self.assertEqual(m['non_finite_record_count'], 0)
        self.assertEqual(d['terminal_numerical_failure_count'], 0)
        for e, v in m['per_epoch'].items():
            self.assertEqual((v['groups'], v['samples'], v['microbatches']), (1105, 8838, 2210), e)

    def test_03_final_epoch_and_tail_group(self):
        fe = self.doc['final_epoch_record']
        self.assertEqual((fe['epoch'], fe['global_step'], fe['optimizer_groups'], fe['microbatches'], fe['rows'],
                          fe['ema_active']), (60, 66300, 1105, 2210, 8838, True))
        lg = self.doc['final_group']
        self.assertEqual((lg['group_samples'], lg['microbatch_sizes']), (6, [4, 2]))

    def test_04_amp_retries_recovered(self):
        a = self.doc['amp_retry_summary']
        self.assertEqual(a['total_retry_events'], sum(len(r['retry_numbers']) for r in a['events']))
        self.assertNotIn('FAIL_CLOSED_AMP_OVERFLOW_FINAL', self.doc['terminal_marker_counts'])
        for r in a['events']:
            self.assertTrue(r['all_optimizer_steps_taken_zero'] and r['all_optimizer_update_false']
                            and r['all_pre_group_state_restored'] and r['update_not_advanced_on_retry'])
            self.assertEqual(r['complete_record_count'], 1)
            self.assertEqual(r['complete_amp_attempts'], len(r['retry_numbers']) + 1)

    def test_05_firewall_zero_non_train(self):
        f = self.doc['firewall']
        for k in ('non_train_images', 'val_images', 'test_images', 'val_metadata', 'test_metadata'):
            self.assertEqual(f[k], 0, k)
        self.assertIs(f['val_split_accessed'], False)
        self.assertIs(f['test_split_accessed'], False)
        self.assertEqual(f['unique_ids_sha256'], '27a8dff74625704c99b7dcbbf38a5b70ee9eaede0fe7b0ea8d6206719d5ebb1d')

    def test_06_recovery_provenance(self):
        r = self.doc['recovery_checkpoint']
        self.assertEqual(r['provenance']['code_commit'], AUTHORITY)
        self.assertEqual(r['provenance']['method'], 'GPAT-B0')
        self.assertEqual((r['position']['global_update'], r['position']['epoch'], r['position']['next_group']),
                         (66300, 61, 1))
        self.assertIn('NOT_ELIGIBLE_FOR_VAL_SELECTION', r['labels'])
        self.assertEqual(len(r['sha256']), 64)

    def test_07_candidates_51_epochs_10_to_60(self):
        e = self.doc['ema_candidates']
        self.assertEqual(e['count'], 51)
        self.assertEqual(e['epoch_range'], [10, 60])
        rows = e['candidates']
        self.assertEqual([r['epoch'] for r in rows], list(range(10, 61)))
        self.assertEqual([r['filename'] for r in rows], [f'ema_epoch_{k:02d}.pt' for k in range(10, 61)])
        self.assertEqual(len({r['sha256'] for r in rows}), 51)
        for r in rows:
            self.assertEqual(r['embedded_global_update'], r['epoch'] * 1105)
            self.assertEqual((r['code_commit'], r['method'], r['seed']), (AUTHORITY, 'GPAT-B0', 42))

    def test_08_no_selection_state(self):
        for r in self.doc['ema_candidates']['candidates']:
            self.assertIs(r['selected'], False)
            self.assertIs(r['index_selected_for_final'], False)
            self.assertEqual(r['val_fields'], [])
            self.assertEqual(r['test_fields'], [])
        c = self.doc['confirmations']
        self.assertFalse(any(c.values()))
        s = self.collected['run_summary']
        self.assertIsNone(s['final_or_selected_checkpoint_path'])
        self.assertIsNone(s['seed_level_evaluation_metrics'])

    def test_09_sha_manifest_deterministic(self):
        rows = self.doc['ema_candidates']['candidates']
        self.assertEqual(self.t.ema_manifest_text(list(reversed(rows))), self.manifest)
        self.assertEqual(len(self.manifest.splitlines()), 51)
        self.assertEqual(hashlib.sha256(self.manifest.encode()).hexdigest(),
                         self.doc['ema_candidates']['sha256_manifest_digest'])

    def test_10_rebuild_reproduces_completion(self):
        doc, text = self.t.build(self.collected, ROOT, self.doc['authority'])
        self.assertEqual(text, self.manifest)
        for k in ('audit_timestamp_utc', 'observations'):
            doc.pop(k)
        mine = dict(self.doc)
        mine.pop('audit_timestamp_utc')
        mine.pop('observations')
        self.assertEqual(json.loads(json.dumps(doc)), mine)

    def test_11_provenance(self):
        p = self.doc['provenance']
        self.assertEqual(p['config_sha256'], '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a')
        self.assertEqual(p['pair_manifest_sha256'], 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243')
        self.assertEqual(p['environment_lock_sha256'],
                         '24c983ebbb308acacd63f32837532e5f9166114962f524e993f243f2ff746114')
        self.assertEqual(p['amp_amendment_sha256'], '4dc8a838cbdce72c454313b4951684dfad8d90f82b8a6a8a090d355e8499e0a7')

    def test_12_run_root_unchanged(self):
        self.assertTrue(self.collected['run_root_unchanged'])
        self.assertEqual(self.collected['run_root_tree_before'], self.collected['run_root_tree_after'])


if __name__ == '__main__':
    unittest.main()
