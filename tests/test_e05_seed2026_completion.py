"""E05 seed2026 closure: evidence checks for the completed E05 PCGAN (controlled architecture resolution) / seed 2026
scientific run (run_id b739351daf9779bc, training authority efdf753, CONTROLLED_ADAPTATION, fresh only).

Stdlib only, any host. The run root lives on the GPU host; these tests validate the committed evidence (completion
JSON with the embedded read-only collected document, checkpoint manifest, SHA-256 manifest) and re-derive the
completion JSON and manifest from the collected document to prove the build is deterministic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
COMPLETION = AUDIT / 'E05_SEED2026_COMPLETION.json'
MANIFEST = AUDIT / 'E05_SEED2026_CHECKPOINT_MANIFEST.txt'
SHA_FILE = AUDIT / 'E05_SEED2026_EVIDENCE_SHA256.txt'
AUTHORITY = 'efdf753156045dd2bed52229d9eac869b18d72f8'
RUN_ID = 'b739351daf9779bc'
CONFIG_SHA = '478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de'
FINAL_SHA = 'b7f0575f1e061936175ac04d2ac333e1008577bdd3654566c009a2f72bf0b166'
FIRST_SHA = '74a62dcc39925845f60596770f3fc9999733d0af6a6ff6400b9d5a71756833e4'
N = 4000


def tool():
    spec = importlib.util.spec_from_file_location('e05_seed_audit', ROOT / 'tools' / 'e05_seed_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE05Seed2026(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(COMPLETION.read_text())
        cls.c = cls.doc['collected']
        cls.m = cls.c['metrics']

    def test_01_status_seed_run_id_authority(self):
        d = self.doc
        self.assertEqual(d['status'], 'E05_SEED2026_COMPLETED_CLEANLY')
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['seed'], d['run_id'], d['scientific_authority'], d['config_sha256']),
                         (2026, RUN_ID, AUTHORITY, CONFIG_SHA))
        self.assertEqual(hashlib.sha256(f'E05|2026|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16], RUN_ID)
        man = self.c['run_manifest']
        self.assertEqual((man['run_id'], man['git_commit'], man['git_dirty'], man['experiment_seed']),
                         (RUN_ID, AUTHORITY, False, 2026))
        self.assertEqual((self.c['git']['head'], self.c['git']['status_porcelain']), (AUTHORITY, ''))
        self.assertEqual((d['method']['fidelity'], d['resume']), ('CONTROLLED_ADAPTATION', 'NOT_QUALIFIED_FRESH_ONLY'))
        self.assertNotIn('--resume', man['command_line'])
        self.assertEqual(self.c['event_counts'], {'e05_checkpoint_event': 2, 'e05_run_start': 1})

    def test_02_completed_exit_zero(self):
        h, s = self.c['host'], self.c['run_summary']
        self.assertEqual((h['exit_file'], h['tmux_pane_dead'], h['tmux_pane_dead_status'], h['runner_processes']),
                         ('0', True, 0, []))
        self.assertEqual((s['completion_status'], s['benchmark_iterations']), ('completed', N))

    def test_03_contiguous_records(self):
        m = self.m
        self.assertEqual((m['records'], m['first_step'], m['last_step'], m['distinct_global_steps']), (N, 1, N, N))
        self.assertTrue(m['global_step_is_1_to_n'])
        self.assertEqual(m['bad_lines'], 0)

    def test_04_predicted_order_and_unique_train_pairs(self):
        o, p = self.m['order'], self.m['pairs']
        self.assertEqual((o['predicted_positions'], o['match']), (N, N))
        self.assertEqual(o['sampler_order_sha256'], o['predicted_order_sha256'])
        self.assertEqual((o['distinct_pair_ids'], o['distinct_sampler_indices']), (N, N))
        self.assertEqual(self.doc['distinct_train_pairs'], N)
        for k in ('manifest_row_match', 'source_spoof_target_live', 'both_faces_train_same_dataset',
                  'face_png_sha256_match_m2_accounting', 'pair_sha256_match'):
            self.assertEqual(p[k], N, k)
        self.assertEqual(sum(p['records_by_dataset'].values()), N)
        self.assertEqual(self.m['consumed_splits'], ['TRAIN'])

    def test_05_optimizer_accounting(self):
        final = {'D': N, 'G': N, 'hook_D': N, 'hook_G': N, 'total': 2 * N}
        self.assertEqual(self.m['applications_exact'], N)
        self.assertEqual(self.m['final_applications'], final)
        self.assertEqual(self.doc['optimizer_applications'], final)
        self.assertEqual(self.c['run_summary']['optimizer_applications'], final)

    def test_06_losses_finite_g_terms_active(self):
        m = self.m
        self.assertEqual((m['losses_finite'], m['nonfinite_values'], m['g_terms_active']), (N, 0, N))
        self.assertLessEqual(m['g_total_unit_sum_max_relative_error'], 1e-5)
        self.assertEqual(self.doc['losses']['g_terms'], ['L_rec', 'L_recblur', 'L_advrec', 'L_advmix', 'L_pat'])

    def test_07_gradient_health(self):
        self.assertEqual(self.m['gradients_exact'], N)

    def test_08_two_checkpoints_semantics(self):
        ck = {x['global_step']: x for x in self.c['checkpoints']}
        self.assertEqual(sorted(ck), [1, N])
        self.assertEqual(self.c['checkpoints_dir_listing'], ['iteration_0001.pth', 'iteration_4000.pth'])
        a, b = ck[1], ck[N]
        self.assertEqual((a['checkpoint_type'], a['selected_for_final'], a['sha256']), ('periodic', False, FIRST_SHA))
        self.assertEqual((b['checkpoint_type'], b['selected_for_final'], b['selection_reason'], b['sha256']),
                         ('selected', True, 'BASELINE_FINAL_STATE_V1', FINAL_SHA))
        for x in (a, b):
            self.assertEqual((x['sha256_actual'], x['size_actual']), (x['sha256'], x['file_size_bytes']))
            self.assertEqual(x['load']['nonfinite_floating_tensors'], 0)
        self.assertEqual(b['load']['provenance']['optimizer_applications']['total'], 2 * N)
        self.assertEqual(self.doc['checkpoints']['final_selected'],
                         {'path': 'checkpoints/iteration_4000.pth', 'sha256': FINAL_SHA,
                          'rule': 'BASELINE_FINAL_STATE_V1'})

    def test_09_no_resume_sidecar(self):
        self.assertEqual((self.c['runner_state_files'], self.c['resume_files'], self.c['partial_files']), ([], [], []))
        self.assertTrue(all(e['resume_sidecar'] is None for e in self.c['checkpoint_events']))

    def test_10_val_test_zero(self):
        f = self.doc['firewall']
        self.assertEqual((f['val_access'], f['test_access'], f['val_changed'], f['test_changed']), (0, 0, 0, 0))
        for k, v in self.c['atime_firewall']['by_class'].items():
            if k.split('|')[1] in ('VAL', 'TEST'):
                self.assertEqual((v['atime_changed'], v['consumed'], v['sensitive_whole_window']), (0, 0, v['files']))
            else:
                self.assertEqual(v['changed_not_consumed'], 0, k)

    def test_11_driver_ruling_recorded(self):
        r = self.doc['driver_ruling']
        self.assertEqual((r['lock_driver_version'], r['observed_driver_version'], r['driver_policy']),
                         ('595.84', '580.178.04', 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY'))
        self.assertEqual(self.c['environment_recheck']['differences_other_than_driver'], [])
        self.assertTrue(self.c['run_start']['environment_identity_equals_lock'])
        self.assertIn('580.178.04', (AUDIT / 'E05_SEED2026_COMPLETION.md').read_text())

    def test_12_run_root_unchanged(self):
        r = self.doc['run_root_integrity']
        self.assertTrue(r['unchanged_whole_audit'])
        self.assertEqual(self.c['run_root_sha_listing_digest_before'], r['sha_listing_digest'])
        lines = ''.join(x + '\n' for x in self.c['run_root_sha_listing']).encode()
        self.assertEqual(hashlib.sha256(lines).hexdigest(), r['sha_listing_digest'])

    def test_13_deterministic_rebuild(self):
        doc, text = tool().build(self.c, ROOT)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(text, MANIFEST.read_text())

    def test_14_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/E05_SEED2026_CHECKPOINT_MANIFEST.txt',
                                         'outputs/audit/E05_SEED2026_COMPLETION.json',
                                         'outputs/audit/E05_SEED2026_COMPLETION.md'])


if __name__ == '__main__':
    unittest.main()
