"""M7D1-N7A closure: evidence checks for the OWNER-STOPPED M7D1-A1 GPAT-B0 / E08 / seed 1337 scientific run
(run_id 62d7cfd98022a95a), stopped at u15107 after direct confirmation of generator identity collapse.

Stdlib + the production runtime contract + the TRAIN relation, any host. The run root lives on the GPU host; these
tests validate the committed evidence (read-only collected document, direct-output diagnostic, closure JSON, report,
SHA-256 manifest), re-derive the diagnostic TRAIN subset, re-check the onset window against the production A1
curriculum, and rebuild the closure JSON from the committed inputs to prove the build is deterministic.
The run is NOT a completed run: nothing here expects 60 epochs / 66300 updates.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
CLOSURE = AUDIT / 'M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.json'
DIRECT = AUDIT / 'M7D1_A1_E08_SEED1337_COLLAPSE_DIRECT.json'
REPORT = AUDIT / 'M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.md'
COLLECTED = AUDIT / 'M7D1_A1_E08_SEED1337_STOPPED_COLLECTED.json'
SHA_FILE = AUDIT / 'M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE_EVIDENCE_SHA256.txt'
TOOL = ROOT / 'tools' / 'm7d1_n7a_seed1337_stopped_collapse.py'
DIAG_TOOL = ROOT / 'tools' / 'm7d1_n7a_seed1337_direct_output_diag.py'
N2_COLLECTED = AUDIT / 'M7D1_E08_SEED42_SATURATION_COLLECTED.json'
V10_EMA = AUDIT / 'M7D1_E08_SEED42_EMA_SHA256.txt'
A1_EMA = AUDIT / 'M7D1_A1_E08_SEED42_EMA_MANIFEST.txt'
N6B_COMPLETION = AUDIT / 'M7D1_A1_E08_SEED42_COMPLETION.json'
AUTHORITY = '4466e1258c24e4a5d448dd5b30e8d8768dfe7174'
RUN_ID = '62d7cfd98022a95a'
STATUS = 'OWNER_STOPPED_CONFIRMED_GENERATOR_COLLAPSE'
SEED42_A1 = '9b1a85eaeaf635450807d43a7e19e97d447d6a18f2585c5fc7720b8efadfadf3'
SEED42_V10 = 'ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a'
RECOVERY_FINAL = 'c1b4295e8c1a2c0838daa5ee833f8da003bb4b09382fa2cf26151dc2f75f6fe1'
RECOVERY_EPOCH13 = '0f845ec5530b0d7ab6eca521eba374e9efd5698da28e1d58271c74ed48ade2b8'
DIAG_TOOL_SHA = 'e1fe0b39698aa014fd20875c6ef68acf97779822f1e2f43e4a670298c3a93640'
DIAG_RAW_SHA = 'baff772722eb697bd4582b89f8fe77d0b21d1c1353cd31d1c1c74f180684b9ae'


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n7a_closure', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticM7D1N7AStoppedCollapse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(CLOSURE.read_text())
        cls.direct = json.loads(DIRECT.read_text())
        cls.collected = json.loads(COLLECTED.read_text())
        cls.t = tool()

    def test_01_status_owner_stopped_not_completed(self):
        d = self.doc
        self.assertEqual(d['status'], STATUS)
        self.assertEqual(d['final_run_status'], STATUS)
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['method'], d['experiment'], d['seed'], d['run_id']), ('GPAT-B0', 'E08', 1337, RUN_ID))
        self.assertEqual(d['launch_authority_commit'], AUTHORITY)
        self.assertEqual(d['provenance']['code_commit'], AUTHORITY)
        self.assertEqual(d['curriculum_id'], 'GPAT-TransferBench-v1.0+M7D1-A1')
        self.assertTrue(d['run_root'].endswith('/runs/m7_a1/E08/seed_1337'))
        c = d['completion']
        self.assertEqual(c['manifest_completion_status'], 'interrupted')
        self.assertEqual(c['summary_completion_status'], 'interrupted')
        self.assertNotEqual(c['last_complete_update'], c['planned_updates'])
        self.assertLess(c['completed_fraction_of_plan'], 1)
        for phrase in ('a completed 60-epoch scientific run', 'a numerical hard failure', 'an AMP hard failure'):
            self.assertIn(phrase, d['this_is_not'])
        self.assertFalse(d['integrity']['terminal_amp_failure'])
        self.assertFalse(d['integrity']['numerical_hard_failure'])

    def test_02_last_complete_update(self):
        c = self.doc['completion']
        self.assertEqual((c['last_complete_update'], c['last_complete_epoch']), (15107, 13))
        self.assertEqual((c['last_complete_record']['epoch'], c['last_complete_record']['group']), (14, 742))
        self.assertEqual(self.collected['run_summary']['global_update'], 15107)
        seq = self.doc['integrity']['update_sequence']
        self.assertEqual(seq, {'first': 1, 'last': 15107, 'contiguous': True, 'duplicates': [], 'gaps': []})
        self.assertEqual(self.doc['integrity']['record_counts'],
                         {'optimizer_group_COMPLETE': 15107, 'epoch': 13, 'amp_retry': 9})
        self.assertEqual(self.doc['integrity']['epoch_records'], list(range(1, 14)))
        self.assertEqual(self.doc['integrity']['non_complete_group_records'], [])
        self.assertEqual(self.doc['integrity']['nonfinite_records'], [])

    def test_03_owner_stop_record(self):
        s = self.doc['owner_stop']
        self.assertEqual(s['signal'], 'SIGTERM')
        self.assertTrue(s['target'].startswith('main training process only'))
        self.assertEqual(s['shell_exit_status'], 0)
        self.assertIn('NOT scientific completion', s['exit_status_interpretation'])
        self.assertLess(s['update_before_diagnostic'], s['update_at_pre_stop_snapshot'])
        self.assertLessEqual(s['update_at_pre_stop_snapshot'], 15107)
        self.assertLess(s['diagnostic_utc'], s['sent_utc'])
        self.assertEqual(self.collected['training_processes'], [])

    def test_04_direct_output_confirmation(self):
        x = self.direct
        self.assertEqual(x['conclusion'], 'DIRECT_COLLAPSE_CONFIRMED')
        self.assertEqual(x['code'], AUTHORITY)
        self.assertIn('production GPATCore', x['forward'])
        self.assertEqual(x['scale_hf'], 0.10)
        self.assertEqual(x['raw_result_sha256'], DIAG_RAW_SHA)
        self.assertEqual(x['diag_tool']['sha256'], DIAG_TOOL_SHA)
        self.assertEqual(hashlib.sha256(DIAG_TOOL.read_bytes()).hexdigest(), DIAG_TOOL_SHA)
        subj = sorted(k for k in x['states'] if k.startswith('s1337_a1/'))
        self.assertEqual(subj, ['s1337_a1/ema_epoch_10', 's1337_a1/ema_epoch_11', 's1337_a1/ema_epoch_12',
                                's1337_a1/ema_epoch_13', 's1337_a1/recovery_ema', 's1337_a1/recovery_live'])
        ix = {r['epoch']: r['sha256'] for r in self.doc['checkpoints']['ema_candidates']}
        for e in (10, 11, 12, 13):
            self.assertEqual(x['states'][f's1337_a1/ema_epoch_{e:02d}']['checkpoint_sha256'], ix[e])
        for k in ('recovery_live', 'recovery_ema'):
            self.assertEqual(x['states'][f's1337_a1/{k}']['checkpoint_sha256'], RECOVERY_EPOCH13)
            self.assertEqual(x['states'][f's1337_a1/{k}']['position']['global_update'], 14365)

    def test_05_exact_target_copy(self):
        for k, s in self.direct['states'].items():
            if not k.startswith('s1337_a1/'):
                continue
            self.assertEqual(s['u8_x_hat_equals_u8_target_count'], 32, k)
            self.assertEqual(s['u8_max_level_diff'], 0, k)
            self.assertEqual(s['u8_pixels_differing_fraction_per_image']['max'], 0.0, k)
            self.assertLess(s['x_hat_minus_target_mean_abs_per_image']['max'], 1e-7, k)
            self.assertLess(s['x_hat_minus_target_max_abs'], 1e-6, k)
            self.assertLess(s['A_mean_per_image']['median'], 1e-17, k)            # ~1e-18 .. 1e-20
            self.assertLess(s['M_mean_per_image']['median'], 1e-16, k)            # ~1e-17 .. 1e-18
            self.assertLess(s['A_pixel_quantiles']['max'], 1e-11, k)
            self.assertLess(s['M_pixel_quantiles']['max'], 1e-10, k)
            self.assertEqual(s['M_fraction_lt_0p01'], 1.0, k)
            self.assertAlmostEqual(s['delta_HF_abs_quantiles']['p50'], 0.10, places=6)   # saturated, masked out
        f = self.doc['direct_collapse']['findings']['s1337_a1']
        self.assertEqual(set(f['exact_u8_target_copies'].values()), {32})
        self.assertEqual(f['M_fraction_lt_0p01'], [1.0])

    def test_06_comparison_provenance(self):
        """Reference checkpoints are exactly the committed EMA candidates of the two seed42 runs."""
        x = self.direct['states']
        v10_sha = {ln.split()[1]: ln.split()[0] for ln in V10_EMA.read_text().splitlines() if ln.strip()}
        a1_sha = {int(ln.split()[0]): ln.split()[2] for ln in A1_EMA.read_text().splitlines()
                  if ln.strip() and not ln.startswith('#')}
        for e in (10, 11, 12, 13):
            v10, ok = x[f's42_v10/ema_epoch_{e:02d}'], x[f's42_a1/ema_epoch_{e:02d}']
            self.assertEqual(v10['checkpoint_sha256'], v10_sha[f'checkpoints/ema_candidates/ema_epoch_{e:02d}.pt'])
            self.assertEqual(ok['checkpoint_sha256'], a1_sha[e])
            self.assertEqual((v10['global_update'], ok['global_update']), (e * 1105, e * 1105))
            self.assertIn('historical v1.0 seed42', v10['role'])
            self.assertIn('healthy M7D1-A1 seed42', ok['role'])
            self.assertEqual(v10['u8_x_hat_equals_u8_target_count'], 32)                 # same failure mode
            self.assertEqual(v10['M_fraction_lt_0p01'], 1.0)
            self.assertEqual(ok['u8_x_hat_equals_u8_target_count'], 0)                   # healthy
            self.assertGreater(ok['A_mean_per_image']['median'], 1e-3)
            self.assertGreater(ok['M_fraction_gt_0p99'], 0.99)
            self.assertGreater(ok['x_hat_minus_target_mean_abs_per_image']['median'], 1e-3)
        f = self.doc['direct_collapse']['findings']
        self.assertEqual(set(f['s42_v10']['exact_u8_target_copies'].values()), {32})
        self.assertEqual(set(f['s42_a1']['exact_u8_target_copies'].values()), {0})

    def test_07_train_subset_identity(self):
        from methods.gpat import runner_io as rio
        sub = self.direct['subset']
        self.assertEqual((sub['size_pairs'], sub['images']), (32, 64))
        records = rio.read_relation()
        salt = 'M7D1-N2|fixed-train-diagnostic-subset|v1'
        ranked = sorted(records, key=lambda r: hashlib.sha256(f"{salt}|{r['pair_id']}".encode()).hexdigest())
        exp = [{'index': r['index'], 'pair_id': r['pair_id'], 'dataset': r['dataset']}
               for r in sorted(ranked[:32], key=lambda r: r['index'])]
        self.assertEqual(sub['rows'], exp)
        self.assertEqual(sub['rows'], json.loads(N2_COLLECTED.read_text())['output_subset'])

    def test_08_firewall_zero_val_test(self):
        f = self.doc['firewall']
        self.assertEqual((f['val_access'], f['test_access']), (0, 0))
        a = f['scientific_run_access']
        self.assertEqual((a['val_images'], a['test_images'], a['val_metadata'], a['test_metadata'],
                          a['non_train_images']), (0, 0, 0, 0, 0))
        self.assertEqual(a['train_images'], a['opened_events'])
        self.assertFalse(f['manifest_val_split_accessed'])
        self.assertFalse(f['manifest_test_split_accessed'])
        d = self.direct['firewall']
        self.assertEqual(d['access_log']['train_images'], 64)
        self.assertEqual((d['access_log']['val_images'], d['access_log']['test_images']), (0, 0))
        self.assertTrue(d['faces_opened_all_in_subset'])
        self.assertEqual(d['suspicious_val_test_paths'], [])
        self.assertEqual(d['run_root_write_opens'], [])

    def test_09_a1_schedule_and_onset_stage1(self):
        from methods.gpat import runtime_contract as rc
        a = self.doc['integrity']['a1_schedule_audit']
        self.assertEqual((a['updates_checked'], a['mismatch_total'], a['mismatches_first']), (15107, 0, {}))
        self.assertEqual(set(a['fields']), {'lambda_adv', 'scale_hf', 'lambda_con', 'lambda_spec', 'learning_rate',
                                            'epoch', 'group'})
        o = self.doc['onset']
        self.assertEqual((o['grad_spike_update'], o['epoch'], o['stage'], o['lambda_adv']), (2167, 2, 1, 0.0))
        self.assertLess(o['G_grad_norm_before'], 0.05)
        self.assertGreater(o['G_grad_norm_at_spike_pre_clip'], 2.0)
        self.assertEqual(o['amp_attempts_at_spike'], 1)
        self.assertNotIn(2167, self.doc['integrity']['amp']['retry_updates'])
        for r in o['window']:
            cur = rc.curriculum_a1(r['u'])
            self.assertEqual((r['stage'], r['lambda_adv'], r['scale_hf']), (cur['stage'], cur['lambda_adv'],
                                                                              cur['s_hf']))
            self.assertEqual(r['lambda_adv'], 0.0)
        self.assertLess(o['grad_spike_update'], rc.A1_RAMP_FIRST)
        self.assertIn('CANNOT BE ATTRIBUTED TO THE M7D1-A1 ADVERSARIAL TRANSITION', o['attribution'])
        self.assertTrue(o['root_cause'].startswith('NOT INFERRED'))
        sg = o['signature']
        self.assertEqual((sg['first'], sg['run_at_end_start'], sg['run_at_end'], sg['updates']),
                         (2174, 2176, 12932, 12933))
        w = {r['u']: r for r in o['window']}
        self.assertTrue(w[2174]['signature'] and not w[2175]['signature'])
        self.assertGreaterEqual(w[2171]['budget'], 0.0096)
        self.assertLess(o['per_epoch']['14']['G_grad_norm_median'], 1e-7)

    def test_10_amp_numerical(self):
        amp = self.doc['integrity']['amp']
        self.assertEqual(amp['retry_event_count'], 9)
        self.assertEqual(amp['retry_updates'], [1798, 2132, 7771, 9802, 12055, 12146, 14057, 14173])
        self.assertEqual(amp['min_scales'], {'D_min': 8192.0, 'G_min': 65536.0})
        self.assertEqual(amp['final_scales'], {'D': 16384.0, 'G': 2097152.0})
        offenders = sorted({o for r in amp['retry_events'] for o in r['offending_optimizers']})
        self.assertEqual(offenders, ['D_OPT', 'G_OPT'])

    def test_11_recovery_states_not_selectable(self):
        r = self.doc['recovery_states']
        a, b = r['final_mid_epoch_after_stop'], r['epoch13_end_copy']
        self.assertEqual(a['sha256'], RECOVERY_FINAL)
        self.assertEqual({k: a['position'][k] for k in ('epoch', 'next_group', 'global_update')},
                         {'epoch': 14, 'next_group': 743, 'global_update': 15107})
        self.assertEqual(b['sha256'], RECOVERY_EPOCH13)
        self.assertEqual({k: b['position'][k] for k in ('epoch', 'next_group', 'global_update')},
                         {'epoch': 14, 'next_group': 1, 'global_update': 14365})
        self.assertTrue(b['path'].startswith('<runtime>/diagnostics/m7/M7D1_N7A_seed1337/'))
        for s in (a, b):
            self.assertFalse(s['selectable_ema_checkpoint'])
            self.assertIn('RECOVERY_ONLY', s['labels'])
            self.assertIn('NOT_ELIGIBLE_FOR_VAL_SELECTION', s['labels'])
            self.assertEqual(s['provenance']['code_commit'], AUTHORITY)
        self.assertFalse(r['pre_onset_state_available'])

    def test_12_no_selection(self):
        ck = self.doc['checkpoints']
        self.assertFalse(ck['selection_performed'])
        self.assertEqual([e['epoch'] for e in ck['ema_candidates']], [10, 11, 12, 13])
        self.assertTrue(all(e['selected_for_final'] is False for e in ck['ema_candidates']))
        ix = self.collected['checkpoint_index']
        self.assertEqual(ix['labels'], [])
        self.assertTrue(all(r['selected_for_final'] is False for r in ix['checkpoints']))
        s = self.collected['run_summary']
        self.assertIsNone(s['final_or_selected_checkpoint_path'])
        self.assertIsNone(s['final_or_selected_checkpoint_sha256'])
        self.assertFalse(self.doc['confirmations']['selection_run'])

    def test_13_root_integrity(self):
        ri = self.doc['root_integrity']
        self.assertEqual(ri['pre_stop']['listing_digest'],
                         '84f5b8cd3fe37406a7e49dbaccb2d5744445753d8f7172efcef32de25394d2f8')
        self.assertEqual(ri['post_stop']['listing_digest'], ri['now']['digest'])
        self.assertEqual(ri['now']['digest'], ri['now']['digest_after_collect'])
        self.assertEqual((ri['pre_stop']['file_count'], ri['now']['file_count']), (11, 12))
        self.assertEqual(ri['closeout_observed_rewritten'], ri['closeout_expected_changes']['rewritten'])
        self.assertEqual(sorted(ri['run_root_files']),
                         sorted(ri['closeout_expected_changes']['added'] + ri['closeout_expected_changes']['rewritten']
                                + ri['closeout_expected_changes']['unchanged']))
        self.assertEqual(ri['run_root_files']['checkpoints/recovery/latest.pt']['sha256'], RECOVERY_FINAL)
        self.assertEqual(ri['a1_runs_listing'], ['E08/seed_1337', 'E08/seed_42'])

    def test_14_seed42_roots_unchanged(self):
        s = self.doc['root_integrity']['seed42_roots']
        self.assertEqual(s['a1_seed42']['digest_before'], SEED42_A1)
        self.assertEqual(s['a1_seed42']['digest_after'], SEED42_A1)
        self.assertEqual(s['v1_0_seed42']['digest_before'], SEED42_V10)
        self.assertEqual(s['v1_0_seed42']['digest_after'], SEED42_V10)
        self.assertIn(SEED42_A1, N6B_COMPLETION.read_text())
        self.assertIn(SEED42_V10, N6B_COMPLETION.read_text())
        o = self.direct['other_roots_listing']
        self.assertTrue(o['unchanged'])
        self.assertEqual((o['before']['s42_a1'][0], o['before']['s42_v10'][0]), (SEED42_A1, SEED42_V10))

    def test_15_confirmations(self):
        c = self.doc['confirmations']
        self.assertEqual(set(c.values()), {False})
        for k in ('replay_started', 'fix_implemented', 'val_run', 'test_run', 'seed_2026_started',
                  'b1_b2_b3_started', 'bank_or_downstream_started', 'resumed', 'production_code_modified'):
            self.assertIn(k, c)

    def test_16_rebuild_deterministic(self):
        doc = self.t.build(self.collected, self.direct, ROOT)
        self.assertEqual(json.loads(json.dumps(doc, sort_keys=True)), self.doc)
        self.assertEqual(self.t.report(doc), REPORT.read_text())

    def test_17_sha256_manifest(self):
        lines = [ln.split() for ln in SHA_FILE.read_text().splitlines() if ln.strip()]
        paths = [p for _, p in lines]
        self.assertEqual(sorted(paths), sorted([
            'outputs/audit/M7D1_A1_E08_SEED1337_COLLAPSE_DIRECT.json',
            'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.json',
            'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.md',
            'outputs/audit/M7D1_A1_E08_SEED1337_STOPPED_COLLECTED.json',
            'tools/m7d1_n7a_seed1337_direct_output_diag.py',
            'tools/m7d1_n7a_seed1337_stopped_collapse.py']))
        for h, p in lines:
            self.assertEqual(hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), h, p)

    def test_18_no_completion_artifacts(self):
        for name in ('M7D1_A1_E08_SEED1337_COMPLETION.json', 'M7D1_A1_E08_SEED1337_EMA_MANIFEST.txt',
                     'M7D1_A1_E08_SEED1337_COMPLETION.md'):
            self.assertFalse((AUDIT / name).exists(), name)
        self.assertFalse((ROOT / 'tools' / 'm7d1_n7a_a1_seed1337_audit.py').exists())
        self.assertFalse((ROOT / 'tests' / 'test_m7d1_n7a_a1_seed1337_completion.py').exists())


if __name__ == '__main__':
    unittest.main()
