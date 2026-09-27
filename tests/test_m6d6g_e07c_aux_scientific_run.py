"""M6D6g E07c auxiliary SCIENTIFIC run (seed 42 x 200 epochs): recorded evidence, identities and final SHA256.

Static and read-only: no Torch, no CUDA, no TRAIN image, no Parquet decode, no rerun, and the 185-MB scientific
checkpoint and the engineering sidecar are never opened (their SHA256 values are checked as recorded evidence).
"""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '7642b23e60a1d89fcd481b03d4c9cb361f4c6e20'
M6D6G_COMMIT = 'ee0a9b9166577cac606c142af5bc2e338d75eea1'
FINAL_SHA = '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c'
PENDING = 'SHA256_RECORDED_PENDING_OWNER_FREEZE'
RUN = ROOT / 'outputs/audit/M6D6G_E07C_AUX_SCIENTIFIC_RUN.json'
FINAL = ROOT / 'outputs/audit/M6D6G_E07C_AUX_FINAL_CHECKPOINT.json'
LOG = ROOT / 'outputs/audit/M6D6G_E07C_AUX_SCIENTIFIC_RUNTIME_LOG.txt'
REPORT = ROOT / 'outputs/audit/M6D6G_E07C_AUX_SCIENTIFIC_RUN.md'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6d6g_preflight_under_test',
                                                  ROOT / 'tools/m6d6g_e07c_aux_scientific_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


class TestM6D6gScientificRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.ev = json.loads(RUN.read_text())
        cls.final = json.loads(FINAL.read_text())

    def test_01_authority_commit(self):
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        self.assertEqual((self.ev['authority_commit'], self.final['authority_commit']), (AUTHORITY, AUTHORITY))

    def test_02_source_config_a3_a6_a7_a8_identity(self):
        self.assertEqual(self.ev['identities'], self.pf.IDENTITIES)
        for rel, digest in self.pf.EXECUTED.items():
            self.assertEqual(self.pf.sha((ROOT / rel).read_bytes()), digest, rel)

    def test_03_scientific_seed_42(self):
        self.assertEqual(self.ev['run_identity']['auxiliary_encoder_training_seed'], 42)
        self.assertIsNone(self.ev['run_identity']['experiment_seed'])
        self.assertEqual((self.ev['seeding']['seed'], self.ev['seeding']['pythonhashseed']), (42, '42'))

    def test_04_one_logical_scientific_run(self):
        self.assertEqual(self.ev['scientific_counters']['scientific_auxiliary_logical_runs'], 1)
        self.assertEqual(len(self.ev['process_sessions']), 1)
        self.assertEqual((self.ev['interruptions'], self.ev['resume_reconciliations']), ([], []))
        self.assertEqual(self.ev['launch']['launches'], 1)
        self.assertIsNone(self.ev['launch']['resume_state_argument'])

    def test_05_completed_epoch_200(self):
        self.assertEqual(self.ev['step_accounting']['completed_epochs'], 200)
        self.assertEqual(self.ev['step_accounting']['committed_boundary']['completed_epoch'], 200)

    def test_06_logical_optimizer_steps_11200(self):
        a = self.ev['step_accounting']
        self.assertEqual((a['logical_authoritative_optimizer_steps'], a['physical_optimizer_steps_executed'],
                          a['superseded_optimizer_steps']), (11200, 11200, 0))

    def test_07_every_epoch_has_56_steps(self):
        eps = self.ev['epochs']
        self.assertEqual([e['epoch'] for e in eps], list(range(1, 201)))
        self.assertTrue(all((e['optimizer_steps'], e['consumed'], e['dropped']) == (56, 14336, 131) for e in eps))
        self.assertEqual(self.ev['metrics_verification']['logical_steps_per_epoch_distinct'], [56])
        self.assertEqual(self.ev['metrics_verification']['logical_batch_sizes'], [256])

    def test_08_final_checkpoint_epoch_200(self):
        self.assertEqual((self.final['epoch'], self.final['global_step'], self.final['checkpoint_type'],
                          self.final['selected_for_final']), (200, 11200, 'selected', True))

    def test_09_final_checkpoint_sha_recorded_and_agrees(self):
        self.assertEqual(self.final['sha256'], FINAL_SHA)
        self.assertEqual(set(self.final['sha256_sources_agree'].values()), {FINAL_SHA})
        self.assertEqual(self.ev['runtime_files']['checkpoints/encoder_final.pkl']['sha256'], FINAL_SHA)

    def test_10_status_pending_owner_freeze(self):
        self.assertEqual(self.final['final_checkpoint_status'], PENDING)
        self.assertFalse(self.final['authoritative_for_main_difffas'])
        self.assertFalse(self.final['owner_freeze_performed'])
        self.assertIn('E07c_AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', self.ev['not_claimed'])
        self.assertIn('AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN', self.ev['not_qualified'])

    def test_11_completion_status_completed(self):
        self.assertEqual(self.ev['run_identity']['completion_status'], 'completed')
        self.assertEqual(self.ev['launch']['exit_status'], 0)

    def test_12_whole_module_checkpoint_semantics_unchanged(self):
        self.pf.checkpoint_semantics_unchanged()
        self.assertTrue(self.final['format'].startswith('torch.save of the WHOLE nn.Module'))
        self.assertFalse(self.final['deserialized_for_evidence'])

    def test_13_no_val_selection(self):
        self.assertTrue(self.final['selection_reason'].startswith('FINAL_STATE_AFTER_EPOCH_200'))
        self.assertEqual(self.ev['checkpoint_events']['periodic_not_selected'], 199)
        self.assertEqual(self.ev['data_access']['VAL_rows_used'], 0)
        self.assertTrue(self.ev['metrics_verification']['val_losses_all_null'])

    def test_14_no_test_use(self):
        d = self.ev['data_access']
        self.assertEqual((d['TEST_rows_used'], d['TEST_image_reads'], d['test_split_accessed']), (0, 0, False))
        self.assertEqual(self.ev['population']['test_rows_materialized'], 0)

    def test_15_no_main_difffas_scientific_run(self):
        self.assertEqual(self.ev['scientific_counters']['main_difffas_scientific_runs'], 0)
        self.assertEqual(self.ev['method_status'], 'IMPLEMENTED_NOT_EXECUTED')
        self.assertIn('MAIN_DIFFFAS_SCIENTIFIC_TRAINING_COMPLETED', self.ev['not_claimed'])

    def test_16_no_m8_bank(self):
        self.assertFalse(self.ev['scientific_counters']['M8_bank'])
        self.assertIn('M8_BANK', self.ev['not_qualified'])

    def test_17_scientific_files_byte_identical_to_authority(self):
        # B1 (M6D6h, PROSPECTIVE_REGRESSION_HARNESS_SCOPE_CORRECTION): the historical claim is that M6D6g itself left
        # the scientific trees unchanged, i.e. over the commit range AUTHORITY..M6D6G_COMMIT; it does not lock HEAD.
        self.assertEqual(git('merge-base', '--is-ancestor', M6D6G_COMMIT, 'HEAD').returncode, 0)
        self.assertEqual(git('rev-parse', M6D6G_COMMIT + '^').stdout.decode().strip(), AUTHORITY)
        diff = git('diff', '--name-only', AUTHORITY, M6D6G_COMMIT, '--', *self.pf.FROZEN_TREES)
        self.assertEqual((diff.returncode, diff.stdout.decode().strip()), (0, ''))
        for rel, digest in self.pf.EXECUTED.items():
            blob = git('show', f'{M6D6G_COMMIT}:{rel}').stdout
            self.assertEqual((self.pf.sha(blob), blob), (digest, git('show', f'{AUTHORITY}:{rel}').stdout), rel)
        spec = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
        self.assertEqual(self.pf.sha(git('show', f'{M6D6G_COMMIT}:{spec}').stdout), self.pf.SPEC_SHA)

    def test_18_runtime_manifest_git_commit(self):
        self.assertEqual((self.ev['runtime_git_commit'], self.ev['runtime_git_dirty']), (AUTHORITY, False))

    def test_19_run_root_is_seed_42_auxiliary_root(self):
        self.assertTrue(self.ev['run_identity']['run_dir'].endswith('/runs/m6/E07c/aux_encoder/seed_42'))
        self.assertEqual(self.final['absolute_path'], self.ev['run_identity']['run_dir'] + '/checkpoints/encoder_final.pkl')

    def test_20_runtime_evidence_hashes_consistent(self):
        run, final = self.pf.check_evidence()
        log = LOG.read_text()
        for name, v in run['runtime_files'].items():
            self.assertIn(v['sha256'], log, name)
        self.assertIn(FINAL_SHA, REPORT.read_text())

    def test_21_logical_train_consumption_and_data_firewall(self):
        d = self.ev['data_access']
        self.assertEqual((d['TRAIN_logical_consumption'], d['superseded_optimizer_steps'], d['raw_benchmark_image_reads'],
                          d['main_8838_row_relation_used']), (2867200, 0, 0, 0))
        # face reads were not instrumented by the production process: explicit null + reason, never a derived number
        for field in ('TRAIN_physical_face_reads', 'TRAIN_superseded_face_reads'):
            self.assertIsNone(d[field])
            self.assertIn('not independently instrumented', d['missing_field_reasons'][field])
        self.assertNotIn('physical_read_basis', d)
        self.assertIn('not a face-image read', d['manifest_read_disclosure'])
        with self.assertRaises(ValueError):
            self.pf.read('outputs/x/encoder_final.pkl')

    def test_22_preflight_is_static(self):
        tree = ast.parse((ROOT / 'tools/m6d6g_e07c_aux_scientific_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertEqual('torch' in sys.modules, self.torch_before)


if __name__ == '__main__':
    unittest.main()
