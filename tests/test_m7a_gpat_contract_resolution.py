"""M7A: GPAT M7 contract resolution (Amendment A10) + historical GPAT reconciliation. Static only.

History assertions read the state at the commit that ADDED this test (candidate: the worktree) and compare it with the
M6E authority, so later milestones never break them (no HEAD lock; owner decision B1).
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '8d4ddb3c55e8809398346dbdc10d15318e14b60d'
THIS = 'tests/test_m7a_gpat_contract_resolution.py'
PREFLIGHT = 'tools/m7a_gpat_contract_resolution_preflight.py'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m7a_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7a_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7a(rel):
    """File bytes at the M7A state: the commit that added this test, else the (candidate) worktree."""
    commit = m7a_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def exists_at_m7a(rel):
    commit = m7a_commit()
    return git('cat-file', '-e', f'{commit}:{rel}').returncode == 0 if commit else (ROOT / rel).exists()


def listdir_at_m7a(rel):
    commit = m7a_commit()
    if not commit:
        return sorted(p.name for p in (ROOT / rel).iterdir())
    return sorted(Path(p).name for p in git('ls-tree', '--name-only', f'{commit}:{rel}').stdout.decode().split())


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM7AContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.record = cls.pf.load_record(at_m7a)
        cls.result = cls.pf.check_record(at_m7a, at_authority)

    def rejects(self, mutate, check='check_decisions'):
        bad = copy.deepcopy(self.record)
        mutate(bad)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            getattr(self.pf, check)(bad)

    def test_00_authority_and_m6_closed(self):
        self.assertEqual(self.pf.AUTHORITY, AUTHORITY)
        self.assertEqual(self.record['authority_commit'], AUTHORITY)
        self.assertIs(self.record['m6_closed'], True)
        self.assertIs(self.record['m7_scientific_training_started'], False)
        self.assertIn(b'M6_CLOSED = true', at_m7a(self.pf.CONFIG_STATUS))
        self.assertIs(json.loads(at_m7a('outputs/audit/M6E_FINAL_M6_CLOSURE.json'))['m6']['M6_CLOSED'], True)

    def test_01_d01_closes_dev003(self):
        d = self.record['decisions']['D01']
        self.assertEqual((d['closes'], d['lambda_dir'], d['meaning']), ('DEV-003', 0.5, 'coefficient_of_S_orient_inside_L_spec'))
        self.rejects(lambda r: r['decisions']['D01'].update(standalone_directional_loss=True))
        self.rejects(lambda r: r['decisions']['D01'].update(inert_key=True))

    def test_02_d02_grl_plus_positive_ce(self):
        d = self.record['decisions']['D02']
        self.assertEqual((d['grl_alpha'], d['idadv_term_sign']), (1.0, '+'))
        self.rejects(lambda r: r['decisions']['D02'].update(idadv_term_sign='-'))
        self.rejects(lambda r: r['decisions']['D02'].update(objective_term='- lambda_idadv * CE(identity_head(GRL(z_a)), source_subject)'))

    def test_03_d03_masks_siw_never_fabricates(self):
        d = self.record['decisions']['D03']
        self.assertEqual((d['deviation'], d['identity_label'], d['identity_ce_masked']), ('DEV-022', 'source_subject', ['siwmv2']))
        self.assertEqual(d['zero_labelled_rows_in_step'], 'L_idadv = 0 exactly')
        self.rejects(lambda r: r['decisions']['D03'].update(datasets=['casia_fasd', 'msu_mfsd']))
        self.rejects(lambda r: r['decisions']['D03'].update(identity_label='video_id'))
        self.rejects(lambda r: r['decisions']['D03']['fabrication_forbidden'].remove('pseudo_subject_id'))

    def test_04_d04_b1_keeps_siw(self):
        self.assertIn('siwmv2', self.record['decisions']['D04']['datasets'])
        self.rejects(lambda r: r['decisions']['D04'].update(datasets=['casia_fasd', 'msu_mfsd']))

    def test_05_d05_exact_k6_order(self):
        self.assertEqual(self.record['decisions']['D05']['class_order'],
                         ['makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay'])
        self.rejects(lambda r: r['decisions']['D05'].update(class_order=['live', 'makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay']))
        self.rejects(lambda r: r['decisions']['D05'].update(apply_to_x_hat=True))
        self.rejects(lambda r: r['decisions']['D05']['head'].update(out_features=7))

    def test_06_d06_warmup(self):
        d = self.record['decisions']['D06']
        self.assertEqual((d['steps_per_epoch'], d['total_steps'], d['carry_optimizer_state']), (139, 1390, False))
        self.rejects(lambda r: r['decisions']['D06'].update(carry_optimizer_state=True))

    def test_07_d07_teachers_frozen(self):
        self.assertIs(self.record['decisions']['D07']['teachers_frozen'], True)
        self.rejects(lambda r: r['decisions']['D07']['preserved'].remove('teacher_weights'))
        self.rejects(lambda r: r['decisions']['D07'].update(teacher_precision='fp16'))

    def test_08_d08_coordinates_not_heatmaps(self):
        self.rejects(lambda r: r['decisions']['D08'].update(heatmaps=True))
        self.rejects(lambda r: r['decisions']['D08'].update(pixel_units_in_training_loss=True))

    def test_09_d09_no_invented_parser_names(self):
        d = self.record['decisions']['D09']
        self.assertIsNone(d['semantic_names'])
        self.rejects(lambda r: r['decisions']['D09'].update(semantic_names={'0': 'background', '1': 'skin'}))
        self.rejects(lambda r: r['decisions']['D09'].update(non_background=list(range(1, 10))))

    def test_10_d10_exact(self):
        self.assertEqual(self.result['d10_items'], 24)
        self.rejects(lambda r: r['decisions']['D10']['04_Normalize'].update(per_image_min_max=True))
        n = self.record['decisions']['D10']['04_Normalize']
        self.assertEqual((n['u'], n['A'], n['operator']), ('M * (0.5 * A_rgb + 0.5 * A_freq)', 'Normalize(u)',
                                                           'Normalize(u) = clip(u / 2.0, 0.0, 1.0)'))
        self.rejects(lambda r: r['decisions']['D10']['04_Normalize'].update(operator='Normalize(A) = clip(A_rgb / 2, 0, 1)'))
        self.rejects(lambda r: r['decisions']['D10']['04_Normalize'].update(u='0.5 * A_rgb'))
        self.rejects(lambda r: r['decisions']['D10']['04_Normalize'].update(u='0.5 * A_rgb + 0.5 * A_freq'))
        self.rejects(lambda r: r['decisions']['D10']['07_M_face_dilated'].update(kernel=[9, 9]))
        self.rejects(lambda r: r['decisions']['D10']['13_PatchGAN'].update(instance_norm_layers=[2]))
        self.rejects(lambda r: r['decisions']['D10']['20_x_hat_range'].update(clamp_during_training=True))

    def test_11_d11_d12_sanity_contracts(self):
        self.assertEqual(self.record['decisions']['D11']['pass_if_less_than'], 1e-5)
        self.rejects(lambda r: r['decisions']['D11'].update(metric='mean_abs'))
        self.rejects(lambda r: r['decisions']['D12'].update(artifact_scale_zero_disables=['delta_HF']))

    def test_12_d13_d14_d16_selection(self):
        self.rejects(lambda r: r['decisions']['D13'].update(artsim='GPAT F_art cosine'))
        self.rejects(lambda r: r['decisions']['D14'].update(test_participates=True))
        self.rejects(lambda r: r['decisions']['D14'].update(data='VAL_and_TEST'))
        self.rejects(lambda r: r['decisions']['D16'].update(evaluate_all_before_selecting=False))
        self.rejects(lambda r: r['decisions']['D16'].update(training_reads_val=True))
        self.rejects(lambda r: r['frozen_spec_facts']['selection_weights'].update(ID=0.30))

    def test_13_d15_1105_updates(self):
        a = self.pf.m7_arithmetic()
        self.assertEqual((a['microbatches_per_epoch'], a['optimizer_updates_per_epoch'], a['final_group_samples']),
                         (2210, 1105, 6))
        self.rejects(lambda r: r['decisions']['D15'].update(optimizer_updates_per_epoch=1104))
        self.rejects(lambda r: r['decisions']['D15'].update(drop_last=True))

    def test_14_only_d03_is_controlled_adaptation(self):
        self.rejects(lambda r: r['decisions']['D04'].update({'class': 'CONTROLLED_ADAPTATION'}))
        self.rejects(lambda r: r['decisions']['D01'].update({'class': 'NEW_SCIENCE'}))

    def test_15_historical_is_reference_only(self):
        h = self.result['historical']
        self.assertGreaterEqual(h['reconciliation_rows'], 20)
        chain = json.dumps(self.record['authority_chain'])
        self.assertNotIn('PRISM_FAS_C_LLM_Project', chain)
        self.rejects(lambda r: r['authority_chain'].update(historical_project_in_chain=True), 'check_historical')
        self.rejects(lambda r: r['historical_reference'].update(may_override_authority=True), 'check_historical')
        self.assertEqual(self.record['historical_reference']['classification'], 'HISTORICAL_IMPLEMENTATION_REFERENCE_ONLY')
        self.rejects(lambda r: r['historical_reference'].update(files_copied_in_m7a=1), 'check_historical')
        self.rejects(lambda r: r['historical_reference']['not_a_source_of'].remove('teacher_choices'), 'check_historical')
        self.rejects(lambda r: r['historical_reference']['historical_alternatives_for_future_ablation'][0].update(adopted=True),
                     'check_historical')

        def adopt_conflict(r):
            row = next(x for x in r['historical_reference']['reconciliation']
                       if x['classification'] == 'CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE')
            row['action'] = 'reuse'
        self.rejects(adopt_conflict, 'check_historical')
        self.rejects(lambda r: r['historical_reference']['implementation_references'][0].update(sha256='unknown'),
                     'check_historical')

    def test_16_nafnet_immutable_pin(self):
        n = self.record['nafnet_pin']
        self.assertRegex(n['commit'], r'^[0-9a-f]{40}$')
        self.assertNotIn('HEAD', n['commit'])
        self.rejects(lambda r: r['nafnet_pin'].update(commit='HEAD'), 'check_nafnet')
        self.rejects(lambda r: r['nafnet_pin'].update(weights_required=True), 'check_nafnet')

    def test_17_deviation_register(self):
        reg = self.result['registers']
        self.assertEqual(reg['new_dev_numbers'], ['DEV-022'])
        dev = at_m7a(self.pf.DEVIATIONS).decode()
        self.assertTrue(set(re.findall(r'\*\*Status:\*\* (\w+)', dev)) <= {'APPROVED', 'UNAPPROVED'})
        self.assertEqual(dev.count('### DEV-022 —'), 1)
        self.assertNotIn('DEV-022', at_authority(self.pf.DEVIATIONS).decode())
        for rel in (self.pf.DEVIATIONS, self.pf.CONFIG_STATUS):
            self.assertTrue(at_m7a(rel).startswith(at_authority(rel)), rel + ' append-only')

    def test_18_fair_track_clarification_a1_unchanged(self):
        f = self.record['fair_track_clarification']
        self.assertEqual(f['variants']['E09_B1']['siwmv2'], 'INCLUDED_ATTACK_MACRO_PRESENT')
        self.rejects(lambda r: r['fair_track_clarification'].update(pseudo_identity_permitted=True), 'check_fair_track')
        for rel in ('configs/frozen/fair_track_v1.yaml',
                    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md'):
            self.assertEqual(at_m7a(rel), at_authority(rel), rel)

    def test_19_m6_and_frozen_authority_unchanged(self):
        for rel in self.pf.PROTECTED:
            self.assertEqual(hashlib.sha256(at_m7a(rel)).hexdigest(), hashlib.sha256(at_authority(rel)).hexdigest(), rel)
        self.assertEqual(hashlib.sha256(at_m7a(self.pf.GPAT_B0)).hexdigest(), self.pf.GPAT_B0_SHA)
        self.assertEqual(self.result['bound'], 24)
        for rel in self.record['bound_authority_sha256']:
            if rel.startswith('docs/spec/amendments/'):
                self.assertEqual(at_m7a(rel), at_authority(rel), rel)

    def test_20_no_b1_b3_configs_no_gpat_code(self):
        self.pf.check_absent(exists_at_m7a, listdir_at_m7a)
        for rel in self.pf.GPAT_ABSENT:
            self.assertFalse(exists_at_m7a(rel), rel)
        self.assertEqual(listdir_at_m7a('methods/gpat'), ['.gitkeep'])

    def test_21_evidence_binds_candidate(self):
        ev = self.pf.check_evidence(at_m7a, self.result)
        for k in ('training_runs', 'checkpoint_writes', 'bank_writes'):
            self.assertEqual(ev[k], 0)
        self.assertIs(ev['historical_project']['modified_by_m7a'], False)

    def test_22_static(self):
        if not self.torch_before:
            self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
