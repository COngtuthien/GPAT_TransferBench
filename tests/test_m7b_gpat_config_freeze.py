"""M7B: GPAT B1/B2/B3 config freeze (E09/E10/E11) on B0 + A10 + owner clarifications R-01/R-02/R-03/R-06/R-07. Static only.

History assertions read the state at the commit that ADDED this test (candidate: the worktree) and compare it with the
M7A authority, so later milestones never break them (no HEAD lock; owner decision B1).
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = 'd59b9a302c61ed440e711abf6ffab9560458f779'
THIS = 'tests/test_m7b_gpat_config_freeze.py'
PREFLIGHT = 'tools/m7b_gpat_config_freeze_preflight.py'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m7b_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7b_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7b(rel):
    """File bytes at the M7B state: the commit that added this test, else the (candidate) worktree."""
    commit = m7b_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def exists_at_m7b(rel):
    commit = m7b_commit()
    return git('cat-file', '-e', f'{commit}:{rel}').returncode == 0 if commit else (ROOT / rel).exists()


def listdir_at_m7b(rel):
    commit = m7b_commit()
    if not commit:
        return sorted(p.name for p in (ROOT / rel).iterdir())
    return sorted(Path(p).name for p in git('ls-tree', '--name-only', f'{commit}:{rel}/').stdout.decode().split())


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM7BConfigFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.cfgs = cls.pf.load_configs(at_m7b)
        cls.a10 = json.loads(at_m7b(cls.pf.A10).decode())
        cls.rec = json.loads(at_m7b(cls.pf.CLARIFICATIONS).decode())
        cls.result = cls.pf.check_configs(at_m7b)

    def rejects(self, mutate, vid='B1', check='variant'):
        cfgs = copy.deepcopy(self.cfgs)
        mutate(cfgs[vid])
        with self.assertRaises((ValueError, KeyError, TypeError)):
            if check == 'variant':
                self.pf.check_variant(cfgs[vid], vid)
            elif check == 'shared':
                self.pf.check_shared(cfgs[vid], self.a10)
            elif check == 'inheritance':
                self.pf.check_inheritance(cfgs['B0'], cfgs[vid], vid)
            elif check == 'matrix':
                self.pf.comparison_matrix(cfgs)
            elif check == 'union':
                self.pf.check_union(cfgs)

    def rejects_record(self, mutate):
        rec = copy.deepcopy(self.rec)
        mutate(rec)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.pf.check_clarifications(rec)

    def test_00_authority_a10_and_m6_closed(self):
        self.assertEqual(self.pf.AUTHORITY, AUTHORITY)
        self.assertEqual(hashlib.sha256(at_m7b(self.pf.A10_DOC)).hexdigest(),
                         '7baf7888394c4cdd57db09e604bd910153a612e1765b8dc57de3c1d5d833d337')
        self.assertEqual(hashlib.sha256(at_m7b(self.pf.A10)).hexdigest(),
                         '57b37f09ca4279ad3999f4f9efa21fbf6fa39836574a0336bd13e34b2bd53203')
        self.assertIs(json.loads(at_m7b('outputs/audit/M6E_FINAL_M6_CLOSURE.json'))['m6']['M6_CLOSED'], True)
        added = at_m7b(self.pf.CONFIG_STATUS)[len(at_authority(self.pf.CONFIG_STATUS)):].decode()
        self.assertTrue(at_m7b(self.pf.CONFIG_STATUS).startswith(at_authority(self.pf.CONFIG_STATUS)))
        for phrase in ('CONFIGS_FROZEN', 'IMPLEMENTATION_NOT_STARTED', 'ENVIRONMENT_NOT_CREATED',
                       'SCIENTIFIC_TRAINING_NOT_STARTED', 'M6_CLOSED = true'):
            self.assertIn(phrase, added)

        def bad_a10(rel):
            raw = at_m7b(rel)
            return raw + b' ' if rel == self.pf.A10_DOC else raw
        with self.assertRaises(ValueError):
            self.pf.check_bindings(bad_a10)

    def test_01_b0_unchanged(self):
        for rel in (self.pf.GPAT_B0, self.pf.SNAP + self.pf.GPAT_B0):
            self.assertEqual(at_m7b(rel), at_authority(rel), rel)
            self.assertEqual(hashlib.sha256(at_m7b(rel)).hexdigest(), self.pf.GPAT_B0_SHA)

        def bad_b0(rel):
            raw = at_m7b(rel)
            return raw.replace(b'lambda_tv: 0.05', b'lambda_tv: 0.06') if rel == self.pf.GPAT_B0 else raw
        with self.assertRaises(ValueError):
            self.pf.check_bindings(bad_b0)

    def test_02_configs_exist_and_match_snapshots(self):
        for v in self.pf.VARIANTS:
            self.assertTrue(exists_at_m7b(self.pf.CONFIGS[v]))
            self.assertEqual(at_m7b(self.pf.CONFIGS[v]), at_m7b(self.pf.SNAPSHOTS[v]))
            self.assertFalse(at_authority(self.pf.CONFIGS[v]), 'absent at M7A authority')
            c = self.result['configs'][v]
            self.assertEqual(c['config_sha256'], c['snapshot_sha256'])

        def bad_snapshot(rel):
            raw = at_m7b(rel)
            return raw + b'\n' if rel == self.pf.SNAPSHOTS['B2'] else raw
        with self.assertRaises(ValueError):
            self.pf.check_configs(bad_snapshot)

    def test_03_experiment_ids_track_datasets(self):
        for v, e in (('B1', 'E09'), ('B2', 'E10'), ('B3', 'E11')):
            c = self.cfgs[v]
            self.assertEqual((c['variant']['experiment_id'], c['variant']['track'], c['method']), (e, 'B', 'GPAT-' + v))
            self.assertEqual(c['data']['datasets'], ['casia_fasd', 'msu_mfsd', 'siwmv2'])
        self.rejects(lambda c: c['variant'].update(experiment_id='E08'))
        self.rejects(lambda c: c['data'].update(datasets=['casia_fasd', 'msu_mfsd']))

    def test_04_exact_six_attack_classes(self):
        for v in ('B1', 'B3'):
            a = self.cfgs[v]['attack_type_supervision']
            self.assertEqual(a['class_order'], ['makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay'])
            self.assertEqual(a['head'], {'type': 'Linear', 'in_features': 512, 'out_features': 6, 'bias': True})
        self.rejects(lambda c: c['attack_type_supervision'].update(
            class_order=['live', 'makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay']))
        self.rejects(lambda c: c['attack_type_supervision'].update(class_order=['mask_2d', 'makeup', 'mask_3d', 'partial',
                                                                                'print', 'replay']))
        self.rejects(lambda c: c['attack_type_supervision'].update(class_weights=[1, 1, 1, 1, 1, 1]))
        self.rejects(lambda c: c['attack_type_supervision'].update(applied_to_x_hat=True))

    def test_05_supervision_matrix(self):
        for v, (lt, li) in self.pf.LAMBDAS.items():
            loss = self.cfgs[v]['loss']
            self.assertEqual((loss['lambda_type'], loss['lambda_idadv']), (lt, li), v)
        self.assertEqual(self.pf.LAMBDAS, {'B0': (0.0, 0.0), 'B1': (0.2, 0.0), 'B2': (0.0, 0.1), 'B3': (0.2, 0.1)})
        self.rejects(lambda c: c['loss'].update(lambda_type=0.1), check='inheritance')
        self.rejects(lambda c: c['loss'].update(lambda_idadv=0.1), check='inheritance')

    def test_06_only_intended_scientific_deltas(self):
        for v in self.pf.VARIANTS:
            self.assertEqual(set(self.result['configs'][v]['b0_deltas']), self.pf.B0_DELTAS[v])
        self.rejects(lambda c: c['loss'].update(lambda_spec=0.6), 'B3', 'inheritance')
        self.rejects(lambda c: c['training'].update(lr=1e-4), 'B2', 'inheritance')
        self.rejects(lambda c: c['residual_generator'].update(delta_scale_hf=0.2), 'B1', 'inheritance')
        self.rejects(lambda c: c['loss'].pop('lambda_bg'), 'B1', 'inheritance')
        self.rejects(lambda c: c['loss'].update(lambda_spec=0.6), 'B3', 'matrix')
        self.rejects(lambda c: c['shared_contract']['image'].update(resolution=224), 'B2', 'matrix')
        self.rejects(lambda c: c['identity_adversary'].update(lambda_idadv=0.2), 'B3', 'matrix')

    def test_07_comparison_matrix_fully_explained(self):
        rows = self.result['matrix']
        self.assertTrue(rows and all(r['class'] in self.pf.MATRIX_CLASSES for r in rows))
        b0_paths = {p for p, _ in self.pf.leaves(self.cfgs['B0'])}
        self.assertTrue(b0_paths <= {r['path'] for r in rows})
        cls = {r['path']: r['class'] for r in rows}
        self.assertEqual(cls['loss.lambda_type'], 'B1_DELTA')
        self.assertEqual(cls['loss.lambda_idadv'], 'B2_DELTA')
        self.assertEqual(cls['data.label_fields_used'], 'B3_UNION')
        self.assertEqual({p for p in b0_paths if cls[p] not in ('SAME', 'PROVENANCE_ONLY')},
                         {'artifact_encoder.attack_type_head', 'artifact_encoder.identity_adversary', 'loss.lambda_type',
                          'loss.lambda_idadv'})

    def test_08_b3_exact_union(self):
        b1, b2, b3 = self.cfgs['B1'], self.cfgs['B2'], self.cfgs['B3']
        self.assertEqual(b3['attack_type_supervision'], b1['attack_type_supervision'])
        self.assertEqual(b3['attack_warmup'], b1['attack_warmup'])
        self.assertEqual(b3['identity_adversary'], b2['identity_adversary'])
        self.rejects(lambda c: c['attack_warmup'].update(epochs=12), 'B3', 'union')
        self.rejects(lambda c: c['identity_adversary']['grl'].update(alpha=0.5), 'B3', 'union')

    def test_09_grl_positive_ce(self):
        for v in ('B2', 'B3'):
            i = self.cfgs[v]['identity_adversary']
            self.assertEqual(i['objective_term'], '+ lambda_idadv * CE(identity_head(GRL(z_a)), source_subject)')
            self.assertEqual((i['grl']['alpha'], i['idadv_term_sign']), (1.0, '+'))
        self.rejects(lambda c: c['identity_adversary'].update(idadv_term_sign='-'), 'B2')
        self.rejects(lambda c: c['identity_adversary'].update(
            objective_term='- lambda_idadv * CE(identity_head(GRL(z_a)), source_subject)'), 'B2')
        self.rejects(lambda c: c['identity_adversary']['grl'].update(alpha=-1.0), 'B3')
        self.rejects(lambda c: c['identity_adversary']['forbidden'].remove('GRL + (- lambda_idadv * CE)'), 'B2')

    def test_10_dev022_only_b2_b3_no_pseudo_ids(self):
        self.assertEqual(self.cfgs['B1']['provenance']['deviations'], [])
        self.assertNotIn(b'DEV-022', at_m7b(self.pf.CONFIGS['B1']))
        for v in ('B2', 'B3'):
            m = self.cfgs[v]['identity_adversary']['dev_022_masking']
            self.assertEqual((m['deviation'], m['zero_labelled_rows'], m['pseudo_identity']),
                             ('DEV-022', 'L_idadv = 0 exactly', 'FORBIDDEN'))
            self.assertEqual(self.cfgs[v]['identity_adversary']['label'], 'source_subject')
        self.rejects(lambda c: c['provenance'].update(deviations=['DEV-022']), 'B1')
        self.rejects(lambda c: c['identity_adversary'].update(note='DEV-022'), 'B1')
        self.rejects(lambda c: c['identity_adversary'].update(label='video_id'), 'B2')
        self.rejects(lambda c: c['identity_adversary']['dev_022_masking']['fabrication_forbidden'].remove(
            'pseudo_subject_id'), 'B3')
        self.rejects(lambda c: c['identity_adversary']['dev_022_masking'].update(pseudo_identity='CLUSTERING'), 'B2')
        self.rejects(lambda c: c['identity_adversary']['dev_022_masking'].update(masked_rows='none'), 'B2')
        self.rejects(lambda c: c['data'].update(pseudo_identity='ALLOWED'), 'B3')

    def test_11_r01_ptwt_mapping(self):
        w = self.cfgs['B1']['shared_contract']['wavelet_bands']
        self.assertEqual(w['mapping'], {'LL': 'cA', 'LH': 'cH', 'HL': 'cV', 'HH': 'cD'})
        self.assertEqual((w['axes'], w['coefficient_order']), ([-2, -1], '(cA, (cH, cV, cD))'))
        self.rejects(lambda c: c['shared_contract']['wavelet_bands'].update(
            mapping={'LL': 'cA', 'LH': 'cV', 'HL': 'cH', 'HH': 'cD'}), check='shared')
        self.rejects(lambda c: c['shared_contract']['wavelet_bands'].update(axes=[-3, -2]), check='shared')

    def test_12_r02_mask_and_d104(self):
        s = self.cfgs['B2']['shared_contract']
        self.assertEqual(s['mask']['artifact_map_resize'],
                         'M_256 = bilinear_interpolate(M, size=(256, 256), align_corners=False)')
        self.assertEqual((s['mask']['wavelet_domain_resize'], s['mask']['threshold']), ('FORBIDDEN', 'none'))
        self.assertEqual((s['artifact_map']['u'], s['artifact_map']['A']),
                         ('M_256 * (0.5 * A_rgb + 0.5 * A_freq_256)', 'clip(u / 2.0, 0.0, 1.0)'))
        self.rejects(lambda c: c['shared_contract']['mask'].update(wavelet_domain_resize='bilinear'), check='shared')
        self.rejects(lambda c: c['shared_contract']['mask'].update(
            artifact_map_resize='M_256 = nearest(M, size=(256, 256))'), check='shared')
        self.rejects(lambda c: c['shared_contract']['mask'].update(threshold=0.5), check='shared')
        self.rejects(lambda c: c['shared_contract']['artifact_map'].update(A='clip(A_rgb / 2, 0, 1)'), check='shared')
        self.rejects(lambda c: c['shared_contract']['artifact_map'].update(u='0.5 * A_rgb + 0.5 * A_freq_256'),
                     check='shared')
        self.rejects(lambda c: c['shared_contract']['artifact_map'].update(per_image_min_max=True), check='shared')

    def test_13_r03_float_val(self):
        vr = self.cfgs['B3']['shared_contract']['checkpoint_selection']['val_representation']
        self.assertEqual((vr['source'], vr['uint8_roundtrip']), ('x_hat_float', 'FORBIDDEN'))
        self.rejects(lambda c: c['shared_contract']['checkpoint_selection']['val_representation'].update(
            source='exported_uint8_png'), check='shared')
        self.rejects(lambda c: c['shared_contract']['checkpoint_selection']['val_representation'].update(
            uint8_roundtrip='ALLOWED'), check='shared')

    def test_14_r06_precision(self):
        p = self.cfgs['B1']['shared_contract']['precision']
        self.assertEqual(p['autocast'], {'device': 'cuda', 'dtype': 'float16'})
        self.assertIn('frozen_teacher_and_evaluator_forwards', p['fp32_outside_autocast'])
        self.rejects(lambda c: c['shared_contract']['precision'].update(autocast={'device': 'cuda', 'dtype': 'bfloat16'}),
                     check='shared')
        self.rejects(lambda c: c['shared_contract']['precision']['fp32_outside_autocast'].remove('DWT'), check='shared')
        self.rejects(lambda c: c['shared_contract']['precision'].update(x_hat_detach='ALLOWED'), check='shared')
        self.rejects(lambda c: c['shared_contract']['precision']['fp32_frozen_modules'].remove('FaceXFormer'),
                     check='shared')
        self.rejects(lambda c: c['attack_warmup']['amp'].update(ce_dtype='float16'))
        self.rejects(lambda c: c['attack_warmup']['grad_clip'].update(max_norm=None))

    def test_15_r07_optimizer_membership(self):
        o = self.cfgs['B2']['shared_contract']['optimizer_ownership']
        self.assertEqual(o['G_OPT_membership']['B2'], ['G_res', 'E_art', 'identity_adversary_head'])
        self.assertEqual(o['D_OPT']['params'], ['PatchGAN_D'])
        self.rejects(lambda c: c['shared_contract']['optimizer_ownership']['G_OPT_membership'].update(
            B2=['G_res', 'E_art']), check='shared')
        self.rejects(lambda c: c['shared_contract']['optimizer_ownership']['D_OPT'].update(
            params=['PatchGAN_D', 'identity_adversary_head']), check='shared')
        self.rejects(lambda c: c['identity_adversary'].update(optimizer='ID_ADV_OPT'), 'B3')
        self.rejects(lambda c: c['attack_warmup']['optimizer'].update(
            params=['E_art', 'attack_type_head', 'identity_adversary_head']), 'B3')
        self.rejects(lambda c: c['attack_warmup'].update(carry_optimizer_state=True), 'B1')

    def test_16_budget_counts(self):
        a = self.pf.m7_arithmetic()
        self.assertEqual((a['microbatches_per_epoch'], a['optimizer_updates_per_epoch'], a['generator_updates_per_run'],
                          a['tail_group_samples']), (2210, 1105, 66300, 6))
        self.assertEqual((a['warmup_steps_per_epoch'], a['warmup_total_steps']), (139, 1390))
        self.rejects(lambda c: c['shared_contract']['training_budget'].update(optimizer_updates_per_epoch=1104),
                     check='shared')
        self.rejects(lambda c: c['shared_contract']['training_budget'].update(drop_last=True), check='shared')
        self.rejects(lambda c: c['attack_warmup'].update(reduces_generator_epochs=True))

    def test_17_selection_51_val_test_forbidden(self):
        c = self.cfgs['B1']['shared_contract']['checkpoint_selection']
        self.assertEqual((c['candidates']['N'], c['data'], c['test_participates']), (51, 'VAL_only', False))
        self.rejects(lambda x: x['shared_contract']['checkpoint_selection'].update(test_participates=True), check='shared')
        self.rejects(lambda x: x['shared_contract']['checkpoint_selection']['candidates'].update(N=50), check='shared')
        self.rejects(lambda x: x['shared_contract']['checkpoint_selection'].update(training_process_reads_val=True),
                     check='shared')
        for v in self.pf.VARIANTS:
            t = self.cfgs[v]['data']['splits']['TEST']
            self.assertEqual((t['allowed'], t['used_for']), (False, []))
        self.rejects(lambda x: x['data']['splits']['TEST'].update(allowed=True), 'B2')
        self.rejects(lambda x: x['data']['splits']['TEST'].update(used_for=['checkpoint_selection']), 'B3')

    def test_18_nafnet_pin_and_environment(self):
        e = self.cfgs['B1']['shared_contract']['environment']
        self.assertEqual(e['nafnet']['commit'], '2b4af71ebe098a92a75910c233a3965a3e93ede4')
        self.assertEqual(e['status'], 'NOT_CREATED_M7B')
        self.rejects(lambda c: c['shared_contract']['environment']['nafnet'].update(commit='main'), check='shared')
        self.rejects(lambda c: c['shared_contract']['environment'].update(status='CREATED'), check='shared')
        self.assertEqual([n for n in listdir_at_m7b('environments') if 'gpat' in n.lower()], [])

    def test_19_no_gpat_code_runs_checkpoints(self):
        self.pf.check_absent(exists_at_m7b, listdir_at_m7b)
        self.assertEqual(listdir_at_m7b('methods/gpat'), ['.gitkeep'])
        with self.assertRaises(ValueError):
            self.pf.check_absent(exists_at_m7b, lambda rel: ['.gitkeep', 'model.py'] if rel == 'methods/gpat'
                                 else listdir_at_m7b(rel))
        with self.assertRaises(ValueError):
            self.pf.check_absent(lambda rel: rel == 'runs/m7' or exists_at_m7b(rel), listdir_at_m7b)

    def test_20_clarification_record(self):
        c = self.pf.check_clarifications(self.rec)
        self.assertEqual((c['resolved'], c['deferred_to_m7c']), (['R-01', 'R-02', 'R-03', 'R-06', 'R-07'], ['R-04', 'R-05']))
        self.rejects_record(lambda r: r['residual_items']['R-04'].update(status='RESOLVED_M7B'))
        self.rejects_record(lambda r: r['residual_items']['R-05'].update(decision={'x': 1}))
        self.rejects_record(lambda r: r['residual_items'].pop('R-03'))
        self.rejects_record(lambda r: r['residual_items']['R-01']['decision'].update(
            mapping={'LL': 'cA', 'LH': 'cV', 'HL': 'cH', 'HH': 'cD'}))
        self.rejects_record(lambda r: r['status'].remove('NOT_FROZEN_SPEC_FACTS'))
        self.rejects_record(lambda r: r.update(a10_edited=True))
        for v in self.pf.VARIANTS:
            self.assertIs(self.cfgs[v]['provenance']['r_items_are_frozen_spec_facts'], False)

    def test_21_protected_authority_unchanged(self):
        for rel in self.pf.PROTECTED:
            self.assertEqual(hashlib.sha256(at_m7b(rel)).hexdigest(), hashlib.sha256(at_authority(rel)).hexdigest(), rel)
        for rel in ('outputs/audit/method_status.csv', 'configs/frozen/fair_track_v1.yaml'):
            self.assertEqual(at_m7b(rel), at_authority(rel), rel)

    def test_22_evidence_binds_candidate(self):
        ev = self.pf.check_evidence(at_m7b, self.result)
        for k in ('training_runs', 'checkpoint_writes', 'bank_writes'):
            self.assertEqual(ev[k], 0)
        self.assertEqual(ev['loader_or_validator_changes'], [])

    def test_24_obs01_resolved_from_frozen_spec(self):
        obs = self.rec['observations']['M7B-OBS-01']
        self.assertEqual((obs['status'], obs['class']), ('RESOLVED_FROM_FROZEN_SPEC', 'FROZEN_SPEC_DERIVED_EXECUTION_POLICY'))
        self.assertNotIn('open_observations_not_resolved', self.rec)
        for v in self.pf.VARIANTS:
            self.assertNotIn(b'OPEN_FOR_OWNER', at_m7b(self.pf.CONFIGS[v]))
        self.rejects_record(lambda r: r['observations']['M7B-OBS-01'].update(status='OPEN_FOR_OWNER'))
        self.rejects_record(lambda r: r.update(open_observations_not_resolved={'M7B-OBS-01': {}}))

    def test_25_discriminator_contract(self):
        op = self.cfgs['B2']['shared_contract']['optimizer_ownership']
        self.assertEqual(op['D_OPT'], {'name': 'Adam', 'lr': 2e-4, 'betas': [0.5, 0.999], 'weight_decay': 0.0,
                                       'params': ['PatchGAN_D'], 'separate': True})
        de = op['D_execution']
        self.assertEqual((de['lr_schedule']['warmup_epochs'], de['lr_schedule']['peak_lr'], de['lr_schedule']['min_lr'],
                          de['lr_schedule']['decay']), (5, 2e-4, 2e-6, 'cosine'))
        self.assertEqual(de['amp'], {'autocast': 'cuda_fp16', 'modules': ['PatchGAN_D']})
        self.assertEqual((de['grad_scaler'], de['grad_clip']['max_norm'], de['grad_clip']['after']),
                         ('D_SCALER', 1.0, 'D_SCALER.unscale_(D_OPT)'))
        sh = lambda f: self.rejects(lambda c: f(c['shared_contract']['optimizer_ownership']), check='shared')
        sh(lambda o: o['D_OPT'].update(lr=1e-4))
        sh(lambda o: o['D_OPT'].update(betas=[0.9, 0.999]))
        sh(lambda o: o['D_execution']['lr_schedule'].update(min_lr=0.0))
        sh(lambda o: o['D_execution']['lr_schedule'].update(warmup_epochs=0))
        sh(lambda o: o['D_execution']['amp'].update(autocast='none'))
        sh(lambda o: o['D_execution'].update(grad_scaler='G_SCALER'))
        sh(lambda o: o['D_execution']['grad_clip'].update(max_norm=None))
        sh(lambda o: o['D_execution']['grad_clip'].update(params='G_and_D'))
        sh(lambda o: o['D_execution'].update(status='OPEN_FOR_OWNER'))

    def test_26_separate_gradscalers(self):
        gs = self.cfgs['B1']['shared_contract']['precision']['grad_scalers']
        self.assertEqual((gs['G_SCALER']['owns'], gs['D_SCALER']['owns'], gs['WARMUP_SCALER']['owns']),
                         ('G_OPT', 'D_OPT', 'WARMUP_OPT'))
        self.assertIs(gs['independent_state_machines'], True)
        for v in ('B1', 'B3'):
            self.assertEqual(self.cfgs[v]['attack_warmup']['amp']['scaler'], 'WARMUP_SCALER')
        sc = lambda f: self.rejects(lambda c: f(c['shared_contract']['precision']['grad_scalers']), check='shared')
        sc(lambda g: g['D_SCALER'].update(owns='G_OPT'))
        sc(lambda g: g.update(independent_state_machines=False))
        sc(lambda g: g.update(warmup_state_reused_by_G_SCALER=True))
        sc(lambda g: g['per_optimizer_sequence'].reverse())
        self.rejects(lambda c: c['attack_warmup']['amp'].update(scaler='G_SCALER'), 'B3')
        self.rejects(lambda c: c['attack_warmup']['amp'].update(scaler_state_carried_to_generator_stage=True), 'B1')
        self.rejects(lambda c: c['shared_contract']['precision']['grad_clip'].update(applies_to=['G_OPT']), check='shared')

    def test_27_identity_bias_and_val_no_clamp(self):
        for v in ('B2', 'B3'):
            i = self.cfgs[v]['identity_adversary']
            self.assertIs(i['head']['bias'], True)
            self.assertEqual(i['head_bias_class'], 'OWNER_IMPLEMENTATION_CLARIFICATION')
        self.rejects(lambda c: c['identity_adversary']['head'].update(bias=False), 'B2')
        vr = self.cfgs['B1']['shared_contract']['checkpoint_selection']['val_representation']
        self.assertEqual((vr['policy'], vr['clamp_for_val_scoring'], vr['uint8_roundtrip'], vr['export_clamp_roundtrip']),
                         ('VAL_PRE_EXPORT_FLOAT_NO_CLAMP', False, 'FORBIDDEN', 'FORBIDDEN'))
        va = lambda f: self.rejects(lambda c: f(c['shared_contract']['checkpoint_selection']['val_representation']),
                                    check='shared')
        va(lambda r: r.update(clamp_for_val_scoring=True))
        va(lambda r: r.update(export_clamp_roundtrip='ALLOWED'))
        va(lambda r: r.update(encoded_image_roundtrip='ALLOWED'))
        self.assertEqual(self.cfgs['B3']['shared_contract']['wavelet_bands']['output_channels'],
                         {'delta_LL': [0, 1, 2], 'delta_LH': [3, 4, 5], 'delta_HL': [6, 7, 8], 'delta_HH': [9, 10, 11],
                          'mask_logit': [12]})

    def test_23_static(self):
        if not self.torch_before:
            self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
