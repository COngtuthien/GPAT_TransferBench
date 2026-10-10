"""M7C2a: GPAT implementation/runtime contract freeze, CPU environment and pinned NAFNet source qualification.

Static tests need only the stdlib, PyYAML and NumPy. Live tests need torch + ptwt (run them in gpat-m7-cpu);
they skip, visibly, anywhere else. History assertions read the state at the commit that ADDED this test
(candidate: the worktree), so later milestones never break them. No GPU, no dataset sample, no training.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '5f04912b652ae93f72ef6fc2dcdaa41dea55760d'
THIS = 'tests/test_m7c2a_gpat_runtime_contract.py'
RECORD = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'
EVIDENCE = 'outputs/audit/M7C2A_GPAT_CPU_QUALIFICATION.json'
LOCK = 'environments/gpat_m7_cpu.lock.json'
FREEZE = 'environments/gpat_m7_cpu.pip-freeze.txt'
BASE_FREEZE = 'environments/m6_core_gpu.pip-freeze.txt'
PINS = 'third_party/source_pins.json'
CONFIG_SHA = {'configs/methods/gpat_b0.yaml': '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
              'configs/methods/gpat_b1.yaml': '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc',
              'configs/methods/gpat_b2.yaml': '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9',
              'configs/methods/gpat_b3.yaml': '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'}
NAF_SHA = {'basicsr/models/archs/NAFNet_arch.py': '01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967',
           'basicsr/models/archs/arch_util.py': '5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005',
           'basicsr/models/archs/local_arch.py': 'c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26',
           'LICENSE': 'a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad'}


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m7c2a_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m7c2a(rel):
    commit = m7c2a_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def has_live_stack():
    return all(importlib.util.find_spec(m) is not None for m in ('torch', 'ptwt', 'pywt', 'cv2', 'PIL'))


LIVE = has_live_stack()


def load_tool():
    spec = importlib.util.spec_from_file_location('m7c2a_cpu_qualification', ROOT / 'tools/m7c2a_gpat_cpu_qualification.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticM7C2A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = json.loads(at_m7c2a(RECORD).decode())
        cls.res = cls.rec['resolutions']
        from methods.gpat import runtime_contract
        cls.rc = runtime_contract

    # ------------------------------------------------------------------ authority, immutability, record
    def test_00_record_bindings_and_immutability(self):
        self.assertEqual(self.rec['authority_commit'], AUTHORITY)
        for rel, digest in self.rec['bound_authority_sha256'].items():
            self.assertEqual(sha(at_m7c2a(rel)), digest, rel)
            self.assertEqual(sha(at_authority(rel)), digest, 'unchanged since M7B: ' + rel)
        for rel, digest in CONFIG_SHA.items():
            self.assertEqual(sha(at_m7c2a(rel)), digest, rel)
            self.assertEqual(at_m7c2a('frozen_config_snapshot/' + rel), at_m7c2a(rel), 'snapshot ' + rel)
        for flag in ('a10_edited', 'm7b_record_edited', 'frozen_spec_edited', 'gpat_configs_edited'):
            self.assertIs(self.rec[flag], False, flag)
        self.assertIsNone(self.rec['new_deviation'])
        self.assertIn('NOT_A_NEW_SCIENTIFIC_VARIANT', self.rec['status'])
        self.assertIn('NO_NEW_DEVIATION', self.rec['status'])
        self.assertEqual(set(self.res), {'R-04', 'R-05', 'N-01', 'N-03', 'N-04', 'N-05', 'N-06', 'N-07',
                                         'ADVERSARIAL_BCE', 'IDENTITY_CLASS_ORDER', 'N-08', 'N-09', 'SHAPE_TRACE'})

    def test_01_protected_authority_unchanged(self):
        for rel in ('docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx',
                    'configs/amendments/gpat_a10_m7_contract_resolution.yaml',
                    'configs/amendments/gpat_m7b_owner_clarifications.yaml',
                    'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md',
                    'outputs/audit/method_status.csv', 'outputs/audit/M6E_FINAL_M6_CLOSURE.json',
                    'outputs/audit/STAGE_STATE.json', 'outputs/audit/deviation_report.md', 'models/registry.yaml',
                    'third_party/registry.yaml', 'configs/frozen/fair_track_v1.yaml', *CONFIG_SHA):
            self.assertEqual(at_m7c2a(rel), at_authority(rel), rel)
        for i in range(1, 10):
            for p in (ROOT / 'docs/spec/amendments').glob(f'*Amendment_A{i}_*'):
                rel = str(p.relative_to(ROOT))
                self.assertEqual(at_m7c2a(rel), at_authority(rel), rel)

    # ------------------------------------------------------------------ R-04
    def test_02_r04_adapters_and_level1_gates(self):
        r = self.res['R-04']
        self.assertEqual(r['class'], 'RUNTIME_COMPATIBILITY')
        self.assertEqual(r['refines'], 'A10 D07')
        fx = r['facexformer_adapter']
        self.assertEqual(fx['name'], 'CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY')
        self.assertEqual(fx['pass_order'], 'horizontal_then_vertical')
        self.assertIn('no integer rounding, no uint8 conversion', fx['steps'])
        self.assertIn('piecewise-linear', fx['properties'])
        self.assertEqual(r['adaface_adapter']['name'], 'AREA_MATRIX_DIFFERENTIABLE_COMPATIBILITY')
        gates = {k: v['threshold'] for k, v in r['level1']['gates'].items()}
        self.assertEqual(gates, {'facexformer_matrix_vs_library_float_operator_max_abs': 1e-6,
                                 'facexformer_clip_emulating_vs_frozen_pil_uint8_max_lsb': 1.13,
                                 'facexformer_clip_emulating_vs_frozen_normalized_max_abs': 0.0198,
                                 'adaface_matrix_vs_library_float_operator_max_abs': 1e-6,
                                 'adaface_adapter_vs_frozen_cv2_uint8_max_abs': 0.5 / 127.5 + 1e-6,
                                 'highpass_float_path_max_abs': 1e-6,
                                 'artifact_probe_float_preprocessing_max_abs': 1e-6})
        self.assertEqual(r['level1']['diagnostics_not_thresholds'], ['mean_abs', 'rmse', 'p99_abs'])
        self.assertEqual(len(r['level1']['gradient_structure']), 4)

    def test_03_r04_level2_deferred_contract(self):
        l2 = self.res['R-04']['level2']
        self.assertIs(l2['executed_in_m7c2a'], False)
        self.assertIs(l2['thresholds_invented_now'], False)
        self.assertEqual(l2['splits'], {'TRAIN': True, 'VAL': False, 'TEST': False})
        self.assertEqual((l2['subset']['per_dataset_max'], l2['subset']['total_target']), (64, 192))
        self.assertEqual(l2['subset']['datasets'], ['casia_fasd', 'msu_mfsd', 'siwmv2'])
        self.assertEqual(l2['requantization_replicas']['count_per_image'], 8)
        self.assertEqual(set(l2['envelope_rules']), {'adaface', 'facexformer_landmarks', 'facexformer_parsing'})

    def test_04_cpu_evidence_gates(self):
        ev = json.loads(at_m7c2a(EVIDENCE).decode())
        self.assertEqual(ev['status'], 'PASS')
        self.assertIs(ev['level2_executed'], False)
        self.assertIs(ev['gpu_used'], False)
        self.assertEqual(ev['dataset_images_read'], 0)
        gates = {k: v['threshold'] for k, v in self.res['R-04']['level1']['gates'].items()}
        l1 = ev['r04_level1']
        self.assertEqual(l1['gates'], gates)
        for k, g in gates.items():
            self.assertLessEqual(l1['measured'][k], g, k)
            self.assertIs(l1['pass'][k], True, k)
        self.assertIs(l1['repeat_bitwise_identical'], True)
        for case in ev['r04_gradient_structure']['cases'].values():
            self.assertTrue(case['input_grad_finite'] and case['teacher_param_grads_none'] and case['targets_detached'])
            self.assertGreater(case['input_grad_abs_sum'], 0.0)
        self.assertLess(ev['ptwt']['D11_max_abs'], 1e-5)
        self.assertEqual(ev['environment']['python'], '3.11.16')
        self.assertIs(ev['environment']['cuda_available'], False)

    # ------------------------------------------------------------------ R-05
    def test_05_r05_pin_and_symbols(self):
        r = self.res['R-05']
        self.assertEqual((r['option'], r['pinned_commit']), ('D', '2b4af71ebe098a92a75910c233a3965a3e93ede4'))
        self.assertEqual(r['files_sha256'], NAF_SHA)
        self.assertEqual(set(r['required_symbols']), {'SimpleGate', 'NAFBlock', 'LayerNormFunction', 'LayerNorm2d'})
        self.assertEqual(r['hand_transcription'], 'FORBIDDEN')
        self.assertIn('verbatim', r['ctx_saved_variables'])
        pins = json.loads(at_m7c2a(PINS).decode())
        old = json.loads(at_authority(PINS).decode())
        self.assertEqual({k: v for k, v in pins['sources'].items() if k != 'nafnet'}, old['sources'])
        self.assertEqual({k: v for k, v in pins.items() if k != 'sources'}, {k: v for k, v in old.items() if k != 'sources'})
        nf = pins['sources']['nafnet']
        self.assertEqual(nf['pinned_commit'], r['pinned_commit'])
        self.assertEqual(nf['commit_tree'], r['commit_tree'])
        self.assertEqual({k: v['sha256'] for k, v in nf['cited_files'].items()}, NAF_SHA)
        self.assertEqual(nf['method_ids'], ['E08', 'E09', 'E10', 'E11'])
        self.assertIs(nf['model_weights_downloaded'], False)
        raw = at_m7c2a(PINS)
        self.assertEqual(raw, (json.dumps(pins, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode())
        from methods.gpat import naf_source
        self.assertEqual(naf_source.FILES_SHA256, NAF_SHA)
        self.assertEqual([n for _, n in naf_source.SYMBOLS], ['LayerNormFunction', 'LayerNorm2d', 'SimpleGate', 'NAFBlock'])
        self.assertFalse((ROOT / 'methods/gpat/basicsr').exists())

    def test_06_ast_selected_symbols_verbatim(self):
        from methods.gpat import naf_source
        try:
            verified = naf_source.verify_source()
        except naf_source.NAFSourceError as exc:
            self.skipTest(f'pinned source cache absent on this host: {exc}')
        seg = naf_source.source_segments(verified['files'])
        self.assertEqual({k: (v['first_line'], v['last_line']) for k, v in seg.items()},
                         {'LayerNormFunction': (264, 289), 'LayerNorm2d': (291, 300),
                          'SimpleGate': (22, 25), 'NAFBlock': (27, 80)})
        self.assertIn('ctx.saved_variables', seg['LayerNormFunction']['source'])
        for name, s in seg.items():
            text = verified['files'][s['file']].decode()
            self.assertIn(s['source'], text, name)

    # ------------------------------------------------------------------ N-items
    def test_07_n01_hp_domain_and_n03_mask(self):
        n1 = self.res['N-01']
        self.assertEqual(n1['applies_to'], ['E_art', 'F_art', 'PatchGAN_D'])
        self.assertEqual((n1['kernel'], n1['sigma'], n1['per_rgb_channel']), (9, 1.5, True))
        self.assertIs(n1['imagenet_normalization'], False)
        self.assertIs(n1['unit_interval_remap_before_hp'], False)
        self.assertIn('[-1, 1]', n1['input'])
        self.assertTrue(n1['artifact_probe'].startswith('UNCHANGED'))
        self.assertEqual(n1['class'], 'OWNER_SCIENTIFIC_CLARIFICATION_OF_UNSPECIFIED_INPUT_DOMAIN')
        n3 = self.res['N-03']
        self.assertEqual(n3['resize'], "torch F.interpolate(mode='nearest-exact') to 256x256")
        self.assertFalse(n3['antialias'] or n3['bilinear'] or n3['class_interpolation'])

    def test_08_n04_n05_n06_order_and_state(self):
        n4 = self.res['N-04']
        self.assertEqual(n4['scheme'], 'PRE_UPDATE_JOINT_GRADIENT')
        self.assertEqual(n4['boundary_sequence'][:8], ['D_SCALER.unscale_(D_OPT)', 'clip_grad_norm_(D parameters, 1.0)',
                                                       'D_SCALER.step(D_OPT)', 'D_SCALER.update()',
                                                       'G_SCALER.unscale_(G_OPT)', 'clip_grad_norm_(G_OPT parameters, 1.0)',
                                                       'G_SCALER.step(G_OPT)', 'G_SCALER.update()'])
        self.assertIs(n4['g_reforward_through_updated_d'], False)
        self.assertEqual(n4['alternating_extra_forward_pix2pix'], 'FORBIDDEN')
        self.assertEqual(self.res['N-05']['d_trains_from_generator_update'], 1)
        self.assertEqual(self.rc.D_FIRST_UPDATE, 1)
        self.assertEqual(self.rc.curriculum(1)['lambda_adv'], 0.0)
        n6 = self.res['N-06']
        self.assertEqual(n6['sync_batchnorm'], 'FORBIDDEN')
        self.assertEqual(n6['ema_scope'], ['E_art', 'G_res'])
        self.assertEqual(n6['ema_evaluation_mode'], 'eval')

    def test_09_lr_schedules_exact(self):
        rc = self.rc
        self.assertEqual(rc.main_lr(1), 0.0)
        self.assertAlmostEqual(rc.main_lr(5525), 2e-4, delta=1e-18)
        self.assertAlmostEqual(rc.main_lr(5526), 2e-4, delta=1e-18)
        self.assertAlmostEqual(rc.main_lr(66300), 2e-6, delta=1e-18)
        self.assertAlmostEqual(rc.main_lr(2), 2e-4 / 5524, delta=1e-20)
        t = (33000 - 5526) / (66300 - 5526)
        self.assertEqual(rc.main_lr(33000), 2e-6 + 0.5 * (2e-4 - 2e-6) * (1 + math.cos(math.pi * t)))
        self.assertTrue(all(rc.main_lr(u) >= rc.main_lr(u + 1) for u in range(5526, 66300, 997)))
        for bad in (0, 66301, 1.0, True):
            with self.assertRaises(ValueError):
                rc.main_lr(bad)
        self.assertEqual(rc.attack_warmup_lr(1), 1e-4)
        self.assertAlmostEqual(rc.attack_warmup_lr(1390), 0.0, delta=1e-20)
        with self.assertRaises(ValueError):
            rc.attack_warmup_lr(1391)
        n7 = self.res['N-07']
        self.assertEqual(n7['main_lr']['endpoints'], {'1': 0.0, '5525': 0.0002, '5526': 0.0002, '66300': 2e-06})
        self.assertEqual(n7['attack_warmup_lr']['endpoints'], {'1': 0.0001, '1390': 0.0})
        self.assertEqual(n7['main_lr']['applies_to'], ['G_OPT', 'D_OPT'])

    def test_10_curriculum_boundaries(self):
        # M7D1-A1 (N6A): these pins encode the ORIGINAL v1.0 curriculum (stage-2 lambda_adv jump 0 -> 0.05 at u5526),
        # now the historical `curriculum_v1_0`; production `curriculum` differs from it only by the A1 lambda_adv ramp
        # (tests/test_m7d1_a1_adversarial_curriculum.py).
        rc = self.rc
        self.assertEqual(rc.curriculum(5526)['lambda_adv'], 0.0)                 # A1: the ramp starts at exactly 0
        self.assertEqual(rc.curriculum_v1_0(1)['s_hf'], 0.02)
        self.assertAlmostEqual(rc.curriculum_v1_0(5525)['s_hf'], 0.05, delta=1e-15)
        self.assertEqual({k: rc.curriculum_v1_0(5525)[k] for k in ('lambda_adv', 'lambda_con', 'lambda_spec')},
                         {'lambda_adv': 0.0, 'lambda_con': 0.5, 'lambda_spec': 0.25})
        self.assertEqual(rc.curriculum_v1_0(5526), {'stage': 2, 's_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0,
                                                      'lambda_spec': 0.5})
        self.assertEqual(rc.curriculum_v1_0(16575)['stage'], 2)
        self.assertEqual(rc.curriculum_v1_0(16576), {'stage': 3, 's_hf': 0.15, 'lambda_adv': 0.10, 'lambda_con': 1.0,
                                                       'lambda_spec': 0.5})
        self.assertEqual(rc.curriculum_v1_0(66300)['stage'], 3)
        stages = self.res['N-07']['curriculum']['stages']
        self.assertEqual([s['updates'] for s in stages], [[1, 5525], [5526, 16575], [16576, 66300]])
        self.assertEqual(rc.UPDATES_PER_EPOCH * 60, rc.TOTAL_UPDATES)
        self.assertEqual(rc.group_weights([4, 2]), [4 / 6, 2 / 6])

    def test_11_loss_and_spectrum_definitions_recorded(self):
        n7 = self.res['N-07']
        sp = n7['soft_parsing_losses']
        self.assertEqual((sp['dice_classes'], sp['dice_eps'], sp['kl_probability_floor']), (list(range(1, 11)), 1e-6, 1e-8))
        self.assertEqual(sp['kl_direction'], 'KL(p_t || p_h)')
        self.assertEqual(sp['L_parse'], 'L_dice + 0.1 * L_KL')
        sr, so = n7['S_radial'], n7['S_orient']
        self.assertEqual((sr['bins'], sr['radius'], sr['transform']), (128, 'r = floor(sqrt(dx^2 + dy^2))', 'logP = log1p(P)'))
        self.assertIn('DC bin r = 0 INCLUDED', sr['bin_range'])
        self.assertEqual((so['bins'], so['exclude_dc']), (8, True))
        self.assertIn('NOT log power', so['power'])
        self.assertEqual(self.res['ADVERSARIAL_BCE']['d_fake_input'], 'x_hat.detach()')
        self.assertEqual(self.res['ADVERSARIAL_BCE']['g_input'], 'x_hat (NOT detached)')
        self.assertIn('M7C2A-OBS-01', self.rec['residual_open_items'])

    def test_12_identity_order_algorithm(self):
        rc = self.rc
        rows = [('casia_fasd', str(i)) for i in range(1, 36)] + [('msu_mfsd', f'{i:03d}') for i in range(1, 26)]
        m = rc.identity_class_map(list(reversed(rows)) + rows[:5], 'a' * 64)
        subjects = [(c['dataset'], c['source_subject']) for c in m['classes']]
        self.assertEqual(len(subjects), 60)
        self.assertEqual(subjects[:4], [('casia_fasd', '1'), ('casia_fasd', '10'), ('casia_fasd', '11'),
                                        ('casia_fasd', '12')])      # UTF-8 lexicographic, never natural
        self.assertEqual(subjects[35], ('msu_mfsd', '001'))
        self.assertEqual([c['index'] for c in m['classes']], list(range(60)))
        self.assertEqual(m, rc.identity_class_map(rows, 'a' * 64))
        with self.assertRaises(rc.IdentityMapError):
            rc.identity_class_map(rows + [('siwmv2', 'x')], 'a' * 64)
        with self.assertRaises(rc.IdentityMapError):
            rc.identity_class_map(rows[:-1], 'a' * 64)
        with self.assertRaises(rc.IdentityMapError):
            rc.identity_class_map(rows[:-1] + [('msu_mfsd', 25)], 'a' * 64)
        io = self.res['IDENTITY_CLASS_ORDER']
        self.assertIs(io['train_data_read_in_m7c2a'], False)
        self.assertEqual(io['shared_by'], ['B2', 'B3'])

    def test_13_n08_n09_shape_trace(self):
        n8 = self.res['N-08']
        self.assertEqual(n8['class'], 'NUMERICALLY_STABLE_EQUIVALENT_IMPLEMENTATION')
        self.assertEqual(n8['re_dwt_of_x_hat'], 'FORBIDDEN (would only measure fp32 reconstruction noise)')
        self.assertIs(n8['lferr_removed'], False)
        n9 = self.res['N-09']
        self.assertEqual(n9['gpu_qualification'], {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'cudnn_benchmark': False,
                                                   'cudnn_deterministic': True, 'tf32_matmul': False,
                                                   'tf32_cudnn': False, 'use_deterministic_algorithms': True})
        self.assertEqual(n9['silent_disable_on_failure'], 'FORBIDDEN')
        self.assertIs(n9['adopted_in_m7c2a'], False)
        st = self.res['SHAPE_TRACE']
        self.assertIs(st['bottleneck_film'], False)
        self.assertEqual(self.rc.SHAPE_TRACE['bottleneck_film'], False)
        self.assertIs(st['scientific_change'], False)

    # ------------------------------------------------------------------ environment and scope
    def test_14_environment_lock(self):
        lock = json.loads(at_m7c2a(LOCK).decode())
        self.assertEqual(lock['environment_name'], 'gpat-m7-cpu')
        self.assertEqual(lock['versions'], {'python': '3.11.16', 'torch': '2.12.1+cu130', 'torchvision': '0.27.1+cu130',
                                            'numpy': '2.4.6', 'Pillow': '12.3.0', 'opencv': '5.0.0',
                                            'opencv_python_headless': '5.0.0.93', 'ptwt': '1.0.1',
                                            'PyWavelets': '1.9.0', 'pip': '26.2.1'})
        self.assertEqual(lock['freeze_diff_vs_base'], {'added': ['PyWavelets==1.9.0', 'ptwt==1.0.1'], 'removed': [],
                                                       'changed': []})
        base = at_m7c2a(BASE_FREEZE).decode().splitlines()
        new = at_m7c2a(FREEZE).decode().splitlines()
        self.assertEqual(sorted(set(new) - set(base)), ['PyWavelets==1.9.0', 'ptwt==1.0.1'])
        self.assertEqual(set(base) - set(new), set())
        for rel, digest in lock['capture_sha256'].items():
            self.assertEqual(sha(at_m7c2a(rel)), digest, rel)
        for name, add in lock['approved_additions'].items():
            self.assertEqual(add['sha256'], add['pypi_sha256'], name)
        self.assertIs(lock['existing_environments_mutated'], False)
        self.assertIs(lock['gpu_environment_created'], False)

    def test_15_no_full_gpat_core(self):
        commit = m7c2a_commit()
        if commit:
            names = sorted(Path(p).name for p in git('ls-tree', '--name-only', f'{commit}:methods/gpat/').stdout.decode().split())
        else:
            names = sorted(p.name for p in (ROOT / 'methods/gpat').iterdir() if p.name != '__pycache__')
        self.assertEqual(names, ['.gitkeep', '__init__.py', 'naf_source.py', 'runtime_contract.py', 'teacher_preprocess.py'])
        for forbidden in ('artifact_encoder.py', 'generator.py', 'discriminator.py', 'losses.py', 'model.py', 'train.py'):
            self.assertNotIn(forbidden, names)

    def test_17_final_owner_resolutions(self):
        bce = self.res['ADVERSARIAL_BCE']
        self.assertEqual(bce['L_D_real'], "BCEWithLogits(D(real), ones, reduction='mean')")
        self.assertEqual(bce['L_D_fake'], "BCEWithLogits(D(fake.detach()), zeros, reduction='mean')")
        self.assertEqual(bce['L_D'], '0.5 * (L_D_real + L_D_fake)')
        self.assertEqual(bce['L_D_class'], 'OWNER_IMPLEMENTATION_CLARIFICATION')
        self.assertIn('equal size', bce['L_D_equivalent_form'])
        self.assertEqual(bce['L_Gadv'], "BCEWithLogits(D(x_hat), ones, reduction='mean'); D parameters gradient-disabled; "
                                        'x_hat NOT detached')
        self.assertIsNone(bce['new_deviation'])
        items = self.rec['residual_open_items']
        self.assertEqual(items['M7C2A-OBS-01']['status'], 'RESOLVED_M7C2A_OWNER')
        obs2 = items['M7C2A-OBS-02']
        self.assertEqual((obs2['class'], obs2['distribution_metadata_version'], obs2['runtime_pywt___version__']),
                         ('PACKAGING_METADATA_ANOMALY', '1.9.0', '1.8.0'))
        self.assertEqual(obs2['wheel_sha256'], 'd76b7fa8fc500b09201d689b4f15bf5887e30ffbe2e1f338eb8470590eb4521a')
        self.assertIs(obs2['pywt___version___patched'], False)
        self.assertEqual(self.rec['unresolved_blockers_for_m7c2b'], [])
        self.assertEqual(list(self.rec['remaining_deferred']), ['R-04_LEVEL2'])
        self.assertIs(self.rec['remaining_deferred']['R-04_LEVEL2']['blocks_static_core_implementation'], False)
        for token in ('STATIC_CORE_IMPLEMENTATION_ALLOWED', 'GPU_RUNTIME_NOT_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_ALLOWED'):
            self.assertIn(token, self.rec['m7_status_after'])
        rep = self.res['R-04']['level2']['requantization_replicas']
        self.assertIn('BEFORE', rep['meaning'])
        self.assertIn('NOT +/-0.5 added to already-quantized integer pixels', rep['meaning'])
        self.assertEqual(len(rep['procedure']), 5)
        self.assertIn('[-0.5, +0.5] LSB', rep['procedure'][1])
        self.assertEqual(rep['count_per_image'], 8)
        ev = json.loads(at_m7c2a(EVIDENCE).decode())
        pv = ev['pywavelets_provenance']
        self.assertEqual((pv['importlib_metadata_version'], pv['pywt___version__']), ('1.9.0', '1.8.0'))
        self.assertTrue(pv['module_in_environment'] and pv['installed_version_py_matches_record'])
        self.assertEqual(pv['environment_name'], 'gpat-m7-cpu')
        self.assertTrue(pv['pywt___file__'].startswith(pv['sys_prefix'] + '/'))
        self.assertLessEqual(ev['adversarial_bce']['abs_difference_float64'], 1e-12)

    def test_16_preflight_checks_at_m7c2a_and_rejections(self):
        spec = importlib.util.spec_from_file_location('m7c2a_preflight', ROOT / 'tools/m7c2a_gpat_runtime_contract_preflight.py')
        pf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pf)
        self.assertEqual(pf.AUTHORITY, AUTHORITY)
        rec = pf.check_record(at_m7c2a)
        pf.check_pins(at_m7c2a, at_authority)
        pf.check_environment(at_m7c2a)
        pf.check_evidence(at_m7c2a)
        pf.check_config_status(at_m7c2a, at_authority)
        for mutate in (lambda r: r['resolutions']['R-05'].__setitem__('option', 'C'),
                       lambda r: r['resolutions']['N-01'].__setitem__('imagenet_normalization', True),
                       lambda r: r['resolutions']['R-04']['level1']['gates']['highpass_float_path_max_abs']
                       .__setitem__('threshold', 1e-3),
                       lambda r: r['resolutions']['R-04']['level2'].__setitem__('executed_in_m7c2a', True),
                       lambda r: r['resolutions']['N-04'].__setitem__('g_reforward_through_updated_d', True),
                       lambda r: r.__setitem__('new_deviation', 'DEV-023')):
            bad = copy.deepcopy(rec)
            mutate(bad)
            raw = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                pf.check_record(lambda rel, raw=raw: raw if rel == RECORD else at_m7c2a(rel))
        pins = json.loads(at_m7c2a(PINS).decode())
        pins['sources']['dsdg']['pinned_commit'] = '0' * 40
        raw = (json.dumps(pins, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode()
        with self.assertRaises(ValueError):
            pf.check_pins(lambda rel: raw if rel == PINS else at_m7c2a(rel), at_authority)


@unittest.skipUnless(LIVE, 'live CPU tier needs torch + ptwt (run in gpat-m7-cpu)')
class LiveM7C2A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from methods.gpat import naf_source, runtime_contract, teacher_preprocess
        cls.torch, cls.naf, cls.rc, cls.tp = torch, naf_source, runtime_contract, teacher_preprocess
        runtime_contract.apply_qualification_determinism(42, gpu=False)
        cls.tool = load_tool()
        try:
            cls.syms = naf_source.load()
        except naf_source.NAFSourceError as exc:
            raise unittest.SkipTest(f'pinned NAFNet cache absent: {exc}')

    def test_20_naf_loader_forward_backward_gradcheck(self):
        import warnings
        torch = self.torch
        self.assertEqual(set(self.syms), {'SimpleGate', 'NAFBlock', 'LayerNormFunction', 'LayerNorm2d'})
        self.assertFalse(any(m.split('.')[0] in ('basicsr', 'lmdb') for m in sys.modules))
        blk = self.syms['NAFBlock'](32)
        self.assertEqual(sum(p.numel() for p in blk.parameters()), 8224)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = blk(x)
            y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
        x64 = torch.randn(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(4, dtype=torch.float64, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertTrue(torch.autograd.gradcheck(lambda a, c, d: self.syms['LayerNormFunction'].apply(a, c, d, 1e-6),
                                                     (x64, w, b), eps=1e-6, atol=1e-6))

    def test_21_naf_native_parity(self):
        import warnings
        torch = self.torch
        LNF = self.syms['LayerNormFunction']
        for dt, tol in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
            x = torch.randn(2, 16, 8, 8, dtype=dt) * 2 + 0.5
            w, b, g = torch.randn(16, dtype=dt), torch.randn(16, dtype=dt), torch.randn(2, 16, 8, 8, dtype=dt)
            a = [t.clone().requires_grad_(True) for t in (x, w, b)]
            r = [t.clone().requires_grad_(True) for t in (x, w, b)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y1 = LNF.apply(*a, 1e-6)
                y1.backward(g)
            y2 = self.tool.ln_native(*r, 1e-6)
            y2.backward(g)
            self.assertLessEqual(float((y1 - y2).detach().abs().max()), tol)
            for p, q in zip(a, r):
                self.assertLessEqual(float((p.grad - q.grad).abs().max()), tol * max(1.0, float(q.grad.abs().max())))

    def test_22_r04_level1_gates(self):
        imgs = self.tool.corpus()
        result = self.tool.level1(imgs)
        for k, v in result['pass'].items():
            self.assertIs(v, True, k)
        ev = json.loads(at_m7c2a(EVIDENCE).decode())
        self.assertEqual(result['measured'], ev['r04_level1']['measured'])

    def test_23_adapters_do_not_modify_x_hat_and_keep_gradient(self):
        torch, tp = self.torch, self.tp
        x = (torch.rand(1, 3, 256, 256) * 2.4 - 1.2).requires_grad_(True)       # deliberately outside [-1, 1]
        before = x.detach().clone()
        fx = tp.facexformer_input(x)
        af = tp.adaface_input(x)
        (fx.sum() + af.sum()).backward()
        self.assertTrue(torch.equal(x.detach(), before))
        self.assertEqual(tuple(fx.shape), (1, 3, 224, 224))
        self.assertEqual(tuple(af.shape), (1, 3, 112, 112))
        self.assertTrue(torch.isfinite(x.grad).all())
        inside = (before.abs() < 0.9)
        self.assertGreater(float(x.grad[inside].abs().sum()), 0.0)
        xt = torch.randint(0, 256, (1, 3, 256, 256)).float() / 127.5 - 1
        self.assertLessEqual(float(tp.adaface_input(xt).abs().amax()), 1.0 + 1e-6)

    def test_24_n01_highpass_domain(self):
        torch, tp = self.torch, self.tp
        c = torch.full((1, 3, 32, 32), 0.37)
        self.assertLessEqual(float(tp.highpass(c).abs().max()), 1e-6)
        x = torch.rand(1, 3, 64, 64) * 2 - 1
        u = (x + 1) / 2
        self.assertLessEqual(float((tp.highpass(x) - 2 * tp.highpass(u)).abs().max()), 1e-6)   # linear, no remap
        g = tp.gaussian_kernel_1d()
        self.assertEqual(len(g), 9)
        self.assertAlmostEqual(float(g.sum()), 1.0, places=12)

    def test_25_n03_nearest_exact_mask(self):
        torch, rc = self.torch, self.rc
        m = torch.zeros(1, 224, 224, dtype=torch.long)
        m[0, 100, 50] = 7
        m[0, 10, 10] = 0
        out = rc.face_mask_dilated(m)
        self.assertEqual(tuple(out.shape), (1, 1, 256, 256))
        src = [math.floor((o + 0.5) * 224 / 256) for o in range(256)]
        rows = [i for i in range(256) if src[i] == 100]
        cols = [j for j in range(256) if src[j] == 50]
        expect = torch.zeros(256, 256)
        expect[max(rows[0] - 7, 0):rows[-1] + 8, max(cols[0] - 7, 0):cols[-1] + 8] = 1
        self.assertTrue(torch.equal(out[0, 0], expect))
        bg = torch.zeros(1, 224, 224, dtype=torch.long)
        self.assertEqual(float(rc.face_mask_dilated(bg).sum()), 0.0)

    def test_26_n04_pre_update_joint_gradient_toy(self):
        torch, rc = self.torch, self.rc
        torch.manual_seed(0)
        G, D = torch.nn.Linear(4, 4), torch.nn.Linear(4, 1)
        g_opt, d_opt = torch.optim.Adam(G.parameters(), 1e-2), torch.optim.Adam(D.parameters(), 1e-2)
        xs = [torch.randn(4, 4), torch.randn(2, 4)]
        real = [torch.randn(4, 4), torch.randn(2, 4)]
        w = rc.group_weights([4, 2])
        g0 = {k: v.clone() for k, v in G.state_dict().items()}
        d0 = {k: v.clone() for k, v in D.state_dict().items()}
        for i in range(2):
            x_hat = G(xs[i])
            d_before = [None if p.grad is None else p.grad.clone() for p in D.parameters()]
            D.requires_grad_(False)
            (w[i] * rc.g_adv_bce(D(x_hat))).backward()
            for p, b in zip(D.parameters(), d_before):      # the G loss never touches D gradients
                self.assertTrue(p.grad is None if b is None else torch.equal(p.grad, b))
            D.requires_grad_(True)
            lr, lf = rc.d_bce_terms(D(real[i]), D(x_hat.detach()))
            (w[i] * (lr + lf)).backward()
        g_grads = [p.grad.clone() for p in G.parameters()]
        # reference: G gradient from the pre-update D, recomputed in one pass
        Gr, Dr = copy.deepcopy(G), copy.deepcopy(D)
        Gr.load_state_dict(g0)
        Dr.load_state_dict(d0)
        Gr.zero_grad()
        Dr.requires_grad_(False)
        sum(w[i] * rc.g_adv_bce(Dr(Gr(xs[i]))) for i in range(2)).backward()
        for a, b in zip(g_grads, Gr.parameters()):
            self.assertTrue(torch.allclose(a, b.grad, atol=1e-7))
        d_opt.step()
        g_opt.step()
        self.assertEqual(rc.JOINT_UPDATE_BOUNDARY[2], 'D_SCALER.step(D_OPT)')
        self.assertEqual(rc.JOINT_UPDATE_BOUNDARY[6], 'G_SCALER.step(G_OPT)')

    def test_27_n06_ema_policy(self):
        torch, rc = self.torch, self.rc
        live = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        ema = copy.deepcopy(live)
        live.train()
        live(torch.randn(4, 3, 8, 8))
        e0 = {k: v.clone() for k, v in ema.state_dict().items()}
        rc.ema_update(ema.state_dict(), live.state_dict(), 0.999)
        for k, v in ema.state_dict().items():
            l = live.state_dict()[k]
            if torch.is_floating_point(v):
                self.assertTrue(torch.allclose(v, 0.999 * e0[k] + 0.001 * l), k)
            else:
                self.assertTrue(torch.equal(v, l), k)
        self.assertEqual(int(ema.state_dict()['1.num_batches_tracked']), 1)

    def test_28_parsing_losses(self):
        torch, rc = self.torch, self.rc
        z = torch.randn(2, 11, 16, 16)
        self.assertLessEqual(float(rc.kl_loss(z, z)), 1e-7)
        p = torch.softmax(z, 1)
        dice = (2 * (p[:, 1:11] ** 2).sum((-2, -1)) + 1e-6) / (2 * p[:, 1:11].sum((-2, -1)) + 1e-6)
        self.assertLessEqual(abs(float(rc.soft_dice_loss(z, z)) - float(1 - dice.mean())), 1e-6)
        zt, zh = torch.randn(2, 11, 16, 16), torch.randn(2, 11, 16, 16)
        pt, ph = torch.softmax(zt, 1), torch.softmax(zh, 1)
        kl = (pt * (pt.clamp_min(1e-8).log() - ph.clamp_min(1e-8).log())).sum(1).mean()
        self.assertLessEqual(abs(float(rc.kl_loss(zt, zh)) - float(kl)), 1e-6)
        self.assertLessEqual(abs(float(rc.parse_loss(zt, zh)) - float(rc.soft_dice_loss(zt, zh) + 0.1 * kl)), 1e-6)

    def test_29_spectra(self):
        torch, rc = self.torch, self.rc
        const = torch.full((1, 3, 256, 256), 0.5)
        sr = rc.s_radial(const)
        self.assertEqual(tuple(sr.shape), (1, 3, 128))
        self.assertTrue(torch.allclose(sr[..., 0], torch.ones(1, 3)))                  # DC bin included
        self.assertEqual(float(rc.s_orient(const).abs().sum()), 0.0)                   # DC excluded
        xx = torch.arange(256.).view(1, 1, 1, 256).expand(1, 3, 256, 256)
        horiz = torch.cos(2 * math.pi * 10 * xx / 256)                                  # varies along x
        so = rc.s_orient(horiz)
        self.assertGreater(float(so[0, 0, 0]), 0.999)                                   # theta 0 -> bin 0
        vert = horiz.transpose(-1, -2)
        self.assertGreater(float(rc.s_orient(vert)[0, 0, 4]), 0.999)                    # theta pi/2 -> bin 4
        r = rc.radial_bin_index()
        self.assertEqual(int(r[128, 128]), 0)
        self.assertEqual(int(r[128, 128 + 127]), 127)
        idx, support = rc.orient_bin_index()
        self.assertFalse(bool(support[128, 128]))
        self.assertFalse(bool(support[0, 0]))                                           # corner radius > 0.5
        self.assertTrue(bool(support[128, 0]))                                          # Nyquist radius == 0.5
        self.assertTrue(int(idx.max()) <= 7 and int(idx.min()) >= 0)
        a, b = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
        ref = (rc.s_radial(a) - rc.s_radial(b)).abs().mean() + 0.5 * (rc.s_orient(a) - rc.s_orient(b)).abs().mean()
        self.assertEqual(float(rc.spec_loss(a, b)), float(ref))

    def test_30_lferr_internal_and_determinism(self):
        torch, rc = self.torch, self.rc
        ll_t = torch.randn(3, 3, 128, 128)
        gamma, d_ll = 0.0, torch.randn(3, 3, 128, 128)
        self.assertTrue(torch.equal(rc.lferr_internal(ll_t + gamma * d_ll, ll_t), torch.zeros(3)))
        self.assertGreater(float(rc.lferr_internal(ll_t + 0.05 * d_ll, ll_t).min()), 0)
        state = rc.apply_qualification_determinism(42, gpu=False)
        self.assertIs(state['use_deterministic_algorithms'], True)

    def test_32_discriminator_and_generator_bce(self):
        torch, rc = self.torch, self.rc
        import torch.nn.functional as F
        for dt, tol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
            real, fake = torch.randn(4, 1, 30, 30, dtype=dt), torch.randn(4, 1, 30, 30, dtype=dt)
            lr = F.binary_cross_entropy_with_logits(real, torch.ones_like(real), reduction='mean')
            lf = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake), reduction='mean')
            self.assertEqual(float(rc.d_loss(real, fake)), float(0.5 * (lr + lf)))
            cat = torch.nn.BCEWithLogitsLoss(reduction='mean')(
                torch.cat([real, fake], 0), torch.cat([torch.ones_like(real), torch.zeros_like(fake)], 0))
            self.assertLessEqual(abs(float(rc.d_loss(real, fake)) - float(cat)), tol)
        z = torch.randn(4, 1, 30, 30)
        self.assertEqual(float(rc.g_adv_bce(z)), float(F.binary_cross_entropy_with_logits(z, torch.ones_like(z))))
        half = z.half()
        self.assertEqual(rc.d_loss(half, half).dtype, torch.float32)       # never below fp32 (R-06)
        # fake detached only for D; x_hat stays differentiable for the G adversarial loss
        G, D = torch.nn.Conv2d(3, 3, 1), torch.nn.Conv2d(3, 1, 4, 2, 1)
        x_hat = G(torch.randn(2, 3, 8, 8))
        D.zero_grad()
        rc.d_loss(D(torch.randn(2, 3, 8, 8)), D(x_hat.detach())).backward()
        self.assertTrue(all(p.grad is None for p in G.parameters()))
        self.assertTrue(all(p.grad is not None for p in D.parameters()))
        d_grads = [p.grad.clone() for p in D.parameters()]
        D.requires_grad_(False)
        rc.g_adv_bce(D(x_hat)).backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in G.parameters()))
        self.assertTrue(all(torch.equal(p.grad, g) for p, g in zip(D.parameters(), d_grads)))

    def test_33_pywavelets_provenance(self):
        import importlib.metadata as md
        import pywt
        self.assertEqual(md.version('PyWavelets'), '1.9.0')
        self.assertEqual(pywt.__version__, '1.8.0')        # recorded PACKAGING_METADATA_ANOMALY, not patched
        prefix = Path(sys.prefix).resolve()
        if prefix.name != 'gpat-m7-cpu':
            self.skipTest(f'environment-provenance assertion applies to gpat-m7-cpu, running in {prefix}')
        self.assertTrue(Path(pywt.__file__).resolve().is_relative_to(prefix))
        self.assertTrue(Path(md.distribution('PyWavelets')._path).resolve().is_relative_to(prefix))

    def test_31_level2_helpers(self):
        torch, tp = self.torch, self.tp
        ids = {'casia_fasd': [f'c{i}' for i in range(100)], 'msu_mfsd': [f'm{i}' for i in range(10)], 'siwmv2': []}
        sub = tp.level2_subset(ids)
        self.assertEqual([len(sub[d]) for d in tp.L2_DATASETS], [64, 10, 0])
        digest = sorted(hashlib.sha256(s.encode()).hexdigest() for s in ids['casia_fasd'])[:64]
        self.assertEqual(sorted(hashlib.sha256(s.encode()).hexdigest() for s in sub['casia_fasd']), digest)
        seeds = {tp.level2_seed('casia_fasd', 'c1', k) for k in range(8)}
        self.assertEqual(len(seeds), 8)
        unit = torch.rand(1, 3, 224, 224)
        r1 = tp.requantization_replica(unit, 'casia_fasd', 'c1', 3)
        self.assertTrue(torch.equal(r1, tp.requantization_replica(unit, 'casia_fasd', 'c1', 3)))
        self.assertLessEqual(float((r1 * 255 - unit * 255).abs().max()), 1.0 + 1e-4)
        self.assertLessEqual(float((r1 * 255 - torch.round(r1 * 255)).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
