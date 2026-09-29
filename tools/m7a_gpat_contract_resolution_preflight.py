"""Verify the M7A candidate: GPAT M7 contract resolution (Amendment A10) with historical GPAT reconciliation.

M7A encodes the owner decisions D01-D17 additively (A10 amendment + resolution record), registers DEV-022 (the one
controlled adaptation: B2/B3 masked identity supervision on SiW-Mv2), closes DEV-003 / Q-05 and Q-21-for-M7, pins the
NAFNet architecture source, and records the historical project /home/cong/PRISM_FAS_C_LLM_Project as a Level-3 reference
only. It creates no GPAT code, no B1/B2/B3 config, no environment, no checkpoint and no bank, trains nothing, reads no
data (bound manifest hashes are read from git blobs only) and never touches TEST.

STATIC: stdlib only (no Torch, YAML parser, numpy, PIL or pyarrow); no GPU. The A10 record is JSON-syntax YAML.
Candidate-time checks (HEAD lock, worktree, ledger, index) run only here; check_* take readers (no HEAD lock) so the
regression tests can pin them to the commit that added them.

  --write-evidence-json                                  candidate creation only
  --before-ledger / --append-ledger --tests JSON / --rebuild-index / (default final)
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = '8d4ddb3c55e8809398346dbdc10d15318e14b60d'
BRANCH = 'm6-baselines'
MILESTONE = 'M7A'
CLASSIFICATION = 'M7A_GPAT_CONTRACT_RESOLUTION'
RECORD_KIND = 'AMENDMENT / OWNER_CONTRACT_RESOLUTION'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 135
LEDGER_PREFIX_SHA = 'c8e5c1a74e62169e79bd0f4481f3eb2ff926a504023c5d16f138337d38452ede'
INDEX_BASELINE_ROWS = 810
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
DEVIATIONS = 'outputs/audit/deviation_report.md'
MODIFIED = tuple(sorted((CONFIG_STATUS, DEVIATIONS)))
A10_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md'
A10 = 'configs/amendments/gpat_a10_m7_contract_resolution.yaml'
EV_JSON = 'outputs/audit/M7A_GPAT_CONTRACT_RESOLUTION.json'
EV_MD = 'outputs/audit/M7A_GPAT_CONTRACT_RESOLUTION.md'
TESTS = 'tests/test_m7a_gpat_contract_resolution.py'
PREFLIGHT = 'tools/m7a_gpat_contract_resolution_preflight.py'
NEW = tuple(sorted((A10_DOC, A10, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
GPAT_B0 = 'configs/methods/gpat_b0.yaml'
GPAT_B0_SHA = '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a'
GPAT_ABSENT = ('configs/methods/gpat_b1.yaml', 'configs/methods/gpat_b2.yaml', 'configs/methods/gpat_b3.yaml',
               'frozen_config_snapshot/configs/methods/gpat_b1.yaml', 'frozen_config_snapshot/configs/methods/gpat_b2.yaml',
               'frozen_config_snapshot/configs/methods/gpat_b3.yaml', 'runs/m7', 'environments/gpat.lock.json')
STAGE_STATE = 'outputs/audit/STAGE_STATE.json'
METHOD_STATUS = 'outputs/audit/method_status.csv'
PROTECTED = ('outputs/audit/method_status.csv', 'outputs/audit/M6E_FINAL_M6_CLOSURE.json',
             'outputs/audit/M6E_FINAL_M6_CLOSURE.md', STAGE_STATE, 'configs/frozen/fair_track_v1.yaml', GPAT_B0,
             'frozen_config_snapshot/configs/methods/gpat_b0.yaml', SPEC, 'third_party/registry.yaml',
             'third_party/source_pins.json', 'models/registry.yaml')
PROTECTED_PREFIXES = ('docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A', 'configs/methods/',
                      'frozen_config_snapshot/', 'configs/frozen/', 'environments/', 'methods/', 'gpatbench/')
HISTORICAL = '/home/cong/PRISM_FAS_C_LLM_Project'
HISTORICAL_HEAD = '6f0642a1d05c35e4c1778d329fb547f1b22a4115'
HISTORICAL_STATUS_SHA = '3b91017a2ecb0d3734b82cc2edc03772da0de9d863697d92f6ad62350a51eb77'
HISTORICAL_DIFF_SHA = '541030621c84ed0154fc0a5714a188a284649814134f84d87982d0bc27cea5c1'
NAFNET_COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'
NEW_DEV = 'DEV-022'
HIST_CLASSES = ('EXACT_MATCH_CURRENT_AUTHORITY', 'COMPATIBLE_IMPLEMENTATION_REFERENCE', 'USEFUL_BUT_REQUIRES_ADAPTATION',
                'SUPERSEDED_BY_CURRENT_AUTHORITY', 'CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE',
                'NOT_RELEVANT_TO_CURRENT_GPAT')
DECISION_CLASSES = ('FROZEN_SPEC_FACT', 'OWNER_SCIENTIFIC_CLARIFICATION', 'IMPLEMENTATION_CLARIFICATION',
                    'RUNTIME_COMPATIBILITY', 'CONTROLLED_ADAPTATION', 'EXECUTION_STORAGE_POLICY',
                    'HISTORICAL_IMPLEMENTATION_REFERENCE')
ATTACK_CLASSES = ['makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay']
HEX40, HEX64 = re.compile(r'^[0-9a-f]{40}$'), re.compile(r'^[0-9a-f]{64}$')


def require(ok, message):
    if not ok:
        raise ValueError('M7A: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, cwd=None):
    return subprocess.check_output(['git', '-C', str(cwd or ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg', '.docx'}, 'no weight/data ' + path)
    return (ROOT / p).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


def load_record(reader):
    record = json.loads(reader(A10).decode('utf-8'))
    require(record['schema'] == 'gpat.m7a.a10_gpat_m7_contract_resolution' and record['schema_version'] == 1, 'schema')
    return record


# ----------------------------------------------------------------- the owner decisions, encoded exactly
def m7_arithmetic():
    """D06/D15 arithmetic derived from the frozen counts (not copied from the record)."""
    pairs, physical, accum, epochs, warmup_epochs = 8838, 4, 2, 60, 5
    micro = -(-pairs // physical)
    updates = -(-micro // accum)
    tail_group = pairs - (updates - 1) * physical * accum
    warm = -(-pairs // 64)
    return {'microbatches_per_epoch': micro, 'optimizer_updates_per_epoch': updates, 'final_group_samples': tail_group,
            'updates_per_run': updates * epochs, 'lr_warmup_updates': updates * warmup_epochs,
            'warmup_steps_per_epoch': warm, 'warmup_total_steps': warm * 10}


def check_decisions(record):
    d = record['decisions']
    require(sorted(d) == [f'D{i:02d}' for i in range(1, 18)], 'exactly D01-D17')
    classes = [v['class'] for k, v in d.items() if k != 'D10'] + [v['class'] for v in d['D10'].values()]
    require(set(classes) <= set(DECISION_CLASSES), 'decision classes')
    require([k for k, v in d.items() if k != 'D10' and v['class'] == 'CONTROLLED_ADAPTATION'] == ['D03'] and
            all(v['class'] != 'CONTROLLED_ADAPTATION' for v in d['D10'].values()), 'D03 is the only controlled adaptation')
    d01 = d['D01']
    require((d01['closes'], d01['lambda_dir'], d01['meaning'], d01['standalone_directional_loss'], d01['inert_key']) ==
            ('DEV-003', 0.5, 'coefficient_of_S_orient_inside_L_spec', False, False), 'D01')
    require('lambda_dir * L1(S_orient' in d01['l_spec'], 'D01 formula')
    d02 = d['D02']
    require((d02['closes'], d02['grl_alpha'], d02['idadv_term_sign'], d02['identity_head_minimizes_ce'],
             d02['single_reversal_via_grl']) == ('Q-05', 1.0, '+', True, True), 'D02')
    require(d02['objective_term'] == '+ lambda_idadv * CE(identity_head(GRL(z_a)), source_subject)', 'D02 term')
    require(d02['forbidden'] == 'GRL + (- lambda_idadv * CE)', 'D02 forbids double reversal')
    d03 = d['D03']
    require((d03['deviation'], d03['identity_label'], d03['identity_classes'], d03['identity_ce_masked']) ==
            (NEW_DEV, 'source_subject', 60, ['siwmv2']), 'D03')
    require(d03['datasets'] == ['casia_fasd', 'msu_mfsd', 'siwmv2'] and d03['variants'] == ['B2', 'B3'], 'D03 scope')
    require(d03['identity_classes_by_dataset'] == {'casia_fasd': 35, 'msu_mfsd': 25, 'siwmv2': 0}, 'D03 subjects')
    require(d03['zero_labelled_rows_in_step'] == 'L_idadv = 0 exactly' and 'labelled_rows_in_optimizer_step' in
            d03['normalization'], 'D03 normalization')
    require({'pseudo_subject_id', 'clustering_id', 'video_id_as_identity', 'adaface_pseudo_identity'} <=
            set(d03['fabrication_forbidden']), 'D03 never fabricates identity')
    require(d['D04']['datasets'] == ['casia_fasd', 'msu_mfsd', 'siwmv2'] and d['D04']['variants'] == ['B1'], 'D04')
    d05 = d['D05']
    require(d05['class_order'] == ATTACK_CLASSES and d05['head'] == {'type': 'Linear', 'in_features': 512,
                                                                    'out_features': 6, 'bias': True}, 'D05 head/order')
    require((d05['live_class'], d05['class_weights'], d05['reduction'], d05['lambda_type'], d05['apply_to_x_hat'],
             d05['warmup_head_reused_in_generator_stage'], d05['samples']) ==
            (False, None, 'mean', 0.2, False, True, 'spoof_sources_only'), 'D05')
    d06, a = d['D06'], m7_arithmetic()
    require((d06['epochs'], d06['optimizer'], d06['lr'], d06['weight_decay'], d06['batch'], d06['drop_last'],
             d06['carry_optimizer_state'], d06['val_selection'], d06['counted_in_generator_epochs']) ==
            (10, 'AdamW', 1e-4, 1e-4, 64, False, False, False, False), 'D06')
    require(d06['carry_into_generator_stage'] == ['E_art_weights', 'attack_head_weights'] and
            d06['generator_stage_optimizer'] == 'fresh_Adam' and d06['schedule'] == 'cosine_per_optimizer_step_to_0', 'D06 carry')
    require((d06['steps_per_epoch'], d06['total_steps']) == (a['warmup_steps_per_epoch'], a['warmup_total_steps']), 'D06 steps')
    d07 = d['D07']
    require(d07['teachers_frozen'] and d07['x_hat_path_differentiable'] and d07['teacher_precision'] == 'fp32' and
            d07['x_t_path'] == 'same_adapter_then_detached' and d07['parity_test_required_before_training'], 'D07')
    require(set(d07['preserved']) == {'channel_order', 'normalization', 'teacher_weights', 'teacher_code', 'eval_mode',
                                      'teacher_parameters_requires_grad_false'}, 'D07 frozen teacher path')
    d08 = d['D08']
    require((d08['representation'], d08['shape'], d08['space'], d08['heatmaps'], d08['pixel_units_in_training_loss']) ==
            ('coordinates', [68, 2], 'facexformer_native_normalized', False, False), 'D08 coordinates, not heatmaps')
    d09 = d['D09']
    require((d09['closes_for_m7'], d09['background'], d09['non_background'], d09['semantic_names'],
             d09['void_or_ignore_class']) == ('Q-21', [0], list(range(1, 11)), None, None), 'D09 no invented names')
    require(d09['val_dice']['omit_class_for_sample_when'] == 'absent_from_both_masks' and
            d09['l_parse']['kl'] == 'KL_over_all_11_classes', 'D09 dice/L_parse')
    check_d10(d['D10'])
    d11 = d['D11']
    require((d11['library'], d11['wavelet'], d11['level'], d11['boundary_mode'], d11['dtype'], d11['metric'],
             d11['pass_if_less_than'], d11['dataset_access']) ==
            ('ptwt', 'haar', 1, 'reflect', 'float32', 'max_abs_over_all_values', 1e-5, False), 'D11')
    require(d11['input'] == {'distribution': 'uniform[-1,1]', 'shape': [100, 3, 256, 256], 'generator': 'cpu', 'seed': 42},
            'D11 input')
    require(d['D12']['artifact_scale_zero_disables'] == ['delta_HF', 'delta_LL'] and
            d['D12']['max_abs_x_hat_minus_x_t_lt'] == 1e-5, 'D12 zeroes both LL and HF')
    d13 = d['D13']
    require(d13['artsim'] == 'M5 ArtifactProbe v1 embedding cosine' and d13['not'] == 'GPAT F_art' and
            d13['checkpoint_sha256'] == 'b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff', 'D13')
    d14 = d['D14']
    require(d14['candidates'] == {'ema_epochs': [10, 60], 'inclusive': True, 'N': 51} and d14['ties'] == 'average_rank' and
            d14['percentile'] == 'p = (rank - 1) / (N - 1)' and d14['tie_break'] == ['higher_raw_ArtSim', 'earlier_epoch'],
            'D14 ranks')
    require(d14['data'] == 'VAL_only' and d14['test_participates'] is False, 'D14 VAL only')
    require(d14['nme_normalizer']['value'] == 'bounding_box_of_target_68_landmarks', 'D14 NME')
    d15 = d['D15']
    for k in ('microbatches_per_epoch', 'optimizer_updates_per_epoch', 'final_group_samples', 'updates_per_run',
              'lr_warmup_updates'):
        require(d15[k] == a[k], 'D15 ' + k)
    require((d15['train_pairs'], d15['drop_last'], d15['discard_tail'], a['optimizer_updates_per_epoch']) ==
            (8838, False, False, 1105), 'D15 1105 optimizer steps')
    d16 = d['D16']
    require((d16['training_reads_val'], d16['save_all_ema_candidates'], d16['evaluate_all_before_selecting'],
             d16['selected_checkpoint_protected']) == (False, 51, True, True), 'D16 all 51 before selection')
    d17 = d['D17']
    require(d17['executed_in_m7a'] is False and d17['reuse_dsdg_or_difffas_env'] is False, 'D17 deferred')
    fs = record['frozen_spec_facts']
    require(fs['selection_weights'] == {'ID': 0.25, 'Dice': 0.15, 'ArtSim': 0.25, 'neg_NME': 0.20, 'neg_LFErr': 0.15},
            'frozen SelectionScore weights')
    require(fs['loss_weights']['lambda_dir'] == 0.5 and fs['training']['seeds'] == [42, 1337, 2026], 'frozen facts')
    return {'decisions': len(d), 'd10_items': len(d['D10']), 'arithmetic': a}


def check_d10(d10):
    require(len(d10) == 24 and sorted(d10) == list(d10), 'D10 has 24 ordered items')
    exp = {
        '01_S_radial': {'transform': 'log(1 + power)', 'bins_at_256': 128, 'normalize': 'sum_to_1_per_spectrum'},
        '02_S_orient': {'angular_bins': 8, 'angle_range': '[0, pi)', 'exclude_dc': True, 'radius_normalized': '(0, 0.5]'},
        '03_TV': {'form': 'anisotropic_L1', 'reduction': 'mean'},
        '04_Normalize': {'per_image_min_max': False, 'u': 'M * (0.5 * A_rgb + 0.5 * A_freq)', 'A': 'Normalize(u)',
                         'operator': 'Normalize(u) = clip(u / 2.0, 0.0, 1.0)', 'owner_confirmed': True,
                         'denominator_applies_to': 'u (the whole pre-normalized artifact map)',
                         'frozen_spec_9_4_unchanged': True},
        '05_A_freq_resolution': {'channel': 'mean', 'upsample': 'bilinear_x2_to_256', 'align_corners': False},
        '06_L_budget': {'hinge': 'per_image', 'aggregate': 'batch_mean'},
        '07_M_face_dilated': {'resize': 'nearest_224_to_256', 'kernel': [15, 15], 'kernel_shape': 'square'},
        '08_FiLM': {'projection': 'Linear(512, 2C)', 'modulation': '(1 + gamma) * h + beta'},
        '09_downsample': {'op': 'Conv2d(C, 2C, kernel_size=2, stride=2)'},
        '10_skip': {'op': 'elementwise_addition'},
        '11_bottleneck': {'projection': '1x1 conv 256 -> 512', 'pool': 'adaptive_avg_pool_8x8'},
        '12_band_order': {'input': ['LL_rgb', 'LH_rgb', 'HL_rgb', 'HH_rgb']},
        '13_PatchGAN': {'instance_norm_layers': [2, 3, 4], 'padding': 1, 'canonical_output': [30, 30]},
        '14_D_real_sample': {'real': 'x_s_of_current_TRAIN_batch'},
        '15_D_G_updates': {'ratio': '1:1_per_optimizer_update'},
        '16_source_band_upsampling': {'mode': 'bilinear_x2', 'align_corners': False},
        '17_high_pass': {'kernel': [9, 9], 'sigma': 1.5},
        '18_F_art': {'feature': 'GAP(layer4)', 'dim': 512},
        '19_L1_reductions': {'rule': 'mean_over_elements_unless_frozen_formula_states_otherwise'},
        '20_x_hat_range': {'clamp_during_training': False},
        '21_EMA_scope': {'ema': ['E_art', 'G_res'], 'no_ema': ['frozen_teachers', 'discriminator']},
        '22_generator_lr_schedule': {'warmup': 'linear_from_0', 'decay': 'cosine_to_2e-6_at_final_optimizer_step'},
        '23_curriculum': {'interpolation': 'per_optimizer_step', 'runtime_residual_scale': 'delta_scale_hf'},
        '24_E_art_init': {'class': 'FROZEN_SPEC_FACT'}}
    require(sorted(exp) == sorted(d10), 'D10 item names')
    for item, fields in exp.items():
        for k, v in fields.items():
            require(d10[item][k] == v, f'D10 {item}.{k}')
    require('clip(A_rgb / 2, 0, 1)' in d10['04_Normalize']['forbidden_readings'] and
            'A_freq' in d10['04_Normalize']['u'] and d10['04_Normalize']['u'].startswith('M * '), 'D10.4 whole-u reading')
    require(d10['12_band_order']['output_channels'] == {'delta_LL': [0, 1, 2], 'delta_HF': list(range(3, 12)), 'mask': [12]},
            'D10 output channels')


# ----------------------------------------------------------------- historical project stays Level 3
def check_historical(record):
    chain = record['authority_chain']
    require(chain['historical_project_in_chain'] is False and HISTORICAL not in json.dumps(chain), 'history not in chain')
    h = record['historical_reference']
    require((h['level'], h['role'], h['may_override_authority'], h['head'], h['modified_by_m7a']) ==
            (3, 'REFERENCE_ONLY_NOT_AUTHORITY', False, HISTORICAL_HEAD, False), 'historical reference only')
    require(h['classification'] == 'HISTORICAL_IMPLEMENTATION_REFERENCE_ONLY' and h['files_copied_in_m7a'] == 0 and
            set(h['not_a_source_of']) == {'authority_chain', 'scientific_hyperparameters', 'dataset_scope',
                                          'teacher_choices'}, 'historical implementation reference only')
    rows = h['reconciliation']
    require(len(rows) >= 20 and all(r['classification'] in HIST_CLASSES for r in rows), 'reconciliation classes')
    for comp in ('E_art', 'G_res', 'DWT/IDWT', 'FiLM', 'NAF blocks', 'PatchGAN', 'loss set', 'identity adversary / GRL',
                 'attack-type head', 'warmup', 'EMA', 'checkpointing', 'VAL selection', 'dataset scope'):
        require(any(r['component'] == comp for r in rows), 'reconciled ' + comp)
    for r in rows:
        if r['classification'] == 'CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE':
            require(r['action'].startswith('do not'), 'conflict never adopted: ' + r['component'])
    refs = h['implementation_references']
    require(refs and all(HEX64.match(r['sha256']) and r['git_commit_at_head'] == HISTORICAL_HEAD and r['path'] and
                         r['symbol'] and len(r['lines']) == 2 and r['classification'] in HIST_CLASSES and
                         r['classification'] != 'CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE' for r in refs),
            'every historical reference has provenance')
    require(all(a['adopted'] is False for a in h['historical_alternatives_for_future_ablation']), 'no alternative adopted')
    require(all(HEX64.match(v['sha256']) for v in h['core_files'].values()), 'historical core file hashes')
    return {'reconciliation_rows': len(rows), 'implementation_references': len(refs),
            'classification_counts': {c: sum(r['classification'] == c for r in rows) for c in HIST_CLASSES}}


def check_nafnet(record):
    n = record['nafnet_pin']
    require(HEX40.match(n['commit']) and n['commit'] == NAFNET_COMMIT and n['symbolic_ref_recorded'] is False,
            'immutable NAFNet SHA')
    require(n['repository'] == 'https://github.com/megvii-research/NAFNet' and n['weights_required'] is False and
            n['architecture_source_only'] is True, 'NAFNet architecture source only')
    require({'basicsr/models/archs/NAFNet_arch.py', 'basicsr/models/archs/arch_util.py'} <= set(n['files']) and
            all(HEX64.match(f['sha256']) for f in n['files'].values()), 'NAFNet file hashes')
    return {'commit': n['commit'], 'files': {k: v['sha256'] for k, v in n['files'].items()}}


def check_bound(record):
    bound = record['bound_authority_sha256']
    require(bound[SPEC] == SPEC_SHA and bound[GPAT_B0] == GPAT_B0_SHA, 'spec and gpat_b0 binding')
    require(sum(p.startswith('docs/spec/amendments/') for p in bound) == 9, 'A1-A9 bound')
    for rel, digest in bound.items():
        require(sha(at_authority(rel)) == digest, 'bound hash equals authority blob ' + rel)
    return len(bound)


def check_registers(reader, base):
    dev_now, dev_base = reader(DEVIATIONS), base(DEVIATIONS)
    cs_now, cs_base = reader(CONFIG_STATUS), base(CONFIG_STATUS)
    require(dev_now.startswith(dev_base) and len(dev_now) > len(dev_base), 'deviation_report append-only')
    require(cs_now.startswith(cs_base) and len(cs_now) > len(cs_base), 'CONFIG_STATUS append-only')
    added = dev_now[len(dev_base):].decode('utf-8')
    require(not re.findall(r'DEV-02[2-9]', dev_base.decode('utf-8')), 'DEV-022 was unused at authority')
    known = set(re.findall(r'DEV-\d{3}', dev_base.decode('utf-8')))
    require(max(known) == 'DEV-021' and sorted(set(re.findall(r'DEV-\d{3}', added)) - known) == [NEW_DEV],
            'exactly one new DEV id, the next unused one')
    require('### DEV-022 — GPAT_B2_B3_SIW_MASKED_IDENTITY_SUPERVISION (CONTROLLED_ADAPTATION)' in added, 'DEV-022 entry')
    require('### DEV-003 — status update (M7A, A10/D01)' in added and '### Q-05 — resolved' in added and
            '### Q-21 — resolved for M7' in added, 'DEV-003 / Q-05 / Q-21 updates')
    statuses = re.findall(r'\*\*Status:\*\* (\w+)', dev_now.decode('utf-8'))
    require(set(statuses) <= {'APPROVED', 'UNAPPROVED'}, 'status vocabulary')
    cs_added = cs_now[len(cs_base):].decode('utf-8')
    require('M7 scientific training HAS NOT STARTED' in cs_added and 'M6_CLOSED = true' in cs_added, 'CONFIG_STATUS M7A')
    return {'new_dev_numbers': [NEW_DEV], 'deviation_bytes_added': len(dev_now) - len(dev_base),
            'config_status_bytes_added': len(cs_now) - len(cs_base)}


def check_fair_track(record):
    f = record['fair_track_clarification']
    require(f['a1_modified'] is False and f['fair_track_v1_modified'] is False and f['pseudo_identity_permitted'] is False,
            'A1/fair_track unchanged')
    require(f['variants']['E09_B1']['siwmv2'] == 'INCLUDED_ATTACK_MACRO_PRESENT' and
            f['variants']['E10_B2']['siwmv2'].endswith('DEV-022') and f['variants']['E11_B3']['siwmv2'].endswith('DEV-022'),
            'per-method Track-B instantiability')
    require(record['new_deviation']['id'] == NEW_DEV and record['new_deviation']['decision'] == 'D03', 'DEV-022 is D03')


def check_record(reader, base=at_authority):
    record = load_record(reader)
    require(record['authority_commit'] == AUTHORITY and record['m6_closed'] is True and
            record['m7_scientific_training_started'] is False and record['gpat_code_created'] is False and
            record['b1_b2_b3_configs_created'] is False, 'record flags')
    require(record['amendment_document'] == {'path': A10_DOC, 'sha256': sha(reader(A10_DOC))}, 'A10 document binding')
    out = {'record_sha256': sha(reader(A10)), 'document_sha256': sha(reader(A10_DOC)), **check_decisions(record),
           'historical': check_historical(record), 'nafnet': check_nafnet(record), 'bound': check_bound(record),
           'registers': check_registers(reader, base)}
    check_fair_track(record)
    return out


def check_absent(exists, listdir):
    require(not [p for p in GPAT_ABSENT if exists(p)], 'B1/B2/B3 configs and GPAT runs absent')
    require(listdir('methods/gpat') == ['.gitkeep'], 'methods/gpat holds only .gitkeep')


def build_evidence(result):
    return {'schema': 'gpat.m7a.gpat_contract_resolution_evidence', 'schema_version': 1, 'milestone': MILESTONE,
            'classification': CLASSIFICATION, 'record_kind': RECORD_KIND, 'authority_commit': AUTHORITY,
            'amendment': {'document': A10_DOC, 'document_sha256': result['document_sha256'], 'record': A10,
                          'record_sha256': result['record_sha256']},
            'evidence_markdown': {'path': EV_MD, 'sha256': sha(read(EV_MD))},
            'decisions_encoded': result['decisions'], 'd10_items': result['d10_items'],
            'arithmetic': result['arithmetic'], 'historical': result['historical'], 'nafnet_pin': result['nafnet'],
            'bound_authority_files': result['bound'], 'registers': result['registers'],
            'historical_project': {'path': HISTORICAL, 'head': HISTORICAL_HEAD,
                                   'status_porcelain_sha256_before': HISTORICAL_STATUS_SHA,
                                   'diff_sha256_before': HISTORICAL_DIFF_SHA, 'modified_by_m7a': False},
            'M6_closed': True, 'M7_scientific_training_started': False, 'gpat_code_created': False,
            'b1_b2_b3_configs_created': False, 'environment_created': False, 'packages_installed': False,
            'GPU_contacted': False, 'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False,
            'training_runs': 0, 'checkpoint_writes': 0, 'bank_writes': 0}


def check_evidence(reader, result):
    ev = json.loads(reader(EV_JSON).decode('utf-8'))
    require(ev['amendment']['record_sha256'] == result['record_sha256'] and
            ev['amendment']['document_sha256'] == result['document_sha256'], 'evidence binds A10')
    require(ev['evidence_markdown']['sha256'] == sha(reader(EV_MD)), 'evidence binds its markdown')
    require(ev['arithmetic'] == result['arithmetic'] and ev['nafnet_pin'] == result['nafnet'], 'evidence values')
    for k in ('M7_scientific_training_started', 'gpat_code_created', 'environment_created', 'GPU_contacted', 'TEST_access'):
        require(ev[k] is False, 'evidence flag ' + k)
    return ev


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M6E authority')


def historical_unchanged():
    if not Path(HISTORICAL).is_dir():
        return 'HISTORICAL_PROJECT_ABSENT_ON_THIS_HOST'
    status = git('status', '--porcelain=v1', '--untracked-files=all', cwd=HISTORICAL)
    require(git('rev-parse', 'HEAD', cwd=HISTORICAL).decode().strip() == HISTORICAL_HEAD, 'historical HEAD unchanged')
    require(sha(status) == HISTORICAL_STATUS_SHA and sha(git('diff', cwd=HISTORICAL)) == HISTORICAL_DIFF_SHA,
            'historical worktree unchanged')
    return 'UNCHANGED'


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only the registers + ledger/index differ: ' + json.dumps(changed))
    require(git('diff', '--name-only', '--diff-filter=D', AUTHORITY).decode().strip() == '', 'no deletion')
    require(git('ls-files', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors').decode().strip() == '', 'no weights')
    require(git('diff', '--cached', '--name-only').decode().strip() == '', 'nothing staged')
    for rel in PROTECTED:
        require(git('diff', '--name-only', AUTHORITY, '--', rel).decode().strip() == '', 'protected ' + rel)
    for rel in changed:
        require(not rel.startswith(PROTECTED_PREFIXES), 'protected tree ' + rel)


def worktree(bookkeeping):
    status = sorted(git('status', '--porcelain', '--untracked-files=all').decode().splitlines())
    require(status == sorted(['?? ' + p for p in NEW] + [' M ' + p for p in (*MODIFIED, *bookkeeping)]),
            'worktree holds exactly the M7A candidate: ' + json.dumps(status))


def whitespace():
    for rel in NEW:
        raw = read(rel)
        require(not re.search(rb'[ \t]\r?\n', raw) and b'\r' not in raw and raw.endswith(b'\n'), 'LF, no trailing ws ' + rel)
    for rel in MODIFIED:
        added = read(rel)[len(at_authority(rel)):]
        require(not re.search(rb'[ \t]\r?\n', added) and b'\r' not in added, 'LF, no trailing ws ' + rel)


def expected_index():
    reader = csv.DictReader(io.StringIO(at_authority(INDEX).decode()))
    baseline = list(reader)
    rows = {r['path']: r for r in baseline}
    require(reader.fieldnames == ['path', 'size_bytes', 'sha256'] and len(rows) == len(baseline) == INDEX_BASELINE_ROWS,
            'baseline index')
    for p in NEW:
        require(p not in rows, 'additive artifact ' + p)
    for p in MODIFIED:
        require(rows[p]['sha256'] == sha(at_authority(p)), 'baseline row ' + p)
    for p in (*NEW, *MODIFIED):
        raw = read(p)
        rows[p] = {'path': p, 'size_bytes': str(len(raw)), 'sha256': sha(raw)}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows[k] for k in sorted(rows))
    return out.getvalue().encode(), len(rows)


LEDGER_EXPECTED = {
    'milestone': MILESTONE, 'classification': CLASSIFICATION, 'status': 'PASS', 'final_status': 'PASS',
    'record_kind': RECORD_KIND, 'exit_code': 0, 'M6_closed': True, 'M7_started': True,
    'M7_scientific_training_started': False, 'amendment_created': True, 'amendment_id': 'A10',
    'owner_decisions': [f'D{i:02d}' for i in range(1, 18)], 'new_deviation_numbers': [NEW_DEV],
    'resolved': ['DEV-003', 'Q-05', 'Q-21 (for M7)'], 'nafnet_commit': NAFNET_COMMIT,
    'historical_project_role': 'REFERENCE_ONLY_NOT_AUTHORITY', 'historical_project_modified': False,
    'gpat_code_created': False, 'b1_b2_b3_configs_created': False, 'environment_created': False,
    'packages_installed': False, 'method_status_modified': False, 'stage_state_modified': False,
    'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False, 'GPU_contacted': False, 'M8_bank': False,
    'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'commit': False,
    'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': list(MODIFIED),
    'input_artifacts': [SPEC, 'docs/spec/amendments (A1-A9)', 'configs/frozen/fair_track_v1.yaml', GPAT_B0,
                        DEVIATIONS, CONFIG_STATUS, 'outputs/audit/M6E_FINAL_M6_CLOSURE.json',
                        'outputs/audit/artifact_probe_v1.sha256', HISTORICAL + ' (read-only, Level 3)',
                        'https://github.com/megvii-research/NAFNet@' + NAFNET_COMMIT + ' (read-only)'],
    'output_artifacts': sorted((*NEW, *MODIFIED))}


def check_ledger_row(row, prefix):
    for k, v in LEDGER_EXPECTED.items():
        require(row[k] == v, 'ledger field ' + k)
    require(sha(prefix) == row['committed_prefix_sha256'], 'ledger prefix binding')
    require(sorted(row['artifacts_sha256']) == sorted((*NEW, *MODIFIED)), 'ledger artifact list')
    for path, h in row['artifacts_sha256'].items():
        require(sha(read(path)) == h, 'ledger artifact ' + path)
    require(row['tests'], 'ledger tests recorded')
    for t in row['tests']:
        require(t['failures'] == 0 or t.get('failures_preexisting'), 'failures classified: ' + t['scope'])
        require(t['errors'] == 0 or t.get('errors_preexisting_environmental'), 'errors classified: ' + t['scope'])


def ledger_row(tests):
    return {**LEDGER_EXPECTED,
            'purpose': 'M7A: GPAT M7 contract resolution (Amendment A10, owner decisions D01-D17) with historical GPAT '
                       'reference reconciliation; no GPAT code, config, environment, training or checkpoint',
            'notes': ('A10 encodes D01-D17. DEV-003 resolved (D01 lambda_dir = S_orient coefficient in L_spec); Q-05 '
                      'resolved (D02 GRL(1.0) + positive CE); Q-21 resolved for M7 (D09 0 background, 1..10 unnamed); '
                      'DEV-022 = D03 masked identity CE on SiW for B2/B3 (the only controlled adaptation). B1 keeps SiW '
                      '(D04). NAFNet architecture pinned to megvii-research/NAFNet@' + NAFNET_COMMIT + ' (not vendored). '
                      'Historical PRISM_FAS_C GPAT (recipe-conditioned M8 design) reconciled as Level-3 reference only; '
                      'no historical scientific choice imported. A1, fair_track_v1, gpat_b0.yaml, method_status.csv, '
                      'M6E closure and STAGE_STATE unchanged. M7 scientific training not started.'),
            'command': ('laptop only: python3 -B tools/m7a_gpat_contract_resolution_preflight.py --write-evidence-json / '
                        '--before-ledger / --append-ledger / --rebuild-index; .venv/bin/python -B -m unittest discover '
                        '-s tests -p test_m7a_gpat_contract_resolution.py'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    result = check_record(read)
    check_absent(lambda p: (ROOT / p).exists(), lambda p: sorted(q.name for q in (ROOT / p).iterdir()))
    check_evidence(read, result)
    hist = historical_unchanged()
    prefix = at_authority(LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == LEDGER_PREFIX_ROWS and sha(prefix) == LEDGER_PREFIX_SHA and
            current.startswith(prefix), f'first {LEDGER_PREFIX_ROWS} ledger rows byte-identical')
    if stage == 'before_ledger':
        require(current == prefix, 'ledger not yet appended')
    else:
        require(len(current.splitlines()) == LEDGER_PREFIX_ROWS + 1 and current.endswith(b'\n'), 'exactly one append')
        check_ledger_row(json.loads(current[len(prefix):]), prefix)
    expected, count = expected_index()
    if stage == 'final':
        require((ROOT / INDEX).read_bytes() == expected, 'CRLF sorted artifact index')
    worktree(bookkeeping)
    require(not {'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [*MODIFIED, *bookkeeping],
            'historical_project': hist, 'record_sha256': result['record_sha256'],
            'document_sha256': result['document_sha256'], 'new_dev_numbers': [NEW_DEV]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.write_evidence_json:
        authority()
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        ev = build_evidence(check_record(read))
        (ROOT / EV_JSON).write_text(json.dumps(ev, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        print(json.dumps({'status': 'WRITTEN', 'path': EV_JSON}))
        return
    if args.before_ledger:
        result = verify('before_ledger')
    elif args.append_ledger:
        verify('before_ledger')
        row = ledger_row(json.loads(args.tests))
        with open(ROOT / LEDGER, 'ab') as fh:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
            fh.flush()
            os.fsync(fh.fileno())
        result = verify('before_index')
    elif args.rebuild_index:
        verify('before_index')
        (ROOT / INDEX).write_bytes(expected_index()[0])
        result = verify('final')
    else:
        result = verify('final')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
