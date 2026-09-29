"""Verify the M7B candidate: GPAT B1/B2/B3 config freeze (E09/E10/E11, Track B).

M7B creates configs/methods/gpat_b{1,2,3}.yaml and their frozen snapshots from B0 (spec §23.2, unchanged) + Amendment
A10 + the owner-approved M7B clarifications R-01/R-02/R-03/R-06/R-07 (configs/amendments/gpat_m7b_owner_clarifications.yaml).
It creates no GPAT code, environment, checkpoint, bank or run, trains nothing, reads no TRAIN/VAL/TEST sample and never
touches TEST. Manifest hashes are compared with the committed A10 binding, never read from the data.

STATIC: stdlib + PyYAML only (no Torch, numpy, PIL or pyarrow); no GPU. Candidate-time checks (HEAD lock, worktree,
ledger, index) run only here; check_* take readers (no HEAD lock) so the regression tests can pin them to the commit
that added them.

  --print-matrix                                          read-only: B0-B3 comparison matrix as markdown
  --write-evidence-json                                   candidate creation only
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

import yaml

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'd59b9a302c61ed440e711abf6ffab9560458f779'
BRANCH = 'm6-baselines'
MILESTONE = 'M7B'
CLASSIFICATION = 'M7B_GPAT_CONFIG_FREEZE'
RECORD_KIND = 'CONFIG_FREEZE / OWNER_CLARIFICATIONS_LAYERED_ON_A10'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
LEDGER_PREFIX_ROWS = 136
LEDGER_PREFIX_SHA = 'ba752b3d10a994543c443acee483976b941148a79aa71232d18f7892e37a13ce'
INDEX_BASELINE_ROWS = 816
CONFIG_STATUS = 'configs/CONFIG_STATUS.md'
MODIFIED = (CONFIG_STATUS,)
A10_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md'
A10_DOC_SHA = '7baf7888394c4cdd57db09e604bd910153a612e1765b8dc57de3c1d5d833d337'
A10 = 'configs/amendments/gpat_a10_m7_contract_resolution.yaml'
A10_SHA = '57b37f09ca4279ad3999f4f9efa21fbf6fa39836574a0336bd13e34b2bd53203'
CLARIFICATIONS = 'configs/amendments/gpat_m7b_owner_clarifications.yaml'
GPAT_B0 = 'configs/methods/gpat_b0.yaml'
GPAT_B0_SHA = '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a'
SNAP = 'frozen_config_snapshot/'
VARIANTS = ('B1', 'B2', 'B3')
CONFIGS = {v: f'configs/methods/gpat_{v.lower()}.yaml' for v in VARIANTS}
SNAPSHOTS = {v: SNAP + CONFIGS[v] for v in VARIANTS}
EV_JSON = 'outputs/audit/M7B_GPAT_CONFIG_FREEZE.json'
EV_MD = 'outputs/audit/M7B_GPAT_CONFIG_FREEZE.md'
TESTS = 'tests/test_m7b_gpat_config_freeze.py'
PREFLIGHT = 'tools/m7b_gpat_config_freeze_preflight.py'
NEW = tuple(sorted((*CONFIGS.values(), *SNAPSHOTS.values(), CLARIFICATIONS, EV_JSON, EV_MD, TESTS, PREFLIGHT)))
SPEC = 'docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'
SPEC_SHA = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
PROTECTED = (SPEC, A10_DOC, A10, GPAT_B0, SNAP + GPAT_B0, 'outputs/audit/method_status.csv',
             'outputs/audit/M6E_FINAL_M6_CLOSURE.json', 'outputs/audit/M6E_FINAL_M6_CLOSURE.md',
             'outputs/audit/STAGE_STATE.json', 'outputs/audit/deviation_report.md', 'configs/frozen/fair_track_v1.yaml',
             'third_party/registry.yaml', 'third_party/source_pins.json', 'models/registry.yaml')
PROTECTED_PREFIXES = ('docs/', 'configs/frozen/', 'environments/', 'methods/', 'gpatbench/', 'manifests/', 'models/',
                      'third_party/', 'runs/')
NAFNET_COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'
PAIRS_TRAIN = ('manifests/pairs_train_v1.parquet', 'a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243')
PAIRS_VAL = ('manifests/val_pairs_v1.parquet', '84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872')
DATASETS = ['casia_fasd', 'msu_mfsd', 'siwmv2']
ATTACK_CLASSES = ['makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay']
SEEDS = [42, 1337, 2026]
EXPERIMENT = {'B1': 'E09', 'B2': 'E10', 'B3': 'E11'}
LAMBDAS = {'B0': (0.0, 0.0), 'B1': (0.2, 0.0), 'B2': (0.0, 0.1), 'B3': (0.2, 0.1)}
SUPERVISION = {'B1': 'binary_plus_attack_macro', 'B2': 'binary_plus_identity_adversary',
               'B3': 'binary_plus_attack_macro_plus_identity_adversary'}
HEADS = {'B0': (False, False), 'B1': (True, False), 'B2': (False, True), 'B3': (True, True)}
MEMBERSHIP = {'B0': ['G_res', 'E_art'], 'B1': ['G_res', 'E_art', 'attack_type_head'],
              'B2': ['G_res', 'E_art', 'identity_adversary_head'],
              'B3': ['G_res', 'E_art', 'attack_type_head', 'identity_adversary_head']}
TOP_KEYS = ['method', 'supervision', 'input_resolution', 'wavelet', 'gamma', 'artifact_encoder', 'residual_generator',
            'training', 'loss', 'variant', 'status', 'provenance', 'data', 'attack_type_supervision', 'attack_warmup',
            'identity_adversary', 'shared_contract']
B0_DELTAS = {'B1': {'method', 'supervision', 'artifact_encoder.attack_type_head', 'loss.lambda_type'},
             'B2': {'method', 'supervision', 'artifact_encoder.identity_adversary', 'loss.lambda_idadv'},
             'B3': {'method', 'supervision', 'artifact_encoder.attack_type_head', 'artifact_encoder.identity_adversary',
                    'loss.lambda_type', 'loss.lambda_idadv'}}
FABRICATION_FORBIDDEN = ['pseudo_subject_id', 'clustering_id', 'video_id_as_identity', 'content_group_id_as_identity',
                         'adaface_pseudo_identity', 'filename_derived_identity', 'DEV-013_as_same_person_guarantee']
R_ITEMS = [f'R-{i:02d}' for i in range(1, 8)]
R_RESOLVED = ['R-01', 'R-02', 'R-03', 'R-06', 'R-07']
R_DEFERRED = ['R-04', 'R-05']
MATRIX_CLASSES = ('SAME', 'B1_DELTA', 'B2_DELTA', 'B3_UNION', 'IMPLEMENTATION_CLARIFICATION', 'PROVENANCE_ONLY')
HEX40, HEX64 = re.compile(r'^[0-9a-f]{40}$'), re.compile(r'^[0-9a-f]{64}$')


def require(ok, message):
    if not ok:
        raise ValueError('M7B: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def read(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts, 'relative path')
    require(not {'data', 'faces_256', 'runs', 'cache', 'manifests'} & set(p.parts), 'data firewall ' + path)
    require(p.suffix not in {'.pkl', '.pt', '.pth', '.ckpt', '.parquet', '.png', '.jpg', '.docx'}, 'no weight/data ' + path)
    return (ROOT / p).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}')


def load_yaml(raw):
    return yaml.safe_load(raw.decode('utf-8'))


def load_configs(reader):
    return {'B0': load_yaml(reader(GPAT_B0)), **{v: load_yaml(reader(CONFIGS[v])) for v in VARIANTS}}


def m7_arithmetic():
    """D06/D15 arithmetic derived from the frozen counts (not copied from any record)."""
    pairs, physical, accum, epochs, warmup_epochs, warm_batch = 8838, 4, 2, 60, 5, 64
    micro = -(-pairs // physical)
    updates = -(-micro // accum)
    warm = -(-pairs // warm_batch)
    return {'microbatches_per_epoch': micro, 'optimizer_updates_per_epoch': updates,
            'tail_group_samples': pairs - (updates - 1) * physical * accum, 'generator_updates_per_run': updates * epochs,
            'lr_warmup_updates': updates * warmup_epochs, 'warmup_steps_per_epoch': warm,
            'warmup_final_batch_samples': pairs - (warm - 1) * warm_batch, 'warmup_total_steps': warm * 10,
            'val_candidates': 60 - 10 + 1}


def leaves(node, prefix=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, prefix + (str(k),))
    else:
        yield '.'.join(prefix), node


# ----------------------------------------------------------------- authority bindings
def check_bindings(reader):
    require(sha(reader(A10_DOC)) == A10_DOC_SHA, 'A10 document hash')
    require(sha(reader(A10)) == A10_SHA, 'A10 resolution record hash')
    require(sha(reader(GPAT_B0)) == GPAT_B0_SHA and sha(reader(SNAP + GPAT_B0)) == GPAT_B0_SHA, 'B0 and its snapshot unchanged')
    a10 = json.loads(reader(A10).decode('utf-8'))
    require(sorted(a10['decisions']) == [f'D{i:02d}' for i in range(1, 18)], 'A10 D01-D17')
    require(a10['deviation_register_updates'] == {'DEV-003': 'RESOLVED_BY_A10_D01', 'Q-05': 'RESOLVED_BY_A10_D02',
                                                  'Q-21': 'RESOLVED_FOR_M7_BY_A10_D09', 'DEV-022': 'CREATED_FOR_A10_D03'},
            'A10 register: DEV-003, Q-05, Q-21 resolved; DEV-022 the only new deviation')
    require(a10['new_deviation']['id'] == 'DEV-022' and a10['nafnet_pin']['commit'] == NAFNET_COMMIT, 'A10 DEV-022 / NAFNet')
    bound = a10['bound_authority_sha256']
    require(bound[PAIRS_TRAIN[0]] == PAIRS_TRAIN[1] and bound[PAIRS_VAL[0]] == PAIRS_VAL[1], 'A10 manifest binding')
    for item in re.findall(r'^\| (R-0\d) \|', reader(A10_DOC).decode('utf-8'), re.M):
        require(item in R_ITEMS, 'A10 section 11 item ' + item)
    require(re.findall(r'^\| (R-\d+) \|', reader(A10_DOC).decode('utf-8'), re.M) == R_ITEMS, 'A10 section 11 = R-01..R-07')
    return a10


# ----------------------------------------------------------------- the M7B owner clarifications
def check_clarifications(rec):
    require(rec['schema'] == 'gpat.m7b.owner_clarifications' and rec['authority_commit'] == AUTHORITY, 'record identity')
    require(rec['a10_document'] == {'path': A10_DOC, 'sha256': A10_DOC_SHA} and
            rec['a10_record'] == {'path': A10, 'sha256': A10_SHA}, 'record binds A10')
    require((rec['a10_edited'], rec['frozen_spec_edited'], rec['gpat_b0_edited']) == (False, False, False), 'no rewrite')
    require('NOT_FROZEN_SPEC_FACTS' in rec['status'] and 'NOT_AN_AMENDMENT' in rec['status'], 'R-* are not spec facts')
    r = rec['residual_items']
    require(sorted(r) == R_ITEMS == rec['r_items_in_a10_section_11'], 'every A10 section 11 item accounted for')
    require([k for k in R_ITEMS if r[k]['status'] == 'RESOLVED_M7B'] == R_RESOLVED, 'resolved set')
    require([k for k in R_ITEMS if r[k]['status'] == 'DEFERRED_TO_M7C'] == R_DEFERRED, 'deferred set')
    require(not [k for k in R_ITEMS if r[k]['blocks_m7b']], 'no item blocks M7B')
    require(all(r[k]['decision'] is None and r[k]['blocks_m7c'] for k in R_DEFERRED), 'deferred items not silently decided')
    d1 = r['R-01']['decision']
    require(d1['mapping'] == {'LL': 'cA', 'LH': 'cH', 'HL': 'cV', 'HH': 'cD'} and d1['axes'] == [-2, -1] and
            d1['display_convention_swap'] == 'FORBIDDEN', 'R-01 mapping')
    d2 = r['R-02']['decision']
    require(d2['A'] == 'clip(u / 2.0, 0.0, 1.0)' and d2['u'] == 'M_256 * (0.5 * A_rgb + 0.5 * A_freq_256)' and
            'align_corners=False' in d2['artifact_map'] and d2['threshold'] == 'none', 'R-02')
    d3 = r['R-03']['decision']
    require(d3['source'] == 'x_hat_float' and d3['uint8_roundtrip'] == 'FORBIDDEN' and d3['test'] == 'FORBIDDEN', 'R-03')
    d6 = r['R-06']['decision']
    require(d6['bf16'] == 'FORBIDDEN' and d6['x_hat_detach'] == 'FORBIDDEN' and d6['grad_scaler'] is True, 'R-06')
    d7 = r['R-07']['decision']
    require(d7['G_OPT_membership'] == MEMBERSHIP and d7['identity_head_in_G_OPT'] is True and
            d7['identity_head_in_warmup'] is False, 'R-07')
    require('open_observations_not_resolved' not in rec, 'no open observation remains')
    obs = rec['observations']
    require(sorted(obs) == ['M7B-OBS-01'] and all(o['status'] == 'RESOLVED_FROM_FROZEN_SPEC' and
            o['class'] == 'FROZEN_SPEC_DERIVED_EXECUTION_POLICY' and not o['blocks_m7b'] and not o['blocks_m7c']
            for o in obs.values()), 'M7B-OBS-01 resolved from frozen spec 10.5')
    od = obs['M7B-OBS-01']['decision']
    require('lr 2e-4, betas (0.5, 0.999), weight_decay 0' in od['D_OPT'] and od['grad_scaler'] == 'D_SCALER' and
            'D trainable parameters only' in od['grad_clip'] and 'cosine decay to 2e-6' in od['lr_schedule'], 'OBS-01 D contract')
    g = rec['implementation_clarifications']['GRADSCALER_TOPOLOGY']
    require(g['class'] == 'IMPLEMENTATION_CLARIFICATION' and g['scientific_deviation'] is False and
            g['independent_state_machines'] is True and {'G_SCALER', 'D_SCALER', 'WARMUP_SCALER'} <= set(g), 'scaler topology')
    oc = rec['owner_confirmed_at_m7b_review']
    require(oc['identity_head_bias']['class'] == 'OWNER_IMPLEMENTATION_CLARIFICATION' and
            'bias=True' in oc['identity_head_bias']['decision'] and oc['identity_head_bias']['new_deviation'] is None,
            'identity head bias owner-confirmed')
    require(oc['val_clamp_policy']['class'] == 'OWNER_EVALUATION_CLARIFICATION' and
            oc['val_clamp_policy']['decision'] == 'VAL_PRE_EXPORT_FLOAT_NO_CLAMP' and
            oc['val_clamp_policy']['new_deviation'] is None, 'VAL no-clamp owner-confirmed')
    require(d3['policy'] == 'VAL_PRE_EXPORT_FLOAT_NO_CLAMP' and d3['clamp_for_val_scoring'] is False, 'R-03 no clamp')
    return {'resolved': R_RESOLVED, 'deferred_to_m7c': R_DEFERRED,
            'observations': {k: v['status'] for k, v in sorted(obs.items())}}


# ----------------------------------------------------------------- per-config checks
def check_inheritance(b0, cfg, vid):
    b0_leaves = dict(leaves(b0))
    cfg_leaves = dict(leaves({k: cfg[k] for k in b0}))
    require(list(cfg)[:len(b0)] == list(b0) and list(cfg) == TOP_KEYS, vid + ' top-level keys')
    require(sorted(b0_leaves) == sorted(cfg_leaves), vid + ' B0 field set')
    diff = {k for k in b0_leaves if b0_leaves[k] != cfg_leaves[k]}
    require(diff == B0_DELTAS[vid], f'{vid} differs from B0 only in its supervision delta: {sorted(diff)}')
    require(cfg['method'] == 'GPAT-' + vid and cfg['supervision'] == SUPERVISION[vid], vid + ' identity')
    ae = cfg['artifact_encoder']
    require((ae['attack_type_head'], ae['identity_adversary']) == HEADS[vid], vid + ' heads')
    require((cfg['loss']['lambda_type'], cfg['loss']['lambda_idadv']) == LAMBDAS[vid], vid + ' lambdas')
    return sorted(diff)


def check_variant(cfg, vid):
    v, s, p, d = cfg['variant'], cfg['status'], cfg['provenance'], cfg['data']
    require((v['experiment_id'], v['variant_id'], v['track']) == (EXPERIMENT[vid], vid, 'B'), vid + ' E09/E10/E11 Track B')
    require(v['derived_from'] == {'config': GPAT_B0, 'sha256': GPAT_B0_SHA}, vid + ' derived from B0')
    require((s['config'], s['implementation'], s['environment'], s['scientific_training'], s['checkpoints'], s['banks']) ==
            ('CONFIG_FROZEN', 'NOT_STARTED', 'NOT_CREATED_M7B', 'NOT_STARTED', 0, 0), vid + ' status')
    require((p['authority_commit'], p['spec_sha256'], p['a10_document_sha256'], p['a10_record_sha256'],
             p['m7b_clarifications']) == (AUTHORITY, SPEC_SHA, A10_DOC_SHA, A10_SHA, CLARIFICATIONS), vid + ' provenance')
    require(p['r_items_are_frozen_spec_facts'] is False, vid + ' R-* are not spec facts')
    require((d['datasets'], d['train_pairs'], d['pseudo_identity']) == (DATASETS, 8838, 'FORBIDDEN'), vid + ' datasets')
    require((d['pair_manifest'], d['pair_manifest_sha256']) == PAIRS_TRAIN and
            (d['splits']['VAL']['val_pairs'], d['splits']['VAL']['val_pairs_sha256']) == PAIRS_VAL, vid + ' manifests')
    check_test_firewall(d['splits'], vid)
    attack, idadv = HEADS[vid]
    variant_decisions = {'B1': ['A10_D04', 'A10_D05', 'A10_D06', 'M7B_R-06', 'M7B_R-07'],
                         'B2': ['A10_D02', 'A10_D03', 'DEV-022', 'M7B_R-06', 'M7B_R-07']}
    variant_decisions['B3'] = sorted(set(variant_decisions['B1']) | set(variant_decisions['B2']))
    require(sorted(p['variant_decisions']) == sorted(variant_decisions[vid]), vid + ' variant provenance')
    require(p['deviations'] == (['DEV-022'] if idadv else []), vid + ' DEV-022 only in B2/B3')
    if not idadv:
        require('DEV-022' not in json.dumps(cfg), vid + ' mentions no DEV-022')
    labels = (['attack_macro'] if attack else []) + (['source_subject'] if idadv else [])
    require(d['label_fields_used'] == labels, vid + ' label fields')
    check_attack(cfg, vid, attack)
    check_identity(cfg, vid, idadv)
    return {'experiment_id': EXPERIMENT[vid], 'labels': labels}


def check_test_firewall(splits, ctx):
    t = splits['TEST']
    require(t['allowed'] is False and t['used_for'] == [] and t['code_path_present'] is False, ctx + ' TEST forbidden')
    for k in ('never_training', 'never_checkpoint_selection', 'never_hyperparameter_selection', 'never_seed_selection'):
        require(t[k] is True, f'{ctx} TEST {k}')
    require(splits['VAL']['read_by_training_process'] is False and
            splits['VAL']['used_for'] == ['post_training_checkpoint_selection'], ctx + ' VAL post-training only')


def check_attack(cfg, vid, enabled):
    a, w = cfg['attack_type_supervision'], cfg['attack_warmup']
    require(a['enabled'] is enabled and w['enabled'] is enabled and a['lambda_type'] == LAMBDAS[vid][0], vid + ' attack')
    if not enabled:
        return
    require(a['head'] == {'type': 'Linear', 'in_features': 512, 'out_features': 6, 'bias': True}, vid + ' attack head')
    require(a['class_order'] == ATTACK_CLASSES and a['class_index'] == {c: i for i, c in enumerate(ATTACK_CLASSES)} and
            a['live_class'] is False, vid + ' exact 6 attack classes, no live')
    require((a['loss'], a['class_weights'], a['reduction'], a['applied_to'], a['applied_to_x_hat'], a['samples']) ==
            ('CrossEntropy', None, 'mean', 'x_s_only', False, 'spoof_sources_only'), vid + ' unweighted mean CE on x_s')
    require(a['optimizer'] == 'G_OPT' and a['precision']['ce_logits_and_reduction'] == 'float32', vid + ' attack head R-06/R-07')
    ar = m7_arithmetic()
    require((w['epochs'], w['counted_in_generator_epochs'], w['reduces_generator_epochs'], w['batch'], w['drop_last'],
             w['steps_per_epoch'], w['final_batch_samples'], w['total_steps']) ==
            (10, False, False, 64, False, ar['warmup_steps_per_epoch'], ar['warmup_final_batch_samples'],
             ar['warmup_total_steps']), vid + ' warmup budget')
    require(w['optimizer'] == {'name': 'WARMUP_OPT', 'type': 'AdamW', 'params': ['E_art', 'attack_type_head'],
                               'lr': 1e-4, 'weight_decay': 1e-4} and w['schedule'] == 'cosine_per_optimizer_step_to_0',
            vid + ' warmup optimizer')
    require(w['trainable'] == ['E_art', 'attack_type_head'] and w['identity_adversary_head_participates'] is False,
            vid + ' identity head not in warmup')
    require(w['amp'] == {'autocast': 'cuda_fp16', 'autocast_modules': ['E_art', 'attack_type_head'], 'ce_dtype': 'float32',
                         'grad_scaler': True, 'scaler': 'WARMUP_SCALER', 'scaler_state_carried_to_generator_stage': False},
            vid + ' warmup AMP with its own WARMUP_SCALER')
    require(w['grad_clip']['max_norm'] == 1.0 and 'unscale_' in w['grad_clip']['after'] and
            w['grad_clip']['exemption_in_authority'] is False, vid + ' warmup grad clip')
    require((w['carry_optimizer_state'], w['val_selection'], w['state_used']) == (False, False, 'after_warmup_epoch_10') and
            w['carry_into_generator_stage'] == ['E_art_weights', 'attack_head_weights'], vid + ' warmup carry')


def check_identity(cfg, vid, enabled):
    i = cfg['identity_adversary']
    require(i['enabled'] is enabled and i['lambda_idadv'] == LAMBDAS[vid][1], vid + ' identity adversary')
    if not enabled:
        require(i['dev_022_masking'] == 'NOT_APPLICABLE', vid + ' no DEV-022')
        return
    require(i['grl']['alpha'] == 1.0 and i['grl']['reverses'] == 'E_art_path_only', vid + ' GRL(1.0)')
    require(i['objective_term'] == '+ lambda_idadv * CE(identity_head(GRL(z_a)), source_subject)' and
            i['idadv_term_sign'] == '+' and i['identity_head_minimizes_ce'] is True and
            i['single_reversal_via_grl'] is True, vid + ' GRL with positive CE')
    require('GRL + (- lambda_idadv * CE)' in i['forbidden'] and 'second_adversarial_optimizer' in i['forbidden'] and
            'detached_z_a' in i['forbidden'], vid + ' double reversal forbidden')
    require(i['head_bias_class'] == 'OWNER_IMPLEMENTATION_CLARIFICATION', vid + ' identity head bias owner-confirmed')
    require(i['head'] == {'type': 'Linear', 'in_features': 512, 'out_features': 60, 'bias': True} and i['classes'] == 60 and
            i['classes_by_dataset'] == {'casia_fasd': 35, 'msu_mfsd': 25, 'siwmv2': 0}, vid + ' identity head')
    require(i['label'] == 'source_subject' and i['optimizer'] == 'G_OPT' and i['takes_part_in_attack_warmup'] is False,
            vid + ' identity label / optimizer')
    m = i['dev_022_masking']
    require(m['deviation'] == 'DEV-022' and m['pseudo_identity'] == 'FORBIDDEN' and
            m['fabrication_forbidden'] == FABRICATION_FORBIDDEN, vid + ' no pseudo identity')
    require(m['zero_labelled_rows'] == 'L_idadv = 0 exactly' and 'labelled rows in the optimizer update' in m['normalization']
            and 'siwmv2' in m['masked_rows'] and 'siwmv2' not in m['active_rows'], vid + ' DEV-022 masking')


def check_shared(cfg, a10):
    s = cfg['shared_contract']
    require(s['applies_to'] == ['B0', 'B1', 'B2', 'B3'] and s['image'] == {'resolution': 256, 'range': [-1.0, 1.0]}, 'image')
    w = s['wavelet_bands']
    require(w['mapping'] == {'LL': 'cA', 'LH': 'cH', 'HL': 'cV', 'HH': 'cD'} and w['coefficient_order'] ==
            '(cA, (cH, cV, cD))' and w['display_convention_swap'] == 'FORBIDDEN' and w['axes'] == [-2, -1],
            'R-01 ptwt mapping (no LH/HL swap)')
    require(w['band_order'] == ['LL_rgb', 'LH_rgb', 'HL_rgb', 'HH_rgb'] and w['output_channels'] ==
            {'delta_LL': [0, 1, 2], 'delta_LH': [3, 4, 5], 'delta_HL': [6, 7, 8], 'delta_HH': [9, 10, 11], 'mask_logit': [12]},
            'A10 D10.12 band order')
    m = s['mask']
    require(m['soft'] is True and m['threshold'] == 'none' and m['native_resolution'] == [128, 128] and
            m['wavelet_domain_resize'] == 'FORBIDDEN' and m['tv_resolution'] == [128, 128] and
            m['artifact_map_resize'] == 'M_256 = bilinear_interpolate(M, size=(256, 256), align_corners=False)', 'R-02 mask')
    a = s['artifact_map']
    require((a['A_rgb'], a['A_freq_256'], a['M_256'], a['u'], a['A'], a['denominator_applies_to'], a['per_image_min_max']) ==
            ('mean_channel(abs(x_hat - x_t))', 'bilinear_x2(A_freq_128, align_corners=False)',
             'bilinear_x2(M, align_corners=False)', 'M_256 * (0.5 * A_rgb + 0.5 * A_freq_256)', 'clip(u / 2.0, 0.0, 1.0)',
             'u', False), 'D10.4 whole-u normalization')
    require('gamma * abs(delta_LL)' in a['A_freq_128'] and 'clip(A_rgb / 2, 0, 1)' in a['forbidden_readings'], 'A_freq')
    o = s['operators']
    require((o['S_orient_angular_bins'], o['face_mask_dilation_kernel'], o['film'], o['x_hat_clamp_during_training']) ==
            (8, [15, 15], '(1 + gamma) * h + beta', False) and o['parser']['foreground'] == list(range(1, 11)) and
            o['parser']['background'] == [0] and o['parser']['semantic_names'] is None, 'shared operators')
    fs = a10['frozen_spec_facts']
    require(s['curriculum']['stages'] == fs['curriculum'], 'curriculum equals A10 frozen facts')
    pr = s['precision']
    require(pr['autocast'] == {'device': 'cuda', 'dtype': 'float16'} and pr['bf16'] == 'FORBIDDEN' and pr['grad_scaler'] is True,
            'R-06 autocast fp16 + GradScaler, no bf16')
    require(pr['autocast_fp16_modules'] == ['E_art', 'G_res', 'PatchGAN_D', 'attack_type_head_when_enabled',
                                            'identity_adversary_head_when_enabled'], 'R-06 fp16 modules')
    for op in ('DWT', 'IDWT', 'x_hat_reconstruction', 'artifact_map_A', 'FFT', 'S_radial', 'S_orient', 'CE_logits_and_loss_reduction',
               'Dice', 'KL', 'frozen_teacher_and_evaluator_forwards'):
        require(op in pr['fp32_outside_autocast'], 'R-06 fp32 ' + op)
    require(pr['fp32_frozen_modules'] == ['AdaFace_F_id', 'FaceXFormer', 'F_art', 'ArtifactProbe_v1_val_selection'] and
            pr['x_hat_detach'] == 'FORBIDDEN' and pr['detached'] == ['x_t_teacher_targets'], 'R-06 teachers fp32, x_hat live')
    require(pr['grad_clip']['max_norm'] == 1.0 and pr['grad_clip']['order'][0] == 'GradScaler.unscale_(optimizer)' and
            pr['grad_clip']['scope'] == 'per_optimizer_trainable_parameters' and
            pr['grad_clip']['applies_to'] == ['G_OPT', 'D_OPT', 'WARMUP_OPT'], 'R-06 clip after unscale, per optimizer')
    gs = pr['grad_scalers']
    require((gs['G_SCALER']['owns'], gs['D_SCALER']['owns'], gs['WARMUP_SCALER']['owns']) == ('G_OPT', 'D_OPT', 'WARMUP_OPT')
            and gs['independent_state_machines'] is True and gs['shared_scaler_state'] == 'FORBIDDEN' and
            gs['warmup_state_reused_by_G_SCALER'] is False and gs['class'] == 'IMPLEMENTATION_CLARIFICATION', 'scaler topology')
    require(gs['per_optimizer_sequence'] == ['scaler.scale(loss).backward()', 'scaler.unscale_(optimizer)',
                                             "clip_grad_norm_(that optimizer's trainable parameters, 1.0)",
                                             'scaler.step(optimizer)', 'scaler.update()'], 'scaler sequence')
    op = s['optimizer_ownership']
    require(op['G_OPT_membership'] == MEMBERSHIP and op['identity_head_in_G_OPT'] is True, 'R-07 G_OPT membership')
    require(op['G_OPT'] == {'name': 'Adam', 'lr': 2e-4, 'betas': [0.5, 0.999], 'weight_decay': 0.0}, 'G_OPT hyperparameters')
    require(op['D_OPT']['params'] == ['PatchGAN_D'] and op['D_OPT']['separate'] is True and op['D_OPT']['name'] == 'Adam' and
            (op['D_OPT']['lr'], op['D_OPT']['betas'], op['D_OPT']['weight_decay']) == (2e-4, [0.5, 0.999], 0.0), 'D_OPT')
    require(op['WARMUP_OPT']['params'] == ['E_art', 'attack_type_head'] and op['WARMUP_OPT']['variants'] == ['B1', 'B3'] and
            op['fresh_G_OPT_after_warmup'] is True and op['frozen_modules_in_any_optimizer'] == 'never', 'R-07 warmup/frozen')
    require('D_OPT_not_decided_by_a10_or_m7b' not in op, 'no open D item')
    de = op['D_execution']
    require((de['class'], de['status']) == ('FROZEN_SPEC_DERIVED_EXECUTION_POLICY', 'RESOLVED_FROM_FROZEN_SPEC') and
            'spec_10_5' in de['authority'], 'M7B-OBS-01 resolved from spec 10.5')
    require(de['lr_schedule'] == {'granularity': 'per_optimizer_step', 'warmup': 'linear_from_0', 'warmup_epochs': 5,
                                  'warmup_updates': 5525, 'peak_lr': 2e-4, 'decay': 'cosine', 'min_lr': 2e-6,
                                  'end': 'final_generator_update', 'same_as': 'G_OPT'}, 'D LR schedule')
    require(de['amp'] == {'autocast': 'cuda_fp16', 'modules': ['PatchGAN_D']} and de['grad_scaler'] == 'D_SCALER' and
            de['grad_clip'] == {'max_norm': 1.0, 'params': 'PatchGAN_D_trainable_only', 'after': 'D_SCALER.unscale_(D_OPT)'},
            'D AMP / D_SCALER / D clip')
    t, ar = s['training_budget'], m7_arithmetic()
    for k in ('microbatches_per_epoch', 'optimizer_updates_per_epoch', 'generator_updates_per_run', 'tail_group_samples'):
        require(t[k] == ar[k], 'budget ' + k)
    require((t['generator_epochs'], t['train_pairs'], t['physical_batch'], t['grad_accum'], t['drop_last'], t['seeds'],
             t['attack_warmup_reduces_generator_epochs']) == (60, 8838, 4, 2, False, SEEDS, False), 'budget')
    require((t['microbatches_per_epoch'], t['optimizer_updates_per_epoch'], t['generator_updates_per_run']) ==
            (2210, 1105, 66300), '2210 / 1105 / 66300')
    require(t['lr_schedule'] == {'granularity': 'per_optimizer_step', 'warmup': 'linear_from_0',
                                 'warmup_updates': ar['lr_warmup_updates'], 'decay': 'cosine', 'min_lr': 2e-6}, 'LR schedule')
    require(t['ema'] == {'decay': 0.999, 'start': 'end_of_epoch_5', 'start_update': ar['lr_warmup_updates'],
                         'scope': ['E_art', 'G_res']}, 'EMA')
    c = s['checkpoint_selection']
    require(c['candidates'] == {'kind': 'EMA', 'epochs': [10, 60], 'inclusive': True, 'N': ar['val_candidates']} and
            ar['val_candidates'] == 51, '51 EMA candidates')
    require((c['training_process_reads_val'], c['evaluate_all_before_selecting'], c['data'], c['test_participates']) ==
            (False, True, 'VAL_only', False), 'VAL only, TEST never')
    require(c['weights'] == fs['selection_weights'] and c['tie_break'] == ['higher_raw_ArtSim', 'earlier_epoch'] and
            c['ranking']['ties'] == 'average_rank' and c['ranking']['percentile'] == 'p = (rank - 1) / (51 - 1)' and
            c['nme_normalizer'] == 'bounding_box_of_target_68_landmarks' and 'ArtifactProbe v1' in c['artsim'], 'selection')
    require(c['val_pairs'] == PAIRS_VAL[0], 'VAL pairs')
    vr = c['val_representation']
    require((vr['source'], vr['range'], vr['uint8_roundtrip'], vr['encoded_image_roundtrip'], vr['clamp_for_val_scoring'],
             vr['export_clamp_roundtrip'], vr['policy'], vr['policy_class']) ==
            ('x_hat_float', [-1.0, 1.0], 'FORBIDDEN', 'FORBIDDEN', False, 'FORBIDDEN', 'VAL_PRE_EXPORT_FLOAT_NO_CLAMP',
             'OWNER_EVALUATION_CLARIFICATION') and
            vr['applies_to_metrics'] == ['ID', 'Dice', 'ArtSim', 'NME', 'LFErr'], 'R-03 float VAL path')
    sr = s['sanity_references']
    require(sr['executed_in_m7b'] is False and sr['dwt_idwt'] == {'input': 'uniform[-1,1]', 'shape': [100, 3, 256, 256],
            'generator': 'cpu', 'seed': 42, 'dtype': 'float32', 'metric': 'max_abs', 'pass_if_less_than': 1e-5}, 'D11')
    require(sr['zero_residual']['disables'] == ['delta_HF', 'delta_LL'] and
            sr['zero_residual']['same_x_t_different_x_s_max_abs_lt'] == 1e-5, 'D12')
    e = s['environment']
    require(e['status'] == 'NOT_CREATED_M7B' and HEX40.match(e['nafnet']['commit']) and e['nafnet']['commit'] == NAFNET_COMMIT
            and e['nafnet']['weights'] == 'none' and sorted(e['deferred_to_m7c']) == R_DEFERRED, 'environment / NAFNet pin')


def check_union(cfgs):
    b1, b2, b3 = cfgs['B1'], cfgs['B2'], cfgs['B3']
    require(b3['attack_type_supervision'] == b1['attack_type_supervision'] and b3['attack_warmup'] == b1['attack_warmup'],
            'B3 attack blocks equal B1')
    require(b3['identity_adversary'] == b2['identity_adversary'], 'B3 identity adversary equals B2')
    require(b1['shared_contract'] == b2['shared_contract'] == b3['shared_contract'], 'shared contract identical')


def check_configs(reader):
    a10 = check_bindings(reader)
    cfgs = load_configs(reader)
    out = {}
    for vid in VARIANTS:
        raw = reader(CONFIGS[vid])
        require(raw == reader(SNAPSHOTS[vid]), vid + ' live config bytes == snapshot bytes')
        out[vid] = {'config': CONFIGS[vid], 'config_sha256': sha(raw), 'snapshot': SNAPSHOTS[vid],
                    'snapshot_sha256': sha(reader(SNAPSHOTS[vid])),
                    'b0_deltas': check_inheritance(cfgs['B0'], cfgs[vid], vid), **check_variant(cfgs[vid], vid)}
        check_shared(cfgs[vid], a10)
    check_union(cfgs)
    shared = [reader(CONFIGS[v]).split(b'\n# ---------------------------------------------------------------- shared')[1]
              for v in VARIANTS]
    require(shared[0] == shared[1] == shared[2], 'shared contract byte-identical')
    matrix = comparison_matrix(cfgs)
    return {'configs': out, 'matrix': matrix, 'matrix_counts': {c: sum(r['class'] == c for r in matrix) for c in MATRIX_CLASSES},
            'clarifications': check_clarifications(json.loads(reader(CLARIFICATIONS).decode('utf-8'))),
            'clarifications_sha256': sha(reader(CLARIFICATIONS)), 'arithmetic': m7_arithmetic(),
            'a10_document_sha256': A10_DOC_SHA, 'a10_record_sha256': A10_SHA, 'b0_sha256': GPAT_B0_SHA}


# ----------------------------------------------------------------- comparison matrix
PROVENANCE_GROUPS = ('variant', 'status', 'provenance')
PROVENANCE_PATHS = ('method', 'supervision', 'data.dataset_scope_authority')


def classify(path, vals, b0_field):
    b0, b1, b2, b3 = vals
    group = path.split('.')[0]
    if path in PROVENANCE_PATHS or group in PROVENANCE_GROUPS:
        return 'PROVENANCE_ONLY'
    if group == 'shared_contract':
        return 'IMPLEMENTATION_CLARIFICATION' if b1 == b2 == b3 else 'UNEXPLAINED'
    if b0_field:
        if b0 == b1 == b2 == b3:
            return 'SAME'
        if b1 != b0 and b2 == b0 and b3 == b1:
            return 'B1_DELTA'
        if b2 != b0 and b1 == b0 and b3 == b2:
            return 'B2_DELTA'
        return 'UNEXPLAINED'
    if b1 == b2 == b3:
        return 'SAME'
    if group in ('attack_type_supervision', 'attack_warmup') or path == 'data.splits.TRAIN.used_for':
        return 'B1_DELTA' if b3 == b1 != b2 else 'UNEXPLAINED'
    if group == 'identity_adversary':
        return 'B2_DELTA' if b3 == b2 != b1 else 'UNEXPLAINED'
    if path == 'data.label_fields_used':
        return 'B3_UNION' if b3 == b1 + b2 else 'UNEXPLAINED'
    return 'UNEXPLAINED'


def comparison_matrix(cfgs):
    flat = {v: dict(leaves(c)) for v, c in cfgs.items()}
    b0_paths = set(flat['B0'])
    paths = list(dict.fromkeys(p for v in VARIANTS for p in flat[v]))
    rows = []
    for p in paths:
        vals = [flat['B0'].get(p) if p in b0_paths else None, *(flat[v].get(p) for v in VARIANTS)]
        cls = classify(p, vals, p in b0_paths)
        require(cls in MATRIX_CLASSES, f'unexplained difference at {p}: {vals}')
        b0_shown = vals[0] if p in b0_paths else ('same (A10/M7B shared)' if p.startswith('shared_contract.') else 'n/a')
        rows.append({'path': p, 'B0': b0_shown, 'B1': vals[1], 'B2': vals[2], 'B3': vals[3], 'class': cls})
    require(not [p for p in b0_paths if p not in paths], 'every B0 field compared')
    return rows


def check_absent(exists, listdir):
    require(listdir('methods/gpat') == ['.gitkeep'], 'methods/gpat holds only .gitkeep (no GPAT implementation)')
    require(not [n for n in listdir('environments') if 'gpat' in n.lower()], 'no GPAT environment lock')
    require(not exists('runs/m7') and not [n for n in listdir('runs') if 'gpat' in n.lower() or 'm7' in n.lower()],
            'no GPAT run / checkpoint / bank')


def build_evidence(result):
    return {'schema': 'gpat.m7b.gpat_config_freeze_evidence', 'schema_version': 1, 'milestone': MILESTONE,
            'classification': CLASSIFICATION, 'record_kind': RECORD_KIND, 'authority_commit': AUTHORITY,
            'a10': {'document': A10_DOC, 'document_sha256': A10_DOC_SHA, 'record': A10, 'record_sha256': A10_SHA},
            'b0': {'config': GPAT_B0, 'sha256': GPAT_B0_SHA, 'snapshot_sha256': GPAT_B0_SHA, 'modified': False},
            'clarifications': {'record': CLARIFICATIONS, 'sha256': result['clarifications_sha256'],
                               **result['clarifications']},
            'evidence_markdown': {'path': EV_MD, 'sha256': sha(read(EV_MD))},
            'configs': result['configs'], 'arithmetic': result['arithmetic'], 'matrix_counts': result['matrix_counts'],
            'comparison_matrix': result['matrix'], 'nafnet_commit': NAFNET_COMMIT,
            'loader_or_validator_changes': [], 'M6_closed': True, 'M7_configs_frozen': True,
            'M7_implementation_started': False, 'environment_created': False, 'packages_installed': False,
            'M7_scientific_training_started': False, 'GPU_contacted': False, 'TEST_access': False,
            'TRAIN_access': False, 'VAL_access': False, 'training_runs': 0, 'checkpoint_writes': 0, 'bank_writes': 0}


def check_evidence(reader, result):
    ev = json.loads(reader(EV_JSON).decode('utf-8'))
    require(ev['authority_commit'] == AUTHORITY and ev['evidence_markdown']['sha256'] == sha(reader(EV_MD)), 'evidence binding')
    require(ev['configs'] == result['configs'] and ev['arithmetic'] == result['arithmetic'] and
            ev['comparison_matrix'] == result['matrix'] and ev['clarifications']['sha256'] == result['clarifications_sha256'],
            'evidence values')
    for k in ('M7_implementation_started', 'environment_created', 'M7_scientific_training_started', 'GPU_contacted',
              'TEST_access', 'TRAIN_access', 'VAL_access'):
        require(ev[k] is False, 'evidence flag ' + k)
    return ev


def check_config_status(reader, base):
    now, old = reader(CONFIG_STATUS), base(CONFIG_STATUS)
    require(now.startswith(old) and len(now) > len(old), 'CONFIG_STATUS append-only')
    added = now[len(old):].decode('utf-8')
    for phrase in ('CONFIGS_FROZEN', 'IMPLEMENTATION_NOT_STARTED', 'ENVIRONMENT_NOT_CREATED',
                   'SCIENTIFIC_TRAINING_NOT_STARTED', 'M6_CLOSED = true'):
        require(phrase in added, 'CONFIG_STATUS M7B ' + phrase)
    return len(now) - len(old)


# ----------------------------------------------------------------- candidate-time checks (run once, here only)
def authority():
    require(git('branch', '--show-current').decode().strip() == BRANCH, 'branch')
    require(git('rev-parse', 'HEAD').decode().strip() == AUTHORITY, 'uncommitted candidate on the M7A authority')


def tracked_unchanged(bookkeeping):
    changed = sorted(git('diff', '--name-only', AUTHORITY).decode().split())
    require(changed == sorted([*MODIFIED, *bookkeeping]), 'only CONFIG_STATUS + ledger/index differ: ' + json.dumps(changed))
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
            'worktree holds exactly the M7B candidate: ' + json.dumps(status))


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
    'record_kind': RECORD_KIND, 'exit_code': 0, 'M6_closed': True, 'M7_started': True, 'M7_configs_frozen': True,
    'M7_implementation_started': False, 'M7_scientific_training_started': False,
    'configs_created': sorted(CONFIGS.values()), 'snapshots_created': sorted(SNAPSHOTS.values()),
    'experiments': ['E09', 'E10', 'E11'], 'owner_clarifications': R_RESOLVED, 'deferred_to_m7c': R_DEFERRED,
    'observations': {'M7B-OBS-01': 'RESOLVED_FROM_FROZEN_SPEC'}, 'new_deviation_numbers': [], 'amendment_created': False,
    'a10_document_sha256': A10_DOC_SHA, 'a10_record_sha256': A10_SHA, 'gpat_b0_sha256': GPAT_B0_SHA,
    'nafnet_commit': NAFNET_COMMIT, 'gpat_code_created': False, 'environment_created': False, 'packages_installed': False,
    'loader_or_validator_changes': [], 'method_status_modified': False, 'stage_state_modified': False,
    'TEST_access': False, 'TRAIN_access': False, 'VAL_access': False, 'GPU_contacted': False, 'M8_bank': False,
    'training_runs': 0, 'scientific_runs': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'commit': False,
    'push': False, 'authority_commit': AUTHORITY, 'git_commit': AUTHORITY, 'committed_prefix_rows': LEDGER_PREFIX_ROWS,
    'committed_prefix_sha256': LEDGER_PREFIX_SHA, 'artifact_index_rows_before': INDEX_BASELINE_ROWS,
    'artifact_index_rows_after': INDEX_BASELINE_ROWS + len(NEW), 'modified_existing_files': list(MODIFIED),
    'input_artifacts': [SPEC, A10_DOC, A10, GPAT_B0, SNAP + GPAT_B0, CONFIG_STATUS,
                        'outputs/audit/M6E_FINAL_M6_CLOSURE.json', 'outputs/audit/M7A_GPAT_CONTRACT_RESOLUTION.json'],
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
            'purpose': 'M7B: GPAT B1/B2/B3 config freeze (E09/E10/E11, Track B) from B0 + A10 + owner clarifications '
                       'R-01/R-02/R-03/R-06/R-07; no GPAT code, environment, training, checkpoint or bank',
            'notes': ('B1 = B0 + attack head Linear(512,6) + lambda_type 0.2 + 10-epoch warmup (A10 D04-D06). B2 = B0 + '
                      'GRL(1.0) identity head Linear(512,60) with + lambda_idadv 0.1 CE, DEV-022 SiW masking (A10 D02/D03). '
                      'B3 = exact union. Shared: R-01 ptwt cA/cH/cV/cD = LL/LH/HL/HH; R-02 M at 128 in the wavelet domain, '
                      'bilinear M_256 only for A; R-03 VAL on float x_hat; R-06 fp16 autocast for trainable nets, fp32 '
                      'boundaries; R-07 heads in G_OPT. R-04/R-05 deferred to M7C. M7B-OBS-01 resolved from frozen spec 10.5 '
                      '(D Adam 2e-4, same warmup+cosine schedule, fp16, D_SCALER, clip 1.0 on D only); separate '
                      'G_SCALER/D_SCALER/WARMUP_SCALER; identity head bias=True and VAL pre-export float no-clamp owner-confirmed. B0, A10, spec, method_status, M6E closure unchanged. No loader/validator change.'),
            'command': ('laptop only: python3 -B tools/m7b_gpat_config_freeze_preflight.py --write-evidence-json / '
                        '--before-ledger / --append-ledger / --rebuild-index; .venv/bin/python -B -m unittest discover '
                        '-s tests -p test_m7b_gpat_config_freeze.py'),
            'artifacts_sha256': {p: sha(read(p)) for p in sorted((*NEW, *MODIFIED))},
            'tests': tests, 'cwd': str(ROOT), 'host': socket.gethostname(), 'user': getpass.getuser(),
            'git_dirty': True, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def listdir_worktree(rel):
    return sorted(q.name for q in (ROOT / rel).iterdir())


def verify(stage):
    authority()
    bookkeeping = {'before_ledger': [], 'before_index': [LEDGER], 'final': [INDEX, LEDGER]}[stage]
    tracked_unchanged(bookkeeping)
    whitespace()
    result = check_configs(read)
    check_absent(lambda p: (ROOT / p).exists(), listdir_worktree)
    check_evidence(read, result)
    status_bytes = check_config_status(read, at_authority)
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
    require(not {'torch', 'pyarrow', 'numpy', 'PIL'} & set(sys.modules), 'static preflight')
    return {'status': 'PASS', 'stage': stage, 'ledger_rows': len(current.splitlines()),
            f'first_{LEDGER_PREFIX_ROWS}_rows_byte_identical': True, 'ledger_prefix_sha256': sha(prefix),
            'artifact_index_rows_before': INDEX_BASELINE_ROWS, 'artifact_index_rows_expected': count,
            'artifact_rows_added': len(NEW), 'modified_existing_files': [*MODIFIED, *bookkeeping],
            'config_status_bytes_added': status_bytes, 'matrix_counts': result['matrix_counts'],
            'configs': {v: {k: result['configs'][v][k] for k in ('config_sha256', 'snapshot_sha256')} for v in VARIANTS}}


def fmt(v):
    return json.dumps(v, ensure_ascii=False).replace('|', '\\|') if not isinstance(v, str) else v.replace('|', '\\|')


def print_matrix():
    result = check_configs(read)
    print('| Field | B0 | B1 | B2 | B3 | Class |\n|---|---|---|---|---|---|')
    for r in result['matrix']:
        print(f"| `{r['path']}` | {fmt(r['B0'])} | {fmt(r['B1'])} | {fmt(r['B2'])} | {fmt(r['B3'])} | {r['class']} |")
    print(json.dumps(result['matrix_counts']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-matrix', action='store_true')
    parser.add_argument('--write-evidence-json', action='store_true')
    parser.add_argument('--before-ledger', action='store_true')
    parser.add_argument('--append-ledger', action='store_true')
    parser.add_argument('--tests', help='JSON list of test-run summaries (with --append-ledger)')
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    if args.print_matrix:
        print_matrix()
        return
    if args.write_evidence_json:
        authority()
        require(not (ROOT / EV_JSON).exists(), 'fresh evidence')
        ev = build_evidence(check_configs(read))
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
