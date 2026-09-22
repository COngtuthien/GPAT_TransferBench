"""Immutable E07c M6B + A1/A2/A3 + A6 authority, without editing base values."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import yaml

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import authoritative, PreparationError

AUTHORITY_COMMIT = '8535b6458a12bb4f87d86320c728ef9f13c8755d'
OVERLAY = 'configs/amendments/e07c_a6_feature_interface_source_correction.yaml'
AMENDMENTS = {
    'A1': 'GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md',
    'A2': 'GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md',
    'A3': 'GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md',
}


def immutable_input(relative_path, *, path=None):
    try:
        expected = subprocess.run(['git', '-C', str(ROOT), 'show',
                                   f'{AUTHORITY_COMMIT}:{relative_path}'],
                                  check=True, capture_output=True).stdout
        raw = (Path(path) if path is not None else ROOT / relative_path).read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenConfigError(f'missing immutable E07c input: {relative_path}') from exc
    if raw != expected:
        raise FrozenConfigError(f'changed immutable E07c input: {relative_path}')
    return raw, hashlib.sha256(raw).hexdigest()


def validate_contract(config=None, *, overlay_path=None, adaptation_path=None):
    cfg = authoritative(config) if config is not None else load_method_config('E07c')
    if cfg['method_id'] != 'E07c':
        raise PreparationError('DiffFAS adapter requires E07c')
    semantics = cfg['adaptation_semantics']
    raw, a1_sha = immutable_input(semantics['frozen_in'], path=adaptation_path)
    if a1_sha != semantics['frozen_config_sha256']:
        raise FrozenConfigError('A1 adaptation config SHA mismatch')
    snapshot, _ = immutable_input('frozen_config_snapshot/' + semantics['frozen_in'])
    if raw != snapshot:
        raise FrozenConfigError('A1 adaptation snapshot mismatch')
    a1 = yaml.safe_load(raw)
    raw, overlay_sha = immutable_input(OVERLAY, path=overlay_path)
    overlay = yaml.safe_load(raw)
    if (overlay['base_config_path'] != cfg['_runtime']['config_path'] or
            overlay['base_config_sha256'] != cfg['_runtime']['config_sha256']):
        raise FrozenConfigError('A6 base config identity mismatch')
    documents = {}
    for name, filename in AMENDMENTS.items():
        rel = 'docs/spec/amendments/' + filename
        _, digest = immutable_input(rel)
        if digest != cfg[f'amendment_{name.lower()}_sha256']:
            raise FrozenConfigError(f'{name} identity mismatch')
        documents[name] = {'path': rel, 'sha256': digest}
    if overlay['amendment_a3'] != documents['A3']:
        raise FrozenConfigError('A6/A3 binding mismatch')
    _, digest = immutable_input(overlay['amendment_document'])
    documents['A6'] = {'path': overlay['amendment_document'], 'sha256': digest}
    if any(overlay[k] for k in ('projection_layers_added', 'official_source_modified', 'architecture_substituted')):
        raise PreparationError('A6 authorizes no architecture or channel adapter')
    effective = deepcopy(cfg['conditioning_encoder'])
    for name, shape in overlay['correction'].items():
        effective['feature_interface_at_256'][name]['shape'] = deepcopy(shape)
    code = a1['official_code_path']
    if (code['use_pair'] or cfg['training']['use_pair'] or
            code['model_in_channels'] != cfg['training']['model_in_channels'] or
            code['content_training_role'] != 'INERT_API_PLACEHOLDER' or
            any(a1['identity_firewall'][k] for k in a1['identity_firewall'] if k != 'siw_participates_without_identity') or
            cfg['subject_id_global_consumed']):
        raise PreparationError('E07c requires the A1 unpaired, identity-free path')
    if (a1['training_manifest']['path'] != cfg['data']['training_manifest'] or
            a1['training_manifest']['rows'] != cfg['n_syn_intended']):
        raise PreparationError('A1 training manifest identity mismatch')
    return {'authority_commit': AUTHORITY_COMMIT, 'documents': documents,
            'adaptation': a1, 'adaptation_path': semantics['frozen_in'],
            'adaptation_sha256': a1_sha, 'overlay': overlay,
            'overlay_path': OVERLAY, 'overlay_sha256': overlay_sha,
            'effective_encoder': effective}


def guide_contract(config, contracts=None):
    cfg = authoritative(config)
    contracts = contracts or validate_contract(cfg)
    a1 = contracts['adaptation']
    return {'policy': deepcopy(a1['guide_selection']),
            'manifest': {**deepcopy(a1['training_manifest']),
                         'sha256': cfg['data']['training_manifest_sha256']},
            'identity_firewall': deepcopy(a1['identity_firewall']),
            'style_supervision': deepcopy(a1['style_supervision']),
            'manifest_opened': False, 'rows_decoded': False,
            'selection_mode': 'CONSUME_FROZEN_MANIFEST_NO_RESAMPLING',
            'future_manifest_byte_verification_required': True}


def training_record(record, config=None):
    """Whitelist metadata from an already-selected row; never choose a guide.

    No image paths are opened. Subject/attack identity fields are never inspected.
    This interface does not replace future verification of the manifest bytes.
    """
    cfg = config if config is not None else load_method_config('E07c')
    a1 = validate_contract(cfg)['adaptation']
    if (record['split'] != a1['training_manifest']['split'] or record['use_pair'] is not False or
            record['style_id'] != a1['style_supervision']['style_id'] or
            record['selection_policy'] != a1['guide_selection']['benchmark_policy']):
        raise PreparationError('record must come from the frozen TRAIN binary-spoof manifest')
    return {key: record[key] for key in ('gt_spoof_id', 'guide_spoof_id', 'dataset', 'split', 'style_id')}
