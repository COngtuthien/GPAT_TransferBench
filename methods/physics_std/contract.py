"""Immutable E04 base + A3 + A4 authority. No merged scientific defaults."""
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import authoritative, PreparationError

# Provenance anchor, not a scientific parameter registry. M6A6's final committed
# document bytes are authoritative (its draft audit retains an earlier MD hash).
CONTRACT_COMMIT = '2bf69b35858d0ae416547a4e320ff9ec9de93bc6'
OVERLAY = 'configs/amendments/e04_a4_execution_resolution.yaml'
A3 = ('docs/spec/amendments/'
      'GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md')


def committed_input(relative_path, *, path=None):
    """Verify an execution input against the immutable M6A6 Git object."""
    expected = subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{CONTRACT_COMMIT}:{relative_path}'],
        check=True, capture_output=True).stdout
    target = Path(path) if path is not None else ROOT / relative_path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise FrozenConfigError(f'missing immutable E04 input: {relative_path}') from exc
    if raw != expected:
        raise FrozenConfigError(f'changed immutable E04 input: {relative_path}')
    return raw, hashlib.sha256(raw).hexdigest()


def validate_overlay(config=None, *, overlay_path=None):
    cfg = authoritative(config) if config is not None else load_method_config('E04')
    if cfg['method_id'] != 'E04':
        raise PreparationError('E04 contract requires method E04')
    raw, digest = committed_input(OVERLAY, path=overlay_path)
    overlay = yaml.safe_load(raw)
    if (overlay['base_config_path'] != cfg['_runtime']['config_path'] or
            overlay['base_config_sha256'] != cfg['_runtime']['config_sha256']):
        raise FrozenConfigError('A4 base E04 config identity mismatch')
    _, a3_digest = committed_input(A3)
    if a3_digest != cfg['amendment_a3_sha256']:
        raise FrozenConfigError('A3 identity mismatch')
    _, a4_digest = committed_input(overlay['amendment_document'])
    return {'overlay': overlay, 'overlay_path': OVERLAY, 'overlay_sha256': digest,
            'authority_commit': CONTRACT_COMMIT,
            'A3': {'path': A3, 'sha256': a3_digest},
            'A4': {'path': overlay['amendment_document'], 'sha256': a4_digest}}


def validate_q140(config=None, *, path=None):
    cfg = authoritative(config) if config is not None else load_method_config('E04')
    contract = cfg['controlled_reconstruction']['q140_vertex_set']
    raw, file_hash = committed_input(contract['frozen_list'], path=path)
    record = json.loads(raw)
    indices = record['vertex_indices']
    digest = hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()
    if (digest != contract['vertex_indices_sha256'] or
            digest != record['vertex_indices_sha256'] or
            len(indices) != contract['Q'] or len(set(indices)) != len(indices) or
            any(type(i) is not int or not 0 <= i < record['num_vertices_total'] for i in indices) or
            record['num_anchors'] + record['num_fps_extension'] != len(indices) or
            not record['image_independent'] or not contract['image_independent'] or
            contract['rederived_per_experiment_seed']):
        raise FrozenConfigError('invalid frozen Q140 ordered vertex contract')
    return {**record, 'file_sha256': file_hash,
            'anchor_indices': indices[:record['num_anchors']],
            'fps_indices': indices[record['num_anchors']:],
            'rederived': False}
