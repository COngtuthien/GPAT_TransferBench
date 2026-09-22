"""Frozen E05 + A2 + A5 identity verification; no scientific overrides."""
import hashlib
import subprocess
from pathlib import Path

import yaml

from methods.common.config import ROOT, FrozenConfigError, load_method_config
from methods.common.learned import authoritative, PreparationError

CONTRACT_COMMIT = '2b38b3a6d97afbac8e79c844bdf3e2c1880495bf'
OVERLAY = 'configs/amendments/e05_a5_blur_operator_resolution.yaml'
A2 = ('docs/spec/amendments/'
      'GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md')


def immutable_input(relative_path, *, path=None):
    """Bind execution input bytes to the committed owner-approved contract."""
    try:
        expected = subprocess.run(
            ['git', '-C', str(ROOT), 'show', f'{CONTRACT_COMMIT}:{relative_path}'],
            check=True, capture_output=True).stdout
        raw = (Path(path) if path is not None else ROOT / relative_path).read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenConfigError(f'missing immutable E05 input: {relative_path}') from exc
    if raw != expected:
        raise FrozenConfigError(f'changed immutable E05 input: {relative_path}')
    return raw, hashlib.sha256(raw).hexdigest()


def validate_contract(config=None, *, overlay_path=None):
    cfg = authoritative(config) if config is not None else load_method_config('E05')
    if cfg['method_id'] != 'E05':
        raise PreparationError('PCGAN adapter requires E05')
    raw, digest = immutable_input(OVERLAY, path=overlay_path)
    overlay = yaml.safe_load(raw)
    if (overlay['base_config_path'] != cfg['_runtime']['config_path'] or
            overlay['base_config_sha256'] != cfg['_runtime']['config_sha256']):
        raise FrozenConfigError('A5 base E05 config identity mismatch')
    _, a2_sha = immutable_input(A2)
    if a2_sha != cfg['amendment_a2_sha256']:
        raise FrozenConfigError('A2 identity mismatch')
    _, a5_sha = immutable_input(overlay['amendment_document'])
    if (cfg['source']['architecture_resolution']['port_stylegan_v1_adain_for_prose_citation_35'] or
            not cfg['source']['architecture_resolution']['disclosure_required_in_report']):
        raise PreparationError('E05 requires the disclosed A2 executable architecture resolution')
    return {'overlay': overlay, 'overlay_path': OVERLAY, 'overlay_sha256': digest,
            'authority_commit': CONTRACT_COMMIT,
            'A2': {'path': A2, 'sha256': a2_sha},
            'A5': {'path': overlay['amendment_document'], 'sha256': a5_sha}}
