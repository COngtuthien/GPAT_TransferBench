"""Hash-verified, read-only loader of the frozen GPAT-B0/B1/B2/B3 method configs (M7C2b).

The four configs are frozen (B0 = spec section 23.2 verbatim; B1/B2/B3 = M7B freeze). The loader verifies the config
SHA-256, optionally its frozen_config_snapshot byte equality, and the SHA-256 of the M7C2b authority records, then
exposes the values as deeply immutable mappings. It never writes and never reinterprets a field. Torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parents[2]

# variant -> (experiment id, method name, config path, frozen SHA-256)
VARIANTS = {
    'B0': ('E08', 'GPAT-B0', 'configs/methods/gpat_b0.yaml',
           '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a'),
    'B1': ('E09', 'GPAT-B1', 'configs/methods/gpat_b1.yaml',
           '60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc'),
    'B2': ('E10', 'GPAT-B2', 'configs/methods/gpat_b2.yaml',
           '4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9'),
    'B3': ('E11', 'GPAT-B3', 'configs/methods/gpat_b3.yaml',
           '620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca'),
}
# Expected supervision flags (spec section 5.3, A10 D05/D02): (attack_type_head, identity_adversary)
SUPERVISION = {'B0': (False, False), 'B1': (True, False), 'B2': (False, True), 'B3': (True, True)}
VARIANT_WEIGHTS = {'B0': (0.0, 0.0), 'B1': (0.2, 0.0), 'B2': (0.0, 0.1), 'B3': (0.2, 0.1)}   # (lambda_type, lambda_idadv)
AUTHORITY_SHA256 = {
    'configs/amendments/gpat_a10_m7_contract_resolution.yaml':
        '57b37f09ca4279ad3999f4f9efa21fbf6fa39836574a0336bd13e34b2bd53203',
    'configs/amendments/gpat_m7b_owner_clarifications.yaml':
        'db13b2c0b17771bf62bcac1ee25e0a51bab60ff649846a186799310b6d8f4825',
    'configs/amendments/gpat_m7c2a_implementation_resolution.yaml':
        'b61555a2ca15954bdd8929d92c28d0d957561c595ea38eeef496d06c871e8e70',
}
# Frozen architecture facts (spec 9.3 / 23.2, A10 frozen_spec_facts); the model refuses any other value.
ARCHITECTURE = {'input_resolution': 256, 'gamma': 0.0, 'width': 32, 'enc_blocks': (2, 2, 4, 8), 'middle_blocks': 12,
                'dec_blocks': (2, 2, 2, 2), 'delta_scale_hf': 0.15, 'delta_scale_ll': 0.05, 'output_channels': 13,
                'input_channels': 12, 'highpass': (9, 1.5),
                'wavelet': (('library', 'ptwt'), ('family', 'haar'), ('level', 1), ('boundary_mode', 'reflect'))}


class GPATConfigError(ValueError):
    """The selected GPAT config or an authority record differs from its frozen state (STOP)."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _variant(name: str) -> str:
    for variant, (experiment, method, _, _) in VARIANTS.items():
        if name in (variant, experiment, method):
            return variant
    raise GPATConfigError(f'unknown GPAT experiment/variant id {name!r}')


def verify_authority(reader=None) -> dict:
    """SHA-256 of the A10, M7B and M7C2a records the static core implements; any change is refused."""
    reader = reader or (lambda rel: (ROOT / rel).read_bytes())
    out = {}
    for rel, digest in AUTHORITY_SHA256.items():
        got = _sha(reader(rel))
        if got != digest:
            raise GPATConfigError(f'authority record changed: {rel}')
        out[rel] = got
    return out


@dataclass(frozen=True)
class GPATConfig:
    variant: str
    experiment_id: str
    method: str
    path: str
    sha256: str
    attack_type_head: bool
    identity_adversary: bool
    lambda_type: float
    lambda_idadv: float
    gamma: float
    delta_scale_hf: float
    delta_scale_ll: float
    raw: MappingProxyType

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self                                                   # immutable: shared, never duplicated

    @property
    def loss(self):
        return self.raw['loss']

    @property
    def training(self):
        return self.raw['training']

    @property
    def residual_generator(self):
        return self.raw['residual_generator']


def _check_architecture(raw) -> None:
    ae, rg = raw['artifact_encoder'], raw['residual_generator']
    got = {'input_resolution': raw['input_resolution'], 'gamma': raw['gamma'], 'width': rg['width'],
           'enc_blocks': tuple(rg['enc_blocks']), 'middle_blocks': rg['middle_blocks'],
           'dec_blocks': tuple(rg['dec_blocks']), 'delta_scale_hf': rg['delta_scale_hf'],
           'delta_scale_ll': rg['delta_scale_ll'], 'output_channels': rg['output_channels'],
           'input_channels': ae['input_channels'], 'highpass': (ae['highpass']['kernel'], ae['highpass']['sigma']),
           'wavelet': tuple(sorted(raw['wavelet'].items()))}
    want = dict(ARCHITECTURE, wavelet=tuple(sorted(ARCHITECTURE['wavelet'])))
    if got != want or rg['name'] != 'NAFResidualUNet' or ae['backbone'] != 'torchvision_resnet18_imagenet1k_v1':
        raise GPATConfigError('config architecture differs from the frozen GPAT architecture')


def load_config(name: str, *, verify_snapshot: bool = True, reader=None) -> GPATConfig:
    """Load GPAT-B0..B3 (accepts B0, E08, GPAT-B0, ...) after hash verification; returns an immutable config."""
    variant = _variant(name)
    experiment, method, rel, digest = VARIANTS[variant]
    reader = reader or (lambda r: (ROOT / r).read_bytes())
    verify_authority(reader)
    data = reader(rel)
    if _sha(data) != digest:
        raise GPATConfigError(f'{rel} SHA-256 differs from the frozen config')
    if verify_snapshot and reader('frozen_config_snapshot/' + rel) != data:
        raise GPATConfigError(f'frozen snapshot of {rel} differs')
    raw = yaml.safe_load(data.decode('utf-8'))
    if raw.get('method') != method:
        raise GPATConfigError(f'{rel} names {raw.get("method")!r}, expected {method}')
    if variant != 'B0' and (raw['variant']['experiment_id'], raw['variant']['variant_id']) != (experiment, variant):
        raise GPATConfigError(f'{rel} variant block mismatch')
    _check_architecture(raw)
    ae, loss = raw['artifact_encoder'], raw['loss']
    flags = (ae['attack_type_head'], ae['identity_adversary'])
    weights = (float(loss['lambda_type']), float(loss['lambda_idadv']))
    if flags != SUPERVISION[variant] or weights != VARIANT_WEIGHTS[variant]:
        raise GPATConfigError(f'{variant} supervision flags/weights differ from spec 5.3')
    rg = raw['residual_generator']
    return GPATConfig(variant=variant, experiment_id=experiment, method=method, path=rel, sha256=digest,
                      attack_type_head=flags[0], identity_adversary=flags[1], lambda_type=weights[0],
                      lambda_idadv=weights[1], gamma=float(raw['gamma']), delta_scale_hf=float(rg['delta_scale_hf']),
                      delta_scale_ll=float(rg['delta_scale_ll']), raw=_freeze(raw))
