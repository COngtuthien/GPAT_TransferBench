"""Hash-bound M6D3d authority; Python 3.8 compatible, no framework/data imports."""
import hashlib
import json
from pathlib import Path

AUTHORITY = '90263169f0731b397dfa459d7ac94eaafd2c69ec'
OVERLAY = 'configs/amendments/e04_training_graph_resolution.yaml'
DOCUMENT = ('docs/spec/amendments/'
            'GPAT_TransferBench_v1_0_E04_Training_Graph_Owner_Resolution_Addendum_M6D3d.md')
DIGESTS = {
    OVERLAY: 'f62f28c8fe8e090f47436544287243c308fb73b32fb26b1b0e55e6bb6b98e3cb',
    DOCUMENT: '1d8ae0a4d5033c973539e13a50e601c34a61f90b10d7dee27e52f5e753ecc35e',
}


def load_contract(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    for relative, expected in DIGESTS.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('M6D3d authority changed: ' + relative)
    contract = json.loads((root / OVERLAY).read_text())
    for relative, expected in contract['immutable_inputs_sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen E04 input changed: ' + relative)
    base = root / 'configs/methods/e04_physics_std.yaml'
    if base.read_bytes() != (root / 'frozen_config_snapshot/configs/methods/e04_physics_std.yaml').read_bytes():
        raise ValueError('Frozen E04 snapshot differs')
    return contract
