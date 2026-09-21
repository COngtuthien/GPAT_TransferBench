"""E01 — FAS-Aug deterministic generator (frozen contract: configs/methods/e01_fas_aug.yaml)."""
from .e01 import (
    AssetIndex,
    E01Generator,
    OperatorBackend,
    ToyOperatorBackend,
    enumerate_assets,
    magnitude_for_level,
    operator_seed,
    operator_seed_preimage,
    select_operator,
)

__all__ = [
    "OfficialFASAugBackend", "OfficialBackendError", "AssetIndex", "E01Generator", "OperatorBackend", "ToyOperatorBackend",
    "enumerate_assets", "magnitude_for_level", "operator_seed",
    "operator_seed_preimage", "select_operator",
]

from .official import OfficialFASAugBackend, OfficialBackendError
