"""The common benchmark-facing generator interface shared by the M6 baselines.

The interface is intentionally small. It does not force upstream methods to share internals:
each method keeps its own frozen scientific contract behind `generate_one`.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class GeneratorResult:
    """One benchmark-facing generation record."""

    method_id: str
    global_seed: int
    pair_id: str
    inputs: dict          # input references (ids / paths / shapes); never pixel payloads
    output: Any           # in-memory array or an output reference
    metadata: dict        # deterministic recipe metadata (seed, operator/blocks, assets, ...)
    success: bool
    failure_reason: str | None
    elapsed_seconds: float

    def to_record(self, *, include_output_ref: str | None = None) -> dict:
        """JSON-serialisable log record. Pixel payloads are never serialised."""
        return {
            "method_id": self.method_id,
            "experiment_seed": self.global_seed,
            "pair_id": self.pair_id,
            "inputs": self.inputs,
            "output_ref": include_output_ref,
            "metadata": self.metadata,
            "generation_status": "success" if self.success else "failure",
            "failure_reason": self.failure_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


class MethodGenerator:
    """Minimal lifecycle shared by E01/E02 (and available to later methods).

    prepare()      -> validate the frozen contract and build any deterministic tables
    generate_one() -> produce exactly one GeneratorResult
    finalize()     -> return a compact summary dict
    """

    method_id: str = ""

    def prepare(self) -> "MethodGenerator":
        raise NotImplementedError

    def generate_one(self, *args, **kwargs) -> GeneratorResult:
        raise NotImplementedError

    def finalize(self) -> dict:
        raise NotImplementedError
