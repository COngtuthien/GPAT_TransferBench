"""Verified reader of an E01/E02 bank directory (methods/common/bank_writer.py layout) for downstream use.

The reader re-verifies the bank (manifest SHA256 file, exact file set, every output SHA256 and pixel identity) before
returning anything. It returns metadata only; it decodes no image beyond what verify_bank_files already checks.
Synthetic outputs are spoof-labelled (y_true = 1). Failure rows are counted and never substituted (spec §7, §27).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from methods.common import bank_writer as bw


class BankReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyntheticSample:
    pair_id: str
    method_id: str
    seed: int
    path: Path
    output_sha256: str
    target_live_id: str
    source_spoof_id: str
    y_true: int = 1


def read_verified_bank(run_dir: str | Path, method_id: str, seed: int, *, global_positions: bool = False) -> dict:
    run_dir = Path(run_dir)
    if method_id not in bw.METHODS:
        raise BankReadError(f"no bank reader contract for {method_id!r}")
    integrity = bw.verify_bank_files(run_dir, method_id, global_positions=global_positions)
    if not integrity["ok"]:
        raise BankReadError(f"bank verification failed: {integrity['problems'][:5]}")
    rows = bw.read_manifest(run_dir)
    if {r["method_id"] for r in rows} != {method_id} or {r["experiment_seed"] for r in rows} != {seed}:
        raise BankReadError("bank manifest method/seed differ from the request")
    samples = [SyntheticSample(r["pair_id"], method_id, seed, run_dir / r["synthetic_output_path"], r["output_sha256"],
                               r["target_live_id"], r["source_spoof_id"])
               for r in rows if r["generation_status"] == "success"]
    failures = [{"pair_id": r["pair_id"], "failure_reason": r["failure_reason"]}
                for r in rows if r["generation_status"] != "success"]
    return {"samples": samples, "failures": failures, "rows": len(rows), "integrity": integrity,
            "bank_manifest_sha256": (run_dir / bw.BANK_MANIFEST_SHA).read_text().split()[0]}
