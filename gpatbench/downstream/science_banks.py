"""Downstream synthetic-bank resolution for the seven previous-work methods (M10 scientific detector inputs).

Contract: configs/amendments/m8_downstream_evaluator_contract.yaml D12 SEED_MATCHED_RAW -- detector seed s uses the bank
of generator seed s; the synthetic pool is every valid output of that bank (RAW-MATCHED, spec section 7); generation
failures are excluded and counted, never substituted; nothing is filtered, enhanced or re-ranked by appearance.

Banks are resolved only through their audited locations and bound to the committed audit records:
  E01 / E02  (methods/common/bank_writer.py layout, runtime root on the system SSD): the family closure
             outputs/audit/E0x_PRODUCTION_FAMILY_CLOSURE.json per_seed run_dir and bank_manifest_sha256; full
             verification by gpatbench.downstream.bank_reader.read_verified_bank (bank_writer.verify_bank_files).
  E03 / E04 / E05 / E06b / E06c  (methods/common/learned_bank.py layout, approved HDD): the activated registry
             (lb.resolve_registered_bank, never a symlink or a typed path) and the committed
             outputs/audit/LEARNED_BANK_PRODUCTION_<M>_SEED<S>.json (production_audit_passed, manifest_sha256,
             provenance_sha256), the family closure LEARNED_BANK_PRODUCTION_<M>_FAMILY_CLOSURE.json; full verification by
             lb.read_bank (manifest, every file and pixel SHA256).
Every sample is labelled spoof (y_true = 1) and carries a stable sample_id: pair_id for pair methods, synthetic_id for
DSDG (E06b / E06c). The physical filesystem differs by family; nothing is moved.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

METHODS = ("E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
BANK_WRITER_METHODS = ("E01", "E02")
LEARNED_METHODS = ("E03", "E04", "E05", "E06b", "E06c")
DSDG_METHODS = ("E06b", "E06c")
SEEDS = (42, 1337, 2026)
N_SYN = 8838
CANONICAL = 256
CLOSURE_E0102 = "outputs/audit/{M}_PRODUCTION_FAMILY_CLOSURE.json"
CLOSURE_E0102_STATUS = "{M}_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED"
RECORD_LEARNED = "outputs/audit/LEARNED_BANK_PRODUCTION_{M}_SEED{S}.json"
CLOSURE_LEARNED = "outputs/audit/LEARNED_BANK_PRODUCTION_{M}_FAMILY_CLOSURE.json"
CLOSURE_LEARNED_STATUS = "LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED"
READINESS_RECORD = "outputs/audit/PREVIOUS_WORK_SEVEN_BASELINE_BANK_READINESS.json"
READINESS_STATUS = "ALL_SEVEN_PREVIOUS_WORK_BANK_FAMILIES_COMPLETED_AND_AUDITED"


class ScienceBankError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str            # pair_id (pair methods) or synthetic_id (DSDG)
    method_id: str
    bank_seed: int
    row_index: int
    path: str
    output_sha256: str
    output_pixel_sha256: str
    pair_id: str | None
    target_live_id: str | None
    source_spoof_id: str | None
    y_true: int = 1


def _json(repo: Path, rel: str) -> tuple[dict, str]:
    p = Path(repo) / rel
    if not p.is_file():
        raise ScienceBankError(f"missing committed record {rel}")
    raw = p.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def bank_binding(repo: str | Path, method_id: str, seed: int) -> dict:
    """The committed audit binding of one bank (expected path, manifest sha256, audit and closure identities)."""
    if method_id not in METHODS or seed not in SEEDS or type(seed) is not int:
        raise ScienceBankError(f"no previous-work bank {method_id!r} seed {seed!r}")
    repo = Path(repo)
    if method_id in BANK_WRITER_METHODS:
        clo, clo_sha = _json(repo, CLOSURE_E0102.format(M=method_id))
        if clo.get("status") != CLOSURE_E0102_STATUS.format(M=method_id):
            raise ScienceBankError(f"{method_id} family closure is not COMPLETED_AND_AUDITED")
        ps = clo["per_seed"][str(seed)]
        return {"method_id": method_id, "seed": seed, "family": "bank_writer", "path": ps["run_dir"],
                "manifest_sha256": ps["bank_manifest_sha256"], "audit_record": ps["audit_record"],
                "audit_record_sha256": ps["audit_record_sha256"], "audit_gates": ps["audit_gates"],
                "closure": CLOSURE_E0102.format(M=method_id), "closure_sha256": clo_sha,
                "output_count": ps["output_count"], "failure_count": ps["failure_count"]}
    rec_rel = RECORD_LEARNED.format(M=method_id, S=seed)
    rec, rec_sha = _json(repo, rec_rel)
    clo, clo_sha = _json(repo, CLOSURE_LEARNED.format(M=method_id))
    if rec.get("production_audit_passed") is not True or rec.get("failed_gates") != [] \
            or rec.get("method") != method_id or rec.get("seed") != seed:
        raise ScienceBankError(f"{rec_rel} is not a passed audit of {method_id} seed {seed}")
    if clo.get("status") != CLOSURE_LEARNED_STATUS:
        raise ScienceBankError(f"{method_id} family closure is not COMPLETED_AND_AUDITED")
    return {"method_id": method_id, "seed": seed, "family": "learned_bank", "path": rec["bank"]["path"],
            "manifest_sha256": rec["bank"]["manifest_sha256"], "provenance_sha256": rec["bank"]["provenance_sha256"],
            "audit_record": rec_rel, "audit_record_sha256": rec_sha,
            "audit_gates": "%d/%d" % (rec["gates_passed"], rec["gates_total"]), "run_id": rec["run_id"],
            "closure": CLOSURE_LEARNED.format(M=method_id), "closure_sha256": clo_sha,
            "output_count": rec["bank"]["outputs"], "failure_count": rec["bank"]["failures"]}


def resolve_bank(repo: str | Path, method_id: str, seed: int, *, verify_pixels: bool = True) -> dict:
    """Bank of (method, generator seed): path from its family's registry, full verification by the family reader,
    manifest sha256 equal to the committed audit. Returns {'binding', 'samples', 'failures', 'reader'}."""
    b = bank_binding(repo, method_id, seed)
    if method_id in BANK_WRITER_METHODS:
        from methods.common import bank_writer as bw
        from gpatbench.downstream.bank_reader import read_verified_bank
        d = Path(b["path"])
        got = read_verified_bank(d, method_id, seed)
        if got["bank_manifest_sha256"] != b["manifest_sha256"] \
                or hashlib.sha256((d / bw.BANK_MANIFEST).read_bytes()).hexdigest() != b["manifest_sha256"]:
            raise ScienceBankError(f"{method_id} seed {seed}: manifest differs from the audited value")
        rows = {r["pair_id"]: r for r in bw.read_manifest(d)}
        samples = [SyntheticSample(s.pair_id, method_id, seed, int(rows[s.pair_id]["position"]), str(s.path),
                                   s.output_sha256, rows[s.pair_id]["output_pixel_sha256"], s.pair_id,
                                   s.target_live_id, s.source_spoof_id) for s in got["samples"]]
        reader = {"library": "gpatbench.downstream.bank_reader.read_verified_bank", "ok": got["integrity"]["ok"],
                  "rows": got["rows"], "files_checked": got["integrity"]["files_checked"]}
        failures = got["failures"]
    else:
        from methods.common import learned_bank as lb
        d = lb.resolve_registered_bank(method_id, seed)
        if str(d) != b["path"]:
            raise ScienceBankError(f"{method_id} seed {seed}: registry path {d} differs from the audited path")
        rb = lb.read_bank(d, expect_method=method_id, expect_seed=seed, expect_mode="PRODUCTION", expect_rows=N_SYN,
                          expect_checkpoint_identity=lb.checkpoint_identity_sha256(method_id, seed),
                          verify_pixels=verify_pixels)
        if not rb["ok"]:
            raise ScienceBankError(f"{method_id} seed {seed}: bank verification failed {rb['problems'][:5]}")
        if rb["manifest_sha256"] != b["manifest_sha256"]:
            raise ScienceBankError(f"{method_id} seed {seed}: manifest differs from the audited value")
        samples, failures = [], []
        for r in rb["rows"]:
            if r["generation_status"] != "success":
                failures.append({"sample_id": r["row_id"], "failure_reason": r["failure_reason"]})
                continue
            if r["label"] != "spoof" or r["method_id"] != method_id or r["bank_seed"] != seed:
                raise ScienceBankError(f"{method_id} seed {seed}: row {r['row_id']} identity")
            sid = r["synthetic_id"] if method_id in DSDG_METHODS else r["pair_id"]
            if sid != r["row_id"]:
                raise ScienceBankError(f"{method_id}: row id {r['row_id']} is not its sample id")
            samples.append(SyntheticSample(sid, method_id, seed, int(r["row_index"]), str(d / r["output_path"]),
                                           r["output_sha256"], r["output_pixel_sha256"], r["pair_id"],
                                           r["target_live_id"], r["source_spoof_id"]))
        reader = {"library": "methods.common.learned_bank.read_bank", "ok": rb["ok"], "rows": len(rb["rows"]),
                  "files_checked": rb.get("files_checked"), "verify_pixels": verify_pixels}
    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ScienceBankError(f"{method_id} seed {seed}: duplicate sample ids")
    if [s.row_index for s in samples] != sorted(s.row_index for s in samples):
        raise ScienceBankError(f"{method_id} seed {seed}: rows not in canonical order")
    if len(samples) + len(failures) != N_SYN or len(failures) != b["failure_count"]:
        raise ScienceBankError(f"{method_id} seed {seed}: {len(samples)} outputs + {len(failures)} failures")
    return {"binding": b, "samples": samples, "failures": failures, "reader": reader,
            "samples_sha256": hashlib.sha256("\n".join(f"{s.sample_id}|{s.output_sha256}" for s in samples)
                                             .encode()).hexdigest()}


def load_synthetic_image(sample: SyntheticSample):
    """-> PIL RGB 256x256: bytes re-hashed against the audited output_sha256 before decoding; no conversion,
    enhancement or quality test (E04 clipped outputs and E05 low-contrast outputs are returned exactly as stored)."""
    from PIL import Image
    raw = Path(sample.path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample.output_sha256:
        raise ScienceBankError(f"synthetic output sha256 mismatch {sample.method_id} {sample.sample_id}")
    with Image.open(io.BytesIO(raw)) as im:
        if im.format != "PNG" or im.mode != "RGB" or im.size != (CANONICAL, CANONICAL):
            raise ScienceBankError(f"{sample.sample_id} is not a 256x256 RGB PNG")
        return im.copy()


def pixel_identity(img) -> str:
    """Encoder-independent identity of a decoded image (the definition of both bank families' manifests)."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(img))
    return hashlib.sha256(f"{a.dtype.str}|{'x'.join(map(str, a.shape))}|".encode() + a.tobytes()).hexdigest()
