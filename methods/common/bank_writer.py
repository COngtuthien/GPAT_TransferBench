"""E01/E02 synthetic-bank writer (non-learned direct transformation banks).

Scope. This module turns the frozen E01/E02 generators (`methods/fas_aug/e01.py`, `methods/freq_sub/e02.py`) into a
per-seed bank at the frozen M6B run layout `<runtime_root>/runs/m6/<method>/seed_<seed>/`. It changes no method
semantics: every pixel comes from the unchanged `generate_one`. It adds only input selection, integrity checks,
deterministic output naming, fail-closed writes, per-sample accounting and a bank manifest.

Authorization. A PRODUCTION bank needs the owner-approved `configs/amendments/e01_e02_bank_milestone_ruling.yaml`
(approved 2026-11-22: Q1 A_M8_SMOKE_GATED, Q2 MANIFEST_POSITION_ZERO_BASED, Q3 CONFIRM_CLIP_THEN_RINT). While that
file is absent, unapproved, incomplete or deviates from the approved decisions, `load_production_authorization`
raises `BankBlocked` with status `E01_E02_BLOCKED_BY_MILESTONE_RULING`. With the approved ruling, it raises
`BankBlocked` with status `E01_E02_BLOCKED_BY_M8_SMOKE_GATE` until a committed, hash-verified M8 deterministic
smoke-test record with smoke_test_passed true is bound in the ruling (M6D6jR O3). QUALIFICATION authorizations exist
only for synthetic fixtures and never accept the production faces root. SMOKE authorizations (M8 global smoke) need
the owner-approved configs/amendments/m8_smoke_test_contract.yaml; they accept only the toy population or a synthetic
smoke fixture, keep canonical global pair positions, read faces with O_NOATIME and write only under a separate smoke
root. The production CLI (tools/e01_e02_bank.py) never constructs a SMOKE authorization.

Inputs. Pairs come from the frozen `manifests/pairs_train_v1.parquet` (SHA256 checked before parse, every row TRAIN,
pair_ids ascending and unique, N rows = n_syn_intended). Every referenced sample must be TRAIN in `split_v1` and
COMPLETE in `manifests/m2_sample_accounting.parquet`. Each face is read from
`<runtime_root>/data/processed/faces_256/<dataset>/<sample_id>.png` and its bytes must match the M2 face_png_sha256
before decoding. The decoded image must be PIL mode RGB, 256x256; nothing is converted silently.

Failure policy. Input integrity violations (hash mismatch, missing file, wrong mode/size, non-TRAIN sample) abort
the whole run, which is closed as `failed` and keeps its partial evidence. A method-level exception for one pair
is recorded as that pair's failure (spec section 7 RAW-MATCHED / section 27: report the failure count; never
duplicate or substitute an image) and the run continues.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ROOT, load_method_config, sha256_file
from .runlog import RunContext, atomic_write_json

PAIRS_PATH = "manifests/pairs_train_v1.parquet"
SPLIT_PATH = "manifests/split_v1.parquet"
M2_ACCOUNTING_PATH = "manifests/m2_sample_accounting.parquet"
M2_ACCOUNTING_SHA256 = "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662"
RULING_PATH = "configs/amendments/e01_e02_bank_milestone_ruling.yaml"
RULING_SCHEMA = "e01_e02_bank_milestone_ruling_v1"
BLOCKED_STATUS = "E01_E02_BLOCKED_BY_MILESTONE_RULING"
M8_SMOKE_GATE_STATUS = "E01_E02_BLOCKED_BY_M8_SMOKE_GATE"
ENV_LOCK_PATH = "environments/m6_core_gpu.lock.json"
LOCKED_ENVIRONMENT = {"python_executable": "/home/student20261/miniconda3/envs/gpat-m5/bin/python",
                      "python_version": "3.11.16", "numpy": "2.4.6", "pillow": "12.3.0", "opencv": "5.0.0"}

MILESTONE_OPTIONS = ("A_M8_SMOKE_GATED", "B_M6_EXECUTION_EXTENSION")
ORDINAL_RULES = ("MANIFEST_POSITION_ZERO_BASED", "PAIR_ID_NUMERIC")
E02_QUANTIZATION = "CONFIRM_CLIP_THEN_RINT"
# Owner ruling 2026-11-22 (configs/amendments/e01_e02_bank_milestone_ruling.yaml). PRODUCTION accepts only these.
APPROVED_MILESTONE_OPTION = "A_M8_SMOKE_GATED"
APPROVED_E01_ORDINAL = "MANIFEST_POSITION_ZERO_BASED"
METHODS = ("E01", "E02")
SMOKE_CONTRACT_PATH = "configs/amendments/m8_smoke_test_contract.yaml"
SMOKE_KINDS = ("SMOKE_TOY_POPULATION", "SMOKE_SYNTHETIC_FIXTURE")
PAIR_ID_RE = re.compile(r"^PTR\d{6}$")
CANONICAL = 256
PNG_COMPRESS_LEVEL = 6

BANK_DIR = "bank"
EDIT_MAP_DIR = "edit_maps"
BANK_MANIFEST = "bank_manifest.parquet"
BANK_MANIFEST_SHA = "bank_manifest.sha256"


class BankError(RuntimeError):
    """An integrity, population or write-safety violation. The run is refused or aborted."""


class BankBlocked(BankError):
    """Production authorization is absent or incomplete. `status` names the blocking gate."""

    def __init__(self, message: str, status: str = BLOCKED_STATUS):
        super().__init__(message)
        self.status = status


def require(ok, message):
    if not ok:
        raise BankError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_sha256(arr: np.ndarray) -> str:
    """Encoder-independent identity of an image: SHA256 over shape, dtype and C-order bytes."""
    a = np.ascontiguousarray(arr)
    return sha256_bytes(f"{a.dtype.str}|{'x'.join(map(str, a.shape))}|".encode() + a.tobytes())


# ============================================================== authorization
@dataclasses.dataclass(frozen=True)
class Authorization:
    mode: str                         # PRODUCTION | QUALIFICATION
    e01_round_robin_ordinal: str | None
    milestone_option: str | None
    ruling_path: str | None
    ruling_sha256: str | None
    label: str

    def record(self) -> dict:
        return dataclasses.asdict(self)


def load_production_authorization(method_id: str, path: str | Path | None = None) -> Authorization:
    """Owner ruling gate. Refuses with BLOCKED_STATUS unless the approved ruling answers what `method_id` needs."""
    if method_id not in METHODS:
        raise BankError(f"unknown bank method {method_id!r}")
    p = Path(path) if path is not None else ROOT / RULING_PATH

    def blocked(why, status=BLOCKED_STATUS):
        raise BankBlocked(f"{status}: {why}", status)

    if p.name.endswith(".DRAFT.yaml"):
        blocked("a DRAFT ruling is never authority")
    if not p.is_file():
        blocked(f"no approved owner ruling at {RULING_PATH}")
    raw = p.read_bytes()
    try:
        ruling = json.loads(raw)
    except ValueError:
        import yaml
        ruling = yaml.safe_load(raw)
    if not isinstance(ruling, dict) or ruling.get("schema_version") != RULING_SCHEMA:
        blocked("ruling schema_version mismatch")
    if ruling.get("approved") is not True or ruling.get("status") != "OWNER_APPROVED":
        blocked("ruling is not owner-approved")
    decisions = ruling.get("decisions") or {}
    option = decisions.get("Q1_milestone_owner")
    if option not in MILESTONE_OPTIONS:
        blocked("Q1 milestone owner unanswered")
    if option != APPROVED_MILESTONE_OPTION:
        blocked(f"Q1 {option} differs from the owner-approved {APPROVED_MILESTONE_OPTION}")
    ordinal = decisions.get("Q2_e01_round_robin_ordinal")
    if method_id == "E01" and ordinal not in ORDINAL_RULES:
        blocked("Q2 E01 round-robin ordinal unanswered")
    if method_id == "E01" and ordinal != APPROVED_E01_ORDINAL:
        blocked(f"Q2 {ordinal} differs from the owner-approved {APPROVED_E01_ORDINAL} (position % 8, never the pair_id suffix)")
    if method_id == "E02" and decisions.get("Q3_e02_uint8_quantization") != E02_QUANTIZATION:
        blocked("Q3 E02 quantization not confirmed as the implemented clip->rint")
    smoke = ruling.get("smoke_test_record") or {}
    sp = ROOT / str(smoke.get("path", ""))
    if not smoke.get("path") or not sp.is_file() or sha256_file(sp) != smoke.get("sha256"):
        blocked("option A requires a committed, hash-verified M8 deterministic smoke-test record", M8_SMOKE_GATE_STATUS)
    if (json.loads(sp.read_text()) if sp.suffix == ".json" else {}).get("smoke_test_passed") is not True:
        blocked("smoke-test record does not state smoke_test_passed true", M8_SMOKE_GATE_STATUS)
    return Authorization(mode="PRODUCTION", e01_round_robin_ordinal=ordinal if method_id == "E01" else None,
                         milestone_option=option, ruling_path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                         ruling_sha256=sha256_bytes(raw), label="production")


def qualification_authorization(label: str, e01_round_robin_ordinal: str = "MANIFEST_POSITION_ZERO_BASED") -> Authorization:
    """Synthetic-fixture authorization. Its outputs are never FAS-Aug/FreqSub benchmark evidence."""
    if e01_round_robin_ordinal not in ORDINAL_RULES:
        raise BankError("unknown ordinal rule")
    if not label or not re.fullmatch(r"[a-z0-9_-]+", label):
        raise BankError("qualification label must be [a-z0-9_-]+")
    return Authorization(mode="QUALIFICATION", e01_round_robin_ordinal=e01_round_robin_ordinal,
                         milestone_option=None, ruling_path=None, ruling_sha256=None, label=label)


def smoke_authorization(path: str | Path | None = None) -> Authorization:
    """M8 smoke authorization from the owner-approved smoke contract (S6 frozen pairs, S7 implemented bank writers)."""
    p = Path(path) if path is not None else ROOT / SMOKE_CONTRACT_PATH
    if p.name.endswith((".DRAFT.yaml", ".FINAL_DRAFT.yaml")) or not p.is_file():
        raise BankBlocked(f"M8_SMOKE_CONTRACT_NOT_APPROVED: no approved contract at {p}", "M8_SMOKE_CONTRACT_NOT_APPROVED")
    raw = p.read_bytes()
    c = json.loads(raw)
    d = c.get("decisions") or {}
    if (c.get("status") != "OWNER_APPROVED" or c.get("approved") is not True
            or d.get("S6_bank_pair_population") != "FROZEN_PAIRS_TARGET_IN_TOY"
            or d.get("S7_methods_covered") != "ALL_METHODS_WITH_BANK_WRITERS"):
        raise BankBlocked("M8_SMOKE_CONTRACT_NOT_APPROVED: contract not approved or not the approved decisions",
                          "M8_SMOKE_CONTRACT_NOT_APPROVED")
    return Authorization(mode="SMOKE", e01_round_robin_ordinal=APPROVED_E01_ORDINAL, milestone_option=None,
                         ruling_path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                         ruling_sha256=sha256_bytes(raw), label="m8_smoke")


def read_bytes_noatime(path: Path) -> bytes:
    """Read a file without updating its atime (O_NOATIME), so smoke reads never disturb atime-based firewalls."""
    fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                return b"".join(chunks)
            chunks.append(b)
    finally:
        os.close(fd)


# ============================================================== inputs
@dataclasses.dataclass(frozen=True)
class PairRow:
    position: int
    pair_id: str
    target_live_id: str
    source_spoof_id: str
    dataset: str


class BankInputs:
    """A verified TRAIN pair population plus a hash-checked face reader."""

    def __init__(self, *, pairs: list[PairRow], faces_root: str | Path, face_sha256: dict[str, str],
                 sample_split: dict[str, str], sample_dataset: dict[str, str], provenance: dict):
        self.faces_root = Path(faces_root)
        self.smoke = provenance.get("kind") in SMOKE_KINDS
        require(self.faces_root.is_dir() and not self.faces_root.is_symlink(), f"faces root {self.faces_root}")
        self.pairs = list(pairs)
        require(self.pairs, "empty pair population")
        ids = [r.pair_id for r in self.pairs]
        require(len(set(ids)) == len(ids), "duplicate pair_id")
        require(ids == sorted(ids), "pairs are not in ascending pair_id order")
        positions = [r.position for r in self.pairs]
        if self.smoke:      # canonical global positions of a frozen sub-population (S6)
            require(all(isinstance(x, int) and x >= 0 for x in positions)
                    and all(a < b for a, b in zip(positions, positions[1:])), "smoke positions not strictly increasing")
        else:
            require(positions == list(range(len(self.pairs))), "positions are not 0..N-1")
        for r in self.pairs:
            require(PAIR_ID_RE.match(r.pair_id) is not None, f"non-TRAIN pair id {r.pair_id!r}")
            for sid in (r.target_live_id, r.source_spoof_id):
                require(sample_split.get(sid) == "TRAIN", f"sample {sid} is not TRAIN")
                require(sample_dataset.get(sid) == r.dataset, f"sample {sid} dataset mismatch")
                require(isinstance(face_sha256.get(sid), str) and len(face_sha256[sid]) == 64, f"face sha256 {sid}")
            require(r.target_live_id != r.source_spoof_id, f"{r.pair_id}: target == source")
        self.face_sha256 = dict(face_sha256)
        self.sample_dataset = dict(sample_dataset)
        self.provenance = dict(provenance)
        self.opened: list[str] = []

    def face_path(self, sample_id: str) -> Path:
        p = self.faces_root / self.sample_dataset[sample_id] / f"{sample_id}.png"
        require(not p.is_symlink(), f"face is a symlink: {p}")
        return p

    def read(self, sample_id: str) -> tuple[np.ndarray, str]:
        """Bytes -> SHA256 == M2 record -> PIL decode; require mode RGB 256x256 uint8."""
        from PIL import Image
        require(sample_id in self.face_sha256, f"sample {sample_id} is outside the verified population")
        p = self.face_path(sample_id)
        try:
            raw = read_bytes_noatime(p) if self.smoke else p.read_bytes()
        except OSError as exc:
            raise BankError(f"face unreadable {sample_id}: {exc}") from exc
        self.opened.append(sample_id)
        digest = sha256_bytes(raw)
        require(digest == self.face_sha256[sample_id], f"face sha256 mismatch {sample_id}")
        with Image.open(io.BytesIO(raw)) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == (CANONICAL, CANONICAL),
                    f"face {sample_id} is not a 256x256 RGB PNG ({im.format}, {im.mode}, {im.size})")
            arr = np.array(im, dtype=np.uint8)
        require(arr.shape == (CANONICAL, CANONICAL, 3), f"decoded shape {arr.shape}")
        return arr, digest


def _read_parquet_checked(path: Path, expected_sha256: str, columns: list[str]):
    import pyarrow.parquet as pq
    require(sha256_file(path) == expected_sha256, f"{path.name} SHA256 differs from the frozen value")
    return pq.read_table(path, columns=columns).to_pydict()


def production_inputs(runtime_root: str | Path, config: dict) -> BankInputs:
    """The frozen TRAIN population of `config` with faces at <runtime_root>/data/processed/faces_256."""
    data = config["data"]
    require(data["pair_manifest"] == PAIRS_PATH and data["split_manifest"] == SPLIT_PATH, "frozen manifest paths")
    p = _read_parquet_checked(ROOT / PAIRS_PATH, data["pair_manifest_sha256"],
                              ["pair_id", "target_live_id", "source_spoof_id", "dataset", "split", "split_manifest_sha256"])
    n = len(p["pair_id"])
    require(n == int(config["generation"]["n_syn_intended"]), f"pair rows {n} != n_syn_intended")
    require(set(p["split"]) == {"TRAIN"}, "pair manifest contains a non-TRAIN row")
    require(set(p["split_manifest_sha256"]) == {data["split_manifest_sha256"]}, "pair manifest split binding")
    pairs = [PairRow(i, p["pair_id"][i], p["target_live_id"][i], p["source_spoof_id"][i], p["dataset"][i])
             for i in range(n)]
    referenced = {r.target_live_id for r in pairs} | {r.source_spoof_id for r in pairs}
    s = _read_parquet_checked(ROOT / SPLIT_PATH, data["split_manifest_sha256"], ["sample_id", "split", "dataset"])
    split = {sid: sp for sid, sp in zip(s["sample_id"], s["split"]) if sid in referenced}
    dataset = {sid: ds for sid, ds in zip(s["sample_id"], s["dataset"]) if sid in referenced}
    a = _read_parquet_checked(ROOT / M2_ACCOUNTING_PATH, M2_ACCOUNTING_SHA256,
                              ["sample_id", "face_png_sha256", "final_status"])
    face = {}
    for sid, digest, status in zip(a["sample_id"], a["face_png_sha256"], a["final_status"]):
        if sid in referenced:
            require(status == "COMPLETE", f"M2 face not COMPLETE: {sid}")
            face[sid] = digest
    require(set(face) == referenced and set(split) == referenced, "referenced samples missing from M2/split")
    runtime_root = Path(runtime_root)
    faces_root = runtime_root / "data" / "processed" / "faces_256"
    require(faces_root.resolve() == faces_root, "faces root must be a resolved, non-symlinked path")
    return BankInputs(pairs=pairs, faces_root=faces_root, face_sha256=face, sample_split=split,
                      sample_dataset=dataset,
                      provenance={"kind": "PRODUCTION_FROZEN_TRAIN_POPULATION",
                                  "pair_manifest": PAIRS_PATH, "pair_manifest_sha256": data["pair_manifest_sha256"],
                                  "split_manifest": SPLIT_PATH, "split_manifest_sha256": data["split_manifest_sha256"],
                                  "m2_accounting": M2_ACCOUNTING_PATH, "m2_accounting_sha256": M2_ACCOUNTING_SHA256,
                                  "faces_root": str(faces_root), "pairs": n, "referenced_samples": len(referenced)})


def smoke_toy_inputs(runtime_root: str | Path, config: dict, positions: list[int]) -> BankInputs:
    """The frozen TRAIN pairs at the toy positions (S6), faces read with O_NOATIME. Every manifest is hash-checked before
    parsing and read with a parquet row filter restricted to the referenced TRAIN samples, so no TEST row reaches Python."""
    import pyarrow.parquet as pq
    data = config["data"]
    require(data["pair_manifest"] == PAIRS_PATH and data["split_manifest"] == SPLIT_PATH, "frozen manifest paths")
    require(len(set(positions)) == len(positions) and positions == sorted(positions) and positions, "toy positions")
    require(sha256_file(ROOT / PAIRS_PATH) == data["pair_manifest_sha256"], "pairs_train_v1 SHA256")
    p = pq.read_table(ROOT / PAIRS_PATH, columns=["pair_id", "target_live_id", "source_spoof_id", "dataset", "split",
                                                  "split_manifest_sha256"]).to_pydict()
    n = len(p["pair_id"])
    require(n == int(config["generation"]["n_syn_intended"]) and set(p["split"]) == {"TRAIN"}, "frozen pair relation")
    require(set(p["split_manifest_sha256"]) == {data["split_manifest_sha256"]}, "pair manifest split binding")
    require(p["pair_id"] == sorted(p["pair_id"]) and positions[-1] < n, "toy positions outside the frozen relation")
    pairs = [PairRow(i, p["pair_id"][i], p["target_live_id"][i], p["source_spoof_id"][i], p["dataset"][i]) for i in positions]
    used = sorted({r.target_live_id for r in pairs} | {r.source_spoof_id for r in pairs})
    require(sha256_file(ROOT / SPLIT_PATH) == data["split_manifest_sha256"], "split_v1 SHA256")
    sp = pq.read_table(ROOT / SPLIT_PATH, columns=["sample_id", "split", "dataset"],
                       filters=[("sample_id", "in", used)]).to_pydict()
    require(sha256_file(ROOT / M2_ACCOUNTING_PATH) == M2_ACCOUNTING_SHA256, "M2 accounting SHA256")
    m2 = pq.read_table(ROOT / M2_ACCOUNTING_PATH, columns=["sample_id", "face_png_sha256", "final_status"],
                       filters=[("sample_id", "in", used)]).to_pydict()
    split = dict(zip(sp["sample_id"], sp["split"]))
    dataset = dict(zip(sp["sample_id"], sp["dataset"]))
    face = {}
    for sid, digest, status in zip(m2["sample_id"], m2["face_png_sha256"], m2["final_status"]):
        require(status == "COMPLETE", f"M2 face not COMPLETE: {sid}")
        face[sid] = digest
    require(set(split) == set(used) == set(face) and set(split.values()) == {"TRAIN"}, "toy samples must be TRAIN, M2 COMPLETE")
    faces_root = Path(runtime_root) / "data" / "processed" / "faces_256"
    require(faces_root.resolve() == faces_root, "faces root must be a resolved, non-symlinked path")
    return BankInputs(pairs=pairs, faces_root=faces_root, face_sha256=face, sample_split=split, sample_dataset=dataset,
                      provenance={"kind": "SMOKE_TOY_POPULATION", "pair_manifest": PAIRS_PATH,
                                  "pair_manifest_sha256": data["pair_manifest_sha256"], "split_manifest": SPLIT_PATH,
                                  "split_manifest_sha256": data["split_manifest_sha256"], "m2_accounting": M2_ACCOUNTING_PATH,
                                  "m2_accounting_sha256": M2_ACCOUNTING_SHA256, "faces_root": str(faces_root),
                                  "pairs": len(pairs), "referenced_samples": len(used),
                                  "toy_positions_sha256": sha256_bytes(json.dumps(positions).encode()),
                                  "test_rows_read": 0})


# ============================================================== environment
def environment_versions() -> dict:
    import platform
    import sys
    import PIL
    out = {"python_executable": sys.executable, "python_version": platform.python_version(),
           "numpy": np.__version__, "pillow": PIL.__version__}
    try:
        import cv2
        out["opencv"] = cv2.__version__
    except Exception as exc:  # pragma: no cover - E01 official backend requires OpenCV
        out["opencv"] = f"UNAVAILABLE: {type(exc).__name__}"
    return out


def verify_locked_environment() -> dict:
    """PRODUCTION only: the live interpreter must be the M6D1-qualified gpat-m5 (A2-03 NumPy pin)."""
    live = environment_versions()
    require(sha256_file(ROOT / ENV_LOCK_PATH) == "53d4910a17c541ad7ac57a1ac410c0fef8c2d1c63399b423c97e490817b335eb",
            "E01/E02 environment lock changed")
    lock = json.loads((ROOT / ENV_LOCK_PATH).read_text())
    require(lock["numpy_execution_version"] == LOCKED_ENVIRONMENT["numpy"], "lock NumPy pin")
    diff = {k: (live.get(k), v) for k, v in LOCKED_ENVIRONMENT.items()
            if (Path(live[k]).resolve() != Path(v).resolve() if k == "python_executable" else live.get(k) != v)}
    require(not diff, f"live environment differs from the M6D1 lock: {diff}")
    return live


# ============================================================== output encoding / exclusive writes
def encode_png(arr: np.ndarray, mode: str) -> bytes:
    """Deterministic lossless PNG: fixed compress level, no metadata chunks."""
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(arr))
    require(im.mode == mode, f"PNG mode {im.mode} != {mode}")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False)
    return buf.getvalue()


def decode_png(raw: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return np.array(im)


def write_exclusive(path: Path, data: bytes) -> None:
    """O_CREAT|O_EXCL: an existing file is a collision, never overwritten."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


# ============================================================== generators
def make_generator(method_id: str, config: dict):
    if method_id == "E01":
        from methods.fas_aug.e01 import E01Generator
        return E01Generator(config).prepare()
    if method_id == "E02":
        from methods.freq_sub.e02 import E02Generator
        return E02Generator(config).prepare()
    raise BankError(f"unknown bank method {method_id!r}")


def e02_edit_mask(selected_blocks_rc, n: int = CANONICAL, block: int = 16) -> np.ndarray:
    """The applied edit map (selected blocks plus pixel-wise conjugates) from the recorded block selection."""
    from methods.freq_sub.e02 import _edit_mask
    _, edit = _edit_mask(np.asarray(selected_blocks_rc, dtype=np.int64).reshape(-1, 2), n, block)
    return edit


def generate_pair(method_id: str, generator, seed: int, row: PairRow, inputs: BankInputs, auth: Authorization):
    """Read the inputs the method contract needs and call the unchanged generate_one."""
    target, target_sha = inputs.read(row.target_live_id)
    lineage = {"target_live_id": row.target_live_id, "target_face_sha256": target_sha,
               "source_spoof_id": row.source_spoof_id, "dataset": row.dataset, "position": row.position}
    if method_id == "E01":
        index = row.position if auth.e01_round_robin_ordinal == "MANIFEST_POSITION_ZERO_BASED" else None
        lineage.update(source_face_sha256=None, source_spoof_used=False, round_robin_ordinal=auth.e01_round_robin_ordinal,
                       round_robin_index=index if index is not None else int(row.pair_id[3:]))
        res = generator.generate_one(row.pair_id, seed, target, index=index, source_spoof_ref=row.source_spoof_id,
                                     split="TRAIN")
    else:
        source, source_sha = inputs.read(row.source_spoof_id)
        lineage.update(source_face_sha256=source_sha, source_spoof_used=True)
        res = generator.generate_one(row.pair_id, seed, target, source, source_spoof_ref=row.source_spoof_id,
                                     target_live_ref=row.target_live_id, split="TRAIN")
    return res, lineage


# ============================================================== writer
MANIFEST_COLUMNS = (
    "position", "pair_id", "method_id", "experiment_seed", "config_sha256", "dataset",
    "target_live_id", "target_face_sha256", "source_spoof_id", "source_face_sha256", "source_spoof_used",
    "deterministic_pair_seed", "recipe_or_operator_metadata", "generator_checkpoint_sha256",
    "generation_status", "failure_reason", "synthetic_output_path", "output_sha256", "output_pixel_sha256",
    "edit_map_path", "edit_map_sha256", "selected_operator_name", "operator_magnitude_level",
    "selected_asset_filename", "is_level_zero_noop", "n_eligible", "k_selected",
    "selected_block_indices_sorted_ascending", "max_abs_imag")


def _manifest_row(method_id, seed, config_sha, row, lineage, res, out_rel, out_sha, out_px, em_rel, em_sha):
    md = res.metadata if res.success else {}
    return {
        "position": row.position, "pair_id": row.pair_id, "method_id": method_id, "experiment_seed": seed,
        "config_sha256": config_sha, "dataset": row.dataset,
        "target_live_id": row.target_live_id, "target_face_sha256": lineage["target_face_sha256"],
        "source_spoof_id": row.source_spoof_id, "source_face_sha256": lineage.get("source_face_sha256"),
        "source_spoof_used": lineage["source_spoof_used"],
        "deterministic_pair_seed": str(md.get("operator_seed", md.get("pair_seed"))) if md else None,
        "recipe_or_operator_metadata": json.dumps(md, sort_keys=True, allow_nan=False),
        "generator_checkpoint_sha256": None,
        "generation_status": "success" if res.success else "failure", "failure_reason": res.failure_reason,
        "synthetic_output_path": out_rel, "output_sha256": out_sha, "output_pixel_sha256": out_px,
        "edit_map_path": em_rel, "edit_map_sha256": em_sha,
        "selected_operator_name": md.get("operator"), "operator_magnitude_level": md.get("level"),
        "selected_asset_filename": md.get("asset"), "is_level_zero_noop": md.get("is_level_zero_noop"),
        "n_eligible": md.get("n_eligible"), "k_selected": md.get("k"),
        "selected_block_indices_sorted_ascending": json.dumps(md["selected_indices"]) if "selected_indices" in md else None,
        "max_abs_imag": md.get("max_abs_imag"),
    }


def _write_manifest(path: Path, rows: list[dict]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        (c, pa.int64() if c in ("position", "experiment_seed", "n_eligible", "k_selected") else
         pa.bool_() if c in ("source_spoof_used", "is_level_zero_noop") else
         pa.float64() if c in ("operator_magnitude_level", "max_abs_imag") else pa.string())
        for c in MANIFEST_COLUMNS]))
    tmp = path.with_name(path.name + ".tmp")
    require(not tmp.exists() and not path.exists(), f"bank manifest collision at {path}")
    pq.write_table(table, tmp, compression="zstd")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.link(tmp, path)                      # fails if `path` appeared meanwhile; never overwrites
    os.unlink(tmp)
    digest = sha256_file(path)
    write_exclusive(path.with_name(BANK_MANIFEST_SHA), f"{digest}  {BANK_MANIFEST}\n".encode())
    return digest


def write_bank(method_id: str, seed: int, inputs: BankInputs, auth: Authorization, runtime_root: str | Path, *,
               config: dict | None = None, progress: Callable[[int, int], None] | None = None,
               environment_lock: dict | None = None) -> dict:
    """Generate the bank for one (method, seed). Returns the run summary. Never overwrites anything."""
    require(method_id in METHODS, f"unknown bank method {method_id!r}")
    config = config if config is not None else load_method_config(method_id)
    runtime_root = Path(runtime_root)
    if auth.mode == "PRODUCTION":
        require(environment_lock is not None and environment_lock.get("verified") is True,
                "production requires a verified environment lock")
        require(inputs.provenance.get("kind") == "PRODUCTION_FROZEN_TRAIN_POPULATION", "production needs frozen inputs")
        require(len(inputs.pairs) == int(config["generation"]["n_syn_intended"]), "production population count")
        require(inputs.faces_root == runtime_root / "data" / "processed" / "faces_256", "faces root under runtime root")
    elif auth.mode == "SMOKE":
        kind = inputs.provenance.get("kind")
        require(kind in SMOKE_KINDS, "smoke needs the toy population or a synthetic smoke fixture")
        if kind == "SMOKE_TOY_POPULATION":
            require(environment_lock is not None and environment_lock.get("verified") is True,
                    "smoke toy banks require the verified gpat-m5 environment lock")
        rr = runtime_root.resolve()
        fr = inputs.faces_root.resolve()
        require(not (rr / "data" / "processed" / "faces_256").exists(), "smoke root may not hold benchmark faces")
        require(not fr.is_relative_to(rr) and not rr.is_relative_to(fr), "smoke root and faces root must be disjoint")
        if kind == "SMOKE_TOY_POPULATION":     # faces root = <production runtime root>/data/processed/faces_256
            require(fr.parts[-3:] == ("data", "processed", "faces_256") and not rr.is_relative_to(fr.parents[2]),
                    "smoke root must lie outside the production runtime root")
    else:
        require(auth.mode == "QUALIFICATION", "unknown authorization mode")
        require(inputs.provenance.get("kind") == "SYNTHETIC_QUALIFICATION_FIXTURE", "qualification needs a synthetic fixture")
        require(not (runtime_root / "data" / "processed" / "faces_256").exists(),
                "qualification may not run in a runtime root holding benchmark faces")
    generator = make_generator(method_id, config)
    ctx = RunContext(method_id=method_id, seed=seed, config=config, runtime_root=runtime_root, resume=False)
    ctx.open()
    run_dir = ctx.run_dir
    t0 = time.monotonic()
    rows: list[dict] = []
    try:
        bank_dir = run_dir / BANK_DIR
        bank_dir.mkdir(exist_ok=False)
        edit_dir = None
        if method_id == "E02":
            edit_dir = run_dir / EDIT_MAP_DIR
            edit_dir.mkdir(exist_ok=False)
        ctx.log_event("bank_start", {"authorization": auth.record(), "inputs": inputs.provenance,
                                     "population": len(inputs.pairs), "environment_versions": environment_versions(),
                                     "png_compress_level": PNG_COMPRESS_LEVEL, "output_naming": "bank/<pair_id>.png"})
        for i, row in enumerate(inputs.pairs):
            res, lineage = generate_pair(method_id, generator, seed, row, inputs, auth)
            out_rel = out_sha = out_px = em_rel = em_sha = None
            if res.success:
                out = res.output
                require(isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.shape == (CANONICAL, CANONICAL, 3),
                        f"{row.pair_id}: generator output is not canonical 256x256x3 uint8")
                raw = encode_png(out, "RGB")
                require(np.array_equal(decode_png(raw), out), f"{row.pair_id}: PNG round trip changed pixels")
                out_rel = f"{BANK_DIR}/{row.pair_id}.png"
                write_exclusive(run_dir / out_rel, raw)
                out_sha, out_px = sha256_bytes(raw), pixel_sha256(out)
                if method_id == "E02":
                    mask = e02_edit_mask(res.metadata["selected_blocks_rc"])
                    require(int(mask.sum()) == res.metadata["edited_pixels"], f"{row.pair_id}: edit map disagrees")
                    em_raw = encode_png(mask.astype(np.uint8) * 255, "L")
                    em_rel = f"{EDIT_MAP_DIR}/{row.pair_id}.png"
                    write_exclusive(run_dir / em_rel, em_raw)
                    em_sha = sha256_bytes(em_raw)
            ctx.log_generation({**res.to_record(include_output_ref=out_rel), "output_path": out_rel,
                                "output_sha256": out_sha, "output_pixel_sha256": out_px, "lineage": lineage,
                                "edit_map_path": em_rel, "edit_map_sha256": em_sha})
            rows.append(_manifest_row(method_id, seed, ctx.config_sha256, row, lineage, res,
                                      out_rel, out_sha, out_px, em_rel, em_sha))
            if progress is not None:
                progress(i + 1, len(inputs.pairs))
        manifest_sha = _write_manifest(run_dir / BANK_MANIFEST, rows)
        integrity = verify_bank_files(run_dir, method_id, global_positions=inputs.smoke)
        require(integrity["ok"], f"post-write integrity failed: {integrity}")
        summary = bank_summary(method_id, rows, generator.finalize())
        summary.update(output_manifest_path=str(run_dir / BANK_MANIFEST), bank_manifest_sha256=manifest_sha,
                       authorization=auth.record(), inputs=inputs.provenance, environment_versions=environment_versions(),
                       generation_wall_time_seconds=time.monotonic() - t0, post_write_integrity=integrity,
                       peak_vram_reason="CPU-only non-learned bank; no GPU used",
                       environment_lock=environment_lock or {"verified": False,
                                                             "reason": "QUALIFICATION run outside the locked gpat-m5"},
                       missing_field_reasons={
                           "final_or_selected_checkpoint_path": "non-learned method: no checkpoint exists",
                           "final_or_selected_checkpoint_sha256": "non-learned method: no checkpoint exists",
                           "seed_level_evaluation_metrics": "bank generation only; evaluation is a later milestone",
                           "training_duration_seconds": "non-learned method: no training",
                           "peak_vram_bytes": "CPU-only non-learned bank; no GPU used",
                           "failure_reason": "run did not fail"})
        status = "completed" if summary["failure_count"] == 0 else "completed_with_failures"
        return ctx.close(completion_status=status, summary=summary)
    except BaseException as exc:
        ctx.close(completion_status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
                  summary={"rows_before_abort": len(rows), "authorization": auth.record()})
        raise


def bank_summary(method_id: str, rows: list[dict], finalize: dict) -> dict:
    ok = [r for r in rows if r["generation_status"] == "success"]
    by_px: dict[str, int] = {}
    for r in ok:
        by_px[r["output_pixel_sha256"]] = by_px.get(r["output_pixel_sha256"], 0) + 1
    out = {"input_count": len(rows), "output_count": len(ok), "failure_count": len(rows) - len(ok),
           "failures": [{"pair_id": r["pair_id"], "failure_reason": r["failure_reason"]} for r in rows
                        if r["generation_status"] != "success"],
           "duplicate_output_pixel_groups": sum(1 for v in by_px.values() if v > 1),
           "duplicate_output_pixel_rows": sum(v for v in by_px.values() if v > 1),
           "generator_finalize": finalize}
    if method_id == "E01":
        ops: dict[str, int] = {}
        for r in ok:
            ops[r["selected_operator_name"]] = ops.get(r["selected_operator_name"], 0) + 1
        out.update(operator_counts=dict(sorted(ops.items())),
                   level_zero_noop_count=sum(1 for r in ok if r["is_level_zero_noop"]),
                   level_zero_noop_fraction=(sum(1 for r in ok if r["is_level_zero_noop"]) / len(ok)) if ok else None)
    else:
        out.update(max_abs_imag=max((r["max_abs_imag"] for r in ok), default=None),
                   k_values=sorted({r["k_selected"] for r in ok}), n_eligible_values=sorted({r["n_eligible"] for r in ok}))
    return out


# ============================================================== verification
def read_manifest(run_dir: str | Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(Path(run_dir) / BANK_MANIFEST).to_pylist()


def verify_bank_files(run_dir: str | Path, method_id: str, *, global_positions: bool = False) -> dict:
    """Manifest <-> files: every success row has exactly its file with its hashes; no extra or missing file.
    `global_positions` (smoke banks only) accepts strictly increasing canonical positions instead of 0..N-1."""
    run_dir = Path(run_dir)
    rows = read_manifest(run_dir)
    problems = []
    recorded = (run_dir / BANK_MANIFEST_SHA).read_text().split()[0]
    if recorded != sha256_file(run_dir / BANK_MANIFEST):
        problems.append("bank manifest sha256 file mismatch")
    ids = [r["pair_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate pair_id rows")
    pos = [r["position"] for r in rows]
    if global_positions:
        if not all(a < b for a, b in zip(pos, pos[1:])) or (pos and pos[0] < 0):
            problems.append("positions not strictly increasing")
    elif pos != list(range(len(rows))):
        problems.append("positions not 0..N-1")
    expected = {r["synthetic_output_path"] for r in rows if r["generation_status"] == "success"}
    actual = {f"{BANK_DIR}/{p.name}" for p in (run_dir / BANK_DIR).iterdir()}
    if expected != actual:
        problems.append(f"bank file set differs: missing {sorted(expected - actual)[:5]}, extra {sorted(actual - expected)[:5]}")
    if method_id == "E02":
        em_expected = {r["edit_map_path"] for r in rows if r["generation_status"] == "success"}
        em_actual = {f"{EDIT_MAP_DIR}/{p.name}" for p in (run_dir / EDIT_MAP_DIR).iterdir()}
        if em_expected != em_actual:
            problems.append("edit map file set differs")
    for r in rows:
        if r["generation_status"] != "success":
            if r["synthetic_output_path"] is not None:
                problems.append(f"{r['pair_id']}: failure row has an output path")
            continue
        raw = (run_dir / r["synthetic_output_path"]).read_bytes()
        if sha256_bytes(raw) != r["output_sha256"]:
            problems.append(f"{r['pair_id']}: output sha256")
        arr = decode_png(raw)
        if arr.dtype != np.uint8 or arr.shape != (CANONICAL, CANONICAL, 3) or pixel_sha256(arr) != r["output_pixel_sha256"]:
            problems.append(f"{r['pair_id']}: output pixels")
        if method_id == "E02" and sha256_bytes((run_dir / r["edit_map_path"]).read_bytes()) != r["edit_map_sha256"]:
            problems.append(f"{r['pair_id']}: edit map sha256")
    return {"ok": not problems, "rows": len(rows), "files_checked": len(expected), "problems": problems[:50]}


def replay_bank(run_dir: str | Path, method_id: str, seed: int, inputs: BankInputs, auth: Authorization, *,
                config: dict | None = None) -> dict:
    """Deterministic replay: regenerate every row in memory and compare pixel identity and recipe metadata."""
    config = config if config is not None else load_method_config(method_id)
    generator = make_generator(method_id, config)
    rows = read_manifest(run_dir)
    require(len(rows) == len(inputs.pairs), "replay population differs from the manifest")
    mismatches = []
    for r, row in zip(rows, inputs.pairs):
        require(r["pair_id"] == row.pair_id and r["position"] == row.position, "replay order differs")
        res, _ = generate_pair(method_id, generator, seed, row, inputs, auth)
        status = "success" if res.success else "failure"
        if status != r["generation_status"]:
            mismatches.append((row.pair_id, "status"))
        elif res.success:
            if pixel_sha256(res.output) != r["output_pixel_sha256"]:
                mismatches.append((row.pair_id, "pixels"))
            if json.dumps(res.metadata, sort_keys=True, allow_nan=False) != r["recipe_or_operator_metadata"]:
                mismatches.append((row.pair_id, "metadata"))
    return {"ok": not mismatches, "rows_replayed": len(rows), "mismatches": mismatches[:50]}
