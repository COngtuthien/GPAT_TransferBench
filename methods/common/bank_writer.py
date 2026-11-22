"""E01/E02 synthetic-bank writer (non-learned direct transformation banks).

Scope. This module turns the frozen E01/E02 generators (`methods/fas_aug/e01.py`, `methods/freq_sub/e02.py`) into a
per-seed bank at the frozen M6B run layout `<runtime_root>/runs/m6/<method>/seed_<seed>/`. It changes no method
semantics: every pixel comes from the unchanged `generate_one`. It adds only input selection, integrity checks,
deterministic output naming, fail-closed writes, per-sample accounting and a bank manifest.

Authorization. A PRODUCTION bank needs the owner-approved `configs/amendments/e01_e02_bank_milestone_ruling.yaml`
(approved 2026-11-22: Q1 A_M8_SMOKE_GATED, Q2 MANIFEST_POSITION_ZERO_BASED, Q3 CONFIRM_CLIP_THEN_RINT). While that
file is absent, unapproved, incomplete or deviates from the approved decisions, `load_production_authorization`
raises `BankBlocked` with status `E01_E02_BLOCKED_BY_MILESTONE_RULING`. With the approved ruling, it raises
`BankBlocked` with status `E01_E02_BLOCKED_BY_M8_SMOKE_GATE` until the additive owner binding
`configs/amendments/e01_e02_m8_smoke_binding.yaml` names the exact approved M8 smoke record and that record passes the
strict validator and every pinned identity (M6D6jR O3; the ruling itself stays byte-unchanged and an inline record in it
is refused). `verify_production_launch` then requires the owner-approved production authority record (candidate commit
ancestry, byte-identical authority files, clean worktree), the approved host, the locked gpat-m5 interpreter with CUDA
hidden, the approved runtime root on the approved filesystem UUID with the free-space floor, and an absent run
directory (`E01_E02_BLOCKED_BY_PRODUCTION_AUTHORITY` / `..._PRODUCTION_PREFLIGHT`). QUALIFICATION authorizations exist
only for synthetic fixtures and never accept the production faces root. SMOKE authorizations (M8 global smoke) need
the owner-approved configs/amendments/m8_smoke_test_contract.yaml; they accept only the toy population or a synthetic
smoke fixture, keep canonical global pair positions, read faces with O_NOATIME and write only under a separate smoke
root. The production CLI (tools/e01_e02_bank.py) never constructs a SMOKE authorization.

Inputs. Pairs come from the frozen `manifests/pairs_train_v1.parquet` (SHA256 checked before parse, every row TRAIN,
pair_ids ascending and unique, N rows = n_syn_intended). Every referenced sample must be TRAIN in `split_v1` and
COMPLETE in `manifests/m2_sample_accounting.parquet`. Each face is read from
`<runtime_root>/data/processed/faces_256/<dataset>/<sample_id>.png` with O_NOATIME in every mode (the atime is
stat-checked unchanged across the read, which is also the read that is hashed), and its bytes must match the M2
face_png_sha256 before decoding. The decoded image must be PIL mode RGB, 256x256; nothing is converted silently.

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
import subprocess
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

# M8 production binding (owner approval 2026-11-22). The ruling above stays byte-unchanged; the additive binding file
# names the one passing smoke record. Every identity is pinned here too, so an edited binding cannot redirect the gate.
SMOKE_BINDING_PATH = "configs/amendments/e01_e02_m8_smoke_binding.yaml"
SMOKE_BINDING_SCHEMA = "e01_e02_m8_smoke_binding_v1"
APPROVED_RULING_SHA256 = "4a89b4e1817a970b21ba9b005fdf0edf8d19e7fc236f542ddcd1c9cd93b6a19e"
APPROVED_SMOKE_RECORD_PATH = "outputs/audit/M8_SMOKE_TEST_RECORD.json"
APPROVED_SMOKE_RECORD_SHA256 = "6eaa865aaff168ba1aeb8ec8fdd915eba45142a4bb4f3608fb008b5ee71d1db2"
APPROVED_SMOKE_CONTRACT_SHA256 = "13704edde0fd56966144063b9f1f66401264eb0e49d6220297591f04bc914ebb"
EVALUATOR_CONTRACT_PATH = "configs/amendments/m8_downstream_evaluator_contract.yaml"
APPROVED_EVALUATOR_CONTRACT_SHA256 = "67e4b62ddc6b81b8b9a8657e71b3979f6e6a186827183b052ae3d29b587fcbe0"
SMOKE_SOURCE_COMMIT = "dd893016ddb0e6e828d867a69c2d7c18ba363834"
SMOKE_TOY_PAIRS = 186
SMOKE_REPLAY_ARTIFACTS = 1721
SMOKE_STAGES = ("generator", "bank", "evaluator", "metrics", "plots")
SMOKE_RUNS = ("A", "B")
SMOKE_BANK_GATES = {"E01": ("completed_no_failures", "file_verification", "full_replay", "rows_equal_toy_pairs",
                            "operator_is_position_mod_8"),
                    "E02": ("completed_no_failures", "file_verification", "full_replay", "rows_equal_toy_pairs",
                            "k40_n159_residual")}
PRODUCTION_SEEDS = (42, 1337, 2026)
APPROVED_METHOD_CONFIGS = {
    "E01": ("configs/methods/e01_fas_aug.yaml", "87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0"),
    "E02": ("configs/methods/e02_freqsub.yaml", "fdf74776cc100d5b5c1ed5f58b84ebb0dfd534782fbb7fc34382919d38d67bf0")}
PRODUCTION_AUTHORITY_PATH = "configs/amendments/e01_e02_production_authority.yaml"
PRODUCTION_AUTHORITY_SCHEMA = "e01_e02_production_authority_v1"
PRODUCTION_AUTHORITY_STATUS = "E01_E02_BLOCKED_BY_PRODUCTION_AUTHORITY"
PRODUCTION_PREFLIGHT_STATUS = "E01_E02_BLOCKED_BY_PRODUCTION_PREFLIGHT"
EXECUTION_CONFIG_PATH = "configs/execution/m5_gpu_3090.yaml"
# Files whose bytes decide what a production bank contains or which gate admits it. The production authority record
# must pin each one, and the launch checkout must hold exactly those bytes.
AUTHORITY_FILES = (
    "methods/common/bank_writer.py", "methods/common/config.py", "methods/common/runlog.py",
    "methods/common/interface.py", "methods/fas_aug/e01.py", "methods/fas_aug/official.py", "methods/freq_sub/e02.py",
    "tools/e01_e02_bank.py", "gpatbench/downstream/smoke_evidence.py", "gpatbench/downstream/audit_hook.py",
    "configs/methods/e01_fas_aug.yaml", "configs/methods/e02_freqsub.yaml", "configs/run_logging_v1.yaml",
    "outputs/audit/M6B_CONFIG_FREEZE.json", EXECUTION_CONFIG_PATH, RULING_PATH, SMOKE_BINDING_PATH,
    "configs/amendments/m8_o3_smoke_completion.yaml", SMOKE_CONTRACT_PATH, EVALUATOR_CONTRACT_PATH,
    APPROVED_SMOKE_RECORD_PATH, PAIRS_PATH, SPLIT_PATH, M2_ACCOUNTING_PATH, ENV_LOCK_PATH, "third_party/source_pins.json")
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
    binding: tuple | None = None      # PRODUCTION only: sorted (key, value) identity of the verified M8 smoke binding

    def record(self) -> dict:
        d = dataclasses.asdict(self)
        if d["binding"] is None:          # QUALIFICATION / SMOKE records keep their historical shape
            del d["binding"]
        else:
            d["binding"] = dict(self.binding)
        return d


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def _is_ancestor(commit: str, of: str = "HEAD") -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", commit or "")) and \
        _git("merge-base", "--is-ancestor", commit, of).returncode == 0


def _load_json_or_yaml(raw: bytes):
    try:
        return json.loads(raw)
    except ValueError:
        import yaml
        return yaml.safe_load(raw)


def _verified_smoke_binding(method_id: str, ruling: dict, ruling_sha: str, binding_path, seed) -> dict:
    """The M8 binding gate (gate items 2-13, 17, 18). Raises BankBlocked; returns the verified identities."""
    from gpatbench.downstream.smoke_evidence import SmokeRecordInvalid, validate_smoke_record

    def m8(why):
        raise BankBlocked(f"{M8_SMOKE_GATE_STATUS}: {why}", M8_SMOKE_GATE_STATUS)

    def authority(why):
        raise BankBlocked(f"{PRODUCTION_AUTHORITY_STATUS}: {why}", PRODUCTION_AUTHORITY_STATUS)

    if ruling.get("smoke_test_record") is not None:
        m8("a smoke_test_record inside the ruling is never authority; the owner binding is the additive "
           f"{SMOKE_BINDING_PATH}")
    bp = Path(binding_path) if binding_path is not None else ROOT / SMOKE_BINDING_PATH
    if bp.name.endswith((".DRAFT.yaml", ".FINAL_DRAFT.yaml")) or not bp.is_file() or bp.is_symlink():
        m8(f"no owner-approved M8 smoke binding at {SMOKE_BINDING_PATH}")
    braw = bp.read_bytes()
    try:
        b = json.loads(braw)
    except ValueError:
        m8("binding is not JSON")
    if not isinstance(b, dict) or b.get("schema_version") != SMOKE_BINDING_SCHEMA:
        m8("binding schema_version mismatch")
    if b.get("status") != "OWNER_APPROVED" or b.get("approved") is not True:
        m8("binding is not owner-approved")
    if ruling_sha != APPROVED_RULING_SHA256 or (b.get("ruling") or {}).get("sha256") != ruling_sha \
            or (b.get("ruling") or {}).get("path") != RULING_PATH:
        m8("binding does not bind the approved, byte-unchanged owner ruling")
    rec = b.get("smoke_test_record") or {}
    if rec.get("path") != APPROVED_SMOKE_RECORD_PATH or rec.get("sha256") != APPROVED_SMOKE_RECORD_SHA256:
        m8("binding does not name the approved smoke record path and SHA256 exactly")
    sp = ROOT / APPROVED_SMOKE_RECORD_PATH
    if not sp.is_file() or sp.is_symlink() or sha256_file(sp) != APPROVED_SMOKE_RECORD_SHA256:
        m8("the approved smoke record is missing or its SHA256 differs")
    record = json.loads(sp.read_bytes())
    try:
        v = validate_smoke_record(record, ROOT)
    except SmokeRecordInvalid as exc:
        m8(f"strict smoke validator: {exc}")
    if not (v["admissible"] and v["passed"]):
        m8(f"strict smoke validator did not pass: {v['problems']}")
    if record.get("smoke_test_passed") is not True or record.get("mode") != "real" \
            or record.get("label") != "SMOKE_NON_SCIENTIFIC" or record.get("scientific_results_produced") is not False:
        m8("smoke record is not a passing real SMOKE_NON_SCIENTIFIC run")
    for name, path, approved, record_key, binding_key in (
            ("smoke contract", SMOKE_CONTRACT_PATH, APPROVED_SMOKE_CONTRACT_SHA256, "contract_sha256", "smoke_contract"),
            ("evaluator contract", EVALUATOR_CONTRACT_PATH, APPROVED_EVALUATOR_CONTRACT_SHA256,
             "evaluator_contract_sha256", "evaluator_contract")):
        p = ROOT / path
        doc = json.loads(p.read_bytes()) if p.is_file() else {}
        if doc.get("status") != "OWNER_APPROVED" or doc.get("approved") is not True or sha256_file(p) != approved \
                or record.get(record_key) != approved or b.get(binding_key) != {"path": path, "sha256": approved}:
            m8(f"{name} is not the approved {approved[:12]}... in the repository, the record and the binding")
    stages = record.get("stages") or {}
    if set(stages) != set(SMOKE_STAGES) or not all(
            stages[s].get("executed") is True and stages[s].get("simulated") is False and stages[s].get("status") == "PASS"
            for s in SMOKE_STAGES):
        m8("smoke record does not hold exactly the five executed, non-simulated, passing O3 stages")
    banks = (record.get("coverage") or {}).get("banks") or {}
    expected = {f"{m}.seed_{s}" for m in METHODS for s in PRODUCTION_SEEDS}
    if set(banks) != expected or not all(x.get("rows") == x.get("outputs") == SMOKE_TOY_PAIRS and x.get("failures") == 0
                                         for x in banks.values()):
        m8(f"smoke bank coverage is not exactly E01/E02 x {PRODUCTION_SEEDS} with {SMOKE_TOY_PAIRS} outputs, 0 failures")
    gates = {**(stages["generator"].get("gates") or {}), **(stages["bank"].get("gates") or {})}
    for s in PRODUCTION_SEEDS:
        for run in SMOKE_RUNS:
            for g in SMOKE_BANK_GATES[method_id]:
                if gates.get(f"{run}.banks.{method_id}.seed_{s}.{g}") is not True:
                    m8(f"smoke bank-stage gate {run}.banks.{method_id}.seed_{s}.{g} is not true")
    rp = record.get("independent_replay") or {}
    if rp.get("identical") is not True or rp.get("artifacts_compared") != SMOKE_REPLAY_ARTIFACTS or rp.get("differences"):
        m8(f"smoke replay is not {SMOKE_REPLAY_ARTIFACTS} artifacts with zero differences")
    fw = record.get("test_firewall") or {}
    if (fw.get("violations"), fw.get("test_rows_used"), fw.get("listing_attempts"), fw.get("atime_changed_files")) \
            != (0, 0, 0, 0):
        m8("smoke TEST firewall evidence is not clean")
    src = record.get("source") or {}
    if src.get("git_commit") != SMOKE_SOURCE_COMMIT or b.get("smoke_source_commit") != SMOKE_SOURCE_COMMIT \
            or src.get("worktree_dirty") is not False:
        m8("smoke source authority is not the clean, approved smoke implementation commit")
    scope = b.get("scope") or {}
    if scope.get("methods") != list(METHODS) or scope.get("seeds") != list(PRODUCTION_SEEDS) or scope.get("split") != "TRAIN":
        m8("binding scope is not exactly E01/E02, seeds 42/1337/2026, TRAIN")
    auth = b.get("authorizes") or {}
    if auth.get("e01_e02_production_bank_generation") is not True or \
            any(v is not False for k, v in auth.items() if k != "e01_e02_production_bank_generation"):
        m8("binding must authorize E01/E02 production banks and nothing else (no TEST, training, GPAT)")
    cfg_path, cfg_sha = APPROVED_METHOD_CONFIGS[method_id]
    if (b.get("method_configs") or {}).get(method_id) != {"path": cfg_path, "sha256": cfg_sha} \
            or sha256_file(ROOT / cfg_path) != cfg_sha or (ruling.get("frozen_authorities") or {}).get(cfg_path) != cfg_sha:
        m8(f"{method_id} scientific configuration identity differs from the approved {cfg_sha[:12]}...")
    if seed is not None and seed not in PRODUCTION_SEEDS:
        authority(f"seed {seed} is outside the bound production seeds {PRODUCTION_SEEDS}")
    if not _is_ancestor(SMOKE_SOURCE_COMMIT):
        authority("the smoke source commit is not an ancestor of this checkout")
    return {"binding_path": SMOKE_BINDING_PATH, "binding_sha256": sha256_bytes(braw),
            "smoke_record_path": APPROVED_SMOKE_RECORD_PATH, "smoke_record_sha256": APPROVED_SMOKE_RECORD_SHA256,
            "smoke_contract_sha256": APPROVED_SMOKE_CONTRACT_SHA256,
            "evaluator_contract_sha256": APPROVED_EVALUATOR_CONTRACT_SHA256, "smoke_source_commit": SMOKE_SOURCE_COMMIT,
            "method_config_sha256": cfg_sha}


def load_production_authorization(method_id: str, path: str | Path | None = None, *,
                                  binding_path: str | Path | None = None, seed: int | None = None) -> Authorization:
    """Governance gate, fail-closed. Refuses with BLOCKED_STATUS unless the approved ruling answers what `method_id`
    needs; then with M8_SMOKE_GATE_STATUS unless the additive owner binding (SMOKE_BINDING_PATH) names the exact
    approved smoke record, the record passes the strict validator and every pinned identity (contracts, five stages,
    E01/E02 x 42/1337/2026 coverage, per-method bank gates, replay, firewall, clean smoke source, scope, method config)
    holds; then with PRODUCTION_AUTHORITY_STATUS unless the smoke source is an ancestor of this checkout and `seed`
    (when given) is bound. Launch-time checks (production authority commit, environment, storage, collisions) are
    `verify_production_launch`."""
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
    ruling = _load_json_or_yaml(raw)
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
    binding = _verified_smoke_binding(method_id, ruling, sha256_bytes(raw), binding_path, seed)
    return Authorization(mode="PRODUCTION", e01_round_robin_ordinal=ordinal if method_id == "E01" else None,
                         milestone_option=option, ruling_path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                         ruling_sha256=sha256_bytes(raw), label="production", binding=tuple(sorted(binding.items())))


def filesystem_identity(path: str | Path) -> dict:
    """Device, mount point and filesystem UUID of `path` from stat + /proc/self/mountinfo + /dev/disk/by-uuid (all
    read-only; no privilege needed)."""
    st = os.stat(path)
    dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    mount = source = fstype = options = None
    best = -1
    rp = os.path.realpath(path)
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        f = left.split()
        if f[2] != dev:
            continue
        mp = f[4].replace("\\040", " ")
        if (rp == mp or rp.startswith(mp.rstrip("/") + "/")) and len(mp) > best:
            best, mount, options = len(mp), mp, f[5]
            fstype, source = right.split()[0], right.split()[1]
    uuid = None
    by_uuid = Path("/dev/disk/by-uuid")
    if by_uuid.is_dir():
        for link in sorted(by_uuid.iterdir()):
            try:
                if os.stat(link).st_rdev == st.st_dev:
                    uuid = link.name
                    break
            except OSError:
                continue
    vfs = os.statvfs(path)
    return {"path": str(path), "device": dev, "source": source, "fstype": fstype, "mount_point": mount,
            "mount_options": options, "filesystem_uuid": uuid, "free_bytes": vfs.f_bavail * vfs.f_frsize,
            "free_inodes": vfs.f_favail, "total_bytes": vfs.f_blocks * vfs.f_frsize}


def verify_production_launch(method_id: str, seed: int, auth: Authorization, runtime_root: str | Path, *,
                             authority_path: str | Path | None = None) -> dict:
    """Launch gate (gate items 12, 14-16), run after load_production_authorization and before any face is opened.
    Refuses with PRODUCTION_AUTHORITY_STATUS unless the owner-approved production authority record names a candidate
    commit that is an ancestor of this clean checkout whose authority files are byte-identical to that commit; with
    PRODUCTION_PREFLIGHT_STATUS unless this is the approved host and locked gpat-m5 interpreter with CUDA hidden, the
    runtime root is the approved one on the approved filesystem UUID with enough free bytes/inodes, and the run
    directory does not exist."""
    def authority(why):
        raise BankBlocked(f"{PRODUCTION_AUTHORITY_STATUS}: {why}", PRODUCTION_AUTHORITY_STATUS)

    def preflight(why):
        raise BankBlocked(f"{PRODUCTION_PREFLIGHT_STATUS}: {why}", PRODUCTION_PREFLIGHT_STATUS)

    from .runlog import git_commit, git_dirty
    if method_id not in METHODS or seed not in PRODUCTION_SEEDS:
        authority(f"{method_id} seed {seed} is not a bound production bank")
    if auth.mode != "PRODUCTION" or auth.binding is None:
        authority("a verified PRODUCTION authorization (with the M8 binding) is required")
    ap = Path(authority_path) if authority_path is not None else ROOT / PRODUCTION_AUTHORITY_PATH
    if not ap.is_file() or ap.is_symlink():
        authority(f"no production authority record at {PRODUCTION_AUTHORITY_PATH}")
    araw = ap.read_bytes()
    try:
        rec = json.loads(araw)
    except ValueError:
        authority("production authority record is not JSON")
    if rec.get("schema_version") != PRODUCTION_AUTHORITY_SCHEMA or rec.get("status") != "OWNER_APPROVED" \
            or rec.get("approved") is not True:
        authority("production authority record is not owner-approved")
    binding = dict(auth.binding)
    if rec.get("binding_sha256") != binding["binding_sha256"] or rec.get("ruling_sha256") != auth.ruling_sha256 \
            or rec.get("smoke_record_sha256") != APPROVED_SMOKE_RECORD_SHA256:
        authority("production authority record does not name this binding, ruling and smoke record")
    if (rec.get("scope") or {}).get("methods") != list(METHODS) or (rec.get("scope") or {}).get("seeds") != list(PRODUCTION_SEEDS):
        authority("production authority scope is not E01/E02 x 42/1337/2026")
    cand = rec.get("candidate_commit", "")
    if not _is_ancestor(cand):
        authority("the production candidate commit is not an ancestor of this checkout")
    files = rec.get("authority_files") or {}
    missing = sorted(set(AUTHORITY_FILES) - set(files))
    if missing:
        authority(f"authority record does not pin {missing[:3]}")
    for rel, digest in sorted(files.items()):
        blob = _git("show", f"{cand}:{rel}")
        if blob.returncode != 0 or sha256_bytes(blob.stdout) != digest:
            authority(f"{rel} at the candidate commit differs from the authority record")
        p = ROOT / rel
        if not p.is_file() or p.is_symlink() or sha256_file(p) != digest:
            authority(f"{rel} in this checkout differs from the production candidate commit")
    if git_dirty():
        authority("git worktree is dirty")
    ex = rec.get("execution") or {}
    machine_id = Path("/etc/machine-id").read_text().strip() if Path("/etc/machine-id").is_file() else None
    if not machine_id or machine_id != ex.get("machine_id"):
        preflight("this is not the approved execution host")
    try:
        live = verify_locked_environment()
    except BankError as exc:
        preflight(f"execution environment: {exc}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        preflight("CUDA_VISIBLE_DEVICES must be set and empty (CPU-only bank)")
    st = rec.get("storage") or {}
    rr = Path(runtime_root)
    import yaml
    exec_cfg = yaml.safe_load((ROOT / EXECUTION_CONFIG_PATH).read_text())
    approved_root = str((exec_cfg.get("storage") or {}).get("runtime_root") or "")
    if str(rr) != st.get("runtime_root") or str(rr) != approved_root or not rr.is_dir() or rr.is_symlink() \
            or os.path.realpath(rr) != str(rr):
        preflight(f"runtime root {rr} is not the approved, non-symlinked {st.get('runtime_root')}")
    fs = filesystem_identity(rr)
    if not fs["filesystem_uuid"] or fs["filesystem_uuid"] != st.get("filesystem_uuid") \
            or fs["mount_point"] != st.get("mount_point"):
        preflight(f"output filesystem {fs['filesystem_uuid']} at {fs['mount_point']} is not the approved storage")
    if not isinstance(st.get("min_free_bytes_before_bank"), int) or fs["free_bytes"] < st["min_free_bytes_before_bank"]:
        preflight(f"free bytes {fs['free_bytes']} below the approved floor {st.get('min_free_bytes_before_bank')}")
    if not isinstance(st.get("min_free_inodes_before_bank"), int) or fs["free_inodes"] < st["min_free_inodes_before_bank"]:
        preflight(f"free inodes {fs['free_inodes']} below the approved floor {st.get('min_free_inodes_before_bank')}")
    run_dir = rr / "runs" / "m6" / method_id / f"seed_{seed}"
    if os.path.lexists(run_dir):
        preflight(f"run directory exists: {run_dir}")
    parent = rr / "runs" / "m6"
    if not parent.is_dir() or parent.is_symlink() or os.stat(parent).st_dev != os.stat(rr).st_dev:
        preflight("runs/m6 must be a real directory on the runtime filesystem")
    return {"head_commit": git_commit(), "candidate_commit": cand, "authority_record_sha256": sha256_bytes(araw),
            "machine_id": machine_id, "environment": live, "filesystem": fs, "run_dir": str(run_dir)}


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
        self.atime_checked_reads = 0

    def face_path(self, sample_id: str) -> Path:
        p = self.faces_root / self.sample_dataset[sample_id] / f"{sample_id}.png"
        require(not p.is_symlink(), f"face is a symlink: {p}")
        return p

    def read(self, sample_id: str) -> tuple[np.ndarray, str]:
        """Bytes -> SHA256 == M2 record -> PIL decode; require mode RGB 256x256 uint8."""
        from PIL import Image
        require(sample_id in self.face_sha256, f"sample {sample_id} is outside the verified population")
        p = self.face_path(sample_id)
        try:        # every mode: O_NOATIME, never a plain read (protects atime-based firewalls such as E04 H.val_zero)
            a0 = os.stat(p).st_atime_ns
            raw = read_bytes_noatime(p)
            a1 = os.stat(p).st_atime_ns
        except OSError as exc:
            raise BankError(f"face unreadable {sample_id}: {exc}") from exc
        require(a0 == a1, f"atime of {sample_id} changed across an O_NOATIME read")
        self.atime_checked_reads += 1
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
