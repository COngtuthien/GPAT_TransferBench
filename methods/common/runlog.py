"""Run directories, manifests and append-only logging for M6, per `run_logging_v1`.

Implements the M6B logging contract: deterministic run directory, resolved_config.yaml,
run_manifest.json, append-only metrics.jsonl, checkpoint_index.json, run_summary.json.

Two safety properties are enforced:
  * atomic writes  — every JSON/YAML artifact is written to a temp file then os.replace()d, so a
    crash never leaves a half-written manifest;
  * resume safety  — an existing run directory is never silently overwritten. It either resumes
    (appending to metrics.jsonl, with an explicit reconciliation record) or raises.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from .config import load_logging_contract, load_method_config
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

RUN_FILES = {
    "resolved_config": "resolved_config.yaml",
    "run_manifest": "run_manifest.json",
    "metrics": "metrics.jsonl",
    "generation": "generation_log.jsonl",
    "checkpoint_index": "checkpoint_index.json",
    "run_summary": "run_summary.json",
    "stdout": "stdout.log",
    "stderr": "stderr.log",
}
CHECKPOINT_TYPES = ("periodic", "official", "terminal", "selected")


class RunDirectoryError(RuntimeError):
    """An existing run directory would have been silently overwritten, or is inconsistent."""


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(root: Path = ROOT) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, check=True)
        return out.stdout.decode().strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "UNKNOWN"


def git_dirty(root: Path = ROOT) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, check=True)
        return bool(out.stdout.decode().strip())
    except Exception:  # pragma: no cover
        return True


def compute_run_id(method_id: str, seed: int, config_sha256: str, commit: str) -> str:
    """run_logging_v1: sha256(method_id|seed|config_sha256|git_commit) hex, first 16 chars."""
    pre = f"{method_id}|{seed}|{config_sha256}|{commit}".encode("utf-8")
    return hashlib.sha256(pre).hexdigest()[:16]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_write_yaml(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).encode("utf-8"))


def environment_metadata() -> dict:
    import numpy as np
    return {
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gpu_model": None,
        "gpu_count": 0,
        "cuda_version": None,
        "cudnn_version": None,
        "framework": "NOT_APPLICABLE_NON_LEARNED_METHOD",
        "framework_version": None,
        "pytorch_version": None,
        "dependency_fingerprint": None,
        "environment_lock_path": None,
        "gpu_note": "E01/E02 are non-learned and CPU-only; no GPU job is run.",
    }


class RunContext:
    """One `<runtime_root>/runs/m6/<method_id>/seed_<seed>/` directory and its logs."""

    def __init__(self, *, method_id: str, seed: int, config: dict, runtime_root: str | Path,
                 command_line: str | None = None, resume: bool = False,
                 source_commits: dict | None = None):
        canonical = load_method_config(method_id)
        if config != canonical:
            raise RunDirectoryError("runtime config differs from verified frozen config")
        if type(seed) is not int or seed not in canonical["seeds"]["experiment_seeds"]:
            raise RunDirectoryError("experiment seed must be one of [42, 1337, 2026]")
        self.contract = load_logging_contract()
        self.method_id = method_id
        self.seed = int(seed)
        self.config = config
        self.runtime_root = Path(runtime_root)
        self.resume = resume
        expected_sources = {}
        source = config.get("source", {})
        if source.get("pinned_commit") not in (None, "NOT_APPLICABLE"):
            expected_sources[source["repository"]] = source["pinned_commit"]
        self.source_commits = source_commits if source_commits is not None else expected_sources
        if self.source_commits != expected_sources:
            raise RunDirectoryError("source commits disagree with frozen config")
        self.command_line = command_line if command_line is not None else " ".join(sys.argv)
        self.config_sha256 = config["_runtime"]["config_sha256"]
        self.config_path = config["_runtime"]["config_path"]
        self.git_commit = git_commit()
        self.run_id = compute_run_id(method_id, self.seed, self.config_sha256, self.git_commit)
        self.run_dir = self.runtime_root / "runs" / "m6" / method_id / f"seed_{self.seed}"
        self.start_utc: str | None = None
        self._metrics_fh = None
        self._records = 0
        self._generation_fh = None
        self._closed_summary = None
        self._lock_fh = None
        self.run_uuid = str(uuid.uuid4())
        self._started = time.monotonic()
        self._environment = environment_metadata()

    # ---------------------------------------------------------------- paths
    def path(self, key: str) -> Path:
        return self.run_dir / RUN_FILES[key]

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> "RunContext":
        if self._metrics_fh is not None or self._closed_summary is not None:
            raise RunDirectoryError("RunContext objects may be opened only once")
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = (self.run_dir.parent / f".seed_{self.seed}.lock").open("a")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._open_locked()
        except BaseException:
            for handle in (self._metrics_fh, self._generation_fh, self._lock_fh):
                if handle is not None:
                    handle.close()
            self._metrics_fh = self._generation_fh = self._lock_fh = None
            raise

    def _open_locked(self) -> "RunContext":
        existed = self.run_dir.exists() and any(self.run_dir.iterdir())
        if existed and not self.resume:
            raise RunDirectoryError(
                f"run directory already exists and is non-empty: {self.run_dir}. "
                f"Refusing to overwrite. Pass resume=True to append, or choose a new runtime root.")
        old = None
        if existed and self.resume:
            required = ("run_manifest", "resolved_config", "checkpoint_index", "metrics", "generation")
            if any(not self.path(key).is_file() for key in required):
                raise RunDirectoryError("resume refused: incomplete run artifacts")
            old = json.loads(self.path("run_manifest").read_text(encoding="utf-8"))
            expected = {"method_id": self.method_id, "experiment_seed": self.seed,
                        "run_id": self.run_id, "config_sha256": self.config_sha256,
                        "git_commit": self.git_commit, "source_commits": self.source_commits}
            if any(old.get(k) != v for k, v in expected.items()):
                raise RunDirectoryError("resume refused: run identity/config/source mismatch")
            if old.get("environment") != self._environment:
                raise RunDirectoryError("resume refused: environment differs")
            resolved = yaml.safe_load(self.path("resolved_config").read_text())
            if {k: v for k, v in resolved.items() if k != "_resolved"} != {
                    k: v for k, v in self.config.items() if k != "_runtime"}:
                raise RunDirectoryError("resume refused: resolved config differs")
            for key in ("metrics", "generation"):
                raw = self.path(key).read_text()
                if raw and not raw.endswith("\n"):
                    raise RunDirectoryError("resume refused: partial JSONL requires explicit reconciliation")
                try:
                    for line in raw.splitlines():
                        if not isinstance(json.loads(line), dict):
                            raise ValueError("record must be an object")
                except ValueError as exc:
                    raise RunDirectoryError("resume refused: invalid JSONL") from exc
            self.run_uuid = old["run_uuid"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self.start_utc = old["start_utc"] if old else _utc()
        if not old:
            atomic_write_yaml(self.path("resolved_config"), self._resolved_config())
        atomic_write_json(self.path("run_manifest"), self._manifest(status="running"))
        for key in ("stdout", "stderr"):
            self.path(key).touch(exist_ok=True)
        if not self.path("checkpoint_index").is_file():
            atomic_write_json(self.path("checkpoint_index"),
                              {"method_id": self.method_id, "experiment_seed": self.seed,
                               "run_id": self.run_id, "checkpoints": [],
                               "note": "E01/E02 are non-learned: no checkpoint is ever written."})
        # append-only: never truncate an existing metrics.jsonl
        prior_records = 0
        if self.path("metrics").is_file():
            with self.path("metrics").open("r", encoding="utf-8") as fh:
                prior_records = sum(1 for _ in fh)
        self._metrics_fh = self.path("metrics").open("a", encoding="utf-8")
        self._generation_fh = self.path("generation").open("a", encoding="utf-8")
        if existed and self.resume:
            self.log_event("resume_reconciliation", {
                "prior_record_count": prior_records,
                "policy": "APPEND_OR_EXPLICITLY_RECONCILE",
                "overwrite": False,
                "resumed_utc": _utc(),
                "prior_manifest": old,
                "prior_summary": json.loads(self.path("run_summary").read_text()) if self.path("run_summary").exists() else None,
            })
        return self

    def _resolved_config(self) -> dict:
        resolved = {k: v for k, v in self.config.items() if k != "_runtime"}
        resolved["_resolved"] = {
            "method_id": self.method_id,
            "experiment_seed": self.seed,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "snapshot_path": self.config["_runtime"]["snapshot_path"],
            "snapshot_verified": True,
            "runtime_root": str(self.runtime_root),
            "run_dir": str(self.run_dir),
            "git_commit": self.git_commit,
            "source_commits": self.source_commits,
            "scientific_override_applied": False,
            "note": "Verbatim copy of the frozen config. No scientific value was overridden.",
        }
        return resolved

    def _manifest(self, *, status: str, end_utc: str | None = None) -> dict:
        return {
            "method_id": self.method_id,
            "experiment_seed": self.seed,
            "run_id": self.run_id,
            "run_uuid": self.run_uuid,
            "git_commit": self.git_commit,
            "git_dirty": git_dirty(),
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "snapshot_verified": True,
            "source_commits": self.source_commits,
            "logging_contract": "run_logging_v1",
            "start_utc": self.start_utc,
            "end_utc": end_utc,
            "command_line": self.command_line,
            "environment": self._environment,
            **self._environment,
            "missing_field_reasons": {k: "Not applicable to CPU non-learned execution" if k not in
                ("dependency_fingerprint", "environment_lock_path") else "Runtime environment freeze pending; synthetic validation only"
                for k, v in self._environment.items() if v is None},
            "completion_status": status,
            "test_split_accessed": False,
        }

    # ---------------------------------------------------------------- logging
    @staticmethod
    def _validate_metadata(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"pixels", "image", "target_live", "source_spoof", "pixel_payload"}:
                    raise ValueError("pixel payloads are forbidden in logs")
                RunContext._validate_metadata(item)
        elif isinstance(value, (list, tuple)):
            if any(isinstance(item, (list, tuple)) for item in value) and len(value) > 40:
                raise ValueError("image-like nested payload forbidden in logs")
            for item in value:
                RunContext._validate_metadata(item)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise TypeError("logs accept metadata only, never arrays or binary payloads")

    def _append(self, record: dict) -> None:
        if self._metrics_fh is None:
            raise RunDirectoryError("RunContext.open() must be called before logging.")
        self._validate_metadata(record)
        self._metrics_fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._metrics_fh.flush()
        os.fsync(self._metrics_fh.fileno())
        self._records += 1

    def log_event(self, event: str, payload: dict | None = None) -> None:
        self._append({"record_type": "event", "event": event, "utc": _utc(),
                      "method_id": self.method_id, "experiment_seed": self.seed,
                      "payload": payload or {}})

    def log_generation(self, record: dict) -> None:
        metadata = record.get("metadata", {})
        item = {"record_type": "generation", "utc": _utc(), **record,
                "method_id": self.method_id, "experiment_seed": self.seed,
                "config_sha256": self.config_sha256,
                "pair_or_sample_id": record.get("pair_id"),
                "deterministic_pair_seed": metadata.get("operator_seed", metadata.get("pair_seed")),
                "operator_or_block_selection_metadata": metadata,
                "output_path": record.get("output_path", record.get("output_ref")),
                "output_sha256": record.get("output_sha256"),
                "failure_reason": record.get("failure_reason")}
        if self.method_id == "E01":
            item.update(selected_operator_name=metadata.get("operator"),
                        magnitude_level=metadata.get("level"),
                        sampled_operator_parameters=metadata.get("sampled_operator_parameters"),
                        selected_asset_filename=metadata.get("asset"),
                        asset_enumeration_rule=self.config["asset_enumeration"]["rule"])
        elif self.method_id == "E02":
            item.update(n_eligible=metadata.get("n_eligible"), k_selected=metadata.get("k"),
                        selected_block_indices_sorted_ascending=metadata.get("selected_indices"),
                        numpy_version=self._environment["numpy_version"], pcg64_seed_integer=metadata.get("pair_seed"))
        item["missing_field_reasons"] = {k: "Not applicable, unavailable on failure, or no persisted output; see generation_status and metadata"
                                          for k, v in item.items() if v is None}
        self._append(item)
        self._generation_fh.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        self._generation_fh.flush()
        os.fsync(self._generation_fh.fileno())

    def log_epoch(self, record: dict) -> None:  # pragma: no cover - learned methods, M6C2+
        if not self.config["learned_method"]:
            raise ValueError("non-learned methods must not invent epochs")
        required = self.contract["trajectory"]["required_fields"]
        reasons = record.get("missing_field_reasons", {})
        if any(k not in record or (record[k] is None and not reasons.get(k)) for k in required):
            raise ValueError("trajectory fields require values or explicit nulls with reasons")
        if any("test" in str(k).lower() for k in record):
            raise ValueError("TEST must not appear in trajectory")
        self._append({"record_type": "epoch", "utc": _utc(), **record})

    # ---------------------------------------------------------------- checkpoints
    def record_checkpoint(self, *, path: str, epoch, global_step, file_size_bytes: int,
                          sha256: str, checkpoint_type: str, selected_for_final: bool,
                          selection_reason: str) -> None:
        if checkpoint_type not in CHECKPOINT_TYPES:
            raise ValueError(f"checkpoint_type must be one of {CHECKPOINT_TYPES}")
        p = self.path("checkpoint_index")
        idx = json.loads(p.read_text(encoding="utf-8"))
        idx["checkpoints"].append({
            "path": path, "epoch": epoch, "global_step": global_step,
            "file_size_bytes": file_size_bytes, "sha256": sha256,
            "checkpoint_type": checkpoint_type, "selected_for_final": selected_for_final,
            "selection_reason": selection_reason,
        })
        atomic_write_json(p, idx)

    # ---------------------------------------------------------------- close
    def close(self, *, completion_status: str = "completed", summary: dict | None = None,
              failure_reason: str | None = None) -> dict:
        if self._closed_summary is not None:
            return self._closed_summary
        if summary and set(summary) & {"method_id", "experiment_seed", "run_id", "config_sha256"}:
            raise RunDirectoryError("summary may not overwrite run identity")
        if self._metrics_fh is None:
            raise RunDirectoryError("cannot close a run that was not opened")
        end = _utc()
        if self._generation_fh is not None:
            self._generation_fh.close()
            self._generation_fh = None
        if self._metrics_fh is not None:
            self._metrics_fh.close()
            self._metrics_fh = None
        atomic_write_json(self.path("run_manifest"),
                          self._manifest(status=completion_status, end_utc=end))
        run_summary = {
            "method_id": self.method_id,
            "experiment_seed": self.seed,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "final_or_selected_checkpoint_path": None,
            "final_or_selected_checkpoint_sha256": None,
            "checkpoint_selection_rule": self.config["checkpoint"]["rule"],
            "seed_level_evaluation_metrics": None,
            "training_duration_seconds": None,
            "peak_vram_bytes": None,
            "peak_vram_reason": "CPU-only non-learned method; no GPU job was run.",
            "completion_status": completion_status,
            "failure_reason": failure_reason,
            "records_written": self._records,
            "wall_clock_seconds": time.monotonic() - self._started,
            "output_manifest_path": None,
            "run_summary": str(self.path("run_summary")),
            "start_utc": self.start_utc,
            "end_utc": end,
            "test_split_accessed": False,
        }
        run_summary["missing_field_reasons"] = {k: "Non-learned synthetic validation: no training, evaluation, bank or checkpoint"
                                                for k, v in run_summary.items() if v is None}
        if summary:
            if set(summary) & {"method_id", "experiment_seed", "run_id", "config_sha256"}:
                raise RunDirectoryError("summary may not overwrite run identity")
            run_summary.update(summary)
        atomic_write_json(self.path("run_summary"), run_summary)
        if self._lock_fh is not None:
            self._lock_fh.close()
            self._lock_fh = None
        self._closed_summary = run_summary
        return run_summary

    def __enter__(self) -> "RunContext":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.close(completion_status="failed", failure_reason=f"{exc_type.__name__}: {exc}")
