"""Hermetic execution-storage environment for the M5 tests.

The M5 loader resolves the physical face store from an execution config selected by
`GPAT_M5_EXEC_CONFIG`, defaulting to the historical laptop config. That default names a path that
exists on the laptop and **not** on the GPU host, so any test that merely wants to prove
"dry-run writes no checkpoint" would otherwise fail on the GPU for an unrelated reason.

This module gives those tests their own temporary execution config with a real (empty) faces
directory, so they assert what they are about instead of where the author's data lives. It changes
no production semantics: the config still goes through `m2b.resolve_roots` containment, and the
authoritative path still requires a real face store.

Tests whose subject *is* the default-selection semantics deliberately do **not** use this; they
call `resolve_exec_config()` / `resolve_storage(require_faces=False)` directly.
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.probe import data as D  # noqa: E402

ENV = D.EXEC_CONFIG_ENV


def write_exec_config(tmp: Path, *, create_faces: bool = True,
                      faces: Path | None = None, runtime_root: Path | None = None) -> Path:
    """A structurally valid execution config inside `tmp`, contained and (by default) existing."""
    runtime_root = runtime_root or (tmp / "runtime")
    faces = faces or (runtime_root / "data" / "processed" / "faces_256")
    if create_faces:
        faces.mkdir(parents=True, exist_ok=True)
    else:
        runtime_root.mkdir(parents=True, exist_ok=True)
    cfg = tmp / "m5_exec_test.yaml"
    cfg.write_text(yaml.safe_dump({
        "version": "m5_execution_v1", "kind": "EXECUTION_INFRASTRUCTURE",
        "storage": {"runtime_root": str(runtime_root),
                    "roots": {"faces_256_root": str(faces)},
                    "runstate_root": str(runtime_root / "runstate"),
                    "tmp_root": str(runtime_root / "tmp")}}, sort_keys=True))
    return cfg


@contextmanager
def hermetic_exec_config(**kw):
    """Install a temporary `GPAT_M5_EXEC_CONFIG` and restore the caller's value afterwards."""
    saved = os.environ.get(ENV)
    with tempfile.TemporaryDirectory() as td:
        cfg = write_exec_config(Path(td), **kw)
        os.environ[ENV] = str(cfg)
        try:
            yield cfg
        finally:
            if saved is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = saved


def subprocess_env(cfg: Path | None = None) -> dict:
    """Environment for a CLI subprocess that never silently drops the selector."""
    env = dict(os.environ)
    if cfg is not None:
        env[ENV] = str(cfg)
    elif ENV not in env:
        raise AssertionError("subprocess_env() needs a config, or the selector already in os.environ")
    return env


class HermeticExecConfigMixin:
    """setUp/tearDown form of `hermetic_exec_config`, for whole storage-dependent test classes."""

    hermetic_create_faces = True

    def setUp(self):
        super().setUp()
        self._m5_saved_env = os.environ.get(ENV)
        self._m5_tmp = tempfile.TemporaryDirectory()
        self.exec_config = write_exec_config(Path(self._m5_tmp.name),
                                             create_faces=self.hermetic_create_faces)
        os.environ[ENV] = str(self.exec_config)

    def tearDown(self):
        if self._m5_saved_env is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._m5_saved_env
        self._m5_tmp.cleanup()
        super().tearDown()
