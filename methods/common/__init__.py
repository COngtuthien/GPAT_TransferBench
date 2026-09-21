"""Common M6 runtime: frozen-config loading, deterministic seeding, run directories and logging.

This is a thin additive layer over the existing repository structure, not a second framework. It
implements the `run_logging_v1` contract frozen in M6B (`configs/run_logging_v1.yaml`).
"""
from .config import (
    FrozenConfigError,
    load_method_config,
    sha256_bytes,
    sha256_file,
)
from .interface import GeneratorResult, MethodGenerator
from .runlog import RunContext, RunDirectoryError, git_commit

__all__ = [
    "FrozenConfigError",
    "GeneratorResult",
    "MethodGenerator",
    "RunContext",
    "RunDirectoryError",
    "git_commit",
    "load_method_config",
    "sha256_bytes",
    "sha256_file",
]
