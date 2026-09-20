"""Freeze the SHA-256 records for every authoritative M4 artifact (prompt §39).

    <m2 venv>/bin/python tools/m4_freeze_hashes.py

One record per artifact actually created. A manifest that was not created gets no record and no
placeholder file.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"
ARTIFACTS = [
    ("pair_train_stats_v1.sha256", E.STATS_PATH, "json"),
    ("pairs_train_v1.sha256", E.MANIFEST_PATH["TRAIN"], "parquet"),
    ("val_pairs_v1.sha256", E.MANIFEST_PATH["VAL"], "parquet"),
]


def main() -> int:
    git = E.git_state()
    common = {
        "pairs_config": "configs/frozen/pairs_v1.yaml",
        "pairs_config_sha256": E.sha256_file(E.PAIRS_CONFIG),
        "split_manifest": "manifests/split_v1.parquet",
        "split_manifest_sha256": E.sha256_file(E.SPLIT_MANIFEST),
        "git_commit_at_creation": git["commit"],
        "git_dirty_at_creation": git["dirty"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "builder": "gpatbench.pairs.execute.run (python -m gpatbench.cli build-pairs)",
        "builder_module_sha256": E.sha256_file(ROOT / "gpatbench/pairs/execute.py"),
        "metric_module_sha256": E.sha256_file(ROOT / "gpatbench/pairs/common.py"),
    }
    out = {}
    for name, path, kind in ARTIFACTS:
        if not path.exists():
            continue
        rec = dict(common)
        rec.update({
            "artifact": path.relative_to(ROOT).as_posix(),
            "sha256": E.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "kind": kind,
        })
        if kind == "parquet":
            import pyarrow.parquet as pq
            rec.update({"rows": pq.read_metadata(path).num_rows,
                        "columns": list(E.COLUMNS),
                        "schema_signature": E.schema_signature(),
                        "row_order": list(E.ROW_ORDER),
                        "writer_contract": dict(E.PARQUET_WRITER)})
        else:
            rec.update({"schema_version": E.STATS_SCHEMA_VERSION,
                        "json_policy": dict(E.JSON_POLICY)})
        (AUDIT / name).write_bytes(E.canonical_json_bytes(rec))
        out[name] = rec["sha256"]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
