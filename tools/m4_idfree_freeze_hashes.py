"""Freeze SHA-256 records for every artifact Amendment A1 introduces (§40).

    <m2 venv>/bin/python tools/m4_idfree_freeze_hashes.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E    # noqa: E402
from gpatbench.pairs import track_a as T     # noqa: E402

AUDIT = ROOT / "outputs/audit"
AMENDMENT = ROOT / "docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md"

ARTIFACTS = [
    ("amendment_a1.sha256", AMENDMENT, "amendment"),
    ("fair_track_v1.sha256", T.FAIR_TRACK_CONFIG, "config"),
    ("dsdg_bin_idfree_v1.sha256", T.DSDG_CONFIG, "config"),
    ("difffas_bin_idfree_v1.sha256", T.DIFFFAS_CONFIG, "config"),
    ("difffas_bin_idfree_train_v1.sha256", T.DIFFFAS_MANIFEST, "parquet"),
    ("third_party_source_pins.sha256", T.SOURCE_PINS, "provenance"),
]


def main() -> int:
    git = E.git_state()
    common = {
        "amendment": "A1",
        "amendment_path": AMENDMENT.relative_to(ROOT).as_posix(),
        "amendment_sha256": E.sha256_file(AMENDMENT),
        "original_spec_sha256": "f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e",
        "original_spec_modified": False,
        "split_manifest_sha256": E.sha256_file(E.SPLIT_MANIFEST),
        "common_pairs_train_sha256": E.sha256_file(E.MANIFEST_PATH["TRAIN"]),
        "fair_track_config_sha256": E.sha256_file(T.FAIR_TRACK_CONFIG),
        "git_commit_at_creation": git["commit"],
        "git_dirty_at_creation": git["dirty"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }
    out = {}
    for name, path, kind in ARTIFACTS:
        if not path.is_file():
            continue
        rec = dict(common)
        rec.update({"artifact": path.relative_to(ROOT).as_posix(),
                    "sha256": E.sha256_file(path),
                    "size_bytes": path.stat().st_size, "kind": kind})
        if kind == "config":
            snap = ROOT / "frozen_config_snapshot" / path.relative_to(ROOT)
            rec["snapshot"] = snap.relative_to(ROOT).as_posix() if snap.is_file() else None
            rec["snapshot_sha256"] = E.sha256_file(snap) if snap.is_file() else None
            rec["snapshot_byte_identical"] = rec["snapshot_sha256"] == rec["sha256"]
        if kind == "parquet":
            rec.update({"rows": pq.read_metadata(path).num_rows,
                        "columns": list(T.DIFFFAS_COLUMNS),
                        "schema_signature": T.difffas_schema_signature(),
                        "row_order": list(T.ROW_ORDER),
                        "writer_contract": dict(E.PARQUET_WRITER),
                        "builder": "gpatbench.pairs.track_a.run",
                        "builder_module_sha256": E.sha256_file(ROOT / "gpatbench/pairs/track_a.py"),
                        "deviation": "DEV-021",
                        "official_source_commit": T.official_source_commit("difffas")})
        (AUDIT / name).write_bytes(E.canonical_json_bytes(rec))
        out[name] = rec["sha256"]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
