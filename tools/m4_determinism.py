"""Deterministic-rerun proof for the authoritative M4 common artifacts (prompt §37).

    <m2 venv>/bin/python tools/m4_determinism.py

Runs the whole build again in independent fresh processes:

    A  canonical input order,       PYTHONHASHSEED=0
    B  deliberately shuffled input, PYTHONHASHSEED=0
    C  canonical input order,       PYTHONHASHSEED=424242
    D  shuffled input (other seed), PYTHONHASHSEED=1

Each writes into its own clean directory; the files are then compared byte-for-byte against the
authoritative manifests. Row-level membership is compared too, so a byte match cannot hide a
schema-level coincidence. Nothing is written to manifests/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"
RUNS = [("A", None, "0"), ("B", 20260814, "0"), ("C", None, "424242"), ("D", 7, "1")]

CHILD = """
import json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from gpatbench.pairs import execute as E
out = E.run(out_dir=Path(sys.argv[1]), shuffle_seed=(None if sys.argv[2] == "none" else int(sys.argv[2])))
print(json.dumps(out["sha256"]))
"""


def main() -> int:
    authoritative = {p.name: E.sha256_file(p) for p in
                     (E.STATS_PATH, E.MANIFEST_PATH["TRAIN"], E.MANIFEST_PATH["VAL"])}
    base = {s: pq.read_table(E.MANIFEST_PATH[s]).to_pylist() for s in ("TRAIN", "VAL")}
    tmp_root = Path(tempfile.mkdtemp(prefix="m4_determinism_"))
    report = {"authoritative_sha256": authoritative, "runs": {}, "mismatches": []}
    try:
        for name, shuffle, hashseed in RUNS:
            d = tmp_root / name
            d.mkdir()
            r = subprocess.run(
                [sys.executable, "-c", CHILD.format(root=str(ROOT)), str(d),
                 "none" if shuffle is None else str(shuffle)],
                capture_output=True, text=True,
                env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin",
                     "HOME": str(Path.home())}, check=True)
            got = json.loads(r.stdout.strip().splitlines()[-1])
            files = {p.name: E.sha256_file(p) for p in sorted(d.iterdir())}
            entry = {"shuffle_seed": shuffle, "pythonhashseed": hashseed,
                     "reported_sha256": got, "on_disk_sha256": files,
                     "byte_identical": files == authoritative}
            rows_identical = {}
            for s in ("TRAIN", "VAL"):
                rows = pq.read_table(d / E.MANIFEST_PATH[s].name).to_pylist()
                rows_identical[s] = rows == base[s]
                schema_ok = (pq.read_schema(d / E.MANIFEST_PATH[s].name) ==
                             pq.read_schema(E.MANIFEST_PATH[s]))
                entry.setdefault("schema_identical", {})[s] = schema_ok
                entry.setdefault("membership_identical", {})[s] = (
                    {(x["source_spoof_id"], x["target_live_id"]) for x in rows} ==
                    {(x["source_spoof_id"], x["target_live_id"]) for x in base[s]})
                entry.setdefault("pair_id_identical", {})[s] = (
                    [x["pair_id"] for x in rows] == [x["pair_id"] for x in base[s]])
            entry["rows_identical"] = rows_identical
            if not (entry["byte_identical"] and all(rows_identical.values())):
                report["mismatches"].append(name)
            report["runs"][name] = entry
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    report["status"] = "PASS" if not report["mismatches"] else "FAIL"
    (AUDIT / "M4_DETERMINISM_COMPARE.json").write_bytes(E.canonical_json_bytes(report))
    print(json.dumps({"status": report["status"],
                      "runs": {k: v["byte_identical"] for k, v in report["runs"].items()},
                      "authoritative": authoritative}, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
