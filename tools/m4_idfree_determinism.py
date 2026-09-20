"""Deterministic-rerun proof for the Track-A (identity-free) relations — Amendment A1 §34.

    <m2 venv>/bin/python tools/m4_idfree_determinism.py

Four independent fresh processes rebuild the DiffFAS Track-A manifest into their own clean
directories, and also rebuild the DSDG Track-A relation in memory:

    A  canonical input order,       PYTHONHASHSEED=0
    B  deliberately shuffled input, PYTHONHASHSEED=0
    C  canonical input order,       PYTHONHASHSEED=424242
    D  shuffled input (other seed), PYTHONHASHSEED=1

Requires identical membership, identical row order, identical bytes and identical SHA-256.
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
from gpatbench.pairs import execute as E    # noqa: E402
from gpatbench.pairs import track_a as T     # noqa: E402

AUDIT = ROOT / "outputs/audit"
RUNS = [("A", None, "0"), ("B", 20260814, "0"), ("C", None, "424242"), ("D", 7, "1")]

CHILD = """
import json, sys, hashlib
sys.path.insert(0, {root!r})
from pathlib import Path
from gpatbench.pairs import track_a as T
out = T.run(out_dir=Path(sys.argv[1]),
            shuffle_seed=(None if sys.argv[2] == "none" else int(sys.argv[2])))
dsdg = out["dsdg_rows"]
blob = "\\n".join(f'{{r["pair_id"]}}|{{r["dataset"]}}|{{r["source_spoof_id"]}}|{{r["target_live_id"]}}'
                 for r in dsdg).encode()
print(json.dumps({{"sha256": out["sha256"],
                  "dsdg_relation_sha256": hashlib.sha256(blob).hexdigest(),
                  "dsdg_rows": len(dsdg)}}))
"""


def main() -> int:
    authoritative = {T.DIFFFAS_MANIFEST.name: E.sha256_file(T.DIFFFAS_MANIFEST)}
    base_rows = pq.read_table(T.DIFFFAS_MANIFEST).to_pylist()
    base_schema = pq.read_schema(T.DIFFFAS_MANIFEST)
    pop = T.load_track_a_population()
    import hashlib
    base_dsdg = T.dsdg_training_relation(pop)
    base_dsdg_sha = hashlib.sha256("\n".join(
        f'{r["pair_id"]}|{r["dataset"]}|{r["source_spoof_id"]}|{r["target_live_id"]}'
        for r in base_dsdg).encode()).hexdigest()

    tmp_root = Path(tempfile.mkdtemp(prefix="m4_idfree_determinism_"))
    report = {"authoritative_sha256": authoritative,
              "dsdg_relation_sha256": base_dsdg_sha, "dsdg_rows": len(base_dsdg),
              "dsdg_separate_manifest": False,
              "dsdg_relation_note": "DSDG-BIN-IDFREE reuses manifests/pairs_train_v1.parquet; the "
                                    "hash above is over the identity-free adapter projection "
                                    "(pair_id|dataset|source_spoof_id|target_live_id).",
              "runs": {}, "mismatches": []}
    try:
        for name, shuffle, hashseed in RUNS:
            d = tmp_root / name
            d.mkdir()
            r = subprocess.run(
                [sys.executable, "-c", CHILD.format(root=str(ROOT)), str(d),
                 "none" if shuffle is None else str(shuffle)],
                capture_output=True, text=True,
                env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                check=True)
            got = json.loads(r.stdout.strip().splitlines()[-1])
            files = {p.name: E.sha256_file(p) for p in sorted(d.iterdir())}
            rows = pq.read_table(d / T.DIFFFAS_MANIFEST.name).to_pylist()
            entry = {
                "shuffle_seed": shuffle, "pythonhashseed": hashseed,
                "reported_sha256": got["sha256"],
                "on_disk_sha256": files,
                "byte_identical": files == authoritative,
                "rows_identical": rows == base_rows,
                "schema_identical": pq.read_schema(d / T.DIFFFAS_MANIFEST.name) == base_schema,
                "membership_identical": (
                    {(x["gt_spoof_id"], x["guide_spoof_id"]) for x in rows} ==
                    {(x["gt_spoof_id"], x["guide_spoof_id"]) for x in base_rows}),
                "track_pair_id_identical": (
                    [x["track_pair_id"] for x in rows] ==
                    [x["track_pair_id"] for x in base_rows]),
                "dsdg_relation_identical": got["dsdg_relation_sha256"] == base_dsdg_sha,
                "dsdg_rows": got["dsdg_rows"],
            }
            if not (entry["byte_identical"] and entry["rows_identical"]
                    and entry["dsdg_relation_identical"]):
                report["mismatches"].append(name)
            report["runs"][name] = entry
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    report["status"] = "PASS" if not report["mismatches"] else "FAIL"
    (AUDIT / "M4_IDFREE_DETERMINISM_COMPARE.json").write_bytes(E.canonical_json_bytes(report))
    print(json.dumps({"status": report["status"],
                      "runs": {k: v["byte_identical"] for k, v in report["runs"].items()},
                      "authoritative": authoritative,
                      "dsdg_relation_sha256": base_dsdg_sha}, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
