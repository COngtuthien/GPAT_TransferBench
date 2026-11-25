"""M10 downstream input verification on the GPU host (read-only; no training, no TEST).

  env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/m10_bank_loader_verification.py --repo <clean clone> \
      --runtime-root ~/workdir/GPAT_TransferBench_runtime > bank_loader_verification.json

1. real populations (gpatbench/downstream/science_inputs.build_populations; TRAIN/VAL only) and their relation to the
   pair manifest (every TRAIN spoof frame is one pair source; every pair target is a TRAIN live frame);
2. every one of the 21 banks through gpatbench/downstream/science_banks.resolve_bank: committed audit binding, the
   family's library reader with FULL file + pixel verification of all 8838 outputs, manifest sha256 = audited value,
   sample ids (pair_id / DSDG synthetic_id), labels, canonical order, seed mapping; then 64 deterministic random samples
   decoded through the training-path loader (load_synthetic_image: sha256 before decode, RGB 256x256, no conversion)
   and re-hashed to their manifest pixel identity;
3. E05: the low-contrast rows (pixel range <= 40 grey levels in the sample) are present and loaded unchanged;
   E04: rows whose raw generator output left [0, 1] (clipped by E04-Q3) are present and loaded unchanged;
4. 64 TRAIN live, 64 TRAIN spoof and 64 VAL faces read through FaceReader (O_NOATIME, M2 sha256, atime unchanged).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def rng_for(text):
    import numpy as np
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runtime-root", required=True)
    a = ap.parse_args()
    repo = Path(a.repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import pyarrow.parquet as pq
    from gpatbench.downstream import science_banks as SB, science_inputs as SI
    from gpatbench.downstream.faceio import FaceReader
    from methods.common import learned_bank as lb
    t0 = time.time()
    doc = {"label": "M10 input verification (read-only)", "repo": str(repo),
           "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pop = SI.build_populations(repo)
    pools = pop["pools"]
    pairs = pq.read_table(str(repo / "manifests/pairs_train_v1.parquet"),
                          columns=["pair_id", "target_live_id", "source_spoof_id"]).to_pylist()
    spoof_ids = {f["sample_id"] for f in pools["train_real_spoof"]}
    live_ids = {f["sample_id"] for f in pools["train_live"]}
    doc["populations"] = {"counts": pop["counts"], "pool_sha256": pop["pool_sha256"],
                          "split_manifest_sha256": pop["split_manifest_sha256"],
                          "pair_sources_equal_train_spoof": {p["source_spoof_id"] for p in pairs} == spoof_ids
                          and len(pairs) == len(spoof_ids),
                          "pair_targets_train_live": {p["target_live_id"] for p in pairs} <= live_ids}
    banks = {}
    for m in SB.METHODS:
        for s in SB.SEEDS:
            key = "%s|%d" % (m, s)
            try:
                r = SB.resolve_bank(repo, m, s)
                smp = r["samples"]
                g = rng_for("m10.loader.verify|%s|%d" % (m, s))
                idx = sorted(int(i) for i in g.choice(len(smp), 64, replace=False))
                loaded, ranges = 0, []
                for i in idx:
                    img = SB.load_synthetic_image(smp[i])
                    if SB.pixel_identity(img) != smp[i].output_pixel_sha256:
                        raise SB.ScienceBankError("pixel identity differs for %s" % smp[i].sample_id)
                    arr = np.asarray(img)
                    ranges.append(int(arr.max()) - int(arr.min()))
                    loaded += 1
                rec = {"binding": r["binding"], "reader": r["reader"], "outputs": len(smp), "failures": len(r["failures"]),
                       "samples_sha256": r["samples_sha256"], "sample_id_kind": "synthetic_id" if m in SB.DSDG_METHODS
                       else "pair_id", "first_ids": [x.sample_id for x in smp[:2]],
                       "labels_spoof": all(x.y_true == 1 for x in smp), "seed_mapping": {x.bank_seed for x in smp} == {s},
                       "random_loaded": loaded, "random_pixel_range": {"min": min(ranges), "median": float(np.median(ranges)),
                                                                       "max": max(ranges)}}
                if m == "E05":
                    low = [smp[i] for i, rg in zip(idx, ranges) if rg <= 40]
                    rec["e05_low_contrast_in_sample"] = len(low)
                    rec["e05_low_contrast_loaded_unchanged"] = all(
                        SB.pixel_identity(SB.load_synthetic_image(x)) == x.output_pixel_sha256 for x in low)
                if m == "E04":
                    d = lb.resolve_registered_bank(m, s)
                    rows = [json.loads(x) for x in (d / lb.MANIFEST).read_text().splitlines()]
                    clipped = [x for x in rows if x["raw_output_min"] < 0 or x["raw_output_max"] > 1]
                    by = {x.sample_id: x for x in smp}
                    pick = [by[x["row_id"]] for x in sorted(clipped, key=lambda x: x["raw_output_max"] -
                                                           x["raw_output_min"], reverse=True)[:16]]
                    rec["e04_rows_raw_outside_0_1"] = len(clipped)
                    rec["e04_clipped_rows_present"] = all(x["row_id"] in by for x in clipped)
                    rec["e04_widest_16_loaded_unchanged"] = all(
                        SB.pixel_identity(SB.load_synthetic_image(x)) == x.output_pixel_sha256 for x in pick)
                rec["ok"] = (rec["outputs"] == SB.N_SYN and rec["failures"] == 0 and rec["labels_spoof"]
                             and rec["seed_mapping"] and loaded == 64 and r["reader"]["ok"] is True
                             and r["reader"]["files_checked"] == SB.N_SYN
                             and rec.get("e05_low_contrast_loaded_unchanged", True)
                             and rec.get("e04_clipped_rows_present", True) and rec.get("e04_widest_16_loaded_unchanged", True))
            except Exception as exc:
                rec = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            banks[key] = rec
    doc["banks"] = banks
    reader = FaceReader(Path(a.runtime_root) / "data" / "processed" / "faces_256")
    faces = {}
    for k in ("train_live", "train_real_spoof", "val"):
        g = rng_for("m10.loader.verify.faces|" + k)
        sel = [pools[k][int(i)] for i in sorted(g.choice(len(pools[k]), 64, replace=False))]
        ok = 0
        for f in sel:
            img = reader.read(f["face_rel_path"], f["face_sha256"])
            ok += img.size == (256, 256) and img.mode == "RGB"
        faces[k] = {"read": len(sel), "ok": ok}
    doc["real_faces"] = {"sampled": faces, "ledger": reader.summary()}
    doc["all_ok"] = (all(b["ok"] for b in banks.values()) and len(banks) == 21
                     and doc["populations"]["pair_sources_equal_train_spoof"] and doc["populations"]["pair_targets_train_live"]
                     and all(v["ok"] == v["read"] for v in faces.values()) and reader.summary()["atime_changed_files"] == 0)
    doc["wall_seconds"] = time.time() - t0
    doc["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
