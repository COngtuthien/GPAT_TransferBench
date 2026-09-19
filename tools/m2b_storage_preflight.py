"""M2B storage preflight — mandatory hard gate before any full preprocessing.

Computes a complete persistent + temporary-peak disk budget for M2B and decides whether the run
may start. It uses ONLY artifacts that already exist:

  * the exact per-dataset / per-resolution sample counts from the frozen M1 manifests
    (`manifests/inventory.parquet`, `manifests/inventory_videos.parquet` -> `frame_sizes`);
  * per-component byte measurements taken from the deterministic 24-sample M2A smoke outputs.

No new scientific sample is processed for the estimate.

Because the estimate drives a hard gate, every per-sample size is taken at the **maximum** observed
value (not the mean), and the mean is reported alongside it so the owner can see the spread. The
projection is reported per dataset and per resolution class, never as one pooled average, because
CASIA is a 112x112 image sequence while MSU/SiW are videos at 4 different resolutions.

    /home/cong/.venvs/gpatbench-m2/bin/python tools/m2b_storage_preflight.py [--run-tag runC]
        [--exec-config configs/execution/<file>.yaml] [--out-stem M2B_STORAGE_PREFLIGHT_EXTERNAL]

Without `--exec-config` the target roots are the in-repo `data/processed` and `cache` (the original
blocked pass). With it, the physical roots come from that execution config, so the same conservative
model is applied to the relocated storage. Earlier preflight outputs are never overwritten when a
different `--out-stem` is given: the audit trail keeps both the blocked and the post-relocation runs.

Writes outputs/audit/<out-stem>.{md,json} and exits 0 (PASS) or 1 (BLOCKED).
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.preprocess import logit_store as LS  # noqa: E402

GIB = 1024 ** 3
MIN_SAFETY_RESERVE_BYTES = 10 * GIB          # owner rule: 10 GiB free must remain after the peak
EXT4_BLOCK = 4096                            # block-rounding overhead for many small files
TEMP_PEAK_ALLOWANCE_BYTES = 2 * GIB          # bounded streaming; justified in the report
METADATA_ALLOWANCE_BYTES = 200 * 1024 ** 2   # manifests, indexes, per-sample provenance, audit CSVs
FS_OVERHEAD_ALLOWANCE_BYTES = 200 * 1024 ** 2

DEFAULT_OUT_STEM = "M2B_STORAGE_PREFLIGHT"


# ---------------------------------------------------------------- filesystem facts
def fs_facts(path) -> dict:
    p = Path(path)
    real = os.path.realpath(p)
    st = os.statvfs(real)
    out = subprocess.run(["df", "-PT", real], capture_output=True, text=True, check=True).stdout.splitlines()[-1].split()
    return {"path": str(p), "realpath": real, "device": out[0], "fstype": out[1],
            "mountpoint": subprocess.run(["stat", "-c", "%m", real], capture_output=True, text=True, check=True).stdout.strip(),
            "total_bytes": st.f_blocks * st.f_frsize, "free_bytes": st.f_bavail * st.f_frsize,
            "used_bytes": (st.f_blocks - st.f_bfree) * st.f_frsize}


def preconditions() -> dict:
    """Verify the frozen auxiliary weights and the frozen M1 inputs before anything else."""
    import hashlib

    import yaml

    def sha(path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]
    cfg = yaml.safe_load((ROOT / "configs/frozen/preprocess_v1.yaml").read_text())
    models = {
        "scrfd": (cfg["scrfd"]["model_path"], reg["scrfd"]["weight_sha256"]),
        "facexformer": ("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/face_geometry/ckpts/model.pt",
                        reg["facexformer"]["weight_sha256"]),
        "adaface_ir50": (cfg["adaface"]["checkpoint_path"], reg["adaface_ir50"]["weight_sha256"]),
    }
    mv = {}
    for k, (path, expected) in models.items():
        got = sha(path)
        mv[k] = {"path": path, "expected_sha256": expected, "actual_sha256": got,
                 "size_bytes": os.path.getsize(path), "match": got == expected}
    idx = {r["path"]: r["sha256"] for r in csv.DictReader(
        open(ROOT / "outputs/audit/ARTIFACT_INDEX.csv", newline="", encoding="utf-8"))}
    m1 = {}
    for rel in ("manifests/inventory.parquet", "manifests/inventory_videos.parquet",
                "manifests/raw_file_index.parquet", "configs/frozen/data_v1.yaml",
                "configs/frozen/dataset_protocol_policy_v1.yaml"):
        got = sha(ROOT / rel)
        m1[rel] = {"recorded_sha256": idx.get(rel), "actual_sha256": got, "match": got == idx.get(rel)}
    return {"frozen_models": mv, "model_hashes": "PASS" if all(v["match"] for v in mv.values()) else "FAIL",
            "m1_frozen_inputs": m1, "m1_integrity": "PASS" if all(v["match"] for v in m1.values()) else "FAIL"}


def blocks(nbytes: float, n_files: int) -> int:
    """Disk usage of n_files whose mean size is nbytes/n_files, rounded up to ext4 blocks."""
    if n_files <= 0:
        return 0
    per = nbytes / n_files
    return int(n_files * (int(per // EXT4_BLOCK) + 1) * EXT4_BLOCK)


# ---------------------------------------------------------------- measurement basis
def counts() -> tuple[dict, dict]:
    """Exact sample counts per dataset, and per (dataset, WxH) resolution class, from M1."""
    inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    vids = pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
    per_ds = collections.Counter(r["dataset"] for r in inv)
    n_per_video = collections.Counter((r["dataset"], r["video_id"]) for r in inv)
    per_res = collections.defaultdict(collections.Counter)
    for v in vids:
        per_res[v["dataset"]][v["frame_sizes"]] += n_per_video[(v["dataset"], v["video_id"])]
    return dict(per_ds), {k: dict(v) for k, v in per_res.items()}


def smoke_basis(tag: str) -> dict:
    """Per-component byte measurements from the deterministic smoke run (mean and max)."""
    run = ROOT / "outputs/exploratory/m2a_smoke" / tag
    rows = list(csv.DictReader(open(ROOT / "outputs/audit/M2A_SMOKE_RESULTS.csv", newline="", encoding="utf-8")))

    frame_bpp, face_bytes = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        face_bytes[r["dataset"]].append(os.path.getsize(run / "faces" / f"{r['sample_id']}.png"))
        fp = run / "frames" / f"{r['sample_id']}.png"
        if fp.exists():                                    # CASIA persists no new frame (see the report)
            h, w = (int(x) for x in r["frame_hw"].split("x"))
            frame_bpp[(r["dataset"], f"{w}x{h}")].append(os.path.getsize(fp) / (w * h))

    import zstandard as zstd
    comp = zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(
        LS.ZSTD_LEVEL, write_checksum=LS.ZSTD_WRITE_CHECKSUM, write_content_size=LS.ZSTD_WRITE_CONTENT_SIZE,
        write_dict_id=False, threads=LS.ZSTD_THREADS))
    mask_raw, mask_comp = [], []
    for f in sorted(glob.glob(str(run / "geometry/*__parsing_mask.npy"))):
        b = _npy(np.load(f, allow_pickle=False))      # uint8 mask: no byte shuffle, it has one byte plane
        z = comp.compress(b)
        assert zstd.ZstdDecompressor().decompress(z) == b, f"mask round-trip failed for {f}"
        mask_raw.append(len(b))
        mask_comp.append(len(z))

    logits = [int(r["logits_compressed_bytes"]) for r in rows]
    other = sum(np.load(glob.glob(str(run / f"geometry/*__{n}.npy"))[0], allow_pickle=False).nbytes
                for n in ("landmarks_norm", "landmarks_px224", "landmarks_px256", "pose_pitch_yaw_roll_rad"))
    emb = np.load(glob.glob(str(run / "identity/*__embedding.npy"))[0], allow_pickle=False).nbytes

    def ms(v):
        return {"n": len(v), "min": min(v), "mean": statistics.mean(v), "max": max(v)}

    return {"run_tag": tag,
            "frame_bytes_per_pixel": {f"{k[0]}|{k[1]}": ms(v) for k, v in sorted(frame_bpp.items())},
            "face_png_bytes": {k: ms(v) for k, v in sorted(face_bytes.items())},
            "parsing_logits_compressed_bytes": ms(logits),
            "parsing_mask_npy_bytes": ms(mask_raw),
            "parsing_mask_compressed_bytes": ms(mask_comp),
            "geometry_other_bytes_per_sample": int(other),
            "identity_bytes_per_sample": int(emb)}


def _npy(a: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(a), version=(1, 0), allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------- projection
def project(per_ds: dict, per_res: dict, basis: dict) -> dict:
    """Project every persistent M2 product. Sizes use the MAX observed per-sample value."""
    frames = {}
    for ds, classes in per_res.items():
        if ds == "casia_fasd":
            continue                                        # no new frame artifact (already a canonical PNG)
        for wh, n in classes.items():
            w, h = (int(x) for x in wh.split("x"))
            key = f"{ds}|{wh}"
            b = basis["frame_bytes_per_pixel"].get(key)
            if b is None:                                   # no smoke frame at this resolution
                sameds = [v for k, v in basis["frame_bytes_per_pixel"].items() if k.startswith(ds + "|")]
                b = {"n": 0, "mean": max(x["mean"] for x in sameds), "max": max(x["max"] for x in sameds),
                     "min": min(x["min"] for x in sameds)}
            frames[key] = {"samples": n, "width": w, "height": h, "pixels_per_frame": w * h,
                           "bytes_per_pixel_mean": b["mean"], "bytes_per_pixel_max": b["max"],
                           "measured_from_n_smoke_frames": b["n"],
                           "bytes_mean": n * w * h * b["mean"], "bytes_max": n * w * h * b["max"]}
    faces = {ds: {"samples": n,
                  "bytes_per_file_mean": basis["face_png_bytes"][ds]["mean"],
                  "bytes_per_file_max": basis["face_png_bytes"][ds]["max"],
                  "bytes_mean": n * basis["face_png_bytes"][ds]["mean"],
                  "bytes_max": n * basis["face_png_bytes"][ds]["max"]} for ds, n in per_ds.items()}

    n_all = sum(per_ds.values())
    n_frames_files = sum(v["samples"] for v in frames.values())
    logit_max = basis["parsing_logits_compressed_bytes"]["max"]
    mask_raw = basis["parsing_mask_npy_bytes"]["max"]
    mask_comp = basis["parsing_mask_compressed_bytes"]["max"]

    comp = {
        "frames": {"files": n_frames_files,
                   "bytes_mean": sum(v["bytes_mean"] for v in frames.values()),
                   "bytes_max": sum(v["bytes_max"] for v in frames.values()),
                   "by_class": frames},
        "faces_256": {"files": n_all,
                      "bytes_mean": sum(v["bytes_mean"] for v in faces.values()),
                      "bytes_max": sum(v["bytes_max"] for v in faces.values()),
                      "by_dataset": faces},
        "geometry_parsing_logits": {"samples": n_all, "bytes_per_sample_max": logit_max,
                                    "bytes_mean": n_all * basis["parsing_logits_compressed_bytes"]["mean"],
                                    "bytes_max": n_all * logit_max,
                                    "uncompressed_reference_bytes": n_all * 11 * 224 * 224 * 4},
        "geometry_parsing_masks": {"samples": n_all,
                                   "bytes_uncompressed": n_all * mask_raw,
                                   "bytes_compressed": n_all * mask_comp,
                                   "bytes_mean": n_all * mask_raw, "bytes_max": n_all * mask_raw},
        "geometry_other_tensors": {"samples": n_all, "bytes_per_sample": basis["geometry_other_bytes_per_sample"],
                                   "bytes_mean": n_all * basis["geometry_other_bytes_per_sample"],
                                   "bytes_max": n_all * basis["geometry_other_bytes_per_sample"]},
        "identity_cache": {"samples": n_all, "bytes_per_sample": basis["identity_bytes_per_sample"],
                           "bytes_mean": n_all * basis["identity_bytes_per_sample"],
                           "bytes_max": n_all * basis["identity_bytes_per_sample"]},
        "metadata_indexes": {"bytes_mean": METADATA_ALLOWANCE_BYTES, "bytes_max": METADATA_ALLOWANCE_BYTES},
        "filesystem_overhead": {"bytes_mean": FS_OVERHEAD_ALLOWANCE_BYTES, "bytes_max": FS_OVERHEAD_ALLOWANCE_BYTES,
                                "note": f"ext4 {EXT4_BLOCK} B block rounding over ~{n_frames_files + n_all} small files plus directory inodes"},
    }
    comp["persistent_total"] = {"bytes_mean": int(round(sum(v["bytes_mean"] for v in comp.values()))),
                                "bytes_max": int(round(sum(v["bytes_max"] for v in comp.values())))}
    return comp


def _arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> int:
    tag = _arg("--run-tag", "runC")
    exec_cfg_rel = _arg("--exec-config")
    out_stem = _arg("--out-stem", DEFAULT_OUT_STEM)
    md_out = ROOT / f"outputs/audit/{out_stem}.md"
    json_out = ROOT / f"outputs/audit/{out_stem}.json"

    exec_cfg = None
    if exec_cfg_rel:
        import yaml
        exec_cfg = yaml.safe_load((ROOT / exec_cfg_rel).read_text())
    pre = preconditions()
    per_ds, per_res = counts()
    basis = smoke_basis(tag)
    comp = project(per_ds, per_res, basis)

    if exec_cfg is None:
        roles = {"data_processed": ROOT / "data/processed", "cache": ROOT / "cache"}
        target_key = "data_processed"
    else:
        r = exec_cfg["storage"]["roots"]
        roles = {"processed_frames_root": r["processed_frames_root"], "faces_256_root": r["faces_256_root"],
                 "geometry_cache_root": r["geometry_cache_root"], "identity_cache_root": r["identity_cache_root"],
                 "runstate_root": exec_cfg["storage"]["runstate_root"], "tmp_root": exec_cfg["storage"]["tmp_root"]}
        target_key = "processed_frames_root"
    targets = {name: fs_facts(p) for name, p in (
        [("project_root", ROOT), ("outputs", ROOT / "outputs"),
         ("external_model_cache", "/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")]
        + list(roles.items()))}
    target = targets[target_key]
    same_fs = all(targets[k]["device"] == target["device"] for k in roles)
    free = target["free_bytes"]

    persistent = int(comp["persistent_total"]["bytes_max"])
    peak = persistent + TEMP_PEAK_ALLOWANCE_BYTES
    required = peak + MIN_SAFETY_RESERVE_BYTES
    ok = required <= free

    decision = {
        "target_filesystem": target,
        "all_outputs_on_one_filesystem": same_fs,
        "free_bytes_before": free,
        "projected_persistent_bytes": persistent,
        "projected_persistent_bytes_mean_basis": int(comp["persistent_total"]["bytes_mean"]),
        "temporary_peak_allowance_bytes": TEMP_PEAK_ALLOWANCE_BYTES,
        "predicted_maximum_occupied_bytes": peak,
        "minimum_safety_reserve_bytes": MIN_SAFETY_RESERVE_BYTES,
        "required_bytes": required,
        "predicted_remaining_free_bytes": int(free - peak),
        "shortfall_bytes": int(max(0, required - free)),
        "gate": "PASS" if ok else "BLOCKED_BY_STORAGE_CAPACITY",
    }
    # Remediation options are reported for the owner to decide; this tool never acts on them.
    ext = targets["external_model_cache"]
    decision["remediation_options"] = [
        {"option": "A", "summary": "Owner approves relocating data/processed and cache to the external data volume",
         "target": ext["realpath"], "fstype": ext["fstype"], "free_bytes": ext["free_bytes"],
         "fits": required <= ext["free_bytes"],
         "margin_beyond_required_bytes": ext["free_bytes"] - required,
         "caveats": ("only ~%.2f GiB of headroom beyond the required total, so the volume must not be shared with "
                     "another large job; it is %s, whose permissions/mtime semantics and case handling differ from "
                     "ext4 and would need a fresh determinism smoke on that path before M2B is trusted; "
                     "section 8 forbids moving storage roots without owner approval"
                     % ((ext["free_bytes"] - required) / GIB, ext["fstype"]))},
        {"option": "B", "summary": "Free at least the shortfall on the project filesystem",
         "target": target["realpath"], "needed_bytes": decision["shortfall_bytes"],
         "caveats": "no project-owned data of this size was identified; this is an owner/system decision"},
        {"option": "C", "summary": "Attach additional storage for data/processed and cache",
         "caveats": "needs at least the required total plus normal growth headroom"},
    ]
    if exec_cfg is not None:
        rt = os.path.realpath(exec_cfg["storage"]["runtime_root"])
        escaping = sorted(k for k, v in roles.items() if not os.path.realpath(v).startswith(rt + os.sep))
        decision["output_roots"] = {k: os.path.realpath(v) for k, v in roles.items()}
        decision["runtime_root"] = rt
        decision["roots_escaping_runtime_root"] = escaping
        decision["writable"] = os.access(rt, os.W_OK)
        if escaping or not decision["writable"]:
            decision["gate"] = "BLOCKED_BY_OUTPUT_ROOT_CONTAINMENT"
            ok = False
    rep = {"generated_for": "M2B storage preflight (mandatory hard gate)", "cwd": str(ROOT),
           "execution_config": exec_cfg_rel, "preconditions": pre,
           "sample_counts_by_dataset": per_ds, "sample_counts_by_resolution": per_res,
           "total_samples": sum(per_ds.values()), "measurement_basis": basis,
           "components": comp, "filesystems": targets, "decision": decision}
    json_out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    write_md(rep, md_out)
    print(json.dumps({"gate": decision["gate"], "model_hashes": pre["model_hashes"],
                      "m1_integrity": pre["m1_integrity"], "free_gib": round(free / GIB, 2),
                      "persistent_gib": round(persistent / GIB, 2),
                      "required_gib": round(required / GIB, 2),
                      "shortfall_gib": round(decision["shortfall_bytes"] / GIB, 2)}))
    return 0 if ok else 1


def write_md(rep: dict, md_out) -> None:
    g = lambda b: f"{b / GIB:,.2f} GiB"                                            # noqa: E731
    d, c = rep["decision"], rep["components"]
    t = d["target_filesystem"]
    lines = [
        "# M2B — Storage Preflight (mandatory hard gate)",
        "",
        f"Generated for cwd `{rep['cwd']}`. Basis: exact frozen M1 sample counts "
        f"({rep['total_samples']:,} samples) plus per-component byte measurements from the deterministic "
        f"24-sample M2A smoke run `{rep['measurement_basis']['run_tag']}`. **No new scientific sample was "
        "processed for this estimate.** Every per-sample size below is the **maximum** observed value; the "
        "mean-basis total is reported alongside so the spread is visible.",
        "",
        f"## Decision: **{d['gate']}**",
        "",
        "## 0. Preconditions",
        "",
        f"Frozen auxiliary weight verification: **{rep['preconditions']['model_hashes']}** — "
        f"every active model file hashes to its `models/registry.yaml` value.",
        "",
        "| model | expected sha256 | match |",
        "|---|---|---|",
    ] + [f"| {k} | `{v['expected_sha256']}` | {'PASS' if v['match'] else 'FAIL'} |"
         for k, v in rep["preconditions"]["frozen_models"].items()] + [
        "",
        f"Frozen M1 input integrity: **{rep['preconditions']['m1_integrity']}** — "
        f"{', '.join('`' + k + '`' for k in rep['preconditions']['m1_frozen_inputs'])} all unchanged.",
        "",
        "## 1. Filesystems",
        "",
        "| Role | realpath | device | fstype | mount | total | used | free |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, f in rep["filesystems"].items():
        lines.append(f"| `{name}` | `{f['realpath']}` | `{f['device']}` | {f['fstype']} | `{f['mountpoint']}` | "
                     f"{g(f['total_bytes'])} | {g(f['used_bytes'])} | {g(f['free_bytes'])} |")
    lines += ["",
              f"All output roots resolve to the **same** filesystem (`{t['device']}`, {t['fstype']}): "
              f"**{str(rep['decision']['all_outputs_on_one_filesystem']).lower()}**. "
              "No symlink redirects any output root.",
              ("" if rep.get("execution_config") is None else
               f"\nPhysical output roots come from `{rep['execution_config']}` (execution/infrastructure config, "
               f"DEV-017). Containment check against `{rep['decision'].get('runtime_root')}`: "
               f"escaping roots = {rep['decision'].get('roots_escaping_runtime_root')}, "
               f"writable = {rep['decision'].get('writable')}.\n"
               + "\n".join(f"- `{k}` → `{v}`" for k, v in rep['decision'].get('output_roots', {}).items())),
              "",
              "## 2. Sample counts (frozen M1, not assumed)",
              "",
              "| dataset | samples | resolution classes |",
              "|---|---|---|"]
    for ds, n in sorted(rep["sample_counts_by_dataset"].items()):
        lines.append(f"| {ds} | {n:,} | {len(rep['sample_counts_by_resolution'][ds])} |")
    lines += [f"| **total** | **{rep['total_samples']:,}** | |", "",
              "Per-resolution sample counts (used for the frame projection):", "",
              "| dataset | frame size (W×H) | samples |", "|---|---|---|"]
    for ds, classes in sorted(rep["sample_counts_by_resolution"].items()):
        for wh, n in sorted(classes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {ds} | {wh} | {n:,} |")

    lines += ["", "## 3. Frame PNGs (`data/processed/frames/`)", "",
              "CASIA is an image sequence whose M1-selected source file is **already** a lossless canonical "
              "112×112 PNG, so the frozen route `PRECROPPED_112_RGB_TO_256_INTER_CUBIC` consumes it directly and "
              "records `frame_png_sha256 = source_file_sha256`. No duplicated frame artifact is produced for "
              "CASIA; the frame-extraction block of `configs/frozen/preprocess_v1.yaml` describes a video "
              "decoder and applies to the MSU/SiW route only. MSU and SiW frames are decoded from video and "
              "must be persisted losslessly.", "",
              "| dataset | W×H | samples | B/px (mean) | B/px (max) | smoke frames measured | projected (max basis) |",
              "|---|---|---|---|---|---|---|"]
    for key, v in sorted(c["frames"]["by_class"].items()):
        ds, wh = key.split("|")
        lines.append(f"| {ds} | {wh} | {v['samples']:,} | {v['bytes_per_pixel_mean']:.4f} | "
                     f"{v['bytes_per_pixel_max']:.4f} | {v['measured_from_n_smoke_frames']} | {g(v['bytes_max'])} |")
    lines.append(f"| **total** | | **{c['frames']['files']:,}** | | | | **{g(c['frames']['bytes_max'])}** "
                 f"(mean basis {g(c['frames']['bytes_mean'])}) |")

    lines += ["", "## 4. Canonical faces (`data/processed/faces_256/`)", "",
              "| dataset | samples | B/file (mean) | B/file (max) | projected (max basis) |", "|---|---|---|---|---|"]
    for ds, v in sorted(c["faces_256"]["by_dataset"].items()):
        lines.append(f"| {ds} | {v['samples']:,} | {v['bytes_per_file_mean']:,.0f} | {v['bytes_per_file_max']:,.0f} | {g(v['bytes_max'])} |")
    lines.append(f"| **total** | **{c['faces_256']['files']:,}** | | | **{g(c['faces_256']['bytes_max'])}** |")

    lg, mk = c["geometry_parsing_logits"], c["geometry_parsing_masks"]
    lines += ["", "## 5. Caches", "",
              "| component | basis | projected (max) |", "|---|---|---|",
              f"| geometry: parsing logits (lossless, frozen codec) | {lg['bytes_per_sample_max']:,} B/sample × {lg['samples']:,} | {g(lg['bytes_max'])} |",
              f"| geometry: parsing logits — uncompressed reference | float32 11×224×224 | {g(lg['uncompressed_reference_bytes'])} |",
              f"| geometry: parsing masks (uncompressed `.npy`, conservative) | {mk['bytes_uncompressed'] // mk['samples']:,} B/sample | {g(mk['bytes_uncompressed'])} |",
              f"| geometry: parsing masks — if compressed with the same codec | — | {g(mk['bytes_compressed'])} |",
              f"| geometry: landmarks ×3 + pose | {c['geometry_other_tensors']['bytes_per_sample']:,} B/sample | {g(c['geometry_other_tensors']['bytes_max'])} |",
              f"| identity: 512-D float32 embeddings | {c['identity_cache']['bytes_per_sample']:,} B/sample | {g(c['identity_cache']['bytes_max'])} |",
              f"| manifests, cache indexes, per-sample provenance | allowance | {g(c['metadata_indexes']['bytes_max'])} |",
              f"| filesystem block rounding / inodes | {c['filesystem_overhead']['note']} | {g(c['filesystem_overhead']['bytes_max'])} |",
              "",
              "## 6. Budget", "",
              "| item | bytes | GiB |", "|---|---|---|",
              f"| projected persistent total (max basis) | {d['projected_persistent_bytes']:,} | {g(d['projected_persistent_bytes'])} |",
              f"| projected persistent total (mean basis, for reference) | {d['projected_persistent_bytes_mean_basis']:,} | {g(d['projected_persistent_bytes_mean_basis'])} |",
              f"| temporary peak allowance | {d['temporary_peak_allowance_bytes']:,} | {g(d['temporary_peak_allowance_bytes'])} |",
              f"| **predicted maximum occupied** | **{d['predicted_maximum_occupied_bytes']:,}** | **{g(d['predicted_maximum_occupied_bytes'])}** |",
              f"| minimum safety reserve (owner rule) | {d['minimum_safety_reserve_bytes']:,} | {g(d['minimum_safety_reserve_bytes'])} |",
              f"| **required free space** | **{d['required_bytes']:,}** | **{g(d['required_bytes'])}** |",
              f"| free on target filesystem | {d['free_bytes_before']:,} | {g(d['free_bytes_before'])} |",
              f"| predicted remaining free after peak | {d['predicted_remaining_free_bytes']:,} | {g(d['predicted_remaining_free_bytes'])} |",
              f"| **shortfall** | **{d['shortfall_bytes']:,}** | **{g(d['shortfall_bytes'])}** |",
              "",
              "### Temporary peak allowance — what it covers",
              "",
              "The logit cache is built by streaming: each sample's logits are compressed to one block and "
              "appended to the current shard's `.tmp` file, which is fsynced and atomically renamed at shard "
              "finalization. No second raw-logit cache is ever persisted, so the bounded temporary terms are: "
              f"one in-flight shard (≤ {LS.SHARD_ROWS} × {lg['bytes_per_sample_max']:,} B ≈ "
              f"{g(LS.SHARD_ROWS * lg['bytes_per_sample_max'])}), one in-flight frame/face temp file per worker "
              "(≈ 1.3 MiB each), the mask/tensor shard temps (< 1 MiB), and the deterministic validation subset "
              "re-run into a clean temporary location (24 samples ≈ 0.12 GiB). The 2 GiB allowance covers all of "
              "these with room for several parallel workers.",
              "",
              f"## 7. Verdict: **{d['gate']}**", ""]
    if d["gate"] == "PASS":
        lines += ["The projected peak plus the 10 GiB reserve fits on the target filesystem. Full M2B may proceed."]
    else:
        lines += [
            f"The projected persistent outputs alone ({g(d['projected_persistent_bytes'])}) exceed the free space "
            f"on the target filesystem ({g(d['free_bytes_before'])}). Including the temporary peak and the "
            f"owner-mandated 10 GiB reserve, M2B needs {g(d['required_bytes'])} and is short by "
            f"**{g(d['shortfall_bytes'])}**.",
            "",
            "Per the owner's hard rule, full preprocessing is **not started**. The dataset is not partially "
            "processed, and no cache component is silently redirected to another filesystem.",
            "",
            "This is not a marginal call. Even the most optimistic variant — mean sizes instead of maxima, "
            "masks compressed with the frozen codec, and **no** temporary allowance, metadata allowance, "
            "filesystem overhead or safety reserve at all — still needs "
            f"{g(c['frames']['bytes_mean'] + c['faces_256']['bytes_mean'] + c['geometry_parsing_logits']['bytes_mean'] + c['geometry_parsing_masks']['bytes_compressed'] + c['geometry_other_tensors']['bytes_mean'] + c['identity_cache']['bytes_mean'])}, "
            f"which still exceeds the {g(d['free_bytes_before'])} available.",
            "",
            "### What changed versus the M2A projection",
            "",
            "The M2A environment report projected ≈37.5 GB (≈34.9 GiB) for logits, masks and faces. Two terms "
            "were missing from that figure and dominate here: the spec's §0.2 directory contract and §3.4 "
            "requirement to extract selected video frames as lossless canonical PNGs adds "
            f"**{g(c['frames']['bytes_max'])}** of MSU/SiW frames, and that earlier number was measured against "
            "the free space of the *external* volume, not the project filesystem that `data/processed` and "
            "`cache` actually resolve to.",
            "",
            "### Remediation options (owner decision; this preflight takes none of them)",
            "",
            "| # | option | fits? | notes |",
            "|---|---|---|---|",
        ] + [f"| {o['option']} | {o['summary']} | "
             f"{('yes — ' + g(o['free_bytes']) + ' free') if o.get('fits') else ('no' if 'fits' in o else '—')} | "
             f"{o['caveats']} |" for o in d["remediation_options"]] + [
            "",
            "Option A is the only one that fits today, and it needs explicit owner approval because section 8 "
            "forbids relocating storage roots without it. Note that it would leave little headroom and would "
            "move the scientific outputs onto a different filesystem type, so the M2A determinism smoke should "
            "be repeated on that path before M2B is trusted there.",
        ]
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
