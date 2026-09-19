# M2B — Storage Preflight (mandatory hard gate)

Generated for cwd `/home/cong/GPAT_TransferBench`. Basis: exact frozen M1 sample counts (20,640 samples) plus per-component byte measurements from the deterministic 24-sample M2A smoke run `runC`. **No new scientific sample was processed for this estimate.** Every per-sample size below is the **maximum** observed value; the mean-basis total is reported alongside so the spread is visible.

## Decision: **PASS**

## 0. Preconditions

Frozen auxiliary weight verification: **PASS** — every active model file hashes to its `models/registry.yaml` value.

| model | expected sha256 | match |
|---|---|---|
| scrfd | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | PASS |
| facexformer | `327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d` | PASS |
| adaface_ir50 | `52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8` | PASS |

Frozen M1 input integrity: **PASS** — `manifests/inventory.parquet`, `manifests/inventory_videos.parquet`, `manifests/raw_file_index.parquet`, `configs/frozen/data_v1.yaml`, `configs/frozen/dataset_protocol_policy_v1.yaml` all unchanged.

## 1. Filesystems

| Role | realpath | device | fstype | mount | total | used | free |
|---|---|---|---|---|---|---|---|
| `project_root` | `/home/cong/GPAT_TransferBench` | `/dev/nvme0n1p7` | ext4 | `/home` | 61.46 GiB | 21.80 GiB | 36.50 GiB |
| `outputs` | `/home/cong/GPAT_TransferBench/outputs` | `/dev/nvme0n1p7` | ext4 | `/home` | 61.46 GiB | 21.80 GiB | 36.50 GiB |
| `external_model_cache` | `/media/cong/Data/AI on IOT/Anti_spoofing/model_cache` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `processed_frames_root` | `/media/cong/Data/GPAT_TransferBench_runtime/data/processed/frames` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `faces_256_root` | `/media/cong/Data/GPAT_TransferBench_runtime/data/processed/faces_256` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `geometry_cache_root` | `/media/cong/Data/GPAT_TransferBench_runtime/cache/geometry` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `identity_cache_root` | `/media/cong/Data/GPAT_TransferBench_runtime/cache/identity` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `runstate_root` | `/media/cong/Data/GPAT_TransferBench_runtime/runstate` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |
| `tmp_root` | `/media/cong/Data/GPAT_TransferBench_runtime/tmp` | `/dev/nvme0n1p5` | ntfs3 | `/media/cong/Data` | 362.18 GiB | 297.07 GiB | 65.11 GiB |

All output roots resolve to the **same** filesystem (`/dev/nvme0n1p5`, ntfs3): **true**. No symlink redirects any output root.

Physical output roots come from `configs/execution/m2b_laptop_external_storage.yaml` (execution/infrastructure config, DEV-017). Containment check against `/media/cong/Data/GPAT_TransferBench_runtime`: escaping roots = [], writable = True.
- `processed_frames_root` → `/media/cong/Data/GPAT_TransferBench_runtime/data/processed/frames`
- `faces_256_root` → `/media/cong/Data/GPAT_TransferBench_runtime/data/processed/faces_256`
- `geometry_cache_root` → `/media/cong/Data/GPAT_TransferBench_runtime/cache/geometry`
- `identity_cache_root` → `/media/cong/Data/GPAT_TransferBench_runtime/cache/identity`
- `runstate_root` → `/media/cong/Data/GPAT_TransferBench_runtime/runstate`
- `tmp_root` → `/media/cong/Data/GPAT_TransferBench_runtime/tmp`

## 2. Sample counts (frozen M1, not assumed)

| dataset | samples | resolution classes |
|---|---|---|
| casia_fasd | 4,800 | 1 |
| msu_mfsd | 2,240 | 2 |
| siwmv2 | 13,600 | 3 |
| **total** | **20,640** | |

Per-resolution sample counts (used for the frame projection):

| dataset | frame size (W×H) | samples |
|---|---|---|
| casia_fasd | 112x112 | 4,800 |
| msu_mfsd | 720x480 | 1,120 |
| msu_mfsd | 640x480 | 1,120 |
| siwmv2 | 1920x1080 | 11,760 |
| siwmv2 | 720x1080 | 1,352 |
| siwmv2 | 1280x720 | 488 |

## 3. Frame PNGs (`data/processed/frames/`)

CASIA is an image sequence whose M1-selected source file is **already** a lossless canonical 112×112 PNG, so the frozen route `PRECROPPED_112_RGB_TO_256_INTER_CUBIC` consumes it directly and records `frame_png_sha256 = source_file_sha256`. No duplicated frame artifact is produced for CASIA; the frame-extraction block of `configs/frozen/preprocess_v1.yaml` describes a video decoder and applies to the MSU/SiW route only. MSU and SiW frames are decoded from video and must be persisted losslessly.

| dataset | W×H | samples | B/px (mean) | B/px (max) | smoke frames measured | projected (max basis) |
|---|---|---|---|---|---|---|
| msu_mfsd | 640x480 | 1,120 | 1.4032 | 1.5115 | 6 | 0.48 GiB |
| msu_mfsd | 720x480 | 1,120 | 0.9744 | 1.0408 | 2 | 0.38 GiB |
| siwmv2 | 1280x720 | 488 | 0.8791 | 0.8791 | 1 | 0.37 GiB |
| siwmv2 | 1920x1080 | 11,760 | 0.5433 | 0.6338 | 6 | 14.39 GiB |
| siwmv2 | 720x1080 | 1,352 | 0.6612 | 0.6612 | 1 | 0.65 GiB |
| **total** | | **15,840** | | | | **16.27 GiB** (mean basis 14.16 GiB) |

## 4. Canonical faces (`data/processed/faces_256/`)

| dataset | samples | B/file (mean) | B/file (max) | projected (max basis) |
|---|---|---|---|---|
| casia_fasd | 4,800 | 68,949 | 77,544 | 0.35 GiB |
| msu_mfsd | 2,240 | 96,366 | 103,275 | 0.22 GiB |
| siwmv2 | 13,600 | 82,358 | 87,926 | 1.11 GiB |
| **total** | **20,640** | | | **1.68 GiB** |

## 5. Caches

| component | basis | projected (max) |
|---|---|---|
| geometry: parsing logits (lossless, frozen codec) | 1,693,141 B/sample × 20,640 | 32.55 GiB |
| geometry: parsing logits — uncompressed reference | float32 11×224×224 | 42.44 GiB |
| geometry: parsing masks (uncompressed `.npy`, conservative) | 50,304 B/sample | 0.97 GiB |
| geometry: parsing masks — if compressed with the same codec | — | 0.04 GiB |
| geometry: landmarks ×3 + pose | 1,644 B/sample | 0.03 GiB |
| identity: 512-D float32 embeddings | 2,048 B/sample | 0.04 GiB |
| manifests, cache indexes, per-sample provenance | allowance | 0.20 GiB |
| filesystem block rounding / inodes | ext4 4096 B block rounding over ~36480 small files plus directory inodes | 0.20 GiB |

## 6. Budget

| item | bytes | GiB |
|---|---|---|
| projected persistent total (max basis) | 55,748,980,440 | 51.92 GiB |
| projected persistent total (mean basis, for reference) | 53,134,710,440 | 49.49 GiB |
| temporary peak allowance | 2,147,483,648 | 2.00 GiB |
| **predicted maximum occupied** | **57,896,464,088** | **53.92 GiB** |
| minimum safety reserve (owner rule) | 10,737,418,240 | 10.00 GiB |
| **required free space** | **68,633,882,328** | **63.92 GiB** |
| free on target filesystem | 69,916,213,248 | 65.11 GiB |
| predicted remaining free after peak | 12,019,749,160 | 11.19 GiB |
| **shortfall** | **0** | **0.00 GiB** |

### Temporary peak allowance — what it covers

The logit cache is built by streaming: each sample's logits are compressed to one block and appended to the current shard's `.tmp` file, which is fsynced and atomically renamed at shard finalization. No second raw-logit cache is ever persisted, so the bounded temporary terms are: one in-flight shard (≤ 256 × 1,693,141 B ≈ 0.40 GiB), one in-flight frame/face temp file per worker (≈ 1.3 MiB each), the mask/tensor shard temps (< 1 MiB), and the deterministic validation subset re-run into a clean temporary location (24 samples ≈ 0.12 GiB). The 2 GiB allowance covers all of these with room for several parallel workers.

## 7. Verdict: **PASS**

The projected peak plus the 10 GiB reserve fits on the target filesystem. Full M2B may proceed.
