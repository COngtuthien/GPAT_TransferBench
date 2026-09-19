# M2B — Storage Decision Trail

The audit trail of the M2B storage question, in order. Neither preflight was rewritten: the blocked
run and the post-relocation run are separate, independently reproducible artifacts.

## 1. PREVIOUS — project filesystem → **BLOCKED**

Evidence: `M2B_STORAGE_PREFLIGHT.md` / `.json` (commit `1fdb73c8`).

| | |
|---|---|
| Target | `/dev/nvme0n1p7` (ext4), mount `/home` |
| Roles | `data/processed/frames`, `data/processed/faces_256`, `cache/geometry`, `cache/identity` inside the repo |
| Free | 36.50 GiB |
| Required | 63.92 GiB (persistent 51.92 + peak 2.00 + 10 GiB reserve) |
| Shortfall | **27.42 GiB** |
| Decision | **BLOCKED_BY_STORAGE_CAPACITY** — full preprocessing was not started, nothing was partially processed |

## 2. OWNER DECISION — external storage approved

The owner approved relocating the *physical* storage of M2 generated artifacts to
`/media/cong/Data`, under the dedicated runtime root
`/media/cong/Data/GPAT_TransferBench_runtime`. Recorded as **DEV-017**
(`APPROVED_OPERATIONAL_STORAGE_RELOCATION`) in `deviation_report.md`.

The Git repository stays at `/home/cong/GPAT_TransferBench` and remains authoritative for code,
configs, manifests, tests, tools, audit records and hashes. The physical roots live in an execution
config (`configs/execution/m2b_laptop_external_storage.yaml`), which is infrastructure only: the
runner refuses it if any root escapes the approved runtime root, and asserts the frozen scientific
contract's sha256 before it will run. No project directory was replaced by a symlink.

Before anything was written, the volume was audited (`findmnt`, `df -B1`, `stat -f`) and proven to
support `fsync`, atomic `rename`, directory `fsync` and case-sensitive names.

## 3. NEW — external filesystem → **PASS**

Evidence: `M2B_STORAGE_PREFLIGHT_EXTERNAL.md` / `.json`. Same conservative model, same 10 GiB
safety reserve — the reserve was **not** reduced to force a pass.

| | |
|---|---|
| Target | `/dev/nvme0n1p5` (ntfs3), mount `/media/cong/Data` |
| Runtime root | `/media/cong/Data/GPAT_TransferBench_runtime` (writable: true; roots escaping it: []) |
| Free | 65.11 GiB |
| Required | 63.92 GiB |
| Predicted free after peak | 11.19 GiB |
| Headroom beyond required | **1.19 GiB** |
| Decision | **PASS** |

⚠️ The margin is thin: the projection clears the gate by only
1.19 GiB. Free space is therefore checked at every chunk boundary during the
run, and execution halts at a resumable boundary if it approaches the 10 GiB reserve.

## 4. Proof that relocation changed nothing scientific

Evidence: `M2B_EXTERNAL_SMOKE_COMPARE.json`.

The frozen 24-sample M2A smoke manifest (sha256 `138f5929f81313069e4a4cb74c4ed54e4957432fe9ebc93cc52fdd14100dbce7`,
**not** reselected) was re-run writing to the relocated filesystem and compared against the frozen
`/home` run:

- **210/210 files byte-identical**;
- results table equal except wall-clock timing;
- max float difference: {};
- status: **PASS**.

Frame pixels, canonical face PNGs, SCRFD detections and selected bbox, crop geometry and
zero-padding metadata, FaceXFormer logits/mask/landmarks/pose, AdaFace embeddings and the
compressed logit shard all match exactly. Absolute path fields differ by construction and are not
scientific values — no stored scientific identity depends on a path string.
