# M5 — GPU Execution Plan (E-M5-01)

**Date:** 2026-09-20 · **Status: PLAN ONLY — nothing was transferred and nothing was executed.**
Every remote value below is a field to be **measured**, never assumed. No number in this document
describes the remote machine; the remote columns are deliberately empty.

| | |
|---|---|
| remote host | `sparc5090` |
| expected remote project root | `/home/sparc/workdir/longnm/GPAT_TransferBench` |
| expected remote runtime root | `/home/sparc/workdir/longnm/GPAT_TransferBench_runtime` |
| local runtime root | `/media/cong/Data/GPAT_TransferBench_runtime` |
| frozen config | `configs/frozen/artifact_probe.yaml` sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` |

## 1. Step 1 — repository

Sync the repository at the frozen M5 commit (this pass's commit, once pushed). Verify remotely:

```
git -C /home/sparc/workdir/longnm/GPAT_TransferBench rev-parse HEAD
git -C /home/sparc/workdir/longnm/GPAT_TransferBench status --porcelain     # must be empty
sha256sum configs/frozen/artifact_probe.yaml                                 # must equal e263b370...
sha256sum manifests/split_v1.parquet                                         # must equal fb9aeb36...
```

## 2. Step 2 — minimal data bundle

M5 needs **only** the frozen canonical faces plus the manifests already in the repository. The
trainer reads `faces_256` and nothing else: no geometry cache, no identity cache, no raw videos,
no extracted frames, no pair manifests.

| item | count | size | transfer |
|---|---|---|---|
| `data/processed/faces_256/casia_fasd` | 4,800 | 318 MB | **yes** |
| `data/processed/faces_256/msu_mfsd` | 2,240 | 209 MB | **yes** |
| `data/processed/faces_256/siwmv2` | 13,575 | 1.2 GB | **yes** |
| **faces total** | **20,615** | **≈ 1.7 GB** | **yes** |
| `cache/geometry` | 972 files | 33.4 GiB | **no** — not read by M5 |
| `cache/identity` | 162 files | 0.05 GiB | **no** |
| `data/processed/frames` | 15,840 | 15.1 GiB | **no** |
| raw datasets | — | — | **no** |

Only 20,615 of the faces are needed and all of them are needed: TRAIN (14,467) + VAL (3,121) =
17,588 are read during training, and the remaining 3,027 are TEST, which the loader refuses to
touch. **Transferring TEST faces is optional**; if the owner prefers, exclude them — the trainer
cannot load them either way.

### Transfer command (resume-safe, not executed)

```
rsync -aHAX --partial --append-verify --info=progress2 \
      --exclude='*.tmp' \
      /media/cong/Data/GPAT_TransferBench_runtime/data/processed/faces_256/ \
      sparc5090:/home/sparc/workdir/longnm/GPAT_TransferBench_runtime/data/processed/faces_256/
```

`--partial --append-verify` makes an interrupted transfer resumable without re-sending completed
files; `-a` preserves times so a re-run is a no-op.

### Post-transfer verification (not executed)

```
# 1. file count must equal the local count exactly
ssh sparc5090 'find /home/sparc/workdir/longnm/GPAT_TransferBench_runtime/data/processed/faces_256 \
                 -type f -name "*.png" | wc -l'          # expect 20615

# 2. per-dataset counts
#    casia_fasd 4800 | msu_mfsd 2240 | siwmv2 13575

# 3. content integrity against the M2 provenance already frozen in the repository:
#    manifests/m2_sample_accounting.parquet carries face_png_sha256 for every sample.
#    Re-hash remotely and compare all 20,615 digests; ANY mismatch stops M5.
python -m gpatbench.cli ...   # or a small remote script that reads m2_sample_accounting.parquet
```

Integrity is verified against the **frozen M2 provenance**, not against a freshly computed local
manifest, so a corrupted local copy could not certify itself.

## 3. Step 3 — remote environment preflight (measure, never assume)

Run before training and record every value into `environments/m5_probe_gpu.lock.txt`:

| check | command | expected |
|---|---|---|
| GPU present | `nvidia-smi` | non-empty |
| GPU model | `nvidia-smi --query-gpu=name --format=csv,noheader` | record |
| driver version | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | record |
| CUDA runtime | `python -c "import torch; print(torch.version.cuda)"` | non-null |
| torch CUDA available | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| torch version | `python -c "import torch; print(torch.__version__)"` | record; must be a CUDA build |
| torchvision version | `python -c "import torchvision; print(torchvision.__version__)"` | record |
| cuDNN | `python -c "import torch; print(torch.backends.cudnn.version())"` | non-null |
| OpenCV | `python -c "import cv2; print(cv2.__version__)"` | record (the high-pass uses `cv2.GaussianBlur`) |
| AMP FP16 smoke | one `autocast(device_type="cuda", dtype=torch.float16)` forward + `GradScaler("cuda")` step on random tensors | no error, finite loss |
| deterministic smoke | `torch.use_deterministic_algorithms(True)` then one ResNet-18 forward/backward on random tensors | no "no deterministic implementation" error |
| free disk | `df -h /home/sparc/workdir/longnm` | ≥ 10 GiB free (≈1.7 GB faces + ≈0.2 GB checkpoints/logs + headroom) |
| RAM | `free -g` | record |
| repo HEAD | `git rev-parse HEAD` | equals the pushed M5 commit |
| frozen config sha | `sha256sum configs/frozen/artifact_probe.yaml` | `e263b370…` |
| split manifest sha | `sha256sum manifests/split_v1.parquet` | `fb9aeb36…` |
| ResNet-18 weights | hash the torchvision cache file | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| faces present | count + M2 hash check as above | 20,615 and all digests match |

**If the AMP FP16 smoke or the deterministic smoke fails, STOP.** Do not fall back to CPU,
bfloat16 or float32, and do not disable the deterministic guard — both are explicit owner
prohibitions (E-M5-01, determinism contract).

## 4. Step 4 — authoritative run (only after every check above passes)

```
python -m gpatbench.cli train-probe --config configs/frozen/artifact_probe.yaml --dry-run   # gate
python -m gpatbench.cli train-probe --config configs/frozen/artifact_probe.yaml             # train
```

The trainer re-verifies the config hash, the split hash, the class counts, the class weights, the
ResNet weight hash and CUDA availability before it builds anything, and refuses on any mismatch.

## 5. What this plan does not do

It does not transfer data, does not connect to `sparc5090`, does not state any measured remote fact,
and does not start M5. Nothing in it may be reported as an environment verification — it is the
list of verifications still to be performed.
