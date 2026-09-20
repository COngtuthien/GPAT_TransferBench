# M5 — ArtifactProbeNet Environment and Backbone Analysis (pre-flight)

> **RESOLVED 2026-09-20.** Every question raised below was answered by the owner and frozen in `configs/frozen/artifact_probe.yaml` (sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`). The decision record is `M5_ARTIFACT_PROBE_OWNER_DECISIONS.md` and the contract summary is `M5_ARTIFACT_PROBE_FROZEN_CONTRACT.md`. **The text below is preserved unchanged as the analysis that motivated those decisions.** M5 is still NOT_STARTED: nothing has been trained.

**Date:** 2026-09-20 · Read-only. No model was trained and no checkpoint was written.

## 1. Measured environment

| item | value |
|---|---|
| platform | `Linux-7.0.0-31-generic-x86_64-with-glibc2.39` |
| python | 3.12.3 |
| torch | **2.14.0+cpu** |
| torchvision | **0.29.0+cpu** |
| `torch.cuda.is_available()` | **False** |
| `torch.version.cuda` | `None` |
| `torch.backends.cudnn.version()` | `None` |
| `nvidia-smi` | not installed |
| CPU threads visible to torch | 8 |
| scikit-learn | **not installed** |

## 2. E-M5-01 — the environment blocks M5 *execution*, not the contract freeze

Spec §12.1 requires **AMP**, which in this stack is a CUDA feature. The only verified environment is
CPU-only, with no CUDA runtime and no NVIDIA driver present. **ArtifactProbeNet cannot be trained as
specified on this machine.**

This is recorded as `E-M5-01`, and it is deliberately separated from the nine contract decisions:

- it does **not** block freezing `configs/frozen/artifact_probe.yaml` once the owner resolves
  D-M5-01…D-M5-08 and Q-07;
- it **does** block running M5.

Related pre-existing record: the M0 environment audit already notes *"GPU environment unknown (SSH
needs interactive password)"* and `models/registry.yaml` still carries `status: NOT_DOWNLOADED` for
the backbone. Before M5 runs, the owner must either provide the GPU machine (and an environment lock
captured there) or approve a CPU fallback, which would mean disabling AMP — itself a deviation from
§12.1 that would need recording.

`scikit-learn` is absent, which interacts with D-M5-05: if the owner chooses
`sklearn.metrics.f1_score(average="macro")` as the frozen macro-F1, it must be added to the M5
environment lock; otherwise a pinned manual implementation is needed. Either way the definition must
be frozen, not left to whatever happens to be installed.

## 3. Backbone contract (D-M5-04 — partially open)

| item | status |
|---|---|
| model | `torchvision.models.resnet18` — **fixed by spec** |
| weight enum | `ResNet18_Weights.IMAGENET1K_V1` — **fixed by spec**; no other enum may be substituted |
| penultimate feature dim | **512**, verified by constructing the model (`model.fc.in_features`) — matches the spec's 512-D embedding |
| classifier replacement | `Linear(512, K)`; K comes from D-M5-01 (6 or 7) |
| **fine-tune scope** | **OPEN** — full fine-tune / partial freeze / classifier-only are all consistent with "Backbone: torchvision ResNet-18 IMAGENET1K_V1" |
| parameters (default 1000-class head) | 11,689,512 |

On the fine-tune scope, the only in-spec signal is weak: §12.1 gives a **single** learning rate
(`1e-4`), whereas §14 and `configs/frozen/downstream_resnet18.yaml` give **separate** backbone
(`1e-4`) and head (`5e-4`) rates. A single rate is mildly suggestive of one uniformly optimised
parameter group, i.e. full fine-tuning — but it is not decisive and must not be treated as such.

## 4. Pretrained weight provenance (§10) — RESOLVED

| item | value |
|---|---|
| official enum | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` |
| official URL | `https://download.pytorch.org/models/resnet18-f37072fd.pth` |
| local cache path | `/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/backbones/torchvision/resnet18-f37072fd.pth` (outside Git) |
| size | 46,830,571 B |
| **SHA256** | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| torchvision | 0.29.0 |
| downloaded | 2026-09-20 |

**Self-verifying:** torchvision names its weight files with the first 8 hex characters of the file's
SHA256. The filename is `resnet18-f37072fd.pth` and the measured digest begins `f37072fd`, so the
downloaded bytes are confirmed to be exactly the official `IMAGENET1K_V1` artifact — the same style
of byte-hash confirmation used for SCRFD in M2A.

The weight file is **not committed** (`*.pth` is git-ignored); only its path and hash are recorded,
in `models/registry.yaml` and here. No alternative ResNet-18 enum may be substituted.

## 5. Determinism intent (D-M5-08 adjacent)

Seed `42` only, per §12.1. Seeds must be set for: Python `random`, NumPy, torch CPU, torch CUDA
(all devices), the DataLoader generator, and each worker via `worker_init_fn`. The following must be
pinned explicitly rather than left at library defaults: `cudnn.benchmark`, `cudnn.deterministic`,
`torch.use_deterministic_algorithms`, TF32 for matmul and cuDNN, and `float32_matmul_precision`.

Two levels of reproducibility are distinguished, and the stronger one is **not promised**:

- **scientific reproducibility** — same data, same frozen contract, same seed ⇒ the same selection
  decision and comparable metrics. This is the level M5 targets.
- **byte-identical checkpoint reproducibility** — will **not** be claimed. AMP and GPU reduction
  order are not guaranteed bit-reproducible across machines or driver versions, and nothing has been
  demonstrated here because nothing has been trained. If the owner wants this claim, it must be
  demonstrated on the actual training machine first.

## Resolution (2026-09-20)

**D-M5-04 resolved to FULL FINE-TUNE** (11,180,103 parameters, all trainable, `fc = Linear(512, 7)`). **E-M5-01 resolved to GPU_REQUIRED**: authoritative training runs on `sparc5090` at `/home/sparc/workdir/longnm/GPAT_TransferBench`; CPU fallback and disabling AMP are both forbidden, and the trainer refuses an authoritative run without CUDA (verified). The remote environment must be measured before training — see `M5_GPU_EXECUTION_PLAN.md`; no remote fact is assumed. The ResNet-18 weight hash was re-verified: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
