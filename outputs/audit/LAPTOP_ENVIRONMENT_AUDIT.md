# Laptop Environment Audit — M0 (read-only)

- Captured: 2026-09-18 (UTC) by `tools/capture_laptop_env.sh` → raw dump `environments/laptop_environment_initial.txt` (SHA256 in `ARTIFACT_INDEX.csv`).
- Nothing was installed, upgraded or configured during this audit.
- Role of this machine (per owner): development, orchestration, Git source of truth, frozen configs/manifests, audit/provenance, result collection. Heavy training is planned on the GPU server.

## System

| Item | Value |
|---|---|
| Hostname | cong-ThinkBook-16-G7-AHP |
| User | cong |
| OS | Ubuntu 24.04.3 LTS (noble) |
| Kernel | Linux 7.0.0-31-generic x86_64 |
| CPU | AMD Ryzen 7 8845H w/ Radeon 780M Graphics — 8 cores / 16 threads |
| RAM | 13 GiB total (~7.2 GiB available at capture), swap 4 GiB |
| Disk `/home` | 62 G total, 40 G free (33% used) |
| Disk `/media/cong/Data` (datasets, model cache) | 363 G total, 67 G free (82% used) |
| Display adapter | AMD Phoenix3 iGPU only — **no NVIDIA GPU** |
| nvidia-smi / driver | NOT FOUND |
| CUDA (nvcc) | NOT FOUND |
| cuDNN | NOT FOUND (ldconfig) |
| Docker | docker-ce 29.5.2 installed (`/usr/bin/docker`); podman not found |

## Tooling

| Tool | Status | Version |
|---|---|---|
| Python | AVAILABLE | 3.12.3 (`/usr/bin/python3`, system) |
| python3.12-venv | AVAILABLE (dpkg) | 3.12.3-1ubuntu0.17 — venv creation possible later |
| pip | MISSING | `python3 -m pip` → "No module named pip" (only `python3-pip-whl` wheel present) |
| conda / mamba / micromamba / uv | MISSING | — |
| Git | AVAILABLE | 2.43.0 |
| Git global identity | **NOT CONFIGURED** | `~/.gitconfig` and `/etc/gitconfig` do not exist → commits need owner-provided identity |
| SSH client | AVAILABLE | OpenSSH_9.6p1, OpenSSL 3.0.13 |
| rsync | AVAILABLE | 3.2.7 |
| ffmpeg | MISSING | — |
| pytest | MISSING | M0 tests use stdlib `unittest` |

## Python dependencies (system interpreter)

Status legend: AVAILABLE / MISSING / VERSION_UNVERIFIED / NOT_REQUIRED_AT_M0.
"Required at" = earliest milestone the spec needs it (planning only, from spec sections).

| Package | Status on laptop | Version | Required at | Spec basis |
|---|---|---|---|---|
| numpy | AVAILABLE | 1.26.4 (apt) | M1 | general |
| opencv (cv2) | AVAILABLE | 4.6.0 (apt) | M1/M2 | decode, INTER_AREA/INTER_CUBIC §4 |
| PyYAML | AVAILABLE | 6.0.1 (apt) | M0 | configs |
| Pillow | AVAILABLE | 10.2.0 (apt) | M2 | images |
| pandas | MISSING — NOT_REQUIRED_AT_M0 | — | M1 | inventory tables |
| pyarrow | MISSING — NOT_REQUIRED_AT_M0 | — | M1 | `*.parquet` manifests |
| torch | MISSING — NOT_REQUIRED_AT_M0 | — | M2+ | all models |
| torchvision | MISSING — NOT_REQUIRED_AT_M0 | — | M5+ | ResNet-18 IMAGENET1K_V1 |
| onnxruntime | MISSING — NOT_REQUIRED_AT_M0 | — | M2 | SCRFD ONNX |
| scipy | MISSING — NOT_REQUIRED_AT_M0 | — | M9/M13 | stats |
| scikit-learn | MISSING — NOT_REQUIRED_AT_M0 | — | M9 | fingerprint LR, GroupKFold |
| ptwt | MISSING — NOT_REQUIRED_AT_M0 | — | M7 | Haar DWT §4 |
| PyWavelets | MISSING — NOT_REQUIRED_AT_M0 | — | M7 (ptwt dep) | — |
| transformers | MISSING — NOT_REQUIRED_AT_M0 | — | M11 | DINOv3 HF model |
| timm | MISSING — NOT_REQUIRED_AT_M0 | — | not named in spec | — |
| lpips | MISSING — NOT_REQUIRED_AT_M0 | — | M9 | LPIPS-Alex |
| insightface | MISSING — NOT_REQUIRED_AT_M0 | — | M2 (maybe) | SCRFD model family |
| pytest | MISSING — NOT_REQUIRED_AT_M0 | — | M1+ (optional) | tests |

No version hash is claimed for any MISSING package. The laptop is not expected to run
training; whether the laptop needs a light CPU environment (pandas/pyarrow for M1
inventory) is a decision for M1 — see `environments/PLANNED_ENVIRONMENT_REQUIREMENTS.md`.
