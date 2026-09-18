# Planned Environment Requirements (M0 — planning only, nothing installed)

Status legend: AVAILABLE / MISSING / VERSION_UNVERIFIED / NOT_REQUIRED_AT_M0.
Versions are intentionally **not pinned** here: pins and lock hashes must come from a real,
created environment (`pip freeze` / `conda list --explicit` + SHA256), never from guesses.

## Environments expected by the spec

| Env ID | Purpose | Spec basis | Host | Status |
|---|---|---|---|---|
| env-core | inventory, preprocessing, split, pairs, probe, GPAT, downstream, stats, plots (PyTorch) | §4, §9–§16, §24 | GPU (training) + laptop (light CPU tasks) | NOT CREATED |
| env-stdn | STDN official stack (TensorFlow); isolated env/container required if incompatible | §8.3, §27 | GPU | NOT CREATED; TF version to be read from official repo at pinned commit (M6) |
| env-dsdg | FaceX-Zoo DSDG official stack | §8.6 | GPU | NOT CREATED (M6) |
| env-difffas | DiffFAS official stack | §8.7 | GPU | NOT CREATED (M6) |
| env-fasaug | FAS-Aug official repo deps | §8.1 | GPU/CPU | NOT CREATED (M6) |

## env-core named dependencies (from spec text)

| Dependency | Spec § | Laptop status | GPU status |
|---|---|---|---|
| torch + CUDA (AMP fp16, GradScaler) | 10.5, 14 | MISSING | UNKNOWN (GPU audit PENDING) |
| torchvision (ResNet-18 IMAGENET1K_V1) | 9.2, 12.1, 14 | MISSING | UNKNOWN |
| ptwt (Haar, level 1, reflect) | 4, 23.2 | MISSING | UNKNOWN |
| onnxruntime (SCRFD ONNX) | 4 | MISSING | UNKNOWN |
| FaceXFormer code + weights | 4 | code/weights candidate in local model_cache (unverified) | UNKNOWN |
| AdaFace IR-50 code + weights | 4 | code/weights candidate in local model_cache (unverified) | UNKNOWN |
| transformers (facebook/dinov3-vits16-pretrain-lvd1689m) | 14.2 | MISSING | UNKNOWN |
| opencv (INTER_AREA / INTER_CUBIC, decode) | 3.4, 4 | AVAILABLE 4.6.0 (apt) | UNKNOWN |
| numpy | — | AVAILABLE 1.26.4 (apt) | UNKNOWN |
| pandas + pyarrow (parquet manifests) | 3.5, 6, 22 | MISSING | UNKNOWN |
| scikit-learn (LogisticRegression, GroupKFold) | 12.2 | MISSING | UNKNOWN |
| scipy (statistics) | 18 | MISSING | UNKNOWN |
| lpips (LPIPS-Alex) | 13 | MISSING | UNKNOWN |
| KID implementation | 13 | not named in spec (Q-10) | — |
| UMAP | 20 fig13 | not named in spec | — |
| PyYAML | configs | AVAILABLE 6.0.1 (apt) | UNKNOWN |

## Lock policy (to apply when environments are created)

1. Create env; capture `python -m pip freeze --all` (and `conda list --explicit` if conda).
2. Save as `environments/<env_id>_<host>.lock.txt`; record SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`.
3. Record CUDA/driver/cuDNN/GPU via `nvidia-smi` and `torch.version.cuda` / `torch.backends.cudnn.version()`.
4. Every run's metadata JSON (§22.3) stores `environment_lock_sha256`.
