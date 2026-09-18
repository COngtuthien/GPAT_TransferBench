# M2A Determinism Report (2026-09-19)

- **Contract:** same host (laptop, CPU), same M2 environment (`environments/m2_preprocess_laptop.lock.txt`), same code, same manifest, same models, 4 torch intra-op threads, 1 ORT thread, `torch.use_deterministic_algorithms(True)`, `inference_mode`, `eval()`.
- **Method:** two separate processes (run3, run4) start from clean output directories; every produced file is compared byte-wise (`tools/m2a_smoke.py compare`).

| Artifact | Files | Byte-identical |
|---|---|---|
| Frame PNGs (MSU/SiW) | 16 | 16 |
| Canonical face PNGs | 24 | 24 |
| FaceXFormer arrays (.npy: parsing_logits, parsing_mask, landmarks ×3, pose) | 144 | 144 |
| AdaFace embeddings (.npy) | 24 | 24 |
| **Total** | **208** | **208** |
| Crop metadata / detections / results table (excluding timing) | – | identical |

- **Max float difference:** 0 (exact byte identity; `max_abs_diff` is empty).
- **Final status: PASS** (same host/device/environment).
- **Cross-device:** not tested and not claimed. Full M2B on another host or device must repeat this smoke and lock (rule 35). Laptop hashes are not evidence of remote determinism.
