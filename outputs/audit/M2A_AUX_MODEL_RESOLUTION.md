# M2A — Auxiliary Model Resolution (2026-09-19)

> **Superseded in part by `M2A_OWNER_DECISIONS.md` (same day, owner-decision pass).** Everything below is
> the investigation as it stood before the owner decided, and is kept unchanged as history. Q-02, Q-03,
> Q-18, Q-19, Q-22 and Q-23 are now resolved; the SCRFD "hash match to an official release: NOT VERIFIED"
> line below has since been closed — the local `scrfd_10g_bnkps.onnx` was confirmed **byte-identical** to
> the member of the official `antelopev2.zip`. See `M2A_OFFICIAL_VERIFICATION.json`.

Evidence table: `M2A_MODEL_PROVENANCE.csv`. Official sources were inspected with lightweight operations only:
`git ls-remote`, shallow fetch of pinned commits for small repos, raw files, and the Hugging Face model API
(LFS SHA-256 without downloading weights). No model weight was downloaded; nothing was selected by
benchmark performance.

| Official source | Pinned ref |
|---|---|
| deepinsight/insightface | `1480e705287bc5d59f923b46c260ec6e3e4150f6` (HEAD) |
| Kartik-3004/facexformer | `10fe8291f8a64e2ca1daf938e3e0007bd860303b` (HEAD) |
| HF kartiknarayan/facexformer | `fd12148d0b1949bb98602544d4b257f42eef774d` (main) |
| mk-minchul/AdaFace | `c60eaa786a42c03444f3df7096dbaf9d57ae010d` (HEAD) |
| HF minchul/cvlface_adaface_ir50_webface4m | `60a65befbcf7e19284c4f3ac730f56867ed29594` (main) |
| mk-minchul/CVLface | `308142aa50adf2e187711354f7524635d3414f1e` (HEAD) |

## 1. SCRFD — Q-02: **OWNER_DECISION_REQUIRED**

| | Candidate A | Candidate B |
|---|---|---|
| Path | `model_cache/face_detectors/scrfd_10g_bnkps.onnx` | `model_cache/face_detectors/det_2.5g.onnx` |
| SHA-256 (recomputed) | `5838f7fe…5b91` | `041f73f4…0af9` |
| Size | 16,923,827 B | 3,292,009 B |
| ONNX | IR 6, opset 11, producer pytorch 1.6, no metadata | IR 6, opset 11, producer pytorch 1.8, no metadata |
| Input | `input.1` [1,3,?,?] (dynamic) | `input.1` [1,3,?,?] (dynamic) |
| Outputs | 9 = 3 strides × (score, bbox, 5-kps) | 9 = 3 strides × (score, bbox, 5-kps) |
| Parameters | 4,225,835 | 817,955 |
| Runs at 320×320 | yes (anchor rows 3200/800/200) | yes (same) |
| Official identity (INFERENCE from official docs) | **SCRFD_10G_KPS**: the official table lists 4.23M params and 5 keypoints. Filename used by the official `antelopev2` pack | **SCRFD_2.5G_KPS**: 0.82M params in the official table. `det_2.5g.onnx` is the detector of the official `buffalo_m` pack |
| Hash match to an official release | NOT VERIFIED (needs antelopev2.zip, 407MB) | NOT VERIFIED (needs buffalo_m.zip, 313MB) |

- **Why OWNER_DECISION_REQUIRED:**
  - Spec §4 says only "SCRFD ONNX, input 320, threshold 0.50".
  - Both files are structurally consistent with two different official SCRFD KPS releases.
  - Neither can be hash-verified without large downloads, which were not performed per policy.
- **Recommendation (clearly labelled; owner decides):** candidate A (SCRFD_10G_KPS). Reasons:
  - it is the highest-capacity official KPS detector available locally;
  - its official WIDER Face numbers are higher (published model-card context, *not* benchmark data).
- The M2A smoke used A **provisionally** only to exercise the pipeline.

## 2. FaceXFormer: **VERIFIED**

- **Weights:** HF `kartiknarayan/facexformer` `ckpts/model.pt` LFS SHA-256 `327a7558…286d` equals the local file. The local HF download metadata (revision `fd12148d…`) equals the repo's current `main`. The official GitHub README directs users to exactly this repository and file.
- **Code:** the local `code/facexformer/network/` (not a git checkout) has 4 files that are byte-identical to the official repo at `10fe8291`. The adapter enforces these hashes at load time.
- **Loading:** `FaceXFormer()` calls `swin_b(weights='IMAGENET1K_V1')`. The adapter builds it with `weights=None` (no download) and then loads `state_dict_backbone` with `strict=True`: every parameter and buffer comes from the checkpoint.
- **Contract:** see `M2A_PREPROCESS_CONTRACT.md` §5.

## 3. AdaFace — Q-03: identity **RESOLVED_AUTHORITATIVE** for the local file; selection **OWNER_DECISION_REQUIRED**

- **Identity (FACT):** the local `model.pt` equals HF `minchul/cvlface_adaface_ir50_webface4m` `pretrained_model/model.pt`, with LFS SHA-256 and revision both matching. So the file is the **CVLFace release of AdaFace IR-50 trained on WebFace4M**, published by the AdaFace first author (the CVLFace README lists it with its own benchmark numbers).
- **Architecture check (FACT):** the state dict (467 tensors, prefix `net.`) loads with `strict=True` into the official AdaFace `net.py` `build_model('ir_50')` (mk-minchul/AdaFace @ `c60eaa78`; the local `adaface_net.py` is byte-identical). Output: 512-D, L2-normalised by `net.py`.
- **Why an owner decision is required:**
  1. Spec §4 says "AdaFace IR-50" without a training set. Three official IR-50 variants exist (CASIA-WebFace, WebFace4M, MS1MV2), in two release lines:
     - the CVLFace HF checkpoints (RGB input);
     - the original AdaFace GitHub checkpoints (gdrive `.ckpt`, BGR input).
  2. The spec cites the original repo (R13) and says "follow the model's official **BGR**/input normalization". The local file's own official contract is **RGB** (Q-18).
- **Recommendation (labelled):** either keep the local CVLFace WebFace4M checkpoint and approve its official RGB contract as a documented deviation from the spec's "BGR" wording, or obtain the original-repo IR-50 checkpoint to follow the BGR contract literally (download required).

## 4. New open questions (genuine gaps)

| ID | Question | Evidence | Options | Blocks M2B |
|---|---|---|---|---|
| Q-18 | AdaFace colour order | Spec §4: "official BGR". Local checkpoint's official `model.yaml`/`config.json`/README: `color_space: RGB`; CVLFace `IR_50` has no internal flip. Smoke: cos(embedding with RGB input, with BGR input) = 0.81–0.97, so the choice materially changes embeddings | RGB (checkpoint contract, deviation from spec wording) / BGR (spec wording; with the original-repo checkpoint) | **YES** |
| Q-19 | AdaFace geometric input | Both official pipelines require a **5-point-aligned 112×112** face (original: MTCNN `align.get_aligned_face`; CVLFace: its alignment app). The GPAT canonical face is an unaligned 1.25× SCRFD square crop at 256. The spec (§4, App. A `cache_…(faces)`) gives no alignment step. The ID metric must also work on synthetic 256 faces | (a) resize 256→112 (smoke PROVISIONAL); (b) similarity alignment from SCRFD 5 keypoints on the canonical face (a new transform); (c) official aligner on the canonical face (another model) | **YES** |
| Q-20 | FaceXFormer input framing | The official demo crops an MTCNN box with a 50% margin before resizing to 224. GPAT feeds the canonical face (SCRFD 1.25× square; CASIA: packager crop), as spec App. A defines caches on `faces` | follow spec App. A (implemented) / add official framing | NO (spec-defined; owner-visible) |
| Q-21 | FaceXFormer parsing class semantics | 11 classes; the official repo documents no class names (only a colour table; index 0 is black) | obtain the official label map; do not assume | NO for M2B (cache stores indices/logits); YES before M9 Dice "non-background" and §21 region crops |
| Q-22 | Crop at image borders | Spec §4: "Square **padded** face" and "clamp to image; no random padding". Clamping makes the crop non-square, so resizing to 256 distorts the aspect ratio. Smoke: **3/8 SiW** crops clamped (0/8 MSU) | (a) literal intersection (smoke PROVISIONAL); (b) constant zero padding to keep the square; (c) shift the square inside the image | **YES** |
| Q-23 | Geometry cache size / dtype of parsing logits | Full-resolution logits float32 11×224×224 = 2.21 MB per sample → **≈45.6 GB** for 20,640 samples (float16 ≈22.8 GB, lossy). Data volume free: 67 GB | float32 full / float16 / store only mask + reduced logits (each changes fidelity) | **YES** (schema freeze) |

## 5. Implementation details (not scientific choices; recorded)

- **SCRFD:**
  - NMS IoU 0.4 is the official InsightFace default (spec silent);
  - letterbox uses cv2.resize default interpolation, top-left placement, zero pad (official);
  - ORT is single-threaded and sequential for determinism.
- **Largest face:** area (x2−x1)(y2−y1) in frame pixels. Tie → higher score → smaller x1 → smaller y1 → lower index.
- **Crop integers:** side_px = round(1.25·max(w,h)); x0 = round(cx − side_px/2) (half-to-even); half-open box.
- **Canonical PNG:** OpenCV PNG, compression level 3, lossless; RGB stored.
- **Torch:** `use_deterministic_algorithms(True)`, 4 intra-op threads, `inference_mode`, `eval()`.
- **FaceXFormer:** one forward per task with batch 1, following the official demo. All heads are computed each time and `tasks` only filters rows.

---

## 6. Owner resolution pass (2026-09-19, additive)

| Q | Outcome | Evidence |
|---|---|---|
| Q-02 | SCRFD_10G_KPS selected; **OFFICIAL_BYTE_HASH_CONFIRMED** against `antelopev2.zip` (sha256 `8e182f14…`) | `M2A_OFFICIAL_VERIFICATION.json` |
| Q-03 | Original repo R50/WebFace4M `adaface_ir50_webface4m.ckpt` sha256 `52cca7c6…`; strict load OK | same |
| Q-18 | BGR (official `inference.py` contract reproduced exactly) | `M2A_OWNER_DECISIONS.md` |
| Q-19 | Canonical face → 112 INTER_AREA; no MTCNN/alignment (DEV-014) | `M2A_OWNER_DECISIONS.md` |
| Q-22 | Requested square preserved with zero padding (DEV-016 interpretation) | `M2A_BORDER_CASES.csv` |
| Q-23 | Full float32 logits + uint8 mask, lossless `npy1+shuffle4+zstd` level 10 | `M2A_COMPRESSION_PILOT.md` |
| Q-20 | Closed as DEV-015 (canonical face is the FaceXFormer source image) | `deviation_report.md` |
| Q-21 | Still OPEN; not required for M2B | `deviation_report.md` |

New authoritative finding: the CVLFace export and the original AdaFace release hold the **same trained
weights**; 466/467 tensors are bit-identical and `input_layer.0.weight` is an exact channel-axis reversal.
The §3 note above ("two release lines, RGB vs BGR") is therefore literally about the *input convention*,
not about two differently trained models.
