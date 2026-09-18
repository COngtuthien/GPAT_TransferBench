# M2A — Preprocessing Contract (PROPOSED; not frozen)

Machine-readable form: `configs/proposed/preprocess_v1.proposed.yaml` (deliberately **not** under
`configs/frozen/`, because Q-02, Q-03, Q-18, Q-19, Q-22 and Q-23 are open). Code: `gpatbench/preprocess/`.

## 1. Routes (dataset_protocol_policy_v1)

| Dataset | Route |
|---|---|
| CASIA-FASD | M1-selected 112×112 PNG → decode (cv2 BGR) → INTER_CUBIC 256×256 → RGB uint8 canonical. **No SCRFD, no extra crop.** `scrfd_applied=false`, `scrfd_status=N/A` (DEV-011) |
| MSU-MFSD | M1 frame index → lossless frame PNG → SCRFD (320, thr 0.50) → largest face → 1.25× square crop → 256 RGB uint8 |
| SiW-Mv2 | as MSU (the SiW deviations DEV-012/013 affect M3/M4 only) |

After the canonical face, FaceXFormer and then AdaFace run for all datasets.

## 2. Frame extraction (MSU/SiW)

- **Decode:** `cv2.VideoCapture(path, CAP_FFMPEG)` (opencv-python-headless 5.0.0.93, avcodec 62.28.101). One sequential `read()` per index from 0 up to the M1 `frame_index`; the loop counter is the identity (DEV-006). `CAP_PROP_POS_FRAMES` is never used.
- **Failure:** a failed read at the requested index raises `FrameReadError`. There is no neighbour fallback.
- **Output:** a lossless PNG of the decoded BGR frame (compression 3).
- **Provenance per frame:** dataset, video_id, frame_index, source raw sha256, decoder versions, frame PNG path, frame PNG sha256.

## 3. SCRFD (official InsightFace math @ `1480e705`, reimplemented)

| Item | Value |
|---|---|
| Model | **OPEN (Q-02)**: candidate A `scrfd_10g_bnkps.onnx` or B `det_2.5g.onnx` |
| Detector input | 320×320 (spec) |
| Channel order | source frame BGR (cv2); `blobFromImage(..., swapRB=True)` gives RGB into the network |
| Resize / letterbox | keep aspect ratio; `cv2.resize` default (INTER_LINEAR); image at top-left; zero padding |
| Normalisation | (x − 127.5) / 128, float32, NCHW |
| Decode | strides 8/16/32, 2 anchors, distance→bbox/kps × stride |
| Threshold | score ≥ 0.50 (spec) |
| NMS | IoU 0.4, official default, stable sort by score |
| Coordinates | bbox / det_scale (det_scale = new_height / frame_height) gives original-frame pixels [x1, y1, x2, y2] |
| Largest face | max area (x2−x1)(y2−y1); tie → score ↓ → x1 ↑ → y1 ↑ → index (IMPLEMENTATION_DETAIL) |
| No face | `SCRFD_NO_FACE`; the sample stops. No whole frame, centre crop, previous bbox, other detector or lower threshold |

## 4. Crop and canonical face

- **Box:** side = 1.25·max(w, h); centre = bbox centre; side_px = round(side); x0 = round(cx − side_px/2), same for y. The box is half-open.
- **Border:** **OPEN (Q-22).** The PROVISIONAL smoke rule is a literal intersection with the image (the crop may become non-square).
- **Resize:** INTER_AREA if the larger crop side ≥ 256, else INTER_CUBIC. The larger-side rule is PROVISIONAL, as part of Q-22.
- **Canonical:** 256×256×3 uint8, RGB channel order, sRGB as decoded. No colour management; files carry no ICC profile.
- **PNG:** OpenCV, compression 3, lossless, deterministic bytes.

## 5. FaceXFormer adapter (VERIFIED model; official transform)

- **Input:** canonical RGB uint8 → PIL image → `Resize((224, 224), BICUBIC)` → `ToTensor` → `Normalize(ImageNet)` → [1, 3, 224, 224] float32. The canonical image itself is never modified.
- **Forward:** 3 forwards with batch 1 (task 0/1/2).

| Output | Shape / dtype | Semantics |
|---|---|---|
| `parsing_logits` | [11, 224, 224] float32 | raw seg logits (class names OPEN, Q-21) |
| `parsing_mask` | [224, 224] uint8 | argmax of the logits (official) |
| `landmarks_norm` | [68, 2] float32 | raw head output in [−1, 1], (x, y) order, 68-point layout |
| `landmarks_px224` | [68, 2] float32 | official `denorm_points` (align_corners=False) on the 224 grid |
| `landmarks_px256` | [68, 2] float32 | linear pixel-centre map to the canonical 256 grid: (p+0.5)·256/224 − 0.5 |
| `pose_pitch_yaw_roll_rad` | [3] float32 | **order (pitch, yaw, roll), radians** (official `inference.py` multiplies by 180/π to print degrees). No unit conversion is stored; yaw = element 1 |

## 6. AdaFace adapter (model identity VERIFIED; contract OPEN)

- **Load:** HF `minchul/cvlface_adaface_ir50_webface4m` `model.pt`, with the `net.` prefix stripped, into the official `net.py` `ir_50` using `strict=True`.
- **Input (PROVISIONAL):**
  - geometry: canonical 256 → `cv2.resize` 112×112 INTER_AREA (Q-19);
  - channel order: **RGB**, per the checkpoint's `model.yaml` (Q-18);
  - normalisation: ((x/255) − 0.5)/0.5, mean = std = 0.5 in both official lines;
  - tensor: [1, 3, 112, 112] float32.
- **Output:** `net.py` returns (feature/‖feature‖, ‖feature‖). The adapter re-normalises explicitly and stores float32 [512]. Smoke L2 norms were in [0.99999994, 1.00000012] (tolerance 1e-5).

## 7. Cache schema (design only; not populated)

- **Unit:** one entry per `sample_id`. The key is `sample_id`, and every row carries:
  - `canonical_face_sha256`;
  - model weight sha256 and code fingerprint (commit + file hashes);
  - `preprocess_config_sha256`;
  - adapter version;
  - environment lock sha256.
- **Geometry arrays:** `parsing_logits` (dtype and resolution OPEN, Q-23), `parsing_mask` uint8 [224,224], `landmarks_norm`/`px224`/`px256` float32 [68,2], `pose_pitch_yaw_roll_rad` float32 [3].
- **Identity arrays:** `embedding` float32 [512] (L2 = 1 ± 1e-5), `raw_norm` float32.
- **Serialization (proposed):**
  - per dataset and field, fixed-order shards of raw `.npy` arrays (deterministic header, no pickle, `allow_pickle=False`);
  - a Parquet index maps `sample_id` → shard/row and holds the sha256 of each row's bytes, so the cache can be validated without rerunning models.
  - `.npz` is avoided because zip timestamps break byte determinism.
- **Location (future M2B):** `cache/geometry/`, `cache/identity/` (git-ignored); only the index plus hashes are committed.
- **Storage estimate (Q-23):** logits float32 ≈ 45.6 GB; mask ≈ 1.0 GB; landmarks, pose and embeddings < 0.1 GB.
