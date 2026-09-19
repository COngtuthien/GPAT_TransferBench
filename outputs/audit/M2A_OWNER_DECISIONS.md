# M2A — Owner Decision Resolution (2026-09-19)

Additive pass on top of `0688103a` ("M2A: resolve auxiliary models and validate preprocessing smoke").
Nothing from the earlier investigation was rewritten: every candidate, hash and status recorded then is
still present in `models/registry.yaml`, `M2A_AUX_MODEL_RESOLUTION.md` and `M2A_MODEL_PROVENANCE.csv`.
This file records the owner's resolutions, what was verified for each, and what remains open.

cwd for this pass: `/home/cong/GPAT_TransferBench`.

| Question | Owner decision | Final status | Independently verified? |
|---|---|---|---|
| Q-02 SCRFD variant | candidate A `scrfd_10g_bnkps.onnx` (SCRFD_10G_KPS) | RESOLVED_BY_OWNER_SELECTION | **Yes — official byte hash confirmed** |
| Q-03 AdaFace checkpoint | original repo R50 / WebFace4M | RESOLVED_BY_OWNER_SELECTION | Official link + strict load; publisher hash not published |
| Q-18 AdaFace colour order | BGR | RESOLVED_BY_OWNER | Yes — official `inference.py` contract reproduced exactly |
| Q-19 AdaFace geometry | canonical face → 112 INTER_AREA, no extra alignment | RESOLVED_BY_OWNER_CANONICAL_FACE_ADAPTER | Recorded as DEV-014 (adapter deviation) |
| Q-22 crop border | requested square preserved, zero padding outside the image | RESOLVED_BY_OWNER | Yes — unit tests + 3 real border samples |
| Q-23 parsing logits | full float32 11×224×224 + uint8 argmax mask, lossless storage | RESOLVED_BY_OWNER_FLOAT32_FULL_LOGITS | Yes — lossless pilot, exact reconstruction |
| Q-20 FaceXFormer framing | canonical face (spec App. A), no MTCNN margin recrop | CLOSED as DEV-015 | — (non-blocking) |
| Q-21 parsing class names | not needed for M2B | OPEN (required before M9) | — (non-blocking) |

---

## Q-02 — SCRFD variant

**Decision:** candidate A, `scrfd_10g_bnkps.onnx`, variant `SCRFD_10G_KPS`.
`det_2.5g.onnx` is NOT used for benchmark preprocessing and is retained in the registry as an
unselected candidate (`selection: NOT_SELECTED_FOR_FINAL_M2`).

**Provenance verification performed in this pass (this was the previous pass's gap):**

| Item | Value |
|---|---|
| Official package | `https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip` |
| Package size / sha256 | 360,662,982 B / `8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5` |
| Member | `antelopev2/scrfd_10g_bnkps.onnx`, 16,923,827 B |
| Member sha256 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| Local file sha256 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| Result | **OFFICIAL_BYTE_HASH_CONFIRMED** (byte-for-byte identical) |

Downloaded 2026-09-19 into `/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/_provenance_tmp/`
(outside Git; the package and the model binary are never committed). Only the member needed for the
check was extracted in memory; no other model pack was downloaded. The structural evidence from the
previous pass (4,225,835 parameters = the official 4.23M figure; 9 outputs = 3 strides × score/bbox/5-kps)
remains and now agrees with an exact byte identity, so the identification no longer rests on inference.

Machine-readable record: `M2A_OFFICIAL_VERIFICATION.json` → `scrfd`.

**Enforcement:** `gpatbench/preprocess/scrfd.py` defaults its expected hash to the selected weights and
refuses `det_2.5g.onnx` by hash with an explicit message, even if that hash is passed as the expectation.

## Q-02 — final detector contract (unchanged; no contradicting evidence found)

input 320×320 · aspect ratio preserved · official top-left letterbox · zero padding · official
`blobFromImage(1/128, mean 127.5, swapRB=True)` · score threshold 0.50 · NMS IoU 0.4 · coordinates
mapped back with `det_scale` · largest bbox by original-frame pixel area. Equal-area tie-break:
higher score → lower x1 → lower y1 → remaining bbox coordinates lexicographically. No benchmark
measurement was used to set or modify any of these rules.

---

## Q-03 / Q-18 / Q-19 — AdaFace

**Decision:** the ORIGINAL `mk-minchul/AdaFace` release **R50 / WebFace4M**, with **BGR** input, fed
from the frozen canonical 256 face resized to 112 with INTER_AREA and **no additional alignment**.
The CVLFace export is not the final identity model.

| Item | Value |
|---|---|
| Official source | `mk-minchul/AdaFace` README table row "R50 \| WebFace4M", repo @ `c60eaa786a42c03444f3df7096dbaf9d57ae010d` |
| Official link | `https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing` |
| File | `adaface_ir50_webface4m.ckpt` |
| Download date | 2026-09-19 |
| Size | 596,517,267 B |
| sha256 | `52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8` |
| Architecture | IR-50 (R50), official `net.py` `build_model('ir_50')` |
| Training dataset | WebFace4M |
| Code commit | `c60eaa786a42c03444f3df7096dbaf9d57ae010d` (local `adaface_net.py` byte-identical) |
| Load | `torch.load(...)["state_dict"]`, 467 keys with prefix `model.` stripped, `strict=True` → **All keys matched successfully**; 512-D output, L2 = 1 |
| Location | `/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/face_identity/adaface_original/` (outside Git) |

**Limit of the verification, stated honestly:** the official README publishes no checksum for this file,
so there is no publisher-side hash to compare against. What is verified is: the file came from the
link published in the official README at the pinned commit, its served filename is
`adaface_ir50_webface4m.ckpt`, and it loads strictly into the official IR-50 definition. Its own sha256
is recorded and is now mandatory at load time.

**New authoritative evidence about the two official releases.** Tensor-by-tensor comparison of the
original checkpoint against the CVLFace export (`43bd2d57…`):

- 466 of 467 tensors are **bit-identical**;
- the single difference, `input_layer.0.weight`, is an **exact reversal of the input convolution's
  channel axis** (`orig.flip(1) == cvlface`, max |Δ| = 0).

So the two are the *same trained model*, published for BGR (original) and RGB (CVLFace) input. The
owner's decision therefore restores literal compliance with the spec's "official BGR" wording without
changing the identity space at all — confirmed empirically in `M2A_CONTRACT_CHANGE_IMPACT.json`:
on the 21 canonical faces that the Q-22 change did not touch, cos(previous CVLFace+RGB embedding,
new original+BGR embedding) ∈ [0.99999994, 1.00000012].

⚠️ This must not be confused with the previous pass's diagnostic `cos(RGB-input, BGR-input) = 0.81–0.97`,
which fed **one** checkpoint the **wrong** channel order. That number measured the cost of a mistake,
not the difference between the two releases.

**Q-18 — final input contract** (frozen module constants, not parameters):

```
canonical 256×256 RGB uint8
  → cv2.resize 112×112, INTER_AREA
  → RGB→BGR channel reversal
  → ((pixel / 255.0) − 0.5) / 0.5          # arithmetic of official inference.py to_input
  → float32 tensor [1, 3, 112, 112]
  → IR-50 → L2-normalised 512-D embedding
```

A unit test reimplements the official `to_input` literally and requires `torch.equal`, and a second
test uses a synthetic R=255/G=128/B=0 image so that a lost channel reversal cannot survive a refactor.

**Q-19 — geometric adapter, and why this is a deviation (DEV-014).** The official AdaFace pipeline
aligns arbitrary photographs with MTCNN 5-point alignment before the 112×112 crop. This benchmark does
not: the spec (§4, App. A) defines the caches on the frozen canonical face, and GPAT must score
synthetic 256×256 images in exactly the same frame. Adding an aligner would mean (a) a second face
model inside the identity path, (b) a transform with no spec basis, and (c) a frame that generated
images could fail. The canonical face is therefore the common geometry frame for identity evaluation.
This is recorded as an implementation/adapter deviation, **not** as a claim that the benchmark runs the
full standalone AdaFace photo pipeline. Consequence to disclose when reporting identity numbers: they
are not comparable with published AdaFace verification benchmarks.

Tests assert: exactly one `cv2.resize` call, to (112,112) with INTER_AREA; no MTCNN / alignment /
detector module is imported by the adapter; the adapter's source contains no alignment or detector
reference; the CVLFace checkpoint is refused by hash.

---

## Q-22 — crop at image borders

**Decision:** preserve the requested 1.25× square geometry with deterministic **zero padding** outside
the image boundary.

```
w = x2 − x1 ;  h = y2 − y1
side_px = round(1.25 · max(w, h))                        # unchanged integer rule
x0 = round(cx − side_px/2) , y0 = round(cy − side_px/2)  # half-to-even, half-open box
requested = (x0, y0, x0+side_px, y0+side_px)             # built FIRST, never shrunk, never shifted
source    = requested ∩ image                            # the ONLY pixels read
crop      = zeros(side_px, side_px, 3) ; crop[pad_top:…, pad_left:…] = image[source]
```

"Clamp to image" is interpreted **operationally**: the clamping applies to the SOURCE READ region,
while the requested square is preserved with constant zero fill. No reflect/replicate border, no random
padding, no shrinking, no shifting. The pre-resize region is therefore always a true square, so the
256 resize never changes the aspect ratio. This is an owner-resolved implementation interpretation of
the spec's simultaneous "Square **padded** face" and "clamp to image" wording.

Unit tests cover: no boundary contact, left, right, top, bottom, both corners, a requested square
larger than one source dimension (padding on two opposite sides at once), determinism, refusal of a
square that does not intersect the image at all, and refusal of a non-square input at the canonical
resize. Real evidence: `M2A_BORDER_CASES.csv`.

---

## Q-23 — parsing-logit representation and storage

**Scientific representation (owner):** full **float32 11×224×224** logits, plus a **uint8** mask that is
the argmax **of the stored logits**. No float16, no mask-only, no lossy reduction; the later
geometry/parsing losses need the probability information.

**Storage (operational choice, not a scientific hyper-parameter):** lossless, deterministic, shardable.

```
block := ZSTD( SHUFFLE4( NPY_v1.0(array) ) )
```

| Property | Value |
|---|---|
| Container | `.npy` version (1,0), `allow_pickle=False`, C order (self-describing dtype/shape) |
| dtype / byte order | `<f4` (little-endian float32) |
| Byte shuffle | element size 4, pure numpy, exactly invertible |
| Library | `zstandard` 0.25.0 (libzstd 1.5.7, `cext` backend) |
| Level / frame params | 10 · `write_checksum=False` · `write_content_size=True` · `write_dict_id=False` · `threads=0` |
| Shard | 256 rows per shard, concatenated blocks, addressed by (shard, byte_offset, byte_length) |
| Integrity | sha256 per block, per uncompressed `.npy`, and per shard |
| Random access | one block read + decode, without touching the rest of the shard |

`compression settings are OPERATIONAL_STORAGE_CHOICE` is recorded explicitly in the frozen config.
Pilot results and the projection: `M2A_COMPRESSION_PILOT.md`.

---

## Q-20 / Q-21 — FaceXFormer

FaceXFormer verification is unchanged; no contradicting authoritative evidence was found in this pass.
Weights `327a7558…` (HF `kartiknarayan/facexformer` @ `fd12148d…`), code `Kartik-3004/facexformer` @
`10fe8291…`, official 224 input, ImageNet normalisation, parsing logits 11×224×224, argmax mask,
landmarks, pose. The frozen canonical 256 face is the source image; the official demo's MTCNN 50%-margin
recrop is not used (**Q-20 closed as DEV-015**).

**Q-21 stays OPEN and does not block M2B**, because the cache preserves the full logits, the raw
FaceXFormer class-channel ordering and the argmax mask, and the official source of the class metadata is
recorded. The macro / non-background mapping is needed before M9 and is deliberately not invented now.
