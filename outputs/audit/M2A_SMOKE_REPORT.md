# M2A Smoke Report — DIAGNOSTIC_ONLY (2026-09-19)

This is not M2 output. The 24-sample size has no inferential meaning. No model choice was tuned from these results.
Outputs (images and tensors) are in `outputs/exploratory/m2a_smoke/run{1..4}/` (git-ignored). Only small text evidence is committed:
- `M2A_SMOKE_MANIFEST.csv` (sha256 `138f5929f81313069e4a4cb74c4ed54e4957432fe9ebc93cc52fdd14100dbce7`);
- `M2A_SMOKE_RESULTS.csv` (run3);
- `M2A_SMOKE_SUMMARY.json`;
- `M2A_DETERMINISM_COMPARE.json`.

## Selection rule (deterministic)
- Per dataset, sort all M1 sample rows by `sha256("gpatbench.m2a_smoke.v1|" + sample_id)` and take the first 4 live + 4 spoof.
- Each sample must come from a distinct canonical video.
- For spoof, a first pass takes only unseen `attack_raw` values; a second pass relaxes this.
- Result: CASIA 8 (codes 3, 5, 8, HR_3 + 4 live), MSU 8 (printed_photo ×2, ipad_video, iphone_video + 4 live), SiW 8 (Paper, Replay, Mask_TransparentMask, Makeup_Impersonation + 4 live). Total 24.
- The manifest was rebuilt twice with an identical sha256.

## PROVISIONAL settings used (open questions — NOT frozen)
- Q-02: SCRFD candidate A (`scrfd_10g_bnkps.onnx`).
- Q-18: AdaFace RGB input.
- Q-19: resize 256→112 INTER_AREA.
- Q-22: literal clamp intersection.

## Results (run3; run4 byte-identical; superseded run1 identical except timing)
| dataset | label | attack_raw | route | SCRFD | n_det | score | clamped | FX | AdaFace | L2 | cos(RGB,BGR input) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| casia_fasd | 1 | HR_3 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.884704 |
| casia_fasd | 1 | 8 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.921866 |
| casia_fasd | 1 | 5 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.942218 |
| casia_fasd | 1 | 3 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.930731 |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.942488 |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.952122 |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.965771 |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | OK | OK | 1.00000000 | 0.951617 |
| msu_mfsd | 1 | printed_photo | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.803592 | False | OK | OK | 0.99999994 | 0.943505 |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.909067 | False | OK | OK | 1.00000000 | 0.929424 |
| msu_mfsd | 1 | printed_photo | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.848040 | False | OK | OK | 1.00000000 | 0.922470 |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.883628 | False | OK | OK | 1.00000000 | 0.866992 |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.905434 | False | OK | OK | 1.00000000 | 0.905765 |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.909971 | False | OK | OK | 1.00000000 | 0.921019 |
| msu_mfsd | 1 | ipad_video | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.918427 | False | OK | OK | 1.00000000 | 0.925718 |
| msu_mfsd | 1 | iphone_video | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.785135 | False | OK | OK | 1.00000000 | 0.934207 |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 2 | 0.875302 | True | OK | OK | 1.00000000 | 0.896209 |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.850651 | True | OK | OK | 1.00000000 | 0.930990 |
| siwmv2 | 1 | Paper | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.881039 | False | OK | OK | 1.00000000 | 0.870636 |
| siwmv2 | 1 | Replay | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.839587 | False | OK | OK | 1.00000000 | 0.946100 |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.891394 | False | OK | OK | 1.00000012 | 0.873529 |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.640992 | True | OK | OK | 1.00000000 | 0.859662 |
| siwmv2 | 1 | Mask_TransparentMask | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.885656 | False | OK | OK | 1.00000000 | 0.811111 |
| siwmv2 | 1 | Makeup_Impersonation | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.914333 | False | OK | OK | 1.00000000 | 0.932044 |

**Summary:**
- **SCRFD:** MSU 8/8 detected, SiW 8/8 detected; `SCRFD_NO_FACE` = 0. One SiW frame had 2 detections; the largest was selected. CASIA: N/A (not run, not counted as success).
- **FaceXFormer:** 24/24 OK, all outputs finite.
- **AdaFace:** 24/24 OK; L2 ∈ [0.99999994, 1.00000012].
- **Failures:** none. No fallback was used.

**Observations relevant to owner decisions (not tuning):**
- 3/8 SiW crops were clamped at the frame border (0/8 MSU). The border rule (Q-22) therefore affects a material fraction of SiW.
- AdaFace embeddings from RGB vs BGR input have cosine 0.811–0.966. The colour order (Q-18) materially changes the identity cache.
- The smoke does not reveal a systematic detection failure. With 16 detector samples it cannot bound the full-data SCRFD_NO_FACE rate.

**Run lineage:**
- run1/run2: first code version, byte-identical to each other.
- A code refactor followed (no-face helper and manifest path parameter; no numerical change).
- run3/run4: final code. 208/208 files byte-identical, and results identical to run1 except timing.
