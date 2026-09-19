# M2A Smoke Report — DIAGNOSTIC_ONLY, frozen contract (2026-09-19)

This is not M2 output. The 24-sample size has no inferential meaning. No model, adapter or storage choice
was selected or tuned from these results; every choice is the owner's, recorded in `M2A_OWNER_DECISIONS.md`.
Outputs (images, tensors, logit shards) are in `outputs/exploratory/m2a_smoke/run{A..D}/` (git-ignored).
Only small text evidence is committed:
- `M2A_SMOKE_MANIFEST.csv` (sha256 `138f5929f81313069e4a4cb74c4ed54e4957432fe9ebc93cc52fdd14100dbce7`) — **unchanged**; the same 24 samples as the previous pass, deliberately not reselected;
- `M2A_SMOKE_RESULTS.csv` (run `runC`), `M2A_SMOKE_SUMMARY.json`, `M2A_BORDER_CASES.csv`;
- `M2A_DETERMINISM_COMPARE.json`, `M2A_COMPRESSION_PILOT.{md,csv,json}`, `M2A_CONTRACT_CHANGE_IMPACT.json`.

## Frozen settings used (no provisional setting remains)

| Q | Setting |
|---|---|
| Q-02 | SCRFD_10G_KPS `scrfd_10g_bnkps.onnx` (official byte hash confirmed) |
| Q-03 | original AdaFace R50 / WebFace4M `adaface_ir50_webface4m.ckpt` |
| Q-18 | BGR input |
| Q-19 | canonical 256 → 112 INTER_AREA, no additional alignment |
| Q-22 | requested square preserved, zero padding outside the image |
| Q-23 | full float32 11×224×224 logits, lossless `npy1+shuffle4+zstd` level 10 |

## Routes exercised

- **CASIA:** 112 RGB → INTER_CUBIC 256 → FaceXFormer → canonical256 → AdaFace 112 INTER_AREA / BGR. `scrfd_applied=false`, `scrfd_status=N/A` (DEV-011); CASIA is never counted as a detector success.
- **MSU:** frozen frame index → SCRFD 10G → requested 1.25× square → zero pad if required → canonical 256 → FaceXFormer → AdaFace adapter.
- **SiW:** same nominal path as MSU.

## Results (run `runC`; run `runD` byte-identical)

| dataset | label | attack_raw | route | SCRFD | n_det | score | border | crop geometry | FX | AdaFace | L2 | logit ratio | lossless |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| casia_fasd | 1 | HR_3 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.305705 | True |
| casia_fasd | 1 | 8 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.311068 | True |
| casia_fasd | 1 | 5 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.304657 | True |
| casia_fasd | 1 | 3 | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.310745 | True |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.312281 | True |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.304010 | True |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.308018 | True |
| casia_fasd | 0 | - | PRECROPPED_112_RGB_TO_256_INTER_CUBIC | N/A | - | - | - | — | OK | OK | 1.00000000 | 1.305650 | True |
| msu_mfsd | 1 | printed_photo | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.803592 | False | 365² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.313600 | True |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.909067 | False | 185² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.310650 | True |
| msu_mfsd | 1 | printed_photo | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.848040 | False | 329² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.312509 | True |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.883628 | False | 200² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.319913 | True |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.905434 | False | 190² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.310916 | True |
| msu_mfsd | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.909971 | False | 229² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.318666 | True |
| msu_mfsd | 1 | ipad_video | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.918427 | False | 261² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.316034 | True |
| msu_mfsd | 1 | iphone_video | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.785135 | False | 268² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.312432 | True |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 2 | 0.875302 | True | 1082² pad(L0,T0,R0,B16) | OK | OK | 1.00000000 | 1.315376 | True |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.850651 | True | 836² pad(L37,T0,R79,B0) | OK | OK | 1.00000000 | 1.309352 | True |
| siwmv2 | 1 | Paper | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.881039 | False | 630² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.321758 | True |
| siwmv2 | 1 | Replay | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.839587 | False | 978² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.315388 | True |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.891394 | False | 498² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.309635 | True |
| siwmv2 | 0 | - | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.640992 | True | 755² pad(L0,T0,R0,B102) | OK | OK | 1.00000000 | 1.316677 | True |
| siwmv2 | 1 | Mask_TransparentMask | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.885656 | False | 481² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.314413 | True |
| siwmv2 | 1 | Makeup_Impersonation | SPEC_SECTION_4_SCRFD | DETECTED | 1 | 0.914333 | False | 448² pad(L0,T0,R0,B0) | OK | OK | 1.00000000 | 1.307594 | True |

**Summary**

- **SCRFD:** MSU 8/8 detected, SiW 8/8 detected; `SCRFD_NO_FACE` = 0. One SiW frame had 2 detections; the largest was selected. CASIA: N/A.
- **Crop (Q-22):** every detected crop is a true square before resizing (`crop_square_before_resize_all = true`); 3/8 SiW and 0/8 MSU crops extend past the frame edge and are zero-padded. Details: `M2A_BORDER_CASES.csv`.
- **FaceXFormer:** 24/24 OK, all outputs finite.
- **AdaFace:** 24/24 OK; 512-D; L2 ∈ [1.0, 1.0].
- **Logit storage (Q-23):** 24/24 exact reconstruction (`np.array_equal` and byte equality), mask re-derived from the decoded logits matches; compression ratio 1.304–1.322.
- **Failures:** none. No fallback was used.

## What the contract change actually changed (`M2A_CONTRACT_CHANGE_IMPACT.json`)

Comparing the previous provisional run (`run3`) with this one:

- **21/24 canonical faces are byte-identical**; the **3** that changed are exactly the SiW crops that touch the frame border — i.e. only Q-22 moved them;
- FaceXFormer parsing logits are byte-identical on all 21 unchanged faces;
- AdaFace cos(previous CVLFace+RGB, new original+BGR) ∈ [0.99999994, 1.00000012] on those faces.

That last line is expected and is itself evidence: the two official AdaFace releases hold the same trained
weights (466/467 tensors bit-identical; `input_layer.0.weight` is an exact channel-axis reversal), so
"CVLFace + RGB" and "original + BGR" are the same function. The owner decision restores literal compliance
with the spec's official-BGR wording without changing the identity space. It is **not** the same quantity as
the previous pass's diagnostic `cos(RGB-input, BGR-input) = 0.81–0.97`, which fed one checkpoint the wrong
channel order and is no longer computed (the adapter has no RGB path).

## Run lineage

- `run1`–`run4`: previous pass, provisional contract (kept as history; superseded).
- `runA`/`runB`: first runs on the frozen contract. They exposed a weakness in the comparison tool — the
  results-table comparison split CSV lines on commas, which mis-aligns the JSON geometry columns, so the
  timing columns were not actually excluded. The file-level byte comparison was unaffected. Fixed to use a
  real CSV parser; recorded in the ledger as a TECHNICAL_FAILURE/RECOVERED entry.
- **`runC`/`runD`: the reported deterministic rerun, produced with the final code from clean output
  directories.** The earlier run3/run4 result is not reused as proof, since the contract changed.
