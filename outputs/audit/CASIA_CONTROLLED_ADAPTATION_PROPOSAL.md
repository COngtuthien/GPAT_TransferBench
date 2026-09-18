# Proposal: CASIA_PRE_CROPPED_112_ADAPTATION

- **Classification:** CONTROLLED_DATASET_ADAPTATION
- **Status:** PROPOSED — requires owner approval before M2. Nothing has been run.
- **Affected dataset:** CASIA-FASD only.
- **Affected milestones:** M2 (preprocessing) and everything that consumes CASIA `faces_256`.

## 1. Why it is necessary (FACT)

- Spec §4 nominal path: raw frame → SCRFD (input 320, thr 0.50, largest face) → square crop 1.25×max(w,h) centred on the bbox, clamped → 256×256 RGB (INTER_AREA down / INTER_CUBIC up).
- The only CASIA-FASD data available locally are 110,859 canonical frames that are **already 112×112 face crops** (`CASIA_SOURCE_AUDIT.md`). Every copy found is the same repack; no raw frames or videos exist on this laptop.
- The unknown upstream crop (detector, margin, alignment) cannot be reproduced or verified. Re-running SCRFD on an already-cropped 112×112 face and then applying a 1.25× bbox crop would produce a **second, nested crop** of unknown geometry, not the nominal crop.

## 2. Proposed exact pipeline (CASIA only)

```
canonical local 112×112 PNG frame (the M1-selected frame index; sha256 = source file bytes)
    │ decode with the pinned decoder (cv2.imdecode, IMREAD_COLOR → BGR uint8) and convert to RGB;
    │ assert shape 112×112×3, 8-bit (PNG IHDR colour type RGB; no ICC/gamma handling assumed = sRGB-as-stored)
    ▼
cv2.resize(→ 256×256, interpolation=INTER_CUBIC)   # the spec §4 upscaling interpolation
    │ clip/round to uint8 (OpenCV saturating cast)
    ▼
canonical faces_256 image for CASIA; its sha256 becomes the sample sha256 for M2 onwards
```

- SCRFD detection and the 1.25× crop are **skipped for CASIA only**. The detector success report records CASIA as `ADAPTATION_PRECROPPED`, never as detector success or failure.
- Geometry (FaceXFormer) and identity (AdaFace) caches are computed on this 256×256 image exactly as for the other datasets.

## 3. What differs from the frozen nominal pipeline

| Aspect | Nominal §4 | Adaptation |
|---|---|---|
| Crop source | SCRFD bbox on the raw frame | unknown upstream crop by the packager |
| Crop margin / centring | 1.25×max(w,h), bbox-centred | unknown (visually tight face crops) |
| Resize direction | usually downscale (INTER_AREA) | always upscale 112→256 (INTER_CUBIC) |
| High-frequency content | native-resolution detail | band-limited: effective source Nyquist is 0.219 c/px of the 256 grid (`CASIA_RESIZE_FREQUENCY_AUDIT.md`) |
| Background / context | 1.25× margin of context | whatever the packager kept |

## 4. What remains equal across methods

- Every method (all generators, the ArtifactProbe, both evaluators) receives the **same** canonical CASIA 256×256 images.
- Same frame indices (M1), same split (M3), same pairs (M4), same budgets, same seeds.
- The adaptation is applied upstream of all method-specific code, so it creates no method-specific advantage in data access.

## 5. Possible scientific impact (INFERENCE — to be reported, not tuned)

1. **Frequency regime shift.** CASIA canonical images have about 5–6× less relative high-frequency energy than a native crop would. Wavelet (GPAT), FFT-block (FreqSub) and high-pass (ArtifactProbe, fingerprint probe) measurements on CASIA are not comparable in absolute terms to MSU/SiW.
2. **Dataset shortcut.** The detector could separate CASIA from the other datasets by blur level. The spec already forbids feeding dataset ID. Subgroup reporting (T14) and the fingerprint probe (§12.2) should be read with this in mind.
3. **Pair building (§6).** Pairs are same-dataset, so CASIA targets are paired only with CASIA sources: no cross-resolution pairs are created.
4. **Geometry/identity teachers** run on upscaled, band-limited faces. Their outputs may be less reliable on CASIA; M2 success/quality reports must be broken down by dataset.

## 6. Fairness argument

The adaptation is applied identically to every method, before any method sees the data, and it
does not use any label. It changes the *difficulty* and *frequency content* of the CASIA subset for all
methods alike, and does not favour one method's data access. It may still interact with method design
(frequency-based methods are affected more), so results must carry the dataset-wise breakdown and
this limitation.

## 7. Limitations and required disclosures

- The CASIA rows are **not** produced by the frozen §4 preprocessing. The paper must say so explicitly.
- The main pooled result mixes one adapted dataset with two nominal ones. An alternative that could be reported is a sensitivity analysis excluding CASIA; that is **not** proposed as the main protocol here.
- If an original CASIA copy becomes available (e.g. on the GPU server), the nominal path should be preferred and this adaptation withdrawn before M2.

## 8. Owner decision required

Choose one of:
- **(a)** approve this adaptation for M2;
- **(b)** provide an original CASIA-FASD source and keep the nominal path;
- **(c)** another controlled option (e.g. exclude CASIA). This would change the dataset protocol and needs its own deviation.
