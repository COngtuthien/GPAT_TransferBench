# CASIA 112→256 Resize Frequency Audit — DIAGNOSTIC ONLY (2026-09-18)

- **Purpose:** quantify how the controlled resize (spec §4 upscaling interpolation INTER_CUBIC) changes image content, especially high frequencies. It is **not** a claim of equivalence with the nominal SCRFD raw-frame path, and no parameter is tuned from it.
- **Sample:** all 4,800 CASIA frames already selected by the M1 inventory. This is a deterministic set fixed before the audit; no frame was chosen by result.
- **Script:** `tools/audit_casia_resize.py`. Per-image values: `CASIA_RESIZE_FREQUENCY_AUDIT.csv`.
- **Luminance:** Y = BT.601 of the decoded BGR, in [0, 1]. Frequencies are in cycles/pixel of each image's **own** grid (Nyquist 0.5).

| Metric | median 112 | median 256 | median ratio 256/112 live | median ratio spoof |
|---|---|---|---|---|
| mean Y | 0.38377 | 0.38377 | 1.000 | 1.000 |
| std Y | 0.12855 | 0.12864 | 1.001 | 1.001 |
| power fraction f ∈ [0, 0.1) | 0.89590 | 0.97190 | 1.084 | 1.084 |
| power fraction f ∈ [0.1, 0.2) | 0.06714 | 0.02186 | 0.334 | 0.315 |
| power fraction f ∈ [0.2, 0.3) | 0.02081 | 0.00315 | 0.147 | 0.152 |
| power fraction f ∈ [0.3, 0.5] | 0.01473 | 0.00293 | 0.198 | 0.210 |
| FreqSub region r ∈ [0.20, 0.50] (spec §8.2) | 0.03573 | 0.00613 | 0.167 | 0.174 |
| power above source Nyquist (0.21875 c/px of the 256 grid) | (0.03104)* | 0.00529 | 0.168 | 0.176 |
| Haar L1 detail energy fraction (LH+HL+HH)/total | 0.03362 | 0.00707 | 0.212 | 0.209 |
| mean \|Y − GaussianBlur(9, σ1.5)\| (spec §4 high-pass) | 0.01486 | 0.00541 | 0.369 | 0.361 |
| Laplacian variance | 0.04356 | 0.00308 | 0.076 | 0.071 |

\*For the 112 image, "above 0.21875" is simply the fraction of its own spectrum above that frequency. For the 256 image, any power above 0.21875 c/px did not exist in the source grid: it is interpolation-generated (0.53%).

## Interpretation

**FACT.** On the 256×256 grid, the upscaled CASIA frames carry about 5–6× less relative power in the bands that GPAT and several baselines operate on:
- Haar level-1 detail energy: −79%;
- FreqSub eligible region: −83%;
- §4 high-pass magnitude: −63%;
- Laplacian variance: −92%.

Mean and contrast are unchanged. The reductions are almost identical for live and spoof frames (median ratios differ by ≤ 0.02).

**INFERENCE (not measured here).** MSU/SiW crops come from full frames through SCRFD and are typically *downscaled* (INTER_AREA) to 256, which preserves high-frequency content relative to the grid. After adaptation, CASIA samples would therefore sit in a systematically different frequency regime from MSU/SiW samples in the pooled benchmark. This affects:
- wavelet-residual methods (GPAT);
- FreqSub;
- the high-pass ArtifactProbe;
- possibly dataset-shortcut learning by the detector.

The effect applies equally to all methods that consume the canonical 256 CASIA images, but it may interact differently with frequency-domain methods.

**Not claimed:** equivalence with the nominal pipeline, or any effect on downstream metrics.
