# M4 — Pair Distance Analysis (diagnostic only)

Evaluated on a deterministic hash-selected sample of 400 split rows (salt `gpatbench.m4_preflight.diag.v1|`), purely to measure how much each *unresolved* convention changes the numbers. **No formula was chosen from these distributions**, and TEST was never inspected for a decision.

## Pose (Q-25)

z-normalization fitted on **14,467 TRAIN rows only**; VAL and TEST never contribute a statistic.

| axis | mean (rad) | std (population) | std (sample) | relative difference |
|---|---|---|---|---|
| pitch | -0.023073 | 0.151126 | 0.151131 | 3.5e-05 |
| yaw | 0.008509 | 0.231691 | 0.231699 | 3.5e-05 |
| roll | -0.016004 | 0.083488 | 0.083491 | 3.5e-05 |

Zero-variance axes: none. Per-dataset std differs from the pooled std ({'casia_fasd': [0.077981, 0.051887, 0.05637], 'msu_mfsd': [0.1021, 0.057406, 0.043483], 'siwmv2': [0.175045, 0.283055, 0.095737]}), so **pooled vs per-dataset scope changes the metric** — it is not a cosmetic choice.

`d_pose` under L2: median 1.4505 (max 20.8266); under L1: median 2.1268 (max 33.1349); L1/L2 median ratio 1.4662. The norm materially rescales the 0.50-weighted term relative to the other two, so it cannot be guessed.

## Scale (Q-26)

| dataset | rows with a face box | rows without |
|---|---|---|
| casia_fasd | 0 | 94 |
| msu_mfsd | 46 | 0 |
| siwmv2 | 260 | 0 |

**CASIA has no face box at all** (`casia_has_no_face_box = True`). CASIA ran no SCRFD (DEV-011), so no detector box exists for any CASIA row; every scale_box variant needing one is undefined there.

This is the hardest of the four gaps: it is not a choice between conventions but a missing quantity. Any resolution (treat the whole CASIA frame as the box, derive a box from FaceXFormer landmarks, drop the term for CASIA and renormalise the weights, or exclude CASIA from common pairs) changes the scientific metric for a third of the sources, so none may be adopted silently.

Where a box exists, `|log(area ratio)|` on raw bbox pixels has median 0.2703 (max 2.1265).
Using the bbox **fraction of the frame** instead gives median 0.5731 — different, because SiW and MSU frames have different resolutions, so raw pixel area is not comparable across videos.

## Luminance (Q-27)

| standard | min | mean | max |
|---|---|---|---|
| BT.601 | 0.1952 | 0.4103 | 0.7442 |
| BT.709 | 0.1921 | 0.4050 | 0.7411 |

Per-image difference between the two standards reaches **0.0132** (mean 0.0054) on a quantity whose whole observed range is about 0.55. OpenCV BGR2YCrCb matches BT.601 to ~1e-4; the BT.601/BT.709 gap is far larger. The choice is therefore material and is left to the owner.

## Combination

`d_pair = 0.50·d_pose + 0.30·d_scale + 0.20·d_luma` with the spec's weights, minimum wins, exact ties broken by lexical target `sample_id` with no fuzzy tolerance. The weights are frozen and were not touched; what is missing is the definition of the three terms they weigh.
