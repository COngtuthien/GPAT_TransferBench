# M2 — SCRFD Success Report

Frozen detector SCRFD_10G_KPS at the spec threshold 0.50, largest bbox, no fallback of any kind.
**CASIA-FASD is N/A** and is excluded from every denominator: its route is the approved pre-cropped adaptation (DEV-011), so the detector is never run for it and it is never counted as a detector success.

| dataset | applicable | success | no-face | other error | failed before detection | success % |
|---|---|---|---|---|---|---|
| msu_mfsd | 2,240 | 2,240 | 0 | 0 | 0 | 100.0000 % |
| siwmv2 | 13,600 | 13,575 | 25 | 0 | 0 | 99.8162 % |
| **all applicable** | **15,840** | **15,815** | **25** | **0** | **0** | **99.8422 %** |
| casia_fasd | 0 (N/A) | N/A | N/A | N/A | N/A | N/A |

## Detection-score and bbox distributions (descriptive only)

These are reported for audit. The threshold and the crop policy are frozen and are **not** revisited on the basis of this table.

| dataset | score min | score p05 | score median | score max | bbox area frac min | median | max | padded crops |
|---|---|---|---|---|---|---|---|---|
| msu_mfsd | 0.7309 | 0.7812 | 0.8625 | 0.9292 | 0.04419 | 0.12607 | 0.27925 | 31 |
| siwmv2 | 0.5888 | 0.7832 | 0.8791 | 0.9369 | 0.01142 | 0.08554 | 0.90559 | 1,807 |
