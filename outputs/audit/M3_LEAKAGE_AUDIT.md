# M3 — Leakage Audit

Manifest `/home/cong/GPAT_TransferBench/manifests/split_v1.parquet` · 20,615 rows · sha256 `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d`.
Overall: **PASS**.

## Global checks

| check | result |
|---|---|
| required columns present | True |
| unique sample ids | True |
| rows equal m2 complete | True |
| no failed row present | True |
| missing M2 COMPLETE rows | 0 |
| every row has a split | True |
| canonical row order | True |
| allocation group leakage total | 0 |
| video leakage total | 0 |
| sample leakage total | 0 |
| subject leakage total casia msu | 0 |
| content group leakage total siw | 0 |
| every dataset in every split | True |

Canonical videos whose frames landed in more than one split: **0**

## Per dataset

### casia_fasd

- grouping: `SUBJECT` (key `subject_id_global`) — fidelity **subject-disjoint**
- subject intersections across splits: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- allocation-group intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- video intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- sample intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}

### msu_mfsd

- grouping: `SUBJECT` (key `subject_id_global`) — fidelity **subject-disjoint**
- subject intersections across splits: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- allocation-group intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- video intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- sample intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}

### siwmv2

- grouping: `CANONICAL_VIDEO_CONTENT_GROUP` (key `content_group_id`) — fidelity **canonical-video / content-group-disjoint**
- **subject leakage: N/A** — trustworthy subject identity unavailable (Q-14)
- `subject_id_global` null for every SiW row: True
- subject-disjoint claim: **FORBIDDEN** (SiW is canonical-video / content-group-disjoint and must never be described as subject-disjoint)
- content-group intersections across splits: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- content groups spanning more than one split: **0** (all exact-byte duplicate videos stay together)
- allocation-group intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- video intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}
- sample intersections: {'TRAIN&VAL': [], 'TRAIN&TEST': [], 'VAL&TEST': []}

