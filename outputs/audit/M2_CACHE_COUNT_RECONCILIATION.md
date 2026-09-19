# M2 Cache Count Reconciliation

**Date:** 2026-09-20 · **Pass:** M4 pre-execution correction · **Mode:** read-only

Two M2 cache counts appear in the audit trail and look inconsistent:

| Where | Statement |
|---|---|
| `EXECUTION_FLOW.md` (M2 completion, 2026-09-19) | "full block-level cache integrity **PASS** across all 7 fields (**567 shards**, …)" |
| `M2_CACHE_INTEGRITY.json` (M2 completion) | `shards: 81` **per field** |
| M4 owner decision report §13 (2026-09-19) | "**972** geometry shards … **162** identity shards" |

**Neither number is wrong.** They count three different units, and one of them was given an
ambiguous name. This document defines every unit and states the measured value of each. No M2
artifact was read for content, rebuilt, re-indexed or modified.

---

## 1. Cache layout (measured)

```
<runtime>/cache/geometry/<field>/<field>-NNNNN.bin             <- data shard file
<runtime>/cache/geometry/<field>/<field>-NNNNN.bin.index.json  <- index file for that shard
<runtime>/cache/identity/embedding/embedding-NNNNN.bin
<runtime>/cache/identity/embedding/embedding-NNNNN.bin.index.json
```

- **Geometry fields (6):** `landmarks_norm`, `landmarks_px224`, `landmarks_px256`,
  `parsing_logits`, `parsing_mask`, `pose_pitch_yaw_roll_rad`
- **Identity fields (1):** `embedding`
- **Total fields: 7**
- Shard ordinals run `00000` … `00080` in every field — the same 81 ordinals everywhere, because
  the ordinal → shard assignment is one plan computed before processing (`m2b.build_plan`) and every
  field follows it. Fields are therefore **row-aligned**, not independently sharded.

## 2. Unit definitions and measured values

| Unit | Definition | Value |
|---|---|---|
| **Logical samples** | rows in the frozen M2 inventory | **20,640** |
| **Logical samples, M2 COMPLETE** | inventory rows that produced a canonical face | **20,615** (25 `SCRFD_NO_FACE` failures) |
| **Logical geometry entries** | index rows per geometry field | **20,615 per field** (6 × 20,615 = 123,690 field-entries) |
| **Logical identity entries** | index rows in `embedding` | **20,615** |
| **Logical shard groups** | distinct shard ordinals in the plan; one group spans all 7 fields | **81** |
| **Geometry data shard files** | `.bin` files, by field | 81 per field × 6 = **486** |
| **Geometry index files** | `.bin.index.json` files, by field | 81 per field × 6 = **486** |
| **Identity data shard files** | `.bin` files | **81** |
| **Identity index files** | `.bin.index.json` files | **81** |
| **Data shard files, all 7 fields** | 7 × 81 | **567** |
| **Index files, all 7 fields** | 7 × 81 | **567** |
| **Total physical cache files** | every regular file under `cache/geometry` + `cache/identity` | **1,134** |

Per field, uniformly: 81 data shard files, 81 index files, 162 physical files, 20,615 entries.

## 3. What each previously reported count meant

- **"567 shards"** (`EXECUTION_FLOW.md`) = **data shard files across all 7 fields** = 7 × 81.
  Accurate, but "shards" there means *per-field data shard files*, not shard groups.
- **`shards: 81`** (`M2_CACHE_INTEGRITY.json`) = **shard groups**, reported once per field because
  the verifier runs per field. Accurate; this is the logical unit.
- **"972 geometry shards / 162 identity shards"** (M4 owner decision report §13) = **physical
  files**, i.e. data shard files **plus** index files: geometry 486 + 486 = 972, identity 81 + 81 =
  162. The counting was correct; calling physical files "shards" was the ambiguity. In the units
  defined above, geometry has **81 shard groups**, **486 data shard files** and **972 physical
  files**.

All three reduce to the same underlying cache: **81 shard groups × 7 fields**, each field holding
**20,615** entries, stored as **567 data shard files + 567 index files = 1,134 physical files**.

## 4. Related non-cache counts (for completeness)

| Artifact | Count | Note |
|---|---|---|
| `data/processed/faces_256` | **20,615** | CASIA 4,800 · MSU 2,240 · SiW 13,575 — one canonical face per M2 COMPLETE sample |
| `data/processed/frames` | **15,840** | MSU 2,240 · SiW 13,600 — **no CASIA entries** |

CASIA contributes 0 persisted frames by design: under **DEV-011** the available CASIA-FASD copy is
already pre-cropped 112×112 face imagery, so no full frame exists to extract and none is invented.
SiW has 13,600 frames but 13,575 faces: the 25 `SCRFD_NO_FACE` samples produced a frame and no face,
which is exactly the M2 failure accounting already recorded in
`manifests/m2_sample_accounting.parquet`.

## 5. Terminology adopted going forward

`shard group` (81) · `data shard file` (567) · `index file` (567) · `physical cache file` (1,134) ·
`entry` (20,615 per field). Reports must name the unit; the bare word "shards" is not sufficient.

## 6. Integrity

Read-only pass. 0 files created, modified or deleted under the M2 runtime root; 0 shards rebuilt;
0 indexes rewritten; no model output touched. Historical reports were **not** rewritten — a
clarification pointing here was appended to `EXECUTION_FLOW.md` instead.
