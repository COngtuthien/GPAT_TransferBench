# M4 — Track-A Identity-Free Relation Audit

**Date:** 2026-09-20 · **Result: PASS — 0 failures** · Machine-readable: `M4_IDFREE_MANIFEST_AUDIT.json`
Tool: `tools/m4_idfree_audit.py` (read-only; re-derives both relations and compares).

## 1. DSDG-BIN-IDFREE (E06c, DEV-020)

**Authoritative training relation: `manifests/pairs_train_v1.parquet`**
(`a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243`) — no separate manifest was
created, and `manifests/dsdg_bin_idfree_train_v1.parquet` is asserted **not** to exist.

| check | result |
|---|---|
| rows | **8,838** = expected |
| CASIA / MSU / SiW | **2,520 / 1,200 / 5,118** = expected |
| SiW included | **YES** |
| source set == TRAIN spoof set | PASS (no missing, no extra, no duplicate) |
| every row TRAIN, both endpoints M2 COMPLETE | PASS |
| source SPOOF, target LIVE, same dataset | PASS |
| `binary_spoof_label` = 1 on every row | PASS |
| `lambda_pair` | **0.0** (official default 0.5) |
| losses removed | `["loss_pair"]` — exactly one |
| losses retained | `loss_rec`, `loss_kl`, `loss_mmd`, `loss_ip`, `loss_cls`, `loss_ort` |
| subject equality required by the method | **NO** |
| `subject_id_global` consumed | **NO** |
| adapter interface fields | `pair_id`, `dataset`, `source_spoof_id`, `target_live_id`, `split`, `seed`, `binary_spoof_label`, `source_sha256`, `target_sha256` — **no identity column exists at all** |
| relation unchanged when every CASIA/MSU subject id is nulled | **PASS** |

The last row is the subject-metadata firewall test in its strongest form: nulling
`source_subject` / `target_subject` on every common pair leaves the Track-A relation byte-identical,
which proves the adapter cannot be consuming them.

## 2. DIFFFAS-BIN-IDFREE (E07c, DEV-021)

**Manifest:** `manifests/difffas_bin_idfree_train_v1.parquet`
**SHA256:** `0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6`
**Schema signature:** `eb43e16359800d116617ffa6677d10c964c1e63de8cc6ba00a625141a265d2f0` (19 columns)

| check | result |
|---|---|
| rows | **8,838** = expected, one per TRAIN spoof GT |
| CASIA / MSU / SiW | **2,520 / 1,200 / 5,118** = expected |
| SiW included | **YES** |
| unique GT | 8,838 (GT set == TRAIN spoof set) |
| written manifest == independent rebuild | PASS (all 19 columns, every row) |
| `track_pair_id` | canonical `DFA000001`…`DFA008838` |
| row order | canonical `(dataset, gt_spoof_id)` |
| every row TRAIN | PASS |
| GT SPOOF and guide SPOOF | PASS |
| guide from the same dataset | PASS |
| both endpoints M2 COMPLETE | PASS |
| `style_id` | `SPOOF_BINARY` on every row |
| `use_pair` | `false` on every row |
| `content_training_role` | `INERT_WHEN_USE_PAIR_FALSE` on every row |
| guide is the deterministic minimum | PASS — recomputed for all 8,838 rows, 0 mismatches |
| `eligible_guide_count` | 2,520 / 1,200 / 5,118 (the full binary spoof pool of each dataset) |
| attack-type column in the schema | **none** |
| identity column in the schema | **none** |
| subject identity consumed | **NO** — not for eligibility, selection, loss or model input |
| provenance hashes on every row | split `fb9aeb36…`, common pair `a5e4fdae…`, config `aa9e9841…`, source commit `23f40519…` |

### Guide diagnostics (reported, never optimised)

- **unique guides used:** 5,580 of 8,838 GTs
- **maximum guide reuse:** 5
- **self-guides:** **3**

The self-guide count is reported because the official `random.choice(os.listdir(style_folder))`
enumerates the GT's own file, so a self-guide is inside the official support. No selection was
modified after the fact to reduce it.

## 3. Shared Track-A invariants

| check | result |
|---|---|
| Track-A datasets | `[casia_fasd, msu_mfsd, siwmv2]` |
| `n_syn_intended` | 8,838 for every Track-A method |
| `pair_train_stats_v1.json` unchanged | `a7ccabb0…` PASS |
| `pairs_train_v1.parquet` unchanged | `a5e4fdae…` PASS |
| `val_pairs_v1.parquet` unchanged | `84d12491…` PASS |
| `split_v1.parquet` unchanged | `fb9aeb36…` PASS |
