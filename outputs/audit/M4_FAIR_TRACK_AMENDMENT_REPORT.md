# M4 — Fair Identity-Free Track Amendment Report (Amendment A1)

**Date:** 2026-09-20 · **Outcome:** M4 **COMPLETE under the amended main-track rule**
**Amendment:** `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md`
sha256 `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`
**Original specification: NOT modified** (`f7d23716…` unchanged).

## 1. What changed and why

The primary comparison must use the **same pooled population** — CASIA-FASD + MSU-MFSD + SiW-Mv2 —
for every method. DSDG and DiffFAS natively need subject identity; SiW-Mv2 has none (Q-14: 0 of
9,507 SiW TRAIN rows carry a `subject_id_global`, against 35/35 CASIA and 25/25 MSU identities with
both live and spoof). M4 execution hit exactly this and correctly stopped
(`BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP`).

The owner's resolution removes the identity-dependent assumption from the primary comparison rather
than dropping SiW, fabricating identity, or blocking the benchmark:

| | **Track A — FAIR / COMMON / IDENTITY-FREE** | **Track B — NATIVE / FULL** |
|---|---|---|
| role | **PRIMARY** fairness table | SECONDARY analysis |
| datasets | CASIA + MSU + SiW, **every** method | may differ per method |
| identity supervision | **FORBIDDEN** | allowed |
| attack-type supervision | **FORBIDDEN** | allowed |
| N_syn | **8,838** for every method | n/a |
| status | **COMPLETE** | `DEFERRED_TO_M6_SECONDARY_TRACK` |

**Progression preserved in the audit trail:** original native requirement → blocker discovered and
reported → owner scientific objective clarified → Amendment A1 → Track-A identity-free adaptation
frozen. Earlier documents carry "superseded in part" banners; none was deleted or rewritten.

## 2. Timing — no result cherry-picking

Adopted **before** M5 probe training, M6 baseline training, M7 GPAT training, any synthetic
generation and any result inspection. No model has been trained in this project, no checkpoint
exists, `runs/`, `probes/` and `downstream/` are empty, and TEST performance has never been
observed. The decision is justified by protocol fairness and data availability only.

## 3. Source pins (authorized; source only)

| method | repository | pinned commit | mode | weights |
|---|---|---|---|---|
| DSDG (E06a/b/c) | `JDAI-CV/FaceX-Zoo` | `16b793a7564a4b9308cf94e62bdb2ffacb3a725a` | sparse `addition_module/DSDG`, blob-filtered, depth 1 | **none downloaded** |
| DiffFAS (E07a/b/c) | `murphytju/DiffFAS` | `23f40519ec25a833ebc06842aa6fbab74fad4d15` | depth 1 | **none downloaded** |

Checkouts live in `third_party/source_cache/` (git-ignored). Commits, trees and per-file sha256 +
git blob ids are in `third_party/source_pins.json` (`6f48063c…`). One stray binary checkpoint that
arrived with the FaceX-Zoo sparse checkout was removed; both caches now contain **0** weight files.
Pinning does **not** start M6.

## 4. DSDG-BIN-IDFREE (E06c, DEV-020) — `CONTROLLED_ADAPTATION`

Read from the pinned source (`M4_DSDG_IDFREE_SOURCE_ANALYSIS.md`):

- **Official relation:** `GenDataset_s` indexes spoof frames; the live partner is
  `random.choice(make_pair_dict[label]['1'])` with `label = video_name[4:6]` = the OULU **user id**.
  Same-subject, online, random, never materialized.
- **Loss dependency classification:** of `loss_rec`, `loss_kl`, `loss_mmd`, `loss_ip`, `loss_cls`,
  `loss_ort`, `loss_pair`, **only `loss_pair`** mathematically requires the two images to be the
  same person. `loss_ip` compares each reconstruction with *its own* input and does not.

**Changes — exactly two:** the relation becomes the frozen common fair pair
(`pairs_train_v1.parquet`), and `lambda_pair = 0` (official default 0.5). Everything else is
retained; nothing was removed for convenience.

**Disclosed honestly:** with the binary collapse `attack_type = 1`, `Cls` is `nn.Linear(hdim, 1)` and
`CrossEntropyLoss` over a single logit is identically 0 with zero gradient, so `loss_cls`
contributes nothing. `attack_macro`/`attack_raw` were **not** substituted and no replacement
supervision was invented.

**Training relation:** `manifests/pairs_train_v1.parquet` **itself** — an explicit frozen decision.
A separate manifest would be a redundant projection plus three constants
(`binary_spoof_label = 1`, `lambda_pair = 0`, the config hash) and a second copy could drift.
8,838 rows · CASIA 2,520 · MSU 1,200 · **SiW 5,118**. The identity-free adapter projection hashes to
`403469859598fce129f4ac8dcacbfa9bcdad4d58065dbc00177c90dc92c4f142` and contains **no identity
column at all**; nulling every CASIA/MSU subject id leaves it byte-identical.

## 5. DIFFFAS-BIN-IDFREE (E07c, DEV-021) — `CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH`

**`use_pair=false` is an official argument, not a source modification.** Traced and then **proven
numerically** against the pinned code (`M4_DIFFFAS_CONTENT_INERTNESS.json`):

| | `use_pair=true` | `use_pair=false` |
|---|---|---|
| model input channels | 6 (`x_t` ‖ content) | **3 (`x_t` only)** |
| changing only the content tensor changes the model input | yes | **no** |
| changing only the content tensor changes the loss | yes | **no** (`max_abs_loss_diff = 0.0`) |
| conditioning is `style_spoof`, not content | yes | yes |

The `use_pair=true` column is the negative control: the same probe *does* detect a dependency when
one exists, so the `false` result is meaningful. **`content_training_role = INERT_API_PLACEHOLDER`.**

**Manifest:** `manifests/difffas_bin_idfree_train_v1.parquet`
`0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6` — 8,838 rows, CASIA 2,520 ·
MSU 1,200 · **SiW 5,118**, one deterministic binary style guide per TRAIN spoof GT.

Guide rule: raw-byte SHA-256 ranking (source digest carried as 32 raw bytes, never hex), lexical
`guide_spoof_sample_id` as a defensive tie-break, over TRAIN + SPOOF + same-dataset candidates in
canonical order. Eligibility excludes subject identity **and** `attack_raw`. `style_id =
SPOOF_BINARY` is provenance only. **Self-guides are allowed** because the official
`random.choice(os.listdir(...))` enumerates the GT's own file; the 3 that occur are reported and
were not adjusted away. `content_live_id` is inherited from the common pair for bookkeeping and is
marked `INERT_WHEN_USE_PAIR_FALSE` — it does **not** form a same-identity reconstruction pair.

## 6. Track-A fairness matrix — 0 violations

All 8 planned Track-A methods (E01–E05, E06c, E07c, E08): TRAIN and VAL on
`casia_fasd + msu_mfsd + siwmv2`, `uses_subject_id = NO`, `uses_attack_type = NO`,
`n_syn_intended = 8838`. Details and the two structurally different rows (E06c is generative, so
not per-output source-conditioned, but its budget is still 8,838) are in
`M4_TRACK_A_METHOD_MATRIX.md`.

## 7. Determinism and audit

- `tools/m4_idfree_determinism.py` — **PASS**. Four fresh processes (canonical and shuffled input ×
  `PYTHONHASHSEED` 0/1/424242) reproduced the DiffFAS manifest **byte-identically** and the DSDG
  adapter relation hash identically, with rows, schema, membership and `track_pair_id` compared
  separately.
- `tools/m4_idfree_audit.py` — **PASS, 0 failures**, exhaustive over both relations.

## 8. Integrity

The three frozen common M4 artifacts and the M3 split are **byte-unchanged**: `a7ccabb0…`,
`a5e4fdae…`, `84d12491…`, `fb9aeb36…`. No M2 or M3 artifact, raw dataset or model cache was touched.
No native Track-B manifest exists. No checkpoint, synthetic bank or M5 artifact exists.

## 9. Claim limits

**Supported:** "all compared methods were trained/adapted on the same pooled CASIA-FASD +
MSU-MFSD + SiW-Mv2 TRAIN population and evaluated under the same frozen protocol."

**Not supported:** "DSDG-BIN-IDFREE reproduces official DSDG training exactly", "DIFFFAS-BIN-IDFREE
is native DiffFAS", or any same-person claim about a SiW pair. Required wording: *controlled
benchmark adaptation*, *identity-free fair-track variant*, *source-based architecture/code-path
adaptation*.

## 10. What M4 completion does and does not mean

M4 is COMPLETE **under the amended main-track rule**. The originally specified native pair manifests
were **not** created; they are Track B and deferred to M6.
`STAGE_STATE.json → milestones.M4.amendment_a1.original_native_manifests_completed = false` records
that plainly, and it must not be reported as though the original rule were satisfied.
