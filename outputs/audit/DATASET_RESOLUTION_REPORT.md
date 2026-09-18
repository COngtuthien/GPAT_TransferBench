# Dataset Resolution Report — final dataset gate before M2/M3 (2026-09-18)

- **Base:** commit `112506b` (M1 corrected). No split, preprocessing, training or push was performed.
- **Labels used in this report:** **FACT** (measured / read), **SOURCE** (source-derived mapping), **MODEL** (model-based support), **INFERENCE**, **OWNER** (owner approval required), **BLOCKER**.

## 1. Readiness per dataset and milestone

| Dataset | M1 | M2 | M3 | Machine-readable |
|---|---|---|---|---|
| CASIA-FASD | COMPLETE | **BLOCKED_PENDING_CONTROLLED_ADAPTATION** (OWNER: DEV-011) | READY | `DATASET_READINESS_MATRIX.csv` |
| MSU-MFSD | COMPLETE | READY | READY | ″ |
| SiW-Mv2 | COMPLETE | READY | **BLOCKED_BY_MISSING_SUBJECT_ID** (BLOCKER) | ″ |

## 2. CASIA-FASD

- **FACT:** 600 canonical sequences, 50 subjects (train s1–20 → 1–20, test s1–30 → 21–50), 150 live / 450 spoof, spoof print 300 / replay 150. This was re-derived from disk independently of the manifests with 0 mismatches (`casia_source_audit.json`).
- **SOURCE:**
  - code semantics from owner-provided CASIA protocol evidence (DEV-010), mapped through SPEC §3.3 (`attack_map_v1` rev 2);
  - HR_1 = real_high → live; the local `spoof/` placement is overridden (50 logged overrides).
- **FACT:** all 123,533 PNGs are 112×112 8-bit RGB face crops. **No original CASIA copy exists on the laptop** (`CASIA_SOURCE_AUDIT.md`; GPU server not inspected).
- **FACT (diagnostic):** 112→256 INTER_CUBIC reduces relative high-frequency power on the 256 grid by about 5–6×; live and spoof are affected alike (`CASIA_RESIZE_FREQUENCY_AUDIT.md`). **Not equivalent** to the nominal §4 path.
- **OWNER:** DEV-011 `CASIA_PRE_CROPPED_112_ADAPTATION` (`CASIA_CONTROLLED_ADAPTATION_PROPOSAL.md`).
- DEV-007 (`bs*`/`fs*` derived copies excluded, still indexed) is APPROVED.

## 3. MSU-MFSD

- **FACT:** 280 videos, 35 subjects (from the clientID in the README naming protocol), each subject exactly 2 live + 6 spoof (70 / 210).
- **SOURCE:** printed_photo → print; ipad_video and iphone_video → replay (README §4 + SPEC).
- **FACT:**
  - 2 videos have one undecodable frame each (client008 @127, client023 @247), proven not to shift later indices;
  - the selected frames avoid those indices;
  - 6 videos log ProRes errors while still decoding every frame (content-aligned with PyAV).

## 4. SiW-Mv2

- **FACT:** 1,700 videos (785 live / 915 spoof), 14 attack folders. Per-type video counts equal the supplementary Table 1 for all 14 types.
- **SOURCE:** `Paper` → print (official `config_siwm.py` `'Print': 'Paper'` @ `8667dbc`, count 135 = Print), still RESOLVED. Supplementary Table 1 calls the 72-video type "Full Mask", while the local folder is `Mask_HalfMask`: mask_3d either way.
- **FACT:** official protocol entries are video stems (`preprocessing.py` L31 → `config_siwm.py` L162–164). **No source maps videos to persons.** `siw_subject_mapping_candidate.csv` = 1,700 × NO_EVIDENCE.
- **MODEL:** none. The AdaFace/SCRFD audit was not run because there are 0 source-derived candidate groups; AdaFace was **not** used as ground truth (`siw_subject_recovery/ADAFACE_AUDIT_STATUS.md`).
- **BLOCKER:** Q-14 → `BLOCKED_BY_MISSING_SUBJECT_ID` (CASE C). Spec §3.5 forbids a video-level fallback as the main protocol.
- **FACT:** 6 byte-identical Replay pairs, all preserved and UNRESOLVED_DUPLICATE. 3 pairs are split across the official testlist_all/trainlist_all (`SIW_DUPLICATE_VIDEO_AUDIT.md`). Any future split must keep identical content together.
- **FACT:** the official lists reference 64 stems that are absent from the local release (Q-17, info).

## 5. Frame semantics (all datasets)

- **DEV-005 APPROVED** (IMPLEMENTATION_DETAIL): nearest unused valid index; lower index wins ties; the earlier position keeps a contested frame. Collision fallback used in 0 videos.
- **DEV-006 APPROVED:** index-bounded decode over [0, N_declared).
  - Continuity was proven with an independent decoder (PyAV 18.1.0) for the 2 mid-stream failures, and via timestamp agreement for all 1,980 videos.
  - OpenCV `POS_FRAMES` lags after a failure and must not be used as the index (`DECODER_INDEX_AUDIT.md`).

## 6. Configs and manifests

- `data_v1.yaml` and `attack_map_v1.yaml` stay at **revision 2** (no change in this pass). Revision 1 is preserved in `frozen_config_snapshot/history/`, with the exact diffs in `CONFIG_REVISION_LOG.md`.
- The manifests were **not regenerated**: no metadata decision required a correction. They were re-validated by tests and by the independent CASIA disk re-audit (0 mismatches). `inventory.parquet` sha256 `f3a882c91eb3de808ded61e92f2bf20a6b611d21669420f1e1a64e244a979173` is unchanged from `112506b`.

## 7. What the owner must decide

1. DEV-011 (CASIA controlled adaptation), or provide an original CASIA source.
2. SiW-Mv2 subject IDs. Options: obtain an authoritative per-video subject list from the dataset provider, or approve a separately named non-main protocol, or exclude SiW-Mv2 from the main split. Each option needs its own deviation.
3. Q-16 policy for M3: recommend keeping identical-content groups split-atomic by sha256.
