# Deviation Report — GPAT-TransferBench v1.0

Per spec §0.1 rule 9: every deviation from the frozen specification is recorded here with
reason, affected experiments and explicit status **APPROVED / UNAPPROVED**. Unapproved
deviations cannot be used for main paper results. Only the project owner may change a
status to APPROVED. This file is append-only in spirit: entries are never deleted; status
changes are appended as dated updates under the entry.

## Effective status summary (latest update wins; history kept below)

| ID | Effective spec status | Owner classification | Updated |
|---|---|---|---|
| DEV-001 | APPROVED | INFRASTRUCTURE_ADAPTATION | 2026-09-18 |
| DEV-002 | APPROVED | INFRASTRUCTURE_ADAPTATION | 2026-09-18 |
| DEV-003 | UNAPPROVED | UNRESOLVED — must be resolved before M7 | 2026-09-18 |
| DEV-004 | APPROVED | APPROVED_OPERATIONAL_DEVIATION (with conditions) | 2026-09-18 |
| DEV-005 | APPROVED | IMPLEMENTATION_DETAIL — frame-selection tie/collision rule | 2026-09-18 |
| DEV-006 | APPROVED | Q-04 valid frame + index-bounded decode; index continuity proven (DECODER_INDEX_AUDIT.md) | 2026-09-18 |
| DEV-007 | APPROVED | APPROVED_WITH_EVIDENCE — CASIA derived copies excluded | 2026-09-18 |
| DEV-008 | UNAPPROVED | NOT APPROVED — superseded by DEV-010 | 2026-09-18 |
| DEV-010 | APPROVED | CASIA code semantics + subject ids train N / test 20+N (owner decision) | 2026-09-18 |
| DEV-009 | APPROVED | M0 record correction (SiW lowercase dirs) | 2026-09-18 |
| DEV-011 | APPROVED | APPROVED_CONTROLLED_DATASET_ADAPTATION — CASIA pre-cropped 112→256 INTER_CUBIC, SCRFD N/A | 2026-09-19 |
| DEV-012 | APPROVED | SiW-Mv2 video-disjoint split fallback (content-group, exact-byte sha256) — subject IDs unavailable | 2026-09-19 |
| DEV-013 | APPROVED | SiW-Mv2 pairing fallback: different video AND different content group (not different-person) | 2026-09-19 |
| DEV-014 | APPROVED | AdaFace runs on the frozen canonical face; no MTCNN/landmark alignment stage (owner decision Q-19) | 2026-09-19 |
| DEV-015 | APPROVED | FaceXFormer runs on the frozen canonical face; no MTCNN 50%-margin recrop (closes Q-20) | 2026-09-19 |
| DEV-016 | APPROVED | Q-22 operational interpretation of "clamp to image": the requested square is preserved with zero padding; clamping applies to the source read region | 2026-09-19 |
| DEV-017 | APPROVED | APPROVED_OPERATIONAL_STORAGE_RELOCATION — M2 generated artifacts physically stored on /media/cong/Data; no scientific change | 2026-09-19 |

---

## DEV-001 — Project root path differs from spec §0.2

- **Spec says (§0.2 "Fixed project root and directory contract"):** `D:\AI on IOT\Anti_spoofing\GPAT_TransferBench` (Windows path).
- **Implemented:** `/home/cong/GPAT_TransferBench` (Linux laptop), per explicit owner instruction for this M0 session. The planned GPU mirror is `/home/sparc/workdir/longnm/GPAT_TransferBench` (not created in M0).
- **Observation:** the Windows `D:` volume appears to correspond to `/media/cong/Data` on this laptop (the spec file and an `AI on IOT/Anti_spoofing/` tree exist there). This is an inference, not verified. No `GPAT_TransferBench` directory exists there.
- **Directory contract inside the root:** preserved (all §0.2 subdirectories created, including `data/raw/{casia_fasd,msu_mfsd,siwmv2}` and `data/processed/{frames,faces_256}`).
- **Scientific impact:** none expected (path only); all manifests must use root-relative paths so the tree is portable.
- **Affected experiments:** none directly.
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: INFRASTRUCTURE_ADAPTATION. Linux root `/home/cong/GPAT_TransferBench` replaces the Windows root example of spec §0.2. No change to the scientific protocol.
  - Supersedes the UNAPPROVED line above (kept for history).

## DEV-002 — Additional directories/files beyond the §0.2 contract

- **Added:** `models/` (spec §4 requires `models/registry.yaml`), `frozen_config_snapshot/` (spec §0.1 rule 1), `gpatbench/` (spec §24 CLI `python -m gpatbench.cli`), `environments/`, `outputs/exploratory/`, `docs/spec/`, `tools/`, `configs/data_source_registry.yaml`, `configs/CONFIG_STATUS.md`.
- **Reason:** infrastructure required elsewhere in the spec or by the owner's M0 instructions. Purely additive; no spec directory removed or renamed.
- **Scientific impact:** none.
- **Affected experiments:** none.
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: INFRASTRUCTURE_ADAPTATION. Additional implementation/audit folders (e.g. `models/`, `environments/`, `gpatbench/`, `docs/`, `tools/`) are allowed provided no spec-mandated structure is removed or changes semantics.
  - Supersedes the UNAPPROVED line above (kept for history).

## DEV-003 — Spec internal inconsistency: `lambda_dir` (SPEC_INTERNAL_INCONSISTENCY / SOURCE_GAP)

- **Where:** §23.2 canonical GPAT-B0 YAML contains `lambda_dir: 0.5`.
- **Conflict:** no `λ_dir` term appears in the §10.3 loss-weight table nor in the §10.3 `L_G` formula. §10.2 `L_spec` contains an internal `0.5·||S_orient(x_hat) − S_orient(x_s)||_1` coefficient; whether `lambda_dir` denotes that coefficient or a separate loss is **not stated**. This session does not guess.
- **Action at M0:** `configs/methods/gpat_b0.yaml` is kept byte-verbatim from the spec (including `lambda_dir`). No code interprets it yet.
- **Affected experiments:** E08–E11 and ablations §17.1/§17.2 (M7).
- **Status:** UNAPPROVED / OPEN — must be resolved by the owner before M7; otherwise M7 is BLOCKED_BY_SOURCE_GAP for this term.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** UNAPPROVED
  - Owner classification: **UNRESOLVED**. Not approved and must not be resolved by the agent. The semantics and loss formula of `lambda_dir` must be traced before GPAT implementation/training (M7). Does not block M0.

## DEV-004 — M0 "environment locks": project environment not yet created

- **Spec M0 gate (§25):** "all dependencies/version hashes recorded; no method trained".
- **State:** no project Python environment exists yet. Laptop system Python 3.12.3 has no pip, and torch/torchvision/pandas/pyarrow/onnxruntime/scipy/scikit-learn/ptwt/transformers/timm are MISSING. GPU environment unknown (SSH needs interactive password). Therefore no environment lock hash can honestly be recorded.
- **What was recorded instead:** raw laptop state (`environments/laptop_environment_initial.txt`, SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`) and the planned requirement list (`environments/PLANNED_ENVIRONMENT_REQUIREMENTS.md`) with every dependency marked AVAILABLE / MISSING / VERSION_UNVERIFIED / NOT_REQUIRED_AT_M0. No version hash is claimed for anything not installed.
- **Affected experiments:** all (a lock is required before any run, §0.1 rule 8).
- **Status:** UNAPPROVED — owner to accept this interpretation of the M0 gate or require environment creation first.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: APPROVED_OPERATIONAL_DEVIATION (APPROVED_WITH_CONDITIONS).
  - Conditions:
    1. Dependencies that are not installed stay MISSING.
    2. Dependencies whose version is not verified stay VERSION_UNVERIFIED; nothing may be reported as VERIFIED without evidence.
    3. The environment a milestone needs must be created and frozen (lock + SHA256) before that milestone's scientific execution.
    4. The current laptop environment audit (`environments/laptop_environment_initial.txt`, `LAPTOP_ENVIRONMENT_AUDIT.md`) is kept unchanged as historical evidence.

---

## Open spec questions (not deviations yet — flagged for the milestone that needs them)

These were noticed while reading the full spec. Nothing below has been decided; each must be
resolved from the spec/source hierarchy or by the owner before the listed milestone, or be
marked BLOCKED_BY_SOURCE_GAP.

| ID | Milestone | Spec § | Question |
|---|---|---|---|
| Q-01 | M3 | 3.5 | Allocator objective names the terms (video-count deviation + penalties for missing binary class and attack_macro coverage) but not the penalty weights or search procedure (exact/greedy/random restarts). |
| Q-02 | M2 | 4 | SCRFD variant is not named (only "SCRFD ONNX, input 320"). Local candidates include `scrfd_10g_bnkps.onnx` and `det_2.5g.onnx`. |
| Q-03 | M2 | 4 | AdaFace IR-50 training-set variant (e.g. which pretraining dataset checkpoint) not named. |
| Q-04 | M1/M2 | 3.4 | Definition of a "valid frame" (decode success only, or face-detected) is not stated. |
| Q-05 | M7 | 5.3, 9.2, 10.3 | Identity adversary uses a GRL (coef 1.0) **and** `L_G` subtracts `λ_idadv·L_identity-adversary`; applying both could double-reverse the gradient. Intended sign convention needs confirmation. |
| Q-06 | M11 | 14.2 | DINOv3 table does not state warmup, scheduler/min LR, AMP or grad clip; only batch composition, augmentation, checkpoint/threshold are "same as ResNet-18". |
| Q-07 | M5 | 12.1 | ArtifactProbeNet input is 224×224 from 256×256 faces; resize vs crop method not stated. |
| Q-08 | M6 | 8.5, 30 | PCGAN (R7) has no code link in spec; status depends on code availability at implementation time. |
| Q-09 | M6 | 8.4, 30 | Physics-Guided STD (R2) has no code link; FAITHFUL_PAPER unless a verified official release is found. |
| Q-10 | M9 | 13 | LPIPS-Alex and KID (Inception feature extractor) implementations/weights not named. |
| Q-11 | M1 | 0 | Datasets are ≥ 3 local sources with mixed packaging (see `configs/data_source_registry.yaml`); whether raw data is referenced in place (read-only external path) or placed under `data/raw/` is not stated. |

---

## M1 entries (2026-09-18)

## DEV-005 — M1 frame-selection interpretation (spec §3.4 "nearest unique valid frame indices")

- **Spec:** positions fixed; "choose the nearest unique valid frame indices"; the collision rule is not stated.
- **Implemented (data_v1.yaml `frame_sampling.selection_rule`):** target_k = lo + p_k·(hi − lo) over the valid index range; for each k in order, take the nearest valid index not already chosen; ties go to the smaller index. With fewer than 8 valid frames, every valid frame is used once.
- **Scientific impact:** only matters when two targets share a nearest index. The number of affected videos is measured by the inventory (`SAMPLING_COLLISION_RESOLUTION_USED` issues).
- **Affected experiments:** all (sample identity).
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner M1 review): NOT APPROVED YET.** The exact rule is documented below. The implementation is unchanged because it does not violate the frozen wording.
  - **Exact rule** (`gpatbench/data/base.py:select_frame_indices`):
    - Let V be the sorted unique valid indices, lo = V[0], hi = V[-1], and P = the 8 frozen positions.
    - If |V| < 8, return V (every valid frame once).
    - Otherwise, for k = 1..8 in order: t_k = lo + p_k·(hi − lo) as a float. Pick the element of V minimising (|i − t_k|, i), so ties go to the smaller index.
    - If that element was already chosen for an earlier position, pick instead the element of V \ chosen minimising (|i − t_k|, i).
  - **Boundaries:** p ∈ [0.10, 0.90], so every target lies strictly inside [lo, hi] and never needs a frame outside the valid range. With |V| ≥ 8 the fallback always finds an unused index.
  - **Against the frozen wording "choose the nearest unique valid frame indices":** the output is always 8 unique valid indices, and each is the nearest valid index to its target unless that index was taken by an earlier target. The wording does not say which target keeps a contested index; this rule gives it to the earlier (lower-p) target.
  - **Measured:** the collision fallback was used for **0** videos in the final inventory (`SAMPLING_COLLISION_RESOLUTION_USED` issues = 0). With |V| ≥ 10 the targets are at least 1.03 indices apart.
- **Status update 2026-09-18 (dataset-resolution pass, owner decision):**
  - **Status:** APPROVED
  - Owner classification: IMPLEMENTATION_DETAIL. Approved rule exactly as documented above: lower index wins exact ties; the earlier p_k keeps a contested frame; the next nearest unused index uses the same key. The frozen positions stay exactly [0.10, 0.2142857, 0.3285714, 0.4428571, 0.5571429, 0.6714286, 0.7857143, 0.90]. Measured collision fallback usage: 0 videos. Toy tests: `tests/test_dataset_resolution.py`.

## DEV-006 — Q-04 valid-frame operational definition (decoder robustness)

- **Implemented (data_v1.yaml `valid_frame`):**
  - image sequences: valid iff the file decodes to a non-empty image.
  - videos: 0-based decode positions with pinned opencv-python-headless 5.0.0.93 / CAP_FFMPEG, one position per `read()`. Position i is valid iff `read()` returns a non-empty frame. A failed read marks that single position invalid and decoding continues. The stream ends after 5 consecutive failed reads. The declared frame count is recorded but not trusted.
- **Why "continue", not "stop at first failure":** in run A (superseded), MSU `attack_client008_laptop_SD_ipad_video` decoded 127 good, 1 failed, then 172 good frames (127+1+172 = 300 declared). `attack_client023_laptop_SD_iphone_video` decoded 247 + 1 + 53 = 301. Stopping at the first failure would have discarded 57% and 18% of those videos and shifted their sampling range. A failed read consumes exactly one position, which is consistent with the declared counts.
- **Other evidence:**
  - declared ≠ decoded at the end of stream (137 SiW-Mv2 videos short by 2–5 frames; MSU as above);
  - FFmpeg error-concealed frames are logged per video (`decoder_error_lines`) and raised as WARNING issues.
- **Limitation:** a run of ≥5 consecutive failed reads inside a stream would be treated as end of stream. Such cases would show `n_valid + n_invalid < declared` and are visible in `inventory_videos.parquet`.
- **Scientific impact:** decides which frame indices are sampleable; frame content is untouched. M2 must reuse the pinned decoder or re-verify the indices.
- **Status:** UNAPPROVED — owner review required. This is the proposed Q-04 resolution.
- **Status update 2026-09-18 (owner M1 review):** the semantic definition is APPROVED by the owner: "a valid frame is an original frame/index that successfully decodes into a non-empty image using the pinned decoder/backend". The "stop after 5 consecutive failures" rule was **not accepted** and has been **replaced** (revision 2, `gpatbench/data/frames.py`, data_v1 revision 2):
  - **videos:** index-bounded over the container-declared range [0, N_declared). One sequential `read()` per original index; a failure marks that index invalid, the index is preserved, and decoding continues. One extra `read()` after the range is recorded as evidence only (`frames_beyond_declared`, never sampled). If N_declared ≤ 0: `DECLARED_COUNT_UNAVAILABLE`, ERROR, no valid frames.
  - **image sequences:** index = existing source file's frame number; valid iff it decodes non-empty.
  - **Justification that read() k ↔ original index k:** in run A, MSU client008 decoded 127 good + 1 failed + 172 good = 300 = declared, and client023 247 + 1 + 53 = 301 = declared. A failed read therefore consumes exactly one position.
  - **Status:** UNAPPROVED — the semantic definition is APPROVED by the owner; the index-bounded implementation (revision 2) awaits owner confirmation. Measured outcomes are in the final inventory (`decode_status`, `frames_beyond_declared`).
- **Status update 2026-09-18 (dataset-resolution pass):**
  - **Status:** APPROVED
  - Evidence: `DECODER_INDEX_AUDIT.md`, `decoder_index_audit.csv`, `decoder_index_global.csv`.
    - The independent decoder (PyAV 18.1.0) fails on the same packets (MSU client008 @127, client023 @247).
    - Every later frame content-aligns at offset 0 (thumbnail diff 0.000).
    - All 1,980 videos have OpenCV timestamps equal to PyAV packet PTS for every valid index.
    - The 137 SiW trailing invalid indices equal declared − packets exactly: they are non-existent frames.
  - Caveat recorded for M2: OpenCV `CAP_PROP_POS_FRAMES` lags by 1 after a failed read and must never be used as the frame index.
  - Owner-approved semantic definition, with the implementation now proven.

## DEV-007 — CASIA-FASD derived copies excluded from samples

- `train/live/fs*` files (exact horizontal flips) and `bs*` files (brightened copies) of canonical frames are indexed as DERIVED_AUGMENTATION_COPY and are **not** sampled. Evidence: `m1_inventory_facts.json` and `M1_DATASET_EVIDENCE.md` §1.
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner M1 review):**
  - **Status:** APPROVED
  - Owner classification: APPROVED_WITH_EVIDENCE. `bs*`/`fs*` files stay in the raw file index and are excluded from canonical sampling. Evidence: `fs` = exact horizontal flip (6337/6337); `bs` = brightness derivative.

## DEV-008 — CASIA-FASD subject identity = native partition + number

- `subject_id_raw = {train|test}_s{N}`. The bare number is shared by different people across the native partitions (visual check of all 20 overlapping numbers).
- **Status:** UNAPPROVED — owner review required. This rule decides CASIA subject-disjointness in M3.
- **Status update 2026-09-18 (owner M1 review): NOT APPROVED — SUPERSEDED by DEV-010.** The `{partition}_s{N}` identity is withdrawn. The visual evidence that train sN ≠ test sN remains valid and is consistent with DEV-010.

## DEV-009 — Correction of an M0 record (SiW-Mv2 lowercase directories)

- M0 `data_source_registry.yaml` / report said SiW-Mv2 had both `Live/Spoof` and `live/spoof`. This was a misread of two concatenated `ls` outputs. The lowercase entries belonged to CASIA `train/`. Corrected in M1 with evidence (`M1_DATASET_EVIDENCE.md` §3). The M0 text is kept for history.
- **Status:** UNAPPROVED (record correction; no scientific impact).
- **Status update 2026-09-18 (owner M1 review):**
  - **Status:** APPROVED
  - The M0 observation error is acknowledged. The M0 evidence and ledger rows are preserved; this correction is appended.

## Open data questions raised in M1 (owner decisions needed)

| ID | Blocks | Question |
|---|---|---|
| Q-12 | M3 (CASIA), Native track | CASIA `HR_1` videos are stored under `spoof/` locally but are genuine per the published protocol and look live. Which label is authoritative? |
| Q-13 | Native/Full track; M3 attack_macro coverage | Approve or reject the CASIA code → attack_macro proposals and SiW `Paper` → print (see `attack_map_v1.yaml` `unmapped_pending_approval`). |
| Q-14 | M3 (SiW-Mv2) | SiW-Mv2 subject IDs are not in the local copy. Can the official per-sample naming or subject list be obtained? If not, spec §3.5 blocks the main split for SiW-Mv2 (`BLOCKED_BY_MISSING_SUBJECT_ID`). A `protocol_v1_video_fallback` is allowed only as a separately named protocol. |
| Q-15 | M2 (CASIA) | The local CASIA copy holds 112×112 face crops, not original frames. How should spec §4 (SCRFD on the frame, 1.25× crop, 256 master) apply? |
| Q-16 | M3 (SiW-Mv2) | 6 pairs of SiW-Mv2 Replay videos are byte-identical (e.g. `Replay_76.mov` = `Replay_83.mov`; full list in `dataset_duplicate_hashes.csv`). Both copies are kept and inventoried as separate canonical videos; nothing was merged or deleted. How should M3 treat them (keep both, keep one, or force them into the same split)? |

---

## M1 correction pass (2026-09-18) — new entries

## DEV-010 — CASIA-FASD code semantics and subject identities (supersedes DEV-008)

- **Source:** owner-provided external CASIA-FASD protocol evidence (M1 review, 2026-09-18). Code semantics:
  - live: 1 real_normal, 2 real_low, HR_1 real_high;
  - print: 3 warped_normal, 4 warped_low, HR_2 warped_high, 5 cut_normal, 6 cut_low, HR_3 cut_high;
  - replay: 7 video_normal, 8 video_low, HR_4 video_high.
- **Implemented:**
  - `label_binary` is taken from the code (HR_1 = live). The local folder label is overridden where it disagrees (HR_1 under `spoof/`) and each override is logged as a `FOLDER_LABEL_OVERRIDDEN` issue.
  - `attack_raw` = code for spoof videos only.
  - `subject_id_raw`: train sN → "N", test sN → str(20+N). A number outside train 1..20 / test 1..30 is a hard error.
- **Verification against all 600 local sequences** (before re-inventory): train has s1..s20 and test has s1..s30 exactly; every subject has all 12 codes; the mapping yields 50 unique ids 1..50. The only disagreement with local evidence is the HR_1 folder placement.
- **Status:** APPROVED (owner decision, recorded 2026-09-18).

## Resolution of M1 open questions (2026-09-18)

| ID | Resolution | Trace |
|---|---|---|
| Q-12 | **RESOLVED** — CASIA HR_1 = real_high → LIVE | Owner-provided CASIA protocol evidence; DEV-010 |
| Q-13 (CASIA) | **RESOLVED** — 3,4,HR_2,5,6,HR_3 → print; 7,8,HR_4 → replay; 1,2,HR_1 live (no attack_raw) | SPEC §3.3 + owner protocol evidence; `attack_map_v1.yaml` rev 2 |
| Q-13 (SiW `Paper`) | **RESOLVED** — `Paper` → print | Official code github.com/CHELSEA234/Multi-domain-learning-FAS @ `8667dbcd316b38141729c057adf7517fe0602608`: `source_SiW_Mv2/config_siwm.py` L63–69 `spoof_type_dict` has `'Print': 'Paper'` and separately `'Paper': 'Mask_Paper'` (sha256 ebe37e87…); `csv_parser.py` L145–177 routes `Paper` videos to `print_list` after excluding `Mask_Paper` and `Partial_Paperglass` (sha256 59159f5d…); official README folder list has both `Print` and `Mask_PaperMask`; ECCV'22 supplementary Table 1 (sha256 b9dba0ff…): Print = 135 videos = local `Paper`; Paper Mask = 17 = local `Mask_PaperMask` |
| Q-14 | **STILL BLOCKED** — no video→subject mapping found | Checked: local README.pdf/DRA.pdf; official repo @8667dbc (README, protocol lists `pro_3_text/*.txt` and FASMD `SIWM_list/*.txt` list video names only, which the README calls "subject names", repeated "for balancing"; `combine_label_illu.csv` is SiW v1 naming); ECCV'22 supplementary (per-type subject counts only); `Dataset request/` (agreement forms only); PRISM archive `FULL_SIW_PHYSICAL_AUDIT.json` (`all_frame_subject_null: true`). → `BLOCKED_BY_MISSING_SUBJECT_ID` for the SiW-Mv2 M3 main split |
| Q-15 | **UNRESOLVED** — no original CASIA-FASD video copy found locally | Read-only search of /home/cong and /media/cong/Data: no CASIA `.avi`, no `train_release`/`test_release`, no `HR_*` video files. Copies found are all the same 112px PNG repack: the selected root, `casia-fasd.zip` (CRC-identical), and `PRISM_FAS_C_LLM_Project/data/raw/casia_fasd` (123,533 PNG; 20/20 spot-check byte-identical). M2 CASIA needs an explicit owner deviation decision |
| Q-16 | **OPEN (no action in M1)** — 6 byte-identical SiW Replay pairs kept as separate raw entries | `dataset_duplicate_hashes.csv`; re-check against subject/split boundaries once subject metadata exists |

Note: supplementary Table 1 names the 72-video mask type "Full Mask", while the local folder is `Mask_HalfMask` (spec concept "half mask"). The mapping to mask_3d is unaffected; recorded for traceability.

---

## Dataset-resolution pass (2026-09-18) — new entries

## DEV-011 — PROPOSED: CASIA_PRE_CROPPED_112_ADAPTATION (CONTROLLED_DATASET_ADAPTATION)

- **Why:** the only local CASIA-FASD data are 112×112 face crops (`CASIA_SOURCE_AUDIT.md`: 110,859/110,859 canonical PNGs are 112×112 RGB; no original copy exists on the laptop). The spec §4 SCRFD raw-frame path cannot be executed for CASIA.
- **Proposal (full text: `CASIA_CONTROLLED_ADAPTATION_PROPOSAL.md`):** decode the canonical 112 crop → RGB → INTER_CUBIC to 256×256; SCRFD/1.25× crop skipped for CASIA only; identical for all methods.
- **Measured impact** (`CASIA_RESIZE_FREQUENCY_AUDIT.md`, diagnostic): after upscaling, relative high-frequency power on the 256 grid drops by about 5–6×; live and spoof are affected alike. **Not equivalent** to the nominal pipeline.
- **Affected experiments:** every experiment that uses CASIA (all of them, through the pooled benchmark).
- **Status:** UNAPPROVED — owner decision required before M2 for CASIA.
- **Status update 2026-09-19 (owner decision, dataset policy freeze):**
  - **Status:** APPROVED
  - Owner classification: APPROVED_CONTROLLED_DATASET_ADAPTATION. The CASIA M2 path is the canonical 112×112 RGB crop → INTER_CUBIC 256×256 uint8 sRGB → common pipeline. SCRFD is not applied (`scrfd_applied=false`, SCRFD status N/A, never reported as detector success).
  - Fairness conditions and required disclosures are frozen in `configs/frozen/dataset_protocol_policy_v1.yaml`.
  - Not equivalent to the nominal §4 path; the frequency audit is preserved as evidence.

## Final classification of dataset questions (dataset-resolution pass)

| ID | Final status | Evidence |
|---|---|---|
| Q-04 | RESOLVED (DEV-006 APPROVED) | `DECODER_INDEX_AUDIT.md` |
| Q-14 | **BLOCKED_BY_MISSING_SUBJECT_ID** (CASE C: no trustworthy mapping; 0/1,700 videos). Official protocol entries are video stems (`preprocessing.py` L31 → `config_siwm.py` L162–164), not person IDs. The AdaFace audit was not applicable (0 source-derived candidate groups) and was not run; no pseudo IDs were created | `SIW_PROTOCOL_NAME_SEMANTICS.md`, `siw_subject_mapping_candidate.csv`, `siw_subject_recovery/ADAFACE_AUDIT_STATUS.md` |
| Q-15 | **UNRESOLVED → DEV-011 proposed**. Final read-only search found no original CASIA copy (every zip ≥ 100 MB on the data volume listed; the 76-part CelebA-Spoof archive not listable; one unrelated .rar not opened; the GPU server not inspected) | `CASIA_SOURCE_AUDIT.md` |
| Q-16 | **OPEN, preserved**. 6 byte-identical SiW Replay pairs, all UNRESOLVED_DUPLICATE; 3 pairs sit in official testlist_all vs trainlist_all | `SIW_DUPLICATE_VIDEO_AUDIT.md/.csv` |
| Q-17 (new) | **INFO**. The official SiW-Mv2 protocol lists reference 1,764 stems, 64 of which are absent from the local release (e.g. `Live_892…902`); the official lists cannot be applied verbatim to the local copy | `SIW_PROTOCOL_NAME_SEMANTICS.md` §3 |

---

## Final dataset policy freeze (2026-09-19) — new entries

## DEV-012 — SiW-Mv2 video-disjoint fallback due to unavailable subject identity

- **Original frozen behaviour (spec §3.5):** subject-disjoint split; a dataset without a recoverable subject ID blocks the main split (`BLOCKED_BY_MISSING_SUBJECT_ID`); a video fallback is allowed only as a separately named protocol.
- **Approved behaviour:** SiW-Mv2 is split inside the dataset by deterministic **canonical-video-disjoint** allocation.
  - The grouping unit is `content_group_id = siwmv2::sha256:<raw video sha256>`: videos with identical raw bytes share one unit and always land in the same split.
  - All sampled frames inherit the video's split; frame-level splitting is forbidden.
  - Ratios are 70/15/15 by canonical-video count, with seed 20260814.
  - Balancing priorities follow `dataset_protocol_policy_v1.yaml`.
  - CASIA-FASD and MSU-MFSD stay subject-disjoint.
- **Evidence:** Q-14 investigation (`SIW_PROTOCOL_NAME_SEMANTICS.md`, `siw_subject_mapping_candidate.csv`): no trustworthy video→person mapping exists. Duplicates: `SIW_DUPLICATE_VIDEO_AUDIT.md`, `siw_content_groups.csv` (1,700 videos → 1,694 content groups).
- **Scientific limitation (must never be hidden):** the same physical person may appear in more than one SiW-Mv2 split. The pooled benchmark is not uniformly subject-disjoint. This replaces the main-protocol rule of §3.5 for SiW-Mv2; it is not a separately named side protocol.
- **Affected:** M3 (split), M4 (pairs, see DEV-013), M5–M13 (every model trained or evaluated on the pooled split), downstream pooled evaluation, T01/T14 reporting, paper limitations.
- **Status:** APPROVED (owner decision, 2026-09-19).

## DEV-013 — SiW-Mv2 different-video pairing fallback due to unavailable subject identity

- **Original frozen behaviour (spec §6):** the target live candidate comes from the same dataset and a **different subject**.
- **Approved behaviour for SiW-Mv2:** `source_video_id != target_video_id` **and** `source_content_group_id != target_content_group_id`. CASIA and MSU keep different-subject pairing.
- **Scientific limitation:** SiW-Mv2 pairs are different-video / different-exact-content pairs, **not proven different-person pairs**, and must never be described as such. The GPAT identity-preservation metric (AdaFace cosine target vs synthetic) is unaffected, because it needs no dataset subject label.
- **Affected:** M4 pair manifests, M8 banks, M9 pair metrics, the fingerprint probe's GroupKFold grouping (§12.2 groups by subject: SiW needs the same fallback key, to be set in the M5/M9 contract), and the paper.
- **Status:** APPROVED (owner decision, 2026-09-19). Recorded only; no pairs built.

## Latest status of dataset questions (policy freeze)

| ID | Latest status |
|---|---|
| Q-04 | RESOLVED (DEV-006 APPROVED) |
| Q-12 | RESOLVED (HR_1 = live; DEV-010) |
| Q-13 | RESOLVED (CASIA codes; SiW Paper → print) |
| Q-14 | **SUBJECT_ID_UNAVAILABLE_ACCEPTED_WITH_VIDEO_DISJOINT_DEVIATION** (DEV-012/013); no pseudo IDs |
| Q-15 | **RESOLVED_WITH_CONTROLLED_ADAPTATION** (DEV-011 APPROVED) |
| Q-16 | **RESOLVED_WITH_EXACT_CONTENT_GROUPING** (6 groups; raw records preserved) |
| Q-17 | DOCUMENTED LIMITATION: 64 official protocol names absent locally (`SIW_PROTOCOL_COVERAGE_AUDIT.md`); local inventory is the source of truth |
| Q-01 | **RESOLVED_BY_OWNER_LEXICOGRAPHIC_ALLOCATOR** (2026-09-19); frozen in configs/frozen/split_v1.yaml |

---

## M2A — auxiliary model resolution (2026-09-19)

No new deviation is proposed in M2A. The items below are open questions for the owner. Implementation details are listed separately; they are not scientific choices.

| ID | Status | Blocks M2B | Summary (full text: `M2A_AUX_MODEL_RESOLUTION.md`) |
|---|---|---|---|
| Q-02 | **OWNER_DECISION_REQUIRED** | YES | Two official SCRFD KPS candidates, identified structurally: A = SCRFD_10G_KPS (`scrfd_10g_bnkps.onnx`, sha `5838f7fe…`), B = SCRFD_2.5G_KPS (`det_2.5g.onnx`, sha `041f73f4…`). The spec does not name a variant, and neither file is hash-verified against an official pack (downloads of 407 MB / 313 MB not performed). Labelled recommendation: A |
| Q-03 | **OWNER_DECISION_REQUIRED** (local identity RESOLVED_AUTHORITATIVE) | YES | Local = HF `minchul/cvlface_adaface_ir50_webface4m` @ `60a65bef` (sha `43bd2d57…`, exact). Alternatives: CVLFace IR-50 CASIA / MS1MV2 (RGB) or original-repo IR-50 .ckpt (BGR) |
| Q-18 | OPEN (new) | YES | AdaFace colour order: spec says "BGR"; the local checkpoint's official contract is RGB. Smoke cos(RGB-input, BGR-input embeddings) = 0.81–0.97 |
| Q-19 | OPEN (new) | YES | AdaFace needs a 5-point-aligned 112×112 face; the canonical face is an unaligned 256 crop; the spec has no alignment step |
| Q-20 | OPEN (new) | NO | FaceXFormer official demo framing (MTCNN + 50% margin) vs spec App. A canonical faces (implemented as spec) |
| Q-21 | OPEN (new) | NO for M2B; YES before M9 / §21 | FaceXFormer's 11 parsing classes have no documented names |
| Q-22 | OPEN (new) | YES | Crop borders: "square padded face" vs "clamp to image". Smoke: 3/8 SiW crops clamped |
| Q-23 | OPEN (new) | YES | Parsing-logit cache: float32 ≈ 45.6 GB for 20,640 samples; dtype/resolution decision |

**Implementation details (recorded, not scientific choices):**
- SCRFD NMS 0.4 (official default); official letterbox and normalisation;
- largest-face tie-break;
- crop integer rounding;
- PNG compression level 3;
- FaceXFormer swin_b constructed with `weights=None` and then a strict checkpoint load (identical final weights, no ImageNet download);
- FaceXFormer one forward per task;
- landmark 224→256 pixel-centre mapping;
- deterministic threading (torch 4, ORT 1).

---

## M2A — owner decision pass (2026-09-19)

Full text and evidence: `M2A_OWNER_DECISIONS.md`. The M2A investigation table above the previous section is
kept as history; this section records the resolutions and the three new deviations they make explicit.

| ID | Owner decision | Final status |
|---|---|---|
| Q-02 | SCRFD_10G_KPS `scrfd_10g_bnkps.onnx` (candidate A) | RESOLVED_BY_OWNER_SELECTION; **OFFICIAL_BYTE_HASH_CONFIRMED** against `antelopev2.zip` |
| Q-03 | original `mk-minchul/AdaFace` R50 / WebFace4M checkpoint | RESOLVED_BY_OWNER_SELECTION |
| Q-18 | BGR | RESOLVED_BY_OWNER |
| Q-19 | canonical-face adapter, no extra alignment | RESOLVED_BY_OWNER_CANONICAL_FACE_ADAPTER (DEV-014) |
| Q-20 | canonical face is the FaceXFormer source image | CLOSED (DEV-015) |
| Q-21 | class names not needed for M2B | **OPEN** — required before M9 / §21 region crops |
| Q-22 | requested square preserved with zero padding | RESOLVED_BY_OWNER (DEV-016) |
| Q-23 | full float32 logits + uint8 argmax mask, lossless storage | RESOLVED_BY_OWNER_FLOAT32_FULL_LOGITS |

## DEV-014 — AdaFace identity cache omits the official MTCNN alignment stage

- **Spec basis:** §4 names AdaFace IR-50 with its official BGR/input normalisation, and App. A defines the
  identity cache on the canonical `faces`. It defines no alignment step.
- **Deviation:** the official AdaFace inference pipeline aligns an arbitrary photograph with MTCNN 5-point
  alignment before the 112×112 crop. This benchmark does not run that stage. The frozen canonical 256×256
  face is resized to 112×112 with INTER_AREA and fed directly.
- **Reason:** the benchmark must score real and *synthetic* 256×256 images in one common geometry frame.
  Adding an aligner would introduce a second face model into the identity path, a transform with no spec
  basis, and a stage that generated images can fail. The owner decided the canonical face is that frame.
- **Affected:** every identity-based number (ID cosine, ID retention). They are **not comparable** with
  published AdaFace verification benchmarks; this must be disclosed wherever identity results are reported.
- **Not affected:** the choice is applied identically to every method and every dataset.
- **Status: APPROVED** (owner, 2026-09-19). Classification: IMPLEMENTATION_ADAPTER_DEVIATION.
- **Enforcement:** the adapter has no alignment code path; tests assert that no MTCNN/alignment/detector
  module is imported and that exactly one resize (112×112, INTER_AREA) occurs.

## DEV-015 — FaceXFormer geometry cache omits the official demo's MTCNN margin recrop (closes Q-20)

- **Spec basis:** App. A defines the geometry cache on the canonical `faces`.
- **Deviation:** the official FaceXFormer demo crops an MTCNN box with a 50 % margin before resizing to 224.
  The benchmark feeds the frozen canonical face instead.
- **Reason:** identical to DEV-014 — one common frame for real and synthetic images, no extra face model.
- **Affected:** parsing/landmark/pose caches; applied identically everywhere.
- **Status: APPROVED** (owner, 2026-09-19). Classification: IMPLEMENTATION_ADAPTER_DEVIATION.

## DEV-016 — Operational interpretation of spec §4 "clamp to image" for the 1.25× square crop

- **Spec basis:** §4 requires a "Square **padded** face … side = 1.25×max(w,h); clamp to image; no random
  padding". Taken literally, clamping the *output* region makes the crop non-square, and resizing a
  non-square region to 256×256 distorts the aspect ratio — which contradicts "square" in the same row.
- **Interpretation (owner):** "clamp" applies to the SOURCE READ region. The requested square is
  constructed first, is never shrunk or shifted, and every requested pixel outside the image is filled with
  constant **zero**. No reflect/replicate border and no random padding, so "no random padding" is honoured.
- **Affected:** only crops whose square extends past a frame edge. Measured on the frozen 24-sample smoke:
  3/8 SiW, 0/8 MSU, 0/8 CASIA (CASIA does not use SCRFD). The previous provisional rule produced different
  canonical faces for exactly those 3 samples and identical faces for the other 21
  (`M2A_CONTRACT_CHANGE_IMPACT.json`).
- **Status: APPROVED** (owner, 2026-09-19). Classification: OWNER_RESOLVED_IMPLEMENTATION_INTERPRETATION.
- **Evidence:** `M2A_BORDER_CASES.csv` (real samples, with pads and pre-resize shapes) and exhaustive
  synthetic unit tests (no contact, four edges, both corners, square larger than one source dimension).

---

## DEV-017 — M2 generated artifacts physically stored outside the project filesystem

- **Classification:** `APPROVED_OPERATIONAL_STORAGE_RELOCATION` (infrastructure; not a scientific deviation).
- **Spec basis:** §0.2 fixes a *directory contract* with `data/processed/frames`, `data/processed/faces_256`,
  `cache/geometry` and `cache/identity`. It names logical artifact roles; it does not bind them to a
  filesystem. DEV-001 already approved a project root that differs from the spec's literal path.
- **Reason:** the M2B storage preflight (`M2B_STORAGE_PREFLIGHT.md`) measured that the project filesystem
  (`/dev/nvme0n1p7`, ext4, `/home`) has 36.50 GiB free while M2B needs 63.92 GiB — a 27.42 GiB shortfall —
  and returned `BLOCKED_BY_STORAGE_CAPACITY`. The owner then approved relocating the physical storage.
- **Deviation:** the four M2 output roles are written under the owner-named runtime root

  ```
  /media/cong/Data/GPAT_TransferBench_runtime/data/processed/frames
  /media/cong/Data/GPAT_TransferBench_runtime/data/processed/faces_256
  /media/cong/Data/GPAT_TransferBench_runtime/cache/geometry
  /media/cong/Data/GPAT_TransferBench_runtime/cache/identity
  ```

  The Git repository stays at `/home/cong/GPAT_TransferBench` and remains authoritative for code,
  configs, manifests, tests, tools, audit records and hashes. No project directory was replaced by a
  symlink; the physical roots come from an execution config
  (`configs/execution/m2b_laptop_external_storage.yaml`), which is infrastructure only and is refused by
  the runner if it collides with a frozen scientific field or escapes the approved runtime root.
- **Affected scientific protocol: NONE.** The frozen `configs/frozen/preprocess_v1.yaml` is unchanged
  (its sha256 is asserted by the runner before execution). Scientific identity is `sample_id` plus content
  hashes; no scientific value depends on an absolute path string, so moving these artifacts again would
  not change any sample identity.
- **Evidence that relocation changed nothing:** the frozen 24-sample M2A smoke manifest (sha256
  `138f5929…`, not reselected) was re-run writing to the new filesystem and compared against the frozen
  `/home` run: **210/210 files byte-identical**, results equal except timing
  (`M2B_EXTERNAL_SMOKE_COMPARE.json`). The deterministic final validation repeats this against the
  finalized full-M2 artifacts.
- **Operational caveats recorded honestly:** the volume is `ntfs3`, not ext4 (verified to support fsync,
  atomic rename, directory fsync and case-sensitive names before use), and the post-relocation preflight
  passes with only ~1.19 GiB of headroom beyond the required total, so free space is monitored during the
  run and execution halts at a resumable boundary if it approaches the 10 GiB reserve.
- **Status: APPROVED** (owner, 2026-09-19).

---

## Q-01 — RESOLVED_BY_OWNER_LEXICOGRAPHIC_ALLOCATOR (2026-09-19)

**Not a deviation**: the spec leaves the allocator's weights and search procedure unspecified, and
this is the owner's decision filling that gap. It is recorded here because the decision registry
lives in this file, and because it must be attributable.

- **Classification:** OWNER_BENCHMARK_DESIGN_DECISION. It is **not** literature-derived and must not
  be presented as such.
- **What the spec fixed:** §3.5 ratios 70/15/15, seed 20260814, group-disjoint within each dataset
  then pooled, balancing terms named (video-count deviation; binary and attack_macro coverage).
- **What was missing (Q-01):** the relative importance of those terms, their weights, and the search
  procedure.
- **Owner resolution:** a strict **lexicographic** priority order instead of one weighted sum, so no
  trade-off is hidden in an unchosen weight:

  | level | meaning |
  |---|---|
  | P0 | hard grouping / leakage feasibility (constraint, never a penalty) |
  | P1 | total canonical-video ratio |
  | P2 | binary live/spoof distribution |
  | P3 | attack_macro distribution |
  | P4 | attack_raw distribution |
  | P5 | deterministic seeded tie-break (seed 20260814) |

  Each level is minimized and then pinned as an exact equality constraint, so a later priority can
  never worsen an earlier one.
- **Population contract:** the allocator balances **canonical videos**, each weighing exactly one
  video regardless of how many sampled frames survived M2; only M2 COMPLETE rows become usable
  downstream image samples, and FAILED rows stay provenance-only. A canonical video with zero usable
  frames has no owner policy and raises rather than being silently dropped (measured: zero such
  videos).
- **Normalization:** per-category weights `W_c = SCALE // N_c` with `SCALE = 10**6`, frozen as *the
  definition* rather than as a float approximation. The exact rational form would need
  `lcm(N_c) ≈ 1.2e17` at the SiW attack_raw level, which no float64 MILP backend can represent; the
  divergence bound of the fixed-point form is stated in `split_v1.yaml` rather than hidden.
- **Optimality status:** EXACT. HiGHS via `scipy.optimize.milp` (scipy 1.18.1, 1 thread, `mip_rel_gap = 0`)
  proves a global optimum at every level; the objective is independently recomputed in exact Python
  integers and the P0 invariants re-checked on the produced assignment.
- **Frozen artifact:** `configs/frozen/split_v1.yaml` (+ byte-identical snapshot). It contains no
  membership.
- **Evidence:** `M3_ALLOCATOR_DESIGN.md`, `M3_ALLOCATOR_FEASIBILITY.md`, `M3_ALLOCATOR_OBJECTIVE.json`,
  `M2_FAILURE_CONCENTRATION.md`, `tests/test_m3_allocator.py`.
- **Final status: RESOLVED_BY_OWNER_LEXICOGRAPHIC_ALLOCATOR.**
