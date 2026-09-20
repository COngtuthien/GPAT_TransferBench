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
| DEV-018 | APPROVED | CASIA_PAIR_SCALE_UNAVAILABLE_CONTROLLED_ADAPTATION — CASIA face_area_fraction = 1.0, so d_scale = 0 for every CASIA pair; weights not renormalized | 2026-09-19 |

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
| Q-07 | M5 | 12.1 | ArtifactProbeNet input is 224×224 from 256×256 faces; resize vs crop method not stated. **Extended 2026-09-20 (M5 pre-flight):** the same pipeline also leaves the resize/high-pass order, the interpolation and the Gaussian border mode open; measured impacts up to 0.084, 0.132 and 0.599 against a typical residual peak of 0.245. Still OPEN. **RESOLVED 2026-09-20 by the owner** as `RESOLVED_BY_OWNER_FULL_FACE_RESIZE_THEN_HP`: whole 256 face, no crop, OpenCV INTER_AREA 256→224, then GaussianBlur 9×9 σ1.5 BORDER_REFLECT_101, signed residual. |
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

---

## M4 pre-flight — new open questions (2026-09-19)

None of these is a deviation: each is a gap the frozen specification leaves open, surfaced rather
than filled silently. Q-24..Q-27 **block** the common pair manifest; Q-28/Q-29 concern native-method
scope and do not block it. Evidence: `M4_PAIR_CONTRACT_ANALYSIS.md`,
`M4_COMMON_PAIR_FEASIBILITY.md`, `M4_PAIR_DISTANCE_ANALYSIS.md`, `M4_NATIVE_PAIR_REQUIREMENTS.md`.
Proposal (not frozen): `configs/proposed/pairs_v1.proposed.yaml`.

| ID | Milestone | Spec § | Question | Blocks M4 |
|---|---|---|---|---|
| Q-24 | M4 | 6, 8.1 | The 64-candidate sampler is described only as "a hash of (source_sample_id, split_seed)". Per-candidate hash ranking, a seeded PRNG shuffle and reservoir sampling all satisfy that wording and pick different subsets. The cap binds for **every** source in both splits, so this decides the evaluated candidate set of every pair. `pair_id` assignment must be frozen with it, because §8.1 derives FAS-Aug operator parameters from `SHA256(pair_id + global_seed)`. | **YES** |
| Q-25 | M4 | 6 | "z-normalization inside TRAIN" does not fix: pooled vs per-dataset statistics, population vs sample std, the distance norm (L1/L2/other — "distance" names no norm), or zero-variance handling. Measured: per-dataset std differs sharply from pooled, and the L1/L2 median ratio is ≈1.47, which rescales the 0.50-weighted term. | **YES** |
| Q-26 | M4 | 6, 4 | "log face-box area ratio" presumes a face box, but **CASIA has none** — the approved DEV-011 route runs no SCRFD, so 0/4,800 CASIA rows carry a bbox, and CASIA supplies 2,520 of the 8,838 TRAIN sources. This is a missing quantity, not a choice of convention. Also open where a box exists: which box, raw pixel area vs fraction of the frame (frames differ in resolution), log base, non-positive-area guard, distance form. | **YES** |
| Q-27 | M4 | 6 | "normalized Y-channel mean" names neither the Y standard (BT.601 / BT.709 / OpenCV) nor what "normalized" means nor the distance form. Measured BT.601 vs BT.709 gap up to 0.0132 per image on a range of ≈0.55. | **YES** |
| Q-28 | M4/M6 | 8.6 | DSDG native identity pairing "uses only TRAIN identities that possess both live and spoof samples"; SiW has **zero** such identities (no trustworthy subject id, Q-14), yet DSDG must still "generate exactly N_syn samples" where N_syn counts SiW sources. Whether DSDG-native trains on CASIA+MSU identities while generating the full pooled budget, or is marked blocked for SiW, is not stated. | no |
| Q-29 | M4/M6 | 8.7 | DiffFAS native training needs same-dataset **same-identity** live/spoof reconstruction pairs; SiW can supply none. The spec already forbids the wrong answer ("do not synthesize identity labels or pair different identities … merely to increase count") and asks for coverage to be reported, so what remains is scope: run DiffFAS-native on CASIA+MSU only, or treat it as blocked. DEV-013 resolves **common** pairing for SiW and must not be extended to a same-identity requirement. | no |

**Nothing here was decided.** `configs/frozen/pairs_v1.yaml` does not exist, no pair manifest was
written, and M4 remains NOT_STARTED.

---

## M4 owner decisions (2026-09-19) — Q-24 … Q-27 resolved

The OPEN / OWNER_DECISION_REQUIRED entries recorded earlier in this file remain above as history.
Full record: `M4_PAIR_OWNER_DECISIONS.md`. Frozen contract: `configs/frozen/pairs_v1.yaml`
(sha256 `16ff68036a9ed3a315df6944697394359cb2ea53e0b5623721d036a3cf170398`). These are owner
benchmark-design decisions, **not** literature-derived facts.

| ID | Final status | Summary |
|---|---|---|
| Q-24 | **RESOLVED_BY_OWNER_HASH_RANKING** | Eligible set ordered lexically before hashing; ≤64 → all evaluated; >64 → exactly 64 by two-stage SHA-256 (`source_seed_digest` then `candidate_rank_digest`) read as an unsigned big-endian 256-bit integer, lexical `target_sample_id` as a defensive collision tie-break. No PRNG. `pair_id` frozen as `PTR%06d` / `PVA%06d` over (dataset, source_spoof_id), assigned **after** membership so selection can never depend on it — necessary because spec §8.1 seeds FAS-Aug from `pair_id`. |
| Q-25 | **RESOLVED_BY_OWNER_DATASET_TRAIN_ZSCORE_L2** | Pose z-scored **per dataset** on TRAIN rows only (population mean/std, `ddof=0`, float64); Euclidean L2, no division by √3; `std <= 1e-12` is a hard error; VAL reuses the TRAIN statistics; TEST never contributes. |
| Q-26 | **RESOLVED_BY_OWNER_NORMALIZED_VISIBLE_BBOX_LOGRATIO** | MSU/SiW use the original SCRFD box clipped to the frame, as a fraction of frame area (removes resolution dependence); `d_scale = |ln f_t − ln f_s|`, natural log, no epsilon; invalid geometry is a hard error. CASIA handled by **DEV-018**. |
| Q-27 | **RESOLVED_BY_OWNER_BT601_UNIT_MEAN_ABSDIFF** | BT.601 `Y = 0.299R + 0.587G + 0.114B` on the frozen canonical 256×256 RGB face, channels in [0,1], full-image mean including zero-padded pixels, `d_luma = |ΔY_mean|`. No OpenCV YCrCb, no BT.709, no z-scoring. |

Q-28 and Q-29 were still OPEN at that date; they are resolved for M4 manifest scope below.

## DEV-018 — CASIA common-pair scale term is unavailable (controlled adaptation)

- **Classification:** `CASIA_PAIR_SCALE_UNAVAILABLE_CONTROLLED_ADAPTATION`.
- **Spec basis:** §6 requires "scale distance from log face-box area ratio".
- **Reason:** CASIA-FASD has **no detector face box at all**. The approved DEV-011 route consumes a
  pre-cropped 112×112 source and runs no SCRFD, so 0 of 4,800 CASIA rows carry a bbox — a missing
  measurement, not a choice of convention. CASIA supplies 2,520 of the 8,838 TRAIN sources.
- **Deviation:** for CASIA only, `face_area_fraction = 1.0` for every usable sample, hence
  `d_scale = 0.0` exactly for every CASIA source–target pair.
- **What was explicitly refused:** rerunning SCRFD on CASIA, inventing a detector bbox, deriving a
  pseudo-box from FaceXFormer landmarks, and pretending the 112 crop has an original detector box.
- **Weights:** the 0.50/0.20 weights are **not** renormalised; the frozen `d_pair` formula is
  unchanged.
- **Scientific limitation (must be disclosed):** CASIA common pairs are ranked by pose and luminance
  only, with the 0.30-weighted scale contribution identically zero. Because pairing is always within
  a dataset, no `d_pair` value is interpreted as a physical distance comparable across datasets.
- **Verified:** on the real split, CASIA `d_scale` unique values = `[0.0]`
  (`M4_PAIR_PREFLIGHT.json` → `frozen_diagnostic`).
- **Status: APPROVED** (owner, 2026-09-19).

## M4 pre-execution correction (2026-09-20) — Q-24 preimage + Q-28/Q-29 manifest scope

### Q-24 preimage — `PRE_EXECUTION_CONTRACT_IMPLEMENTATION_CORRECTION`

**Not a new scientific decision.** The owner's Q-24 contract did not change. What changed is how the
candidate preimage is written down. The frozen rule is, in explicit byte terms:

```
source_seed_digest_bytes = SHA256(UTF8("gpatbench.pair.source_seed.v1|" + source_sample_id + "|" + str(split_seed))).digest()
                           -> RAW 32 BYTES, never rendered as hex/base64/decimal/text
candidate_rank_digest    = SHA256(source_seed_digest_bytes || UTF8("|gpatbench.pair.candidate.v1|" + target_sample_id)).digest()
```

The forbidden variant `SHA256(UTF8("gpatbench.pair.candidate.v1|" + source_seed_hex + "|" +
target_sample_id))` is a different function of the same inputs and is now named and test-guarded.

Findings, verified rather than asserted:

- `gpatbench/pairs/common.py` **already implemented the raw-byte rule** and needed no change. Proven
  by importing the module as committed at `e4d167b` (blob sha256 `8f2faaa3…`) and re-running
  selection: **0 of 240** candidate-64 sets changed.
- Counterfactual on the forbidden hex variant: **200 of 240 (83.33%)** sets would have differed.
  MSU VAL is 0% only because every MSU VAL source has exactly 64 eligible targets.
- The defect was in the prose of `configs/frozen/pairs_v1.yaml` (under-specified) and in §3 of the
  2026-09-19 owner decision report (which described the hex variant). Both are corrected.
- No authoritative pair manifest, `pair_train_stats_v1.json` or native manifest existed, so **no
  scientific result and no frozen pair membership was invalidated**.
- Config sha256 `16ff68036a9ed3a315df6944697394359cb2ea53e0b5623721d036a3cf170398` →
  **SUPERSEDED_BY_Q24_PREIMAGE_CORRECTION**, retained as history. Current:
  `f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985`.

Record: `M4_Q24_PREIMAGE_CORRECTION.md`. Q-25, Q-26, Q-27 and DEV-018 were not reopened.

### Q-28 / Q-29 — native manifest scope

| ID | Final status | Decision |
|---|---|---|
| Q-28 | **RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE** | `dsdg_identity_pairs_v1.parquet` contains CASIA + MSU only. SiW: `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`, `native_identity_pair_coverage = 0`, represented in the audit coverage table, never as rows. |
| Q-29 | **RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE** | `difffas_recon_pairs_v1.parquet` contains CASIA + MSU only. Same-dataset, same-trustworthy-identity, live/spoof. SiW not instantiable; DIFFFAS-BIN does not rescue it, because binary style collapse does not recover identity. |

Forbidden in both: invented or pseudo SiW subject ids, `content_group_id` or `video_id` as person
identity, DEV-013 as a same-identity substitute, and pairing arbitrary live/spoof rows to imitate
same-ID reconstruction.

**Scope limit:** these resolve M4 **manifest scope only**. The complete M6 DSDG and DiffFAS
training-adaptation strategy for SiW-Mv2 is a separate later decision and is not settled here.

**DEV-013 boundary (reasserted):** DEV-013 governs the COMMON SiW pairing rule (different video AND
different exact-content group). It is a different-video / different-exact-content guarantee, never a
same-person or different-person claim, and it does not extend to native same-identity pairing.

## M4 execution (2026-09-20)

### No new deviation

M4 execution introduced **no new deviation**. DEV-018 (CASIA `face_area_fraction = 1.0`, hence
`d_scale = 0`) held on the authoritative data: CASIA `d_scale` unique values = `[0.0]` across all
3,096 CASIA pairs in both splits, and the 0.50/0.20 weights were not renormalised. Every other
frozen decision executed as written.

### Determinism defect found and fixed before the freeze

`fit_pose_stats` stacked TRAIN rows in arrival order. Floating-point accumulation is not
associative, so the **population standard deviation** differed in the last ulp between input
orderings, which perturbs the per-axis z-scores and can flip a near-tied `d_pair` to a different
target. The deterministic rerun gate caught this on the shuffled-input runs.

- **Fix:** stack TRAIN rows in canonical `sample_id` order, so the fit is a function of the row set.
- **Q-25 unchanged:** per-dataset TRAIN z-score, population std, `ddof = 0` and L2 are untouched.
- **Blast radius: none.** The defect surfaced before any manifest was frozen; the authoritative
  artifacts were regenerated with the corrected fit and are the only ones ever recorded.
- This is a **defect fix in service of the already-frozen determinism contract**, not a new
  scientific decision. Evidence: `M4_DETERMINISM_REPORT.md`.

### Native pair manifests — BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP

`manifests/dsdg_identity_pairs_v1.parquet` and `manifests/difffas_recon_pairs_v1.parquet` were
**not created**. Q-28/Q-29 settled *which datasets* may appear; they do not define *how* rows are
constructed. `third_party/registry.yaml` records `pinned_commit: null` and
`url_verification: UNVERIFIED` for E06b (DSDG-NATIVE) and E07b (DiffFAS-NATIVE), and
`methods/dsdg/` and `methods/difffas/` contain only `.gitkeep`, so the official data-loader
semantics cannot be inspected: whether pairing is materialized or sampled online, how many tuples
exist, whether frames repeat, what seed applies, and for DiffFAS how the guide is drawn and what
fallback the official code supports.

Nothing was invented to make M4 pass — no first-lexical pick, no seeded random draw, no Cartesian
product, no one-to-one or cyclic matching, and the common-pair Q-24 ranking was not repurposed.
CASIA (35/35 identities) and MSU (25/25) are `SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP`; SiW-Mv2 remains
`NOT_INSTANTIABLE_MISSING_SUBJECT_ID` with coverage 0 and is represented in the coverage table,
never as manifest rows.

**Consequence:** M4 is `IN_PROGRESS`, phase `COMMON_PAIRS_COMPLETE_NATIVE_PAIR_SOURCE_BLOCKED`. It is
not marked COMPLETE merely because the common pairs succeeded. Evidence:
`M4_NATIVE_PAIR_SOURCE_AUDIT.md`, `M4_NATIVE_PAIR_COVERAGE.md`.

## Owner Protocol Amendment A1 (2026-09-20) — DEV-019, DEV-020, DEV-021

Amendment: `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md`
(sha256 `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`).
The frozen specification is **not edited** and remains historically authoritative. The earlier M4
blocker history above is preserved: original native requirement → blocker discovered
(`BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP`) → owner scientific objective clarified →
Amendment A1 → Track-A identity-free adaptation frozen.

**Timing:** adopted before M5 probe training, M6 baseline training, M7 GPAT training, any synthetic
generation and any result inspection. Justified by protocol fairness and data availability, never by
observed performance. No model has been trained and TEST performance has never been observed.

### DEV-019 — MAIN_FAIR_TRACK_IDENTITY_FREE_PROTOCOL_AMENDMENT

- **Classification:** owner scientific protocol amendment (benchmark design), not a literature-derived
  fact and not a dataset limitation.
- **Reason:** the primary comparison must use the same pooled population — CASIA-FASD + MSU-MFSD +
  SiW-Mv2 — for every method. DSDG and DiffFAS natively require subject identity, and SiW-Mv2 has
  none (Q-14: 0 of 9,507 SiW TRAIN rows carry a `subject_id_global`, against 35/35 CASIA and 25/25
  MSU identities with both live and spoof). Dropping SiW for two methods would confound method with
  data; fabricating identity is forbidden; blocking the benchmark on a field the primary question
  does not need is wasteful.
- **Deviation:** subject identity becomes **forbidden supervision in Track A** (the primary fair
  comparison). Identity-dependent native behaviour moves to **Track B**, secondary, which may have
  partial coverage and never appears in a main table without a track column.
- **Frozen contract:** `configs/frozen/fair_track_v1.yaml`
  (sha256 `d9b02214d1e4701fe7d53972e94be0c2e68c6199a3378cacd69f2c1c87826d9d`).
- **Scientific limitation (must be disclosed):** Track A compares methods under harmonized
  supervision constraints, so DSDG and DiffFAS appear there in adapted form. Their native,
  identity-supervised behaviour is a different question and is answered only in Track B, on
  CASIA + MSU.
- **Status: APPROVED** (owner, 2026-09-20).

### DEV-020 — DSDG_BIN_IDFREE_COMMON_PAIR_ADAPTATION

- **Classification:** `CONTROLLED_ADAPTATION`. **Not** `FAITHFUL_OFFICIAL`.
- **Source of truth:** FaceX-Zoo at pinned commit `16b793a7564a4b9308cf94e62bdb2ffacb3a725a`,
  `addition_module/DSDG`; analysis in `M4_DSDG_IDFREE_SOURCE_ANALYSIS.md`.
- **Key change 1 — the relation.** The official `GenDataset_s.get_pair` draws the live partner with
  `random.choice(make_pair_dict[label]['1'])`, where `label = video_name[4:6]` is the OULU-NPU user
  id: same-subject, online, random. It is replaced by the frozen common fair pair
  (`manifests/pairs_train_v1.parquet`, `source_spoof_id → target_live_id`), which gives all three
  datasets the same split, source population and target-selection protocol.
- **Key change 2 — `lambda_pair = 0`** (official default 0.5).
  `loss_pair = lambda_pair * MSE(rec_spoof_identity_feature, rec_live_identity_feature)` forces the
  identity features of the two reconstructions to coincide. Under the common fair pair the two
  images are deliberately different people (CASIA/MSU) or a different video **and** content group
  (SiW), so optimising it would assert something false by construction.
- **Not changed:** `loss_rec`, `loss_kl`, `loss_mmd`, `loss_ip`, `loss_cls`, `loss_ort`, the
  architecture and the warm-up schedule. Only the term whose correctness mathematically requires
  shared identity was disabled. In particular `loss_ip` compares each reconstruction with **its own**
  input, so it carries no cross-pair identity requirement and stays.
- **Disclosed degeneracy:** `Cls(hdim, attack_type)` is `nn.Linear(hdim, attack_type)`; with the
  binary collapse `attack_type = 1`, `CrossEntropyLoss` over a single logit is identically 0 with
  zero gradient, so `loss_cls` contributes nothing. This is stated rather than patched, and
  `attack_macro`/`attack_raw` are **not** secretly substituted.
- **Frozen contract:** `configs/frozen/dsdg_bin_idfree_v1.yaml`
  (sha256 `a0f84fac2e893158410aab6d3cd11f464f6385edad2403c1f8ce4a66615b15de`).
- **Verified:** 8,838 rows (CASIA 2,520 · MSU 1,200 · SiW 5,118); the relation is byte-identical when
  every CASIA/MSU subject id is nulled; no identity field exists in the adapter interface.
- **Status: APPROVED** (owner, 2026-09-20).

### DEV-021 — DIFFFAS_BIN_IDFREE_UNPAIRED_ADAPTATION

- **Classification:** `CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH`. **Not**
  `FAITHFUL_NATIVE`.
- **Source of truth:** murphytju/DiffFAS at pinned commit
  `23f40519ec25a833ebc06842aa6fbab74fad4d15`; analysis in `M4_DIFFFAS_IDFREE_SOURCE_ANALYSIS.md`.
- **Key change 1 — `use_pair = false`.** This is an official argument, not a source modification:
  the model input becomes `x_t` alone, the conditioning is `style_spoof` alone and the target is the
  epsilon of `GT`. **Proven numerically** against the pinned code
  (`M4_DIFFFAS_CONTENT_INERTNESS.json`): with everything else held fixed, changing only the content
  tensor leaves the model input and every loss term bit-identical under `use_pair=false`
  (`max_abs_loss_diff = 0.0`) and changes both under `use_pair=true`, which shows the probe can
  detect a dependency. `content_training_role = INERT_API_PLACEHOLDER`.
- **Key change 2 — deterministic guide.** The official `random.choice(os.listdir(style_folder))` is
  replaced by a frozen raw-byte SHA-256 ranking over the binary spoof pool of the same dataset,
  with lexical `guide_spoof_sample_id` as a defensive tie-break. Eligibility excludes subject
  identity **and** `attack_raw`; `style_id = SPOOF_BINARY` is provenance only and the guide is a real
  TRAIN spoof image. Self-guides are allowed because the official support includes the GT file; the
  count (3) is reported and never adjusted after the fact.
- **Frozen contract:** `configs/frozen/difffas_bin_idfree_v1.yaml`
  (sha256 `aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb`);
  manifest `manifests/difffas_bin_idfree_train_v1.parquet`
  (sha256 `0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6`, 8,838 rows).
- **Scientific limitation (must be disclosed):** `use_pair=false` removes the content conditioning
  entirely, so this variant does not reproduce the paired DiffFAS training signal. It may never be
  described as native DiffFAS.
- **Status: APPROVED** (owner, 2026-09-20).

### M4 status under the amendment

M4 is **COMPLETE under the amended main-track rule**, not under the originally specified rule. The
originally specified native pair manifests were **not** created and are deferred to M6 as a
secondary track. The audit trail preserves that distinction
(`STAGE_STATE.json → milestones.M4.amendment_a1.original_native_manifests_completed = false`).

## M5 pre-flight (2026-09-20) — ArtifactProbeNet contract audit

**No deviation was opened and nothing was trained.** M5 remains NOT_STARTED, no checkpoint exists,
and `configs/frozen/artifact_probe.yaml` was deliberately NOT created. The proposed contract is
`configs/proposed/artifact_probe.proposed.yaml`; the analyses are the `M5_ARTIFACT_PROBE_*` files.

**Q-07 already existed** in this file and in `configs/CONFIG_STATUS.md`. It was extended above, not
duplicated, and no historical question ID was invented.

### New M5 decision ids — all OWNER_DECISION_REQUIRED

| ID | Spec § | Question | Measured impact |
|---|---|---|---|
| D-M5-01 | 12.1 | Classification population: SPOOF `attack_macro` classes only (K=6) or LIVE + spoof (K=7)? `attack_macro` is a §3.2 enum that includes `live` and is populated on every row; "real TRAIN only" contrasts real with synthetic, not spoof with live. | Changes K, every class weight, the macro-F1 denominator and therefore checkpoint selection. Recommendation: **B (live + spoof)**, because §13's "attack consistency" compares a *predicted* `attack_macro` to the source's, and a probe that cannot emit `live` cannot report that a synthetic lost all spoof evidence. |
| D-M5-02 | 12.1 | "inverse-frequency weights clipped to [0.5, 3.0]" — which normalisation? | **The literal `w_c = 1/n_c` clips EVERY class to exactly 0.5 in both populations**, so the weighting vanishes. Recommendation: `w_c = N/(K·n_c)`, the only scale-free non-degenerate reading under which the frozen clip binds a minority of classes. |
| D-M5-03 | 12.1 | Post-high-pass normalisation and value domain. | ImageNet mean/std applied to the signed residual maps the input to ≈[−4.17, +0.81]. Settled by measurement within this item: value domain (3.3e-07) and Gaussian implementation (6.6e-07) are not execution-affecting; the **signed residual must be retained** (range ≈ [−0.47, +0.59]). |
| D-M5-04 | 12.1 | Backbone fine-tune scope: full / partial / classifier-only. | Different probes. §12.1 gives one LR where §14 gives two, which is weak evidence for full fine-tuning but not decisive. |
| D-M5-05 | 12.1 | VAL macro-F1 class set, `zero_division`, implementation. | Today every TRAIN class also appears in VAL, so the readings coincide; they would diverge under a different split, and `scikit-learn` is not installed, so the definition must be frozen rather than inherited. |
| D-M5-06 | 12.1 | Checkpoint tie-break when two epochs share the maximum macro-F1. | Candidate for owner review: higher macro-F1 wins, exact tie → earlier epoch (consistent with §10.6, which is a different rule and confers no authority). |
| D-M5-07 | 12.1 | "cosine decay": implementation, stepping, `T_max`, `eta_min`, warmup. | §14 / `downstream_resnet18.yaml` use cosine with 3-epoch warmup and `min_lr` 1e-6, but that is scoped to the downstream evaluator; importing it silently would be an invention. |
| D-M5-08 | 12.1 | Augmentation and DataLoader. | **`NO_AUGMENTATION_SPECIFIED`.** §14.1 "FAS-safe augmentation — frozen" is scoped to the downstream evaluator, and adopting it would also silently resolve Q-07 through `RandomResizedCrop`. Forbidden regardless: `WeightedRandomSampler` on top of weighted CE, and per-dataset loss weighting. |

### Environment blocker

| ID | Item | Finding |
|---|---|---|
| E-M5-01 | training device | torch 2.14.0+cpu / torchvision 0.29.0+cpu, `cuda_available = False`, no `nvidia-smi`. Spec §12.1 requires AMP, so **M5 execution cannot run on this machine**. This blocks M5 *execution*, not the contract freeze. |

### Disclosed structural limitation (not a deviation)

Four of the six spoof `attack_macro` classes (`makeup`, `mask_2d`, `mask_3d`, `partial`) occur in
**SiW-Mv2 only**; only `print` and `replay` span all three datasets. The probe can therefore score
partly by recognising the dataset rather than the attack family. The §13 metrics that consume it are
computed within a dataset, so they remain usable, but no cross-dataset reading of probe classes is
supported and none may be claimed.

## M5 owner resolution (2026-09-20) — ArtifactProbeNet contract frozen

**No deviation was opened and nothing was trained.** All nine contract decisions and the
environment blocker raised at the M5 pre-flight are resolved by the owner and frozen in
`configs/frozen/artifact_probe.yaml` (sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`), with a byte-identical snapshot and **0**
unresolved execution-affecting fields. The pre-flight analysis is preserved unchanged under
"RESOLVED" banners; `configs/proposed/artifact_probe.proposed.yaml` is kept as history.

**These are owner resolutions of details spec §12.1 left under-specified. The frozen specification
did not uniquely dictate them and no report may say otherwise.**

| id | resolution token | effect |
|---|---|---|
| D-M5-01 | `RESOLVED_BY_OWNER_LIVE_PLUS_SPOOF_ATTACK_MACRO` | K = 7 (`live` included); TRAIN 14,467 / VAL 3,121; `other_spoof` excluded (0 observations, no unused logit) |
| D-M5-02 | `RESOLVED_BY_OWNER_BALANCED_INVERSE_FREQUENCY` | `w = clip(N/(K·n_c), 0.5, 3.0)`, float64, TRAIN only, no renormalisation. `live` 0.5 (clipped low), `mask_2d` 3.0 (clipped high), five classes unclipped |
| Q-07 | `RESOLVED_BY_OWNER_FULL_FACE_RESIZE_THEN_HP` | whole face, no crop, INTER_AREA 256→224, then Gaussian 9×9 σ1.5 `BORDER_REFLECT_101`, signed residual |
| D-M5-03 | `RESOLVED_BY_OWNER_NO_POST_HP_IMAGENET_NORMALIZATION` | the only scaling is `uint8/255`; pretrained initialization is explicitly **not** a reason to ImageNet-normalise a signed residual |
| D-M5-04 | `RESOLVED_BY_OWNER_FULL_FINE_TUNE` | `fc → Linear(512, 7)`; all 11,180,103 parameters trainable under one AdamW |
| D-M5-05 | `RESOLVED_BY_OWNER_FIXED_CLASS_SET_MACRO_F1` | 7×7 int confusion matrix, zero-division → 0, unpredicted classes never dropped, no sklearn |
| D-M5-06 | `RESOLVED_BY_OWNER_MAX_VAL_MACRO_F1_EARLIEST_TIE` | strict `>` replacement, so an exact tie keeps the earlier epoch |
| D-M5-07 | `RESOLVED_BY_OWNER_LITERAL_COSINE_NO_WARMUP` | `CosineAnnealingLR(T_max=30, eta_min=0)` stepped once per epoch; epoch 1 at 1e-4; the §14 warmup is not imported |
| D-M5-08 | `RESOLVED_BY_OWNER_NO_DATA_AUGMENTATION` | deterministic resize + high-pass only; §14.1 stays scoped to the downstream evaluator |
| E-M5-01 | `RESOLVED_BY_OWNER_GPU_REQUIRED` | authoritative training on `sparc5090`; CPU fallback and disabling AMP are forbidden; the trainer refuses a CPU authoritative run (verified) |

### Preserved limitation (deliberately not corrected)

`makeup`, `mask_2d`, `mask_3d` and `partial` occur **only** in SiW-Mv2 in the current population.
The owner directed that this must **not** be "fixed" by resampling, dataset balancing, label
merging or dropping classes. ArtifactProbeNet is a frozen pooled measurement tool and some attack
classes are dataset-specific in the available corpus; later generator metrics must be interpreted
accordingly.

### Milestone status

M5 remains **NOT_STARTED** with pre-flight status `READY_FOR_GPU_EXECUTION_PREFLIGHT`. Freezing a
contract, implementing a trainer and running a CPU-safe smoke do not start a milestone. No remote
GPU fact has been measured; `M5_GPU_EXECUTION_PLAN.md` lists the checks still to perform.

## M5 validation-epoch contract correction (2026-09-21)

**Classification: `CONFIG_SEMANTIC_AMBIGUITY_FOUND_BEFORE_EXECUTION`.** Not an executed scientific
defect and not a result invalidation — **no model has ever been trained in this project.**

The frozen ArtifactProbeNet config wrote the validation schedule as
`validation.checkpoint.evaluate_epochs: [1, 30]`. As YAML that is a two-element list, so it could be
read as "validate at epochs 1 and 30 only" rather than "validate at the end of every epoch from 1 to
30" — 2 passes versus 30, which would select different checkpoints.

**Why nothing was invalidated:** a repository-wide search found the field in exactly one place (the
frozen config, plus its byte-identical snapshot) and **no code consumed it** — no hit in
`gpatbench/`, `tests/` or `tools/`. `gpatbench/probe/train.py::run(dry_run=False)` still raises
before any optimization, the training loop is not implemented, no checkpoint exists and M5 is
`NOT_STARTED`.

**Correction:** the field is replaced by an explicit triple —
`evaluate_every_epoch: true`, `epoch_start: 1`, `epoch_end: 30` (plus `evaluation_count: 30` and
`range_as_two_element_list: FORBIDDEN`). `gpatbench.probe.contract.validation_epochs()` is the
**single** source of the sequence `(1, 2, …, 30)` and hard-fails on a wrong start, a wrong end, a
wrong length or `evaluate_every_epoch != true`. `contract.select_best_epoch()` implements the
D-M5-06 rule purely so it can be tested without training; on a sequence peaking at epoch 17 it
selects 17.

Everything else is unchanged: population, class weights, high-pass transform, ResNet contract,
optimizer, scheduler, DataLoader, macro-F1, tie-break and embedding. The CPU-safe smoke still passes
22/22.

| config sha256 | status |
|---|---|
| `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` | **SUPERSEDED_BY_VALIDATION_EPOCH_CONTRACT_CORRECTION** (retained above as history) |
| `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe` | current; snapshot byte-identical |

Record: `M5_VALIDATION_EPOCH_CONTRACT_CORRECTION.md`. M5 remains **NOT_STARTED** /
`READY_FOR_GPU_EXECUTION_PREFLIGHT`.

