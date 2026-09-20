# Dataset Protocol Decisions — consolidated history (as of 2026-09-19)

- **Machine-readable policy:** `configs/frozen/dataset_protocol_policy_v1.yaml` (sha256 in `ARTIFACT_INDEX.csv`).
- **Full entries:** `deviation_report.md`. **Ledger rows:** `EXECUTION_LEDGER.jsonl`.
- **History policy:** rejected or superseded decisions stay listed; nothing is deleted.

## Deviations

| ID | Original question / issue | Evidence | Original status | Owner decision | Latest status | Milestones | Implementation consequence |
|---|---|---|---|---|---|---|---|
| DEV-005 | Spec §3.4 "nearest unique valid frame indices" does not say who keeps a contested frame | `base.py:select_frame_indices`; measured collision use = 0 | UNAPPROVED | Approved as IMPLEMENTATION_DETAIL (2026-09-18) | **APPROVED** | M1, M2 | nearest unused valid index, lower index wins ties, earlier position keeps; frozen positions unchanged |
| DEV-006 | Q-04: what is a valid frame, and how are indices kept after decode failures? | `DECODER_INDEX_AUDIT.md`, `decoder_index_*.csv` (PyAV independent; 1,980/1,980 consistent) | UNAPPROVED; "stop after 5 failures" rule rejected by owner | Semantic definition approved; index-bounded rule proven | **APPROVED** | M1, M2 | decode [0, N_declared); failure = invalid index, later indices unshifted; `CAP_PROP_POS_FRAMES` never used as the index |
| DEV-007 | CASIA `bs*`/`fs*` files | fs = exact flips 6337/6337; bs = brightened | UNAPPROVED | Approved with evidence | **APPROVED** | M1, M2 | indexed, never canonical samples |
| DEV-008 | CASIA subject = `{partition}_s{N}` | visual: train sN ≠ test sN | UNAPPROVED | **Not approved** (owner correction) | **SUPERSEDED by DEV-010** | M1, M3 | withdrawn; kept for history |
| DEV-009 | M0 note "SiW has lowercase live/spoof dirs" | concatenated `ls` output misread | UNAPPROVED | Approved as a record correction | **APPROVED** | M0/M1 records | M0 text kept; correction appended |
| DEV-010 | CASIA code semantics and identities | owner protocol evidence; 600/600 sequences verified; disk re-audit 0 mismatches | — | Owner decision | **APPROVED** | M1, M3, M4 | 1, 2, HR_1 live; 3–6, HR_2, HR_3 print; 7, 8, HR_4 replay; train N → N, test N → 20+N |
| DEV-011 | Q-15: CASIA exists only as 112×112 crops | `CASIA_SOURCE_AUDIT.md`, `CASIA_RESIZE_FREQUENCY_AUDIT.md` | PROPOSED / UNAPPROVED (2026-09-18) | Approved (2026-09-19) | **APPROVED — CONTROLLED_DATASET_ADAPTATION** | M2+ (all CASIA use) | 112 → INTER_CUBIC 256, SCRFD N/A, same images for all methods, disclosed |
| DEV-012 | Q-14: SiW-Mv2 has no person IDs; spec §3.5 would block the main split | `SIW_PROTOCOL_NAME_SEMANTICS.md`, `siw_subject_mapping_candidate.csv` | — (previously `BLOCKED_BY_MISSING_SUBJECT_ID`) | Owner approves the video-disjoint fallback (2026-09-19) | **APPROVED** | M3, M5–M13, paper | content-group (sha256) units, 70/15/15, frames inherit video split; not subject-disjoint |
| DEV-013 | Spec §6: pairs need different subjects; unavailable for SiW | as DEV-012 | — | Owner approves (2026-09-19) | **APPROVED** | M4, M8, M9, fingerprint-probe grouping, paper | different video AND different content group; never claimed different-person |
| DEV-003 | `lambda_dir` in the GPAT-B0 YAML (unrelated to datasets) | spec §10.3 vs §23.2 | UNAPPROVED | Owner: keep UNRESOLVED | **UNRESOLVED** | M7 | must be resolved before M7 |

## Questions

| ID | Question | Evidence | Original status | Owner decision | Latest status | Milestones | Consequence |
|---|---|---|---|---|---|---|---|
| Q-04 | Definition of a valid frame | DEV-006 evidence | OPEN | Semantic definition approved | **RESOLVED** | M1, M2 | see DEV-006 |
| Q-12 | CASIA HR_1 under `spoof/` | owner protocol evidence + visual check | OPEN (label conflict) | HR_1 = real_high = live | **RESOLVED** | M1, M3 | 150 live / 450 spoof |
| Q-13 | Unmapped attack tokens (CASIA codes, SiW `Paper`) | `attack_map_v1` rev 2 traces; official `config_siwm.py` `'Print': 'Paper'` @8667dbc | OPEN | CASIA via owner evidence; Paper → print via official code | **RESOLVED** | M1, native track | 0 unmapped tokens |
| Q-14 | SiW-Mv2 subject identity | official repo/protocol/supplementary; 1,700 × NO_EVIDENCE; AdaFace not used | BLOCKED_BY_MISSING_SUBJECT_ID | Subject ID not required under DEV-012/013 | **SUBJECT_ID_UNAVAILABLE_ACCEPTED_WITH_VIDEO_DISJOINT_DEVIATION** | M3, M4 | no pseudo IDs; stop inferring identities |
| Q-15 | CASIA pre-cropped source | `CASIA_SOURCE_AUDIT.md` | UNRESOLVED | Accept the local source under DEV-011 | **RESOLVED_WITH_CONTROLLED_ADAPTATION** | M2 | see DEV-011 |
| Q-16 | Byte-identical SiW Replay videos (6 pairs) | `SIW_DUPLICATE_VIDEO_AUDIT.*`, `siw_content_groups.csv` | OPEN | Group by exact raw sha256 for splitting | **RESOLVED_WITH_EXACT_CONTENT_GROUPING** | M3, M4 | same content → same split; raw records preserved |
| Q-17 | Official SiW protocol names absent locally | `SIW_PROTOCOL_COVERAGE_AUDIT.*` (64 names) | INFO | Keep as a documented limitation | **DOCUMENTED LIMITATION** | reporting | local inventory is the source of truth |
| Q-01 | Split allocator objective weights | spec §3.5 names terms only | RESOLVED | Owner lexicographic allocator P0-P5 (2026-09-19) | **RESOLVED_BY_OWNER_LEXICOGRAPHIC_ALLOCATOR** | frozen in configs/frozen/split_v1.yaml | see M3_ALLOCATOR_DESIGN.md |

## Distinction to keep

Dataset subject labels (used for split and pairing guarantees) are **not** the same thing as GPAT
identity preservation (AdaFace cosine target vs synthetic). The latter needs no dataset label and is
unaffected by DEV-012/013.

## M4 pre-flight additions (2026-09-19)

| ID | Topic | Basis | Status | Decision | Effective | Needed by | Next step |
|---|---|---|---|---|---|---|---|
| Q-24 | 64-candidate sampler + pair_id rule | spec §6 names hash inputs only; §8.1 consumes pair_id | OPEN | none | **OWNER_DECISION_REQUIRED** | M4 execution | owner picks the sampling algorithm |
| Q-25 | Pose z-normalization scope/std/norm | spec §6 wording | OPEN | none | **OWNER_DECISION_REQUIRED** | M4 execution | owner picks scope, std convention and norm |
| Q-26 | Scale face-box definition; CASIA has no bbox | spec §6 + DEV-011 | OPEN | none | **OWNER_DECISION_REQUIRED** | M4 execution | owner resolves the missing CASIA quantity |
| Q-27 | Luminance Y standard and normalization | spec §6 wording | OPEN | none | **OWNER_DECISION_REQUIRED** | M4 execution | owner picks BT.601 vs BT.709 and the range |
| Q-28 | DSDG native identity pairing for SiW | spec §8.6 | OPEN | none | OWNER_DECISION_REQUIRED (non-blocking) | M6 | owner sets scope; coverage already measured (0) |
| Q-29 | DiffFAS native same-ID pairs for SiW | spec §8.7 | OPEN | none | OWNER_DECISION_REQUIRED (non-blocking) | M6 | owner sets scope; spec forbids fabricating identity |

## M4 owner decisions (2026-09-19)

| ID | Topic | Basis | Status | Decision | Effective |
|---|---|---|---|---|---|
| Q-24 | 64-candidate sampler + pair_id | spec §6/§8.1 under-specified | RESOLVED | two-stage SHA-256 ranking; PTR/PVA ids assigned after membership | **RESOLVED_BY_OWNER_HASH_RANKING** |
| Q-25 | Pose normalization + distance | spec §6 under-specified | RESOLVED | per-dataset TRAIN z-score, population std ddof=0, Euclidean L2 | **RESOLVED_BY_OWNER_DATASET_TRAIN_ZSCORE_L2** |
| Q-26 | Scale face-box definition | spec §6 under-specified; CASIA has no bbox | RESOLVED | clipped-visible bbox fraction of frame area, natural-log ratio; CASIA d_scale = 0 (DEV-018) | **RESOLVED_BY_OWNER_NORMALIZED_VISIBLE_BBOX_LOGRATIO** |
| Q-27 | Luminance definition | spec §6 under-specified | RESOLVED | BT.601 Y on canonical 256 face, [0,1], full-image mean, absolute difference | **RESOLVED_BY_OWNER_BT601_UNIT_MEAN_ABSDIFF** |
| Q-28 | DSDG native scope for SiW | spec §8.6 | OPEN at 2026-09-19; see below | none | OWNER_DECISION_REQUIRED (non-blocking for common pairs) |
| Q-29 | DiffFAS native scope for SiW | spec §8.7 | OPEN at 2026-09-19; see below | none | OWNER_DECISION_REQUIRED (non-blocking for common pairs) |

## M4 pre-execution correction (2026-09-20)

| ID | Topic | Basis | Status | Decision | Effective |
|---|---|---|---|---|---|
| Q-24 | Candidate preimage byte layout | frozen contract was written ambiguously | CORRECTED | candidate preimage hashes the **RAW 32-byte** source digest concatenated with `UTF8("\|gpatbench.pair.candidate.v1\|" + target_sample_id)`; hex-text variant forbidden and test-guarded | **PRE_EXECUTION_CONTRACT_IMPLEMENTATION_CORRECTION** (owner contract unchanged; 0/240 candidate sets changed) |
| Q-28 | DSDG native manifest scope | spec §8.6 | RESOLVED | `dsdg_identity_pairs_v1` = CASIA + MSU only; SiW `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`, coverage 0, no fake rows | **RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE** |
| Q-29 | DiffFAS native manifest scope | spec §8.7 | RESOLVED | `difffas_recon_pairs_v1` = CASIA + MSU only; same dataset + same trustworthy identity + live/spoof; DIFFFAS-BIN does not rescue SiW | **RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE** |

Q-28/Q-29 fix **M4 manifest scope only**; the later M6 DSDG / DiffFAS adaptation strategy for SiW
remains an open, separate decision. DEV-013 stays confined to the common pairing contract and is
never a same-identity substitute.

## M4 execution (2026-09-20)

| ID | Topic | Basis | Status | Decision | Effective |
|---|---|---|---|---|---|
| Q-25 impl | Pose-statistics row order | frozen determinism contract | FIXED | `fit_pose_stats` stacks TRAIN rows in canonical `sample_id` order, so the population std cannot vary with arrival order | defect fix; Q-25 semantics unchanged |
| Native construction | DSDG / DiffFAS row enumeration | source-of-truth hierarchy level 1 empty | **BLOCKED** | no native manifest written; nothing invented | `BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP` |

The native blocker is a **source-of-truth gap**, not a dataset-protocol decision: the datasets
support the row semantics (CASIA 35/35, MSU 25/25 TRAIN identities with both live and spoof), but
the official repositories are unpinned and unvendored, and pinning them is an M6 first-setup action
under `third_party/registry.yaml` and spec §8.1. SiW-Mv2 stays `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`
under Q-14 regardless; identity is never fabricated and DEV-013 is never same-identity evidence.

