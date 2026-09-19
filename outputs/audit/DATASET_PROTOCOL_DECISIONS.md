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
