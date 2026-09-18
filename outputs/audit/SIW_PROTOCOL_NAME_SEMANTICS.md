# SiW-Mv2 Protocol Name Semantics & Subject-Mapping Recovery (Q-14) — 2026-09-18

**Official source:** github.com/CHELSEA234/Multi-domain-learning-FAS. Commit
`8667dbcd316b38141729c057adf7517fe0602608` was verified by `git ls-remote` on 2026-09-18: it is the current `HEAD`/`main`.
Files were fetched at that commit into `siw_subject_recovery/official_repo_cache/` (git-ignored). Their
sha256 values are tracked in `siw_subject_recovery/official_repo_manifest.json`. Script: `tools/audit_siw_resolution.py`.

## 1. What a protocol entry (e.g. `Live_1`, `Makeup_Co_5`) is — traced through official code (FACT)

1. **Raw video:** `SiW-Mv2/Live/Live_1.mov`. The official README §3.1 describes `Live (contain 785 raw video files)` and `Spoof (contain 14 folders, each of which has raw videos)`.
2. **Preprocessing** `source_SiW_Mv2/preprocessing.py` (sha256 `091849da…`):
   - L29 `def video_process(vlist, folder_dir)`;
   - L31 `folder_name = vd.split('/')[-1].split('.')[0]`, i.e. the **video filename stem**;
   - L32 `folder = os.path.join(folder_dir, folder_name)`: one folder of frames and landmarks per video.
3. **Protocol list** `pro_3_text/trainlist_live.txt` etc. contains stems such as `Live_1`.
4. **Path construction** `source_SiW_Mv2/config_siwm.py` (sha256 `ebe37e87…`):
   - L132–L142 read the lists;
   - L162–L164 `self.LI_DATA_DIR.append(self.live_img_root + x)`, and likewise for spoof (`spoof_img_root + x`).

   The entry is used **only** as the name of one video's preprocessed folder.

**Conclusion.** Protocol entries are **B/C: video stems = preprocessed per-video folder IDs**. They are not person identities.
- The README text "live subjects … spoof subjects" and "(Repetitive subject names are for balancing the number between live and spoof subjects.)" uses "subject" to mean these per-video entries.
- The ECCV'22 supplementary Table 1 lists *Video #* and *Subject #* as different numbers per attack (e.g. Print: 135 videos, 61 subjects), so a video stem cannot be a subject.

## 2. Search for a video → person mapping

| Source | Inspected | Person-identity mapping? |
|---|---|---|
| Local `README.pdf`, `DRA.pdf` | full text | no (counts only; per-sample naming deferred to an external link) |
| Official `README.md`, `source_SiW_Mv2/README.md` | full | no |
| `pro_3_text/*.txt` (14 lists) | all tokens | no: video stems only |
| `config_siwm.py`, `preprocessing.py`, `dataset.py`, `csv_parser.py` | grep subject/identity/person/client/sub_id | no SiW-Mv2 identity field. `sub_id` parsing in `source_multi_domain/config.py` applies to SiW v1/OULU names (`device_sess_sub_spoof`) |
| `source_multi_domain/combine_label_illu.csv` | header + rows | SiW **v1** names (`053-1-3-1-1`), not SiW-Mv2 |
| `source_multi_domain/FASMD/SIWM_list/{age,race}/sub*/` | directory names | undocumented image-gallery folders; not referenced by code; semantics not stated → not a mapping source (and the images show faces, so they were not viewed, per DRA) |
| ECCV'22 supplementary (sha256 `b9dba0ff…`) | text | per-attack *Subject #* counts only |
| Local `Dataset request/` | file names | agreement forms only |
| PRISM archive `FULL_SIW_PHYSICAL_AUDIT.json` | keys | `all_frame_subject_null: true` |

## 3. Coverage facts (measured)

- All 1,700 local videos appear in ≥ 1 official list, under their stem.
- The official lists name 1,764 distinct stems. **64 listed stems are absent locally** (e.g. `Live_892…Live_902`, `Mask_Silicone_18/19`, `Partial_Paperglass_100…`), so the local release is a subset of what the protocol files reference.
- `trainlist_live` ∩ `testlist_live` = ∅ at the stem level.
- 592 stems are repeated inside a list (balancing).
- `siw_subject_mapping_candidate.csv`: 1,700 rows, **all `NO_EVIDENCE`, candidate_subject_id empty** (no pseudo IDs).

## 4. Decision

**CASE C — NO TRUSTWORTHY MAPPING.** Q-14 remains `BLOCKED_BY_MISSING_SUBJECT_ID` for the SiW-Mv2 M3 main split.
- The official protocol lists could at most support a *video-level* partition. That is exactly the `protocol_v1_video_fallback` that spec §3.5 forbids as the main result.
- The AdaFace/SCRFD identity audit was **not run**. There is no source-derived candidate grouping to validate (see `siw_subject_recovery/ADAFACE_AUDIT_STATUS.md`).
