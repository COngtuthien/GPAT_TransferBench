# M1 Dataset Evidence (qualitative findings behind adapter rules)

Measured counts live in the generated artifacts (`dataset_report.html`, `dataset_*.csv`,
`manifests/*.parquet`). This file records **how** each rule was established. No face images are
stored in the repository. Visual spot checks were rendered to the session scratchpad only;
SiW-Mv2 images were not viewed because its DRA forbids displaying subject faces.

## 1. CASIA-FASD (local copy = repackaged face-crop image sequences)

| Finding | Evidence | Consequence |
|---|---|---|
| No documentation ships with the local copy | Every file under the root is a `.png`; `casia-fasd.zip` holds the same 123,533 PNGs and no other file | Code → attack-concept mapping is not locally documented, so all CASIA spoof codes are UNMAPPED (proposals in `attack_map_v1.yaml`) |
| Frames are 112×112 face crops, not the original videos | `cv2` decode of sample frames returned (112,112,3); per-video sizes in `inventory_videos.parquet` `frame_sizes` | Spec §4 SCRFD on the full frame then a 1.25× crop cannot be applied to original frames. **Owner question for M2** |
| Filename grammar `s{N}v{CODE}f{F}.png` under `{train\|test}/{live\|spoof}/`, CODE ∈ {1..8, HR_1..HR_4} | Exhaustive regex over all 123,533 names: every name matches the canonical or derived pattern | Adapter fails loudly on any other name |
| Subject numbers restart per native partition | train has s1–s20 and test has s1–s30. A visual check of **all 20** `train sN` vs `test sN` pairs showed different people; several pairs differ in apparent sex (s3, s7, s10, s17, s18). This agrees with the published CASIA-FASD structure (20 train + 30 test = 50 subjects, external context) | `subject_id_raw = {partition}_s{N}`. The bare number would merge different people. The PRISM adapter used the bare number: REJECTED (see PRISM_REUSE_AUDIT.md) |
| Within a partition, `sN` is one person across all codes | Visual check: `s1` train live, v3, v5, v7, HR_1, HR_2, HR_4 show the same person, and the same holds for test | Subject key is valid within a partition |
| `fs*` = exact horizontal flip of `s*`; `bs*` = brightened `s*` | Measured on every derived file (`m1_inventory_facts.json` → `derived_copy_checks`). Frame-index sets of `s`/`bs`/`fs` per video are identical. They exist only in `train/live/` | DERIVED_AUGMENTATION_COPY: indexed, never sampled. Including them would triplicate train-live videos and break sample semantics |
| HR_1 stored under `spoof/` | The published CASIA-FASD protocol lists HR_1 as the high-quality **genuine** video (external). A visual check of 5 subjects × train/test showed natural, talking, live-looking faces for HR_1, unlike HR_2 (printed photo) | `label_binary` keeps the local folder value (1) and `label_conflict` is set. **Owner must decide the label** before M3 |
| Visual content agrees with the published code order | v5/v6 and HR_3 show cut-out eye holes (cut photo); v3 and HR_2 show flat paper prints; v7 and HR_4 show screen replays | Supports the proposals in `attack_map_v1.yaml`; they are not applied |

## 2. MSU-MFSD

- The selected root is a **directory** named `MSU-MFSD-Publish.zip` holding the extracted release: README.txt, `scene01/{real,attack}`, `train_sub_list.txt`, `test_sub_list.txt`, `DecFrames*.m`, `ffmpeg/`. The multi-part `.zip.001–.016` pieces beside it are not used or cross-checked.
- All video names match the README naming protocol exactly: 35 clients × (2 real + 6 attack) = 280 videos, each with a `.face` annotation file.
- Native subject lists are disjoint (train 15 / test 20). They are kept only as `native_protocol_split`.
- Attack tokens `printed_photo`, `ipad_video` and `iphone_video` are defined in README §4. They map via spec §3.3 (MSU printed photograph → print; replay/video → replay).
- The trial run showed FFmpeg logging ProRes decode errors on some laptop `.mov` files. These are captured per video as `decoder_error_lines` (see §4).

## 3. SiW-Mv2

- **Correction of an M0 note.** M0 recorded "both `Live/Spoof` and `live/spoof`". The M0 command was `ls ".../SiW-Mv2/SiW-Mv2" | head; ls ".../casia-fasd/train" | head -5`: the lowercase `live`/`spoof` lines were the output of the **second** command (CASIA `train/`). In M1, `ls -la` of the SiW root shows only `DRA.pdf Live README.pdf Spoof`, `ls live` fails ("No such file"), and the inventory's case-insensitive collision check reports none. The filesystem is ntfs3 (mount shows case-sensitive lookups for these names). No duplicate case directories exist.
- 785 files in `Live/` and 915 in `Spoof/` across 14 attack folders. This matches README.pdf ("785 videos … 915 spoof videos").
- Subject identity is **not recoverable**. Names are `Live_{n}` / `{Prefix}_{n}`, a per-class sequence number. README.pdf reports 785 live videos from 493 subjects, so `n` is not a subject. README.pdf says "specific naming of each samples can be found at this link", and that link is not local. `SiW-Mv2.zip` uses the same names. PRISM's adapter also had `subject_id=None`.
  → `BLOCKED_BY_MISSING_SUBJECT_ID` for the M3 main split of SiW-Mv2. No heuristic applied.
- 13 of the 14 folder names match a spec §3.3 concept literally. `Paper` is the exception (probably "print", but ambiguous with "paper mask"), so it is left UNMAPPED pending the owner's decision.
- DRA §4: subject faces must not be published or displayed; "no attempt … to determine the identity of a subject". The report therefore contains no images, and no face-identity method is used to recover SiW subjects.

## 4. Q-04 valid frame — evidence

- Declared vs decoded frame counts differ in practice: SiW `Live/Live_889.mp4` declares 111 frames and decodes 108, with both OpenCV 4.6.0 (apt) and 5.0.0 (pip). The declared count cannot be trusted.
- A failed read can be followed by decodable frames. MSU client008 (laptop, iPad) decodes 127 good, 1 failed, then 172 good frames, totalling the declared 300; client023 decodes 247 + 1 + 53 = 301. The video rule therefore treats a failed read as one invalid position and continues (end of stream = 5 consecutive failures).
- FFmpeg can return frames while logging decode errors (ProRes "ac tex damaged", "invalid plane data size" in the trial run). Such frames are error-concealed, and OpenCV's API cannot flag them per frame. The inventory therefore counts stderr lines per video (`decoder_error_lines`) and raises a WARNING issue, so these videos are visible for review rather than silently accepted.

---

## 5. M1 correction pass (2026-09-18) — appended; the sections above are kept as the original record

### 5.1 CASIA-FASD (supersedes the §1 rows on subject identity and HR_1 label)

- **Code semantics (owner-provided protocol evidence):**
  - live: 1 real_normal, 2 real_low, HR_1 real_high;
  - print: 3 warped_normal, 4 warped_low, HR_2 warped_high, 5 cut_normal, 6 cut_low, HR_3 cut_high;
  - replay: 7 video_normal, 8 video_low, HR_4 video_high.

  → HR_1 is LIVE. The local packager's `spoof/` placement is overridden and each override is logged (`FOLDER_LABEL_OVERRIDDEN`). The §1 visual observation ("HR_1 looks live") agrees.
- **Subject identity:** canonical identities are train 1..20 and test 21..50, so train sN → "N" and test sN → "20+N". Checked against all 600 local sequences:
  - train has exactly s1..s20 and test exactly s1..s30;
  - each subject has all 12 codes;
  - the result is 50 unique ids.

  No local evidence contradicts the mapping. The §1 visual check (train sN ≠ test sN) is consistent with it. `{partition}_s{N}` (DEV-008) is withdrawn.
- **Original-video search (Q-15):** none found locally. There are three copies of the same 112px PNG repack:
  - the selected root;
  - `casia-fasd.zip`;
  - `/media/cong/Data/AI on IOT/Anti_spoofing/PRISM_FAS_C_LLM_Project/data/raw/casia_fasd` (123,533 PNG; 20/20 random spot-check byte-identical).

  An old PRISM-FAS-B note (`reports/m2/casia_media_verification.md`) shows that project grouped `bs*`/`fs*` files into the canonical sequence (e.g. "`s10v1` begins at `train/live/bs10v1f0.png`"). GPAT-TransferBench does not do this (DEV-007).

### 5.2 SiW-Mv2 (official sources checked)

- **Official repository** github.com/CHELSEA234/Multi-domain-learning-FAS, pinned HEAD `8667dbcd316b38141729c057adf7517fe0602608` (git ls-remote, 2026-09-18). Files fetched to the session scratchpad only; nothing vendored.
- **`Paper` token → print:**
  - `source_SiW_Mv2/config_siwm.py` L63–69 `spoof_type_dict` contains `'Paper': 'Mask_Paper'` and `'Print': 'Paper'` (sha256 `ebe37e87…`);
  - `csv_parser.py` L145–177 puts `Paper` videos in `print_list`, after routing `Mask_Paper` and `Partial_Paperglass` elsewhere (sha256 `59159f5d…`);
  - the README folder list includes both `Print` and `Mask_PaperMask`;
  - ECCV'22 supplementary Table 1 (sha256 `b9dba0ff…`) gives video counts per type that equal the local folder counts for **all 14** types (Print 135 = `Paper` 135; Paper Mask 17 = `Mask_PaperMask` 17; …). It calls the 72-video type "Full Mask", which is the local `Mask_HalfMask`.
- **Subject IDs:** still unrecoverable.
  - The official protocol lists contain video names only; the README calls them "subject names", repeated "for balancing".
  - The supplementary gives only per-type subject counts.
  - No mapping file ships with the dataset or the repo.
  - PRISM's audit JSON records `all_frame_subject_null: true`.

  → `BLOCKED_BY_MISSING_SUBJECT_ID` for the SiW-Mv2 M3 main split. No heuristic applied.
