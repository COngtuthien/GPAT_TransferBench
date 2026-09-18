# Frozen Config Revision Log

Spec §0.1 rule 1 / owner rule §23: frozen configs are never silently overwritten. Every revision keeps the previous file under `frozen_config_snapshot/history/`, with its hash and the exact diff.

| Config | Revision | Commit | sha256 | Used by scientific execution? |
|---|---|---|---|---|
| data_v1.yaml | 1 | ce12a58 | `a22eeeecf73f595731f9ed06368089ec5795c17c103f26ef93f8887d363cb9d8` | M1 inventory runs B/C (outputs superseded in 112506b); run A used an earlier uncommitted draft (valid_frame text differed); no M2+ run |
| data_v1.yaml | 2 (current) | 112506b | `7227c1bd589a29750495971899f0caf685976585d8610dbcbdeb77a730902885` | M1 inventory runs D/E (current manifests); no M2+ run |
| attack_map_v1.yaml | 1 | ce12a58 | `ca8791ac3f9111568fc66302fefd1a38cbdf74f79925409ebb64dae65bb5f3b1` | M1 trial run, run A and runs B/C (all superseded); no M2+ run |
| attack_map_v1.yaml | 2 (current) | 112506b | `3755a479de9c57a4d92db6a5370bff2b902511ac88395d84ce5ae94212419c0d` | M1 inventory runs D/E; no M2+ run |

Reasons: revision 2 of both files implements owner decisions DEV-010 / Q-12 / Q-13 and the revised Q-04 index-bounded decoder rule (DEV-006). No revision is made in the dataset-resolution pass: the current files stay at revision 2.

## Exact diffs (rev1 → rev2)

### data_v1.yaml

```diff
@@ -1,4 +1,4 @@
-# data_v1 — frozen data configuration (spec §23: dataset roots, 8-frame policy, preprocessing, label mapping).
+# data_v1 — frozen data configuration (revision 2: M1 correction pass; revision 1 = commit ce12a58) (spec §23: dataset roots, 8-frame policy, preprocessing, label mapping).
 # Scientific values are copied from the spec; absolute paths are machine bindings (laptop), not
 # hyperparameters. Items marked M1_INTERPRETATION are operational decisions recorded in
 # outputs/audit/deviation_report.md for owner review.
@@ -43,13 +43,14 @@ frame_sampling:                     # spec §3.4
     already chosen, processing k in order; ties -> smaller index
   extraction: deferred_to_M2   # canonical lossless PNG extraction happens in M2
 
-valid_frame:                        # Q-04, M1_INTERPRETATION (decoder robustness)
-  image_sequence: "frame index = filename frame number; valid iff the file decodes (cv2.imdecode IMREAD_COLOR) to a non-empty image"
+valid_frame:                        # Q-04 (DEV-006 revised): a valid frame is an original frame/index that
+                                    # successfully decodes into a non-empty image with the pinned decoder
+  image_sequence: "index = filename frame number of an existing source file; valid iff cv2.imdecode(IMREAD_COLOR) returns a non-empty image"
   video_file: >-
-    frame index = 0-based decode position with cv2.VideoCapture(path, cv2.CAP_FFMPEG), one position per read();
-    position i is valid iff read() returns ok and a non-empty array; a failed read marks position i invalid and
-    decoding continues; the stream ends after 5 consecutive failed reads (those trailing failures are not positions);
-    the container-declared frame count is recorded, not trusted; FFmpeg stderr lines are counted per video
+    index-bounded: original indices 0..N_declared-1 from the container (cv2.CAP_PROP_FRAME_COUNT); one sequential
+    read() per index with cv2.VideoCapture(path, cv2.CAP_FFMPEG); success + non-empty => valid, failure => invalid,
+    index preserved and decoding continues; one extra read() after the range is evidence only (frames_beyond_declared,
+    never sampled); N_declared <= 0 => DECLARED_COUNT_UNAVAILABLE (no valid frames, ERROR); FFmpeg stderr lines counted per video
   orientation: "OpenCV default CAP_PROP_ORIENTATION_AUTO (recorded; affects pixels in M2, not indices)"
 
 decoder:                            # spec §3.4 'decode with the same backend and record decoder/version'
```

### attack_map_v1.yaml

```diff
@@ -1,13 +1,31 @@
 # attack_map_v1 — frozen attack-label harmonization (spec §3.3).
-# Only tokens whose mapping traces to SPEC and/or DATASET_DOCUMENTATION/local dataset tokens are
-# under `mapped`. Observed tokens without such a trace are under `unmapped_pending_approval`:
-# the inventory assigns them attack_macro=other_spoof (binary benchmark only) and the
-# Native/Full track is BLOCKED for them until the owner approves an explicit mapping here.
+# Every mapped token traces to SPEC plus DATASET_DOCUMENTATION / official source / owner-provided
+# protocol evidence. Unmapped spoof tokens (none at this revision) would get attack_macro=other_spoof
+# for the binary benchmark only, with Native/Full BLOCKED for them.
 # Any change to this file requires an owner-approved entry in outputs/audit/deviation_report.md.
 version: attack_map_v1
+revision: 2
+revision_history:
+  - {revision: 1, commit: ce12a58fc9e0aca7b3cf56255d73cb85d696b36a, status: PARTIAL_PENDING_OWNER_APPROVAL,
+     note: "MSU 3 + SiW 13 mapped; CASIA 10 codes and SiW 'Paper' pending"}
+  - {revision: 2, commit: "M1 correction pass (this commit)", status: COMPLETE_FOR_OBSERVED_TOKENS,
+     note: "CASIA codes mapped from owner-provided protocol evidence (Q-12/Q-13 RESOLVED); SiW 'Paper' -> print traced to official code (Q-13 RESOLVED)"}
 spec_sha256: f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e
-status: PARTIAL_PENDING_OWNER_APPROVAL
+status: COMPLETE_FOR_OBSERVED_TOKENS
 mapped:
+  casia_fasd:
+    # Code semantics: owner-provided external CASIA-FASD protocol evidence (M1 review, 2026-09-18):
+    # 1 real_normal, 2 real_low, HR_1 real_high (LIVE, not attack tokens); spoof codes below.
+    # Visual spot check (M1_DATASET_EVIDENCE.md §1) is consistent: cut-eye holes in 5/HR_3, prints in 3/HR_2, screens in 7/HR_4.
+    "3":  {attack_macro: print,  semantics: warped_normal, trace: ["SPEC §3.3: 'CASIA warped printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: 3 = warped_normal"]}
+    "4":  {attack_macro: print,  semantics: warped_low,    trace: ["SPEC §3.3: 'CASIA warped printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: 4 = warped_low"]}
+    HR_2: {attack_macro: print,  semantics: warped_high,   trace: ["SPEC §3.3: 'CASIA warped printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: HR_2 = warped_high"]}
+    "5":  {attack_macro: print,  semantics: cut_normal,    trace: ["SPEC §3.3: 'CASIA cut-eye printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: 5 = cut_normal"]}
+    "6":  {attack_macro: print,  semantics: cut_low,       trace: ["SPEC §3.3: 'CASIA cut-eye printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: 6 = cut_low"]}
+    HR_3: {attack_macro: print,  semantics: cut_high,      trace: ["SPEC §3.3: 'CASIA cut-eye printed photograph -> print'", "OWNER_PROTOCOL_EVIDENCE: HR_3 = cut_high"]}
+    "7":  {attack_macro: replay, semantics: video_normal,  trace: ["SPEC §3.3: 'CASIA video attack -> replay'", "OWNER_PROTOCOL_EVIDENCE: 7 = video_normal"]}
+    "8":  {attack_macro: replay, semantics: video_low,     trace: ["SPEC §3.3: 'CASIA video attack -> replay'", "OWNER_PROTOCOL_EVIDENCE: 8 = video_low"]}
+    HR_4: {attack_macro: replay, semantics: video_high,    trace: ["SPEC §3.3: 'CASIA video attack -> replay'", "OWNER_PROTOCOL_EVIDENCE: HR_4 = video_high"]}
   msu_mfsd:
     printed_photo:
       attack_macro: print
@@ -22,10 +40,16 @@ mapped:
       trace: ["SPEC §3.3: 'MSU replay/video attack -> replay'",
               "DATASET_DOCUMENTATION README.txt §4: 'mobile phone replay video attacks using an iPhone 5S screen'; attackType 'iphone_video'"]
   siwmv2:
-    # Local token = spoof folder name (dataset-native). Each entry is a literal name match to a spec concept.
+    # Local token = spoof folder name. Official source: github.com/CHELSEA234/Multi-domain-learning-FAS
+    # @ 8667dbcd316b38141729c057adf7517fe0602608 (source_SiW_Mv2/config_siwm.py spoof_type_dict,
+    # csv_parser.py) and ECCV'22 supplementary Table 1 (video counts equal local folder counts for all 14 types).
     Replay:                  {attack_macro: replay,  trace: ["SPEC §3.3: 'SiW-Mv2 replay -> replay'", "LOCAL_TOKEN folder 'Replay'"]}
-    Mask_PaperMask:          {attack_macro: mask_2d, trace: ["SPEC §3.3: 'SiW-Mv2 paper mask -> mask_2d'", "LOCAL_TOKEN folder 'Mask_PaperMask'"]}
-    Mask_HalfMask:           {attack_macro: mask_3d, trace: ["SPEC §3.3: 'SiW-Mv2 half mask -> mask_3d'", "LOCAL_TOKEN folder 'Mask_HalfMask'"]}
+    Paper:                   {attack_macro: print,   trace: ["SPEC §3.3: 'SiW-Mv2 print -> print'",
+                                                             "OFFICIAL_CODE config_siwm.py L67 / csv_parser.py L149 @8667dbc: spoof_type_dict 'Print': 'Paper' (paper mask is separately 'Paper': 'Mask_Paper'); csv_parser appends 'Paper' videos to print_list after excluding Mask_Paper and Partial_Paperglass",
+                                                             "OFFICIAL_README source_SiW_Mv2/README.md: dataset folder list contains 'Print' and 'Mask_PaperMask' as distinct types",
+                                                             "SUPPLEMENTARY Table 1: Print = 135 videos = local 'Paper' folder count; Paper Mask = 17 = local 'Mask_PaperMask' count"]}
+    Mask_PaperMask:          {attack_macro: mask_2d, trace: ["SPEC §3.3: 'SiW-Mv2 paper mask -> mask_2d'", "LOCAL_TOKEN folder 'Mask_PaperMask' (file prefix 'Mask_Paper' = official 'Paper' mask type)"]}
+    Mask_HalfMask:           {attack_macro: mask_3d, trace: ["SPEC §3.3: 'SiW-Mv2 half mask -> mask_3d'", "LOCAL_TOKEN folder 'Mask_HalfMask' (supplementary Table 1 names this 72-video type 'Full Mask'; macro unchanged)"]}
     Mask_TransparentMask:    {attack_macro: mask_3d, trace: ["SPEC §3.3: 'SiW-Mv2 transparent mask -> mask_3d'", "LOCAL_TOKEN folder 'Mask_TransparentMask'"]}
     Silicone:                {attack_macro: mask_3d, trace: ["SPEC §3.3: 'SiW-Mv2 silicone mask -> mask_3d'", "LOCAL_TOKEN folder 'Silicone' (file prefix 'Mask_Silicone')"]}
     Mannequin:               {attack_macro: mask_3d, trace: ["SPEC §3.3: 'SiW-Mv2 mannequin -> mask_3d'", "LOCAL_TOKEN folder 'Mannequin' (file prefix 'Mask_Mann')"]}
@@ -36,20 +60,4 @@ mapped:
     Makeup_Cosmetic:         {attack_macro: makeup,  trace: ["SPEC §3.3: 'SiW-Mv2 cosmetic makeup -> makeup'", "LOCAL_TOKEN folder 'Makeup_Cosmetic'"]}
     Makeup_Impersonation:    {attack_macro: makeup,  trace: ["SPEC §3.3: 'SiW-Mv2 impersonation makeup -> makeup'", "LOCAL_TOKEN folder 'Makeup_Impersonation'"]}
     Makeup_Obfuscation:      {attack_macro: makeup,  trace: ["SPEC §3.3: 'SiW-Mv2 obfuscation makeup -> makeup'", "LOCAL_TOKEN folder 'Makeup_Obfuscation'"]}
-  casia_fasd: {}
-unmapped_pending_approval:
-  casia_fasd:
-    # The local CASIA copy ships no documentation; code -> concept requires the published CASIA-FASD
-    # protocol (external). Proposals below are NOT applied.
-    "3":    {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: codes 3,4 = warped printed photo (external); visual spot check consistent"}
-    "4":    {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: codes 3,4 = warped printed photo (external)"}
-    "5":    {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: codes 5,6 = cut-eye printed photo (external); visual spot check shows cut eye holes"}
-    "6":    {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: codes 5,6 = cut-eye printed photo (external)"}
-    "7":    {proposed_attack_macro: replay, basis: "published CASIA-FASD protocol: codes 7,8 = video replay (external)"}
-    "8":    {proposed_attack_macro: replay, basis: "published CASIA-FASD protocol: codes 7,8 = video replay (external)"}
-    HR_1:   {proposed_attack_macro: null,   basis: "LABEL CONFLICT: published protocol lists HR_1 as genuine high-resolution video, but local copy stores it under spoof/; visual spot check looks live. Owner must decide the label before any mapping."}
-    HR_2:   {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: HR_2 = high-res warped printed photo (external)"}
-    HR_3:   {proposed_attack_macro: print,  basis: "published CASIA-FASD protocol: HR_3 = high-res cut-eye photo (external); visual spot check shows cut eye holes"}
-    HR_4:   {proposed_attack_macro: replay, basis: "published CASIA-FASD protocol: HR_4 = high-res video replay (external)"}
-  siwmv2:
-    Paper:  {proposed_attack_macro: print,  basis: "the only local folder without a literal spec concept match; spec concept 'SiW-Mv2 print' is the only unmatched concept (14 folders <-> 14 concepts). Name 'Paper' is ambiguous with 'paper mask', so not applied without owner approval."}
+unmapped_pending_approval: {}
```
