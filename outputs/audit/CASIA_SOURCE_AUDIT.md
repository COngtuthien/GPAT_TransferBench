# CASIA-FASD Source Audit (Q-15) — 2026-09-18, read-only

## 1. Independent re-audit of the selected source (FACT)

`tools/audit_casia_source.py` walks the selected root directly, reads PNG headers, and re-derives
semantics from filenames only, without trusting generated manifests → `casia_source_audit.json`.

| Check | Result |
|---|---|
| Files parsed by the canonical/derived grammar | 123,533 / 123,533 (0 unparsed) |
| PNG header of canonical frames | 110,859 × **112×112, 8-bit, RGB** |
| PNG header of derived `bs*`/`fs*` frames | 12,674 × 112×112, 8-bit, RGB |
| Canonical sequences | 600 |
| Subject numbers | train s1..s20, test s1..s30 |
| Every subject has all 12 codes | yes |
| Identity mapping train N → N, test N → 20+N | 50 unique ids, 1..50 |
| Live / spoof sequences (code rule 1, 2, HR_1 live) | 150 / 450 |
| Spoof macro counts | print 300, replay 150 |
| Folder vs code disagreements | 50 (all `spoof/…HR_1`) |
| Mismatches vs `manifests/inventory_videos.parquet` | **0** |

## 2. Search for an original (uncropped / video) CASIA-FASD copy

Scope: `/home/cong`, `/media/cong/Data` (the only mounted data volume; `mount` shows one ntfs3
volume), old PRISM projects, dataset roots, and the contents of every large archive on the Data volume.

| Candidate | Source type | Files | Resolution sample | Relation to current repack | Usable as original? | Reason |
|---|---|---|---|---|---|---|
| `/media/cong/Data/AI on IOT/Anti_spoofing/Dataset/casia-fasd` (selected) | PNG frame sequences | 123,533 | all 112×112 RGB | — | NO | pre-cropped repack |
| `/media/cong/Data/AI on IOT/Anti_spoofing/Dataset/casia-fasd.zip` | zip of the same PNGs | 123,533 entries | (same) | 123,533/123,533 size+CRC32 identical | NO | identical content |
| `/media/cong/Data/AI on IOT/Anti_spoofing/PRISM_FAS_C_LLM_Project/data/raw/casia_fasd` | PNG frame sequences | 123,533 | 112×112 | 20/20 random files byte-identical | NO | another copy of the repack |
| PRISM configs `configs/data/casia_fasd.yaml` (PRISM-C, PRISM-B) | config only | — | — | pattern `(b\|f)?s…v…f….png` | NO | points to the same repack |
| `/media/cong/Data/AI and Application/Cuối kì/data.zip` | zip | 953 entries | — | 0 CASIA / `.avi` / `HR_*` entries | NO | unrelated |
| `Dataset/OULU_NPU.zip` | zip | 247,374 entries | — | 0 CASIA entries | NO | OULU-NPU |
| `Dataset/Anti-Spoofing-Replay-Dataset-main.zip` | zip | 64 entries | — | 0 CASIA entries | NO | unrelated |
| `Dataset/CelebA-Spoof-zips/` (76 parts) | split archive | — | — | not listable without joining; CelebA-Spoof by name | NO | different dataset |
| Any `*.avi` outside SiW-Mv2 on /home or /media | — | 1 dir (`AI and Application/Cuối kì`) | — | not CASIA | NO | unrelated coursework |
| GPU server copies | — | — | — | **not inspected** (SSH needs a password) | UNKNOWN | GPU audit PENDING |

**Result: NO original CASIA-FASD source exists on this laptop.** Q-15 needs either an original copy
(owner action, e.g. on the GPU server or re-download) or the controlled adaptation in
`CASIA_CONTROLLED_ADAPTATION_PROPOSAL.md`.

Not opened: `/media/cong/Data/Bosch Embedded Academy.rar` (1.5 GB; name unrelated; no `unrar` used). Archives under 100 MB were not listed.
