# SPEC PROVENANCE — GPAT-TransferBench v1.0

The frozen specification below is the **highest scientific source of truth** for this
project. If any instruction (prompt, task document, convenience) conflicts with it, the
spec wins and the conflict is recorded in `outputs/audit/deviation_report.md`.
The original file is never edited.

## Identity

| Field | Value |
|---|---|
| Project | GPAT-TransferBench |
| Spec version | 1.0 — August 2026 |
| Spec title (body) | GPAT-TRANSFERBENCH — Frozen Research, Reimplementation, Training, Evaluation and Paper-Evidence Specification |
| Spec title (page header) | GPAT-TransferBench v1.0 \| Frozen Research & Reimplementation Specification |
| Actual absolute path (primary reference) | `/home/cong/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` |
| Filename | `GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` |
| Size | 710996 bytes |
| SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| File mtime | 2026-08-14 16:20:55 +0700 |
| Date checked | 2026-09-18T04:15:38Z |
| Machine | cong-ThinkBook-16-G7-AHP |
| User | cong |

## Candidate search

Search: `find` over `/` (xdev), `/home`, `/mnt`, `/media` for `*GPAT*TransferBench*` and `*Frozen_Specification*`.

| # | Path | Size | SHA256 | Decision |
|---|---|---|---|---|
| 1 | `/home/cong/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` | 710996 | f7d23716…281489e | Selected as primary reference |
| 2 | `/media/cong/Data/AI on IOT/Anti_spoofing/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` | 710996 | f7d23716…281489e | Byte-identical duplicate (same hash, same mtime) |

No `(N)`-suffixed duplicates were found. Both candidates have the same SHA256 → **no conflict**.

## Content verification

| Check | Result | Evidence |
|---|---|---|
| Contains "GPAT-TransferBench" | PASS | body line 1, page header |
| "Version 1.0" | PASS | body: "Version 1.0 — August 2026"; header "v1.0" |
| "August 2026" | PASS | body line 3 |
| "Frozen Research & Reimplementation Specification" | PASS | `word/header1.xml` text |
| 33 pages | **UNVERIFIED** | No DOCX renderer installed (no LibreOffice/pandoc). `docProps/app.xml` says `Pages=1`, which is stale metadata written by python-docx (`core.xml` creator = python-docx) and is not a reliable page count. Document contains 1 explicit page break and 7 embedded figures. |
| Full text read | DONE | All sections 0–32 and Appendices A–C read from `docs/spec/spec_extracted_text.txt` (987 lines) |

## Project copy

| Field | Value |
|---|---|
| Copy path | `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` (chmod 444) |
| Copy SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` (cmp: byte-identical) |
| Extracted text | `docs/spec/spec_extracted_text.txt` — derived, non-authoritative; regenerate with `python3 tools/docx_extract.py <docx>` |
| Extraction method | Python stdlib `zipfile` + `xml.etree` (no third-party packages) |

The DOCX remains authoritative; the extracted text is a convenience view only.

## Related (non-authoritative) documents observed

| Path | SHA256 | Status |
|---|---|---|
| `/media/cong/Data/AI on IOT/Anti_spoofing/GPAT/GPAT-TransferBench Task Assignments.pdf` (9 pages, producer "ChatGPT Canvas") | 60b658c7f49882eeb9878e99f110b55bf717ba310600e8bd2c2f60a0ff252a2c | Task-assignment document (e.g. STDN reimplementation task). **Not** a source of truth; the frozen spec overrides it. Not copied into the project. |

## Update 2026-09-18 — page count (owner statement)

The owner states that SHA256 `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`
was independently confirmed byte-identical to the 33-page frozen specification.
**File identity matches the externally verified 33-page frozen specification.**
The local machine still has **not** rendered the document: the 33-page count was verified
externally (per the owner), not by a local renderer. The "UNVERIFIED" row above remains
accurate for local verification and is kept for history.
