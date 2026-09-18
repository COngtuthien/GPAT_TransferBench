# Decoder Index Audit — DEV-006 / Q-04 (2026-09-18)

**Question:** after a failed decode at original index f, are later successfully decoded frames
still assigned their correct original index (no backward shift)?

**Answer:** yes, for every video in the local datasets, using the index-bounded rule of
`gpatbench/data/frames.py` (one sequential `read()` per original index in [0, N_declared)).
The evidence below is FACT (measured). Scripts: `tools/audit_decoder_index.py` (targeted +
`--global`). Raw data was opened read-only.

## Pinned decoders

| Role | Package | FFmpeg libs | API |
|---|---|---|---|
| Inventory decoder (pinned) | opencv-python-headless 5.0.0.93 | avcodec 62.28.101, avformat 62.12.101 | `cv2.VideoCapture(path, cv2.CAP_FFMPEG)`, sequential `read()` |
| Independent reference | PyAV 18.1.0 (separate FFmpeg build) | avcodec 62.28.102, avformat 62.12.102, avutil 60.26.102 | demux packets, presentation index = PTS rank; per-packet decode, `thread_type="NONE"` |

The audit environment is separate from the inventory environment: a venv kept **outside** the project root (session scratchpad; recreate from the lock), lock
`environments/m1_decoder_audit_laptop.lock.txt`. The inventory venv and its lock are unchanged.
**Limitation:** both decoders are builds of the same FFmpeg release line, so they are
independent *builds and APIs* but not independent *codec implementations*. No `ffmpeg`/`ffprobe`
binary exists on this laptop.

## A. Targeted audit (per-frame content alignment) — `decoder_index_audit.csv`

For each OpenCV-valid index i, a 64×48 grayscale thumbnail is compared with PyAV frames at
presentation index i+d, d ∈ [−3, 3]. Continuity holds at i if d = 0 is the best match.

| Video | Declared | Packets | OpenCV failed | PyAV decode errors | Aligned (unique d=0) | Static/ambiguous | Misaligned | Verdict |
|---|---|---|---|---|---|---|---|---|
| MSU attack_client008_laptop_SD_ipad_video | 300 | 300 | [127] | [127] | 299 | 0 | 0 | CONTINUITY_PROVEN |
| MSU attack_client023_laptop_SD_iphone_video | 301 | 301 | [247] | [247] | 298 | 2 | 0 | CONTINUITY_PROVEN |
| MSU attack_client028_laptop_SD_printed_photo (FFmpeg error lines) | 205 | 205 | – | – | 201 | 4 | 0 | CONTINUITY_PROVEN |
| MSU attack_client049_laptop_SD_printed_photo (error lines) | 213 | 213 | – | – | 211 | 2 | 0 | CONTINUITY_PROVEN |
| MSU attack_client051_laptop_SD_printed_photo (error lines) | 211 | 211 | – | – | 209 | 2 | 0 | CONTINUITY_PROVEN |
| MSU attack_client053_laptop_SD_ipad_video (error lines) | 301 | 301 | – | – | 301 | 0 | 0 | CONTINUITY_PROVEN |
| MSU real_client001_laptop_SD (control) | 301 | 301 | – | – | 299 | 2 | 0 | CONTINUITY_PROVEN |
| SiW Live/Live_889.mp4 (trailing) | 111 | **108** | [108,109,110] | – | 108 | 0 | 0 | CONTINUITY_PROVEN |

"Static/ambiguous" means the neighbouring frames are nearly identical, so the offset cannot be
distinguished by content alone. It is never a misalignment.

### Neighbourhood of the failures (from the CSV)

| Video | index | OpenCV | POS_FRAMES before→after | POS_MSEC after | PyAV | PyAV PTS ms | best offset / diff |
|---|---|---|---|---|---|---|---|
| client008 | 126 | OK | 126→127 | 4199.200 | OK | 4199.200 | 0 / 0.000 |
| client008 | **127** | **READ_FAILED** | 127→**127** | 0.0 | **ERROR** | – | – |
| client008 | 128 | OK | **127→128** | 4265.867 | OK | 4265.867 | 0 / 0.000 |
| client008 | 129 | OK | 128→129 | 4299.200 | OK | 4299.200 | 0 / 0.000 |
| client023 | 246 | OK | 246→247 | 8172.067 | OK | 8172.067 | 0 / 0.000 |
| client023 | **247** | **READ_FAILED** | 247→**247** | 0.0 | **ERROR** | – | – |
| client023 | 248 | OK | **247→248** | 8238.733 | OK | 8238.733 | 0 / 0.000 |

**Interpretation (FACT):**
1. The failure at 127/247 is a real packet decode failure; the independent PyAV decode fails on the same packet.
2. A failed `read()` consumes exactly one frame position. The next successful read returns the frame whose PTS belongs to index f+1 (4265.867 ms = PyAV PTS of index 128; 33.333 ms spacing), and its pixels equal PyAV's frame f+1 (thumbnail diff 0.000 at offset 0).
3. **OpenCV's `CAP_PROP_POS_FRAMES` is NOT trustworthy after a failure.** It does not advance on the failed read, so it then lags the true index by 1 (it reports 128 after returning original frame 128). Any implementation that used `POS_FRAMES` as the frame index would shift every later frame backwards by one. GPAT-TransferBench does not use it: the index is the loop counter over [0, N_declared). This finding is recorded so M2 does not re-introduce it.

## B. Global audit (all 1,980 MSU + SiW videos) — `decoder_index_global.csv`

For every video and every OpenCV-valid index i: `POS_MSEC` after the read vs the PyAV packet PTS at
presentation rank i (tolerance 1 ms). Runtime 5 min 59 s with 8 workers.

| Result | Count |
|---|---|
| Videos checked | 1980 (280 MSU + 1700 SiW) |
| `INDEX_TIMESTAMPS_CONSISTENT` | **1980** |
| `INDEX_TIMESTAMP_MISMATCH` | **0** |
| Videos with a failed read inside the stream | 2 (MSU client008 @127, client023 @247) |
| Videos with trailing failed indices only | 137 (all SiW) |
| …of which n_failed == N_declared − N_packets | **137/137** |
| Declared > packets, by dataset / difference | SiW: +2 (22 videos), +3 (49), +4 (48), +5 (18) |

**Interpretation (FACT):** the 137 "trailing" invalid indices do not exist in the container. The
container header over-declares the frame count by 2–5, and the demuxer holds exactly
N_declared − n_failed packets. These indices are correctly marked invalid and are never sampled
(they lie outside the valid range [lo, hi]).

## C. Decision

- The index-bounded rule preserves original indices across mid-stream failures (proven for both
  cases and for timestamps of every valid frame in every video).
- No video needs to be BLOCKED for indexing.
- DEV-006 → recommended **APPROVED** (see deviation_report.md). The M2 contract must reuse the loop-counter
  indexing and the pinned decoder, or re-run this audit.
