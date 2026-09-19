# M2 — Failure Concentration Audit (canonical-video level)

M2 is COMPLETE and untouched. This is a read-only audit of where its 25 technical failures fall, so
that M3 can be designed on facts rather than on the assumption that failures are harmless.

All 25 failures are `SCRFD_NO_FACE` on SiW-Mv2 — the frozen no-fallback outcome
(no lowered threshold, no second detector, no centre crop, no neighbouring frame, no fabricated
geometry or identity). Per-video detail: `M2_FAILURE_CONCENTRATION.csv`.

## Hard gate: zero-usable canonical videos

**0** canonical videos have zero COMPLETE samples → gate **PASS**.

Every one of the 2,580 canonical videos retains at least one usable
sample, so no video is silently lost from the benchmark and no deletion/replacement policy is needed.

## Headline numbers

| | |
|---|---|
| expected samples | 20,640 |
| COMPLETE | 20,615 |
| FAILED | 25 |
| canonical videos (all) | 2,580 |
| affected canonical videos | 23 |
| max failed frames in one video | 2 |
| min COMPLETE frames in an affected video | 6 |

Failures are **diffuse, not concentrated**: they touch 23 distinct
videos out of 2,580, at most 2 frames each.

## Usable-frame counts per canonical video (all 2,580 videos)

| COMPLETE frames | videos |
|---|---|
| 8 | 2,557 |
| 7 | 21 |
| 6 | 2 |

Affected videos only:

| COMPLETE frames | videos |
|---|---|
| 7 | 21 |
| 6 | 2 |

## Failure distribution by label

By binary class (0 = live, 1 = spoof):

| label_binary | failed samples |
|---|---|
| 0 | 4 |
| 1 | 21 |

By `attack_macro`:

| attack_macro | failed samples |
|---|---|
| replay | 11 |
| print | 5 |
| live | 4 |
| partial | 3 |
| makeup | 2 |

By `attack_raw`:

| attack_raw | failed samples |
|---|---|
| Replay | 11 |
| Paper | 5 |
| (none) | 4 |
| Partial_FunnyeyeGlasses | 3 |
| Makeup_Impersonation | 2 |

## Failure distribution by content-group size

How many canonical videos share the exact raw bytes of each affected SiW video
(`content_group_id`, the frozen SiW allocation key). Size 1 means the video is its own group.

| content group size | affected videos |
|---|---|
| 1 | 21 |
| 2 | 2 |

No identity is inferred from SiW filenames anywhere in this audit.

## Consequence for M3

These are frame-level technical failures inside otherwise healthy canonical videos. They do not
change any video's group identity, and the M3 population contract (see `M3_ALLOCATOR_DESIGN.md`)
therefore balances **canonical videos**, not surviving frame counts, while only COMPLETE rows become
usable downstream image samples.
