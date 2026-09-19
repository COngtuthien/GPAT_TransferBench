# M2 — Cache Integrity Audit

Overall: **PASS**. Block-level verification: yes.

| field | entries | shards | bytes | duplicates | orphan shards | bad block hashes | bad shape/dtype | non-finite | shard hash failures | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| `parsing_logits` | 20,615 | 81 | 32.32 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `parsing_mask` | 20,615 | 81 | 0.97 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `landmarks_norm` | 20,615 | 81 | 0.01 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `landmarks_px224` | 20,615 | 81 | 0.01 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `landmarks_px256` | 20,615 | 81 | 0.01 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `pose_pitch_yaw_roll_rad` | 20,615 | 81 | 0.00 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| `embedding` | 20,615 | 81 | 0.04 GiB | 0 | 0 | 0 | 0 | 0 | 0 | PASS |

| accounting | value |
|---|---|
| expected (frozen M1 inventory) | 20,640 |
| COMPLETE | 20,615 |
| FAILED | 25 |
| missing state (unaccounted) | 0 |
| COMPLETE samples missing a cache entry | 0 |
| orphan cache rows (no COMPLETE sample) | 0 |
| COMPLETE samples missing a face file | 0 |

Per dataset:

| dataset | COMPLETE | FAILED |
|---|---|---|
| casia_fasd | 4,800 | 0 |
| msu_mfsd | 2,240 | 0 |
| siwmv2 | 13,575 | 25 |

Failure classes:

- `SCRFD_NO_FACE`: 25
