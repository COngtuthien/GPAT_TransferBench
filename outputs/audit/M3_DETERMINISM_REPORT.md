# M3 — Determinism Report

Status: **PASS**.

The whole allocator was re-executed in **fresh interpreters** into clean temporary directories, including a run whose input metadata order was deliberately permuted and runs under different `PYTHONHASHSEED` values. A manifest is accepted only if the Parquet bytes are identical, not merely the assignment.

| run | fresh process | metadata order | PYTHONHASHSEED | rows | manifest sha256 |
|---|---|---|---|---|---|
| authoritative | False | canonical | unset | 20,615 | `fb9aeb369a124fc96ba855ef2ce26923…` |
| rerun_A_canonical_order | True | canonical | 0 | 20,615 | `fb9aeb369a124fc96ba855ef2ce26923…` |
| rerun_B_shuffled_metadata_order | True | shuffled (seed 20260814) | 1 | 20,615 | `fb9aeb369a124fc96ba855ef2ce26923…` |
| rerun_C_shuffled_other_seed | True | shuffled (seed 7) | 12345 | 20,615 | `fb9aeb369a124fc96ba855ef2ce26923…` |

- all manifest hashes identical: **True**
- all group-manifest hashes identical: **True**
- all allocator objectives identical: **True**
- byte-identical split manifest: **True**

Serialization is pinned by the frozen writer settings recorded in `M3_SPLIT_REPORT.md`; row order is frozen as `dataset > video_id > frame_index > sample_id` and is applied inside the writer, so input row order cannot reach the file.
