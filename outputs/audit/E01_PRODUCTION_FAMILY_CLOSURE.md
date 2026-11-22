# E01 production family closure

Status: `E01_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`

Storage: `/home/student20261/workdir/GPAT_TransferBench_runtime` on /dev/nvme0n1p3 (ext4 UUID `227ce2f6-373e-4f13-9cdb-84b911775706`, mount `/`)
Launch commit `0ca57e3bcd77e1a0c54004c7fd4c21e5fe101822` (candidate `d995f556a8adcc5a032ba266a022b470117d1c1c`), config `87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0`

| seed | run_id | outputs | bank manifest SHA256 | audit |
|---|---|---|---|---|
| 42 | `80ba99c10c270fc7` | 8838 | `abd93be76b2659563599fb3cbcb5afaa72b56a7465b52db1e9e75bc1d6873e5f` | 52/52 |
| 1337 | `84f9df641837ea8f` | 8838 | `38a8c6c5caa7735da475862636ce856dafb91660ba8b1cb43431e23e69017a4b` | 52/52 |
| 2026 | `006c7827c1a10965` | 8838 | `ff7361426c6af222b11db0c86e7ac71c5601ae4f06f83180119525350ec35933` | 52/52 |

| seed | level-0 no-op copies | duplicate-pixel groups |
|---|---|---|
| 42 | 857 | 132 |
| 1337 | 872 | 136 |
| 2026 | 889 | 137 |

Disclosures:
- Official RizhaoCai/FAS-Aug@0da1dd79 operators, round-robin operator_set[position % 8] over the frozen pair order (owner ruling Q2), per-pair operator seed SHA256(pair_id + seed) mod 2^31 (A2-05), NUM_MAG 10 grid.
- Magnitude level 0 is retained: some outputs are unchanged copies of the live target (per_seed level_zero_noop_count; each was verified pixel-identical to its target); they are not substituted or removed.
- Duplicate output pixels: frozen pairs reuse live targets, so level-0 copies of the same target are identical (per_seed duplicate_output_pixel_groups); every row is kept, as spec section 7 requires.
- Cross-environment variation: in M6 qualification 3/72 synthetic Hand_Trembling outputs differed by <= 1 LSB between the laptop .venv and gpat-m5. These banks were generated and replayed only under gpat-m5, and byte identity on other stacks is not claimed.
- The bank is non-learned. No checkpoint exists, and generator_checkpoint_sha256 is null by design.

Remaining:
- M8+ downstream evaluation of these banks (ResNet-18 per the approved evaluator contract; DINOv3 needs its frozen config and pinned weights): separate owner authorization required
- TEST evaluation: not authorized; no TEST row or TEST face was read
- outputs/audit/method_status.csv is unchanged; updating it needs a separately verified registry-authority decision
- learned-method banks (E03-E07) and GPAT are outside this work
