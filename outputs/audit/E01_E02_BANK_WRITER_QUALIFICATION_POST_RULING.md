# E01/E02 bank-writer qualification after the owner ruling (synthetic fixture)

Date: 2026-11-22. Source commit `86dac00f77a5b5768412a57cf40bdb0afa12662a`; the tree was clean in both runs.

**Status: `E01_E02_BANK_WRITER_QUALIFIED_POST_RULING_SYNTHETIC`. 10/10 gates passed in the laptop `.venv` and 10/10 in the
locked `gpat-m5`.**

Scope:

- The runs used a synthetic fixture, the frozen manifests (metadata only) and pure recipes.
- No benchmark image was decoded and no production bank was written.
- Production stays refused with `E01_E02_BLOCKED_BY_M8_SMOKE_GATE`.

Authority:

- Owner ruling `configs/amendments/e01_e02_bank_milestone_ruling.yaml` (`4a89b4e1…a19e`, commit `656cc8a`).
- Gate correction `86dac00`.
- The pre-ruling records `E01_E02_BANK_WRITER_QUALIFICATION{,_GPAT_M5}.json` (`028b5c1`) are unchanged and serve as
  G10 references.

## Why a fresh qualification

- The approved Q2 rule, zero-based position % 8, is the rule the writer already used at `028b5c1`.
- The production gate changed at `86dac00`: Q1 and Q2 are pinned, and the M8 status is new.
- The owner required re-testing the exact assignment instead of relying on the earlier record.
- G10 shows that no output changed.

## Gates

| Gate | Check | Laptop `.venv` | `gpat-m5` (GPU host, CPU only) |
|---|---|---|---|
| G1 | The approved ruling is present (Q1 A, Q2 zero-based, Q3 clip→rint, Q4 separate clone, `smoke_test_record` null). Production authorization gives `E01_E02_BLOCKED_BY_M8_SMOKE_GATE` for both methods. The CLI exits 3 with that status for 2 methods × 3 seeds and writes nothing | PASS | PASS |
| G2 | Frozen population as metadata: 8838 TRAIN pairs `PTR000001…PTR008838`, 12,668 TRAIN samples, all M2 COMPLETE, 0 images read | PASS | PASS |
| G3 | 24-pair fixture, E01/E02 × 42/1337/2026: completed, 24/24 outputs, 0 failures, file and SHA256 verification, full replay | PASS | PASS |
| G4 | An independent second root reproduces every bank manifest byte for byte, with the same run id | PASS | PASS |
| G5 | E01 operator = position mod 8; A2-05 seed 795981663. E02 k = 40, n_eligible = 159, residual ≤ 1e-8 (max 1.0e-13); A2-04 seed | PASS | PASS |
| G6 | A rerun into an existing root is refused with outputs byte-unchanged. A 1-bit source-face corruption aborts the run (`failed` after 3 rows) | PASS | PASS |
| G7 | 0 opens under any runtime root or `faces_256`. 0 image opens outside the fixture and the pinned FAS-Aug assets. Manifests opened: pairs_train_v1, split_v1 (metadata), m2_sample_accounting. 0 TEST rows | PASS | PASS |
| G8 | **Q2, exact assignment** (details below) | PASS | PASS |
| G9 | **Q3, independent quantization** (details below) | PASS | PASS |
| G10 | Each of the 6 bank manifests is byte-identical to the pre-ruling record of the **same** environment | PASS | PASS |

## G8: E01 operator assignment (Q2)

**Full frozen population, 8838 pairs × seeds 42/1337/2026, from pure recipes with no image:**

- The operator is `operator_set[position % 8]` and `operator_index = position % 8` in every case.
- The first 8 pairs map to Color_Diversity … Low_Resolution in pinned order.
- 0 pairs agree with the pair_id-suffix rule, which is shifted by one for every pair.
- The sequence is identical for all three seeds, so there is no seed rotation.
- Counts per seed: 1105 for each of operators 0–5 and 1104 for each of operators 6–7.
- `operator_seed` equals the A2-05 value recomputed independently for every pair, with 0 mismatches, and the level-0 flag
  equals the first `randrange(10)` draw of that seed.

**Predicted level-0 no-ops**, unchanged copies of the live image that must be disclosed:

| Seed | Count | Share |
|---|---|---|
| 42 | 857 | 9.70 % |
| 1337 | 872 | 9.87 % |
| 2026 | 889 | 10.06 % |

These values are identical in both environments.

**Fixture banks:** the recorded `operator_index` equals position % 8. All 8 level-0 fixture outputs are pixel-identical to
their live input.

**Production-gate probes:** these used temporary ruling copies and a temporary smoke record in the work directory. The
repository ruling was not modified.

| Ruling variant | Gate result |
|---|---|
| Approved decisions plus a passing smoke record | AUTHORIZED (E01 ordinal `MANIFEST_POSITION_ZERO_BASED`, option A) |
| `PAIR_ID_NUMERIC` | refused, `E01_E02_BLOCKED_BY_MILESTONE_RULING` |
| Option B (E01 or E02) | refused, `E01_E02_BLOCKED_BY_MILESTONE_RULING` |
| Q3 `OTHER` | refused, `E01_E02_BLOCKED_BY_MILESTONE_RULING` |

## G9: E02 quantization (Q3)

All 72 E02 fixture outputs (24 pairs × 3 seeds) were recomputed independently in float64 from the recorded block
selection and the pair's own source face. The recomputation pipeline:

1. per-channel `fft2` and `fftshift`;
2. the source spoof magnitude replaces the target in the selected blocks and their pixel-wise conjugates; the target phase
   is kept;
3. `ifftshift` and `ifft2`;
4. real part, clip to [0,255], `numpy.rint`, uint8.

| Check | Result |
|---|---|
| Output pixel mismatches | 0 rows |
| Edit-map mismatches | 0 rows |
| `source_face_sha256` mismatches | 0 rows |
| Pixels where floor or truncation would give a different byte | 7,077,635 |
| Values clipped | 311 |

The floor/truncation count shows that the rounding rule determines the output bytes. That rule is the M6D1
implementation-defined choice that Q3 confirmed.

## Cross-environment (unchanged finding)

| Bank | Laptop manifest | gpat-m5 manifest |
|---|---|---|
| E01 s42 | `f4758044…` = ref | `f039af12…` = ref |
| E01 s1337 | `6f62c32a…` = ref | `6f62c32a…` = ref |
| E01 s2026 | `b71e717e…` = ref | `d2307136…` = ref |
| E02 s42 / s1337 / s2026 | `f3aa5426…` / `c577c75d…` / `e529fdd1…` = ref | `73987bf0…` / `00b5b935…` / `bb333606…` = ref |

- Each environment reproduces its own `028b5c1` manifests byte for byte.
- The laptop-versus-`gpat-m5` differences are the ones already recorded in `E01_E02_BANK_WRITER_CROSS_ENVIRONMENT.json`:
  - E01 Hand_Trembling: ≤ 1 LSB, from OpenCV `filter2D`.
  - E02: only the `max_abs_imag` diagnostic differs (~1e-13); the pixels are identical.
- E01 laptop and GPU-host bytes are **not** claimed identical. E02 pixel identity holds for these two stacks only.

## Tests at `86dac00`

| Suite | Laptop `.venv` | `gpat-m5` |
|---|---|---|
| `test_e01_e02_bank_writer` + `test_e01_e02_bank_owner_ruling` | 34/34 | 34/34, including `verify_locked_environment` |
| `test_m6c1_*` | 83/83 | 83/83 |
| `tools/m6b_validate_configs.py` | PASS | — |
| full `test_m6*` | 975 ran, only the 4 known `/media/cong/Data`-unmounted errors | — |

## gpat-m5 run isolation

- The run used a separate clone of `86dac00` from a git bundle (`~/workdir/e01e02_bank_qual_86dac00`).
- The FAS-Aug cache was symlinked read-only from the E04 checkout.
- Settings: `CUDA_VISIBLE_DEVICES` empty, `nice 19`, `ionice -c3`.
- The clone, the bundle, the work directory and the host copy of the record were removed afterwards.

E04 seed 2026 was untouched throughout:

- PID 435565 kept running (global_step 10875 → 11198).
- The checkout stayed at `18ba00e`, clean.
- There is no exit marker.
