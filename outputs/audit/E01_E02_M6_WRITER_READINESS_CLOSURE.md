# E01 FAS-Aug / E02 FreqSub: M6 writer-readiness closure

Date: 2026-11-22. Branch `m6-baselines`. Machine-readable records:

- `outputs/audit/E01_M6_WRITER_READINESS_CLOSURE.json`
- `outputs/audit/E02_M6_WRITER_READINESS_CLOSURE.json`

| Method | Status |
|---|---|
| E01 FAS-Aug | **`E01_M6_WRITER_QUALIFIED_M8_BANK_PENDING`** |
| E02 Frequency Substitution | **`E02_M6_WRITER_QUALIFIED_M8_BANK_PENDING`** |

**These are not bank-generation completion records.** No real E01 or E02 bank exists. Every production-bank count and hash
in the JSON records is `null` and marked not generated.

## Authority chain

| Step | Commit |
|---|---|
| Contract and milestone assessment (DRAFT Q1–Q4) | `6c16fe8` |
| Writer and CLI | `01de91a` |
| Pre-ruling harness | `241321f` |
| Pre-ruling evidence | `028b5c1` |
| Owner ruling Q1–Q4 (`OWNER_APPROVED`) | `656cc8a` |
| Production-gate correction (approved Q1/Q2 pinned; M8 smoke status) | `86dac00` |
| Fresh post-ruling qualification, 10/10 in laptop `.venv` and `gpat-m5` | `6f15f8a` |

## Milestone state (both methods)

| Item | State |
|---|---|
| Scientific method implementation | COMPLETE: M6C1 implementation and M6D1 core-environment qualification; frozen configs unchanged |
| Production writer | QUALIFIED after the ruling (synthetic fixture; laptop `.venv` and locked `gpat-m5`) |
| Frozen TRAIN population | VERIFIED as metadata only: 8838 TRAIN pairs `PTR000001…PTR008838`; 12,668 TRAIN samples, M2 COMPLETE; 0 images read |
| Real production banks | **NOT GENERATED** |
| Production gate now | refuses with `E01_E02_BLOCKED_BY_M8_SMOKE_GATE` (CLI exit 3) |
| M8 deterministic smoke test | **PENDING**: M6D6jR O3 `smoke_test_completed: false`; hard gate |
| Downstream evaluation | PENDING, not authorized |
| `method_status.csv` | not updated; this requires a separately verified registry-authority decision |

## E01 FAS-Aug

- **Operator order.** The official `FAS-Aug@0da1dd79` augment list #0–#7 is pinned in the frozen config and checked by the
  backend: Color_Diversity, Color_Distortion, Reflection, BN_Halftone, Moire_Pattern, SFC_Halftone, Hand_Trembling,
  Low_Resolution.
- **Ordinal rule (Q2).**
  - The operator is `operator_set[position % 8]`, with zero-based position over `pairs_train_v1` sorted ascending.
  - PTR000001 maps to Color_Diversity. The rule never uses the pair_id suffix and never rotates by seed.
  - Counts per seed: operators 0–5 get 1105 pairs each; operators 6–7 get 1104 each.
- **Per-pair RNG (A2-05).**
  - The seed is `SHA256(UTF-8(pair_id + str(seed))) mod 2^31`, and it seeds `random.Random`. PTR000001 with seed 42 gives
    795981663.
  - The level is the first `randrange(10)` draw; the asset and operator draws follow.
  - G8 recomputed the seed independently for all 8838 × 3 pairs with 0 mismatches.
- **Level 0.** Level 0 is retained. The predicted unchanged copies of the live image, from recipes rather than bank counts,
  are:

  | Seed | Unchanged copies |
  |---|---|
  | 42 | 857 |
  | 1337 | 872 |
  | 2026 | 889 |

- **Disclosure.** Hand_Trembling varies by ≤ 1 LSB between environments. Production and any replay audit must run in
  `gpat-m5`.

## E02 Frequency Substitution

- **Frequency source.** The pair's own source spoof magnitude replaces the target live magnitude in the 40 selected blocks
  and their pixel-wise conjugates. There are 159 eligible 16×16 blocks (0.20 ≤ r ≤ 0.50, ≥ 75 % coverage). The target
  phase is kept. `source_face_sha256` is recorded and verified.
- **Quantization (Q3).** Real part, clip to [0,255], `numpy.rint`, uint8. This is the M6D1 implementation-defined rule, not
  an explicit instruction of spec §8.2.
- **G9 result.** An independent float64 recomputation reproduced all 72 fixture outputs and edit maps exactly. The maximum
  imaginary residual is 1.0e-13, against a tolerance of 1e-8.
- **Disclosure.** Pixels were identical between laptop and `gpat-m5`. This is evidence for those two stacks only.

## Remaining before any production bank (M8)

1. The M8 deterministic smoke test passes (spec §26/§32; M6D6jR O3).
2. A separate owner authorization binds the passing smoke-test record (`smoke_test_record` in the ruling) and names the
   production authority commit.
3. The production run follows Q4: CPU, locked `gpat-m5` on the GPU host, isolated clean clone at that authority, with
   `--check-only` first. `--check-only` performs the first real-face read.
4. The M8 bank completion audit: counts, manifests, hashes and replay.

**Next authorized milestone: the M8 deterministic smoke test.**

## E04 isolation

E04 seed 2026 was observed only with read-only `ps`, `tmux ls`, `git rev-parse`, `git status` and a one-line tail of
`metrics.jsonl`. The observations:

- run `e7100eb9a5ab0537`, PID 435565, tmux `e04_physics_std_seed2026`;
- global_step 10875 at 23:20Z and 11198 at 23:31Z on 2026-11-21;
- no exit marker; checkout `18ba00e`, clean.

No E04 process, checkout, environment, runtime file or GPU allocation was touched.
