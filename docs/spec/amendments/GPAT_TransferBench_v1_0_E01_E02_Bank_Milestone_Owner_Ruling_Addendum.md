# GPAT-TransferBench v1.0: E01/E02 bank milestone owner ruling (approved addendum)

Date: 2026-11-22. Branch `m6-baselines`. Machine-readable record:
`configs/amendments/e01_e02_bank_milestone_ruling.yaml` (`OWNER_APPROVED`).

The ruling approves the DRAFT `configs/amendments/e01_e02_bank_milestone_ruling.DRAFT.yaml`. That DRAFT was committed at
`01de91a` (SHA256 `e49df97f…3482`) and is preserved unchanged, together with its assessment
`outputs/audit/E01_E02_BANK_CONTRACT_AND_MILESTONE_ASSESSMENT.md`. For each question Q1–Q4, the owner chose the option the
DRAFT recommended. The ruling introduces no scientific choice that differs from the DRAFT.

**This ruling authorizes no bank.**

- No real E01 or E02 bank may be generated until the M8 deterministic smoke test has passed and a separate owner
  authorization binds it.
- No smoke-test waiver is granted.
- No training, downstream evaluation, TEST access, package installation or `method_status.csv` change is authorized.

## Q1: Milestone ownership. Decision `A_M8_SMOKE_GATED`

M8 owns the real production synthetic-bank generation for E01 and E02.

| Milestone | Owns |
|---|---|
| M6 | method implementation, frozen-contract verification, production writer qualification, readiness evidence |
| M8 | production synthetic-bank generation, final bank manifests, bank completion audits |

- E01 and E02 have no generator training. The correct M6 closure for both is **writer-qualified, not bank-completed**.
- The deterministic smoke-test gate of M6D6jR O3 (spec §26/§32) stays mandatory before any production bank.
- The frozen milestone specification (§25) and O3 are not altered.

Enforcement:

- `tools/e01_e02_bank.py` refuses production under option A until a committed, hash-verified smoke-test record with
  `smoke_test_passed: true` is bound in the ruling.
- The approved ruling carries `smoke_test_record: null` on purpose. Binding the record is M8 work and needs its own owner
  authorization.

## Q2: E01 round-robin ordinal. Decision `MANIFEST_POSITION_ZERO_BASED`

The canonical TRAIN pair relation `manifests/pairs_train_v1.parquet` is sorted by `pair_id` ascending. It holds
`PTR000001…PTR008838` (verified, 8838 rows).

```
operator_index = position % 8        # position is zero-based; PTR000001 has position 0
operator       = operator_set[operator_index]
```

The pinned operator order is identical to the official `RizhaoCai/FAS-Aug@0da1dd79` `FAS_Augmentations.augment_list`
entries #0–#7 and to `configs/methods/e01_fas_aug.yaml` `generation.operator_set`. `OfficialFASAugBackend.prepare()` refuses
any mismatch. The order is unambiguous and is not reordered.

| index | operator | pairs per seed |
|---|---|---|
| 0 | Color_Diversity | 1105 |
| 1 | Color_Distortion | 1105 |
| 2 | Reflection | 1105 |
| 3 | BN_Halftone | 1105 |
| 4 | Moire_Pattern | 1105 |
| 5 | SFC_Halftone | 1105 |
| 6 | Hand_Trembling | 1104 |
| 7 | Low_Resolution | 1104 |

- PTR000001 maps to Color_Diversity and PTR000002 maps to Color_Distortion. The assignment continues cyclically.
- The operator index is **not** derived from the numeric suffix of `pair_id`.
- The sequence is **not** rotated by experiment seed.

Preserved:

- the A2-05 per-pair seed `int.from_bytes(SHA256(UTF-8(pair_id + str(seed))), "big") mod 2^31`. PTR000001 with seed 42
  gives 795981663;
- all eight official operators and their magnitude ranges, and the NUM_MAG = 10 grid;
- level 0. A level-0 output is an unchanged copy of the live image, and these copies are disclosed.

Implementation status at the ruling:

- The qualified writer (`01de91a`) already passes `index = position` under this rule.
- The `028b5c1` qualification used it (gate G5: operator = position mod 8).
- The approved rule is therefore **not** a mapping change. The production gate is nevertheless narrowed to this rule. A
  fresh post-ruling synthetic qualification re-tests the exact assignment rather than relying on the earlier record.

## Q3: E02 uint8 quantization. Decision `CONFIRM_CLIP_THEN_RINT`

The existing M6C1/M6D1 rule is kept. The steps are:

1. Compute the frozen frequency substitution: the source spoof magnitude replaces the target live magnitude in the
   selected blocks and their pixel-wise conjugates.
2. Keep the target phase.
3. `ifftshift`, then `ifft2`.
4. Take the real part.
5. Clip to [0,255].
6. Apply `numpy.rint`: round to nearest, ties to even.
7. Convert to `uint8`.

The final rounding is an **implementation-defined choice qualified in M6D1, not an explicit instruction of spec §8.2**.

The following are unchanged:

- band 0.20 ≤ r ≤ 0.50 with at least 75 % block coverage;
- 16×16 blocks; n_eligible = 159, k = 40;
- source and target semantics; conjugate closure; the inverse transform;
- the 1e-8 imaginary-residual tolerance.

The E02 numerical regression fixtures and the cross-environment evidence are retained.

## Q4: Execution environment. Decision `SEPARATE_CLONE_CPU_CONCURRENT`

Future production banks run as follows:

- on CPU in the locked `gpat-m5` environment on the GPU host (`environments/m6_core_gpu.lock.json`, NumPy 2.4.6);
- from an isolated clean checkout or clone pinned to the approved production authority;
- never from the GPU checkout used by E04 scientific training.

E04's environment and runtime are not modified, and no package is installed or upgraded.

"Concurrent" means only that the isolated CPU-only run need not wait for an E04 run to end. It never touches the E04
checkout, environment, runtime root, tmux session or GPU (`CUDA_VISIBLE_DEVICES` empty). It does not lift the M8 gate.

Disclosures:

- **E01.** Cross-environment variation exists in Hand_Trembling (OpenCV `filter2D` rounding). 3 of 72 synthetic outputs
  differ by at most 1 LSB between the laptop `.venv` and `gpat-m5`. Laptop and GPU-host E01 outputs are **not** claimed
  byte-identical.
- **E02.** The observed laptop-versus-`gpat-m5` pixel identity is evidence for those two stacks only. It is not an
  unrestricted guarantee.

## Qualification implications

- Q2 and Q3 change no pixel path. Q1 changes the status the CLI reports while it refuses production: the refusal is now
  caused by the absent M8 smoke-test record, not by the absent milestone ruling.
- The `028b5c1` qualification records and the DRAFT stay byte-unchanged. They are not rewritten.
- A fresh post-ruling synthetic qualification is recorded separately.

## Next authorized milestone

**M8 deterministic smoke test.** E01/E02 production banks may be generated only after it passes and a separate owner
authorization binds its record.
