# M8 smoke and downstream evaluator contracts: finalization for owner approval

Date: 2026-11-22. Branch `m6-baselines`. Starting HEAD = origin = `d77742055b4731b126e035a88ac0b41c0feb22d7`.

**Status: `M8_CONTRACT_READY_FOR_FINAL_APPROVAL`.**

- Toy feasibility is verified from TRAIN/VAL metadata.
- Every owner-selected option is recorded, and every execution detail is pinned.
- The items that need explicit confirmation are listed in §9.
- Nothing is approved or activated, and nothing was executed (no smoke stage, training, bank, TEST access or installation).

Files:

| File | Content |
|---|---|
| `configs/amendments/m8_smoke_test_contract.FINAL_DRAFT.yaml` | S1–S10, the smoke schedule and acceptance gates |
| `configs/amendments/m8_downstream_evaluator_contract.FINAL_DRAFT.yaml` | D1–D14, including the proposed D10 rule |
| `outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json` | `TOY_SUBSET_FEASIBLE`, written by `tools/m8_toy_plan.py` at `b4f4876`, clean tree |
| `environments/m8_downstream.planned.json` | environment plan (`PLANNED_NOT_CREATED`) |

The earlier DRAFTs (`025d9e8`, `d777420`) and the S1 record are unchanged.

## 1. Smoke decisions S1–S10

| ID | Option | Status |
|---|---|---|
| S1 | A_GLOBAL_END_TO_END | owner-approved (`550d5cc`) |
| S2 | BINARY_LABEL_PER_DATASET | owner-selected; pinned |
| S3 | CONTENT_GROUP_ID | owner-selected. **Disclosure: a SiW-Mv2 content group is not a subject identity** |
| S4 | HASH_RANK | owner-selected; key bytes and eligibility pinned (confirm) |
| S5 | TRAIN_VAL_ONLY_TEST_PROXY | owner-selected; select and holdout are disjoint by construction; TEST is filtered at read |
| S6 | FROZEN_PAIRS_TARGET_IN_TOY | owner-selected; canonical positions kept |
| S7 | ALL_METHODS_WITH_BANK_WRITERS | owner-selected; E01/E02 now, learned methods later; no GPAT |
| S8 | SMOKE_SCHEDULE_CPU | owner-selected; exact schedule in §4 (confirm) |
| S9 | BITWISE_ALL_STAGES | owner-selected; conditional on environment check V4, with stop-on-failure |
| S10 | ALL_THREE_BANK_SEED42_EVALUATOR | owner-selected |

## 2. Evaluator decisions D1–D14

The owner-selected options are LIVE_PASS, RESIZE_FULL_BICUBIC, SPOOF_IS_ONE, SPOOF_IF_SCORE_GE_THRESHOLD,
EPOCH_OWN_VAL_THRESHOLD, PER_STEP_FROM_ZERO, LINEAR_INTERPOLATION, SPOOF_POSITIVE, VIDEO_THRESHOLD,
RESNET18_ONLY_IN_SMOKE, SEED_MATCHED_RAW, CPU_NO_AMP_SMOKE_GPU_AMP_SCIENCE and CROP_FLIP_JITTER_NORMALIZE.

**D10 is the proposed RAW_TYPE_COVERAGE_V1 (§6), which still needs approval.** Every frozen value in spec §14–§16 and
`downstream_resnet18.yaml` takes precedence. The pins fill only the gaps:

- the sampler remainder rule and the RNG scheme;
- the exact resize call;
- the LR formula and parameter groups;
- the EER and TPR@FPR formulas;
- the CPU thread and determinism flags.

## 3. Toy-subset feasibility

These counts are exact, from `split_v1` TRAIN/VAL rows read through the TEST row filter and `pairs_train_v1`.

| Role | casia_fasd | msu_mfsd | siwmv2 (content groups) | Pooled |
|---|---|---|---|---|
| TRAIN live | 2 subjects, 48 frames, 6 videos | 2, 32, 4 | 2, 16, 2 | **96 frames, 12 videos** |
| TRAIN spoof | 2 subjects, 144 frames, 18 videos | 2, 96, 12 | 2, 16, 2 | **256 frames, 32 videos** |
| VAL select live / spoof | 6 / 18 videos | 4 / 12 | 2 / 2 | **12 / 32 videos** (96 / 256 frames) |
| VAL holdout live / spoof | 6 / 18 videos | 4 / 12 | 2 / 3 | **12 / 33 videos** (96 / 263 frames) |

Further results:

- **Units and pairs:** there were no shortfalls and no skipped units. There are **186 eligible frozen toy pairs** (CASIA
  141, MSU 27, SiW-Mv2 18). They cover all 8 E01 operators: 30/14/24/21/23/30/18/26.
- **Disjointness:** VAL select and holdout share no unit in any dataset. MSU works with its 5 VAL subjects because select
  happens to use only 3 of them. Had it used 4, the planner would report a shortfall and the contract treats that as a
  blocker; the synthetic test `test_06b` demonstrates that case.
- **Minibatches:** LIVE_PASS gives floor(96/32) = **3 full batches per epoch, with 0 tail frames dropped**. One epoch
  needs 48 real spoof (mixed) or 96 (E00 real-only) out of 256, and 48 synthetic out of 186 per bank.
- **Batch contract:** the frozen 32/16/16 and 32/32 batches can be satisfied exactly. **No replacement sampling and no
  smaller smoke batch is needed.**
- **Optimizer steps:** a LIVE_PASS epoch contains 3 valid optimizer steps.
- **VAL classes:** select and holdout each hold both classes in every dataset. Pooled, VAL threshold selection, AUC and
  dataset-wise AUC are therefore defined. Holdout AUC is defined both pooled and per dataset.
- **TEST:** no stage needs TEST. Banks read TRAIN faces (toy targets and their frozen TRAIN sources). The evaluator reads
  toy TRAIN, VAL select and VAL holdout. Metrics and plots read only prediction files and tables.

## 4. Smoke-only schedule (SMOKE_NON_SCIENTIFIC; S8, needs confirmation)

| Item | Value |
|---|---|
| Detectors | seed 42: E00 real-only, E01 mixed (bank seed 42), E02 mixed (bank seed 42) |
| Banks | E01 and E02 for seeds 42, 1337 and 2026 over 186 toy pairs (seeds 1337/2026 bank-stage only) |
| Schedule | 2 epochs × 3 steps = 6 optimizer steps per detector; warmup 1 epoch (3 steps, D6 formula); cosine over steps 3–5 to 1e-6 |
| Exceptions | epochs 2 (not 40), warmup 1 epoch (not 3), AMP off (CPU) |
| Unchanged | everything else: architecture, weights, loss, batches, AdamW, learning rates, weight decay, clip, augmentation, selection rules |
| VAL | VAL select after each epoch → checkpoint (D5) → threshold (§15); VAL holdout predicted once |
| Production | the frozen 40 / 3 epochs with AMP on CUDA are restored; the exception applies only inside the smoke root |
| Out of scope | the subject bootstrap CI (§18, M13) is N/A in smoke tables; DINOv3 (D11) |

## 5. Checkpoint and threshold validity

Per epoch, VAL select holds 12 live and 32 spoof videos (each dataset has both classes), so:

- the §15 sweep has both classes;
- the epoch ACER and the AUC tie-break are defined;
- D5 can select among the 2 epochs.

A third-level threshold tie raises `UnresolvedThresholdTie` (no silent pick). The holdout (12 live, 33 spoof videos) gives
defined holdout ACER, AUC and EER, and defined per-dataset ACER and AUC.

## 6. D10 RAW_TYPE_COVERAGE_V1 (proposed)

**Label-validity gate.** Every SiW-Mv2 spoof video in the evaluated prediction set must have exactly one non-null
`attack_raw`, constant over its frames, that is one of the 14 `attack_map_v1.yaml` siwmv2 tokens and maps to the video's
`attack_macro`. If any video fails, the whole raw table is N/A (`RAW_LABEL_COVERAGE_INVALID`); there is no partial table.

**Per type.** APCER is reported with its numerator and denominator. Zero videos gives N/A (`ABSENT_IN_SPLIT`). There is
no pooling, imputation, cross-split borrowing or minimum-support filtering, and all 14 types are always listed.

**Support counts (videos), TRAIN / VAL:**

| Raw type | TRAIN | VAL |
|---|---|---|
| Makeup_Cosmetic | 36 | 8 |
| Makeup_Impersonation | 43 | 9 |
| Makeup_Obfuscation | 16 | 3 |
| Mannequin | 28 | 6 |
| Mask_HalfMask | 50 | 11 |
| Mask_PaperMask | 12 | 3 |
| Mask_TransparentMask | 42 | 9 |
| Paper | 95 | 20 |
| Partial_Eye | 40 | 8 |
| Partial_FunnyeyeGlasses | 125 | 27 |
| Partial_Mouth | 21 | 4 |
| Partial_PaperGlasses | 53 | 12 |
| Replay | 68 | 15 |
| Silicone | 12 | 2 |

**Missing categories in TRAIN/VAL: none.** There are 0 null labels and 0 multi-label videos. TEST support is not
inspected; it is computed only at the final TEST evaluation. On the toy holdout, Replay has 2 videos and
Makeup_Impersonation 1; the other 12 types are N/A, so the absent-type path is exercised.

## 7. Environment plan (`environments/m8_downstream.planned.json`; installation not performed)

**Name:** `gpat-m8-downstream`, CPU only, on the GPU host.

**Base:** `gpat-m5`, unchanged (Python 3.11.16, glibc 2.39, torch 2.12.1+cu130, torchvision 0.27.1+cu130, numpy 2.4.6,
pillow 12.3.0, pyarrow 25.0.1, pandas 3.0.6, packaging 26.3, python-dateutil 2.9.0.post0).

**Added wheels** (all cp311 or py3, hashes from PyPI metadata, re-hashed after download):

| Package | Version | Note |
|---|---|---|
| matplotlib | 3.11.2 | |
| contourpy | 1.3.3 | 1.4.0 needs Python ≥ 3.12 |
| cycler | 0.12.1 | |
| fonttools | 4.68.0 | |
| kiwisolver | 1.5.1 | |
| pyparsing | 3.3.3 | |

Every reused `gpat-m5` package already satisfies matplotlib's constraints, so nothing pinned is upgraded.

**Recommended construction: A_VENV_OVERLAY.** It uses `gpat-m5`'s Python, `venv --system-site-packages`, and
`pip --no-index --no-deps --require-hashes`. It takes about 60 MB, and `gpat-m5` is untouched. The alternative, a conda
clone of about 5.4 GB, is not recommended: the root filesystem is at 96 % (42 GB free) while E04 writes checkpoints.

**Verification:**

| Check | What it verifies |
|---|---|
| V1 | `gpat-m5` pip-freeze hash is unchanged and its lock still verifies |
| V2 | imports and exact versions |
| V3 | ResNet-18 `f37072fd…` loads on CPU |
| V4 | two-process CPU training determinism, the S9 precondition |
| V5 | plot PNG determinism |
| V6 | lock capture to `environments/m8_downstream.lock.json` |

E04's TF1 environment is not used.

## 8. Code and tests (this task)

**Commit `b4f4876`:**

- `gpatbench/downstream/toy_subset.py` and `tools/m8_toy_plan.py`: the metadata-only planner.
- `metrics.py`: EER by linear interpolation (D7), spoof F1 (D8) and frame rates at the video threshold (D9).
- `firewall.py`: the TRAIN/VAL parquet row filter (allowlist lookup; no TEST metadata reaches Python).

New tests in `tests/test_m8_toy_and_metric_resolutions.py`:

- hand-derived EER cases: 2/3 interpolated, 1/2 exact, 0 separated;
- spoof F1 and frame rates;
- the firewall row filter;
- synthetic planner rules: disjointness, rank key, eligibility skip, shortfall reporting, refusals;
- a re-derivation of the committed feasibility record.

Not yet implemented, awaiting approval or the environment:

- TPR@FPR and the D10 table;
- the bank-writer SMOKE mode;
- the ResNet-18 runner;
- the plot renderer.

## 9. Owner approvals still required

1. **Final approval of both contracts.** This means committing `m8_smoke_test_contract.yaml` and
   `m8_downstream_evaluator_contract.yaml`, which explicitly confirms:
   - the S8 schedule (§4);
   - the S4/S5 pins;
   - CI out of smoke scope;
   - D10 RAW_TYPE_COVERAGE_V1;
   - the D6 warmup form `base·k/W` (first step at lr 0) versus `base·(k+1)/W`;
   - the D8 TPR@FPR rule;
   - the D1 remainder-discard rule and RNG scheme.
2. **Approval to install** the six pinned wheels into `gpat-m8-downstream` (A_VENV_OVERLAY).
3. Later, outside the smoke: the D13 science AMP dtype/GradScaler before M10; DINOv3 pins before M11; and binding of a
   passing smoke record.

## 10. Next executable milestone

After approvals 1–2, the next milestone is **M8-SMOKE-IMPL**:

1. Create and verify the environment (V1–V6). If V4 fails, stop under S9.
2. Implement and qualify on synthetic fixtures: the bank-writer SMOKE mode, the ResNet-18 smoke runner, TPR@FPR, the D10
   table and the renderer.
3. Run the global smoke and its replay on CPU in an isolated clone.
4. Validate the record with `validate_smoke_record`, re-derive it independently, and commit it.

E04 seed 2026 was observed read-only at 2026-11-22T10:06Z: global_step 31127, PID 435565, checkout `18ba00e` clean, no
exit marker.
