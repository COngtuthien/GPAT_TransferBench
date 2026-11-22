# GPAT-TransferBench v1.0: M8 downstream evaluator contract, owner approval

Date: 2026-11-22. Machine-readable record: `configs/amendments/m8_downstream_evaluator_contract.yaml` (`OWNER_APPROVED`).

The owner approved `configs/amendments/m8_downstream_evaluator_contract.FINAL_DRAFT.yaml` exactly as committed at
`60b50ef`. The FINAL_DRAFT and the original DRAFT (`d777420`) are preserved unchanged. The governance fields differ:
status, approved, approval date, approval pin, document, owner ruling, confirmed items and authorizations. In addition, D10
is recorded as `RAW_TYPE_COVERAGE_V1`, and its "proposed" block is renamed with its content unchanged.

**Approved options:** D1 LIVE_PASS, D2 RESIZE_FULL_BICUBIC, D3 SPOOF_IS_ONE, D4 SPOOF_IF_SCORE_GE_THRESHOLD,
D5 EPOCH_OWN_VAL_THRESHOLD, D6 PER_STEP_FROM_ZERO, D7 LINEAR_INTERPOLATION, D8 SPOOF_POSITIVE, D9 VIDEO_THRESHOLD,
D10 RAW_TYPE_COVERAGE_V1, D11 RESNET18_ONLY_IN_SMOKE, D12 SEED_MATCHED_RAW, D13 CPU_NO_AMP_SMOKE_GPU_AMP_SCIENCE,
D14 CROP_FLIP_JITTER_NORMALIZE.

**Confirmed pins:**

- **D1:** one LIVE_PASS epoch; without-replacement permutations per pool; the remainder is discarded and the pool
  reshuffled at its boundary; domain-separated sampler RNG; a dedicated augmentation generator; seeded model
  initialization; no hidden global-RNG dependence.
- **D6:** warmup `lr_g(k) = base_g * k / W`, so the first optimizer step is at lr 0; cosine to 1e-6 afterwards; two
  parameter groups; AdamW defaults; clip 1.0.
- **D8:** `TPR@FPR(x)` is the maximum TPR over the section-15 candidates with BPCER ≤ x, without interpolation; an undefined
  value is N/A.
- **D10:** the label-validity gate (the whole table is N/A on any violation), all 14 categories always listed, numerator and
  denominator shown, absent categories N/A, no pooling, imputation or minimum-support filter, and no TEST metadata during
  the smoke.
- **D13:** the science-time CUDA AMP dtype and GradScaler policy remain separately gated before M10.

Frozen spec §14–§16 and `configs/frozen/downstream_resnet18.yaml` values take precedence. This approval is not
authorization for scientific training.
