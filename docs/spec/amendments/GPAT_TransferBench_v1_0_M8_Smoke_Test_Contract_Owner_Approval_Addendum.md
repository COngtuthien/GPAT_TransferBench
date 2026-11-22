# GPAT-TransferBench v1.0: M8 smoke-test contract, owner approval

Date: 2026-11-22. Machine-readable record: `configs/amendments/m8_smoke_test_contract.yaml` (`OWNER_APPROVED`).

The owner approved `configs/amendments/m8_smoke_test_contract.FINAL_DRAFT.yaml` exactly as committed at `60b50ef`. The
FINAL_DRAFT, the original DRAFT (`025d9e8`) and the S1 partial record (`550d5cc`) are preserved unchanged. Only governance
fields differ from the FINAL_DRAFT: status, approved, approval date, approval pin, document, owner ruling, confirmed items
and authorizations.

**Approved options:** S1 A_GLOBAL_END_TO_END, S2 BINARY_LABEL_PER_DATASET, S3 CONTENT_GROUP_ID, S4 HASH_RANK,
S5 TRAIN_VAL_ONLY_TEST_PROXY, S6 FROZEN_PAIRS_TARGET_IN_TOY, S7 ALL_METHODS_WITH_BANK_WRITERS, S8 SMOKE_SCHEDULE_CPU,
S9 BITWISE_ALL_STAGES, S10 ALL_THREE_BANK_SEED42_EVALUATOR.

**Confirmed pins:**

- the rank key bytes and the frozen global pair positions;
- live-unit eligibility and the skip/shortfall rules;
- disjoint VAL select and holdout units, with no TEST access;
- the SiW-Mv2 content-group disclosure (a content group is not a subject);
- 186 toy pairs, subject to re-derivation;
- E01/E02 toy banks at seeds 42/1337/2026;
- E00/E01/E02 at detector seed 42;
- 2 epochs × 3 steps with 1 warmup epoch, frozen full batches, CPU without AMP;
- separate runtime roots, the SMOKE_NON_SCIENTIFIC label, and the bootstrap CI out of scope.

**Authorized:**

- implementation of the smoke pipeline;
- the `gpat-m8-downstream` environment with the six hash-pinned wheels;
- synthetic qualification;
- one smoke execution and one independent replay;
- committing genuine evidence.

**Not authorized:** production banks, full scientific downstream training, TEST access, GPAT, DINOv3 execution, binding
of `smoke_test_record`, or marking O3 completed. The smoke schedule never replaces the frozen 40-epoch schedule. S7 does not
authorize learned-method production banks; each method needs its own bank-stage acceptance.
