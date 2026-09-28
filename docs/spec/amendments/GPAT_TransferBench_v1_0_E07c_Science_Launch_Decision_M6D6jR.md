# GPAT-TransferBench v1.0 — E07c Science Launch Decision (M6D6jR)

**Record kind:** OWNER_DECISION / SCIENCE_LAUNCH_POLICY
**Classification:** DETERMINISTIC_IMPLEMENTATION_CLARIFICATION
**Status:** OWNER_APPROVED · ADDITIVE · NON-DESTRUCTIVE · PROSPECTIVE
**Authority commit:** `ade5900db1dbff73917a2eb2280f8f2fe7afc418` (M6D6j)
**Machine-readable record:** `configs/amendments/e07c_m6d6jr_science_launch_decision.yaml`

This is not an amendment; no A9 is created. It is not a scientific adaptation, not a new deviation, not a data, model
or metric change. E07c remains DiffFAS-BIN-IDFREE (controlled encoder reconstruction), CONTROLLED_ADAPTATION, DEV-021;
`new_deviation = false`, `new_fidelity_class = false`. It resolves the two pre-science blockers M6D6j recorded (O3, O4).

## 1. O3 — smoke-test timing: STAGED_MILESTONE_PRECEDENCE_CLARIFICATION

The frozen specification orders the milestones M6 Baselines → M7 GPAT → M8 Banks → M9 Generator eval → M10 ResNet
downstream → M11 DINOv3 downstream → … . It also states (§26) "A full deterministic smoke test on a 2-subject-per-class
toy subset must finish before full runs" and lists in the final go/no-go checklist (§32) "smoke test passes for
generator → bank → evaluator → metrics → plots". Completing that end-to-end smoke literally before M6/M7 generator
training would require implementing and executing the M8–M14 stages ahead of their own milestones.

Owner interpretation (execution order only):

- M6 and M7 generator scientific training MAY proceed once that generator's production runner, data, checkpoint and
  firewall contracts are qualified (for E07c: M6D6j).
- The full deterministic 2-subject-per-class smoke test is NOT deleted or waived. It is a mandatory HARD GATE before
  the first FULL M8+ end-to-end bank / evaluator / metrics / plots execution, and it must exercise
  generator → bank → evaluator → metrics → plots on the required deterministic toy subset.
- M6D6j is NOT retroactively labelled that smoke test. No scientific result may be fabricated from it.

`smoke_test_required = true`, `smoke_test_completed = false`, `blocks_E07c_M6_training = false`,
`blocks_full_M8_plus_execution_until_completed = true`.

## 2. O4 — resume before science: RESUME_NOT_REQUIRED_BEFORE_SCIENCE

MAIN_CHECKPOINT_RESUME remains UNQUALIFIED; M6D6k is NOT a prerequisite for E07c scientific training. Scientific E07c
runs are FRESH_ONLY; no resume flag or path may be used. If a scientific seed is interrupted or fails for a technical or
runtime reason: do not resume through an unqualified path; preserve the failed/interrupted run evidence; diagnose the
technical cause; after an authorized technical fix, rerun the SAME experiment seed from the beginning under the same
frozen scientific settings. Never substitute another seed; never rerun to obtain a better result. This follows the
frozen specification failure policy ("One seed fails — rerun same seed after technical fix; never replace with a
different seed").

`resume_required_before_science = false`, `resume_qualified = false`,
`recovery_policy = FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX`.

## 3. Launch status

FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION = false; O3 = RESOLVED_STAGED_MILESTONE_PRECEDENCE;
O4 = RESOLVED_NOT_REQUIRED. E07c main scientific training is AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED.
MAIN_PRODUCTION_RUNNER stays QUALIFIED; MAIN_CHECKPOINT_RESUME, MAIN_DIFFFAS_SCIENTIFIC_TRAINING and M8_BANK stay
UNQUALIFIED. Nothing is marked completed.

## 4. Scientific run contract (unchanged)

Seeds 42, 1337, 2026. Per seed: 400 epochs, batch size 4, drop_last false, 2210 iterations per epoch, 884000 optimizer
iterations; entrypoint `tools/run_e07c_main.py`; exact-source visualization every 1000 steps (M6D6j D1); periodic
checkpoints every 10000 steps and the terminal checkpoint at 884000 after the final visualization (M6D6j D2); M6D6iR
successor-gated pruning. No value is changed by this decision.
