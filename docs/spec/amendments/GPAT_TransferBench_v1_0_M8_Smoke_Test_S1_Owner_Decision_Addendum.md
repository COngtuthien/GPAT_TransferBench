# GPAT-TransferBench v1.0: M8 smoke test, owner decision S1 (partial)

Date: 2026-11-22. Branch `m6-baselines`. Machine-readable record: `configs/amendments/m8_smoke_test_s1_owner_decision.yaml`
(`OWNER_APPROVED_PARTIAL_DECISION`).

**This record answers S1 only.** The M8 smoke-test contract stays a DRAFT:

- `configs/amendments/m8_smoke_test_contract.DRAFT.yaml` (`025d9e8`, SHA256 `1e3eef54…b2a7`) is preserved unchanged.
- S2–S10 remain pending.
- `configs/amendments/m8_smoke_test_contract.yaml`, the activation file, does not exist.

## Decision

**S1 = `A_GLOBAL_END_TO_END`.** The owner's label is `A_GLOBAL_END_TO_END_SMOKE_REQUIRED`; it is recorded under the DRAFT's
exact identifier. Option `B_STAGED_AMENDMENT` is rejected, and M6D6jR O3 is not amended.

The decision means:

- M8 requires one global deterministic end-to-end smoke test (spec §26, §32).
- The smoke test covers the generator or baseline transformation, the bank, the evaluator, the metrics and the plots.
- It must pass before any full M8+ production execution.
- The M6 E01/E02 writer qualification (`6f15f8a`, `fa239ad`) does not substitute for it.
- E01/E02 real production banks remain blocked. `tools/e01_e02_bank.py` keeps refusing with
  `E01_E02_BLOCKED_BY_M8_SMOKE_GATE`.
- GPAT remains deferred until the previous-work evaluation is completed.

## What this record does not authorize

- smoke-test execution or toy-subset selection;
- downstream training or any TEST access;
- production banks or package installation;
- GPAT work;
- a `method_status.csv` change;
- binding a smoke record.

Each of these requires the remaining decisions and the separate approvals listed in
`outputs/audit/M8_DOWNSTREAM_READINESS_ASSESSMENT.md`.
