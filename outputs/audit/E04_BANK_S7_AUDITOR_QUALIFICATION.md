# E04 S7 independent auditor qualification

**Status:** `E04_S7_AUDITOR_QUALIFIED`, 8/8 gates; auditor commit `9e15795cd1ab707ad5ec236cc90b048b37be3c50`, baseline `2caff0d166d2b347e31e96b9ecf96294aa4ae779`.

| gate | result |
|---|---|
| `Q.targeted_regression_modules_pass` | PASS |
| `Q.new_e04_audit_module_ran` | PASS |
| `Q.host_collector_interpreter_tests` | PASS |
| `Q.host_real_e04_environment_probe` | PASS |
| `Q.dry_collection_fail_closed` | PASS |
| `Q.real_feed_pack_allowlist_crosscheck` | PASS |
| `Q.existing_s7_records_rebuild_byte_for_byte` | PASS |
| `Q.full_suite_no_new_failures` | PASS |

## Targeted regression modules (repo .venv, one `discover` per module)

| module | exit | result |
|---|---|---|
| `test_e01_e02_bank_owner_ruling.py` | 0 | Ran 7 tests in 0.217s OK |
| `test_e01_e02_bank_writer.py` | 0 | Ran 27 tests in 18.919s OK |
| `test_e01_e02_m6_writer_closure.py` | 0 | Ran 6 tests in 0.052s OK |
| `test_e01_e02_production_authority.py` | 0 | Ran 5 tests in 0.137s OK |
| `test_e01_e02_production_banks.py` | 0 | Ran 3 tests in 0.933s OK |
| `test_e01_e02_production_gate.py` | 0 | Ran 18 tests in 0.607s OK |
| `test_e04_bank_adapter.py` | 0 | Ran 18 tests in 0.943s OK |
| `test_e04_bank_fixture_qualification_record.py` | 0 | Ran 2 tests in 0.004s OK |
| `test_e04_bank_readiness.py` | 0 | Ran 15 tests in 0.285s OK |
| `test_e04_bank_s7_audit.py` | 0 | Ran 39 tests in 3.617s OK |
| `test_e04_bank_s7_prep.py` | 0 | Ran 5 tests in 0.031s OK |
| `test_e04_seed1337_completion.py` | 0 | Ran 17 tests in 0.120s OK |
| `test_e04_seed2026_completion.py` | 0 | Ran 21 tests in 0.208s OK |
| `test_e04_seed42_completion.py` | 0 | Ran 12 tests in 0.111s OK |
| `test_e04_three_seed_training_closure.py` | 0 | Ran 9 tests in 5.265s OK |
| `test_e05_s7_b1_b2.py` | 0 | Ran 16 tests in 0.139s OK |
| `test_e05_s7_contrast_diagnostic.py` | 0 | Ran 3 tests in 0.259s OK |
| `test_learned_bank.py` | 0 | Ran 40 tests in 7.422s OK |
| `test_learned_bank_authority_assessment.py` | 0 | Ran 13 tests in 0.033s OK |
| `test_learned_bank_production.py` | 0 | Ran 12 tests in 59.934s OK |
| `test_learned_bank_s7_acceptance.py` | 0 | Ran 4 tests in 1.366s OK |
| `test_m8_bank_smoke_mode.py` | 0 | Ran 6 tests in 3.095s OK |
| `test_m8_contract_finalization.py` | 0 | Ran 6 tests in 0.302s OK |
| `test_m8_contract_owner_approval.py` | 0 | Ran 3 tests in 0.009s OK |
| `test_m8_downstream_infrastructure.py` | 0 | Ran 20 tests in 0.919s OK |
| `test_m8_downstream_readiness_package.py` | 0 | Ran 5 tests in 0.315s OK |
| `test_m8_smoke_decision_package.py` | 0 | Ran 4 tests in 0.211s OK (skipped=1) |
| `test_m8_smoke_pipeline_units.py` | 0 | Ran 7 tests in 11.060s OK |
| `test_m8_smoke_record.py` | 0 | Ran 6 tests in 0.292s OK |
| `test_m8_smoke_s1_owner_decision.py` | 0 | Ran 5 tests in 0.011s OK (skipped=1) |
| `test_m8_toy_and_metric_resolutions.py` | 0 | Ran 9 tests in 0.339s OK |

## Whole suite (clean worktrees, identical conditions)

- Baseline `2caff0d166d2b347e31e96b9ecf96294aa4ae779`: Ran 2517 tests in 191.262s FAILED (failures=18, errors=56, skipped=213) rc=1
- Candidate `5fbc6b389db073741603013cbcafec0967c4b7ed` (auditor branch; the later auditor commit only tightened one presence gate and is covered by the targeted modules): Ran 2556 tests in 198.805s FAILED (failures=18, errors=56, skipped=213) rc=1
- Failing tests: baseline 74, candidate 74; new failures: none; fixed: none.
- The pre-existing failures are environment- or history-dependent (see `fullsuite/`); their identities are listed in the JSON record and are unchanged by this work.

## Scope

qualifies the E04 branch of tools/learned_bank_s7_audit.py before the E04 S7 launch authority is activated; no GPU inference, no face read, no output was produced.
