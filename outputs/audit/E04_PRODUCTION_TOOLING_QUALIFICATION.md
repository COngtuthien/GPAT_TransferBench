# E04 production tooling qualification

**Status:** `E04_PRODUCTION_TOOLING_QUALIFIED` at tooling commit `ce16e3a`. Nothing activated or launched.

- Focused suites: 165 tests OK (2 skips: activation-state checks), exit 0.
- Full suite in clean worktrees at `ce16e3a`: 2588 tests, the identical 74 pre-existing failing IDs as baseline `5214363`, both as committed and with simulated activations (no test breaks on activation).
- Host (auditor interpreter Python 3.11.16, clone at ce16e3a): checkpoints_15_ok, feed_pack_crosscheck, independent_clean, png_structure_canonical, s7_equivalence_bytes_and_pixels, s7_equivalence_rejects_non_production_mode.
- Host E04 preflight before activation: authority refused (`LEARNED_BANK_BLOCKED_BY_PRODUCTION_AUTHORITY`); TF 1.15.5 / Python 3.8.20 env-lock probe OK, 0 precision errors, 15/15 checkpoints OK.

## Mandatory cases

- valid production authority accepted: `ProductionAuthority.test_valid_authority_accepted`
- missing authority refused: `ProductionAuthority.test_missing_authority_storage_or_s7_refused; host preflight (authority refused)`
- wrong contract hash refused: `ProductionAuthority.test_tampered_authority_refused[contract hash]; test_wrong_or_altered_records_refused`
- wrong S7 record refused: `ProductionAuthority.test_tampered_authority_refused[s7 record sha/path]; test_wrong_or_altered_records_refused`
- wrong checkpoint refused: `ProductionAuthority.test_tampered_authority_refused[checkpoint sha/run id/identity/iteration/seed swap]`
- wrong method/seed refused: `ProductionAuthority.test_tampered_authority_refused[method, seeds, seed order]`
- wrong inference config refused: `ProductionAuthority.test_tampered_authority_refused[config, BN decision, conversion decision]; Auditor.test_training_and_checkpoint_integrity`
- wrong GPU environment refused: `Auditor.test_wrong_gpu_environment`
- wrong HDD UUID refused: `StorageExtension.test_prerequisite_failures_are_refused; test_tampered_extension_record_refused; Auditor.test_wrong_hdd_or_storage`
- symlink/path traversal refused: `StorageExtension.test_symlink_traversal_and_unexpected_contents_refused`
- unexpected storage contents refused: `StorageExtension.test_symlink_traversal_and_unexpected_contents_refused; test_prerequisite_failures_are_refused`
- modified or missing face hashes refused: `Auditor.test_face_lineage_allowlist_and_split_crosscheck`
- incorrect TRAIN allowlist refused: `Auditor.test_face_lineage_allowlist_and_split_crosscheck`
- split_v1/M2 cross-check failure refused: `Auditor.test_face_lineage_allowlist_and_split_crosscheck`
- VAL/TEST metadata access refused: `Auditor.test_val_test_access`
- missing or altered PNG rejected: `CollectorHelpers.test_altered_missing_and_extra_png_detected; Auditor.test_output_population`
- wrong output count rejected: `Auditor.test_output_population`
- missing or duplicate pair ID rejected: `Auditor.test_output_population`
- incorrect S7-to-production correspondence rejected: `Auditor.test_s7_correspondence; host qualification s7_equivalence_rejects_non_production_mode`
- nonzero optimizer activity rejected: `Auditor.test_training_and_checkpoint_integrity`
- checkpoint mutation rejected: `Auditor.test_training_and_checkpoint_integrity`
- partial or failed execution refused: `Auditor.test_partial_or_failed_execution; Auditor.test_01_empty_collection_never_passes`
- family closure with fewer than three PASS banks refused: `Auditor.test_family_closure_needs_three_passed_banks`
- historical authority and audit tampering rejected: `Auditor.test_historical_authority_and_audit_tampering; tests/test_learned_bank_production.ProductionRecords.test_fail_closed`
- positive: synthetic full bank passes all 43 gates: `Auditor.test_00_synthetic_bank_passes_every_gate`

## Defects found and fixed before activation

- tools/learned_bank.py production refused E04 unconditionally (storage resolved without method_id)
- independent_verification raised on an undecodable PNG instead of recording a problem (still fail-closed, now recorded)
- tests/test_e04_bank_adapter CLI test would have invoked a real E04 production on the GPU host after activation; it now invokes the CLI only where refusal is guaranteed
