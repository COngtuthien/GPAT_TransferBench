# E01 production bank, seed 1337: independent audit

Status: `E01_SEED1337_PRODUCTION_BANK_AUDIT_PASS` (52/52 gates)

- run_id `84f9df641837ea8f`, run dir `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E01/seed_1337`
- outputs 8838 / 8838, failures 0
- bank manifest SHA256 `38a8c6c5caa7735da475862636ce856dafb91660ba8b1cb43431e23e69017a4b`
- filesystem UUID `227ce2f6-373e-4f13-9cdb-84b911775706` at `/` (/dev/nvme0n1p3)
- launch commit `0ca57e3bcd77e1a0c54004c7fd4c21e5fe101822` (candidate `d995f556a8adcc5a032ba266a022b470117d1c1c`), gpat-m5, CPU-only
- replay: CLI in-process 8838/8838 identical; independent process 8838 rows, 0 mismatches
- face atime: 0 changed of 20615 (VAL 0, TEST by elimination 0); every read was O_NOATIME
- TEST firewall: CLI violations 0, audit violations 0, TEST rows parsed 0
- E04: checkout `18ba00e9` clean=True, process running, exit marker None

| gate | result |
|---|---|
| A.exit_zero | PASS |
| A.cli_completed | PASS |
| A.progress_complete | PASS |
| A.launch_head_is_launch_authority | PASS |
| A.launch_worktree_clean | PASS |
| A.process_exited | PASS |
| A.time_order | PASS |
| A.authority_candidate_ancestor | PASS |
| A.launch_gates_passed | PASS |
| B.run_manifest_commit_clean | PASS |
| B.config_identity | PASS |
| B.run_id_recomputed | PASS |
| B.summary_completed | PASS |
| B.production_authorization_bound | PASS |
| B.locked_environment | PASS |
| B.run_dir_listing_exact | PASS |
| C.frozen_manifests_sha | PASS |
| C.rows_8838 | PASS |
| C.population_equals_frozen_order | PASS |
| C.all_samples_train | PASS |
| C.no_duplicates | PASS |
| C.all_success | PASS |
| C.no_test_rows_parsed | PASS |
| D.manifest_sha_matches | PASS |
| D.bank_file_set_exact | PASS |
| D.output_sha_and_pixels | PASS |
| D.face_lineage_sha | PASS |
| D.read_only_files | PASS |
| D.metadata_identity | PASS |
| E.operator_is_position_mod_8 | PASS |
| E.operator_counts_1105_1104 | PASS |
| E.operator_seed_sha256_mod_2_31 | PASS |
| E.levels_in_grid_and_level0_retained | PASS |
| E.level_zero_noop_outputs_equal_target | PASS |
| F.cli_full_replay | PASS |
| F.cli_file_verification | PASS |
| F.independent_process_full_replay | PASS |
| G.cli_firewall_clean | PASS |
| G.cli_o_noatime_reads | PASS |
| G.audit_firewall_clean | PASS |
| G.audit_reads_noatime | PASS |
| H.snapshot_complete | PASS |
| H.val_atime_unchanged | PASS |
| H.test_atime_unchanged | PASS |
| H.train_atime_unchanged | PASS |
| H.no_mtime_change | PASS |
| I.output_filesystem_approved | PASS |
| J.e04_checkout_untouched | PASS |
| J.e04_running_or_exited_zero | PASS |
| J.e04_run_identity | PASS |
| J.cpu_only | PASS |
| K.audit_tool_identity | PASS |
