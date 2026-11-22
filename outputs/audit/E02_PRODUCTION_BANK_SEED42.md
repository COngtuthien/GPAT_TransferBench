# E02 production bank, seed 42: independent audit

Status: `E02_SEED42_PRODUCTION_BANK_AUDIT_PASS` (55/55 gates)

- run_id `77dce92a0f1b6751`, run dir `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E02/seed_42`
- outputs 8838 / 8838, failures 0
- bank manifest SHA256 `0b0aa41d714316601f0d9e2120a7d107f5a58d414a9475ab9f0e83feac284d1b`
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
| D.edit_map_file_set_exact | PASS |
| D.edit_map_sha | PASS |
| E.pair_seed_namespace_sha256 | PASS |
| E.n_eligible_159_k_40 | PASS |
| E.pcg64_selection_and_blocks | PASS |
| E.edit_maps_conjugate_closed | PASS |
| E.independent_fft_clip_rint_pixels | PASS |
| E.imag_residual_le_1e-8 | PASS |
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
