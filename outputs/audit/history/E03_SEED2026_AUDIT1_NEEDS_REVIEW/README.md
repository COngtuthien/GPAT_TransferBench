# E03 seed2026, first completion-audit build (historical diagnostic, NOT the accepted evidence)

This directory keeps the first build of the E03 STDN seed 2026 completion audit (run_id `237887eebab18637`) exactly as
it was produced on 2026-10-29. The files are byte-identical; they were moved here from `outputs/audit/` before the
re-audit and were never committed under their original paths.

- Status `E03_SEED2026_NEEDS_REVIEW`, **86/88** gates. Failed: `B.process_exited_zero` and `J.single_e03_session`.
- Cause: the run completed at 2026-10-28T18:52:21Z (exit 0). The host was then cleanly shut down at about
  2026-10-29T02:44Z and rebooted. The reboot removed the tmux server, so the live pane state that both gates read no
  longer existed. Every scientific gate passed.
- On 2026-10-29 the owner approved `OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION` for exactly these two
  gates. The accepted evidence is `outputs/audit/E03_SEED2026_*`, built from fresh read-only collections with the
  amended `tools/e03_seed_completion_audit.py`.
- `E03_SEED2026_EVIDENCE_SHA256.txt` lists the original `outputs/audit/E03_SEED2026_*` paths. The files themselves are
  in this directory.
- `E03_SEED2026_AUDIT1_collected{,_tf}.json` are the raw collector outputs. They are equal to the documents embedded in
  `E03_SEED2026_COMPLETION.json`.

| file | sha256 |
|---|---|
| `E03_SEED2026_COMPLETION.json` | `62bbc549c838f96be97cfc093fa0b6be66a3a52d0925cbb12858c1583b079ceb` |
| `E03_SEED2026_COMPLETION.md` | `f84abc0d17858e11f60d27d35d0afcf116d50807e5d60bd6d4e94578a35e84aa` |
| `E03_SEED2026_CHECKPOINT_MANIFEST.txt` | `74f4dea62ad74d7fc60f486799261efe58c993ee63a14bccb611240a7b9df7cb` |
| `E03_SEED2026_EVIDENCE_SHA256.txt` | `900401ce85f3c3c4837da3475a8cf89172de7e33ff2dea8bb885cbff777a6aa1` |
| `E03_SEED2026_AUDIT1_collected.json` | `66dfac988c9920987c1317b5b73ab18297855bb6dec30f6c121f8493f366d01b` |
| `E03_SEED2026_AUDIT1_collected_tf.json` | `d86a87fad0b241022bedff2155e1a4f32e69d428fe5bc39c97dddf8fc41aee2c` |
