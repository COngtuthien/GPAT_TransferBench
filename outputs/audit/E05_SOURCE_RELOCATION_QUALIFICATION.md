# E05 PCGAN: O2 source-relocation correction qualified; S7 blocked by two further defects

**Status:** `E05_O2_CORRECTION_QUALIFIED_S7_BLOCKED_BY_ADDITIONAL_DEFECTS`. No E05 S7 run was started.

## O2 correction (code authority `5f81d704b6da24165c4d9fe466afaf5da239046a`)

- The original failure was re-verified on the host, and its evidence is unchanged: only `source.root` differed, and the run failed before any model, checkpoint state, generation or face read.
- `build_models` now passes from an isolated clone on the GPU:
  - relocated source, `root` the only differing field;
  - 23 inventory files and 67 code files (sha256 + git blob) re-hashed;
  - no extra executables or symlinks; owner ruling bound;
  - all 7 upstream modules resolve to the qualified generated closure;
  - both real CUDA extensions verified; no JIT build.
- The original location `~/workdir/GPAT_TransferBench/...` is still accepted (read-only check).
- Tests: laptop 60 + 374 OK; GPU host (gpat-m6-e05) 105 OK.

## Additional blockers found by the load-only probes

1. **B1:** `bank_adapter.load_real` looks for `tensor2im` in `util/__init__.py` (a package) and raises StopIteration. The function is defined in `util/util.py`, and its generated segment equals the pinned one (`9973b640…`). Proposed fix: point the check at the defining module.
2. **B2:** a second `existing_runtime` call in the same process is refused ("both real CUDA extensions loaded"), because the extensions load at import and the modules stay cached. The `s7` CLI runs three seeds in one process. Proposed fixes:
   - **P1 (recommended):** make the runtime load process-idempotent, with full re-verification on each call;
   - **P2:** run one process per seed.

Neither fix is within the O2 authorization. Owner decision required.
