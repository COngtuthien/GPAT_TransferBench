# M8 learned-bank S7: E05 PCGAN attempt 1, run A FAILED (owner review)

**Status:** `S7_FAILED_NEEDS_OWNER_REVIEW`. Run A exited 1 at 2026-11-24T18:49:39Z, 2 s after launch, before any generation. Run B was not started and there was no retry.

**Failed gate:** `existing_runtime` → `require(manifest["source"] == source, "source identity")` while loading the qualified PCGAN CUDA runtime (`RuntimeError: M6D4d: source identity`).

**Diagnosis (read-only probe):** the only differing key is `root`. The qualified runtime manifest `environments/e05.compatibility_patch_manifest.json` binds the upstream source root as an absolute path inside the main GPU checkout (`~/workdir/GPAT_TransferBench/...`). The S7 plan requires an isolated clone, whose identical source lives at a different path. Commit, tree, every file SHA-256, every git blob and the worktree status are identical. This is an execution-path binding, not a scientific, numerical or integrity failure. The CPU fixture qualification never reached this CUDA-only path.

**Effects:** 0 outputs (the out root holds only empty `run_A/E05/` directories), 0 face reads, 0 of 20615 face atimes changed, all three E05 checkpoints unchanged.

**Owner options:**

- **O1:** run E05 S7 from the main GPU checkout path after fast-forwarding it from `18ba00e` to the S7 execution commit. E04 is closed, but this deviates from the plan rule "never use the E04 checkout".
- **O2:** authorize a narrow gate amendment that accepts a byte-identical pinned source at another root, with tests, then rerun in new roots.
- **O3:** keep E05 blocked.
