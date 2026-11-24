# E05 B1/B2 correction qualification

**Status:** `E05_B1_B2_QUALIFIED_FOR_S7_ATTEMPT_2`. Execution authority `7ff4d36f74cd64d6c405944d3b9fa4aee1fae521`, ruling `configs/amendments/e05_s7_b1_b2_ruling.yaml`.

- Tests:
  - laptop: 390 (`.venv`, 22 torch/GPU skips) + 76 (python3) OK;
  - GPU host, gpat-m6-e05: 121 OK.
- Real runtime: one fresh process per seed ran the genuine `load_real`.
  - B1 resolves `util.util` / `util/util.py`: file `bd534619…` = the patch manifest value; segment `9973b640…` = the pinned value.
  - O2 relocation passes; 7 imports resolve to the qualified closure; both CUDA extensions verified; no JIT build.
  - Exact checkpoint per seed: 42 `2b55dcf2…`, 1337 `f3a687d8…`, 2026 `ee10a3b4…`. Eval mode, training guards 0.
  - A second runtime load in the same process is still refused, so the B2 isolation is genuine. Another seed's checkpoint path is refused.
- No generation was run, no face was read, and nothing was written.
