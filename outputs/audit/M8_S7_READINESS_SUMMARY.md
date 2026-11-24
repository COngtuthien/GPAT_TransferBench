# M8 learned-bank S7 readiness summary

**Status:** `E04_FAMILY_CLOSED_S7_PARTIALLY_QUALIFIED`. S7-qualified: E06c, E03, E06b. Blocked: E05.

Authority `configs/amendments/learned_bank_s7_launch_authority.yaml` (`0be8f0e14578…`, commit `a5e3132`). E04 seed 2026 closed at `0e3f1d1`; E04 family closed at `eeff32d`. Execution clone at `a5e3132` on the RTX 3090 `GPU-b722cd9d…`, driver 580.178.04, one GPU job at a time.

| Method | Seed | Run A | Run B | Bitwise replay | Audit | Status |
|---|---|---|---|---|---|---|
| E06c | 42 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E06c | 1337 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E06c | 2026 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E05 | 42 | FAILED (seed 42 load, before generation) | NOT_STARTED | not run | failure record | BLOCKED |
| E05 | 1337 | NOT_REACHED | NOT_STARTED | not run | failure record | BLOCKED |
| E05 | 2026 | NOT_REACHED | NOT_STARTED | not run | failure record | BLOCKED |
| E03 | 42 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E03 | 1337 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E03 | 2026 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E06b | 42 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E06b | 1337 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |
| E06b | 2026 | PASS | PASS | identical | 98/98 | S7_QUALIFIED |

## E05

Run A failed while loading the qualified PCGAN CUDA runtime. The gate `existing_runtime` "source identity" failed because the qualified runtime manifest binds the upstream source root to the main GPU checkout path, while S7 runs from an isolated clone. Only `root` differs. Nothing was generated or read, and the checkpoints are unchanged. Owner options are in `LEARNED_BANK_S7_E05_ATTEMPT1_FAILURE.md`.

## Production readiness

- E06c, E03 and E06b are ready for a **separately authorized** production launch. Each needs its own owner `configs/amendments/learned_bank_production_<M>.yaml` binding its S7 record.
- E05 is blocked until the owner rules on the path binding and a new S7 attempt passes.
- Not done: production banks, downstream ResNet-18, TEST evaluation, DINOv3, GPAT.
