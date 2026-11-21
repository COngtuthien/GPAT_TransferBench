# E01/E02 bank-writer qualification (synthetic fixture)

Date: 2026-11-22. Source commit `241321f760bfcc9e3bbdfb4088b73ba8fc9bc727` (clean tree in both runs).

**Status: `E01_E02_BANK_WRITER_QUALIFIED_SYNTHETIC` in the laptop `.venv` and in the locked `gpat-m5`.**

Scope: synthetic fixture only, with no benchmark image decoded and no production bank written. Production stays
`E01_E02_BLOCKED_BY_MILESTONE_RULING` (see `E01_E02_BANK_CONTRACT_AND_MILESTONE_ASSESSMENT.md`).

## What was qualified

`methods/common/bank_writer.py` and `tools/e01_e02_bank.py` (commit `01de91a`) were qualified with
`tools/e01_e02_bank_qualification.py` (commit `241321f`).

| Gate | Check | Laptop .venv | gpat-m5 (GPU host, CPU only) |
|---|---|---|---|
| G1 | No approved ruling; CLI exits 3 `E01_E02_BLOCKED_BY_MILESTONE_RULING` for 2 methods × 3 seeds; runtime root untouched | PASS | PASS |
| G2 | Frozen population, metadata only, against an empty faces root: 8838 TRAIN pairs `PTR000001…PTR008838`; 12,668 TRAIN samples, all M2 COMPLETE; 0 images read | PASS | PASS |
| G3 | 24-pair fixture, E01 and E02 × seeds 42/1337/2026: completed, 24/24 outputs, 0 failures, file verification, full replay | PASS | PASS |
| G4 | An independent second root reproduces every bank manifest byte for byte, with the same run id | PASS | PASS |
| G5 | E01 operator = position mod 8, each operator 3× per seed, A2-05 seed 795981663. E02 k = 40, n_eligible = 159, residual ≤ 1e-8 (max 1.0e-13), A2-04 seed | PASS | PASS |
| G6 | Rerun into an existing root is refused with outputs byte-unchanged; a 1-bit corruption of a source face aborts the run (`failed`, 3 rows before abort, no manifest) | PASS | PASS |
| G7 | Audit hook: 0 opens under any runtime root or `faces_256`; 0 image opens outside the fixture and the pinned FAS-Aug assets; manifests opened = pairs_train_v1, split_v1 (metadata), m2_sample_accounting | PASS | PASS |

Tests at `241321f`:

| Suite | Laptop | gpat-m5 |
|---|---|---|
| `tests/test_e01_e02_bank_writer.py` | 27/27 | 27/27, including `verify_locked_environment` PASS |
| `test_m6c1_e01` / `test_m6c1_e02` / `test_m6c1_runtime` | 26 / 29 / 28 OK | 26 / 29 / 28 OK |
| `tools/m6b_validate_configs.py` | PASS | — |

The full laptop `test_m6*` set ran 975 tests with only the 4 known errors caused by `/media/cong/Data` being unmounted.
`test_m0_bootstrap` has 3 failures, identical at `1630d6a` without these changes.

## gpat-m5 run isolation

- A separate clone of `241321f` was made from a git bundle at `/home/student20261/workdir/e01e02_bank_qual_241321f`.
- The FAS-Aug cache was symlinked read-only from the E04 checkout; the backend runs only `git rev-parse` and `ls-tree`.
- Settings: `CUDA_VISIBLE_DEVICES` empty, `nice 19`, `ionice -c3`.
- The clone, the bundle and the temp files were removed afterwards.
- The E04 checkout stayed at `18ba00e`, clean. E04 PID 435565 was untouched and advanced from step 426 to 784 during the work.

## Cross-environment finding (`E01_E02_BANK_WRITER_CROSS_ENVIRONMENT.json`)

| Bank | Pixel-identical rows, laptop vs gpat-m5 | Differences |
|---|---|---|
| E01 s42 | 22/24 | PTR000007 and PTR000023, both Hand_Trembling: max \|Δ\| 1, about 3k of 65,536 pixels |
| E01 s1337 | 24/24 | — |
| E01 s2026 | 23/24 | PTR000015, Hand_Trembling: max \|Δ\| 1 |
| E02 all seeds | 24/24 | only the `max_abs_imag` diagnostic (about 1e-13) |

The E01 recipes (operator, level, asset and sampled parameters) are identical. The ±1 LSB differences come from rounding in
OpenCV `filter2D`. **E01 output bytes are therefore bound to the execution environment and host.** Production banks and any
replay audit must run in the locked `gpat-m5` on the GPU host, which `tools/e01_e02_bank.py` enforces through
`verify_locked_environment`. E02 pixels are environment-stable across these two stacks.

## Writer behaviour (summary)

**Output layout.** Outputs go to `<runtime_root>/runs/m6/<E0x>/seed_<s>/bank/<pair_id>.png`, written as lossless PNG with
compress level 6, no metadata chunks and `O_EXCL` mode 0444. E02 also writes `edit_maps/<pair_id>.png`: the applied block
mask plus its conjugates, as single-channel 0/255.

**Bank manifest.** `bank_manifest.parquet` carries one row per pair in canonical order:

- position, pair_id, method, seed and config SHA256;
- target and source ids with their M2 face SHA256s, and `source_spoof_used`;
- the deterministic pair seed and the full recipe JSON;
- status and failure reason;
- the output path with its file and pixel SHA256s;
- edit-map path and SHA256;
- for E01: operator, level, asset and level-zero flag;
- for E02: n_eligible, k, indices and imaginary residual.

It is written with tmp + `os.link` (no overwrite), and `bank_manifest.sha256` sits beside it.

**Run files.** The `run_logging_v1` files come from the existing `RunContext`: `run_manifest`, `resolved_config`,
`metrics` / `generation_log` and `run_summary`. The run summary adds the following:

- input, output and failure counts, with the failure list;
- E01 operator counts and level-zero no-op count and fraction;
- E02 k, n_eligible and the maximum residual;
- duplicate output-pixel groups, which are reported rather than refused, because shared targets make level-zero copies coincide legitimately;
- authorization, input provenance, environment and lock, wall time, post-write integrity.

**Failure policy.**

- Input integrity violations abort the run, which is closed as `failed` with its partial evidence preserved.
- A method exception for one pair is recorded and counted, with no substitute or duplicate. The run is then `completed_with_failures`.

**TEST firewall.**

- Only `PTR\d{6}` pair ids are accepted, and every referenced sample must be TRAIN in split_v1 and M2 COMPLETE.
- The faces root must be `<runtime_root>/data/processed/faces_256`.
- A qualification run refuses any runtime root that holds `faces_256`.
- There is no TEST, VAL, limit, subset or resume option.

## Remaining before production

1. Owner ruling `configs/amendments/e01_e02_bank_milestone_ruling.yaml` (Q1 milestone, Q2 E01 ordinal, Q3 E02 rint, Q4 host).
   With option A, a passing deterministic smoke-test record is also needed.
2. Run in `gpat-m5` on the GPU host from a clean checkout or clone at the authority commit. Run `--check-only` first,
   which exercises gate 4 against the real faces.
