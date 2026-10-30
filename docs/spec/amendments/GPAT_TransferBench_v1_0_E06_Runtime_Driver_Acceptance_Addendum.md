# GPAT-TransferBench v1.0 — E06b / E06c retrospective runtime-driver acceptance addendum

| | |
|---|---|
| Status | `OWNER_APPROVED` |
| Ruling | `OWNER_APPROVED_RETROSPECTIVE_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY_E06` |
| Date | 2026-10-30 |
| Machine-readable record | `configs/amendments/e06_runtime_driver_acceptance.yaml` |
| Approved draft | `configs/amendments/e06_runtime_driver_acceptance.DRAFT.yaml` at `09b452415988f03bae074a134890c6438639ff02` |
| Disclosure | `configs/amendments/e06_runtime_driver_retrospective_disclosure.yaml` and its addendum (unchanged) |
| Evidence | `outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE.json` (`E06_HISTORICAL_DRIVER_ESTABLISHED`, unchanged) |

The owner approved this ruling on 2026-10-30. It decides the question left open by §5 of the retrospective
runtime-driver provenance addendum. It does not change the frozen specification, any other amendment, any config, the
environment lock, any scientific source, the training data, any checkpoint or any run output.

## 1. Ruling

The six completed E06b/E06c scientific runs below are accepted as valid comparison baselines, without rerunning or
discarding them. They are accepted as runtime-compatible with `environments/e06c.lock.json` (sha256 `91416a20…`)
**except for the GPU driver version**: the lock records **595.84**; the runs executed under **580.178.04**. The GPU
model and UUID (RTX 3090, `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`) are unchanged.

| Method | Seed | run_id | Training authority | Seed closure |
|---|---|---|---|---|
| E06c DSDG-BIN-IDFREE | 42 | `27dcaf23aac18d7a` | `84809868` | `cf5ed6c` |
| E06c DSDG-BIN-IDFREE | 1337 | `49a179a4dca77f41` | `84809868` | `9281c79` |
| E06c DSDG-BIN-IDFREE | 2026 | `86b631403e3c4ee1` | `84809868` | `81a8ef6` |
| E06b DSDG-NATIVE | 42 | `aa96129289d080ba` | `d9ecbef7` | `1f546c9` |
| E06b DSDG-NATIVE | 1337 | `9ffda3ea3f6cc767` | `d9ecbef7` | `a8a7b5d` |
| E06b DSDG-NATIVE | 2026 | `ec71778b28479de5` | `d9ecbef7` | `1729333` |

E06b seed-42 attempt 1 (`d3c0ddd5dc82a4a1`, FAILED_TECHNICAL, excluded) is not a scientific result and is not covered.

The draft terms committed at `09b4524` are approved as written. They are listed in `draft_terms_approved`.

## 2. Scope limits

This acceptance:

- is **retrospective**. It is not a pre-execution approval, and it does not reuse the E05 or E03 driver rulings;
- applies **only** to the six runs above;
- does **not** establish bitwise equivalence between NVIDIA drivers 595.84 and 580.178.04;
- does **not** claim that the runs used driver 595.84;
- does **not** retroactively change `environments/e06c.lock.json` or any other environment lock;
- does **not** modify any checkpoint, weight, log, run output or training data;
- does **not** authorize future E06 runs, including synthetic-bank generation, under arbitrary drivers. Future E06 work on this host must record the live driver;
- does **not** waive future runtime qualification requirements;
- does **not** authorize rerunning a seed because its performance is unfavorable.

## 3. Disclosure requirement

Every use of these six runs as comparison baselines carries this disclosure:

- the lock records 595.84;
- the runs executed under 580.178.04;
- they are accepted retrospectively, for the driver version only.

## 4. Remaining reproducibility limitations

- The E06 qualifications ran under 595.84: M6D5a–e on 2026-09-25 and M6FC/FD on 2026-09-29. No E06 requalification under 580.178.04 exists.
- The 580.178.04 driver was not sampled inside any E06 scientific process. It is established with HIGH confidence from four sources:
  - package history;
  - kernel identity;
  - the driver library inventory;
  - same-boot observations.
- The E06 runs cannot be reproduced bitwise under 595.84 on this host: the driver is uninstalled and has no module for the running kernel.
- Per-run package versions, torchvision and the Python binary hash were not recorded per run.
