# GPAT-TransferBench v1.0 — E06b / E06c retrospective runtime-driver provenance addendum

| | |
|---|---|
| Status | `OWNER_AUTHORIZED_RETROSPECTIVE_DISCLOSURE_RECORDED` |
| Date | 2026-10-30 |
| Machine-readable record | `configs/amendments/e06_runtime_driver_retrospective_disclosure.yaml` |
| Evidence | `outputs/audit/E06_RUNTIME_DRIVER_PROVENANCE.{json,md}`, built by `tools/e06_runtime_driver_provenance_record.py` |
| Runtime-compatibility acceptance | **not granted by this addendum**; `configs/amendments/e06_runtime_driver_acceptance.DRAFT.yaml` (`approved: false`) awaits an owner decision |

The owner authorized this addendum on 2026-10-30. It records a retrospective disclosure. It does not change the
frozen specification, any amendment, any config, the environment lock, any scientific source, the training data, any
checkpoint or any run output.

## 1. Discrepancy

The six completed E06 scientific runs ran under NVIDIA driver **580.178.04**. The environment of record,
`environments/e06c.lock.json` (sha256 `91416a20…`), records **595.84** (`identity.gpu_driver`). The GPU model and UUID
are the same: RTX 3090, `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`.

| Method | Seed | run_id | Training authority | Seed closure |
|---|---|---|---|---|
| E06c DSDG-BIN-IDFREE | 42 | `27dcaf23aac18d7a` | `84809868` | `cf5ed6c` |
| E06c DSDG-BIN-IDFREE | 1337 | `49a179a4dca77f41` | `84809868` | `9281c79` |
| E06c DSDG-BIN-IDFREE | 2026 | `86b631403e3c4ee1` | `84809868` | `81a8ef6` |
| E06b DSDG-NATIVE | 42 | `aa96129289d080ba` | `d9ecbef7` | `1f546c9` |
| E06b DSDG-NATIVE | 1337 | `9ffda3ea3f6cc767` | `d9ecbef7` | `a8a7b5d` |
| E06b DSDG-NATIVE | 2026 | `ec71778b28479de5` | `d9ecbef7` | `1729333` |

Family closures: E06c `24f7049`, E06b `3b0d308`. E06b seed-42 attempt 1 (`d3c0ddd5dc82a4a1`, FAILED_TECHNICAL, excluded) is not a
scientific result.

**Why the discrepancy was not caught at launch:**
- `tools/run_e06c.py` and `tools/run_e06b.py` check the lock SHA256 and the source identity.
- Neither one compares the live runtime identity, driver included, with the lock.
- No E06 run root and no E06 closure records the driver.
- The E05 and E03 runners do record it, and those methods hold `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`, granted before their launches. **That ruling is not reused here as a prior approval of E06.**

## 2. Evidence for the historical driver

All host data were collected read-only, and the collected document is embedded in the evidence record.

1. **Package history** (`/var/log/apt/history.log*`, every rotation).
   - On 2026-10-03, apt transaction 11:27:17–11:27:35 +07 installed `nvidia-driver-580-open 580.178.04` and `linux-modules-nvidia-580-open-7.0.0-38-generic`.
   - The same transaction removed `nvidia-driver-595-open 595.84`, `libnvidia-compute-595 595.84` and the 595 kernel modules for kernels 7.0.0-28 and 7.0.0-31.
   - No 595 kernel module was ever installed for kernel 7.0.0-38.
   - The only later NVIDIA transaction (2026-10-10) repackaged the same-version 580 module.
2. **Kernel boots** (wtmp). The two boots that span all six runs, 2026-10-03 11:22:33 +07 and 2026-10-20 15:56:42 +07, both ran kernel `7.0.0-38-generic`. 595.84 had no kernel module for that kernel.
3. **Same-boot observations of 580.178.04.**
   - **Boot of E06c seed 42:** M7C3 GPU qualification on 2026-10-03.
   - **Boot of the other five runs:** the run_start records of E05 seeds 42/1337/2026 and E03 seeds 42/1337/2026, from 2026-10-25 to 2026-10-28.
4. **Live host.** `libcuda.so.580.178.04` is the only CUDA driver library, `/proc/driver/nvidia/version` reports 580.178.04, and only `nvidia-580-open` module directories exist.
5. **Other recorded runtime fields.** Each run_manifest/run_start records Python, NumPy, CUDA, cuDNN, torch, executable, GPU model and the precision settings. These are compared with the lock in the evidence record. The live `gpat-m6-e06c` environment (Python sha256 and the 45 package versions) is re-checked against the lock as supporting evidence.

**Confidence: HIGH for every run.** The evidence combines package exclusivity, kernel identity and same-boot observation. It is not a log line written during each run. See the limitations in §4.

**Erratum to `34d1543`.** The results audit cited "boot 2026-10-03 11:28 +07". 11:28:24 is only the first user-visible journal entry. wtmp records the kernel boot at 11:22:33 +07, before the install. The 580 module was therefore loaded within that boot, after the install. The conclusion is unchanged.

## 3. Statements

- The difference is disclosed **retrospectively**. The runs are not described as executed under the locked driver.
- The environment lock is not edited, and no scientific source, training data, checkpoint or run output is modified.
- This disclosure does **not** establish bitwise runtime equivalence between 595.84 and 580.178.04.
- All three seeds of each method are preserved as completed. Nothing is retrained, rerun, dropped or selected.

## 4. Remaining limitations

- The E06 qualifications ran under 595.84: M6D5a–e on 2026-09-25 and M6FC/FD on 2026-09-29. No E06 requalification under 580.178.04 exists.
- No driver observation falls inside any E06 run window. The kernel journal is not readable by the run account.
- The E06 runs cannot be reproduced under 595.84 on this host: the driver is uninstalled and has no module for the running kernel.
- Per-run package versions, torchvision and the Python binary hash were not recorded. Their historical equality with the lock is supported by the lock-hash gate at launch and the live re-check, not proven per run.

## 5. Remaining owner decision

Whether to accept the completed E06 results as runtime-compatible with the lock except for the driver version (a
retrospective, driver-version-only acceptance) is a separate owner decision. The draft terms are in
`configs/amendments/e06_runtime_driver_acceptance.DRAFT.yaml`. Until the owner approves them, the E06 results carry this
disclosure as a runtime-provenance limitation.
