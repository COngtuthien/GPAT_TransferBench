# E06b / E06c retrospective runtime-driver provenance

**Status:** `E06_HISTORICAL_DRIVER_ESTABLISHED`. Lock driver `595.84`; historical driver `580.178.04`.

This is an evidence record for the retrospective disclosure in `configs/amendments/e06_runtime_driver_retrospective_disclosure.yaml` (addendum `docs/spec/amendments/GPAT_TransferBench_v1_0_E06_Retrospective_Runtime_Driver_Provenance_Addendum.md`). It modifies no scientific source, lock, checkpoint or run output. Host data were collected read-only at 2026-10-30T02:32:36Z (boot `a74f71de-4761-4d83-81b5-3a3f56741765`).

## Checks

| check | result |
|---|---|
| `lock_records_595_84_unchanged` | pass |
| `switch_transaction_installs_580_removes_595` | pass |
| `no_driver_change_after_switch` | pass |
| `no_595_kernel_module_ever_installed_for_running_kernel` | pass |
| `observations_all_580` | pass |
| `every_e06_run_bound_to_580_boot` | pass |
| `recorded_runtime_fields_equal_lock` | pass |
| `live_e06_environment_equals_lock_packages` | pass |
| `live_host_580_only` | pass |
| `e06_qualification_was_under_595_84` | pass |

## Driver switch

apt transaction `2026-10-03  11:27:17` (`apt-get install -o DPkg::options::=--force-confnew -y nvidia-driver-580-open linux-modules-nvidia-580-open-generic-hwe-24.04`) ended 2026-10-03T04:27:35Z UTC: it installed `nvidia-driver-580-open 580.178.04` and `linux-modules-nvidia-580-open-7.0.0-38-generic`, and removed `nvidia-driver-595-open 595.84`, `libnvidia-compute-595 595.84` and the 595 kernel modules for 7.0.0-28/-31. No 595 kernel module was ever installed for kernel 7.0.0-38. The only later NVIDIA transaction is a same-version 580 module repackage.

## Kernel boots (wtmp, since 2026-09-01)

| boot start (UTC) | kernel |
|---|---|
| 2026-10-02T06:11:52Z | 7.0.0-34-generic |
| 2026-10-02T09:58:02Z | 7.0.0-34-generic |
| 2026-10-03T04:03:07Z | 7.0.0-34-generic |
| 2026-10-03T04:22:33Z | 7.0.0-38-generic |
| 2026-10-20T08:56:42Z | 7.0.0-38-generic |
| 2026-10-29T02:44:27Z | 7.0.0-38-generic |

## Same-host driver observations

| time (UTC) | boot | driver | source |
|---|---|---|---|
| 2026-10-03T07:27:21Z | 2026-10-03T04:22:33Z | 580.178.04 | outputs/audit/M7C3_GPAT_GPU_QUALIFICATION.json:environment/nvidia_smi (time: host file mtime) |
| 2026-10-25T04:39:34Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E05_SEED42_COMPLETION.json:collected/run_start/gpu_driver/live_driver_version |
| 2026-10-25T07:12:02Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E05_SEED1337_COMPLETION.json:collected/run_start/gpu_driver/live_driver_version |
| 2026-10-25T07:44:37Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E05_SEED2026_COMPLETION.json:collected/run_start/gpu_driver/live_driver_version |
| 2026-10-25T10:56:49Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E03_SEED42_COMPLETION.json:collected/run_start/gpu_driver/observed_driver_version |
| 2026-10-27T03:15:48Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E03_SEED1337_COMPLETION.json:collected/run_start/gpu_driver/observed_driver_version |
| 2026-10-28T11:50:05Z | 2026-10-20T08:56:42Z | 580.178.04 | outputs/audit/E03_SEED2026_COMPLETION.json:collected/run_start/gpu_driver/observed_driver_version |

## E06 runs

| method | seed | run_id | window (UTC) | boot (kernel) | same-boot observations | driver |
|---|---|---|---|---|---|---|
| E06c | 42 | `27dcaf23aac18d7a` | 2026-10-18T06:44:08Z → 2026-10-19T00:39:46Z | 2026-10-03T04:22:33Z (7.0.0-38-generic) | 1 | 580.178.04 |
| E06c | 1337 | `49a179a4dca77f41` | 2026-10-20T10:49:23Z → 2026-10-20T18:10:12Z | 2026-10-20T08:56:42Z (7.0.0-38-generic) | 6 | 580.178.04 |
| E06c | 2026 | `86b631403e3c4ee1` | 2026-10-21T04:55:25Z → 2026-10-21T13:25:12Z | 2026-10-20T08:56:42Z (7.0.0-38-generic) | 6 | 580.178.04 |
| E06b | 42 | `aa96129289d080ba` | 2026-10-21T17:18:10Z → 2026-10-21T20:58:11Z | 2026-10-20T08:56:42Z (7.0.0-38-generic) | 6 | 580.178.04 |
| E06b | 1337 | `9ffda3ea3f6cc767` | 2026-10-22T01:33:11Z → 2026-10-22T05:12:28Z | 2026-10-20T08:56:42Z (7.0.0-38-generic) | 6 | 580.178.04 |
| E06b | 2026 | `ec71778b28479de5` | 2026-10-24T08:01:12Z → 2026-10-24T11:40:54Z | 2026-10-20T08:56:42Z (7.0.0-38-generic) | 6 | 580.178.04 |

None of the six run roots records a driver string (run_manifest, run_summary, resolved_config, run_start).

## Limitations

- The kernel journal is not readable by the run account, so no NVRM module-load line is available; the driver is established from package history, kernel identity, library inventory and same-boot observations, not from a log line written during each E06 run.
- No observation was taken inside any E06 run window; the closest same-boot observations are cited per run.
- The E06 qualifications (M6D5e, M6FC/FD) ran under 595.84; no E06 requalification was performed under 580.178.04. Bitwise runtime equivalence between the two drivers is neither claimed nor established.

## Erratum to 34d1543

The results audit 34d1543 (PREVIOUS_WORK_RESULTS_AUDIT) stated "boot 2026-10-03 11:28 +07"; 11:28:24 is the first user-visible journal entry. wtmp records the kernel 7.0.0-38 boot at 11:22:33 +07, before the 580 install (11:27:17-11:27:35); no 595 module existed for that kernel, so the 580 module was loaded within that boot after the install. The conclusion (580.178.04) is unchanged.
