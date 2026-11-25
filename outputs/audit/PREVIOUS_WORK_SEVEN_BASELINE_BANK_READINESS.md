# Previous-work synthetic banks: seven-baseline readiness

**Status:** `ALL_SEVEN_PREVIOUS_WORK_BANK_FAMILIES_COMPLETED_AND_AUDITED`. 185598 of 185598 images (synthetic PNG images only; E02 edit maps, manifests and provenance sidecars are not counted).

| Method | Name | Seeds | Banks verified | Family closure | Downstream readable |
|---|---|---|---|---|---|
| E01 | FAS-Aug | 3 | 3/3 | E01_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E02 | FreqSub | 3 | 3/3 | E02_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E03 | STDN | 3 | 3/3 | LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E04 | Physics-STD (controlled geometry/depth reconstruction) | 3 | 3/3 | LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E05 | PCGAN (controlled architecture resolution) | 3 | 3/3 | LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E06b | DSDG-NATIVE | 3 | 3/3 | LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |
| E06c | DSDG-BIN-IDFREE | 3 | 3/3 | LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED | yes |

| Method | Seed | Images | Manifest SHA256 | Audit gates | Host verification |
|---|---|---|---|---|---|
| E01 | 42 | 8838 | `abd93be76b265956…` | 52/52 | verified |
| E01 | 1337 | 8838 | `38a8c6c5caa7735d…` | 52/52 | verified |
| E01 | 2026 | 8838 | `ff7361426c6af222…` | 52/52 | verified |
| E02 | 42 | 8838 | `0b0aa41d71431660…` | 55/55 | verified |
| E02 | 1337 | 8838 | `cd34bc02e94ddcb7…` | 55/55 | verified |
| E02 | 2026 | 8838 | `9eb09bae542d37b7…` | 55/55 | verified |
| E03 | 42 | 8838 | `5cc8b9e26c4f5679…` | 34/34 | verified |
| E03 | 1337 | 8838 | `c414758d373de246…` | 34/34 | verified |
| E03 | 2026 | 8838 | `51fa6bafdf7ea1fd…` | 34/34 | verified |
| E04 | 42 | 8838 | `2ffbf95f968d0acd…` | 43/43 | verified |
| E04 | 1337 | 8838 | `83993dc73979b8bd…` | 43/43 | verified |
| E04 | 2026 | 8838 | `14569ec629b9dffe…` | 43/43 | verified |
| E05 | 42 | 8838 | `079077c86c12a562…` | 37/37 | verified |
| E05 | 1337 | 8838 | `f8e9ff6782058c74…` | 37/37 | verified |
| E05 | 2026 | 8838 | `e2407a9ce312e349…` | 37/37 | verified |
| E06b | 42 | 8838 | `882017b8fadc5326…` | 34/34 | verified |
| E06b | 1337 | 8838 | `40564cd58982eae4…` | 34/34 | verified |
| E06b | 2026 | 8838 | `e1dd3a2eaa990b05…` | 34/34 | verified |
| E06c | 42 | 8838 | `8da6d40458a2ee42…` | 34/34 | verified |
| E06c | 1337 | 8838 | `e88e1ca631bd5e63…` | 34/34 | verified |
| E06c | 2026 | 8838 | `d30ad186264074a6…` | 34/34 | verified |

## Checks

- all_21_banks_verified: pass
- all_7_family_closures: pass
- total_185598_images: pass
- verify_clone_clean: pass
- storage_live_ok: pass

## Known limitations

- E01: level-zero no-op operator outputs reproduce the target face (disclosed in the E01 closure); duplicate pixel groups exist by construction
- E02: edit maps preserved beside every output (8838 per bank)
- E04: CONTROLLED_ADAPTATION (controlled geometry/depth reconstruction; BN moving statistics; clip-truncate); not an official native Physics-STD reproduction
- E05: PCGAN controlled architecture resolution; low-contrast outputs, seed 1337 near-collapse disclosed and retained under RAW-MATCHED
- E06b: DSDG-NATIVE trained on CASIA+MSU TRAIN spoof coverage only (SiW-Mv2 not instantiable); synthetic ids without pair lineage
- E06c: DSDG-BIN-IDFREE adapted binary variant; synthetic ids without pair lineage
- banks exist only on the GPU host (E01/E02 runtime root, learned banks on the HDD); git holds manifests, hashes and audits; no independent binary backup

## Remaining downstream prerequisites (not executed)

- frozen downstream evaluator / bank-loader readiness review and owner authorization
- scientific ResNet-18 training and validation per method and seed
- authorized TEST evaluation
- mean +/- standard deviation across seeds; ACER, APCER, BPCER, EER, AUC; ROC/DET
- final previous-work comparisons
