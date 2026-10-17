# M7D1-N6B — M7D1-A1 GPAT-B0 / E08 / seed 42 scientific run: completion audit

**Status:** `M7D1_N6B_A1_SEED42_COMPLETED` — 54 gates, failed: none  
**Audit:** 2026-10-17T08:55:10Z (collected 2026-10-17T08:54:07Z); tool `tools/m7d1_n6b_a1_seed42_audit.py` (read-only, CPU-only on the GPU host)

## Authority / identity

| field | value |
|---|---|
| launch authority commit | `4466e1258c24e4a5d448dd5b30e8d8768dfe7174` |
| run_id | `92e91ddb78f1e242` |
| method / experiment / seed | `GPAT-B0 / E08 / 42` |
| curriculum_id | `GPAT-TransferBench-v1.0+M7D1-A1` |
| amendment | `M7D1-A1 (`outputs/audit/M7D1_A1_ADVERSARIAL_CURRICULUM_AMENDMENT.json`)` |
| frozen spec sha256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| B0 config sha256 | `0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a` |
| run root | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7_a1/E08/seed_42` |
| run window (first/last COMPLETE, UTC) | `2026-10-10T15:27:13Z → 2026-10-16T05:15:09Z` |
| runner exit | `tmux `m7_a1_e08_s42`: `{"status": "finished"}`, `EXIT_STATUS=0`; process gone; stdout/stderr.log empty` |

The evidence-only closure commit does not change the scientific launch authority (4466e12).

## Training completion

- completion_status `completed` (manifest + summary); final epoch 60, global update 66300.
- 66300 COMPLETE optimizer-group records, contiguous 1..66300; 0 missing, 0 duplicate, 0 non-COMPLETE; 60 epoch records (1105 groups / 2210 microbatches / 8838 rows each; group 1105 is the [4,2] tail group).
- Fresh start: first COMPLETE update u1 at 2026-10-10T15:27:13Z; no historical checkpoint resumed (recovery/EMA provenance = 4466e12 + A1 curriculum_id).

## M7D1-A1 schedule audit (all 66300 updates)

Compared by exact, type-checked float equality against `methods.gpat.runtime_contract.curriculum_a1 + main_lr (exact float ==, type-checked)` (runtime_contract sha256 `8645d7bb911b8a02cad5a2db656fac16bbcf469c3e772f7ada418833666c0017`): fields lambda_adv, scale_hf, lambda_con, lambda_spec, learning_rate, epoch, group. **Mismatches: 0.**

| u | logged lambda_adv | production | hex |
|---|---|---|---|
| 5525 | 0.0 | 0.0 | `0x0.0p+0` |
| 5526 | 0.0 | 0.0 | `0x0.0p+0` |
| 5527 | 4.528985507246377e-05 | 4.528985507246377e-05 | `0x1.7beb3922e017cp-15` |
| 6629 | 0.04995471014492754 | 0.04995471014492754 | `0x1.993a9ecb50e1ap-5` |
| 6630 | 0.05 | 0.05 | `0x1.999999999999ap-5` |
| 6631 | 0.05 | 0.05 | `0x1.999999999999ap-5` |
| 16575 | 0.05 | 0.05 | `0x1.999999999999ap-5` |
| 16576 | 0.1 | 0.1 | `0x1.999999999999ap-4` |
| 16577 | 0.1 | 0.1 | `0x1.999999999999ap-4` |
| 66300 | 0.1 | 0.1 | `0x1.999999999999ap-4` |

## AMP / numerical

- 49 amp_retry events on 46 updates; ownership by update: D_OPT 23, G_OPT 23; max 2 retries per update.
- Every retry atomic (0 optimizer steps, optimizer_update false, pre-group state restored, update index not advanced) and its update COMPLETE exactly once with amp_attempts = retries+1.
- Scales: min D 1024, min G 4096; final D 131072, G 65536 (= recovery scalers).
- 0 non-finite loss/grad-norm records; 0 terminal markers (FAIL_CLOSED*, NON_FINITE_LOSS, FAILED_NUMERICAL_POST_STEP, OOM) in metrics and logs; no post-step non-finite-parameter stop (would be terminal). tmux scrollback shows only a benign torch UserWarning and the clean finish.

| u | epoch/group | retries | offending | accepted D scale | accepted G scale |
|---|---|---|---|---|---|
| 1971 | 2/866 | 2 | D_OPT | 16384 | 65536 |
| 1985 | 2/880 | 1 | D_OPT | 8192 | 65536 |
| 2587 | 3/377 | 1 | D_OPT | 4096 | 131072 |
| 3002 | 3/792 | 1 | D_OPT | 2048 | 131072 |
| 5031 | 5/611 | 1 | D_OPT | 2048 | 262144 |
| 5234 | 5/814 | 1 | D_OPT | 1024 | 262144 |
| 5890 | 6/365 | 1 | G_OPT | 1024 | 131072 |
| 6893 | 7/263 | 1 | G_OPT | 1024 | 65536 |
| 9154 | 9/314 | 1 | G_OPT | 2048 | 65536 |
| 11396 | 11/346 | 1 | G_OPT | 8192 | 65536 |
| 14815 | 14/450 | 1 | G_OPT | 16384 | 65536 |
| 14896 | 14/531 | 1 | G_OPT | 16384 | 32768 |
| 15235 | 14/870 | 1 | D_OPT | 16384 | 32768 |
| 16612 | 16/37 | 1 | G_OPT | 16384 | 16384 |
| 17061 | 16/486 | 1 | D_OPT | 8192 | 16384 |
| 19236 | 18/451 | 1 | D_OPT | 8192 | 32768 |
| 19332 | 18/547 | 1 | G_OPT | 8192 | 16384 |
| 20736 | 19/846 | 2 | G_OPT | 8192 | 4096 |
| 23136 | 21/1036 | 1 | D_OPT | 8192 | 8192 |
| 26331 | 24/916 | 1 | D_OPT | 8192 | 16384 |
| 28455 | 26/830 | 1 | D_OPT | 8192 | 32768 |
| 30621 | 28/786 | 1 | D_OPT | 8192 | 65536 |
| 30737 | 28/902 | 1 | G_OPT | 8192 | 65536 |
| 31415 | 29/475 | 1 | G_OPT | 8192 | 32768 |
| 32745 | 30/700 | 1 | G_OPT | 16384 | 16384 |
| 32784 | 30/739 | 1 | D_OPT | 8192 | 16384 |
| 35124 | 32/869 | 1 | D_OPT | 8192 | 32768 |
| 37532 | 34/1067 | 1 | D_OPT | 8192 | 65536 |
| 37857 | 35/287 | 1 | G_OPT | 8192 | 32768 |
| 39732 | 36/1057 | 1 | D_OPT | 8192 | 32768 |
| 40099 | 37/319 | 1 | G_OPT | 8192 | 32768 |
| 40465 | 37/685 | 1 | G_OPT | 8192 | 16384 |
| 41766 | 38/881 | 1 | D_OPT | 8192 | 16384 |
| 42760 | 39/770 | 1 | G_OPT | 8192 | 16384 |
| 44077 | 40/982 | 1 | D_OPT | 8192 | 16384 |
| 46297 | 42/992 | 1 | G_OPT | 16384 | 16384 |
| 46328 | 42/1023 | 1 | D_OPT | 8192 | 16384 |
| 49386 | 45/766 | 1 | D_OPT | 8192 | 32768 |
| 50447 | 46/722 | 1 | G_OPT | 8192 | 32768 |
| 52705 | 48/770 | 1 | G_OPT | 16384 | 32768 |
| 53730 | 49/690 | 2 | D_OPT | 8192 | 32768 |
| 55266 | 51/16 | 1 | G_OPT | 8192 | 32768 |
| 56357 | 52/2 | 1 | D_OPT | 8192 | 32768 |
| 57329 | 52/974 | 1 | G_OPT | 8192 | 32768 |
| 60530 | 55/860 | 1 | G_OPT | 32768 | 32768 |
| 63498 | 58/513 | 1 | G_OPT | 65536 | 32768 |

## EMA candidates

- 51 candidates, epochs 10..60, no missing/duplicate; checkpoint_index.json agrees (sha256/size/epoch/global_step); all loaded with `weights_only=True`; all carry run provenance 4466e12 + GPAT-TransferBench-v1.0+M7D1-A1; selected=false everywhere.
- Manifest `outputs/audit/M7D1_A1_E08_SEED42_EMA_MANIFEST.txt` (epoch size sha256 path) — digest `51f9648ebefa0d41f85277ce646280d33c02911b9de04171ddd61b8e9ba79406`.
- **No best checkpoint selected; epoch 60 is not marked best.** Selection is N6C (VAL-only).

## Recovery checkpoint (not a selection candidate)

- `checkpoints/recovery/latest.pt` — 720644632 bytes, sha256 `25bdd59b9dfddec29d497a7742997ad0dcc03608a77639fca2e1f3693ce88623`
- position: epoch 61 / next_group 1 / global_update 66300 (= post-completion state: last completed epoch 60); curriculum_id `GPAT-TransferBench-v1.0+M7D1-A1`; labels RECOVERY_ONLY, NOT_ELIGIBLE_FOR_VAL_SELECTION, NOT_ELIGIBLE_FOR_BANK, NOT_ELIGIBLE_FOR_PAPER_RESULT.

## Firewall / scope

- TRAIN images 1060560 (= 2 × 8838 × 60), unique ids 12668 (sha256 `27a8dff74625704c99b7dcbbf38a5b70ee9eaede0fe7b0ea8d6206719d5ebb1d`); VAL images 0, VAL metadata 0, TEST images 0, TEST metadata 0, non-TRAIN 0.
- runs/m7_a1 contains only E08/seed_42. No checkpoint selection, bank, downstream FAS, seed1337/2026, B1/B2/B3.

## Historical v1.0 root

- `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7/E08/seed_42`: listing digest `ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a` (59 files) before and after the audit = the known value from N4/N5/N6B launch. Unchanged; nothing written.

## Run-root integrity (scientific root not mutated by the audit)

| item | sha256 |
|---|---|
| metrics_jsonl_sha256 | `fa05cfb6151ad932d52dc6c810c8d86711ee748e228ec376632fdf9b1f71717e` |
| run_manifest_sha256 | `4e0685bc43f9c18d355c67e3db07613051643c4c49fe8bb304f4c26e081be627` |
| run_summary_sha256 | `2ed833736b6a57e5e090bfdd1eac5f9a37a131efbaf8d121681ea5f8109d7e35` |
| checkpoint_index_sha256 | `0428e7e500996d3d2fb4b5f9ed42e5d73e3d225aad312737f720ae16b7e19343` |
| resolved_config_sha256 | `55b9ab973544fe17a24e56e569e16781faf38febb7042a9f9e2a586100a3ec03` |
| recovery_sha256 | `25bdd59b9dfddec29d497a7742997ad0dcc03608a77639fca2e1f3693ce88623` |
| ema_manifest_digest | `51f9648ebefa0d41f85277ce646280d33c02911b9de04171ddd61b8e9ba79406` |
| listing_digest | `9b1a85eaeaf635450807d43a7e19e97d447d6a18f2585c5fc7720b8efadfadf3` |
| tree_sha256 | `50d20a2881685e488cf6beeff38c0ebe880dfcbfc3c790bf82517c7a20705e89` |

File count 59; listing convention `find . -type f -printf "%P %s %T@\n" | sort | sha256sum`.

## Scientific health (observations, not gates)

| epoch | D_total | gadv | G grad | D grad | G clip | D clip | budget hinge | bg | lm |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.422 | 1.2 | 0.0235 | 2.07 | 0.00 | 0.68 | 0.781 | 0.000475 | 0.00067 |
| 5 | 0.000151 | 11 | 0.173 | 0.012 | 0.00 | 0.16 | 0.030 | 0.00276 | 0.00328 |
| 6 | 0.515 | 1.82 | 0.615 | 12.2 | 0.06 | 0.97 | 0.662 | 0.00316 | 0.0036 |
| 7 | 0.329 | 2.52 | 0.743 | 8.76 | 0.19 | 1.00 | 0.787 | 0.00317 | 0.00255 |
| 10 | 0.148 | 4.94 | 0.839 | 8.19 | 0.24 | 0.99 | 0.622 | 0.00351 | 0.00327 |
| 15 | 0.0572 | 8.1 | 1.05 | 6.04 | 0.60 | 0.91 | 0.553 | 0.0043 | 0.00358 |
| 16 | 0.0551 | 8.31 | 2.32 | 6.14 | 1.00 | 0.88 | 0.272 | 0.00538 | 0.00406 |
| 20 | 0.0269 | 10.3 | 2.3 | 3.96 | 1.00 | 0.79 | 0.151 | 0.00555 | 0.00426 |
| 25 | 0.00846 | 12.4 | 2.22 | 1.76 | 1.00 | 0.60 | 0.155 | 0.00588 | 0.00422 |
| 30 | 0.00263 | 14.4 | 2.53 | 0.661 | 1.00 | 0.43 | 0.056 | 0.00639 | 0.00424 |
| 35 | 0.000642 | 16.5 | 2.7 | 0.198 | 1.00 | 0.26 | 0.076 | 0.00635 | 0.00411 |
| 40 | 0.000172 | 17.4 | 2.76 | 0.0518 | 1.00 | 0.13 | 0.054 | 0.00672 | 0.00415 |
| 45 | 4.07e-05 | 18 | 2.58 | 0.0127 | 1.00 | 0.04 | 0.032 | 0.00706 | 0.00417 |
| 50 | 1.6e-05 | 18.5 | 2.25 | 0.00477 | 1.00 | 0.00 | 0.028 | 0.00717 | 0.00415 |
| 55 | 1.11e-05 | 19.1 | 1.88 | 0.00339 | 1.00 | 0.00 | 0.028 | 0.00728 | 0.0042 |
| 60 | 8.02e-06 | 19.5 | 1.74 | 0.00262 | 1.00 | 0.00 | 0.029 | 0.00726 | 0.00418 |

(medians per epoch; grad norms are the pre-clip total norms; clip = norm > 1.0; budget hinge = fraction of updates with budget > 0.)

- **M7D1-N6B-OBS-01 — discriminator saturation (limitation, not a gate).** per-epoch medians D_total / gadv: ep1 0.422/1.20; ep10 0.148/4.94; ep15 0.0572/8.10; ep20 0.0269/10.3; ep30 2.63e-3/14.4; ep40 1.72e-4/17.4; ep50 1.60e-5/18.5; ep60 8.02e-6/19.5. D grad norm median falls 8.19 (ep10) -> 2.62e-3 (ep60); D clip fraction 0 from ep55. D wins progressively in stage 3, as in N4/N5 qualification and the v1.0 run. Recorded for owner review before N6C; no model/loss/config change made.
- **M7D1-N6B-OBS-02 — generator gradient and clipping.** G grad norm (pre-clip) epoch medians 0.84 (ep10), 2.32 (ep16), 2.53 (ep30), 2.76 (ep40), 1.74 (ep60); run-wide minimum after epoch 5 is 0.278. G clip fraction (norm > 1.0) >= 0.99 in every epoch from 16 to 60. No vanishing G gradient: the v1.0 seed42 run had G grad median ~5e-11 from ~epoch 30. Clipping is ~always active in stage 3 (N5 limitation reproduced).
- **M7D1-N6B-OBS-03 — identity-collapse signature (v1.0 F01) absent.** signature budget >= 0.0099 and bg < 1e-6 and lm < 1e-6 per COMPLETE update: A1 run 0/66300 matches; the same scan of the historical v1.0 metrics.jsonl (read-only) gives 60520 matches from u5781 (= M7D1-N2 F01). A1 after epoch 5: bg min 9.99e-4, lm min 1.19e-3, budget max 2.80e-3 (< artifact_min hinge 0.01); budget hinge active fraction 0.272 (ep16) -> 0.029 (ep60). No closed-mask / x_hat == x_t collapse signature in the training metrics. Mask statistics are not logged in metrics.jsonl, and the direct EMA-candidate x_hat-vs-target check (M7D1-N2 style) was NOT run in N6B; the absence is inferred from the training metrics. An open-mask saturation, which the N4/N5 diagnostics reported (M_mean ~1), cannot be confirmed or excluded from these metrics.
- **M7D1-N6B-OBS-04 — AMP retries.** 49 amp_retry events on 46 updates (D_OPT 23 updates, G_OPT 23 updates; max 2 retries per update); every retry was atomic (0 optimizer steps, state restored) and its update later COMPLETE exactly once. Min D scale 1024, min G scale 4096; final D 131072, G 65536. Recovered retries under ATOMIC_AMP_BACKOFF_RETRY; not failures. No FAIL_CLOSED event.

**Comparison with the v1.0 failure.** Old run: G grad → ~5e-11, mask closed, x_hat = target from u5781 (signature budget ≥ 0.0099 ∧ bg < 1e-6 ∧ lm < 1e-6 matches 60520 updates). A1 run: 0 matching updates; G grad stays O(1) (clipped); bg/lm stay ≥ ~1e-3. No comparable collapse in the training metrics.

## Not done (stop boundary)

N6C (VAL-only selection), VAL, TEST, best-EMA selection, seed1337/2026, B1/B2/B3, bank and downstream FAS were not started. No model/loss/config/scientific-implementation change.
