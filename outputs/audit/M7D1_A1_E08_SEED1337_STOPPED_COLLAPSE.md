# M7D1-N7A — GPAT-B0 / E08 / seed1337 (run 62d7cfd98022a95a): OWNER_STOPPED_CONFIRMED_GENERATOR_COLLAPSE

Status: **OWNER_STOPPED_CONFIRMED_GENERATOR_COLLAPSE** (failed gates: none).

This run is **not** a completed 60-epoch run, **not** a numerical hard failure and **not** an AMP hard failure. It is an owner-authorized graceful stop after direct TRAIN-only confirmation of catastrophic generator identity collapse. No checkpoint of this run is selectable or bank-eligible.

## Identity
- launch authority `4466e1258c24e4a5d448dd5b30e8d8768dfe7174` (pinned launch clone `/home/student20261/workdir/GPAT_TransferBench_M7A1_4466e12`), curriculum `GPAT-TransferBench-v1.0+M7D1-A1`
- run root `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7_a1/E08/seed_1337`, launched 2026-10-17T15:02:30Z, SCIENTIFIC mode, config `0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a`
- environment: NVIDIA GeForce RTX 3090, torch 2.12.1+cu130, CUDA 13.0

## Stop
- last COMPLETE update **u15107** (epoch 14, group 742); last complete epoch **13**; 15107/66300 planned updates
- SIGTERM to the main process only at 2026-10-17T19:11:35Z; graceful exit after 24 s; shell EXIT_STATUS=0; manifest/summary completion_status = `interrupted`
- EXIT_STATUS=0 and the CLI word "finished" only mean the graceful-stop path returned normally; the runner recorded completion_status=interrupted. It is NOT scientific completion.

## Direct collapse confirmation (TRAIN only, 32 pairs)
- rule: sha256("M7D1-N2|fixed-train-diagnostic-subset|v1|"+pair_id) ascending, first 32 of 8838 TRAIN rows (M7D1-N2 rule); production GPATCore forward at 4466e12, CPU fp32, s_hf 0.10
- seed1337 (6 states: EMA 10–13, recovery live/EMA at u14365): u8 x_hat == u8 target in 32/32 for every state; median mean |x_hat − x_t| 6.07e-08; max 2.38e-07; A median 5.4e-20–1.2e-18; M median 7.2e-19–1.6e-17; M < 0.01 on 100% of pixels; HF residual saturated but masked out
- historical v1.0 seed42 (EMA 10–13): 32/32 exact target copies (same failure mode); A median 1.1e-11–4.7e-11
- healthy A1 seed42 (EMA 10–13): 0/32 exact copies; mean |x_hat − x_t| 0.00866–0.0112; A median 0.00946–0.0122; M median ≈ 1

## Training-time onset
- u2167 (epoch 2, stage 1, lambda_adv 0.0): G grad norm 0.0206 → 2.45 (pre-clip), amp_attempts 1 (no AMP event; nearest earlier retry u2132)
- historical-collapse signature (budget >= 0.0099 and bg < 1e-6 and lm < 1e-6 (M7D1-N2 historical-collapse training signature)): first at u2174; single non-signature update u2175 (lm 1.047e-06, just above the 1e-6 threshold; budget and bg in signature); then uninterrupted u2176..u15107 (12932 updates) through the owner stop
- budget pinned near the artifact-min hinge, id/lm/bg collapse rapidly; generator gradient later decays to ~7e-09 (epoch-14 median)
- **THIS FAILURE CANNOT BE ATTRIBUTED TO THE M7D1-A1 ADVERSARIAL TRANSITION, because it begins before adversarial loss is active (lambda_adv = 0 throughout stage 1, u < 5526).**
- root cause: NOT INFERRED (next milestone: deterministic fresh-from-u1 replay with pre-onset snapshots; the seed42 N3 adversarial mechanism is not assumed).

## Integrity
- COMPLETE u1..u15107 contiguous, 0 duplicates, 0 gaps; epoch records 1..13
- A1 schedule mismatches 0 over 15107 updates (fields: lambda_adv, scale_hf, lambda_con, lambda_spec, learning_rate, epoch, group)
- non-finite records 0; AMP: 9 recovered retries on 8 updates [1798, 2132, 7771, 9802, 12055, 12146, 14057, 14173], no terminal AMP failure
- VAL access 0, TEST access 0 (scientific run: 241660 TRAIN image opens; diagnostic: 64 TRAIN images)

## Recovery states (neither is a selectable EMA checkpoint)
- A. final mid-epoch after stop: u15107, epoch 14 / next group 743, sha256 `c1b4295e8c1a2c0838daa5ee833f8da003bb4b09382fa2cf26151dc2f75f6fe1`
- B. epoch-13-end copy `<runtime>/diagnostics/m7/M7D1_N7A_seed1337/recovery_latest_epoch13_end_u14365.pt`: u14365, sha256 `0f845ec5530b0d7ab6eca521eba374e9efd5698da28e1d58271c74ed48ade2b8`
- pre-onset state available: no — recovery/latest.pt is overwritten at every epoch end and the first EMA candidate is epoch 10, so no state before u2167 survives; the nearest exact pre-onset state is a fresh start (u0) at 4466e12 / seed1337 / SCIENTIFIC mode.

## Root integrity
- seed1337 root listing: pre-stop `84f5b8cd3fe37406a7e49dbaccb2d5744445753d8f7172efcef32de25394d2f8` (11 files) → post-stop `07fd4e0bfeef18f13ed8977d5cafd8fc7d40c22d002ab4069552c75779df04b7` (12 files), unchanged since
- close-out changes: added ['run_summary.json'], rewritten ['checkpoint_index.json', 'checkpoints/recovery/latest.pt', 'metrics.jsonl', 'run_manifest.json']; resolved_config, EMA candidates and logs unchanged
- a1_seed42 root `9b1a85eaeaf635450807d43a7e19e97d447d6a18f2585c5fc7720b8efadfadf3` (59 files) unchanged
- v1_0_seed42 root `ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a` (59 files) unchanged

No resume, replay, fix, VAL, TEST, selection, seed2026, B1/B2/B3, bank or downstream work was started.
