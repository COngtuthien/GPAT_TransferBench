# M6D6j — E07c MAIN DiffFAS production-runner qualification (QUALIFICATION ONLY)

**Method:** E07c DiffFAS-BIN-IDFREE (controlled encoder reconstruction) · CONTROLLED_ADAPTATION · DEV-021 · method status
remains **IMPLEMENTED_NOT_EXECUTED**. **Authority:** `39508719a57e5afde0a8c71bf7db0f9f09ec25e0` (M6D6iR).
**Qualification seed:** 60608 (next unused E07c qualification seed; 60601–60607 used once each). `experiment_seed = null`
(QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN). No scientific run, no experiment seed, no scientific checkpoint, no M8 output.

## Owner decisions recorded in the production-runner contract

- **D1 = EXECUTE_EXACT_SOURCE.** FAS_train.py:117–163 runs at `iters % 1000 == 0` in source order: a second TRAIN iterator
  on the same loader (CPU RNG), `torch.randperm` (CPU RNG), the pinned DDPM `p_sample_loop` (250 reverse steps,
  cond_scale 2, 500 model forwards, 2 encoder calls) in TRAIN mode (CUDA RNG, dropout, BatchNorm running-buffer updates),
  PNG under `diagnostics/visualization/`. No switch, isolation, reduction, DDIM substitution or RNG restore exists.
- **D2 = AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION.** At 884000: train → report → periodic test false (884000 %
  10000 = 4000) → visualization (884000 % 1000 = 0) → terminal checkpoint → verify → prune model_880000.pt → completion.

## Implementation (additive; no existing file modified)

`configs/amendments/e07c_m6d6j_main_production_runner_contract.yaml`, `methods/difffas/main_runner_io.py` (no torch),
`methods/difffas/main_checkpoint.py`, `methods/difffas/main_runner.py`, `methods/difffas/main_runner_qualification.py`,
`tools/run_e07c_main.py`. `main_runner` reuses `main_graph` (M6D6i) and adds AST-verified statement replicas of the
report slot (:86–98), payload/path (:103–113) and visualization (:118–163); the iteration slot order
train → report → metrics → save → visualize → terminal is verified from the AST. Science and qualification share the
same dataset, reader (M6D6e CanonicalFaceReader), checkpoint store, retention, visualization, logging and loop code.

## Qualification results (one fresh GPU process, gpat-m6-e07c)

| Item | Result |
|---|---|
| A7 construction order | seed → precision → transform → dataset → DataLoader → objects → encoder_eval |
| Frozen encoder | through `main_runner_encoder` only; 1 SHA-verified event; 2 read-only opens; stat/SHA unchanged |
| A1 relation | 8838 rows (2520 / 1200 / 5118), 11 allowlisted columns, 3 self-guide rows, 5580 guides, 3830 content ids |
| TRAIN membership | 12668 relation ids all TRAIN / M2-COMPLETE; 14467 TRAIN rows materialized; 0 VAL / 0 TEST rows |
| Real B=4 step (global step 1) | loss 1.0016389, mse 0.9996135, vb 0.0020255; lr 4.0e-7; peak 13,690,067,456 B |
| Real B=2 tail step (global step 2) | loss 1.0104611; lr 4.0192e-7; peak 9,367,319,040 B; 256 nonzero-grad tensors |
| Visualization (direct production slot) | 500 model forwards, 2 encoder calls, CPU+CUDA RNG advanced, BN buffers changed; parameters, EMA, optimizer, scheduler, encoder unchanged; 73.9 s; peak 4,938,468,864 B |
| Checkpoint size / time | 2,475,667,461 B (full payload); write 5.7–6.2 s; SHA256 ≈ 1.2 s |
| Lifecycle (labelled steps) | 10000 → 20000 (10000 pruned) → terminal 884000 (20000 pruned) → terminal prune refused |
| Round-trip of first file | keys model/ema/scheduler/optimizer/conf; model, EMA, scheduler equal live; conf = DiffusionConfig |
| Physical face opens | B=4 12, tail 6, visualization 12 (3 per item: content, GT, guide); 30 unique TRAIN ids; 0 prefetch |
| Firewall | 0 denials; only the two frozen manifests, TRAIN faces, the frozen encoder path, qualification roots |
| nvidia-smi process peak | 15,206 MiB of 24,576 MiB |

The lifecycle writes the real full main state at LABELLED qualification steps (the executed global step is 2); because the
state and source payload (which carries no step) do not change between them, the three files are byte-identical — pruning
and verification are exact-path and index based. All qualification checkpoint bytes (7,427,002,383 B) were removed
under the qualification cleanup policy (not the M6D6iR policy); records and SHA256 remain in the qualification index.

**Accounting.** qualification_optimizer_steps = 2, qualification_backward_calls = 2, qualification_scheduler_steps = 2,
qualification checkpoint deserializations = 1 (frozen encoder) + 1 round-trip of its own file, torch.save = 3.
scientific_optimizer_steps = 0, scientific_main_runs = 0, experiment_seed_runs = 0, scientific_checkpoint_writes = 0.

## Disclosures

- **Attempt 1 aborted (retained).** The first process stopped at a harness gate: model forwards were counted with
  `nn.Module` forward hooks, which the pinned `forward_with_cond_scale` bypasses (`self.forward`, unet_autoenc.py:129/134),
  so 0 of the actual 500 forwards were seen. B=4 and B=2 had passed; no checkpoint was written. Its root is retained as
  `q60608-c463e33ac6940f3c.aborted-attempt-1`. Attempt 2 counts through a pass-through instance `forward` wrapper and
  passed; its visualization PNG is byte-for-byte the same size as attempt 1 (same seed and order).
- **stdout handling (non-scientific I/O).** The FAS_train.py:40 tqdm wrapper is constructed with `disable=True`; the report
  slot runs verbatim; the trajectory is kept by one metrics.jsonl record per optimizer step (884000 per scientific seed).
- **GPU historical test.** `test_m6d6e_e07c_aux_runner::test_24_cli_accepts_only_seed42` errors on the GPU host because
  the M6D6g run root exists; unchanged and unrelated. A new-test issue (`test_23` compared torch-import state absolutely)
  was corrected before the final GPU regression (371/372, only the known historical error).
- **Rough wall-clock planning (not measured steady state):** 884 visualizations × ~74 s ≈ 18 h per seed; 89 checkpoint
  transitions × ~10 s; per-step training time needs steady-state measurement (only two steps ran here).

## Statuses

Qualified: MAIN_PRODUCTION_RUNNER, A1_TRAIN_DATASET_INTEGRATION, CANONICAL_FACE_READER, GUIDE_MAPPING,
RUN_LOGGING_V1_MAIN_RUN, MAIN_CHECKPOINT_CADENCE, MAIN_TERMINAL_CHECKPOINT_CREATION,
MAIN_CHECKPOINT_RETENTION_RUNTIME_INTEGRATION (plus E07c_-prefixed aliases).
Not qualified: MAIN_CHECKPOINT_RESUME, MAIN_DIFFFAS_SCIENTIFIC_TRAINING, M8_BANK.
FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION = true (O3); O4 (M6D6k resume before science) pending. No 400-epoch seed may
start after M6D6j. Next: M6D6k (MAIN_CHECKPOINT_RESUME) and the O3/O4 owner decisions.
