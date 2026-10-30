"""Write the classification matrices (C, E, F, G, H) of the previous-work results audit.

Classifications are audit judgements; every row names the source it rests on. Writes only into --out.
usage: python3 -I write_matrices.py --out <dir>
"""
import argparse
import csv
from pathlib import Path

M = 'metrics.jsonl'
LEARNED = ('E06c', 'E06b', 'E05', 'E03')

# (item, scope, status, class, source, reason)
FIG = [
    # ---------------------------------------------------------------- training figures
    ('total generator loss vs iteration', 'E06c', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses.total_loss by global_step (7400/seed)',
     'Caveat: epoch 1 uses 0.01 weights on aux terms (pinned train_generator.py:174-178), so steps 1-37 sit on a different scale; mark the epoch-1/2 boundary.'),
    ('total generator loss vs iteration', 'E06b', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses.total_loss by global_step (3200/seed)',
     'Same epoch-1 weighting caveat (boundary at step 16/17).'),
    ('total generator loss vs iteration', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses.L_G_total by global_step (4000/seed)',
     'Batch 1: per-iteration values are very noisy; binned means are postprocessing (curves/E05_seed*_curve.csv, 100-iteration bins).'),
    ('total generator loss vs iteration', 'E03', 'READY', 'READY_WITH_POSTPROCESSING', M + ' TRAIN_STEP train_losses.g_loss + a_loss',
     'The STDN G optimizer minimizes g_loss + a_loss (pinned model/model.py get_train_op(g_loss+a_loss)); both are logged per step, so the sum is exact. Index: train_step_total is 0-based.'),
    ('total discriminator loss vs iteration', 'E06c', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'pinned DSDG train_generator.py', 'DSDG generator training has no discriminator.'),
    ('total discriminator loss vs iteration', 'E06b', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'pinned DSDG train_generator.py', 'DSDG generator training has no discriminator.'),
    ('total discriminator loss vs iteration', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses.L_D_total', 'L_D_total = L_D_image + L_D_patch re-derived (<= 6e-8).'),
    ('total discriminator loss vs iteration', 'E03', 'READY', 'READY_WITH_POSTPROCESSING', M + ' TRAIN_STEP train_losses.d_loss + optimizer_ops_applied',
     'd_loss is fetched every TRAIN step but d_op is applied only on even steps (50000 of 100000); filter on optimizer_ops_applied for the applied-D curve.'),
    ('loss components vs iteration', 'E06c', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses.{loss_rec,loss_kl,loss_mmd,loss_ip,loss_ort}',
     'loss_cls and loss_pair are identically 0 by design (DEV-020: one spoof class, lambda_pair=0); plotting them as learned curves would mislead.'),
    ('loss components vs iteration', 'E06b', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses (7 terms)', 'All 7 terms active (loss_cls K=2, loss_pair lambda=5).'),
    ('loss components vs iteration', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' train_losses (5 G terms, 7 D terms)', 'Terms are defined by M6D4c; G total is their unit sum.'),
    ('loss components vs iteration', 'E03', 'PARTIAL', 'PARTIALLY_AVAILABLE', M + ' g_loss/d_loss/a_loss (TRAIN and VAL)',
     'The official source exposes only these three aggregates. esr/gan/reg/pixel sub-terms were never logged and cannot be recovered for historical steps. a_loss mixes two definitions per step (random `dec` branch, pixel weight 0.0 or 0.1), and the branch was not logged.'),
    ('learning rate vs iteration', 'E06c', 'READY', 'READY_FROM_EXISTING_DATA', M + ' learning_rate', 'Constant 2e-4 for all 7400 steps. A table entry is more informative than a plot.'),
    ('learning rate vs iteration', 'E06b', 'READY', 'READY_FROM_EXISTING_DATA', M + ' learning_rate', 'Constant 2e-4.'),
    ('learning rate vs iteration', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' learning_rates.{G,D}', 'Constant 1e-6 (G and D).'),
    ('learning rate vs iteration', 'E03', 'READY', 'READY_FROM_EXISTING_DATA', M + ' learning_rates.{G,D}', 'Staircase 6e-5*0.9^floor(gs/20000) on the shared global_step; read-back max rel err 2.5e-7.'),
    ('training runtime vs iteration', ','.join(LEARNED), 'READY', 'READY_FROM_EXISTING_DATA', M + ' step_seconds, wall_clock_seconds, utc',
     'MISLEADING for method-cost or seed comparisons without disclosure. The GPU was shared and contention was not recorded (E06c s42 ran 2.2-2.5x slower than s1337/s2026; E03 s1337 took 1.35x as long). step_seconds also differs in definition across methods.'),
    ('gradient diagnostics', 'E06c', 'PARTIAL', 'PARTIALLY_AVAILABLE', M + ' owned_gradients_finite', 'Only a finite-gradient count per step (55/55); no norms.'),
    ('gradient diagnostics', 'E06b', 'PARTIAL', 'PARTIALLY_AVAILABLE', M + ' owned_gradients_finite, netCls_grad_nonzero', 'Counts only; no norms.'),
    ('gradient diagnostics', 'E05', 'PARTIAL', 'PARTIALLY_AVAILABLE', M + ' gradients.{D,G}.{finite,non_none,nonzero,tensors}', 'Counts only; no norms.'),
    ('gradient diagnostics', 'E03', 'NOT_READY', 'NOT_RECORDED', 'methods/stdn/production_runner.py fetch list', 'No gradient quantity is fetched; cannot be recovered without retraining.'),
    ('checkpoint chronology', ','.join(LEARNED), 'READY', 'READY_FROM_EXISTING_DATA', M + ' *_checkpoint_event (utc, epoch/step) + checkpoint_index.json',
     'E06c/E06b 21 events, E05 2, E03 50; all payloads present and equal to the index.'),
    ('scientific budget verification', ','.join(LEARNED), 'READY', 'READY_FROM_EXISTING_DATA', 'run_analysis.json checks', 'Re-derived in this audit: full budget, contiguous index, counters equal budget.'),
    ('seed-to-seed variability', ','.join(LEARNED), 'READY', 'READY_WITH_POSTPROCESSING', 'curves/*_curve.csv; D3 table',
     'Within-method mean and sample std over 3 seeds of the final-window means. Never across methods.'),
    ('method-specific diagnostics', 'E06c,E06b', 'READY', 'READY_FROM_EXISTING_DATA', M + ' batch/epoch-order hashes, epsilon hashes; *_visualization_diagnostic + diagnostics/visualization PNGs (126 files)',
     'The visualization PNGs are TRAIN-sample grids, not validation.'),
    ('method-specific diagnostics', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' pair_id, dataset, sampler_index, PNG sha256 per iteration', '4000 distinct TRAIN pairs per seed (single partial pass over 8838).'),
    ('method-specific diagnostics', 'E03', 'READY', 'READY_FROM_EXISTING_DATA', M + ' VAL_STEP val_losses; flips; brightness; diagnostics/figures PNGs',
     'The VAL losses are diagnostics only (never used for selection). Label any VAL curve accordingly.'),
    ('peak GPU memory', ','.join(LEARNED), 'READY', 'READY_FROM_EXISTING_DATA', M + ' gpu_memory_bytes (max)',
     'The semantics differ: torch max_memory_allocated reset per step (E05/E06) vs TF MaxBytesInUse, cumulative allocator (E03). E03 run_summary.peak_vram_bytes is null by design; its value is the maximum over metrics.'),
    ('parameter counts', 'E05', 'READY', 'READY_FROM_EXISTING_DATA', M + ' run_start payload.runtime_closure.models / optimizers', 'Encoder 929960, Generator 3699233, ImageD 28859521, PatchD 24522145.'),
    ('parameter counts', 'E03,E06c,E06b', 'PARTIAL', 'READY_WITH_POSTPROCESSING', 'checkpoints (read-only inspection)',
     'Not in the run logs. Countable from the checkpoints (TF .index shapes; DSDG weights_only sidecar / raw storages) without training.'),
    ('cross-method loss magnitude comparison', 'all', 'NOT_APPLICABLE', 'NOT_APPLICABLE', '-',
     'The architectures and loss definitions differ, so a shared axis would mislead. Plot each method separately.'),
    # ---------------------------------------------------------------- spec figure catalog (section 20)
    ('fig24_training_curves.png (Train/VAL losses and ACER/AUC per epoch)', 'previous work', 'PARTIAL', 'PARTIALLY_AVAILABLE', 'spec s20; run logs',
     'Generator TRAIN-loss curves are ready. VAL losses exist only for E03 (diagnostic). ACER/AUC per epoch belong to downstream detector training, which has not run.'),
    ('fig14/14b/15/16 downstream ACER/AUC/EER bars', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'spec s14-16, s18-20', 'No bank, detector, prediction or metric exists.'),
    ('fig17_downstream_roc.png / fig18_det_curve.png', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'spec s22.2 prediction file',
     'Requires per-video TEST scores (video_score per sample) from the prediction file. An aggregate AUC/EER cannot produce ROC or DET curves, and no prediction file exists.'),
    ('fig19 attack-wise APCER / fig20 dataset-wise metrics', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'spec s16 subgroups, T13/T14', 'Pending downstream evaluation.'),
    ('fig21 binary vs native delta (E06c vs E06b)', 'E06c,E06b', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'T09', 'Pending downstream evaluation.'),
    ('fig25 compute trade-off (ACER vs generation cost)', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'T15', 'Needs downstream ACER and generation sec/image.'),
    ('fig04 pair examples grid; fig05-fig13 generator metrics; qualitative grid (s21)', 'all', 'NOT_READY', 'PENDING_BANK_GENERATION', 'spec s13, s20, s21',
     'No synthetic bank exists for any method.'),
    ('fig22/23/26 GPAT ablations', 'GPAT', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'spec s17.1/17.2', 'GPAT-only ablations; GPAT is owner-paused. None exist for the previous-work baselines.'),
    ('fig27 ranking parallel coordinates; fig28 failure counts', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'T04/T06/T15/T17', 'Pending bank generation and evaluation.'),
    # ---------------------------------------------------------------- tables
    ('Table A training configuration', 'E06c,E06b,E05,E03,E04,E01,E02', 'READY', 'READY_FROM_EXISTING_DATA', 'configs/methods/*.yaml (= frozen_config_snapshot), run_start payloads',
     'Draft in D1_table_A_training_configuration.csv.'),
    ('Table B training execution', ','.join(LEARNED), 'READY', 'READY_FROM_EXISTING_DATA', 'run_summary.json, metrics.jsonl',
     'GPU contention and CPU/RAM/IO are NOT_RECORDED. The E06c/E06b observed driver is not in the run artifacts.'),
    ('Table C generator training diagnostics', ','.join(LEARNED), 'READY', 'READY_WITH_POSTPROCESSING', 'metrics.jsonl',
     'Within-method mean ± sample std only. Final-window definitions differ by method and are stated per row.'),
    ('T03 reimplementation fidelity', 'all', 'READY', 'READY_FROM_EXISTING_DATA', 'method_status.csv, configs, amendments',
     'Fidelity fields are current; the scientific execution columns in method_status.csv are stale (see G).'),
    ('T05 generator distribution metrics', 'all', 'NOT_READY', 'PENDING_BANK_GENERATION', 'spec s13', 'No bank.'),
    ('T06/T07/T08 downstream (video/DINOv3/frame)', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'spec s14-16',
     'The DINOv3 frozen config is NOT CREATED (CONFIG_STATUS Q-06), and the evaluator is not implemented.'),
    ('T13/T14/T16 attack-wise, dataset-wise, statistics', 'all', 'NOT_READY', 'PENDING_DOWNSTREAM_EVALUATION', 'spec s16, s18', 'Pending.'),
    ('T15 compute', ','.join(LEARNED), 'PARTIAL', 'PARTIALLY_AVAILABLE', 'run_summary, metrics.jsonl',
     'Training time and peak VRAM are ready. Params: E05 ready; E03/E06 need postprocessing. Gen sec/image and evaluator time are pending.'),
    ('T17 failures', 'all', 'PARTIAL', 'PARTIALLY_AVAILABLE', 'E06B_SEED42_ATTEMPT1_CHECKPOINT_WRITER_FAILURE.*', 'Training failures are documented; generation failures are pending.'),
    ('cross-domain (source->target domain) performance tables', 'all', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'spec s3 (pooled held-out protocol); spec limitations (line 840 of spec_extracted_text.txt)',
     'The frozen protocol is pooled held-out, not cross-dataset DG. The defined analogue is the dataset-wise subgroup table T14 (pending).'),
]

# E04 pre-launch logging checklist: (requirement, status, evidence, note)
E04 = [
    ('per-iteration training losses', 'READY', 'production_runner.py:448-454 ctx.log_epoch once per iteration; fsync per record (runlog.py:314-315)',
     'Verified in D12 requalification records (I.* gates).'),
    ('individual E04 loss components', 'READY', 'production_io.py:76 STEP_LOSSES; losses.py objectives',
     'step1 loss_G_step1, L_depth, L_G, L_P, L_R, P_positive, P0_negative_prior; step2 loss_D, L_D; step3 loss_G_step3, L_S, L_H. These are pre-apply values fetched in the apply run itself.'),
    ('generator and discriminator learning rates', 'READY', 'production_runner.py:440-450 learning_rates {G, D}', 'Asserted against 5e-5*10^-floor(t/45000) (D = G/2) every iteration.'),
    ('optimizer update counters', 'READY_WITH_POSTPROCESSING', 'global_step = global_iteration after Step 3 (asserted t+1, phase 0) per record',
     'Per-iteration apply counts are not logged explicitly, but they are exact: each logged iteration ran apply[1..3], or the run raised. G applications = 2*global_step, D = global_step. The Adam slots in each checkpoint allow a cross-check.'),
    ('seed and RNG provenance', 'READY', 'resolved_config rng plan; run_start seeding; per record pair_rows, pair_ids, pair_passes, hard_scales, hard_keep, input_sha256', 'D4/D7 domain-separated streams; replay-verifiable.'),
    ('training timing', 'READY', 'step_seconds (3 steps, excl. feed), wall_clock_seconds, utc per record', 'Feed time = wall-clock delta - step_seconds (postprocessing).'),
    ('checkpoint chronology', 'READY', 'e04_checkpoint_event every 15000 (10 events, 30 files) with utc, state_sha256, round_trip_equal, all_finite; checkpoint_index.json', '-'),
    ('final checkpoint at iteration 150000', 'READY', 'production_io.checkpoint_kind -> selected at 150000; run_summary final_checkpoint_files', 'BASELINE_FINAL_STATE_V1.'),
    ('data and config identities', 'READY', 'production_io.identities(): config, spec, A4, M6D3d, ruling yaml/doc, locks, pairs_train_v1, split, prepass, feed pack, worker graph, production file hashes', '-'),
    ('numerical reproducibility disclosure (D12)', 'READY_WITH_POSTPROCESSING', 'run_manifest.git_commit (launch commit) contains configs/amendments/e04_d12_procedural_reproducibility.yaml (OWNER_APPROVED, c60481b)',
     'The D12 hash is not embedded in identities(). The seed completion audit must cite it explicitly. This needs no runner change.'),
    ('scientific completion evidence', 'READY', 'run_summary completion_status/global_iteration/face_reads; checkpoint events', 'The exit marker/atime firewall evidence is external, as in E03/E05.'),
    ('peak GPU memory', 'READY', 'per-record gpu_memory_bytes (TF MaxBytesInUse, cumulative)', 'run_summary.peak_vram_bytes is null with a stated reason, as in E03.'),
    ('GPU driver identity', 'READY', 'run_start payload.gpu_driver (D9 driver-version-only tolerance)', 'Unlike E06c/E06b, the driver is recorded.'),
    ('paper plotting / result extraction', 'READY', 'same JSONL schema family as E03/E05', 'Expected size ~2.4 KB/record, so ~0.36 GB metrics + ~10x24 MB checkpoints per seed. The host has 89 GB free (90% used).'),
    ('gradient diagnostics', 'NOT_RECORDED (not required)', 'production loop fetches no gradients', 'Adding fetches would change the executed subgraph and need requalification. The frozen contract does not require them.'),
    ('per-discriminator loss split', 'NOT_RECORDED (not required)', 'L_G/L_D are add_n over 4 discriminators (losses.py)', 'Aggregates are the defined Eq.23/L_D terms.'),
    ('shared-GPU contention', 'NOT_RECORDED (optional external)', 'none in runner',
     'Optional, non-invasive: an external nvidia-smi sampler during the run, with no runner change and no requalification. At audit time another user job was using the GPU (95% util, ~5.2 GB). Timing: the original qualification measured 0.944/0.942 s/iteration (~39 h/seed, E04_PRODUCTION_QUALIFICATION.md); the D12 requalification 3.10-3.20 s/iteration was measured under concurrent CPU pre-pass load and is disclosed as not representative (E04_PRODUCTION_QUALIFICATION_D12_NOTES.md). Live co-tenant load will slow the run.'),
]

# F. missing information and recovery plan: (field, scope, class, recovery, impact)
F = [
    ('observed GPU driver version during training', 'E06c x3, E06b x3', 'RECOVERABLE_BY_READ_ONLY_POSTPROCESSING (host evidence) + OWNER_DISCLOSURE_RULING',
     'Host dpkg.log: 580.178.04 installed 2026-10-03 11:27 +07, booted 11:28 +07, no later driver install. M7C3 (cff688d) observed 580.178.04 on 2026-10-03. The lock says 595.84. The DSDG runners never record the driver.',
     'Disclosure only. E05/E03 carry OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY; E06c/E06b have no equivalent record. No rerun is warranted.'),
    ('GPU contention / co-tenant load', 'all 12 runs (notably E06c s42, E03 s1337)', 'IRRECOVERABLE (measurement) / FUTURE_OBSERVATIONAL for E04',
     'Disclose qualitatively. For E04, optionally run an external nvidia-smi sampler.', 'Runtime cannot be presented as intrinsic method cost.'),
    ('STDN sub-loss terms (esr, gan, reg, pixel)', 'E03 x3', 'IRRECOVERABLE_WITHOUT_RERUNNING (not required)',
     'None. The official source exposes only g/d/a aggregates, and the logging contract forbids fabricating them.', 'The component plot for E03 is limited to g/d/a.'),
    ('STDN a_loss branch indicator (dec) per step', 'E03 x3', 'IRRECOVERABLE_WITHOUT_RERUNNING (not required)', 'Use epoch means; disclose the mixture.', 'Per-step a_loss is a mixture of two definitions.'),
    ('gradient norms', 'E06c, E06b, E05, E03', 'IRRECOVERABLE_WITHOUT_RERUNNING (not required)', 'None; only finite/nonzero counts exist (E05/E06).', 'No gradient-norm figure.'),
    ('parameter counts', 'E03, E06c, E06b', 'RECOVERABLE_BY_READ_ONLY_POSTPROCESSING', 'Count from the checkpoints (TF .index shapes; DSDG weights_only sidecar or raw storages).', 'Needed for T15.'),
    ('split manifest sha256 in run root', 'E06c (and any run without the field)', 'RECOVERABLE_FROM_EXISTING_ARTIFACTS',
     'pairs_train_v1 a5e4fdae... is recorded and is derived from split_v1 fb9aeb36...; the closures verified TRAIN membership.', 'None.'),
    ('source_commits in run_manifest', 'E05 x3', 'RECOVERABLE_FROM_EXISTING_ARTIFACTS', 'run_start payload.identities.source_commit 6baa180f + source_tree 25a434f1.', 'None.'),
    ('spec s22.3 run metadata (status enum, deviations list)', 'all 12', 'RECOVERABLE_BY_READ_ONLY_POSTPROCESSING',
     'Map completion_status completed -> COMPLETED; deviations from fidelity/amendment records.', 'None.'),
    ('peak VRAM in run_summary', 'E03 x3', 'RECOVERABLE_BY_READ_ONLY_POSTPROCESSING', 'Max of metrics gpu_memory_bytes (done in D2).', 'None.'),
    ('CPU/RAM/IO/energy measurements', 'all', 'IRRECOVERABLE (not required)', 'None; T15 needs only params/time/VRAM.', 'None.'),
    ('VAL losses', 'E06c, E06b, E05', 'NOT_APPLICABLE', 'No VAL protocol in the source or frozen contract.', 'None.'),
    ('synthetic banks', 'all 7 methods', 'REQUIRES_FUTURE_APPROVED_EXECUTION',
     'No generation runner or bank writer exists for any method. E06b N_syn is DEFERRED_TO_M8. E01/E02 need the owner M6/M8 gate ruling. The M8 smoke-test hard gate applies.',
     'Blocks all generator-quality and downstream results.'),
    ('downstream detectors, VAL/TEST predictions, ACER/APCER/BPCER/EER/AUC, ROC/DET', 'all 7 methods + E00', 'REQUIRES_FUTURE_APPROVED_EXECUTION',
     'Implement the frozen ResNet-18 evaluator (configs/frozen/downstream_resnet18.yaml) and create the DINOv3 config (Q-06) in M10-M11; the per-sample prediction file follows spec s22.2.',
     'PENDING_DOWNSTREAM_EVALUATION.'),
    ('authoritative registry state', 'outputs/audit/method_status.csv, STAGE_STATE.json', 'RECOVERABLE_FROM_EXISTING_ARTIFACTS (separate reconciliation commit)',
     'See G_method_status_reconciliation_PROPOSED.csv. Not applied here.', 'Tables built from the registry would misreport training status.'),
]

# G. proposed method_status.csv reconciliation (NOT applied)
G = [
    ('E06c', 'production_qualification_status', 'PRODUCTION_RUNNER_QUALIFIED_ON_REAL_TRAIN (M6D5e; incl. checkpoint writer and resume)', 'unchanged'),
    ('E06c', 'scientific_execution_status', 'NOT_EXECUTED', 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS (24f7049; authority 84809868)'),
    ('E06c', 'completed_scientific_seeds', '0', '3'),
    ('E06c', 'checkpoint_status', 'NO_SCIENTIFIC_CHECKPOINT', 'OFFICIAL_GENERATOR_EPOCH_200 netG x3 (665684fd / 43ffa30d / dc6c54b4)'),
    ('E06c', 'disclosures', '... unchanged by M6F and A9', 'append: observed driver 580.178.04 vs lock 595.84 not recorded in run artifacts (pending owner disclosure ruling)'),
    ('E06c', 'latest_authoritative_milestone', 'M6D5e', 'BASELINE-EXEC E06c closure 24f7049'),
    ('E06b', 'scientific_execution_status', 'SCIENTIFIC_TRAINING_NOT_EXECUTED', 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS (3b0d308; authority d9ecbef; seed42 attempt 1 FAILED_TECHNICAL excluded)'),
    ('E06b', 'completed_scientific_seeds', '0', '3'),
    ('E06b', 'checkpoint_status', 'NO_SCIENTIFIC_CHECKPOINT', 'OFFICIAL_GENERATOR_EPOCH_200 netG x3 (d56122e0 / 645538e8 / e8d0d01c)'),
    ('E06b', 'disclosures', 'SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED; CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED; ...',
     'remove the two lifecycle items (exercised, fixed at d9ecbef); keep N_syn DEFERRED_TO_M8 and the SiW note; add the driver disclosure'),
    ('E06b', 'latest_authoritative_milestone', 'M6FD', 'BASELINE-EXEC E06b closure 3b0d308'),
    ('E05', 'production_qualification_status', 'NOT_QUALIFIED_ON_REAL_TRAIN', 'PRODUCTION_QUALIFIED_ON_REAL_TRAIN (efdf753)'),
    ('E05', 'scientific_execution_status', 'NOT_EXECUTED', 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS (7677c3a; authority efdf753)'),
    ('E05', 'completed_scientific_seeds', '0', '3'),
    ('E05', 'checkpoint_status', 'NO_SCIENTIFIC_CHECKPOINT', 'BASELINE_FINAL_STATE_V1 iteration_4000 x3 (4ad80b35 / b6a4cb2c / b7f0575f)'),
    ('E05', 'latest_authoritative_milestone', 'M6D4e', 'BASELINE-EXEC E05 closure 7677c3a'),
    ('E03', 'production_qualification_status', 'NOT_QUALIFIED_ON_REAL_TRAIN', 'PRODUCTION_QUALIFIED_ON_REAL_TRAIN (d547fc9; rulings bc7d8b9)'),
    ('E03', 'scientific_execution_status', 'NOT_EXECUTED', 'SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS (374de40 + ruling 320b42c; authority d547fc9)'),
    ('E03', 'completed_scientific_seeds', '0', '3'),
    ('E03', 'checkpoint_status', 'NO_SCIENTIFIC_CHECKPOINT', 'OFFICIAL_LATEST_FINAL_CKPT_50 x3 (data 5b337a63 / 316c94c0 / 9dbd463c)'),
    ('E03', 'disclosures', '... checkpoint rule OFFICIAL_LATEST_FINAL_CKPT_50', 'append: TF1 not bitwise reproducible across processes (disclosed); VAL diagnostics only'),
    ('E03', 'latest_authoritative_milestone', 'M6D2b', 'BASELINE-EXEC E03 closure 374de40 / 320b42c'),
    ('E04', 'production_qualification_status', 'NOT_QUALIFIED_ON_REAL_TRAIN', 'E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION 52/52 (b0f0ffd; D1-D9 13c981e; D12 c60481b)'),
    ('E04', 'scientific_execution_status', 'NOT_EXECUTED', 'unchanged (NOT_EXECUTED; seed launch needs a separate owner decision)'),
    ('E04', 'disclosures', 'controlled reconstruction ...', 'append: D12 procedural (non-bitwise) reproducibility, OWNER_APPROVED'),
    ('E04', 'latest_authoritative_milestone', 'M6D3e', 'BASELINE-EXEC E04 requalification b0f0ffd'),
    ('E01', '(all)', 'current', 'unchanged (IMPLEMENTED_NOT_EXECUTED)'),
    ('E02', '(all)', 'current', 'unchanged (IMPLEMENTED_NOT_EXECUTED)'),
]

TA_HEADER = ['method_id', 'method_name', 'track', 'architecture_and_source', 'fidelity', 'config_sha256',
             'dataset_population', 'input_resolution', 'batch', 'optimizer', 'learning_rate_and_schedule',
             'training_budget', 'seeds', 'checkpoint_rule', 'sources']
TA = [
    ('E06c', 'DSDG-BIN-IDFREE', 'A', 'DSDG VAE generator: netE_nir/netE_vis encoders + netG decoder, frozen LightCNN-29v2 identity net (FaceX-Zoo@16b793a7)',
     'CONTROLLED_ADAPTATION (DEV-020)', '7176dd4c...', 'pairs_train_v1 (a5e4fdae): 8838 TRAIN pairs (CASIA 2520, MSU 1200, SiW-Mv2 5118); 12668 faces; binary (1 spoof class)',
     '256', 'global 240 = 12 x 20 microbatches (M6D5c/d V2), tail 198; 37 steps/epoch', 'Adam over netE_nir+netE_vis+netG (netCls not optimized)',
     '2e-4 constant', '200 epochs = 7400 optimizer steps', '42, 1337, 2026', 'OFFICIAL_GENERATOR_EPOCH_200 (netG_model_epoch_200_iter_0.pth)',
     'configs/methods/e06c_dsdg_bin_idfree.yaml; metrics.jsonl e06c_run_start'),
    ('E06b', 'DSDG-NATIVE', 'B', 'same DSDG architecture, native type supervision (FaceX-Zoo@16b793a7)', 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
     '9d665dc2...', 'CASIA+MSU TRAIN: 3720 spoof rows with same-subject live partner, 60 subjects (print=0, replay=1); SiW not instantiable (no subject id)',
     '256', 'global 240 = 12 x 20 microbatches, tail 120; 16 steps/epoch', 'Adam over netE_nir+netE_vis+netG (netCls not optimized)', '2e-4 constant',
     '200 epochs = 3200 optimizer steps', '42, 1337, 2026', 'OFFICIAL_GENERATOR_EPOCH_200', 'configs/methods/e06b_dsdg_native.yaml; e06b_run_start'),
    ('E05', 'PCGAN', 'A', 'controlled architecture resolution on swapping-autoencoder@6baa180f (Encoder 0.93M, Generator 3.70M, ImageD 28.9M, PatchD 24.5M params)',
     'CONTROLLED_ADAPTATION (A2-07, A5, M6D4c)', '478756e1...', 'pairs_train_v1: x_src = source spoof, x_tgt = target live; 4000 distinct pairs consumed per seed (one partial pass)',
     '256', '1', 'Adam (0.9, 0.999), G and D', '1e-6 constant (G and D)', '4000 iterations (D then G each)', '42, 1337, 2026',
     'BASELINE_FINAL_STATE_V1 @ 4000 (iteration_4000.pth)', 'configs/methods/e05_pcgan.yaml; e05_run_start'),
    ('E03', 'STDN', 'A', 'official STDN (yaojieliu/ECCV20-STDN@c79f1f8c), TF 1.15 runtime compatibility', 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY',
     '08dde851...', 'feed pack c8c9850e: pairs_train_v1 TRAIN (12668 faces + M2 landmarks); VAL val_pairs_v1 1905 pairs for diagnostics only', '256 (map 32)',
     '2 pairs; G:D = 2:1', 'Adam (0.9, 0.999, 1e-8) G and D; weight EMA 0.9999', '6e-5 x 0.9^floor(global_step/20000), staircase on shared global step',
     '50 epochs x 2000 steps (G 100000, D 50000, global_step 150000)', '42, 1337, 2026', 'OFFICIAL_LATEST_FINAL_CKPT_50', 'configs/methods/e03_stdn.yaml; e03_run_start'),
    ('E04', 'Physics-STD', 'A', 'controlled reconstruction (A3 s4, A4): STDN lineage + 3DDFA_V2@1b6c6760/BFM depth-M0 pre-pass, Q=140; 1763902 trainable params',
     'CONTROLLED_ADAPTATION (D1-D9 13c981e, D12 c60481b)', '41861794...', 'pairs_train_v1 via feed pack db1908d5 / pre-pass 274a4b0a; 4 live + 4 spoof per iteration',
     '256 (depth 32)', '8 (4 live + 4 spoof)', 'Adam (0.9, 0.999, 1e-8), G and D (A4)', '5e-5 x 10^-floor(t/45000); D = G/2',
     '150000 iterations (3 steps each: G, D, G)', '42, 1337, 2026', 'BASELINE_FINAL_STATE_V1 @ 150000; periodic every 15000',
     'configs/methods/e04_physics_std.yaml; production_io.py; A4'),
    ('E01', 'FAS-Aug', 'A', 'official FAS-Aug operators (RizhaoCai/FAS-Aug@0da1dd79), non-learned', 'FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION', '87012ea7...',
     'pairs_train_v1 target lives; 8838 outputs per seed (n_syn_intended)', '256', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'NOT_APPLICABLE',
     'NOT_APPLICABLE (8838 transformations/seed)', '42, 1337, 2026 (operator seeds)', 'NOT_APPLICABLE', 'configs/methods/e01_fas_aug.yaml'),
    ('E02', 'FreqSub', 'A', 'spec-defined FFT block substitution (spec 8.2, A2-02..04), non-learned', 'SPEC_DEFINED', 'fdf74776...',
     'pairs_train_v1 (target live, source spoof); 8838 outputs per seed', '256', 'NOT_APPLICABLE', 'NOT_APPLICABLE', 'NOT_APPLICABLE',
     'NOT_APPLICABLE (8838 transformations/seed)', '42, 1337, 2026 (PCG64 pair seeds)', 'NOT_APPLICABLE', 'configs/methods/e02_freqsub.yaml'),
]

STAGES = ('generator_scientifically_trained', 'final_generator_checkpoint_available', 'synthetic_bank_generated',
          'downstream_detector_trained', 'val_predictions_available', 'test_predictions_available',
          'final_evaluation_metrics_available', 'dataset_wise_subgroup_tables_available')


def downstream_rows():
    rows = []
    for m in ('E06c', 'E06b', 'E05', 'E03', 'E04', 'E01', 'E02'):
        for s in (42, 1337, 2026):
            r = {'method_id': m, 'seed': s}
            if m in LEARNED:
                r.update(generator_scientifically_trained='YES', final_generator_checkpoint_available='YES (sha256 verified 2026-10-30)')
            elif m == 'E04':
                r.update(generator_scientifically_trained='NO (production-qualified, not launched)', final_generator_checkpoint_available='NO')
            else:
                r.update(generator_scientifically_trained='NOT_APPLICABLE (non-learned)', final_generator_checkpoint_available='NOT_APPLICABLE')
            for k in STAGES[2:]:
                r[k] = 'NO'
            r['cross_domain_tables'] = 'NOT_APPLICABLE (pooled held-out protocol)'
            r['status'] = 'PENDING_DOWNSTREAM_EVALUATION'
            rows.append(r)
    rows.append({'method_id': 'E00 (real-only lower bound)', 'seed': '42/1337/2026', 'generator_scientifically_trained': 'NOT_APPLICABLE',
                 'final_generator_checkpoint_available': 'NOT_APPLICABLE', **{k: 'NO' for k in STAGES[2:]},
                 'cross_domain_tables': 'NOT_APPLICABLE', 'status': 'PENDING_DOWNSTREAM_EVALUATION'})
    return rows


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    write(out / 'D1_table_A_training_configuration.csv', TA_HEADER, TA)
    write(out / 'C_figure_table_readiness_matrix.csv', ['item', 'scope', 'status', 'detail_class', 'source', 'reason'], FIG)
    write(out / 'E_e04_prelaunch_logging_checklist.csv', ['requirement', 'status', 'evidence', 'note'], E04)
    write(out / 'F_missing_information_recovery_plan.csv', ['field', 'scope', 'classification', 'recovery', 'impact'], F)
    write(out / 'G_method_status_reconciliation_PROPOSED.csv', ['experiment_id', 'column', 'current_value', 'proposed_value'], G)
    rows = downstream_rows()
    with open(out / 'H_downstream_evaluation_status.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
