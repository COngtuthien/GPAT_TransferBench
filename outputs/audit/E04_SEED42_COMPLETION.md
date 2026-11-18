# E04 seed42: scientific run completed cleanly

**Status:** `E04_SEED42_COMPLETED_CLEANLY`, 90/90 gates passing. The machine-readable record is `E04_SEED42_COMPLETION.json`; it embeds both read-only host collections, so `tools/e04_seed_completion_audit.py build` can re-derive it.

This commit is reporting evidence only. The E04 scientific launch authority stays **`18ba00e94813f005876c8d6235b0b848226b2c1d`** (production qualification `b0f0ffd`, D12 ruling `c60481b`, criteria `9f1a2471…`).

| | |
|---|---|
| method | E04 **Physics-STD**, track A_FAIR_COMMON_IDENTITY_FREE, fidelity `CONTROLLED_ADAPTATION` (reporting label "Physics-STD (controlled geometry/depth reconstruction)") |
| run | seed = 42, run_id = **`40befa6004aee85c`** = sha256("E04\|42\|41861794…\|18ba00e…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`: one run_start, fresh run root |
| window | 2026-10-30T02:54:14Z → 2026-11-01T19:30:02Z, 232547 s of training; step time median 1.251 s (p05 1.165, p95 2.225, max 2.8) |
| exit | `~/workdir/e04_s42.exit` = 0, written by the launch wrapper after the runner returned; end marker 2026-11-01T19:30:04Z; the runner completion record is the last pane-log line; no runner process; stdout/stderr/generation_log empty |
| host | no reboot or shutdown inside the run window (run boot `7.0.0-38-generic 2026-10-29T09:44:27+07:00`; next boot `7.0.0-43-generic 2026-11-13T13:16:33+07:00`, after the run ended); the tmux server is absent now (later reboots), so the live pane is not used as evidence |

## Completion accounting

- **Iterations = 150000** (contiguous 0..149999, global_step = iteration + 1), **final global_iteration = 150000**, minibatch_phase 0 at every checkpoint.
- **G applications = 300000** (Step 1 Eq.23 and Step 3 Eq.24, one shared G Adam), **D applications = 150000** (Step 2 L_D). Each iteration record holds the losses fetched in the three apply runs; the graph asserts the G → D → G phase order.
- **Adam beta powers:** 7 checkpoint values in the normal float32 range equal float32(0.999)^(applications+1) exactly, e.g. beta2_power at 15000 = 30000 applications; beta2_power_1 at 15000 = 15000 applications; beta2_power at 30000 = 60000 applications. The rest are flushed to 0 as expected.
- **Learning rate:** G 5e-5·10^-⌊t/45000⌋, D half; the largest relative float32 read-back error is 2.5e-08.
- **Metrics log:** 150011 records (1 run_start, 150000 iterations, 10 checkpoint events), every line parses, no missing or duplicate iteration, every loss component finite.

Training-loss block means (iterations 0–1499 and 148500–149999), for orientation only:

| loss | first block | last block |
|---|---|---|
| step1/loss_G_step1 | 23351.6 | 8848.46 |
| step1/L_depth | 0.197053 | 0.114151 |
| step2/loss_D | 529.087 | 829.198 |
| step3/loss_G_step3 | 116931 | 19554.2 |

## Data feed and RNG (D4, D7, D8, D2/D3)

- **Seeds:** experiment 42; PAIR_ORDER 679869707; HARD_SAMPLE 320493565; TF graph seed 42.
- **Streams:** pair stream `0dea19a0889cdf6c494fb76d7f6684b504124d1897af71c74191c7b99c26a20e` and hard stream `ed4f32ed21e819f02b588efd49cd3cc91151ad73c945f3bbab14a4f6731cfaf3` equal the approved vectors and the production stream function.
- **Record by record:** an independent numpy RandomState replay of the rule text matches the pair rows, pair ids, pass indices and hard draws of all 150000 iterations. The hard and depth input sha256 recomputed from the replay and the feed pack also match. The dense offsets of 83 sampled iterations match, recomputed from the pack geometry.
- **Faces:** 12668 TRAIN samples consumed (the whole pack). Split, label, dataset and M2 face sha256 match for all of them. Feed pack `db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b`, pre-pass `274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279`, 0 VAL and 0 TEST rows.

## Checkpoints (BASELINE_FINAL_STATE_V1)

- iteration_015000 … iteration_150000 (30 files) exist and equal the index and the checkpoint events.
- Every checkpoint holds the 500 graph variables, all finite. Its state fingerprint equals the live-state fingerprint logged by the runner at save time.
- Trainables that changed against the previous checkpoint (of 144): 30000: 144, 45000: 144, 60000: 144, 75000: 144, 90000: 144, 105000: 144, 120000: 144, 135000: 144, 150000: 138. In the final block, the unchanged ones are float32 absorption, not a missing update: at lr 5e-8 an Adam step is below half an ulp of a BN gamma near 1.
- The final checkpoint is **iteration_150000**, selected by the frozen rule, not by loss or VAL:
  - path `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E04/seed_42/checkpoints/iteration_150000`
  - `.data-00000-of-00001` **`55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6`**
  - `.index` `f83983429ee5f7f8b82e8802439173f54e6d7684b4ffab4ad9f6425fb40dcc02`
  - `.meta` `fc441707a1300b74a1eaea53f1265bbbf05cb25b7204d7457925cbea66b24933`
- No partial, temporary or resume file. Full list: `E04_SEED42_CHECKPOINT_MANIFEST.txt`.

## Firewall

Stat-only atime snapshots: pre 2026-10-30T02:53:53Z, post 2026-11-18T11:14:08Z, 20627 files. No image was opened by the audit.

| split | files | atime-sensitive | changed | changed, consumed | changed, not consumed | consumed |
|---|---|---|---|---|---|---|
| TEST | 3027 | 3027 | 0 | 0 | 0 | 0 |
| TRAIN | 14467 | 1799 | 12668 | 12668 | 0 | 12668 |
| VAL | 3121 | 3121 | 0 | 0 | 0 | 0 |

**TEST = 0** and **VAL = 0**: none of those faces changed, and no consumed sample is VAL or TEST. `test_split_accessed` and `val_split_accessed` are false.

## Environment, D9 and D12

- Driver: lock 595.84, observed **580.178.04**, `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional narrow scope)`. GPU model and UUID are exact, and there is no other lock difference, both at run_start and on the live re-check.
- TensorFlow 1.15.5, CUDA runtime 11080, cuDNN 8907, 59 locked packages; Python 3.8 pathlib backport disclosed.
- OS kernel during the run: `Linux-7.0.0-38-generic-x86_64-with-glibc2.17`. At audit time: `Linux-7.0.0-43-generic-x86_64-with-glibc2.17`. The kernel is not part of the E04 lock; it is disclosed only.
- D12: D12 procedural reproducibility: independent GPU executions of the frozen graph are not bitwise identical (GPU ScatterNd gradients of the warp GatherNd ops and the forward AddN T_A); this run is not compared bitwise to any other execution. It never excuses a missing or wrong optimizer update, non-finite state or a corrupted checkpoint, which are gated above (D, F, G).
- GPU sharing: not sampled continuously. 43 of 100 blocks of 1500 iterations have a median step time of at least 1.25 × the overall median. That is consistent with intermittent sharing and is not a validity criterion.
- Host after completion: the run boot ended without a wtmp shutdown record when the host restarted at `2026-11-13T13:16:33+07:00` (after the run ended). Later shutdowns: `7.0.0-43-generic 2026-11-16T20:37:29+07:00`. Kernel updates installed after the run (unattended upgrades) changed only the OS kernel; the NVIDIA driver version is unchanged.

## Run-root integrity

- 39 files; LC_ALL=C `<path> <size> <sha256>` listing digest **`c50d18e832557c70e257bd81e33bdbb0d2e58adedec46ff20d076ed00fbf4431`**, identical before and after the audit.
- metrics.jsonl `57f233118e28041d5ad1e5985fa30454de5442f42a0ba382167cf7d6494d4b1b`. It holds every per-iteration loss, learning rate and timing value needed for later plots.

## Boundary

No downstream metric, synthetic bank, best-checkpoint or best-seed logic. ACER, APCER, BPCER, EER and ROC need the later frozen downstream pipeline; none is derived from training losses.
