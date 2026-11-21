# E04 seed1337: scientific run completed cleanly

**Status:** `E04_SEED1337_COMPLETED_CLEANLY`, 90/90 gates passing. The machine-readable record is `E04_SEED1337_COMPLETION.json`; it embeds both read-only host collections, so `tools/e04_seed_completion_audit.py build` can re-derive it.

This commit is reporting evidence only. The E04 scientific launch authority stays **`18ba00e94813f005876c8d6235b0b848226b2c1d`** (production qualification `b0f0ffd`, D12 ruling `c60481b`, criteria `9f1a2471…`).

| | |
|---|---|
| method | E04 **Physics-STD**, track A_FAIR_COMMON_IDENTITY_FREE, fidelity `CONTROLLED_ADAPTATION` (reporting label "Physics-STD (controlled geometry/depth reconstruction)") |
| run | seed = 1337, run_id = **`a42fdd2a2accaf0a`** = sha256("E04\|1337\|41861794…\|18ba00e…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`: one run_start, fresh run root |
| window | 2026-11-18T11:28:43Z → 2026-11-21T08:09:39Z, 247255 s of training; step time median 1.526 s (p05 1.249, p95 2.015, max 2.7) |
| exit | `~/workdir/e04_s1337.exit` = 0, written by the launch wrapper after the runner returned; end marker 2026-11-21T08:09:40Z; the runner completion record is the last pane-log line; no runner process; stdout/stderr/generation_log empty |
| host | no reboot or shutdown inside the run window: the host has not rebooted since before launch (run boot = current boot `7.0.0-43-generic 2026-11-17T15:16:38+07:00`, wtmp "still running", boot_id `5c3d5b941cc940a18ddfb5a105433743`, btime 2026-11-17T08:16:30Z; journal boot spans 2026-11-17T08:16:56Z → 2026-11-21T17:13:17Z); the tmux session is still present (pane dead after the wrapper finished) but is not required as evidence |

## Completion accounting

- **Iterations = 150000** (contiguous 0..149999, global_step = iteration + 1), **final global_iteration = 150000**, minibatch_phase 0 at every checkpoint.
- **G applications = 300000** (Step 1 Eq.23 and Step 3 Eq.24, one shared G Adam), **D applications = 150000** (Step 2 L_D). Each iteration record holds the losses fetched in the three apply runs; the graph asserts the G → D → G phase order.
- **Adam beta powers:** 7 checkpoint values in the normal float32 range equal float32(0.999)^(applications+1) exactly, e.g. beta2_power at 15000 = 30000 applications; beta2_power_1 at 15000 = 15000 applications; beta2_power at 30000 = 60000 applications. The rest are flushed to 0 as expected.
- **Learning rate:** G 5e-5·10^-⌊t/45000⌋, D half; the largest relative float32 read-back error is 2.5e-08.
- **Metrics log:** 150011 records (1 run_start, 150000 iterations, 10 checkpoint events), every line parses, no missing or duplicate iteration, every loss component finite.

Training-loss block means (iterations 0–1499 and 148500–149999), for orientation only:

| loss | first block | last block |
|---|---|---|
| step1/loss_G_step1 | 28483.3 | 5878.78 |
| step1/L_depth | 0.198214 | 0.112697 |
| step2/loss_D | 486.937 | 807.838 |
| step3/loss_G_step3 | 121892 | 10257.3 |

## Data feed and RNG (D4, D7, D8, D2/D3)

- **Seeds:** experiment 1337; PAIR_ORDER 1349475396; HARD_SAMPLE 35554372; TF graph seed 1337.
- **Streams:** pair stream `5c18cc0823127f18df8527d79a7d17a40525b6400d1534224c327c9d20caa802` and hard stream `2cdc01ca64445e2a48933e7dbbd9bd4fe449b4dcbe9c45d643b5cabed8431e04` equal the approved vectors and the production stream function.
- **Record by record:** an independent numpy RandomState replay of the rule text matches the pair rows, pair ids, pass indices and hard draws of all 150000 iterations. The hard and depth input sha256 recomputed from the replay and the feed pack also match. The dense offsets of 83 sampled iterations match, recomputed from the pack geometry.
- **Faces:** 12668 TRAIN samples consumed (the whole pack). Split, label, dataset and M2 face sha256 match for all of them. Feed pack `db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b`, pre-pass `274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279`, 0 VAL and 0 TEST rows.

## Checkpoints (BASELINE_FINAL_STATE_V1)

- iteration_015000 … iteration_150000 (30 files) exist and equal the index and the checkpoint events.
- Every checkpoint holds the 500 graph variables, all finite. Its state fingerprint equals the live-state fingerprint logged by the runner at save time.
- Trainables that changed against the previous checkpoint (of 144): 30000: 144, 45000: 144, 60000: 144, 75000: 144, 90000: 144, 105000: 144, 120000: 144, 135000: 144, 150000: 130. In the final block, the unchanged ones are float32 absorption, not a missing update: at lr 5e-8 an Adam step is below half an ulp of a BN gamma near 1.
- The final checkpoint is **iteration_150000**, selected by the frozen rule, not by loss or VAL:
  - path `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E04/seed_1337/checkpoints/iteration_150000`
  - `.data-00000-of-00001` **`fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5`**
  - `.index` `1f284d8be5a4234bc4a7cdead2aa49b00ee413907c83c2d962ac1cbc06fea560`
  - `.meta` `bade2ca14ae01e67384705543789bf39d4baabd8411f7253fbbda7ba139af233`
- No partial, temporary or resume file. Full list: `E04_SEED1337_CHECKPOINT_MANIFEST.txt`.

## Firewall

Stat-only atime snapshots: pre 2026-11-18T11:28:38Z, post 2026-11-21T17:13:00Z, 20627 files. No image was opened by the audit.

| split | files | atime-sensitive | changed | changed, consumed | changed, not consumed | consumed |
|---|---|---|---|---|---|---|
| TEST | 3027 | 3027 | 0 | 0 | 0 | 0 |
| TRAIN | 14467 | 14467 | 12668 | 12668 | 0 | 12668 |
| VAL | 3121 | 3121 | 0 | 0 | 0 | 0 |

**TEST = 0** and **VAL = 0**: none of those faces changed, and no consumed sample is VAL or TEST. `test_split_accessed` and `val_split_accessed` are false.

## Environment, D9 and D12

- Driver: lock 595.84, observed **580.178.04**, `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional narrow scope)`. GPU model and UUID are exact, and there is no other lock difference, both at run_start and on the live re-check.
- TensorFlow 1.15.5, CUDA runtime 11080, cuDNN 8907, 59 locked packages; Python 3.8 pathlib backport disclosed.
- OS kernel during the run: `Linux-7.0.0-43-generic-x86_64-with-glibc2.17`. At audit time: `Linux-7.0.0-43-generic-x86_64-with-glibc2.17`. The kernel is not part of the E04 lock; it is disclosed only.
- D12: D12 procedural reproducibility: independent GPU executions of the frozen graph are not bitwise identical (GPU ScatterNd gradients of the warp GatherNd ops and the forward AddN T_A); this run is not compared bitwise to any other execution. It never excuses a missing or wrong optimizer update, non-finite state or a corrupted checkpoint, which are gated above (D, F, G).
- GPU sharing: not sampled continuously. 18 of 100 blocks of 1500 iterations have a median step time of at least 1.25 × the overall median. That is consistent with intermittent sharing and is not a validity criterion.
- Host after completion: no reboot or shutdown since before launch (the audit ran in the same boot, `5c3d5b941cc940a18ddfb5a105433743`); the kernel during the run and at audit time are the same boot.

## Run-root integrity

- 39 files; LC_ALL=C `<path> <size> <sha256>` listing digest **`6a8f471effd41d0e1eb93bbfa455c86cd64518a30fc36278bd3cd9e6fec7585c`**, identical before and after the audit.
- metrics.jsonl `eda6e3940136bd3a9ec8b092ef3559e5c77ad0e5dc690edc53e02dd5047748a9`. It holds every per-iteration loss, learning rate and timing value needed for later plots.

## Boundary

No downstream metric, synthetic bank, best-checkpoint or best-seed logic. ACER, APCER, BPCER, EER and ROC need the later frozen downstream pipeline; none is derived from training losses.
