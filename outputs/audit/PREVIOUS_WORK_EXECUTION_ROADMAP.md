# Previous-work execution roadmap (BASELINE-EXEC-P0)

Status: **PREVIOUS_WORK_EXECUTION_STAGED**. Date 2026-10-18. Authority `03217143a766f0bbeae6fcf90c45c4226881a0fc`
(branch `m6-baselines`). Machine-readable record: `PREVIOUS_WORK_EXECUTION_ROADMAP.json`.

This roadmap documents the current state. It does not rewrite any frozen spec, amendment or config, and nothing was
trained or launched. GPAT is deferred (`M7D1_GPAT_DEFERRED_FIX_BACKLOG.md`).

Authority read: `configs/CONFIG_STATUS.md` (M6E section), `outputs/audit/method_status.csv`, `M6E_FINAL_M6_CLOSURE.md`,
`EXECUTION_LEDGER.jsonl`, `third_party/registry.yaml`, `third_party/source_pins.json`, `docs/spec/amendments/` and
`configs/methods/`.

## 1. Scope

**Active previous work (7):** E01 FAS-Aug, E02 FreqSub, E03 STDN, E04 Physics-STD, E05 PCGAN, E06c DSDG-BIN-IDFREE,
E06b DSDG-NATIVE.

**Excluded or superseded (4; not revived):**

| Row | Scope | Authority |
|---|---|---|
| E07c DiffFAS-BIN-IDFREE | owner-excluded | A9, DEV-021 (seed-42 attempt 2 interrupted at step 54836; `model_050000.pt` is non-scientific evidence only) |
| E07b DiffFAS-NATIVE | owner-excluded | A9 (never implemented) |
| E06a DSDG-BIN | superseded | DEV-019 (A1): identity pairing is forbidden in Track A; replaced by E06c |
| E07a DiffFAS-BIN | superseded | DEV-019 (A1): replaced by E07c |

## 2. Readiness matrix

The frozen config SHA256 is given in short form below; full hashes are in the JSON. Every config is byte-identical to
its `frozen_config_snapshot/` copy. All rows use seeds 42 / 1337 / 2026.

| Row | Track | Fidelity | Config | Source pin | Lifecycle | Env | Real TRAIN | Scientific runner | Checkpoint rule | Resume |
|---|---|---|---|---|---|---|---|---|---|---|
| E06c | A | CONTROLLED_ADAPTATION (DEV-020) | `7176dd4c` | FaceX-Zoo `16b793a7` | learned training | gpat-m6-e06c ✔ | **qualified** (M6D5e-r) | **qualified** `tools/run_e06c.py` | OFFICIAL_GENERATOR_EPOCH_200 | **qualified** (explicit sidecar) |
| E06b | B | FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY | `9d665dc2` | FaceX-Zoo `16b793a7` | learned training | gpat-m6-e06c ✔ | **qualified** (M6FD) | CLI preflight qualified; lifecycle not exercised | OFFICIAL_GENERATOR_EPOCH_200 | fresh only |
| E05 | A | CONTROLLED_ADAPTATION (A2-07, A5) | `478756e1` | swapping-autoencoder `6baa180f` | learned training | gpat-m6-e05 (synthetic) | not qualified | not implemented | BASELINE_FINAL_STATE_V1 @ 4000 | none |
| E03 | A | FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY | `08dde851` | ECCV20-STDN `c79f1f8c` | learned training | gpat-m6-e03 TF1 (synthetic) | not qualified | not implemented | OFFICIAL_LATEST_FINAL_CKPT_50 | none |
| E04 | A | CONTROLLED_ADAPTATION (A3 §4, A4) | `41861794` | none official; 3DDFA_V2 `1b6c6760` + BFM + Q=140 | learned training + fixed depth-M0 pre-pass | gpat-m6-e03 + gpat-m6-e04-geometry (synthetic) | not qualified | not implemented | BASELINE_FINAL_STATE_V1 @ 150000 | none |
| E01 | A | FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION | `87012ea7` | FAS-Aug `0da1dd79` | **direct transformation bank, non-learned, CPU** | gpat-m5 (M6D1, synthetic) | not qualified | `generate_one` only | not applicable | not applicable |
| E02 | A | SPEC_DEFINED (gate adapted) | `fdf74776` | none (spec 8.2) | **direct transformation bank, non-learned, CPU** | gpat-m5 (M6D1, synthetic) | not qualified | `generate_one` only | not applicable | not applicable |

Budgets:
- E06c: 200 epochs × 37 steps (8838 rows, global batch 240) = 7400 steps/seed.
- E06b: 200 × 16 steps (3720 rows) = 3200 steps/seed.
- E05: 4000 iterations at batch 1.
- E03: 50 epochs × 2000 steps at batch 2.
- E04: 150000 iterations at batch 8.
- E01 and E02: 8838 outputs per seed each.

## 3. Blockers by method

- **E06c:** none. Before launch: fast-forward the GPU clone (`ffffa9b` at P0) to the launch commit, re-run the
  preflight, and get the owner's launch instruction.
- **E06b:** two lifecycle items are open, `CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED` and
  `SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED`. Close them with a short qualification-root lifecycle check or an explicit
  owner acceptance. N_syn stays DEFERRED_TO_M8; that blocks the bank, not training.
- **E05:** needs four things:
  - a real-TRAIN pair data path with a TRAIN-only audited firewall;
  - a production loop, a terminal-4000 checkpoint writer and the run lifecycle;
  - a fail-closed CLI;
  - real-TRAIN qualification.
- **E03:** needs four things:
  - a TF1 production controller (faces + M2 landmarks, flips, ckpt-1..50 writer, lifecycle);
  - a ruling on the official `val_steps = 500` block. VAL is diagnostics-only and never selects, so the block must
    either be firewall-audited or shown to be non-influential;
  - a fail-closed CLI;
  - real-TRAIN qualification.
- **E04:** needs four things:
  - live-TRAIN depth-M0 production, with fixed assets hashed once and reused for all seeds;
  - a TF1 controller (shared with E03), a terminal-150000 writer and the lifecycle;
  - a CLI;
  - real-TRAIN qualification.

  It has the largest budget, and its wall time is not yet measured.
- **E01 / E02:** need a real-TRAIN bank writer with per-sample accounting (E02 also needs the NumPy pin, A2-03). They
  also need an **owner ruling** on whether a per-seed bank counts as M6 method execution or as an M8 bank. An M8 bank
  is held by the deterministic smoke test (`HARD_GATE_BEFORE_FIRST_FULL_M8_PLUS_EXECUTION`).

## 4. Recommended execution queue

The GPU is one shared RTX 3090, so exactly one GPU training job runs at a time. No GPAT job runs during this program.

| Rank | Row | Resource | Reason |
|---|---|---|---|
| 1 | E06c | 3090 | The only Track-A baseline whose real-TRAIN runner, checkpoint writer and resume are qualified. Live preflight passed. |
| 2 | E06b | 3090 | Same environment; data path and CLI are qualified; only the lifecycle exercise is open; 3200 steps/seed. |
| 3 | **E05** | 3090 | **Moved up from 5.** It is in the same tier as E03/E04 (synthetic only), but it is PyTorch, so it reuses the qualified E06c RunContext, firewall and pair-loader patterns. Its iteration is already qualified and its budget is the smallest. |
| 4 | E03 | 3090 | A new TF1 controller. It must precede E04 (E04 trains in the E03 interpreter and uses the STDN lineage and A4 Adam). |
| 5 | E04 | 3090 + depth pre-pass | Depends on the E03 controller and on depth-M0 production; largest budget. |
| 6 | E01 | CPU | Non-learned bank. Needs a bank writer and the gate ruling. It does not need the 3090, so once qualified it can run alongside GPU work. |
| 7 | E02 | CPU | Same lifecycle and gate as E01. |

Only one relative position changed: E05 moved ahead of E03 and E04.

## 5. Scientific fairness

- Each baseline keeps its own frozen protocol. GPAT's optimizer, epochs, batch size, architecture and selection rule are
  not imposed on any baseline.
- Controlled adaptations are never called faithful or native. Reporting labels and forbidden wording come from each
  frozen config.
- Checkpoint semantics follow each frozen rule:
  - E03: `ckpt-50`.
  - E06b and E06c: epoch-200 `netG`.
  - E04 and E05: BASELINE_FINAL_STATE_V1.
  - E01 and E02: not applicable.
- Seeds 42 / 1337 / 2026 are all primary evidence. Using a best seed, or dropping or rerunning only a poor seed, is
  forbidden. A failed seed is rerun as the same seed after a technical fix (spec §27).
- TEST is never used for model or checkpoint selection.

## 6. Next method: E06c DSDG-BIN-IDFREE (staged, NOT launched)

| Item | Value |
|---|---|
| Host | `student20261@100.121.84.44`, one RTX 3090. At P0 it was idle: 198 MiB used, 0 %, no compute processes |
| Repo | `/home/student20261/workdir/GPAT_TransferBench`, clean at `ffffa9b`. Fast-forward it to the launch commit first. E06c code is byte-identical between `ffffa9b` and P0 |
| Environment | `/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python` (lock `91416a20…`) |
| Runtime root | `/home/student20261/workdir/GPAT_TransferBench_runtime` |
| Run roots | `…/runs/m6/E06c/seed_42`, `seed_1337`, `seed_2026`. None exists at P0 |
| Run ID | `sha256("E06c\|<seed>\|7176dd4c…\|<launch commit>")[:16]` (run_logging_v1). It is fixed only once the launch commit is known |
| Seeds | 42, then 1337, then 2026, strictly in sequence |
| Budget | 200 epochs, 37 optimizer steps/epoch, 7400 steps/seed. Global batch 240 (12 × 20), tail 198. FP32, TF32 off, no AMP |
| Duration (estimate) | about 8.4 h/seed and 25 h for three seeds (M6D5e-r real-TRAIN epoch took 150.83 s) |
| Storage (estimate) | 21 checkpoint events × 726 MB, about 15.3 GB/seed and 46 GB in total; 220 GB free |
| Assets | `pairs_train_v1.parquet` `a5e4fdae…` (8838 rows), faces_256 via `m5_gpu_3090.yaml` `c9d22abf…`, LightCNN `d0750746…` (123,844,849 B), FaceX-Zoo `16b793a7` tree `0d2216bd…`, A1 adaptation `a0f84fac…`, M6D5c/M6D5d/M6D5e overlays |
| Checkpoint rule | OFFICIAL_GENERATOR_EPOCH_200. The authoritative final is `netG_model_epoch_200_iter_0.pth`. Saves happen at epoch 1 and every 10 epochs. Resume sidecars are never selectable |
| VAL/TEST | TRAIN only. E06c uses no VAL: its periodic block writes TRAIN-sample grids only. TEST is never touched |

Command, run inside tmux on the GPU host. Run it with `--preflight-only` first; it must exit 0 with
`torch_imported = false`. Then run it again without that flag.

```
cd /home/student20261/workdir/GPAT_TransferBench
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=<seed> NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python tools/run_e06c.py --seed <seed> \
    --execution-config configs/execution/m5_gpu_3090.yaml [--preflight-only]
```

After a technical stop only, resume with the same command plus `--resume-state checkpoints/runner_state_epoch_<E>.pth`.

**Hard gates** (enforced by the CLI unless noted):
- The worktree must be clean.
- `PYTHONHASHSEED` must equal the seed; TF32 must be off; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- The seed must be in {42, 1337, 2026}.
- Every identity is verified before torch is imported.
- An existing seed root is refused unless an indexed `--resume-state` is given.
- Override flags are refused.
- Data access is TRAIN-only.
- The GPU is exclusive (operational rule).
- Spec §27 applies: rerun the same seed, never substitute one.

**Live preflight at P0:** E06c `--preflight-only` for seed 42 exited 0 at `ffffa9b` with `torch_imported = false`. The
E06b preflight also exited 0.

GPU-host contact in P0 was read-only (git, listing, nvidia-smi, df, the two preflights). One temporary `/tmp` file was
written on the host and then removed.
