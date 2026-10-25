# E05 seed42: scientific launch and early qualification

**Status:** `E05_SEED42_LAUNCHED_EARLY_QUALIFICATION_PASS`. The machine-readable record is `E05_SEED42_LAUNCH.json`.

This commit is reporting evidence only. The E05 scientific training authority stays **`efdf753156045dd2bed52229d9eac869b18d72f8`**, and the GPU checkout trains from that commit: it was fast-forwarded from `eede868` and is clean.

| | |
|---|---|
| method | E05, **PCGAN (controlled architecture resolution)**, CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE |
| host | `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`. Reached via the fallback route `100.121.84.44`, because `10.31.131.201` returned "No route to host". |
| run | seed 42, run_id **`18463bf5cd471e55`** (recomputed independently), fresh and with no resume. Root `runs/m6/E05/seed_42` was absent before launch. |
| launch | 2026-10-25T04:39:31Z, tmux `m6_e05_s42` (remain-on-exit), PID 3455662. Exit marker `~/workdir/e05_s42.exit`; exact command in `~/workdir/e05_s42.command`. |

## Owner rulings applied

- **Driver:** `lock_driver_version = 595.84`, `observed_driver_version = 580.178.04`, `driver_policy = OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`.
  - Only the driver version field is tolerated. The GPU model and UUID match, and every other lock field is exact; this is recorded in the run_start event.
  - Nothing was re-locked, rewritten, installed or downgraded.
  - This is not evidence that arbitrary driver changes are acceptable.
- **Production resolutions:** these are approved controlled implementation resolutions, not native, faithful or official PCGAN choices.
  - x_src is the source_spoof face and x_tgt the target_live face.
  - Pairs are drawn by a seeded shuffle with batch 1, 0 workers and drop_last false; iteration t uses sampler position t−1.
  - Inputs are mapped to [-1, 1], with no resize, crop or flip.
  - Checkpoints are written at iteration 1 (periodic) and iteration 4000 (selected, BASELINE_FINAL_STATE_V1).

## Identities and preflight

- **Identities:**
  - config `478756e1…` = snapshot; pair manifest `a5e4fdae…`; lock `849a1004…`;
  - source `6baa180f…` / tree `25a434f1…`; Python `d3255ce6…`;
  - fused_bias_act `5afdb236…`, upfirdn2d `40ba9db7…`, with no JIT build;
  - FP32, with TF32, AMP, autocast and cuDNN benchmark off and cuDNN deterministic on;
  - seeds set by `apply_framework_seed(42)`.
- **Preflight** (`--preflight-only`, PYTHONHASHSEED=42): exit 0, `torch_imported = false`, method E05, seed 42, commit `efdf753`, run root absent and not created, no image read and no model executed.

## Early qualification (same scientific run)

The run was checked at 366 completed iterations, as of 04:41:34Z:

- **Identity:** run_id, seed, authority, CONTROLLED_ADAPTATION, `NOT_QUALIFIED_FRESH_ONLY`, a single run_start event and no `--resume`.
- **Data:**
  - All 366 sampler positions equal the independent seed-42 prediction (one int64 draw, then `randperm(8838)`), with no duplicates.
  - In every pair the source has label spoof and the target has label live, and both are TRAIN.
  - Every input is [1,3,256,256] torch.float32 in [-1, 1].
- **Optimizer:** after each iteration t, D = G = t and the total is 2t, both in the runner and in the independent hooks.
- **Losses:** all 14 values are finite and the five G terms are active. `L_G_total` equals the unit sum of the five terms (maximum relative FP32 rounding 1.3e-7). Iteration 1 → 366:

  | | iteration 1 | iteration 366 |
  |---|---|---|
  | L_rec | 224.0 | 220.0 |
  | L_recblur | 105.6 | 112.3 |
  | L_advrec | 0.704 | 0.703 |
  | L_advmix | 0.705 | 0.704 |
  | L_pat | 0.712 | 0.814 |
  | L_D_total | 2.831 | 2.601 |

- **Gradients:** 80/80 D and 59/59 G finite in every iteration, with the inactive group's gradients None.
- **Checkpoint at iteration 1:**
  - `checkpoints/iteration_0001.pth`, 232,113,504 bytes, sha256 `82d06063…` (recomputed, equal to the index).
  - `periodic`, `selected_for_final = false`.
  - All 167 tensors round-trip equal; the applications recorded at write time are 1/1 and iteration 2 shows a total of 4, so no extra step was taken.
  - No runner_state or resume sidecar.
- **Speed:** median 0.31 s per iteration, peak 3.08 GB, stderr empty.
- **Firewall** (stat-only atime: pre 04:39:17Z, mid 04:40:55Z, 20 627 files, no image opened):

  | class | files | atime-sensitive | changed |
  |---|---|---|---|
  | TEST | 3027 | 3027 | **0** |
  | VAL | 3121 | 3121 | **0** |
  | siwmv2 TRAIN | 9507 | 9501 | 293, all faces of consumed pairs |
  | CASIA/MSU TRAIN | 4960 | 0 | read by E06b < 24 h earlier, so relatime does not update them |

  No manifest changed. **VAL 0, TEST 0.**

## Boundary

The run is left running untouched. No seed1337/2026 run, other method, bank, downstream evaluation or GPAT work was started, no production code changed, and the E06b/E06c runs were not touched.
