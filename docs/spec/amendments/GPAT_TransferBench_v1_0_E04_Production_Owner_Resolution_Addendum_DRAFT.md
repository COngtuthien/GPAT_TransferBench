# GPAT-TransferBench v1.0: E04 Physics-STD production owner resolution — **DRAFT, NOT APPROVED**

**Status:** `DRAFT_FOR_OWNER_FINAL_APPROVAL`. Nothing in this document is an owner ruling until the owner approves it
explicitly. It creates no E04 scientific authority and authorizes no execution (pre-pass, training, bank or evaluation).
**Machine-readable companion (DRAFT):** `configs/amendments/e04_production_resolution.DRAFT.yaml`.
**Evidence:** `outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION.{json,md}`, with the readiness audit
`outputs/audit/E04_PRODUCTION_READINESS_AUDIT.*` (a36acb5).

Unchanged frozen authority: `configs/methods/e04_physics_std.yaml` (`41861794…`, equal to its snapshot), Amendments A3 §4
and A4, A2-06 `BASELINE_FINAL_STATE_V1`, the M6D3d owner resolution and the M6D3e implementation record. Fidelity stays
`CONTROLLED_ADAPTATION`; reporting label "Physics-STD (controlled geometry/depth reconstruction)".

Classes used below: **SOURCE FACT** (pinned code or frozen data, verified), **FROZEN** (already required by authority),
**PROPOSED** (controlled adaptation or implementation resolution awaiting approval), **RUNTIME** (measured
compatibility observation). "Accepted in principle" records the owner's earlier direction; it is not final approval.

## D1 — 3DDFA crop (accepted in principle: official landmark rule, frozen M2 landmarks)

- SOURCE FACT: pinned 3DDFA_V2 `1b6c6760` `TDDFA.__call__(img, objs, crop_policy='landmark')` computes
  `roi_box = parse_roi_box_from_landmark(pts)` (`utils/functions.py`, sha `4245b8c5…`): the extent box of `pts[0]` (x)
  and `pts[1]` (y), squared about its centre to half-side `r = max(w, h)/2`, then a square of side `√2·2r` centred
  there, with no vertical shift. `crop_img` rounds the four corners with Python `round()` (half to even) and zero-pads
  everything outside the image. The crop is resized to 120×120 with `cv2.INTER_LINEAR`, then `ToTensorGjz` and
  `NormalizeGjz(127.5, 128)`, the pinned MobileNet, and `param·std + mean`. `recon_vers(dense_flag=True)` uses the
  unrounded float `roi_box` in `similar_transform`.
- SOURCE FACT: coordinate conventions agree. 3DDFA vertices are 0-based pixel coordinates with pixel centres at
  integers (`similar_transform` subtracts 1 from the MATLAB-based x and maps y as `size − y`). M2 `landmarks_px256`
  are the official FaceXFormer `denorm_points` (`align_corners=False`) mapped centre-preserving to 256
  (`gpatbench/preprocess/aux_models.py:128-130`). The same convention applies, so no offset or rescale is needed.
  Only one landmark-based convention exists.
- PROPOSED: `pts = landmarks_px256.T` (`[2, 68]`, float32, all 68 iBUG points, unmodified) from `e03_feed_pack_v1`
  (manifest `c8c9850e…`, payload verified). The image is the canonical face, verified against the M2 accounting
  `face_png_sha256` and read with `cv2.imread(path, cv2.IMREAD_COLOR)` (BGR uint8, the official demo convention).
  Everything else is the official path above. Fail closed (no fallback) on non-finite landmarks, a zero-extent box or
  an empty crop. Disclosure: the landmarks come from the frozen FaceXFormer M2 cache, not from 3DDFA's own tracking
  landmarks; both are 68-point iBUG.
- Test vectors: `D1_crop_vectors` in the evidence (synthetic landmarks; the pinned functions).
- Still to measure in qualification, with no choice involved: how often real ROIs exceed the 256 frame and are
  zero-padded.

## D2 — pre-pass population (accepted in principle: TRAIN only)

- SOURCE FACT (manifest-only read): `pairs_train_v1` (`a5e4fdae…`) has 8838 rows, stored sorted by `pair_id`.
  It holds 3830 unique `target_live_id` (all TRAIN, label 0; reuse 1–23, mean 2.308) and 8838 unique
  `source_spoof_id` (all TRAIN, label 1), with no overlap. All 12668 faces are covered by the frozen M2 landmarks in
  `e03_feed_pack_v1`.
- FROZEN: spoof `M₀` = exact float32 zeros 32×32 (paper Eq. 21). Live `M₀` follows the A3/A4 pipeline. Assets are
  generated once and reused unchanged for seeds 42/1337/2026 (A3 §4.2b). No VAL or TEST face is included.
- PROPOSED: for each of the 3830 live faces, store the fit, the ROI, the dense mesh, live `M₀` (A3/A4), the z-buffer
  face mask and the Q140 xy (`dense_vertices[q140_indices, :2]`, canonical-256 pixels). For each of the 8838 spoof
  faces, store the fit, the ROI and the Q140 xy only. Spoof faces are never rendered and no spoof depth is stored or
  derived; spoof `M₀` is the constant zero target. Per-pair offsets are
  `dense_offsets(source_xy = spoof Q140, target_xy = live Q140)` (M6D3e). The pack carries per-sample sha256 and has
  0 TEST rows. The manifest's `attack_macro` column is never read.

## D3 — pre-pass device (accepted in principle: CPU)

- RUNTIME: 3 clean CPU processes of the M6D3a geometry worker on the current host are byte-identical in every field.
  Fitted parameters, normalised input, mesh and depth equal the committed M6D3a values, which were recorded under
  driver 595.84. CPU geometry is driver-independent. CUDA fits and meshes differ from CPU; final depth is identical.
- PROPOSED: CPU only, in `gpat-m6-e04-geometry`, with the qualified settings
  (`tools/m6d3a_e04_geometry_preflight.py:90-96`): `torch.set_num_threads(1)`, `torch.manual_seed(0)`,
  `torch.use_deterministic_algorithms(True)`, cuDNN benchmark off and deterministic on, TF32 off. Sim3DR is CPU by
  construction. Two independent pre-pass executions must produce byte-identical packs before acceptance.

## D4 — pair sampling (accepted in principle: per-pass permutations continuing across passes)

- PROPOSED algorithm (reference `tools/e04_rng_test_vectors.py`):
  - **Seed:** `pair_seed = 1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|PAIR_ORDER|<s>")[0:4]) mod (2^31 − 1))`.
  - **Generator:** one `numpy.random.RandomState(pair_seed)` (legacy MT19937; NumPy freezes its stream).
  - **Passes:** pass `p = 0, 1, …` is `rng.permutation(8838)`. The indices address `pairs_train_v1` rows in stored
    order. The passes are concatenated and nothing is dropped.
  - **Iteration t** (0-based) consumes `stream[4t : 4t+4]`. Slot `j` gives `rgb[j] = target_live`,
    `rgb[4+j] = source_spoof`, `depth[j] = M₀(live)`, `depth[4+j] = 0`, and `offsets[j]` for that pair.
  - **Totals:** 600000 draws = 67 full permutations + 7854 rows of the 68th.
  - **No replacement, no oversampling, no reordering.** A batch that straddles a pass boundary may in principle hold
    one row twice. For seeds 42/1337/2026 this never happens (34 straddling iterations each, 0 repeats).
  - **Restart:** E04 is fresh-only, and a failed seed is rerun as the same seed from iteration 0 (spec §27). The stream
    is a pure function of the seed, so any iteration's rows are re-derivable by replay. Each iteration's row indices
    are logged in `metrics.jsonl` for independent audit.
- Test vectors: `seeds` in `E04_RNG_TEST_VECTORS.json`. They are identical under numpy 1.23.5 (Python 3.8, training
  env) and numpy 2.5.3 (Python 3.12, geometry env and laptop).

## D5 — augmentation (accepted in principle: none added)

- FROZEN, intrinsic to the model and retained: the frequency decomposition (bilinear 32/128 low-pass, ×15/×25 bands),
  geometric trace warping with the pre-pass offsets, the hard-sample perturbation (D7) and the additive/inpainting
  synthesis.
- PROPOSED: no data-pipeline augmentation. No flip, brightness, crop, colour or noise. STDN's TRAIN flips and
  brightness jitter are not inherited (M6D3d `stdn_reference_policy` is reference-only, and flips would also require
  flipping `M₀` and Q140).

## D6 — VAL diagnostics (accepted in principle: none)

- FROZEN: VAL is optional and diagnostics-only and never selects (`data.splits.VAL`, `run_logging_v1`
  `val_role: DIAGNOSTIC_ONLY`). `val_losses` / `val_metrics` are "null + reason if the method exposes none". The final
  checkpoint is `BASELINE_FINAL_STATE_V1` at iteration 150000.
- PROPOSED: no VAL block. The E04 runner reads neither VAL nor TEST, and VAL fields are explicit null with the reason
  `NO_VAL_PROTOCOL_IN_SOURCE_OR_FROZEN_CONTRACT`. Precedent: the accepted E06c, E06b and E05 closures had no VAL
  access. Disclosure: the spec's schematic loop passes `validate=split.VAL`, and fig24 will show no E04 VAL curve.

## D7 — hard-sample RNG (direction acceptable; algorithm requires explicit approval)

- SOURCE FACT: the qualified graph contains no TF random op apart from the weight initialisers
  (`RandomStandardNormal`, seeded by the graph seed). The hard-sample draws enter as two placeholders:
  `hard_scales [4,1,1,3]` (uniform[0,1] per spoof for B/C/T) and `hard_keep [4,1,1,1]` (Bernoulli(.5) inpainting
  keep). The qualified worker draws them from one `RandomState` with `uniform(0,1,(4,1,1,3))`, then
  `uniform(0,1,(4,1,1,1)) >= .5`, and feeds the same arrays to all three steps of the minibatch. **No graph change is
  needed** to give them an independent stream.
- PROPOSED algorithm:
  - **Seed:** `hard_seed = 1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|HARD_SAMPLE|<s>")[0:4]) mod (2^31 − 1))`.
  - **Generator:** one `numpy.random.RandomState(hard_seed)`.
  - **Before Step 1 of each iteration t:** `u = uniform(0,1,(4,1,1,3))`, `k = uniform(0,1,(4,1,1,1))`,
    `hard_scales = float32(u)`, `hard_keep = float32(k >= 0.5)` (comparison in float64). That is 16 doubles per
    iteration; the same arrays are fed to Steps 1–3.
  - **Independence:** the stream is independent of the D4 pair stream and of the data. The TF graph seed stays the
    experiment seed (`methods/common/learned.py` `seed_plan`) for weight initialisation only. All derived seeds are
    distinct from each other and from the experiment seeds.
  - **Audit:** the draws for every iteration are logged. The stream position after t iterations is 16t, and the
    whole stream can be replayed in about 1 s.
- Evidence: the realization with seed 314159 reproduces the M6D3e fixture hashes `b735482c…` / `7472a1a3…`. Full
  150000-iteration streams are identical across numpy 1.23.5 / 2.5.3. Keep rates are 0.50016 / 0.50019 / 0.49986.

## D8 — input encoding (accepted in principle: RGB / 255)

- SOURCE FACT: the canonical faces are RGB uint8 256×256 sRGB PNG (`preprocess_v1.canonical_face`). The generator
  receives the raw tensor and applies only its paper frequency decomposition; there is no input normalisation layer.
  The discriminators take RGB. L_P thresholds `|T_A| > β = 0.1`, depth `M₀` is in [0,1], and the pinned STDN
  predecessor feeds `float32 / 255` (`model/dataset.py:144`). The M6D3e fixtures lie in [0.2, 0.8].
- PROPOSED: RGB channel order (decode to RGB; for the training feed this is independent of the D1 BGR geometry
  read), `float32(uint8) / 255` applied once in the feed to live and spoof alike. No mean/std normalisation, no
  clipping and no second scaling.

## D9 — runtime compatibility (requires explicit approval)

- RUNTIME, measured on synthetic data with qualification seed 314159 in 4 clean processes plus the readiness run:
  - **Identical to the committed M6D3e evidence:** the TF 1.15.5 / CUDA 11080 / cuDNN 8907 identity, GraphDef
    `7e3e84cc…`, fixtures, initial state, BN update sets, changed-trainable sets, every pre-step loss and the final
    losses.
  - **Step 1:** 4 tensors, `D/D3/conv{1..4}/BN/moving_mean` (shapes 32/64/96/128), differ from history. The
    historical raw tensors were not retained, so this rests on their recorded summaries: min/max move by at most
    2 float32 ulps at each tensor's largest magnitude (|Δ| ≤ 4.66e-10), and the relative ΔL2 is ≤ 8.8e-9. They are identical across all current processes, and the
    two historical processes were identical to each other. The difference is deterministic per runtime, and only the
    driver differs (595.84 → 580.178.04).
  - **Steps 2/3:** history varied in 51 D3 tensors, 13 of them trainable. All current processes reproduce historical
    process 2 on 47 of them, including every trainable. The only current variation is the same 4 moving means
    (2 variants, |Δ| ≤ 4.66e-10).
  - **Consumers:** the moving statistics are read only by their moving-average update ops (`Identity`, `Sub`,
    `AssignSub`) and the initial `Assign`. With `training=True` FusedBatchNorm never reads them. They therefore cannot
    affect any loss, gradient, trainable variable or generator output.
  - **Open:** the exact kernel is not identified (CUPTI unavailable), the same limitation M6D3e accepted.
- PROPOSED: extend `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY` (driver version field only; GPU model,
  UUID and every other lock field exact) to E04 training in `gpat-m6-e03`, disclosing the Step-1 D3 BN finding. The
  pre-pass runs on CPU (D3) and is measured driver-independent.

## Implementation-defined within authority (frozen later in the production contract, not owner decisions)

- **Checkpoints:** `implementation_defined_periodic_plus_terminal`, e.g. every 15000 iterations plus
  `iteration_150000` with zero extra optimizer steps.
- **Metrics:** one `metrics.jsonl` record per iteration, including pair rows and hard draws.
- **Feed and workers:** dense offsets computed in the feed with the pinned SciPy 1.10.1; workers per
  `IMPLEMENTATION_ADAPTABLE_NO_SCIENTIFIC_IMPACT`.
