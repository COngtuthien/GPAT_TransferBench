# E03 STDN: production runner qualification (Track-A, real TRAIN/VAL data)

**Status:** `E03_PRODUCTION_RUNNER_QUALIFIED`, 49/49 gates passing.

| | |
|---|---|
| Qualification seed | 60903 (QUALIFICATION_ONLY, not a scientific result) |
| Qualified candidate | `c07ecdd72c227c7b8fcace15e12dc23b28e9c8c8` |
| Production code | byte-identical in this commit (gate M) |
| Machine-readable record | `E03_PRODUCTION_RUNNER_QUALIFICATION.json`, which embeds the collected results; rebuild with `tools/e03_production_qualification_evidence.py build` |
| Scientific roots | none created (`runs/m6/E03` absent); no scientific seed launched |

**Execution fidelity:** `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY`. The frozen field `FAITHFUL_OFFICIAL` stays as history. The three layers are kept separate:

- **Official, unchanged:** the pinned `train._step` graph (Gen/Disc_s, losses, hard samples, Adam with exponential decay on the shared global_step, loss and weight EMA, inline BN updates) and the `train.py` epoch order and fetches.
- **Controlled Track-A adaptation (owner rulings V/R1/R2/R3, decisions D1–D3):**
  - frozen pair relations and the M2 faces/landmarks feed pack;
  - independent TRAIN/VAL parser RNG objects;
  - a serialized map and `prefetch(1)`.
- **Runtime compatibility:** the M6D2b NVIDIA TF 1.15.5 runtime, plus two disclosed Python 3.8 infrastructure items (below).

No exact official dataset reproduction or bitwise reproduction of the authors' environment is claimed.

## Observed evidence

**Data access**

- **External drive:** `/dev/nvme0n1p5` (NTFS, label `Data`) was already mounted at `/media/cong/Data` (udisks2), so no mount action was needed.
- **M2 shards:** the 81 `landmarks_px256` shards and 81 indexes match `M2_CACHE_SHARD_INDEX.csv` byte for byte (20 615 rows, no extra file). No source byte was changed.

**Feed pack** (`tools/e03_feed_pack.py`, verifier `methods/stdn/feed_pack.py`)

- **Determinism and pin:** two independent builds are byte-identical, with manifest sha256 `c8c9850e…`, pinned in `production_io`.
- **Contents:**
  - TRAIN: 8838 pairs over 12 668 samples (3830 live, 8838 spoof);
  - VAL: 1905 pairs over 2653 samples (748 live, 1905 spoof);
  - 15 321 landmark rows (exact `<f4` M2 values); 5294 M2 rows were not taken, including every TEST row.
- **Per-block verification:** every selected block was checked by the M2 procedure: block sha256, `np.load(allow_pickle=False)`, shape/dtype, and the npy re-encode sha256.
- **TEST exclusion:** 0 TEST rows or samples. The canonical `split_v1` is checked in the builder, the receiver and `tests/test_e03_production_runner.py`.
- **Transfer:** staged at `…/data/processed/.e03_feed_pack_v1.partial` on the GPU host and verified there by the receiver (Python 3.8, NumPy 1.23). It was then published by an atomic rename to `/home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/e03_feed_pack_v1`, and its hashes are unchanged at the end of qualification.

**Environment**

- **Host:** `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`.
- **Exact matches to the E03 lock:** the Python string and executable, TF 1.15.5, the 59 packages (inventoried at the fixed site-packages path), native CUDA 11080 / cuDNN 8907, the precision environment and the session config.
- **Driver:** lock 595.84, observed 580.178.04 (`OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`). The lock was not modified.

**Graph** (identical in all cases)

- 747 global variables, 170 trainable (98 G, 72 D, equal to the M6D2b oracle), 170 Adam ops, 60 BN moving, 170 weight EMA, 2 loss EMA.
- The feed and the VAL graph create 0 variables.
- The VAL graph reaches no state-mutating op, and its 48 FusedBatchNorm ops all have `is_training=False`.

**Schedule** (`val_a`, `val_b`, `no_val`: 201 + 8 TRAIN steps)

- Each step is one `sess.run`: 105 runs of `losses + [g_op, d_op, fig]` and 104 of `losses + [g_op, fig]`.
- The shared global_step went +2 / +1 at every step; the final values are G = 209, D = 105, global_step = 314.
- The `beta1_power` values equal the float32 emulation for 209 G and 105 D applications.
- An odd step changes no D variable. An even step changes all 98 G and 72 D trainables.
- The learning rate is float32(6e-5) on both groups, read from the shared step. Decay boundaries at 20 000 k are covered by the static tests.

**Data and RNG**

- **Prediction:** the independent predictor (TF index pipelines plus NumPy RandomState, no controller code) matches all 418 consumed TRAIN positions and flips and all 102 VAL positions and flips. Both splits reshuffle on every pass and each pass is a permutation.
- **Seeds:** TRAIN uses parser and shuffle seed 60903; VAL uses 2050698081, from D2.
- **Determinism across processes:** the data decisions (pairs, flips, brightness deltas) are identical in all three processes.
- **TRAIN parser state:** it is exactly a function of its parse count.
- **Mapping:** target_live → live and source_spoof → spoof.

**VAL non-influence** (hard gate, real VAL block, 51 steps)

- **In-process byte comparison** before and after the VAL block, covering:
  - the sha256 of each of the 747 variable value bytes;
  - the shared global_step and lr;
  - the TRAIN parser call/draw counts and RandomState state.
- **Result:** all identical in `val_a` and `val_b`. Only the VAL parser and iterator advanced, and VAL never changes the global_step.
- **Cross-process:** the TRAIN pair, flip and brightness decisions after VAL are identical with VAL (`val_a`, `val_b`) and without it (`no_val`).

**Checkpoints**

- **ckpt-1:** written after the epoch-1 TRAIN block and before VAL, at global_step 302. It holds 3 files (`.data`, `.index`, `.meta`), each indexed with its sha256 and size. The `.data` and `.index` sizes, 20 749 008 and 28 161 bytes, equal those of the authors' removed `ckpt-50` files.
- **Round trip:** all 747 tensors equal and finite, and saving changed no state.
- **ckpt-50 semantics probe:** written by the official saver at global_step 3, recorded as `selected` with `OFFICIAL_LATEST_FINAL_CKPT_50` and labelled `QUALIFICATION_TERMINAL_SEMANTICS_PROBE_NOT_50_EPOCHS`. It is not a trained ckpt-50.

**Figures (R3)**

- Written: `Epoch-1-1`, `Epoch-1-201`, `Epoch-2-1`, `Epoch-1-Val-1` and `Epoch-1-Val-51`, each from the `fig`/`fig_val` of that same run.
- Every session run is accounted for exactly (279 / 279 / 228 / 17 / 7), so figures add no run, iterator pull or RNG draw.

**Lifecycle**

- The qualification cases close `interrupted` with QUALIFICATION_ONLY labels.
- `failure_probe` closes `failed` with no checkpoint or `.partial` file, and a second open of the same root is refused (fresh only).
- No runner_state or resume sidecar exists.

**Firewall**

- **Access audit:** the audit hook denied nothing, and nothing was written in the runtime outside the qualification roots. Faces were read only for feed-pack samples.
- **Snapshots:** stat-only atime, pre 10:07:57Z and post 10:19:09Z.
- **Results:** TEST 0 changed of 3027 (all sensitive). The VAL and TRAIN changes are all faces the parsers actually read. No manifest was modified.

**CLI** (`tools/run_e03.py`)

- **Preflights:** 42, 1337 and 2026 exit 0, with no TensorFlow import, no image or array decode and no run root.
- **Refusals (exit 2, nothing written):** seed 60903, `--resume`, `--epochs`, `--val-steps`, `--test` and a wrong `PYTHONHASHSEED`.

## Interpretation and disclosed limitations

- **Numeric training is not bitwise reproducible across processes.**
  - The initial state and step-0 losses are identical. The first float difference appears after the first optimizer update (step 1, before any VAL) and grows chaotically, to a maximum relative loss difference of 0.29 over 209 steps.
  - The two identical with-VAL processes differ exactly like with-VAL vs no-VAL.
  - The unchanged official graph contains atomic-accumulation GPU kernels (10 `ResizeBilinearGrad`, 4 `ScatterNd` from the `gather_nd` warp), so this is a property of graph plus runtime. VAL does not cause it, and the graph was not changed to suppress it.
  - The VAL non-influence verdict therefore rests on the in-process byte identity and the cross-process identity of data decisions, not on equal losses.
- **Python 3.8 infrastructure compatibility** (not STDN or TF APIs):
  1. An exact CPython 3.9 `PurePath.is_relative_to` backport is installed only in the E03 controller process, because the shared hash-pinned `methods/common/config.py` uses it.
  2. During `LandmarkAdapter` construction only, the source check is answered by `production_io.verify_source_checkout`, a Python 3.8 implementation of the same checks as `learned.verify_source` (pin, top level, HEAD, tree, origin, blobs, cited digests). That function uses `str.removesuffix`, and the laptop run of both checks gives identical results.
  3. The package inventory reads the fixed site-packages path, because TF/setuptools add vendored packages to `sys.path`.
- **Round 1:** candidate `68a42c2` stopped in `failure_probe` (case not registered). The fix is `c07ecdd`; its other round-1 cases had also passed the in-process VAL gate. Round-1 roots remain as diagnostics.

## Tests

- `tests/test_e03_production_runner.py`: 20 static tests covering the rulings, D2 vectors, split streams, relations, schedule, LR, ckpt-before-VAL, cadence, pipeline, flip rule, pack verification and TEST exclusion, the CLI and fresh-only behaviour.
- `tests/test_e03_production_qualification_evidence.py`: the evidence and rebuild tests.
- The existing M6D2a and M6D2b tests are retained.
