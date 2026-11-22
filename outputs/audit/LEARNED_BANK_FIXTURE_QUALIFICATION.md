# M8 learned-bank adapters: CPU synthetic-fixture qualification

Date: 2026-11-22, run 15:32–15:37Z.

Code authority: `b09a1cc`, a clean, isolated clone at
`~/workdir/m8_learned_banks/clones/qual_b09a1cc` on the GPU host. The E04 checkout `18ba00e` was not used; the pinned
source caches were copied from it read-only.

Contract: `configs/amendments/learned_bank_generation_contract.yaml` (`d0b4e4e`).

**Label: `SYNTHETIC_FIXTURE_QUALIFICATION_NOT_S7_NOT_BENCHMARK`.**

- These runs prove nothing about GPU bitwise reproducibility (C-Q2).
- They do not replace the real S7 toy acceptance (C-Q3). S7 stays mandatory and is gated by E04 completion (C-Q5).

| Family | Environment | Gates | Evidence (SHA256) |
|---|---|---|---|
| E06b + E06c DSDG | gpat-m6-e06c, Python 3.11.16, torch 2.12.1+cu130, NumPy 2.4.6 | **60 / 60** | `LEARNED_BANK_FIXTURE_QUALIFICATION_DSDG.json` (`e16cbf06…`) |
| E05 PCGAN | gpat-m6-e05, Python 3.11.16, torch 2.12.1+cu130, NumPy 2.4.6 | **33 / 33** (one is informational, see E05) | `LEARNED_BANK_FIXTURE_QUALIFICATION_E05.json` (`dd3b57c6…`) |
| E03 STDN | gpat-m6-e03, Python 3.8.20, TF 1.15.5 (NVIDIA), NumPy 1.23.5 | **32 / 32** | `LEARNED_BANK_FIXTURE_QUALIFICATION_E03.json` (`555d1668…`) |

## Execution conditions

- Every run used `CUDA_VISIBLE_DEVICES=` with 2 CPU threads, at `nice 19` and `ionice -c3`. No GPU was used.
- E03 ran with the qualified M6D2b precision environment, with CUDA hidden.
- All inputs were artificial:
  - random-init fixture weights;
  - synthetic PNG faces and synthetic landmarks written into the qualification work directory;
  - fixture seed 60911, which is never an experiment seed.
- An audit hook counted every `open` under `<runtime_root>/data` (benchmark faces and records): **0** in every family.
- No VAL or TEST data and no real TRAIN face was read.

## Real checkpoints

All twelve checkpoints were hash-verified before and after each family's run and found unchanged. They were inspected
read-only, with no forward pass and no model output.

| Method | Inspection |
|---|---|
| DSDG | official payload `{'epoch': 200, 'model': DataParallel(Decoder_s)}`; keys and shapes equal a pinned `Decoder_s(hdim=128)`; all tensors finite |
| E05 | `weights_only` load; `E05_BASELINE_STATE_V1`, iteration 4000; FP32 Encoder/Generator states finite |
| E03 | `tf.train.list_variables`: all 134 inference-graph variables are present in each ckpt-50 with equal shapes; the 170 EMA shadows in the checkpoint are not restored |

## What each family qualified

### DSDG (E06b, E06c)

Loading:

- The official `misc/util.py::load_model` loads into `DataParallel(Decoder_s)`, with full key coverage and bytes equal
  to the checkpoint.
- netG has 19,887,494 parameters in 21 tensors and is in eval mode.

Stream:

- `torch.randn(100, 128, generator=g)` is bitwise equal to the official `torch.zeros(100, 128).normal_(0, 1)` draw on
  the same generator state.
- The 186-row toy stream uses 2 batches and discards 14 samples.
- The latent stream hash is identical across independent loads.
- An independent recomputation of rows 0, 85, 99, 100 and 185 (official draw expression, then reference
  `(255*x).astype(uint8)`) equals the bank pixels.

Outputs:

- The sigmoid output lies in [0, 1] with shape (100, 6, 256, 256) FP32.
- Rows are identified by synthetic ids with null pair lineage.
- The toy rows are byte-identical to the first 186 rows of a 250-row stream. This is the "toy rows = production rows"
  property.
- E06b and E06c streams differ.

### E05 PCGAN

Checks that use the real code:

- The pinned `util.tensor2im` segment equals the reference conversion, including clipping at both ends.
- The approved input mapping `(x/255-0.5)/0.5` is applied.
- The pinned encoder source draws no random numbers.

Checks that use the interface fixture:

- The per-pair seed fixes the generator noise.
- Without a seed, consecutive forwards differ, so no zero-noise substitution happens.
- One generator forward runs per pair, with no reconstruction forward.
- Pair lineage, global positions, pair seeds, 10 O_NOATIME TRAIN-allowlist reads and order independence (a single-pair
  bank reproduces the row) all hold.
- An independent recomputation equals the bank pixels.

**Limitation.** The real StyleGAN2 Encoder/Generator run only through the qualified CUDA kernels (`fused_bias_act`,
`upfirdn2d`), so the genuine E05 forward cannot execute on CPU.

- The fixture modules are labelled `INTERFACE_FIXTURE_NOT_PCGAN_ARCHITECTURE`.
- The gate `real_forward_not_run_on_cpu` is informational, not a pass of the real forward.
- The real E05 forward is S7 GPU work.

### E03 STDN

Source and graph:

- The cited lines match the pinned source byte for byte: train.py:44/47–50, utils.py:24/37, test.py:55/58 and
  dataset.py:142.
- The inference graph is all `STDN/*` with no EMA, no BN update ops, and no mutating or random ops reachable from the
  outputs.
- The graph is finalized, so adding an op raises.

Weights:

- The raw-variable Saver restore is exact for fixture weights, including non-trivial BN moving statistics.

Inputs and conversion:

- Landmarks are unflipped and divided by 256 through the unchanged `LandmarkAdapter`.
- Inputs are RGB256 / 255.
- The offset map `generate_offset_map(lm_sp, lm_li)` is finite.
- The in-graph `clip`/`cast` equals the reference `clip(x, 0, 1) * 255 → uint8`.
- The row is not stochastic, and reads are O_NOATIME.

### Common to all families

Bank mechanics:

- The bank reader verifies schema, identity, file set and file and pixel SHA256.
- Outputs are 256×256×3 uint8 PNGs with 0 failures.
- Two fresh roots are bitwise identical on CPU.
- Files are mode 0444 and directories 0555.
- A real overwrite attempt is refused and leaves the bank unchanged.

Authority and safety:

- The approved contract, pinned-source digests and frozen config identities were all verified.
- S7 refuses with `LEARNED_BANK_BLOCKED_BY_E04_COMPLETION`.
- Production refuses for every method and seed.
- Optimizer construction, `backward` and `torch.save` are forbidden in inference processes: zero calls, and an attempt
  raises.

## Disclosures

1. **DSDG checkpoint loading.** The official DSDG checkpoints pickle the `DataParallel` module. torch ≥ 2.6 defaults to
   `weights_only=True`, so `load_model` is called with `weights_only=False`, and only for this call. The file SHA256 is
   verified first; these are the project's own closed training artifacts.
2. **E03 fixture identity hash.** The fixture-checkpoint identity hash for E03 includes TF's `checkpoint` index file,
   which records the absolute work-directory path. A dry run (`dryrun1` clone, an uncommitted copy of the same code at
   `d0b4e4e`) therefore has a different E03 fixture identity. Its bank pixels and PNG bytes are identical to the
   evidence run's. Real banks use the approved checkpoint identity from the committed table instead.
3. **Earlier attempt.** The first E03 dry run stopped at `pinned_sources_verified` because the dry-run clone's source
   cache was laid out wrong (a `cp` into a missing directory). The evidence clone was built correctly. Both dry runs
   stay on the host under `~/workdir/m8_learned_banks/qualification/dry1_*`.

## Per-method status

| Method | Status |
|---|---|
| E03 | `BANK_ADAPTER_IMPLEMENTED_SYNTHETIC_QUALIFIED_S7_PENDING` |
| E05 | `BANK_ADAPTER_IMPLEMENTED_SYNTHETIC_QUALIFIED_S7_PENDING` (real forward qualifiable only on GPU) |
| E06b | `BANK_ADAPTER_IMPLEMENTED_SYNTHETIC_QUALIFIED_S7_PENDING` |
| E06c | `BANK_ADAPTER_IMPLEMENTED_SYNTHETIC_QUALIFIED_S7_PENDING` |

No method is production-ready. The real S7 toy acceptance on the GPU is required first, then a per-family owner
production authority.
