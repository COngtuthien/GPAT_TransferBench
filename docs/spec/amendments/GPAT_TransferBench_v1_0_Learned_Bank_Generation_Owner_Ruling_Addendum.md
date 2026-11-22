# GPAT-TransferBench v1.0: learned-method bank generation owner ruling (approved addendum)

Date: 2026-11-22. Branch `m6-baselines`. Machine-readable record:
`configs/amendments/learned_bank_generation_contract.yaml` (`OWNER_APPROVED`).

Scope: bank generation for E03 STDN, E05 PCGAN, E06b DSDG-NATIVE and E06c DSDG-BIN-IDFREE, seeds 42 / 1337 / 2026.

The ruling approves the DRAFT `configs/amendments/learned_bank_generation_contract.DRAFT.yaml`. That DRAFT was committed
at `5f6aabb` (SHA256 `217e5145…1aad`) and stays unchanged. Its assessment is
`outputs/audit/LEARNED_BANK_AUTHORITY_ASSESSMENT.md` (`3433ad0`).

All 14 owner decisions were checked against the DRAFT before activation. For each one:

- the option identifier exists in the DRAFT for that question;
- it equals the DRAFT's recommended option.

The ruling introduces no scientific choice that differs from the DRAFT.

**This ruling authorizes no bank.**

- No generator is retrained and no checkpoint changes.
- No VAL or TEST data is used for bank generation.
- No full production bank is authorized. Each production family needs its own later owner authorization, granted only after
  that method's S7 toy acceptance has passed.
- S7 toy acceptance on the GPU stays mandatory, and it is gated by E04 completion (C-Q5).
- The storage root is governed separately by `configs/amendments/learned_bank_storage_location.*`.

## Decisions

| Question | Decision |
|---|---|
| E03-Q1 inference mode | `A_OFFICIAL_TEST_MODE`: `Gen(x, training_nn=False)`, moving BN statistics |
| E03-Q2 weight source | `A_RAW_VARIABLES`: `tf.train.Saver` restore of the raw variables (test.py:58), never the EMA shadows |
| E03-Q3 inference augmentation | `A_NONE`: no brightness change, no flip; RGB256 / 255; feed-pack landmarks / 256, unflipped |
| E03-Q5 auxiliary outputs | `A_NONE`: only the synthetic spoof image |
| E05-Q1 StyleGAN2 noise | `A_PER_PAIR_SEEDED`: `torch.manual_seed(pair_seed)` immediately before the single mix forward |
| DSDG-Q1 latent stream | `A_OFFICIAL_BATCHES_SEEDED`: seeded official batches of 100 (identity noise, then spoof noise) |
| DSDG-Q2 output half | `A_SPOOF_ONLY`: channels 0:3 |
| DSDG-Q4 bank identity | `A_SYNTHETIC_ID`: `<method>_S<seed>_<index:06d>`, zero-based; pair fields null |
| E06b-Q0 N_syn | `A_POOLED_8838`: Track B coverage distinction preserved and disclosed |
| Q4 float → uint8 | `A_SOURCE_IMAGE_WRITER`: each pinned source's own clip / scale / truncation |
| C-Q1 encoding | `A_PNG_LOSSLESS`: E01/E02 PNG writer settings; pixel and file SHA256 |
| C-Q2 device / reproducibility | `A_GPU_FP32_BITWISE_REPLAY` |
| C-Q3 S7 acceptance | `A_TOY_TWO_RUN_BITWISE` |
| C-Q5 GPU timing | `A_AFTER_E04_COMPLETES` |

## Operational definitions

### E03 STDN

1. Gen runs on the source spoof image at batch 1 (`test.py:49`).
2. The synthesis lines then run verbatim:
   - `recon1 = (1-s)*img - b - resize(C) - T` (`train.py:47`)
   - `trace = img - recon1` (`train.py:48`)
   - `trace_warp = warping(trace, reg, 256)` (`train.py:49`)
   - `synth1 = img_live + trace_warp` (`train.py:50`)
3. The offset map is `reg = generate_offset_map(lm_sp, lm_li)` (`dataset.py:142`).
4. Conversion runs in-graph: `tf.cast(tf.clip_by_value(synth1, 0, 1) * 255, tf.uint8)` (`utils.py:24,37`).

Two other steps in `plotResults` do not apply to the bank:

- The 256→256 `resize_images` is skipped by TF1 at equal static size.
- The RGB→BGR swap only offsets the channel order of `cv2.imwrite`.

Neither changes the stored RGB values.

### E05 PCGAN

```
pair_seed = int.from_bytes(sha256(b'GPAT_TRANSFERBENCH|E05|BANK|' + str(seed) + '|' + pair_id)[:8], 'big') mod (2^31 - 1)
```

1. Map inputs as `(x/255 - 0.5)/0.5`.
2. Run the encoder on `x_src = source_spoof` and on `x_tgt = target_live`.
3. Call `torch.manual_seed(pair_seed)`.
4. Run one mix forward: `G(spatial(x_src), global(x_tgt))`.
5. Convert with the pinned `util.tensor2im`: `((x+1)/2*255)`, clip to `[0, 255]`, `astype(uint8)`.

### E06b / E06c DSDG

```
bank_seed = int.from_bytes(sha256(b'GPAT_TRANSFERBENCH|' + method + b'|BANK|' + str(seed))[:8], 'big') mod (2^31 - 1)
```

The stream is generated as follows:

1. `g = torch.Generator().manual_seed(bank_seed)`.
2. For each batch of 100, in order:
   - `noise = randn(100, 128, generator=g)`
   - `noise_s = randn(100, 128, generator=g)`
   - `z = cat(noise_s, noise, noise)`
   - `fake = netG(z)`
   - keep `spoof = fake[:, 0:3]`
3. Convert each kept row with `(255*x).astype(uint8)` (no clip, `generated.py:63`).
4. Keep the first N rows in stream order. The discarded tail of the final batch is counted in provenance:
   - N = 8838: 89 batches, 62 discarded.
   - Toy N = 186: 2 batches, 14 discarded.

Further DSDG rules:

- `pair_id`, `target_live_id` and `source_spoof_id` are null.
- No `pairs_train_v1` identifier is ever assigned.
- E06b and E06c keep distinct methods, checkpoints and seed domains.

The byte encoding of the seed preimages (UTF-8 of the ASCII string) and test vectors are recorded in the YAML.

## Reproducibility, S7 and GPU timing

### Reproducibility (C-Q2)

- Inference runs on the RTX 3090 in each method's locked environment, in FP32.
- Determinism settings:
  - E03 uses the qualified M6D2b precision environment and session configuration.
  - E05, E06b and E06c use the qualified PyTorch flags: no TF32, cudnn.benchmark False, deterministic True.
  - DSDG's official `cudnn.benchmark = True` is overridden to False as a determinism control only.
- None of these settings changes an architecture, a checkpoint or a graph operation.
- If bitwise replay fails: keep the evidence, stop, and report to the owner. Never weaken the requirement.

### S7 toy acceptance (C-Q3)

Inputs:

- E03/E05: the 186 S6 toy pairs, with toy manifest SHA256 `50ac8fb8…`, at their original global positions.
- E06b/E06c: stream indices 0..185.

Each input set runs twice, in fresh roots. The gates are listed in the YAML. Synthetic fixtures never replace this
acceptance.

### GPU timing (C-Q5)

Learned-bank GPU inference runs only after all of the following:

1. genuine E04 runner completion;
2. the E04 seed-2026 completion audit PASS;
3. the formal E04 three-seed family closure, or documented sufficient roadmap authority;
4. a resource preflight.

Neither tmux disappearance nor a final checkpoint counts as E04 completion.
