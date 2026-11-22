# M8 learned-method bank authority assessment (E03, E05, E06b, E06c)

Status: **LEARNED_BANK_AUTHORITY_ASSESSED_OWNER_DECISIONS_REQUIRED** (110/110 gates). Base commit `35ad1fdb242809aae822a413ba6f13179bb5060e`; host collection 2026-11-22T14:43:16Z.

This record authorizes nothing. No inference, bank, GPU use or HDD write happened.

## 1. Checkpoint authority (12 runs)

| Method | Seed | Run | Checkpoint | SHA256 | Rule | Final step | Seed closure | Family closure | Verified |
|---|---|---|---|---|---|---|---|---|---|
| E03 | 42 | `a565b76bc7349532` | `checkpoints/ckpt-50` | `5b337a63db63e405… (data)` | OFFICIAL_LATEST_FINAL_CKPT_50 | 150000 | `b95c127` | `374de40` | yes |
| E03 | 1337 | `e8c85cc378e778f5` | `checkpoints/ckpt-50` | `316c94c066c32314… (data)` | OFFICIAL_LATEST_FINAL_CKPT_50 | 150000 | `5cdb65d` | `374de40` | yes |
| E03 | 2026 | `237887eebab18637` | `checkpoints/ckpt-50` | `9dbd463cb3eb56c9… (data)` | OFFICIAL_LATEST_FINAL_CKPT_50 | 150000 | `15bfc84` | `374de40` | yes |
| E05 | 42 | `18463bf5cd471e55` | `checkpoints/iteration_4000.pth` | `4ad80b35e759154f…` | BASELINE_FINAL_STATE_V1 | 4000 | `072ce7d` | `7677c3a` | yes |
| E05 | 1337 | `036c806bb2f339b7` | `checkpoints/iteration_4000.pth` | `b6a4cb2c61b1b582…` | BASELINE_FINAL_STATE_V1 | 4000 | `023beb4` | `7677c3a` | yes |
| E05 | 2026 | `b739351daf9779bc` | `checkpoints/iteration_4000.pth` | `b7f0575f1e061936…` | BASELINE_FINAL_STATE_V1 | 4000 | `21d47c2` | `7677c3a` | yes |
| E06b | 42 | `aa96129289d080ba` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `d56122e0cf363a6c…` | OFFICIAL_GENERATOR_EPOCH_200 | 3200 | `1f546c9` | `3b0d308` | yes |
| E06b | 1337 | `9ffda3ea3f6cc767` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `645538e866a6d835…` | OFFICIAL_GENERATOR_EPOCH_200 | 3200 | `a8a7b5d` | `3b0d308` | yes |
| E06b | 2026 | `ec71778b28479de5` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `e8d0d01cfdb59b92…` | OFFICIAL_GENERATOR_EPOCH_200 | 3200 | `1729333` | `3b0d308` | yes |
| E06c | 42 | `27dcaf23aac18d7a` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `665684fddf11ec28…` | OFFICIAL_GENERATOR_EPOCH_200 | 7400 | `cf5ed6c` | `24f7049` | yes |
| E06c | 1337 | `49a179a4dca77f41` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `43ffa30dc7eaa568…` | OFFICIAL_GENERATOR_EPOCH_200 | 7400 | `9281c79` | `24f7049` | yes |
| E06c | 2026 | `86b631403e3c4ee1` | `checkpoints/netG_model_epoch_200_iter_0.pth` | `dc6c54b4357a5154…` | OFFICIAL_GENERATOR_EPOCH_200 | 7400 | `81a8ef6` | `24f7049` | yes |

Full hashes (E03: data, index and meta) are in the JSON. Every hash was recomputed on the host from the run root and equals the committed three-seed closure CSV.

## 2. E04 seed 2026 (read-only)

- run `e7100eb9a5ab0537`, launch commit `18ba00e94813f005876c8d6235b0b848226b2c1d`, GPU checkout `18ba00e94813f005876c8d6235b0b848226b2c1d`
- last metrics global_step 39770 of 150000; latest checkpoint files: iteration_015000.index, iteration_030000.index
- process: [' 435565   77063 Sl+  python -u -B tools/run_e04.py --seed 2026 --execution-config configs/execution/m5_gpu_3090.yaml']; exit marker present: False; status RUNNING_NOT_COMPLETE
- none: stat/read of metrics tail, checkpoint listing and process table only

## 3. Method bank contracts

### E03 STDN (Track A)

Fidelity: FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY. Environment: gpat-m6-e03 (Python 3.8, TF1; environments/e03_stdn.lock.json).
Population: 8838 (frozen pairs_train_v1 row; spec 6, 7; A1 N_syn).

Frozen:

- checkpoint: checkpoints/ckpt-50 of seed s (OFFICIAL_LATEST_FINAL_CKPT_50; A2-06 clause 1)
- bank seed s uses generator seed s (evaluator contract D12 SEED_MATCHED_RAW)
- synthesis graph: trace = s*x_sp + b + resize(C) + T from Gen(x_sp); synth = x_li + warping(trace, offset(lm_sp, lm_li)) (train.py:44-50, dataset.py:142); spec 8.3 "spoof trace from source_spoof_id, geometric correction to target_live_id"
- inputs: canonical RGB256 uint8 / 255 (dataset.py:144, production runner); landmarks_px256 via e03_feed_pack_v1, which covers every pairs_train_v1 row
- output: one canonical 256x256 RGB uint8 synthetic spoof per pair (spec 7, 8.3)

Open (owner decisions):

- E03-Q1 Gen inference mode (BN moving statistics vs batch statistics)
- E03-Q2 weight source (raw variables vs weight-EMA shadow)
- E03-Q3 inference-time augmentation (training brightness/flip off vs on)
- E03-Q5 auxiliary outputs
- Q4 float -> uint8 conversion
- C-Q1 encoding
- C-Q2 device/reproducibility
- C-Q3 S7 acceptance
- C-Q5 GPU timing
- storage location (storage DRAFT)

Auxiliary outputs: none by spec (output only the synthetic spoof direction); trace maps would be optional diagnostics (E03-Q5). Stochastic at inference: False.

Status: `E03_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY`

### E05 PCGAN (Track A)

Fidelity: owner-approved controlled implementation resolutions (not native/faithful/official). Environment: gpat-m6-e05 (Python 3.11, torch; environments lock in E05 closure).
Population: 8838 (frozen pairs_train_v1 row; spec 6, 7, 8.5; A1 N_syn).

Frozen:

- checkpoint: checkpoints/iteration_4000.pth of seed s (BASELINE_FINAL_STATE_V1; A2-06)
- bank seed s uses generator seed s (D12)
- generation path: mix = G(E(x_src).spatial, E(x_tgt).global), x_src = source_spoof (artifact), x_tgt = target_live (content) (spec 8.5; methods/pcgan/training_runner.py:63 explicit_pair)
- inputs: canonical RGB256 uint8 -> float32 (x/255-0.5)/0.5, no resize/crop/flip (E05 closure protocol.input_mapping)
- output: full 256x256 RGB synthetic image (spec 8.5)

Open (owner decisions):

- E05-Q1 StyleGAN2 noise at inference (fresh noise is drawn on every forward)
- Q4 float [-1,1] -> uint8 conversion
- C-Q1 encoding
- C-Q2 device/reproducibility
- C-Q3 S7 acceptance
- C-Q5 GPU timing
- storage location (storage DRAFT)

Auxiliary outputs: none (spec 8.5 evaluates only the Live->Spoof mix). Stochastic at inference: True.

Status: `E05_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY`

### E06b DSDG-NATIVE (Track B (secondary; SiW NOT_INSTANTIABLE_MISSING_SUBJECT_ID))

Fidelity: FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY. Environment: gpat-m6-e06c (Python 3.11, torch; environments lock 91416a20...).
Population: DEFERRED_TO_M8 (synthetic sample from noise (not a pair method); configs/methods/e06b_dsdg_native.yaml generation.n_syn_intended = DEFERRED_TO_M8).

Frozen:

- checkpoint: checkpoints/netG_model_epoch_200_iter_0.pth of seed s (OFFICIAL_GENERATOR_EPOCH_200; generated.py:23)
- bank seed s uses generator seed s (D12)
- generation graph: netG = Decoder_s(hdim=128); z = cat(noise_s, noise, noise); output = sigmoid RGB 256x256, channels 0:3 = spoof (generated.py:47-54; generator.py:150)
- DSDG does not consume the pair manifest; pairwise metrics N/A (spec 6, 8.6)

Open (owner decisions):

- E06b-Q0 N_syn (8838 pooled vs 3720 native coverage) - deferred to M8 by the frozen config
- DSDG-Q1 latent sampling stream and seed
- DSDG-Q2 which output half
- Q4 float -> uint8 conversion
- DSDG-Q4 bank identity schema for non-pair rows
- C-Q1 encoding (official JPEG vs PNG)
- C-Q2 device/reproducibility (official cudnn.benchmark=True)
- C-Q3 S7 acceptance
- C-Q5 GPU timing
- storage location (storage DRAFT)

Auxiliary outputs: none for the spoof bank; the paired live half is discarded unless DSDG-Q2 says otherwise. Stochastic at inference: True.

Status: `E06b_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY`

### E06c DSDG-BIN-IDFREE (Track A)

Fidelity: CONTROLLED_ADAPTATION (DEV-020). Environment: gpat-m6-e06c (Python 3.11, torch; environments lock 91416a20...).
Population: 8838 (synthetic sample from noise (not a pair method); configs/methods/e06c_dsdg_bin_idfree.yaml n_syn_intended 8838; A1).

Frozen:

- checkpoint: checkpoints/netG_model_epoch_200_iter_0.pth of seed s (OFFICIAL_GENERATOR_EPOCH_200; A2-06 clause 1)
- bank seed s uses generator seed s (D12)
- generation graph identical to E06b (same pinned FaceX-Zoo 16b793a7, same Decoder_s); DEV-020 changes only training (pair relation, lambda_pair = 0, binary spoof class), not generation
- separate scientific identity: E06c config 7176dd4c, launch authority 84809868, own checkpoints and bank

Open (owner decisions):

- DSDG-Q1 latent sampling stream and seed
- DSDG-Q2 which output half
- Q4 float -> uint8 conversion
- DSDG-Q4 bank identity schema for non-pair rows
- C-Q1 encoding
- C-Q2 device/reproducibility
- C-Q3 S7 acceptance
- C-Q5 GPU timing
- storage location (storage DRAFT)

Auxiliary outputs: none for the spoof bank. Stochastic at inference: True.

Status: `E06c_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY`

## 4. Storage

- root `/` /dev/nvme0n1p3 ext4: 34.7 GiB free; floor 24 GiB; E04 growth reserve 0.41 GiB; headroom above floor 10.3 GiB
- `/media/ext_data` /dev/sda2 ext4: 3214 GiB free; `root:shared 2775 directory`; outside the runtime_root containment contract
- estimate for 12 banks x 8838 outputs: typical 9.1 GiB (E01 mean 90231 B/PNG), worst case 19.7 GiB (incompressible PNG); per family worst 4.9 GiB
- all four families fit above the floor: typical True, worst False; one family (worst): True
- Full learned-bank production for all four families is not proven safe on the root SSD above the approved 24 GiB floor in the worst (incompressible) case. Either (a) run one family at a time on the root SSD with the per-bank 24 GiB floor gate re-checked before every bank, or (b) approve the additive storage-location amendment DRAFT (configs/amendments/learned_bank_storage_location.DRAFT.yaml) for a dedicated owner-only bank root on /media/ext_data. Nothing is written to the HDD until approved.

## 5. Source citations

| Id | File:line | Pin match | OK |
|---|---|---|---|
| STDN.gen_call_train | `third_party/source_cache/stdn/train.py:44` | True | True |
| STDN.recon | `third_party/source_cache/stdn/train.py:47` | True | True |
| STDN.trace | `third_party/source_cache/stdn/train.py:48` | True | True |
| STDN.trace_warp | `third_party/source_cache/stdn/train.py:49` | True | True |
| STDN.synth | `third_party/source_cache/stdn/train.py:50` | True | True |
| STDN.offset_direction | `third_party/source_cache/stdn/model/dataset.py:142` | True | True |
| STDN.input_scale | `third_party/source_cache/stdn/model/dataset.py:144` | True | True |
| STDN.test_inference_mode | `third_party/source_cache/stdn/test.py:55` | True | True |
| STDN.test_batch1 | `third_party/source_cache/stdn/test.py:49` | True | True |
| STDN.test_raw_saver | `third_party/source_cache/stdn/test.py:58` | True | True |
| STDN.weight_ema_maintained | `third_party/source_cache/stdn/model/model.py:122` | True | True |
| STDN.figure_clip | `third_party/source_cache/stdn/model/utils.py:24` | None | True |
| STDN.figure_truncate | `third_party/source_cache/stdn/model/utils.py:37` | None | True |
| PCGAN.basis_fresh_noise | `third_party/source_cache/swapping_autoencoder/models/networks/stylegan2_layers.py:342` | None | True |
| PCGAN.basis_tensor2im_scale | `third_party/source_cache/swapping_autoencoder/util/util.py:125` | None | True |
| PCGAN.basis_tensor2im_clip | `third_party/source_cache/swapping_autoencoder/util/util.py:128` | None | True |
| PCGAN.basis_tensor2im_cast | `third_party/source_cache/swapping_autoencoder/util/util.py:131` | None | True |
| DSDG.final_checkpoint | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:23` | None | True |
| DSDG.cudnn_benchmark | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:36` | None | True |
| DSDG.eval | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:43` | None | True |
| DSDG.batches | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:46` | None | True |
| DSDG.noise_identity_first | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:47` | None | True |
| DSDG.noise_spoof_second | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:48` | None | True |
| DSDG.latent_layout | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:49` | None | True |
| DSDG.spoof_half | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:54` | None | True |
| DSDG.truncate_no_clip | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:63` | None | True |
| DSDG.jpeg | `third_party/source_cache/facexzoo/addition_module/DSDG/generated.py:65` | None | True |
| DSDG.sigmoid_output | `third_party/source_cache/facexzoo/addition_module/DSDG/networks/generator.py:150` | True | True |
| DSDG.netG_is_decoder_s | `third_party/source_cache/facexzoo/addition_module/DSDG/networks/__init__.py:11` | True | True |

## 6. Production proposal (not authorized)

- method order ['E06c', 'E05', 'E03', 'E06b'] (Track-A methods first (E06c, E05, E03), smallest/fastest inference first; E06b (Track B, budget deferred) last); seed order [42, 1337, 2026]
- sources: one new learned-bank authority commit (adapters + writer + approved contract) for all four methods; checkpoints stay those of the training authorities d547fc9 (E03), efdf753 (E05), d9ecbef (E06b), 84809868 (E06c); the E04 checkout 18ba00e is never used or moved
- environments: {'E03': 'gpat-m6-e03', 'E05': 'gpat-m6-e05', 'E06b': 'gpat-m6-e06c', 'E06c': 'gpat-m6-e06c'}
- GPU memory: {'E03': 'TF1 graph, batch 1; must set allow_growth (TF1 otherwise reserves the whole GPU); inference footprint to be measured in S7', 'E05': '< training peak 3.08 GB (no D, no grads)', 'E06b': 'Decoder_s only, batch 100: expected ~2-4 GB (to be measured in S7)', 'E06c': 'as E06b'}
- runtime: minutes per bank on an unshared 3090 (8838 forwards); E03 adds CPU offset-map interpolation per pair (matplotlib triangulation)
- layout: {'clone': '~/workdir/m8_learned_banks/clones/<authority>', 'evidence': '~/workdir/m8_learned_banks/evidence/<M>_s<S>/', 'bank_root': 'per the storage ruling (L1)', 'tmux': 'm8bank_<method>_s<seed> (remain-on-exit, pipe-pane log)', 'exit_marker': '~/workdir/m8bank_<method>_s<seed>.exit'}
- preconditions: learned_bank_generation_contract.yaml OWNER_APPROVED; storage ruling approved; adapters qualified on synthetic fixtures (CPU) and S7 toy acceptance passed per method; E04 seed 2026 exited and audited (C-Q5 A) or concurrent use approved; checkpoint SHA256 re-verified at launch; free-space floor gate; clean isolated clone
- bank audit: per bank: row count, 0 or counted failures, file and pixel hashes, checkpoint hash, toy-prefix byte equality with the S7 acceptance outputs, TRAIN-only O_NOATIME reads with VAL/TEST atime unchanged, run-root listing digest; then a family closure per method

## 7. Gates

Failed: none

