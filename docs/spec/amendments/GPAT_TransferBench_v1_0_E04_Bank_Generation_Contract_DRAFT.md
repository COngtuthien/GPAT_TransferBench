# GPAT-TransferBench v1.0 — E04 Physics-STD bank-generation contract (DRAFT)

**Status:** `DRAFT_PENDING_OWNER_DECISION` · approved: `false` · 2026-11-25

Machine-readable source: `configs/amendments/e04_bank_generation_contract.DRAFT.yaml`. This page is rendered from it by `tools/e04_bank_readiness.py`; the YAML governs.

Decide how the three closed E04 generators (iteration_150000 of seeds 42, 1337, 2026) are turned into M8 synthetic banks. The frozen authorities define the E04 training graph completely but contain no bank-inference rule; this DRAFT lists every pixel-affecting point they leave open, with options, source evidence, expected effect and a recommendation. Nothing here is approved, implemented or executed.

**Approval:** The owner approves by committing configs/amendments/e04_bank_generation_contract.yaml (status OWNER_APPROVED, approved true) that names one option for every question in owner_questions and keeps the frozen_by_existing_authority list unchanged, in a commit of its own. This DRAFT stays byte-unchanged. Only after that may the E04 bank adapter be implemented and fixture-qualified; S7 and production each need their own later authority (see governance).

## Source situation

- **official phystd code:** NONE (config source.official_code_release NONE_LOCATED_AS_OF_2026_09_21; A3 section 1). No official PhySTD inference or synthesis-at-test procedure exists to pin.
- **paper:** arXiv:2012.05185 v1 / TPAMI 2022. Defines the spoof synthesis (v1 Eq.8, Eq.13-15, section 3.3.1) and the harder-sample synthesis (section 3.3, Algorithm 1) as training-time procedures. It is silent on test-time normalization statistics, on producing a synthetic bank, and on float->uint8 conversion.
- **official predecessor:** yaojieliu/ECCV20-STDN @ c79f1f8c615d2b8471b3df29da881bb18dd54c90. Frozen role (e04_training_graph_resolution.yaml stdn_reference_policy): OFFICIAL_PREDECESSOR_REFERENCE_ONLY, use_only_where_phystd_silent. Its test.py scores images only (no synthesis at test time); its train.py synthesizes only inside training.
- **frozen e04 graph:** methods/physics_std/training_graph.py (M6D3e, qualified, used for all three seeds) realizes the paper synthesis as the tensor synthesized_spoof = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w (training_graph.py:32-39, trace_transfer.py:42-43), where B~ = low32(B), C~ = low128(C) and _w is the bilinear CPU warp with dense_offsets(source=spoof Q140, target=live Q140). It is computed only as the D4 discriminator input during training.
- **not used as authority:** The E03, E05 and DSDG bank rulings of configs/amendments/learned_bank_generation_contract.yaml are not inherited. They are cited nowhere below as a reason for an E04 choice.

## Evidence (read-only CPU probe)

- **record:** outputs/audit/E04_BANK_CONTRACT_EVIDENCE.json (read-only CPU probe, clone 0728cb1b, TF 1.15.5)
- **frozen graph bn mode:** all 14 FusedBatchNormV3 ancestors of synthesized_spoof have is_training=True: the trained synthesis used batch statistics of the 8-row batch (4 target lives + 4 source spoofs), so each training-time synthesized spoof depended on the other 7 images of its batch
- **synthesized spoof inputs:** only placeholders synthetic_rgb and sampling_offsets; only generator variables; no moving statistic, no discriminator variable, no hard-sample placeholder
- **t a addn:** the T_A AddN that D12 found non-bitwise on GPU is NOT an ancestor of synthesized_spoof (no AddN ancestor at all)
- **moving statistics:** each final checkpoint holds 30 finite generator BN moving statistics (15 layers), min moving_variance 1.3e-4 .. 2.6e-4; the original-pass and hard-pass BN update ops assign the same 30 variables (momentum 0.99), so these statistics are an EMA over original (4 live + 4 spoof) and hard (4 live + 4 hard-synthesized) batches
- **weights:** no ExponentialMovingAverage shadow variable exists; generator trainables are the raw variables (72 trainable + 30 moving per seed); 288 optimizer slots and the discriminator are not needed for synthesis
- **cpu pinned ops:** 576 of 850 ancestor ops request /cpu:0 (resize, warp) exactly as qualified

## Owner questions

### E04-Q1_normalization_statistics (pixel-affecting)

Which batch-normalization statistics does the generator use when it decomposes a source spoof for a bank output?

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_MOVING_STATISTICS_PER_PAIR` | Build the frozen generator topology (network.generator with identical variable names and shapes) with batch normalization in inference mode (moving_mean / moving_variance of iteration_150000); forward the source spoof alone (batch 1); restore the 72 generator trainables and 30 generator moving statistics only. Every bank row depends only on its own pair. | OFFICIAL_PREDECESSOR_REFERENCE_ONLY where PhySTD is silent: ECCV20-STDN test.py:34 Gen(img, training_nn=training_nn), test.py:55 training_nn=False, test.py:49 BATCH_SIZE=1, test.py:63 saver.restore; standard TF1 inference semantics of tf.layers.batch_normalization(training=False). | Deterministic per-pair output, independent of batch neighbours. Normalization uses the EMA statistics accumulated during training, which mix original and hard batches; this differs from the batch statistics the trained synthesis saw (disclosure). The 186 S7 toy rows can equal their production rows byte for byte. |
| `B_TRAINING_BATCH_STATISTICS_FROZEN_BATCHES` | Run the frozen TrainingGraph forward and fetch synthesized_spoof with batch statistics over 8 rows (4 target lives + 4 source spoofs). | FROZEN_M6D3E_GRAPH (exactly the training-time computation); no paper or predecessor support for test-time batch statistics. | Reproduces the training-time synthesis computation, but every output depends on three unrelated pairs. Needs two further new rules: the batch-composition rule (for example ascending pair_id in consecutive groups of 4) and the remainder rule (8838 = 4 x 2209 + 2; training D4 forbids padding and duplication). S7 toy rows equal production rows only if each toy row is computed inside its production batch. |
| `C_TRAINING_BATCH_STATISTICS_SINGLE_PAIR` | Batch statistics over the pair's own two images (target live + source spoof). | No source precedent. | Statistics from 2 samples differ strongly from both training batches and moving averages. Listed for completeness only. |

**Recommended:** `A_MOVING_STATISTICS_PER_PAIR`. PhySTD is silent. The only pinned official code in the lineage evaluates with moving statistics at batch 1, which the frozen stdn_reference_policy allows where PhySTD is silent. A keeps each output a function of its own pair and checkpoint, needs no invented batching or remainder rule, and keeps S7-to-production byte equivalence possible. The training/inference statistics difference must be disclosed.

### E04-Q2_synthesis_variant (pixel-affecting)

Which synthesis produces the bank image?

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_PAPER_SPOOF_SYNTHESIS` | out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w with B~ = low32(B), C~ = low128(C), P = sigmoid(raw[...,9]), decomposition of the source spoof, warp with dense_offsets(source_xy = spoof Q140, target_xy = live Q140), exactly the frozen synthesized_spoof operations and operand order (training_graph.py:32-39). No random draw. | PAPER arXiv v1 Eq.8, Eq.13-15, section 3.3.1; M6D3d row 'spoof synthesis / trace transfer' CONTRACT_RESOLVED; M6D3e realization. | The method's own spoof synthesis; deterministic given the checkpoint and pair. |
| `B_HARDER_SAMPLE_SYNTHESIS` | Algorithm 1 harder samples: B~, C~, T scaled by uniform[0,1] weights and P removed with probability 0.5, with a new bank RNG domain. | PAPER section 3.3 / Algorithm 1 defines this as a training augmentation for L_S and L_H; no bank RNG domain exists. | Weaker, randomized traces; adds an RNG rule. Not the method's synthesized spoof. |

**Recommended:** `A_PAPER_SPOOF_SYNTHESIS`. The paper's synthesis is the generator's spoof output; harder samples are a training-only perturbation.

### E04-Q3_float_to_uint8 (pixel-affecting)

How does the float32 synthesis (D8 scale: input = uint8/255; additive traces can leave [0, 1]) become the canonical uint8 RGB image?

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_PREDECESSOR_CLIP_TRUNCATE` | u8 = truncate(clip(out, 0, 1) * 255) in float32 (the inverse of D8 with clip then cast). | OFFICIAL_PREDECESSOR_REFERENCE_ONLY where PhySTD is silent: ECCV20-STDN model/utils.py:24 tf.clip_by_value(fig, 0.0, 1.0), model/utils.py:37 tf.cast(column * 255, tf.uint8). | Saturates out-of-range values; truncation biases each channel by -0.5 grey level on average relative to rounding. |
| `B_CLIP_ROUND_HALF_EVEN` | u8 = numpy.rint(clip(out, 0, 1) * 255) cast to uint8. | BENCHMARK (no E04 or predecessor source). | Unbiased quantization; differs from A on about half of all pixels by 1 grey level. |
| `C_CLIP_ROUND_HALF_UP` | u8 = floor(clip(out, 0, 1) * 255 + 0.5). | BENCHMARK. | As B except exact .5 ties. |

**Recommended:** `A_PREDECESSOR_CLIP_TRUNCATE`. PhySTD has no writer. The frozen reference policy points to the official predecessor where PhySTD is silent, and its writer is clip then truncating cast. B is the alternative if the owner prefers unbiased rounding over predecessor behaviour. In every option a NaN/Inf output is a counted RAW-MATCHED failure (spec section 7), never cast or substituted.

### E04-Q4_auxiliary_outputs

Are auxiliary maps stored with the bank?

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_NONE` | The bank holds the synthetic RGB image only. | spec section 7 requires only the canonical RGB output | Same bank format as every other method. |
| `B_DIAGNOSTIC_SIDECAR` | Also store P_w and the warped additive trace per row as diagnostics, never read downstream. | BENCHMARK | Larger bank; diagnostics only. |

**Recommended:** `A_NONE`. No downstream consumer; keeps the bank format uniform.

### E04-Q5_device_and_reproducibility (pixel-affecting)

On which device does bank inference run, and what replay standard does S7 accept? D12 governs E04 training (procedural, non-bitwise) and does not set a bank-inference standard.

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_GPU_FP32_BITWISE_REPLAY` | Approved RTX 3090, environment gpat-m6-e03 under the E04 training lock with the D9 driver-version-only policy, PRECISION_ENV and the qualified session configuration (production_runner.session_config), resize and warp pinned to /cpu:0 as frozen. S7 runs each seed twice; acceptance requires bitwise-identical manifests and PNG bytes. A bitwise failure stops for an owner ruling; the requirement is never weakened and no graph or output is changed to force determinism. | BENCHMARK; evidence: synthesized_spoof excludes the only forward op D12 observed varying (T_A AddN). | Strongest standard; not guaranteed to pass (conv / transpose-conv / BN kernels are not proven bitwise here). |
| `B_GPU_PRE_REGISTERED_TOLERANCE` | As A, but accept a replay within a tolerance fixed before S7 (a number the owner would have to set). | BENCHMARK, D12-like. | Weaker; requires a new numeric threshold. |
| `C_CPU_FP32_BITWISE_REPLAY` | CPU-only inference in gpat-m6-e03 (CUDA_VISIBLE_DEVICES empty, single thread) with bitwise two-run acceptance. | BENCHMARK. | Different kernels from training (float differences vs GPU); slower. |

**Recommended:** `A_GPU_FP32_BITWISE_REPLAY`. Same device, environment and graph operations as training, and the probe shows the synthesis path avoids the known non-bitwise op. If A fails, the owner decides between B and C with the failure evidence in hand.

### E04-Q6_governance_inclusion

Does E04 join the existing learned-bank output, S7, production and storage governance additively?

| option | rule | provenance | expected effect |
|---|---|---|---|
| `A_ADDITIVE_E04_EXTENSION` | Additive records only, none of the existing ones edited: (1) the approved E04 contract adopts for E04 the common lossless PNG encoding (PIL compress_level 6, optimize False, no metadata), the create-only writer and reader, pixel and file SHA256 per row, RAW-MATCHED failure accounting; (2) a new E04 S7 launch authority (learned_bank_s7_launch_authority_E04.yaml) using the approved 186-pair toy manifest; (3) a new additive storage record (learned_bank_storage_location_E04.yaml) registering <bank_root>/E04/seed_<seed>/ on the same HDD; (4) a production authority learned_bank_production_E04.yaml (all three seeds) binding the passed E04 S7 record. | BENCHMARK governance; the existing learned-bank contract and storage authority cover only E03, E05, E06b, E06c. | E04 banks become comparable files under the same audits; nothing existing changes. |
| `B_SEPARATE_E04_PIPELINE` | A self-contained E04 bank writer, reader, storage root and audit. | BENCHMARK | Duplicated infrastructure; downstream readers need a second path. |

**Recommended:** `A_ADDITIVE_E04_EXTENSION`. Reuses audited infrastructure without editing any existing authority.

## Already fixed by existing authority

| item | rule | source |
|---|---|---|
| checkpoints | iteration_150000 of each seed (BASELINE_FINAL_STATE_V1, within seed); no other checkpoint, no VAL/TEST selection, no best seed | A2-06; configs/methods/e04_physics_std.yaml checkpoint; E04_THREE_SEED_TRAINING_CLOSURE.json |
| population and row identity | one output per pairs_train_v1 row (8838, ascending pair_id); identity pair_id, target_live_id, source_spoof_id, dataset; TRAIN only | spec sections 6-7; config data.pair_manifest a5e4fdae...; D2/D4 pair order |
| roles | source_spoof_id is decomposed (B, C, T, P, I_P); target_live_id receives the warped traces | PAPER Eq.8/13-15; M6D3e training_graph.py:30-39 |
| input encoding | canonical 256x256 RGB uint8 faces, SHA256-checked, float32(uint8)/255 once; no mean/std, no clip, no second scaling | D8 |
| augmentation | none (no flip, brightness or crop) | D5 data_pipeline_augmentation NONE |
| geometry | frozen CPU pre-pass Q140 xy from e04_feed_pack_v1 (manifest db1908d5...; pre-pass 274a4b0a...); offsets = dense_offsets(source_xy=spoof Q140, target_xy=live Q140) with pinned scipy 1.10.1; no refit, tracker or redetection | D1-D3, D2 offsets, implementation_defined_within_authority.dense_offsets |
| frequency input and decomposition | network.frequency_input and generator heads B, C, T, P, I_P unchanged | M6D3d / M6D3e |
| warp | trace_transfer.warp bilinear, border clamp, /cpu:0 | M6D3e |
| weights | raw checkpoint variables; no EMA shadows exist | E04_BANK_CONTRACT_EVIDENCE |
| output size | canonical 256x256 RGB uint8 | spec section 7 |
| RAW-MATCHED | retain every numerically valid output; only NaN/Inf, decode failure or missing output excluded; no quality filter | spec section 7 |
| fidelity | CONTROLLED_ADAPTATION; label 'Physics-STD (controlled geometry/depth reconstruction)'; forbidden wording native / official reproduction / faithful reproduction | config; A3 |
| supervision | binary only; attack_macro never read | spec 8.4; D2 forbidden_columns |
| data firewall | no VAL or TEST row, face or metadata | config data.splits; D6 |

## Needs implementation qualification after approval

- inference generator construction with variable names and shapes equal to the trained generator (102 restored variables per seed), finalized graph, no optimizer, gradient or assign op
- restore of generator variables only, full checkpoint SHA256 verified before and after
- CPU fixture equality of the inference synthesis subgraph with the frozen TrainingGraph synthesized_spoof operations when both use the same normalization statistics
- offset and warp equality with the training feed on synthetic Q140 fixtures
- the approved float->uint8 conversion, PNG writer and reader round trip, create-only roots, manifest and provenance
- TRAIN-only O_NOATIME face reader, VAL/TEST refusal, negative authorization gates
- real-device S7 on the GPU (not provable by CPU tests)

**Randomness:** Under the recommended options no random draw exists: the experiment seed only selects the checkpoint. Option E04-Q2 B would need a new RNG domain.

**Not authorized by this DRAFT:** adapter implementation, S7 GPU inference, production banks, creating <bank_root>/E04, retraining or any change to E04 training code, checkpoints or evidence, downstream training, TEST access, DINOv3, GPAT.
