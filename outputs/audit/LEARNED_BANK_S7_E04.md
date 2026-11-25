# M8 learned-bank S7 toy acceptance: E04 Physics-STD

**Status:** `E04_S7_ALL_THREE_SEEDS_QUALIFIED`, 180/180 gates passing; `s7_acceptance_passed` = true.

Label: S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence). Fidelity: CONTROLLED_ADAPTATION (not an official Physics-STD inference reproduction).

Authority `configs/amendments/learned_bank_s7_launch_authority_E04.yaml` (`5043a7bfc69de4892f129b597ca21e895f68f25d0350887598f17636bbe66777`), contract `45240cd25e840c16a11256639cab52c2f67c73d2a5a853284d7d9aa96b6df13f`, family closure `a14d98fdf5a9a0e279ef09ca231d01261a349b9e7b86fc86d7d69c9b8734c55d`, input-allowlist ruling `c74217624cfa10421532946b36c0dbdfc3d8236aa01be3f8da66476cd6916713`, execution commit `d5d0cb3ea5d2c15aba4d0d2ddb84cade72ef279d`.

| seed | run A | run B | A/B replay | audit | final |
|---|---|---|---|---|---|
| 42 | PASS | PASS | identical | 55 gates, 0 failed | QUALIFIED |
| 1337 | PASS | PASS | identical | 55 gates, 0 failed | QUALIFIED |
| 2026 | PASS | PASS | identical | 55 gates, 0 failed | QUALIFIED |

Failed gates: none.

## Execution

- Environment `gpat-m6-e03` (`/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python`, Python 3.8.20, TensorFlow 1.15.5), GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04 (OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY (E04 D9, conditional narrow scope)).
- One OS process per seed and run, strictly sequential: seed 42 run A 2026-11-25T13:43:13Z → 2026-11-25T13:43:28Z exit 0; seed 42 run B 2026-11-25T13:43:44Z → 2026-11-25T13:43:58Z exit 0; seed 1337 run A 2026-11-25T13:44:39Z → 2026-11-25T13:44:54Z exit 0; seed 1337 run B 2026-11-25T13:45:02Z → 2026-11-25T13:45:16Z exit 0; seed 2026 run A 2026-11-25T13:45:55Z → 2026-11-25T13:46:09Z exit 0; seed 2026 run B 2026-11-25T13:46:17Z → 2026-11-25T13:46:31Z exit 0.
- Population: 186 toy TRAIN pairs (`50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690`) per seed and run; 1116 images in total.
- BN: tf.layers.batch_normalization(training=False): restored moving statistics, batch 1 (E04-Q1).
- Synthesis: out = (1-P_w)*(I_live + B~_w + C~_w + T_w) + P_w*I_source_w (E04-Q2); conversion `(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8) float32 [E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE; reference stdn model/utils.py:24,37]`; lossless PNG, PIL compress_level 6, optimize False, no metadata chunks (IHDR/IDAT/IEND only).
- Output root (GPU host): `/home/student20261/workdir/m8_learned_banks/s7/d5d0cb3ea5d2c15aba4d0d2ddb84cade72ef279d/E04`.
- Manifests (run A = run B when replay passed): seed 42 `3d92249683f4725749341bf0bb665b8c68cb2b1144cfbb9c0c89280e991b7674`; seed 1337 `aefd56e2e27500d0a925ece582541f65d4c7c1208c573861cdcc856b353c9336`; seed 2026 `ee5962301af9dc850999a3c70c470c66a7529365a28d29fc0379ff89f98eab24`.
- GPU memory (nvidia-smi, new PIDs): {"1337": {"A": {"peak_used_mib_new_pids": {"932732": 582}, "samples": 23}, "B": {"peak_used_mib_new_pids": {"935010": 582}, "samples": 23}}, "2026": {"A": {"peak_used_mib_new_pids": {"940293": 582}, "samples": 20}, "B": {"peak_used_mib_new_pids": {"942615": 582}, "samples": 20}}, "42": {"A": {"peak_used_mib_new_pids": {"924198": 582}, "samples": 23}, "B": {"peak_used_mib_new_pids": {"927236": 582}, "samples": 20}}}.

## Disclosures

- CONTROLLED_ADAPTATION: bank inference uses the stored BN moving statistics (batch 1); training-time synthesis used 8-image batch statistics; not official Physics-STD test-time behaviour
- clip-truncate float->uint8 follows the ECCV20-STDN predecessor (reference only)
- TRAIN allowlist from the pinned e04_feed_pack_v1 in the inference process (owner ruling e04_s7_input_allowlist_ruling.yaml); split_v1 / M2 cross-check performed by this auditor
- S7 outputs live only on the GPU host S7 root; git holds hashes, not image or checkpoint bytes
- toy population (186 pairs per seed); production equivalence is checked by a later production audit

## Scope

S7 establishes E04 readiness for a separately authorized production launch only. No production bank, storage directory, retraining, downstream training, TEST evaluation, DINOv3 or GPAT work was performed.
