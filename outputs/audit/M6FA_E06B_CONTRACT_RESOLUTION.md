# M6FA — E06b DSDG-NATIVE contract resolution (M6F-A)

| Field | Value |
|---|---|
| Milestone | M6FA, classification `M6FA_E06B_CONTRACT_RESOLUTION` (audit; no implementation, no training) |
| Authority | `8e9ccc371f6c53b75ec6b473d2306aefa9aa672b` (M6A9), branch `m6-baselines` |
| Contract candidate | `outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json` |
| Contract SHA256 | `fe287ebe1ec9c19e5f63a40f1498e41f5b7044014c2cc7eb6279a32604555058` |
| Verifier | `tools/m6fa_e06b_contract_preflight.py` (stdlib only; re-derives every value from authority bytes) |
| Frozen config | **not created**; target `configs/methods/e06b_dsdg_native.yaml` |

Every value below comes from one of these authorities:
- the frozen specification;
- the adopted amendments (A1, A9);
- the pinned official source (FaceX-Zoo `16b793a7`, `addition_module/DSDG`), whose files match their SHA256 pins;
- the frozen dataset population and labels;
- existing benchmark execution records.

No owner preference was needed. **Unresolved scientific fields: none.**

## 1. Dataset scope (derived)

E06b uses a native same-subject relation. Spec §8.6 says: "Training live/spoof identity pairing uses only TRAIN
identities that possess both live and spoof samples." A1 §3 gives Track B the coverage "CASIA + MSU and SiW
`NOT_INSTANTIABLE_MISSING_SUBJECT_ID`", and `fair_track_v1.yaml` records the same.

| Dataset | Spoof frames (print / replay) | Live frames | Subjects with both | Live frames per subject |
|---|---|---|---|---|
| CASIA-FASD | 2,520 (1,680 / 840) | 840 | 35 / 35 | 24 |
| MSU-MFSD | 1,200 (400 / 800) | 400 | 25 / 25 | 16 |
| **Total** | **3,720 (2,080 / 1,640)** | **1,240** | **60** | — |
| SiW-Mv2 | 0 used | 0 used | 0 (0 of 9,507 TRAIN rows carry a subject) | `NOT_INSTANTIABLE_MISSING_SUBJECT_ID` |

- No spoof frame lacks a same-subject live frame.
- Identity is never fabricated. DEV-013 is a common-pair rule and is never used as same-person evidence.
- Counts come from the frozen M3 TRAIN video table (× 8 frames per video) and the frozen M4 coverage. A
  TRAIN-filtered, allowlisted-column metadata read of `split_v1` confirmed them at frame level; that read opened no
  image and materialized no TEST row.

## 2. Spoof-type target (derived): K = 2

Spec §8.6 says: "Use attack_macro as the spoof-type target". Under `attack_map_v1`, every CASIA and MSU spoof token maps
to `print` or `replay`, and both classes appear for all 60 subjects. So the vocabulary is {print, replay} and K = 2.
That is a data-derived class count, not a hyperparameter.

- Class index: print = 0, replay = 1. The official code orders OULU print codes before replay codes, and the frozen
  taxonomy also orders print before replay.
- The official `attack_type = 4` is OULU-specific and is not kept. There are no unused logits, following the
  `artifact_probe_classes_v1` precedent.
- `loss_cls` is active and non-degenerate. `netCls` stays outside the optimizer, as in the official code.

## 3. Official values (pinned source)

| Value | E06b | Source |
|---|---|---|
| lambda_pair | **5**; `loss_pair` active | `train_generator.sh:17` `lambda_pair=5`, passed at `:29`. The argparse default 0.5 (`train_generator.py:41`) is not the value actually executed. |
| lambda_mmd / ip / type / ort | 50 / 1000 / 10 / 1 | `train_generator.sh:15,16,18,19` |
| epochs | 200 | `train_generator.sh:8` |
| effective batch | 240 | `train_generator.sh:7` |
| hdim | 128 | `train_generator.sh:10` |
| lr | 2e-4, Adam over netE_nir + netE_vis + netG | `train_generator.py:22` (the launch script passes no `--lr`); `:87-88` |
| warm-up | epoch < 2 scales every term except `loss_rec` by 0.01 | `train_generator.py:174-176` |
| checkpoint | `OFFICIAL_GENERATOR_EPOCH_200`; saved at epoch 1 and every 10 epochs | `generated.py` pre_model; `train_generator.py:214` |
| seeds | 42, 1337, 2026 | spec |
| identity net | LightCNN-29 v2, frozen, SHA256 `d0750746…` | M6A4 |

All seven official loss terms are kept, including `loss_pair` and an active `loss_cls`.

`train_generator.sh` is not listed per-file in `third_party/source_pins.json`. Its bytes (SHA256 `bdb08394…`, git blob
`7fcb3f86…`) were verified identical to the pinned commit's tree entry.

## 4. Native relation: SAME_SUBJECT_ONLINE_RANDOM

- The dataset is indexed by TRAIN spoof frames. The live partner is `random.choice` over the same subject's TRAIN
  live frames, drawn in `__getitem__` every time an item loads (`generation_dataset.py:48, 87, 93`). There is no
  stored pair list.
- `manifests/dsdg_identity_pairs_v1.parquet` is **not** created; historical tests assert it does not exist. The
  relation is built at runtime, in memory, from the TRAIN rows of `split_v1`.
- Spoof index order and per-subject live pool order are ascending `sample_id` (bytewise).
  - The official `make_train_list.py` depends on `os.listdir` and `random.sample` for OULU frame selection.
  - The benchmark's frames are fixed by the frozen dataset contract, and M6B forbids behaviour that depends on
    `os.listdir` order.
  - Lexical order follows the A2-01 precedent. This is a deterministic clarification with no scientific effect.
- `subject_id_global` is used only for sampling (Track B allows native metadata).

**Randomness contract.** E06b reuses the E06c production loader:
- batch 240;
- `shuffle` with `torch.Generator().manual_seed(seed)`;
- **num_workers = 8** (`train_generator.sh:6`);
- `worker_init_fn = seed_torch_worker`;
- `persistent_workers=False` and `drop_last=False`.

Chunking happens on the GPU after loading. The worker count **affects results**: each worker serves every 8th global
batch from its own Python `random` stream. So for E06b the worker count is frozen at 8; E06c's
"implementation adaptable" worker policy does not carry over.

## 5. Execution compatibility

Physical batch 240 ran out of memory on the RTX 3090 (E06c, M6D5b). The M6D5c
`GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1` path (microbatch 20) and the M6D5d tail handling can be reused
unchanged, with effective batch 240:
- The networks have no BatchNorm.
- `loss_cls` (CrossEntropy mean) and `loss_pair` (MSE mean) are per-sample means, already weighted by m/B in
  `chunk_objective`.
- MMD and ORT use the global pass-1 statistics.

Epoch plan: **15 × 240 + 120** = 16 optimizer steps per epoch and 3,200 in total. The tail of 120 is **6 × 20**.

This is classified as `EXECUTION_RUNTIME_COMPATIBILITY`, not a scientific change. Bitwise equivalence to a physical
240 step is not claimed. For E06c the same path was recorded as `CONTROLLED_EXECUTION_ADAPTATION` inside an already
adapted row. Its effect on E06b's final fidelity disclosure is assessed after qualification.

Qualification still required:
- a reference check with `lambda_pair = 5` and K = 2;
- the B = 120 tail;
- memory at microbatch 20.

## 6. E06b compared with E06c

| | E06c | E06b |
|---|---|---|
| Spoof-type supervision | K = 1 (`loss_cls` always 0) | attack_macro K = 2 (active) |
| Pairing | common fair pair (different subjects) | official same-subject online random |
| lambda_pair | 0 | 5 |
| Datasets / rows | CASIA+MSU+SiW / 8,838 | CASIA+MSU / 3,720 |
| Epoch plan | 36 × 240 + 198 | 15 × 240 + 120 |
| Identity metadata | forbidden | sampling only |
| Workers | adaptable | frozen at 8 |
| Fidelity | CONTROLLED_ADAPTATION (DEV-020) | target FAITHFUL_OFFICIAL, not yet assessed |

Unchanged between them: architecture, hdim, optimizer and lr, 200 epochs, effective batch 240, the other lambdas,
warm-up, LightCNN, checkpoint rule, seeds and microbatch execution.

## 7. Deferred questions (not M6)

- **M8, Track-B generation budget.** Should E06b's N_syn be the pooled 8,838 (spec §7) or its native 3,720? This is
  genuinely ambiguous: `pairs_v1` leaves the M6 SiW strategy unsettled, and M4 raised it as Q-28. It does not block
  M6, because the M6 gate is adapters plus `method_status.csv`.
- **M12, T09 interpretation.** E06c and E06b differ in coverage by their own contracts. Neither method is altered to
  suit T09.

## 8. State

- Configs: no frozen config, native manifest or `method_status.csv` was created.
- Runtime: no training, GPU, image read or TEST access.
- **M6_CLOSED = false. M7 HAS NOT STARTED.**
- Next step, M6F-B: freeze `configs/methods/e06b_dsdg_native.yaml` from this contract, then implement the adapter
  statically.
