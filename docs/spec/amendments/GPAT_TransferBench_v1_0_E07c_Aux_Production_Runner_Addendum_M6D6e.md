# GPAT-TransferBench v1.0 — E07c Auxiliary-Encoder Production Runner Addendum (M6D6e)

**Status:** PRODUCTION_RUNNER_CONTRACT_ADDITIVE · **Classification:** implementation contract, not a scientific amendment
**Method:** E07c DiffFAS-BIN-IDFREE (controlled encoder reconstruction) · `CONTROLLED_ADAPTATION` · `DEV-021` (unchanged)
**Authority commit:** `9bfb8dc0fc1e1a4be7ea7f1d4c788b6d2dbbd00f` (M6D6d)
**Machine-readable companion:** `configs/amendments/e07c_m6d6e_aux_production_runner_contract.yaml`

This addendum is **not** an A8 amendment. No owner decision was made. It records how the frozen
A3 §5.4 auxiliary-encoder training contract is executed by one production engine, and what the
bounded M6D6e qualification established. It changes no frozen config, amendment, snapshot, seed,
budget, source pin or environment.

## 1. Source semantics executed (pinned `models/pretrain_classifier.py`, SHA256 `3444691f…`)

| Source line | Statement | Production engine (`methods/difffas/aux_runner.py`) |
|---|---|---|
| 15–17 | `resnet18 = resnet18(); resnet18.fc = nn.Linear(512,17); .cuda()` | `encoder.encoder_model()` → `custom_rn.resnet18(pretrained=False)`, `fc = Linear(512, 7)` (A3), `.cuda()` |
| 18–23 | `Resize((256,256)) → ToTensor() → Normalize([0.5]*3, [0.5]*3)` | `build_transform` — verbatim |
| 25 | `ImageFolder(PADISI-USC)` | `AuxTrainDataset` over the frozen TRAIN K7 population (A3) |
| 26 | `DataLoader(batch_size=256, shuffle=True, num_workers=6, drop_last=True)` | `build_loader` — verbatim; no generator, worker_init_fn, pin_memory or sampler override |
| 27 | `SGD(lr=0.002, momentum=0.9, weight_decay=5e-3)` | `build_optimizer` — verbatim; no scheduler |
| 28 | `CrossEntropyLoss()` | `build_criterion` — verbatim |
| 36–41 | `zero_grad; _,_,_,outputs = model(x); CE; backward; step` | `Trainer.step` — same order, fourth output |
| 42–43 | `running_loss += loss.item()*inputs.size(0); rus = running_loss/len(train_dataset)` | `Trainer.run_epoch` — denominator **14467** (not 14336) |
| 45 | `torch.save(resnet18, './PADISI.pkl')` after every epoch | `Trainer.checkpoint_event` → M6D6c `save_whole_module` (whole `nn.Module`), same path overwritten, written to `<path>.partial` then `os.replace` |

A3 changes only: the K = 7 label space, the TRAIN data source, auxiliary seed 42 and the single
run. A7 adds FP32 / TF32 off / cuDNN deterministic without benchmark / no autocast / no GradScaler.
The source-native disconnected `norm.weight` / `norm.bias` (BatchNorm1d(18), never called) keep
`grad = None` and are skipped by SGD, exactly as in M6D6b; they are not "fixed".

## 2. TRAIN population

* Source: `manifests/split_v1.parquet` (`fb9aeb36…`) + `manifests/artifact_probe_classes_v1.json`
  (`c7d23e3e…`), both SHA256-checked before parsing. **Not** `manifests/difffas_bin_idfree_train_v1.parquet`
  (the 8838-row main DiffFAS spoof relation), which the auxiliary runner never opens.
* Scan: `pyarrow.parquet.read_table(columns=[sample_id, dataset, attack_macro, split, m2_status],
  filters=[split == TRAIN])`. Subject, video, raw-attack and hash columns are never read.
* Physical disclosure: the manifest holds TRAIN, VAL and TEST rows in **one** row group. pyarrow reads
  the footer and decodes the allowlisted column chunks natively, then filters in C++. Only TRAIN rows
  reach Python; no VAL/TEST row becomes a Python object or a Dataset item. That is a physical file
  read, not VAL/TEST use.
* Result: **14467** TRAIN rows — live 5629, makeup 759, mask_2d 96, mask_3d 1056, partial 1911,
  print 2838, replay 2178; `other_spoof` 0. All `m2_status = COMPLETE`.
* Order: ImageFolder-equivalent `(class index, sample_id)`. ImageFolder indexes sorted class folders,
  then sorted file names, and the frozen K7 order is alphabetical. This is an implementation
  derivation, not an owner decision.
* Canonical face: `<faces_256_root>/<dataset>/<sample_id>.png` (the M2 writer / M5 probe / M6D5e reader
  convention), RGB PNG 256×256 decoded by PIL exactly as ImageFolder's `pil_loader`. A face is
  accepted only if it is already mode RGB and 256×256, so there is no implicit conversion.
  Symlinks, path escape and non-population sample ids are refused, and nothing is enumerated.
  `Resize((256,256))` is an identity copy on the canonical face; evidence compares the bytes.

## 3. Seeds and modes

| Mode | Seed | Entry | Root |
|---|---|---|---|
| SCIENTIFIC | 42 (`auxiliary_encoder_training_seed`, A3 §5.4b) via `seed_adapter.apply_seed(42, cuda=True, auxiliary=True)` | `tools/run_e07c_aux.py` (**not launched in M6D6e**) | `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/` |
| QUALIFICATION | 60605 (engineering; the same framework calls, the scientific API is not called) | `methods/difffas/aux_runner_qualification.py` | `<runtime_root>/qualification/m6d6e/E07c_aux/q60605-<run_id>/` |

The seed is applied before model construction. The source passes no DataLoader generator, so
`RandomSampler` draws its seed from the process-seeded CPU generator, as the source would. The run
id uses the run_logging_v1 formula with the label `E07c/aux_encoder`, so it never collides with the
future main E07c seed-42 run. `experiment_seed` is null with a reason, because the auxiliary run is
not an experiment-seed run.

## 4. Scientific CLI (hard-frozen; not launched)

`tools/run_e07c_aux.py` accepts only `--seed 42`, `--execution-config` and `--preflight-only`. Before
anything executes, it refuses any epoch, batch, worker, drop_last, LR, momentum, weight-decay,
scheduler, AMP/FP16/BF16/TF32, microbatch, accumulation, activation-checkpointing, label-map,
alternate-manifest/data-root, VAL/TEST, checkpoint-selection, resume or qualification-seed flag. It
also refuses:

* a dirty worktree or wrong branch;
* a PYTHONHASHSEED other than 42;
* a launch environment other than the lock's (`NVIDIA_TF32_OVERRIDE=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`);
* an interpreter other than the locked `gpat-m6-e07c` python;
* a source pin that differs from the contract or lock;
* an A3/A6/A7 or contract-bound input that differs by SHA256;
* an existing `seed_42` root.

Torch is not imported before every gate has passed.

## 5. Checkpoint semantics

* Format: `torch.save(model, path)` of the whole `nn.Module` (M6D6c seam). Not a state_dict,
  safetensors, TorchScript or ONNX file.
* SCIENTIFIC: `checkpoints/encoder_final.pkl` is overwritten after every epoch. It is **not**
  authoritative and **not** consumable before epoch 200 completes. At epoch 200 the final bytes are
  closed and hashed, and the SHA256 is recorded in `run_summary.json` as
  `SHA256_RECORDED_PENDING_OWNER_FREEZE`. Main DiffFAS may consume the file only after a later
  milestone freezes that SHA256, and only through `aux_checkpoint.load_frozen_aux_encoder`
  (SHA256 before deserialization).
* No selection metric, no VAL selection, no TEST.
* Resume: `AUX_RESUME_NOT_QUALIFIED`. The pinned script has none, and a non-empty root is refused.

## 6. B = 256 OOM policy

CUDA OOM at any stage of the exact B = 256 step would have produced `BLOCKED_BY_AUX_B256_MEMORY`,
with no smaller batch, microbatch, accumulation, AMP, TF32, activation checkpointing, CPU offload,
model change, skipped backward or worker change. **It did not occur** (§7).

## 7. M6D6e qualification result (QUALIFICATION_ONLY)

* **B256 feasibility** (2 fresh processes, one production step each, real TRAIN, A7 FP32).
  Peak allocated 16,400.6 MiB and peak reserved 19,294 MiB on a 24,123 MiB RTX 3090, so exact B = 256
  fits. Loss `0x1.27fb6p+1`. 110 parameter tensors were updated; `norm.weight`/`norm.bias` were untouched.
* **Repeatability:** process 2 is bitwise equal to process 1. That covers the batch, input tensor,
  loss, all 110 gradients, all 112 post-step parameters, momentum and RNG, with max |Δ| = 0 and no
  tolerance.
* **One real-TRAIN epoch** through the same engine:
  * 56 optimizer steps × 256; 14336 unique TRAIN samples consumed, 131 dropped; no tail batch.
  * Source-native epoch loss 20556.797 / **14467** = 1.4209439913190094.
  * The first step equals the B256 step bitwise.
  * The epoch-boundary whole-module checkpoint was 185,136,819 B, SHA256 `ad124c52…`. It was reloaded
    through the SHA-first secure loader, found equal to the in-memory model, then removed.
  * An earlier fully executed but superseded attempt, in a separate process, produced the
    bitwise-identical epoch: the same order hash, loss and checkpoint SHA256.
* **Access:**
  * TRAIN rows exposed: 14467. VAL/TEST rows exposed: 0.
  * TRAIN face reads equal exactly the 14336 consumed samples in the epoch process. Each B256
    process read 3329 faces: 13 prefetched batches plus 1 transform check.
  * VAL/TEST/raw image reads: 0. Unauthorized manifest reads: 0. Firewall denials in the
    authoritative processes: 0.

Qualification loss is not benchmark performance, and nothing was retuned from it.

## 8. What remains NOT qualified

`AUXILIARY_ENCODER_200_EPOCH_TRAINING`, `AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT`,
`AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`, `AUX_RESUME`, `MAIN_DIFFFAS_TRAINING_GRAPH`,
`MAIN_CHECKPOINT_RESUME`, `MAIN_RUNNER_ENCODER_LOAD_INTEGRATION`, `MAIN_PRODUCTION_RUNNER`,
`SCIENTIFIC_TRAINING`, `M8_BANK`. The auxiliary encoder is **not** scientifically trained. Method
status stays `IMPLEMENTED_NOT_EXECUTED` / `CONTROLLED_ADAPTATION`.
