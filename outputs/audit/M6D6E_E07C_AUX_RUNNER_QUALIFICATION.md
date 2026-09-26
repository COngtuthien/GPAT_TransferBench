# M6D6e — E07c auxiliary-encoder production runner · real TRAIN path · exact B=256 bounded qualification

**Final status: PASS (candidate, uncommitted).** Method status is still `IMPLEMENTED_NOT_EXECUTED`, with
fidelity `CONTROLLED_ADAPTATION` and deviation `DEV-021`. Seed 60605 was used for qualification only.
This was a **ONE_REAL_TRAIN_EPOCH_QUALIFICATION, not 200_EPOCH_SCIENTIFIC_TRAINING.** The auxiliary
encoder has **not** been scientifically trained.

This is the first E07c milestone with **authorized real TRAIN access**: TRAIN metadata and TRAIN
canonical-face bytes. VAL and TEST were not used.

## 1. Authority

* Laptop: `m6-baselines` @ `9bfb8dc0fc1e1a4be7ea7f1d4c788b6d2dbbd00f` = `origin/m6-baselines`, clean.
* GPU: `e70c221` (M6D6d parent). It was verified clean and an ancestor, then fast-forwarded to `9bfb8dc`
  and left clean. No reset, clean, rebase or forced checkout.
* Environment `gpat-m6-e07c` = lock `0c909de1…` (conda `2c5a2bbc…`, pip `42cebd80…`, python `83c01e86…`);
  protected environments unchanged. Pinned DiffFAS `23f40519…` / tree `d190a5fb…` is clean.

## 2. Source semantics re-derived (pinned `models/pretrain_classifier.py` `3444691f…`)

The source executes these steps in order:

1. `resnet18(); fc = Linear(512,17); .cuda()`.
2. `Resize((256,256)) → ToTensor → Normalize(0.5,0.5)`.
3. `DataLoader(batch_size=256, shuffle=True, num_workers=6, drop_last=True)`.
4. `SGD(lr=0.002, momentum=0.9, weight_decay=5e-3)` and `CrossEntropyLoss()`.
5. 200 epochs of: `zero_grad` → fourth output → CE → `backward` → `step`.
6. `running_loss += loss.item()*inputs.size(0)`; `rus = running_loss / len(train_dataset)`.
7. `torch.save(resnet18, './PADISI.pkl')` after every epoch.

A3 changes only the K = 7 head, the TRAIN source, seed 42 and the single run. With drop_last,
`len(dataset) = 14467`, and 56 × 256 = 14336 samples are consumed. **The source denominator 14467 is
preserved**; 14336 and 131 appear only as audit metadata.

## 3. TRAIN relation

* `manifests/split_v1.parquet` (`fb9aeb36…`) and `manifests/artifact_probe_classes_v1.json` (`c7d23e3e…`)
  are SHA256-checked before parsing.
* The schema has 14 columns and 20615 rows in **one** row group.
* Scan: `read_table(columns=[sample_id, dataset, attack_macro, split, m2_status], filters=[split==TRAIN])`.
  pyarrow decodes those column chunks of the single all-split row group natively and filters them in C++.
* **TRAIN rows materialized = 14467; VAL = 0; TEST = 0.** No VAL/TEST row became a Python object or a
  Dataset item. Reading the Parquet footer and column chunks is a physical file access, not VAL/TEST use.
* Class counts: live 5629, makeup 759, mask_2d 96, mask_3d 1056, partial 1911, print 2838, replay 2178.
  `other_spoof` = 0.
* Rows by dataset: casia 3360, msu 1600, siwmv2 9507.
* Order is ImageFolder-equivalent `(class index, sample_id)`, since the K7 order is alphabetical.
  `population_order_sha256 = c890ed8d…`.
* `manifests/difffas_bin_idfree_train_v1.parquet`, the 8838-row main spoof relation, was **never opened**.

## 4. Canonical faces, transform, loader, seed, precision

* Path: `<faces_256_root>/<dataset>/<sample_id>.png`, the convention of the M2 writer
  (`encode_png_rgb`), the M5 probe and the M6D5e reader. The root is resolved from
  `configs/execution/m5_gpu_3090.yaml` through `m2b.resolve_roots`.
* PIL decode (ImageFolder `pil_loader`). A face must already be mode RGB and 256×256.
  Non-population ids, symlinks and escapes are refused, and nothing is enumerated.
* Transform: `Resize((256,256)) → ToTensor → Normalize([0.5]*3,[0.5]*3)`, verbatim. On a canonical face,
  Resize is a byte-identity copy. Loader row 0 equals the transform output and the manual formula
  bitwise. Input is float32 `[256,3,256,256]` in [−1, 1], RGB.
* No augmentation, no class balancing, no weighted or replacement sampling, and no ArtifactProbeNet
  preprocessing.
* Loader: `DataLoader(batch_size=256, shuffle=True, num_workers=6, drop_last=True)`. No generator,
  worker_init_fn or pin_memory (all source defaults; `prefetch_factor` 2; `persistent_workers` False).
  56 batches.
* Seed: qualification 60605 goes through `aux_runner.seed_process`, which makes the same framework
  calls as `seed_adapter.apply_seed`; the scientific API is **not** called with 60605. The seed is
  applied before model construction, and `PYTHONHASHSEED=60605`. Scientific seed 42 was not used.
* A7: FP32, no autocast, no GradScaler, matmul/cuDNN TF32 off, `cudnn.benchmark=False`,
  `cudnn.deterministic=True`, `use_deterministic_algorithms` not set. Launch environment:
  `NVIDIA_TF32_OVERRIDE=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## 5. Production engine and scientific CLI

* `methods/difffas/aux_runner_io.py`: torch-free contract, population, reader, records and CLI refusals.
* `methods/difffas/aux_runner.py`: the one engine (components, step, epoch, whole-module save, run context).
* `methods/difffas/aux_runner_qualification.py`: the qualification harness with the audit-hook firewall.
* `tools/run_e07c_aux.py`: the scientific CLI. It accepts only seed 42 and refuses every training
  override, VAL/TEST, resume, a dirty worktree, a wrong branch, wrong PYTHONHASHSEED or launch
  environment, a wrong interpreter, source, A3/A6/A7 or contract input, and an existing seed_42 root.
  It was **not launched**.
* The scientific and qualification modes share the dataset, transform, loader, K7 map, model, SGD,
  CE, step, epoch-loss and checkpoint writer. Only the seed (42 / 60605), the epoch bound (200 / 1)
  and the root differ.

## 6. Exact B=256 feasibility (2 fresh processes)

| Stage | allocated MiB | peak allocated MiB | reserved MiB |
|---|---|---|---|
| after model construction | 176.5 | 176.5 | 204 |
| after B256 host→device | 368.5 | 368.5 | 396 |
| after forward | 14257.1 | 14257.1 | 14348 |
| after backward | 642.9 | 16400.6 | 19288 |
| after optimizer.step | 820.7 | 16400.6 | 19294 |

Overall peak allocated was **16,400.6 MiB** and peak reserved **19,294 MiB**, on an RTX 3090 with
24,123 MiB total (23,608 MiB free at start). **Exact B = 256 FP32 fits; there was no OOM and no fallback
of any kind.**

* First batch: `sample_ids_sha256 37b59ee5…`; class histogram [113, 12, 3, 25, 22, 50, 31].
* Loss `2.312358856201172` (`0x1.27fb6p+1`), finite.
* Gradients present on 110 tensors, all finite; `norm.weight`/`norm.bias` have `grad=None`.
* 110 parameter tensors were updated; `norm.*` was unchanged; 108 BatchNorm buffers were updated.

## 7. First-step repeatability

Process 2 equals process 1 **bitwise**, with no tolerance, on every compared field: RNG after seeding;
initial state; RNG before the iterator; first-batch indices, ids, histogram, input tensor and target;
loss; gradient and parameter aggregates; momentum; buffers; RNG after the step. The tensor comparison
found 112/112 parameters and 110/110 gradients bitwise equal, max |Δ| = 0. The process-1 probe
was removed.

## 8. One complete real-TRAIN qualification epoch

* **56** optimizer steps, **56** backward calls and **56** `zero_grad` calls; every batch was 256 and
  no tail batch was produced.
* **14336** unique TRAIN samples consumed and **131** dropped (dropped histogram
  [57, 8, 0, 10, 14, 24, 18]); no duplicate and no VAL/TEST sample. All losses were finite, ranging
  1.1456 to 2.3124, and every update was finite.
* Ordered sample sequence `epoch_sample_order_sha256 = 76e887fe8f448da0af94c3c5331ea3c9033fed407fa07219f3da698b0feb6127`.
* Source-native epoch loss: running 20556.79672241211 / **14467** = **1.4209439913190094**. The
  consumed count 14336 is recorded but **not** used as the denominator. Nothing was retuned from this
  qualification loss, which is not benchmark performance.
* The first step equals the B256 step bitwise (batch, loss, post-step parameters).
* Peak per-step allocated memory was 16,578.5 MiB. The epoch took 74.5 s.
* run_logging_v1 root `…/qualification/m6d6e/E07c_aux/q60605-8473ee804f8bc698` holds resolved_config,
  run_manifest, metrics.jsonl (56 step records + 3 events), checkpoint_index, run_summary, stdout.log
  and stderr.log. VAL fields are null with a `NOT_USED` reason, and no TEST field exists.

## 9. Epoch checkpoint integration

* The epoch-boundary save went through the M6D6c seam: `torch.save(model, path)` of the whole module,
  written to `.partial` and then `os.replace`. Result: `checkpoints/encoder_final.pkl`,
  **185,136,819 B**, SHA256 `ad124c5246ae220ab7bcbc2c6795b01d53bd466fe6ba2534efd88030a16a2d04`,
  epoch 1, step 56, `periodic`, not selected, `QUALIFICATION_ONLY`.
* It was reloaded through `load_verified_whole_module`, SHA256 first: pinned `custom_rn.ResNet`, fc
  512→7, 223 state entries equal to the in-memory model.
* **Cleanup:** the bytes were removed after the evidence was recorded. The checkpoints directory is
  empty, no weight file remains in the run root, and the checkpoint index records the removal.
* **The scientific checkpoint is absent:** `runs/` stayed absent throughout the session.
* In scientific mode the same file is overwritten every epoch and is **not** authoritative before
  epoch 200. Its SHA256 is recorded at epoch 200 as `SHA256_RECORDED_PENDING_OWNER_FREEZE`.

## 10. Access audit (per process, never combined)

| Process | TRAIN rows exposed | TRAIN face reads | VAL/TEST rows | VAL/TEST image reads | raw reads | unauthorized manifests | denials |
|---|---|---|---|---|---|---|---|
| b256_1 | 14467 | 3329 (13 prefetched batches + 1 transform check) | 0 | 0 | 0 | 0 | 0 |
| b256_2 | 14467 | 3329 | 0 | 0 | 0 | 0 | 0 |
| epoch | 14467 | **14336 = exactly the consumed set** | 0 | 0 | 0 | 0 | 0 |

In each process:

* The split manifest had 2 Python SHA256 opens (the contract bound check and the pre-parse check)
  plus 3 pyarrow API calls (`read_metadata`, `read_schema`, `read_table`), all on that exact path.
* The class map had 2 reads.
* Subprocesses were limited to git, nvidia-smi, `ldconfig -p` and `uname -p`.

Superseded development attempts are disclosed in the runtime log §A6:

* 2 B256 processes: PASS, bitwise equal to the authoritative ones.
* An epoch attempt aborted before any step by a firewall denial of the qualification parent `mkdir`.
* A full epoch attempt, bitwise equal to the authoritative epoch, whose final gate tripped on a
  stdlib `uname -p` subprocess denial.

Their TRAIN face reads (3329, 3329, 0, 14336) are reported separately. They had 0 VAL/TEST accesses
and 0 benchmark-path denials. Their artifacts are retained, not deleted.

## 11. Tests and preflight

* `tests/test_m6d6e_e07c_aux_runner.py` (39 tests):
  * Laptop: 34 pass, 5 Torch-live tests skipped.
  * GPU: 39/39 pass, CUDA hidden, CPU-only synthetic live checks.
* Regressions M6D6d, M6D6c, M6D6b, M6D6a, M6C2b3 (contract, encoder, runtime): all pass on both
  hosts, with the Torch tests skipped on the laptop only.
* Every run was under an audit hook with 0 denials. The M6D6e runner counts authorized split-manifest
  and class-map reads; the regressions deny all manifest access.
* `tools/m6d6e_e07c_aux_runner_preflight.py` is static: no Torch, no YAML parser, no Parquet or image
  read. It validates the recorded evidence and does not rerun training.

## 12. Statuses

* **Qualified:** `E07c_AUX_REAL_TRAIN_PATH_QUALIFIED`, `E07c_AUX_B256_TRAINING_MEMORY_QUALIFIED`,
  `E07c_AUX_PRODUCTION_RUNNER_QUALIFIED`, `E07c_AUX_EPOCH_CHECKPOINT_INTEGRATION_QUALIFIED`.
* **Still not qualified:** `AUXILIARY_ENCODER_200_EPOCH_TRAINING`,
  `AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT`, `AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`, `AUX_RESUME`,
  `MAIN_DIFFFAS_TRAINING_GRAPH`, `MAIN_CHECKPOINT_RESUME`, `MAIN_RUNNER_ENCODER_LOAD_INTEGRATION`,
  `MAIN_PRODUCTION_RUNNER`, `SCIENTIFIC_TRAINING`, `M8_BANK`.
* **Scientific counters:** scientific auxiliary runs 0, scientific main runs 0; auxiliary seed 42
  used = false; seeds 42/1337/2026 were not used for any run.
* No owner decision was made, there is no new amendment or deviation, and the source is not patched.
