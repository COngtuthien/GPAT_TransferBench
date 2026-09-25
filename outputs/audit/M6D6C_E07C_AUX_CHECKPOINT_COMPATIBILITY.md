# M6D6c — E07c DiffFAS-BIN-IDFREE auxiliary checkpoint: whole-module serialization + loader compatibility

**Classification:** `M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY_QUALIFICATION` · **Result:** PASS ·
**Method status:** `IMPLEMENTED_NOT_EXECUTED` · **Fidelity:** `CONTROLLED_ADAPTATION` (DEV-021, unchanged)

Every checkpoint in this milestone is `QUALIFICATION_ONLY` · `NOT_ELIGIBLE_FOR_BANK` · `NOT_ELIGIBLE_FOR_DOWNSTREAM` ·
`NOT_ELIGIBLE_FOR_REPORTING` · `NOT_A_SCIENTIFIC_CHECKPOINT`. No auxiliary encoder was trained. No benchmark data
was read. No scientific checkpoint exists.

## 1. Authority

| Item | Value |
|---|---|
| Laptop / remote `m6-baselines` | `b89b8d0e0ac5f12fe43a7cfb09b02cda255f4661`, clean |
| GPU | `ca206cd` → fast-forward only → `b89b8d0`, clean before and after |
| Frozen specification | SHA256 `f7d23716…`; §8.7 names the pinned repository as the architecture/config source of truth |
| Pinned DiffFAS | `murphytju/DiffFAS@23f40519…`, tree `d190a5fb…`, unmodified |
| Environment | `gpat-m6-e07c`, lock SHA256 `0c909de1…`; conda/pip/python hashes re-verified before and after; no mutation |

## 2. The checkpoint format is frozen. It is not an open choice.

- `configs/methods/e07c_difffas_bin_idfree.yaml` → `conditioning_encoder.checkpoint_format`:
  *"torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())"*.
- Amendment A3 §5.2: *"Checkpoint format is preserved: `torch.save(model)` of the **whole `nn.Module`**"*. A6 leaves it
  unchanged.
- Pinned writer `models/pretrain_classifier.py`: `:17 resnet18 = resnet18.cuda()`, then the `:45` call
  `torch.save(resnet18, './PADISI.pkl')` inside the epoch loop. The whole module object is saved, not a state_dict.
- Pinned consumer `models/unet_autoenc.py::BeatGANsAutoencModel.encoder` `:74-77`: `resnet18()` → `torch.load(path)`
  (no `weights_only`, no `map_location`) → `.cuda()`. The caller `FAS_train.py:34-35` then calls `encoder.eval()`.

These facts were re-derived by AST from the pinned bytes. No upstream file was imported or executed.
M6D6c keeps the whole-module pickle. It does not convert to state_dict, safetensors, TorchScript or ONNX.

## 3. Observed runtime behaviour (torch 2.12.1+cu130), recorded rather than assumed

`inspect.signature(torch.load)`:
`(f, map_location=None, pickle_module=None, *, weights_only: bool | None = None, mmap: bool | None = None, **pickle_load_args)`.
`TORCH_FORCE_WEIGHTS_ONLY_LOAD` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` were both unset.

Each probe ran against the qualification checkpoint only. Its SHA256 was verified immediately before each call.

| Call | Pinned `custom_rn` importable | Outcome |
|---|---|---|
| `torch.load(path)` (the upstream call form) | yes | **REJECTED** `_pickle.UnpicklingError`: *Weights only load failed*. `custom_rn.ResNet` is not an allowed global |
| `torch.load(path, weights_only=True)` | yes | **REJECTED**, with the identical exception and message digest |
| `torch.load(path, weights_only=False)` | yes | **LOADED** `custom_rn.ResNet` (the pinned module's class) on `cuda:0`. Parameter and buffer aggregates equal the writer's |
| `torch.load(path, weights_only=False)` | **no** | **REJECTED** `ModuleNotFoundError: No module named 'custom_rn'`. Module identity is required during unpickle |

The default rejection is expected compatibility evidence. It is not a model failure. The default resolves to
`weights_only=True`, and the restricted unpickler refuses a whole pickled custom module.

## 4. Compatibility decision: the minimum GPAT-owned seam

**Selected:** `EXPLICIT_WEIGHTS_ONLY_FALSE_AFTER_SHA256_VERIFICATION`. It is classified as
`RUNTIME_COMPATIBILITY_ADAPTATION_RESTORING_LEGACY_WHOLE_MODULE_LOAD_SEMANTICS`. It is **not** a scientific
adaptation, a new checkpoint format or a source modification. There is no new deviation; DEV-021 is unchanged.

The loader harness gates the decision on the probes. If the default call had restored the module, the process
would have stopped with "no override justified".

`methods/difffas/aux_checkpoint.py`:

- `save_whole_module(model, path)` performs exactly `torch.save(model, str(path))` inside the committed
  `encoder_model()` context, so the pickle records `custom_rn.ResNet`. It refuses paths inside the repository.
- `read_verified(path, expected_sha256)` accepts only a 64-character lowercase-hex SHA256 and an external path. It
  reads the bytes, hashes them, and raises `PreparationError` on mismatch. It never deserializes.
- `load_verified_whole_module(path, sha, config)` is a context manager with this fixed order:
  **read bytes → SHA256 verified → pinned source re-verified → `upstream_modules` imports `custom_rn` →
  `torch.load(io.BytesIO(raw), weights_only=False)` → A3/A6 identity check → `.cuda()`**. The SHA256 is taken over
  the exact bytes that are unpickled, so the file is never read a second time. `custom_rn` stays imported while
  the caller uses the encoder.
- `load_frozen_aux_encoder(runtime_root, recorded_sha256)` is the future main-run entry point. It reuses
  `encoder.verify_future_checkpoint` on the frozen external path, then delegates to the gated load.
- `torch.load` is not patched globally. No `add_safe_globals` allowlist is used, because that would be a separate
  restricted-loader contract GPAT would have to maintain. It is recorded as an alternative that was not adopted.

## 5. Writer qualification (1 fresh process, `qualification_seed` 60603)

| Item | Value |
|---|---|
| Construction | committed A3 seam `encoder_model()` → `custom_rn.ResNet`, BasicBlock `[3,4,6,3]`, widths `[64,256,512,512]`, `fc = Linear(512, 7)` with bias |
| State at save | 46,233,707 parameters (112 tensors), 111 buffers, all on `cuda:0`, FP32 parameters; train mode (the source default) |
| Serialization | `torch.save(model, path)`, called once, with no extra arguments |
| Path | `<runtime_root>/builds/e07c_difffas/m6d6c/checkpoint/e07c_aux_encoder_QUALIFICATION_ONLY.pkl` (not under `runs/`) |
| File | **185,141,005 bytes**, SHA256 **`cf89680a56cdb257e9739fb0ded55522631a9f8291719d972a05c1168460522b`** |
| Archive | torch zip, pickle protocol 2, 223 storage records. Globals include `custom_rn.ResNet` and `custom_rn.BasicBlock`; no torchvision class (listed by `pickletools.genops`, nothing unpickled) |

The disclosed writer attempt 1 produced a byte-identical file (same SHA256). Its JSON was superseded only
because its signature field had been captured after the counting wrapper was installed. See the runtime log, §A4.

## 6. Loader qualification (2 fresh processes, same checkpoint, same expected SHA256)

- **SHA before deserialization:** the event order is `open_checkpoint → sha256_verified → torch.load(bytes,
  weights_only=False) → find_class ×15`. The first `pickle.find_class` audit event comes after the verification event.
- **Wrong SHA** (one hex digit flipped): the events are `open_checkpoint → sha256_rejected`. `torch.load` was called
  0 times and `find_class` fired 0 times. A **malformed SHA** is rejected before the file is even opened.
- **Identity:** `custom_rn.ResNet`, the live module file is the pinned `models/custom_rn.py`, and the class is the
  pinned module's class. The module tree is equal to the writer's, with no projection or adapter. `fc` is 512 → 7.
- **Parameters:** 112/112 tensors bitwise equal, 46,233,707 values. **Buffers:** 111/111 bitwise equal.
  Both sit on `cuda:0`. Parameters are float32; buffers are float32 plus int64 `num_batches_tracked`.
- **Forward round trip:** on a synthetic analytic B=4 input at 256×256 (phase 53), in eval mode as in
  `FAS_train.py:35`: `x32x32 [4,256,32,32]`, `x16x16 [4,512,16,16]`, `x8x8 [4,512,8,8]` and the fourth output
  `[4,7]` are **all bitwise equal** to the writer's pre-save reference. No tolerance was used.
- **Repeatability:** the two loader process JSONs are byte-identical (SHA256 `70d0dd73…`).

## 7. Cleanup

The checkpoint bytes were removed only after the writer and both loader records existed. At handoff:
`QUALIFICATION_PKL_ABSENT`, `ENCODER_FINAL_ABSENT`, `RUNS_ABSENT`, and zero `*.pkl/*.pt/*.pth` files in the m6d6c build root.
No checkpoint bytes are in Git.

## 8. Precision policy audit: PRECISION_POLICY_DECISION_REQUIRED (still open)

I searched the frozen specification text, Amendments A1/A2/A3/A6, both frozen E07c configs, the pinned source and
the prior E07c evidence. The specification's AMP entries cover the GPAT generator, the ArtifactProbeNet and the
downstream detector, not E07c. §8.7 defers to the repository, and the repository sets no autocast, GradScaler,
TF32 or cudnn flag in `pretrain_classifier.py`. The E03–E06c precision decisions are method-specific.
`NVIDIA_TF32_OVERRIDE=0` in the E07c lock is a recorded qualification launch variable, and M6D6a explicitly labeled
it as not deciding the production policy. M6D6c used the same ENGINEERING_QUALIFICATION_CONTROLS as M6D6a/b (FP32,
TF32 off). It **decides nothing** and adds no precision value to any config. Serialization qualification did not
need this decision.

## 9. Integration observations for later milestones (not resolved here)

- `FAS_train.py:34` obtains the encoder through `BeatGANsAutoencModel.encoder(path)`, whose bare `torch.load(path)`
  is exactly the call observed to be rejected. A future main-run integration must obtain the encoder through
  `load_frozen_aux_encoder` rather than that method. This change is still not a source patch.
- That upstream method first executes `model_autoencoder = resnet18()` (`unet_autoenc.py:74`). This builds a 17-way
  model and consumes RNG before the result is overwritten by `torch.load`. Whether the main runner must reproduce
  that RNG consumption is an open main-runner decision.
- The writer pickles the class as top-level `custom_rn` (`from custom_rn import resnet18`), while the consumer
  imports `.custom_rn` inside the `models` package. The GPAT seam supplies the pinned file as top-level `custom_rn`,
  which is the identity the writer recorded.

## 10. Status

**Qualified in M6D6c:** `E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED`,
`E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED`, `E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED`.
These qualify the serialization and loader **mechanism** only. They do not cover epoch-200 checkpoint timing, the
production runner, or the creation of any scientific checkpoint.

**Retained:** `E07c_EXECUTION_ENVIRONMENT_QUALIFIED`, `E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED`,
`E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED`, `E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED`,
`E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED`, `E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED`.

**Still not qualified:** REAL_TRAIN_PATH · AUXILIARY_ENCODER_200_EPOCH_TRAINING · AUX_B256_TRAINING_MEMORY ·
AUX_PRODUCTION_RUNNER · PRODUCTION_PRECISION_POLICY (`PRECISION_POLICY_DECISION_REQUIRED`) ·
MAIN_DIFFFAS_TRAINING_GRAPH · MAIN_CHECKPOINT_RESUME · MAIN_RUNNER_ENCODER_LOAD_INTEGRATION · SCIENTIFIC_TRAINING ·
M8_BANK. The B256 scientific training memory remains unqualified.

Counters across all M6D6c processes: backward 0 · optimizer construction/step 0 · epochs 0 ·
`torch.save` 1 per writer process and `torch.load` 5 per loader process, all QUALIFICATION_ONLY ·
benchmark manifest/image/TRAIN/VAL/TEST reads 0 · firewall denials 0.
