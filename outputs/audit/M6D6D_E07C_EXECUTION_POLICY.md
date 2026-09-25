# M6D6d — E07c DiffFAS-BIN-IDFREE production execution-policy freeze + upstream RNG-consumption compatibility

Method: **E07c**, DiffFAS-BIN-IDFREE (controlled encoder reconstruction). Fidelity: **CONTROLLED_ADAPTATION**
(DEV-021, unchanged). Method status: **IMPLEMENTED_NOT_EXECUTED**. Result: **PASS** (uncommitted candidate).

Authority: laptop = remote = `e70c221464b4a2db4ea3d45176a0cee52a7747cf` (M6D6c). The GPU was fast-forwarded
`b89b8d0 → e70c221`, clean. Source pin unchanged: `murphytju/DiffFAS@23f40519…`, tree `d190a5fb…`. Environment
`gpat-m6-e07c` unchanged; the verifier's output was identical before and after.

## 1. Decisions frozen (Amendment A7, owner-frozen, additive)

Artifacts: `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A7_E07c_Execution_Policy.md`
(`0a46c3b06e277ad38b17a6e85fe2924441d8d13b1de887bf54903dda9aa292ba`) and
`configs/amendments/e07c_a7_execution_policy.yaml` (`3e4c758c1421aac6e993724a8a80b99e8b0754e75b983cec5ffca41479f492a3`).
The overlay SHA256 is pinned in `methods/difffas/execution_policy.py`. Class:
`DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`. A7 is an owner-frozen benchmark execution policy,
not a claim that the upstream authors chose it. It adds no new fidelity class and no new scientific deviation. A1, A2, A3 and A6,
the A6 overlay, and the frozen E07c configs and snapshots are byte-identical to `e70c221`.

| Control | Frozen | Observed after `apply_e07c_precision_policy` (both processes) |
| --- | --- | --- |
| compute dtype | FP32 | `torch.float32` |
| autocast | disabled | `is_autocast_enabled('cuda'/'cpu')` False; autocast entries 0 |
| GradScaler | disabled | constructions 0 |
| `cuda.matmul.allow_tf32` | False | False (library default was False) |
| `cudnn.allow_tf32` | False | False (**library default was True**) |
| `cudnn.benchmark` | False (main: existing; aux: A7) | False |
| `cudnn.deterministic` | True (main: existing; aux: A7) | True (library default was False) |
| `float32_matmul_precision` | consequence of matmul TF32 off | `highest` |
| `use_deterministic_algorithms` | NOT set | False |

Pre-existing precision authority for E07c: **none**. Spec §8.7 defers executable settings to the repository. The
spec's AMP rows cover only GPAT §10.5 and the probe/detector. M6D6c recorded `PRECISION_POLICY_DECISION_REQUIRED`.
The pinned source contains no autocast/GradScaler/TF32/half-precision token, so nothing conflicts with the owner
decision. TF32-off needed no source modification.

**Throwaway `resnet18()` RNG:** `PRESERVE_UPSTREAM_THROWAWAY_ENCODER_CONSTRUCTOR_RNG = TRUE`. The helper is
`consume_upstream_encoder_loader_rng` (RNG_COMPATIBILITY_ONLY). It builds exactly one pinned
`custom_rn.resnet18(pretrained=False)` on the CPU with the default 17-way head (no A3 K=7 replacement), as a bare
expression statement, and discards it. It has no conditioning role. The M6D6c secure loader stays mandatory:
`main_runner_encoder` = helper → `load_frozen_aux_encoder` (SHA256 of the exact bytes → pinned `custom_rn` →
`torch.load(bytes, weights_only=False)` → identity → `.cuda()`), replacing `FAS_train.py:34`. The caller then runs
`.eval()` (`:35`).

## 2. Source evidence

`models/unet_autoenc.py` (`127ecd59…`) `BeatGANsAutoencModel.encoder`, lines 73–77:
`:74 model_autoencoder = resnet18()` (imported `from .custom_rn import resnet18`, `:10`) → `:75 model_autoencoder =
torch.load(path)` overwrites it → `:76 .cuda()` → `:77 return`. In `models/custom_rn.py` (`fa788c4d…`),
`resnet18(pretrained=False, progress=True)` at `:367` returns `_resnet('resnet18', BasicBlock, [3, 4, 6, 3], …)`.
`ResNet.__init__` builds `fc = nn.Linear(512 * block.expansion, 17)` (`:271`) and runs `kaiming_normal_` on every
Conv2d (`:275–277`). It makes no CUDA call, so it consumes CPU RNG only.

Source-derived order (pinned `FAS_train.py`, `c5eb42a1…`; each step checked by AST at the exact line):
`:180 seed_torch` → `:184 transform` → `:189 dataset` → `:205 DataLoader(shuffle=True)` → `:216–217 model` →
`:218–219 EMA` → `:222 optimizer` → `:223 scheduler` → `:226 resume branch (not taken)` → `:234 betas` →
`:235 diffusion` → `:236 train()` → `:34 model.encoder(path)` = [`unet_autoenc :74` throwaway → `:75` load → `:76`
cuda] → `:35 encoder.eval()` → `:37` epoch loop → `:40–41` iterator (CPU RNG) → `:50 time_t` (CUDA RNG).

Why preservation matters: the next **CPU**-generator consumers after `:74` are the DataLoader iterator base seed and
the `RandomSampler` permutation at every epoch, plus `torch.randperm` at `:125`. Skipping the constructor therefore
changes the batch order of every epoch. `time_t`, noise, `cond_mask` and dropout use the CUDA generator, which the
constructor does not touch (verified below), but their pairing with batches still changes. The bypass control
confirms the constructor is not trajectory-neutral. `FAS_sample.py:48` contains the same call. A7 does **not**
decide the generation runner.

## 3. RNG equivalence (qualification_seed 60604; two fresh processes; byte-identical JSON)

Initial CPU RNG state (after `torch.manual_seed(60604)`): `c73af325596cd62bc1ba2ed2e6be7a49ffd5ea6da044ac565f01cf8bfc0e8d01`.

| Path | before | after | next `torch.rand(16)` |
| --- | --- | --- | --- |
| helper (cold `custom_rn` import inside window) | `c73af325…` | `530629ef50c480dd347ccd9ccccf3fdd4858f40074879f773893e45d91c06b71` | `ad0e952d042755ac13e21c5fe8dbb3880191b565ced7a01c517f5fcf25636464` |
| direct pinned `custom_rn.resnet18(pretrained=False)` | `c73af325…` | `530629ef…` | `ad0e952d…` |
| upstream call site `encoder(path)`, stopped at `:75` before I/O | `c73af325…` | `530629ef…` | `ad0e952d…` |
| helper, upstream main modules imported | `c73af325…` | `530629ef…` | `ad0e952d…` |
| bypass (no constructor, control) | `c73af325…` | `c73af325…` | `65ddce509a9c14c7382ecca332e26f81f88c4ec1f44b667e9743b5aa2d2890f9` |

- The comparisons use the bytes of `torch.get_rng_state()`. CUDA, Python `random` and NumPy RNG states are
  unchanged on every path (CUDA `9c185b2d…` before = after).
- The call-site throwaway (`models.custom_rn.ResNet`) and the direct object (`custom_rn.ResNet`) are both on the
  CPU with head `[512, 17]`, and their parameters are identical (`2800ca74…`).
- The RNG state captured at the `:75` interception equals the state after return.
- Discard: the helper returns a plain record (`object_returned: false`). After every path, zero throwaway
  instances remain alive (gc scan) and the CUDA allocation delta is 0. No source was mutated (source identity before
  = after, worktree clean). Normal allocator reclamation is not claimed to be exact.

## 4. Runtime under the policy (synthetic, eval, `torch.no_grad`)

The K7 encoder (A3 seam, untrained, in-memory) produced finite A6 shapes `[4,256,32,32] / [4,512,16,16] /
[4,512,8,8]` plus `embg [4,7]`. The main `BeatGANsAutoencModel` output is `[4,6,256,256]` FP32 and finite; the
middle block and last output block are finite and non-trivial. The policy state was observed unchanged inside
every encoder and main-model forward and at process end. No architecture change, backward, optimizer or
checkpoint. B256 scientific training memory remains unqualified.

## 5. Scope and access

- Counters per process: backward 0, optimizer construction/step 0, `torch.save` 0, real `torch.load` 0, and one
  pre-I/O interception of the upstream `torch.load(path)` stand-in, which opens no file.
- The firewall recorded 0 denials, and every subprocess was on the allowlist.
- Benchmark manifests, images and TRAIN/VAL/TEST reads: 0.
- Scientific auxiliary runs 0 and main runs 0; seeds 42/1337/2026 were not consumed.
- Tests: new suite 25/25 (laptop 2 skips; GPU 0 skips). The M6D6c, M6D6b, M6D6a and M6C2b3 regressions all pass.
  One first laptop run failed in a test-only assertion; it was fixed and is disclosed in the runtime log.

## 6. Status

New: **E07c_PRODUCTION_PRECISION_POLICY_FROZEN**, **E07c_MAIN_ENCODER_LOADER_RNG_POLICY_FROZEN**,
**E07c_THROWAWAY_ENCODER_RNG_COMPATIBILITY_QUALIFIED**, **E07c_EXECUTION_POLICY_RUNTIME_QUALIFIED**;
IMPLEMENTED_NOT_EXECUTED; CONTROLLED_ADAPTATION.

Retained: E07c_EXECUTION_ENVIRONMENT_QUALIFIED · E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED ·
E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED · E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED ·
E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED · E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED ·
E07c_AUX_CHECKPOINT_WHOLE_MODULE_SERIALIZATION_QUALIFIED · E07c_AUX_CHECKPOINT_LOADER_COMPATIBILITY_QUALIFIED ·
E07c_AUX_CHECKPOINT_SHA_BEFORE_DESERIALIZE_QUALIFIED.

Removed from the unresolved list: PRODUCTION_PRECISION_POLICY; the open throwaway-resnet18 RNG decision.

Still not qualified: REAL_TRAIN_PATH · AUXILIARY_ENCODER_200_EPOCH_TRAINING · AUX_B256_TRAINING_MEMORY ·
AUX_PRODUCTION_RUNNER · MAIN_DIFFFAS_TRAINING_GRAPH · MAIN_CHECKPOINT_RESUME · MAIN_RUNNER_ENCODER_LOAD_INTEGRATION ·
MAIN_PRODUCTION_RUNNER · SCIENTIFIC_TRAINING · M8_BANK. M6 is not closed.
