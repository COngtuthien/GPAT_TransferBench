# Amendment A7 — E07c production execution policy

Status: **OWNER-FROZEN · ADDITIVE · NON-DESTRUCTIVE**.
Class: **`DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`** (A2 §1 taxonomy).
Milestone: M6D6d. Method: E07c — DiffFAS-BIN-IDFREE (controlled encoder reconstruction).

A7 fixes two execution details that the frozen specification, A1/A2/A3/A6, the frozen E07c
configs and the pinned DiffFAS source leave open, and that M6D6c recorded as open decisions
(`PRECISION_POLICY_DECISION_REQUIRED`; the throwaway `resnet18()` RNG question). Both are
**owner-frozen benchmark execution policy**. Neither is a claim that the DiffFAS authors chose
it, and neither changes a scientific quantity: architecture, objective, K7 label space, data,
seeds, budgets, checkpoint format and rules are untouched.

E07c fidelity stays **`CONTROLLED_ADAPTATION`** under **DEV-021** (A1 + A3). A7 adds **no new
fidelity class and no new scientific deviation**. The source pin is unchanged
(`murphytju/DiffFAS` @ `23f40519ec25a833ebc06842aa6fbab74fad4d15`). A1, A2, A3, A6, the A6
overlay and the frozen E07c configs are not edited.

## 1. Precision policy (auxiliary encoder and main DiffFAS processes)

| Control | Frozen value |
| --- | --- |
| tensor compute | **FP32** (`torch.get_default_dtype() == torch.float32`) |
| autocast | **disabled** — no autocast context anywhere in E07c |
| GradScaler | **disabled** — never constructed |
| `torch.backends.cuda.matmul.allow_tf32` | **False** |
| `torch.backends.cudnn.allow_tf32` | **False** |
| AMP / FP16 / BF16 / mixed precision | **forbidden** |

Rationale of record:

1. The pinned E07c source requests no AMP, autocast or GradScaler (no occurrence in any pinned
   `.py` file).
2. The pinned source requests no TF32 setting either way.
3. Relying on library defaults would make numerics depend on the PyTorch/CUDA/hardware
   combination; the qualified stack's own defaults differ between cuBLAS (matmul TF32 off) and
   cuDNN (TF32 on), as observed in M6D6c.
4. Every live E07c qualification (M6D6a, M6D6a-r, M6D6b, M6D6c) ran FP32 with TF32 disabled.
5. An explicit FP32 / TF32-off contract is reproducible and host-independent.

Spec AMP rows (§10.5 GPAT, ArtifactProbeNet, downstream detector) do not govern E07c: spec §8.7
defers E07c's executable settings to the pinned repository.

## 2. cuDNN algorithm flags

| Flag | Main DiffFAS | Auxiliary encoder |
| --- | --- | --- |
| `torch.backends.cudnn.benchmark` | **False** (existing authority) | **False** (A7) |
| `torch.backends.cudnn.deterministic` | **True** (existing authority) | **True** (A7) |

The main-training values already exist in `configs/methods/e07c_difffas_bin_idfree.yaml`
(`training.cudnn`) and in pinned `FAS_train.py::seed_torch` (lines 174–175). A7 extends the same
controls to the auxiliary encoder, which is trained once and reused by all three main seeds, so
its artifact must not depend on autotuner choice. The pinned `pretrain_classifier.py` sets no
flag.

A7 does **not** set `torch.use_deterministic_algorithms(True)`; no existing E07c authority requires
it, and A7 introduces no stronger determinism policy. A7 also does not change the recorded
launch environment of `environments/e07c.lock.json`.

## 3. Throwaway `resnet18()` RNG semantics — preserved

Pinned `models/unet_autoenc.py::BeatGANsAutoencModel.encoder(path)` (lines 73–77):

```python
def encoder(self,path):
    model_autoencoder = resnet18()          # :74  (from .custom_rn import resnet18, :10)
    model_autoencoder = torch.load(path)    # :75  overwrites :74
    model_autoencoder = model_autoencoder.cuda()
    return model_autoencoder
```

The object built at line 74 is overwritten at line 75, and its parameters are never used. Its
construction (`custom_rn.py:367–369 → _resnet → ResNet.__init__`) runs on the CPU:
`kaiming_normal_` on every `Conv2d` (`:275–277`) and the default `nn.Linear` initialisation of
the 17-way head (`:271`). That consumes **PyTorch CPU RNG** state.

**Owner decision:** `PRESERVE_UPSTREAM_THROWAWAY_ENCODER_CONSTRUCTOR_RNG = TRUE`.

Immediately before the secure checkpoint load, the future GPAT main runner executes exactly one
pinned, source-native `custom_rn.resnet18(pretrained=False)` on the CPU and discards it. The
default 17-way head is kept, and the A3 `fc → Linear(512, 7)` replacement is **not** applied,
because line 74 does not apply it. The object is never used, moved to GPU, saved, retained,
returned or modified. Manual `torch.rand` calls, estimated draw counts and torchvision ResNet are
forbidden substitutes. Its **only** role is `RNG_COMPATIBILITY_CONSUMPTION`; it has no
conditioning role.

Implementation: `methods/difffas/execution_policy.py::consume_upstream_encoder_loader_rng`
(`RNG_COMPATIBILITY_ONLY`). The M6D6c secure loader stays **mandatory** for the actual encoder:
SHA256 of the exact bytes → pinned `custom_rn` module identity →
`torch.load(bytes, weights_only=False)` → identity validation → `.cuda()`
(`methods/difffas/aux_checkpoint.py::load_frozen_aux_encoder`). The composed seam
`execution_policy.main_runner_encoder` replaces `FAS_train.py:34`. The caller then calls
`encoder.eval()` (`:35`).

### 3.1 Why the consumption matters

In pinned `FAS_train.py` the throwaway constructor runs after the main/EMA models were built on
the CPU and before the first training iterator. The next **CPU**-generator consumers are the
DataLoader iterator's base seed and `RandomSampler` permutation, created at every epoch
(`:40–41`, `shuffle=True`), and the periodic `torch.randperm` at `:125`. Skipping the constructor
shifts every later CPU draw, so the batch order of every epoch changes. The later per-step draws
(`time_t` at `:50–55` with `device=cuda`, `randn_like` noise and `prob_mask_like` condition masks
in `diffusion.py::training_losses`, UNet dropout) use the **CUDA** generator, which the
constructor does not touch. Their pairing with batches still changes, because the batches
change. Bypassing the constructor is therefore **not trajectory-neutral**.

### 3.2 Source-derived order (pinned `FAS_train.py`, fresh run, `use_pair=False`)

| # | Line | Upstream statement | GPAT future main runner |
| --- | --- | --- | --- |
| 1 | 180 | `seed_torch(seed=1)` | experiment seed via `seed_adapter.apply_seed` (42/1337/2026); A7 precision/cuDNN policy |
| 2 | 184 | `transform = transforms.Compose([...])` | frozen preprocessing |
| 3 | 189 | `if args.protocol == 'PADISI': ...` (dataset) | A1 TRAIN manifest dataset |
| 4 | 205 | `dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)` | same |
| 5 | 216–217 | `model = get_model_conf().make_model()` · `.to(args.device)` | same (CPU init, then CUDA) |
| 6 | 218–219 | `ema = get_model_conf().make_model()` · `.to(args.device)` | same |
| 7 | 222 | `optimizer = DiffConf.training.optimizer.make(model.parameters())` | same |
| 8 | 223 | `scheduler = DiffConf.training.scheduler.make(optimizer)` | same |
| 9 | 226 | `if args.pretrain_path is not None:` (resume; not taken on a fresh run) | resume contract not qualified |
| 10 | 234 | `betas = DiffConf.diffusion.beta_schedule.make()` | same |
| 11 | 235 | `diffusion = create_gaussian_diffusion(betas, predict_xstart=False)` | same |
| 12 | 236 | `train(...)` | same |
| 13 | 34 → `unet_autoenc.py:74` | throwaway `resnet18()` | `consume_upstream_encoder_loader_rng` |
| 14 | 34 → `unet_autoenc.py:75–76` | `torch.load(path)` · `.cuda()` | `load_frozen_aux_encoder` (M6D6c secure loader) |
| 15 | 35 | `encoder.eval()` | same |
| 16 | 37, 40–41 | epoch loop · `tqdm(loader)` · iterator creation (CPU RNG) | same |
| 17 | 50 | `time_t = torch.randint(..., device=device)` (CUDA RNG) | same |

### 3.3 Scope note — sampler

Pinned `FAS_sample.py:48` contains the same `encoder = model.encoder(args.pretrain_classifier)`
call. A7 freezes the rule for the **main training runner** only. Whether the future generation
(M8) runner replays the constructor is **not decided here** and is left to the owner.

## 4. What A7 does not do

A7 does not modify the pinned source, change the environment, train anything, read benchmark
data, create or load any checkpoint, set `use_deterministic_algorithms`, or change a frozen config,
amendment or snapshot. Scientific seeds (auxiliary 42; main 42/1337/2026) are not consumed.

## 5. Additive immutable execution input

Overlay: `configs/amendments/e07c_a7_execution_policy.yaml`. Its SHA256 is pinned in
`methods/difffas/execution_policy.py` (`A7_OVERLAY_SHA256`). The overlay records this document's
SHA256, the base config identity, the A3 document and A6 overlay identities, and the pinned source
file digests. Future E07c production processes must verify it through
`execution_policy.load_policy` before they apply the policy.
