# M6D6b — E07c auxiliary conditioning-encoder training-graph qualification

**M6D6b final status: PASS** · Classification: **M6D6B_E07C_AUX_TRAINING_GRAPH_QUALIFICATION**.
This checks that the future auxiliary-encoder training graph can run. It is not auxiliary training and not a
scientific milestone.

| Status | Basis |
|---|---|
| **E07c_AUX_ENCODER_TRAINING_GRAPH_QUALIFIED** | pinned `custom_rn.resnet18` + A3 `Linear(512, 7)`, source-default train mode, CE on the fourth output, backward, finite and connected gradients |
| **E07c_AUX_ENCODER_OPTIMIZER_STEP_QUALIFIED** | exactly one `SGD(model.parameters(), lr 0.002, momentum 0.9, weight_decay 0.005)` step; update matches the first-step SGD formula within 1.49e-8 |
| **IMPLEMENTED_NOT_EXECUTED** | E07c scientific method status unchanged; 0/3 main seeds run; the one auxiliary encoder has not been trained |
| **CONTROLLED_ADAPTATION** | A1 identity-free + A3 controlled encoder reconstruction, DEV-021; A6 source-traceability only |

Retained from M6D6a/M6D6a-r (re-verified, not re-claimed): E07c_EXECUTION_ENVIRONMENT_QUALIFIED,
E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED, E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED,
E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED.

Reporting label: DiffFAS-BIN-IDFREE (controlled encoder reconstruction). It is not native, not faithful and not an
official reproduction. All artifacts of this milestone are **QUALIFICATION_ONLY · NOT_ELIGIBLE_FOR_BANK ·
NOT_ELIGIBLE_FOR_DOWNSTREAM · NOT_ELIGIBLE_FOR_REPORTING**.

## 1. Authority and synchronization

- Laptop: `m6-baselines`, HEAD `ca206cdc15de830e467484baace84208678fb5ce` = `git ls-remote` remote; worktree clean.
- GPU: started at `596d845` (clean, ancestor). `git fetch origin` → `git merge-base --is-ancestor HEAD origin/m6-baselines`
  (OK) → `git merge --ff-only origin/m6-baselines` → `ca206cd`, clean. No reset, clean, rebase or forced checkout.
- Ledger 116 rows; artifact index 656 rows. M6D6a/M6D6a-r evidence byte-identical to the authority commit.

## 2. Upstream training semantics (pinned `models/pretrain_classifier.py`, SHA256 `3444691f…7b16`; AST only, never imported)

| Question | Pinned source evidence | M6D6b behaviour |
|---|---|---|
| Train/eval mode | no `.train()` or `.eval()` call anywhere in the script; `nn.Module` defaults to `training=True` | every submodule verified `training=True` at construction; `model.train()` (idempotent) |
| BatchNorm | batch statistics, running stats updated (default train mode) | same; 36 BatchNorm2d layers, ≥256 values per channel at B=4; no frozen-BN or running-stat workaround |
| Construction order | `:15` `resnet18()` → `:16` `fc = nn.Linear(512,17)` → `:17` `.cuda()` → `:27` SGD → `:28` CE | committed A3 seam (`resnet18()` then `fc = Linear(512, 7)`) → `.cuda()` → SGD → CE |
| Optimizer set | `:27` `optim.SGD(resnet18.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-3)` | `SGD(model.parameters(), …)`: the whole network, checked by object identity: 112 tensors / 46,233,707 parameters |
| Loss | `:28` `nn.CrossEntropyLoss()`; `:37-38` `_,_,_,outputs = resnet18(inputs)`; `criteria(outputs, labels)` | plain CE (mean, no weight, no smoothing) on `outputs[3]` only |
| Step order | `:36` zero_grad → `:37` forward → `:38` CE → `:40` backward → `:41` step | identical; counted by pass-through wrappers |
| Scheduler / clipping | none | none (constructor/clip calls forbidden and counted: 0) |
| Precision | no autocast, GradScaler, TF32 flag, dtype cast or cudnn flag | see §4 |

The frozen A3 `training_contract` (SGD, 0.002, 0.9, 5e-3, scheduler NONE, `CrossEntropyLoss_on_fourth_forward_output`)
equals these source values exactly.

**Source-native disconnected parameters.** `custom_rn.py:273` builds `self.norm = nn.BatchNorm1d(18)` and `:274`
`self.in1 = nn.InstanceNorm2d(64)`. Neither is called by `_forward_impl`. `norm.weight` and `norm.bias` (36 values) are
therefore in the pinned SGD parameter set but receive no gradient. This is how the pinned source behaves; it is not an
M6D6b choice. The harness accepts exactly these two tensors as `None`-gradient and fails on any other disconnected
tensor. Separately, the live Torch test walks the autograd graph without calling backward and finds the same two
tensors. PyTorch SGD skips parameters whose gradient is `None`, weight decay included, so both stayed bitwise
unchanged.

## 3. Qualification inputs (synthetic only)

- `qualification_seed = 60602` (not `auxiliary_encoder_training_seed` 42, not experiment seeds 42/1337/2026; no
  scientific seed consumed). `PYTHONHASHSEED=60602`.
- `qualification_batch_size = 4`; `scientific_aux_batch_size = 256` (frozen, unchanged);
  `batch_256_training_memory_not_qualified = true`.
- Input: `[4,3,256,256]` float32 in [-1,1], built from the same analytic sin/cos family as M6D6a (`rq.synthetic`,
  phase 37). No RNG, no file. SHA256 `621d27eb…bdba`.
- Target: `LongTensor[4] = [1, 4, 0, 3]` from the rule `(3i+1) mod 7`. These are class indices only and are not
  benchmark class samples. SHA256 `ee95e5ce…7032`.

## 4. Precision: engineering controls only

M6D6b reused the M6D6a qualification controls: FP32, matmul and cuDNN TF32 off, cudnn benchmark False,
deterministic True, float32 matmul precision `highest`, no autocast, no GradScaler, deterministic algorithms not
forced. These are **ENGINEERING_QUALIFICATION_CONTROLS** and do not set the scientific precision policy. The pinned
auxiliary script has no explicit policy, so it runs under library defaults. In this runtime those defaults were
observed before the controls were applied: `cudnn.allow_tf32 = True`, `matmul.allow_tf32 = False`, cudnn
benchmark/deterministic False/False. Default execution could therefore use TF32 cuDNN convolutions on Ampere. The
production auxiliary precision policy remains **open**.

## 5. Results (two fresh GPU processes, `gpat-m6-e07c`, RTX 3090)

Both `M6D6B_E07C_AUX_TRAINING_PROCESS_1.json` and `_2.json` report `status = PASS`, have no runtime warnings and are
**byte-identical** (SHA256 `7390c6c9836b5a9de6a80be528ef9fc01a1bafadef69aca87e7a268d279ab31b`).

| Item | Value (identical in both processes) |
|---|---|
| Encoder | `custom_rn.ResNet`, BasicBlock `[3,4,6,3]`, widths 64/256/512/512, only `fc` differs from pinned `resnet18()` (17 → 7) |
| Features / logits | `[4,256,32,32]`, `[4,512,16,16]`, `[4,512,8,8]` / logits `[4,7]` |
| CE loss (pre-backward) | **2.049288272857666** (float32 `0x1.064f14p+1`); float64 reference deviation 1.63e-7 |
| Counters per process | SGD constructions 1, zero_grad 1, **backward 1**, **optimizer.step 1**, autograd.grad 0, scheduler 0, clipping 0, torch.save 0, torch.load 0 |
| Optimized set | 112 tensors, 46,233,707 parameters, 1 param group; lr 0.002, momentum 0.9, dampening 0, weight_decay 0.005, nesterov False, maximize False |
| Gradients (after backward, before step) | 110 tensors with gradient, all finite, all 110 non-zero; `None`: `norm.bias`, `norm.weight` only (source-native); unexpected disconnected: 0 |
| Gradient magnitudes | global L2 64.7126; max abs 3.3011 (stem `conv1.weight`); `fc.weight` 3584/3584 non-zero, L2 9.2388; `fc.bias` 7/7 non-zero, L2 0.3852 |
| Backbone reach | non-zero gradient at `conv1`, `layer1.0`, `layer2.0`, `layer3.0`, `layer4.0` |
| Parameter update | 110 tensors changed (2 fc + 108 backbone); unchanged: `norm.bias`, `norm.weight`; all finite; changed outside optimizer scope 0 |
| SGD check | max \|p_after − (p − 0.002·(g + 0.005·p))\| = 1.49e-8 (tolerance 1e-6); 110 momentum buffers |
| Aggregate parameter SHA256 | before `aa4c2cdd…ba46` → after `3fc8d8b3…8aa6` |
| Buffers | train-mode forward updated only BatchNorm2d `running_mean/var/num_batches_tracked`; backward/step changed no buffer |

**Repeatability:** bitwise. Initial state, synthetic input and target hashes, pre-step loss, all 112 per-tensor
gradient summaries, all 112 post-step parameter hashes and the changed-tensor counts are identical, so no tolerance
was needed. Deterministic algorithms were not forced. The equality was observed, not engineered.

**AUX_ENCODER_SYNTHETIC_TRAINING_STEP_MEMORY_OBSERVATION** (B=4): 185,067,520 B allocated / 213,909,504 B reserved after
model construction; one-step peak 1,192,865,280 B allocated / 1,300,234,240 B reserved.
**B256 scientific training memory remains unqualified.**

## 6. Access audit (complete M6D6b session, laptop + GPU)

Every command used explicit allowlisted paths. There was no `git ls-files | xargs`, no recursive repository or runtime
scan, and no shell here-document. The log was written by a Python writer. `tools/build_artifact_index.py`, which
hashes every file, was **not** run. The index was rebuilt from the committed rows plus only the new artifacts.

| Count (complete session) | Value |
|---|---:|
| benchmark manifest reads | **0** |
| benchmark image reads | **0** |
| TRAIN / VAL / TEST reads | **0 / 0 / 0** |
| firewall denials (2 GPU processes + 11 audited test runs: 5 laptop, 6 GPU) | **0** |
| `backward()` calls | **2** (1 per qualification process) |
| `optimizer.step()` calls | **2** (1 per qualification process) |
| torch.save / torch.load | **0 / 0** |
| scientific checkpoints created | **0** |
| auxiliary encoder training runs | **0** |
| scientific seed runs completed | **0 / 3** |

`<runtime_root>/runs` was absent before and after both processes and after cleanup. No `runs/m6/E07c`,
`aux_encoder/seed_42` or `encoder_final.pkl` exists. Qualification scratch lives only in
`<runtime_root>/builds/e07c_difffas/m6d6b/`, outside the scientific run roots.

## 7. Tests (actual counts, all audited)

| Suite | Laptop (no Torch) | GPU (`gpat-m6-e07c`, CUDA hidden) |
|---|---|---|
| `tests.test_m6d6b_e07c_aux_training_graph` (new) | 17 run, 16 pass, 1 skip (live Torch) | 17 run, 17 pass |
| `tests.test_m6d6a_e07c_runtime` (unchanged) | 16 run, 15 pass, 1 skip | 16 run, 16 pass |
| `tests.test_m6c2b3_{contract,encoder,runtime}` (unchanged) | 46 run, 45 pass, 1 skip | 46 run, 46 pass |

Disclosure: the first GPU run of the new suite had 1 failure, and it was in the test helper, not the harness. The
live graph walk de-duplicated autograd nodes by the `id()` of short-lived Python wrappers. CPython reuses those ids once
the wrappers are freed, so the walk stopped early. The fix keeps each visited node alive. The rerun passed, and so did
a laptop rerun. No harness, evidence or model code changed. The live test calls neither `backward()` nor `step()`.

## 8. Preflight, ledger, index

`python3 -I -S -B tools/m6d6b_e07c_aux_training_graph_preflight.py` is static: no Torch, CUDA, model, manifest or image.
It validates:
- the authority commit and the unchanged frozen and M6D6a inputs;
- the source pin, and the upstream semantics re-derived independently with the stdlib AST;
- the environment lock against the executed identity;
- both process records and their bitwise repeatability;
- the session access audit;
- that the report avoids forbidden wording;
- the single ledger append and the CRLF index.

Ledger 116 → 117 (one M6D6b row). The artifact index was rebuilt last.

## 9. Open future issues (recorded, not resolved)

- **torch.load / weights_only:** the qualified runtime's `torch.load` defaults to `weights_only=True`. The whole-module
  auxiliary checkpoint (`torch.save(model)` at `pretrain_classifier.py:45`, consumed by `torch.load(path)` in
  `unet_autoenc.encoder`) needs an explicit loader decision. M6D6b neither saved nor loaded anything.
- **Production precision policy:** see §4; undecided.
- **Real auxiliary B256 training memory:** not exercised; the observation here is B=4 only.

## 10. Still not qualified

Real TRAIN path · auxiliary encoder training (200 epochs) · aux B256 training memory · aux production runner ·
checkpoint writer · checkpoint loader · resume · main DiffFAS training graph · scientific training (0/3 seeds) · M8 bank.

**NO BENCHMARK DATA ACCESS. EXACTLY 2 SYNTHETIC QUALIFICATION STEPS (1 PER PROCESS). NO CHECKPOINT. NO AUXILIARY ENCODER
TRAINING. NO SCIENTIFIC TRAINING. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
