# M6D5b — E06c DSDG-BIN-IDFREE: exact training graph and physical batch 240 (STOP)

**M6D5b STOP.** Decision: **STOP_AND_REPORT**. Failure classification: **CUDA_OOM_PHYSICAL_BATCH_240**
(clean resource state, not RESOURCE_CONTAMINATION).

| Status | |
|---|---|
| E06c_RUNTIME_ENVIRONMENT_QUALIFIED | retained from M6D5a |
| E06c_ARCHITECTURE_RUNTIME_QUALIFIED | retained from M6D5a |
| **E06c_TRAINING_GRAPH_PHYSICAL_BATCH_240_STOPPED_OOM** | this milestone |
| **E06c_PHYSICAL_BATCH_240_NOT_QUALIFIED** | this milestone |
| E06c_FULL_TRAINING_NOT_EXECUTED | |
| IMPLEMENTED_NOT_EXECUTED | E06c method status unchanged |
| CONTROLLED_ADAPTATION | preserved (Amendment A1, DEV-020) |

Reporting label: DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation).
No optimizer step, no backward pass and no training of any kind has happened for E06c.

## Authority and GPU synchronization

Laptop HEAD = origin/m6-baselines = `6e3dd3384add2ef723c4fc00e30512e58cd761c5`
("M6D5a: qualify E06c runtime architecture"). Divergence is 0 0 and the worktree is clean. Ledger has 111 rows; the artifact
index has 584 rows. `AGENTS.md` was checked with `test -f`: absent. No recursive filesystem search was done.

The GPU started at `13fc86c` (= its origin, 0 0, clean). We fetched only `refs/heads/m6-baselines`. Ancestry was proved
(distance 1), and `git merge --ff-only` brought it to `6e3dd33` (0 0, clean). No reset, clean, stash, commit or push was used.
During execution the two new harness files were staged on the GPU as temporary untracked copies and removed afterwards
(see runtime log).

## Identity (unchanged before and after)

| Item | Value |
|---|---|
| Environment | `gpat-m6-e06c`. Lock SHA256 `91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95` was verified in-process. All 14 non-launch identity fields match the lock: Python 3.11.16, torch 2.12.1+cu130, CUDA 13.0, cuDNN 92000, driver 595.84, package set. No install, update or removal. |
| Pinned source | `JDAI-CV/FaceX-Zoo@16b793a7564a4b9308cf94e62bdb2ffacb3a725a`, tree `0d2216bd…`. The 13-file closure SHA256 equals the lock. Status is only the recorded CDCN weight removal. Source identity was checked before and after the run. |
| Transcription | All 46 transcribed statements of `train_generator.py` (lines 87–182) match the pinned bytes verbatim (`verify_pinned_statements` → `[]`). |
| LightCNN-29 v2 | GPU runtime path: 123,844,849 B, SHA256 `d0750746…9964`. Load: 61 checkpoint tensors, 60/60 matched, 0 missing, 0 shape mismatches; only `module.fc2.weight` was filtered. Identity was checked before and after. |
| Precision | FP32. TF32 off (matmul and cuDNN), autocast off, matmul precision `highest`, cuDNN deterministic, benchmark off. These are the M6D5a qualified runtime controls; upstream `cudnn.benchmark=True` is a runtime control, not model semantics. |

`compatibility_patch = NONE`. The M6D5a files (`runtime.py`, environments, audits, preflight, test), the configs,
`adapter.py`, `source.py`, `source_pins.json` and the source cache were not modified. `training_qualification.py`
imports the M6D5a helpers unchanged.

## Clean GPU resource state

We checked `nvidia-smi` on the shell at 03:47:42Z, and the harness checked it again in-process right before
the attempt. The RTX 3090 has 24,576 MiB in total: 254 MiB used, 23,870 MiB free.
**No compute processes were running.** The only other processes were display graphics processes
(Xorg, gnome-shell and similar, about 160 MiB). Nothing was killed. `resource_clean = true`, so the OOM is an E06c OOM,
not RESOURCE_CONTAMINATION.

## Exact B=240 attempt (process 1, diagnostic seed 60502)

Seed 60502 (PYTHONHASHSEED 60502) is for qualification only. It is not experiment seed 42, 1337 or 2026.

- **Inputs (SYNTHETIC_PHYSICAL_BATCH_240):** from `training_graph.synthetic_pair`. These are analytic in-memory patterns with
  per-row frequencies and phases: row b uses fx = 1 + (b mod 17), fy = 1 + (7b mod 13) and phase 2πb/240 + c, built as sin/cos
  products in [0,1]. They are computed in CPU float64, cast to FP32 and moved to CUDA. `x_spoof` and `x_live` are [240,3,256,256],
  FP32, in [0,1]. **All 480 rows are distinct.** `label_spoof` is [240] int64, all 0. No RNG, file, decoder or manifest was used.
- **Models:** `define_G(hdim=128, attack_type=1)` and `define_IP(is_train=False)`, each with the upstream
  `DataParallel(...).cuda()` wrapper (device_ids [0]). Live counts match exactly: netE_nir 13,790,048/17, netE_vis
  11,692,640/17, netG 19,887,494/21, netCls 129/2, netIP 10,475,872/60 (eval mode, requires_grad=false).
  Encoders and netG are in train mode.
- **Optimizer (exactly one, as at train_generator.py:87-88):**
  `torch.optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()), lr=2e-4)`.
  - Ownership: 1 param group; **45,370,182 parameters / 55 tensors**, in exact order. 0 duplicates; 0 overlap with netCls or netIP.
  - Resolved runtime defaults: betas (0.9, 0.999), eps 1e-8, weight_decay 0, amsgrad false, maximize false,
    foreach None, fused None, capturable false, differentiable false, decoupled_weight_decay false.
  - The only explicit argument is `lr`.
- **Sequence (pinned order):** construct models, load and freeze LightCNN, construct Adam, allocate B=240. Then
  forward (encoders, pinned `reparameterize` in the order z_cls, z_nir, z_vis, then netCls, then netG), the seven losses and the
  epoch-1 total, `optimizer.zero_grad()` (line 180, after the forward as in the source), `backward()` and `step()`.

### Result: CUDA out of memory in the netG forward

| Field | Value |
|---|---|
| Phase | `forward_classifier_generator`: netCls(z_cls) + netG(cat(z_cls,z_nir,z_vis)), train_generator.py:130-133 |
| Error | `torch.OutOfMemoryError`: "CUDA out of memory. Tried to allocate 960.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 634.44 MiB is free. … 21.23 GiB is allocated by PyTorch, and 1.13 GiB is reserved by PyTorch but unallocated." |
| Allocator at OOM | allocated 22,799,887,872 B (peak 22,800,297,984); reserved 24,008,196,096 B (peak 24,010,293,248); cudaMalloc retries 2; OOMs 1 |
| Device at OOM | free 665,255,936 of 25,294,995,456 B. nvidia-smi: 23,489 MiB used; the only compute process was the harness itself |
| Not reached | LightCNN identity path, loss construction, `zero_grad`, backward, optimizer step |
| Optimizer | constructed 1; step entered **no**; applications **0** |
| Backward passes | **0** |
| Parameters | **unchanged**: 0 changed elements in netE_nir, netE_vis, netG, netCls and netIP (compared against CPU copies taken at construction) |

The failed 960 MiB request equals one FP32 [240,64,128,128] activation (240·64·128·128·4 B). That is a feature
map of the final 128-px Decoder_s residual stage. The full `torch.cuda.memory_summary()` is kept in
the process JSON.

### CUDA memory by phase

| Phase | Allocated | Reserved | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| after model construction + LightCNN load | 229,401,088 | 367,001,600 | 353,616,896 | 367,001,600 |
| after B=240 input allocation (+ Adam, no state yet) | 606,890,496 | 744,488,960 | 606,891,008 | 744,488,960 |
| at OOM (netG forward) | 22,799,887,872 | 24,008,196,096 | 22,800,297,984 | 24,010,293,248 |
| after forward/loss construction | not reached | | | |
| after backward | not reached | | | |
| after optimizer step | not reached | | | |

We did not extrapolate a B=240 requirement. The observed fact is that the forward activations alone,
before the LightCNN path and before any backward, need more than the 23.56 GiB this device can address.
The pinned `train_generator.sh` sets `gpu_ids='0,1,2,3'`: upstream `DataParallel` spread the global batch of 240 over four GPUs.
In this qualified runtime `replica_factor = 1` and there is a single RTX 3090.

### Loss components and semantics at B=240

The OOM came before loss construction, so **no B=240 loss value exists**. The same applies to the epoch-1 total,
the post-warmup total, the gradient inventories, the Adam state and the parameter-change evidence. We did not produce
or fabricate any of them. What M6D5b did establish:

- **Exact graph code.** The additive `methods/dsdg/training_graph.py` transcribes train_generator.py:118-178 statement
  for statement. It calls the pinned `reparameterize`, `kl_loss`, `reconstruction_loss(rec, img, True)/2.0` (per-sample sum,
  then batch mean) and `rgb2gray`. MMD is `50·|mean₀(z_nir) − mean₀(z_vis)|.mean()` over the whole physical batch.
  Orthogonality is `1·|Σ(z_cls·z_nir).mean()|` over the whole batch. `loss_cls = 10·CE(one logit, 0)`. `loss_ip` is
  `1000·(MSE(rec_nir_fc, nir_fc.detach()) + MSE(rec_vis_fc, vis_fc.detach()))/2`, with the reconstructed features not detached.
  `loss_pair = 0.0·MSE(rec_nir_fc, rec_vis_fc)` because lambda_pair = 0.0 under A1 (upstream default 5 not restored).
  The warmup (`epoch < 2`) multiplies every term except loss_rec by 0.01.
- **Focused unit tests.** These use tiny CPU toy nets with the pinned util and the same transcription; there was no
  optimizer and no backward. They confirm: the loss_rec formula (it differs from default MSELoss), the KL assembly, the
  full-batch MMD (the mean of two half-batch values differs), one-logit CE = 0 exactly, the orthogonality formula,
  bilinear-128 + rgb2gray LightCNN preprocessing (align_corners default False), detached target features, a live
  reconstructed-feature gradient path through frozen LightCNN parameters, loss_pair = 0 with a nonzero raw MSE, and
  the epoch-1 and post-warmup formulas.
  This is **not** B=240 qualification evidence.

## Modern `zero_grad()` compatibility (observation only)

The pinned code calls `optimizer.zero_grad()` with no argument. In PyTorch 1.6 that zeroes existing gradients in place.
In the runtime (torch 2.12.1) `Optimizer.zero_grad(set_to_none=True)` is the default, as the process recorded. For the
**first** step both leave every gradient `None`, because freshly constructed parameters have no gradient, so backward
assigns identical fresh gradients. This could **not be shown numerically** at B=240, because the OOM came before line 180.
**M6D5c issue (recorded, not decided):** for later iterations the two semantics differ only if an optimizer-owned tensor
ever receives no gradient. Adam skips a `None` gradient but applies a moment-decay update for a zero-filled one. A
future production runner must either verify full gradient coverage of the 55 tensors on every step or make an explicit,
owner-approved `set_to_none` decision. The frozen config was not changed.

## Possible future resolution categories (NONE adopted, none authoritative)

Listed for owner review only. Each needs its own owner-approved execution-resolution milestone:
(a) a larger-memory single device, or a multi-GPU global-batch DataParallel runtime like the upstream 4-GPU launch;
(b) an owner-approved, *demonstrated* semantics-preserving execution of the batch-dependent MMD/orthogonality terms;
(c) recompute-based activation checkpointing; (d) allocator reconfiguration (the 1.13 GiB of fragmentation is far
smaller than the unreached forward/backward remainder); (e) reduced precision; (f) offload. Categories (b)–(f)
are workarounds the M6D5b brief prohibits. None was tried, and `DSDGAdapter.validate_batch` still refuses 120×2,
60×4 and 1×240.

## Process accounting

| Count (entire milestone) | Value |
|---|---:|
| GPU qualification processes | 1 (process 2 not launched after a clean OOM; nothing fabricated) |
| Optimizer constructions | 1 |
| Optimizer step entries / applications | 0 / 0 |
| Backward passes (Tensor.backward / autograd.backward) | 0 / 0 |
| autograd.grad, torch.save, activation-checkpoint calls | 0 |
| Firewall denials | 0 (LightCNN read opens: 3) |
| Benchmark data / manifest / TEST opens | 0 |

## Tests, preflight, ledger, index

See the audit JSON for the exact test counts. `tests/test_m6d5b_e06c_training_graph.py` (32 tests) passed in
gpat-m6-e06c on the GPU. On the laptop (no Torch) 22 passed and 10 Torch formula tests were skipped.
`tools/m6d5b_e06c_training_preflight.py` is static: no Torch, no CUDA, no model, no optimizer, no data. It validates the
retained STOP state. The full repository suite was not run, and existing tests were not modified.
Ledger: **111 → 112**, one `M6D5B_E06C_TRAINING_GRAPH_STOPPED_OOM` row, first 111 rows byte-identical. The artifact
index was rebuilt last with CRLF line endings.

benchmark_training=false · benchmark_data_access=false · TEST_access=false · checkpoint_created=false ·
synthetic_bank=false · optimizer_constructed=true (diagnostic, 1) · optimizer_applications=0 · backward_passes=0 ·
physical_batch_size=240 · gradient_accumulation_steps=1 · replica_factor=1 · physical_batch_240_qualified=false.

E06c is not described as trained, faithful, native or an official reproduction; it remains CONTROLLED_ADAPTATION.

**NO RECURSIVE FILESYSTEM SEARCH. NO BENCHMARK DATA ACCESS. NO TEST ACCESS. NO BENCHMARK TRAINING. NO SCIENTIFIC
CHECKPOINT. NO SYNTHETIC BANK. NO WORKAROUND. NO COMMIT. NO PUSH.**
