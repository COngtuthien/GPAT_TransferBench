# M6C2b3 — E07c implementation audit

Status: **IMPLEMENTED_NOT_EXECUTED**. Static preparation verified; no model construction, training or benchmark execution.

Starting HEAD: `8535b6458a12bb4f87d86320c728ef9f13c8755d`, branch `m6-baselines`; clean preflight passed.

## Identity and immutable authority

Reporting label: **DiffFAS-BIN-IDFREE (controlled encoder reconstruction)**. Fidelity remains **CONTROLLED_ADAPTATION**, A1 ID-free adaptation plus A3 controlled encoder reconstruction, DEV-021. A6 is a source-traceability correction only. It authorizes no channel projection or architecture substitution.

The M6B YAML and snapshot, frozen A1 adaptation and snapshot, A1/A2/A3 documents, and A6 overlay/document are verified against their committed identities. The effective encoder structure applies only A6's corrected feature dimensions without changing any authority file. All paths and SHA256s are recorded in the companion JSON.

## Source and feature interface

Verified repository `https://github.com/murphytju/DiffFAS`, commit `23f40519ec25a833ebc06842aa6fbab74fad4d15`, required files and Git blob bytes. No source-cache weights were read or restored.

`models/custom_rn.py:367::resnet18` calls BasicBlock `[3,4,6,3]`; `ResNet._forward_impl` at line 329 returns the source-native stages. File SHA256: `fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3`.

Effective HWC interface: **32×32×256, 16×16×512, 8×8×512**, followed by **[B,7]**. `models/unet_autoenc.py:138::BeatGANsAutoencModel.forward` discards the fourth output. The future construction context imports the verified custom module with pretrained=False and changes only fc to Linear(512,7). No torchvision ResNet18 or channel adapter is introduced. Real CPU construction was not run because Torch/torchvision are absent.

`diffusion.py:797::GaussianDiffusion.training_losses` preserves the unpaired `torch.cat([x_t],1)` path, style_spoof condition and epsilon_of_GT target, with no content path in the variational term. `content_live_id` is inert bookkeeping, not an identity pair. Main metadata adapters do not consume subject identity, attack_raw or attack_macro. Macro labels belong only to the auxiliary objective.

## Frozen guides and one auxiliary encoder

Guide selection consumes the existing frozen manifest identity (8838 rows, SHA256 `0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6`) and deterministic raw-byte SHA256 ranking contract. No random.choice, new sampler or manifest regeneration. Eligibility remains m2_complete/TRAIN/SPOOF/same dataset; self-guide allowed. Preparation binds manifest metadata without opening it; future execution must verify bytes before consuming rows. Membership was not independently recounted here.

One static auxiliary plan: **seed42, runs1**, K7 class order live, makeup, mask_2d, mask_3d, partial, print, replay; other_spoof excluded; TRAIN-only total14467. Frozen counts are retained in JSON. No augmentation, balancing or weighted CE. Resize256→ToTensor→Normalize(.5,.5); CE on fourth output; SGD lr.002/momentum.9/weight_decay.005; no scheduler; batch256/drop_last/shuffle/workers6; 200 epochs and final-state selection. The taxonomy/ArtSim interaction disclosure remains in the plan.

Future checkpoint: `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl`, whole `torch.save(model)` module, compatible with the pinned `torch.load(path).cuda()` path. Its SHA is explicitly **RECORDED_AFTER_TRAINING**. No checkpoint exists or is fabricated; future byte verification requires the recorded frozen SHA and does not unpickle the asset.

Exactly three main plans (42/1337/2026) reference that same future auxiliary checkpoint. Three encoder trainings, candidate search, downstream selection and VAL/TEST selection are forbidden.

## Main execution mapping

Each plan binds base/A1/A6 SHA256s, A1/A2/A3/A6 document identities, source commit and required file hashes. Seed hooks replace the official hard-coded seed1 (`FAS_train.py:167::seed_torch`) with the experiment seed across launch-time PYTHONHASHSEED, Python, NumPy, Torch CPU/CUDA and cuDNN benchmark=false/deterministic=true. No globally forced deterministic algorithms or cross-device bitwise guarantee.

Main budget400 epochs, batch4, resolution256, normalization mean/std .5, use_pair=false, channels3, guidance.2, means_size5, var_size3. AdamW lr1e-5; cycle scheduler lr1e-5/n_iter2400000/warmup5000/decay[linear,flat]; EMA .9999, before_warmup0. Linear1000-step diffusion, beta1e-4→2e-2, EPSILON, LEARNED_RANGE.

DDIM skip10, initial_noise250, effective25 steps, cond_scale2.0. `FAS_sample.py:32::main` loads checkpoint["model"], never EMA for authoritative generation. `diffusion.py:47::ddim_steps` initializes noise through q_sample of the generation target; this is distinct from the inert training-content placeholder. No sampling performed.

BASELINE_FINAL_STATE_V1 selects the terminal state at completion of epoch400, within seed; periodic cadence10000 iterations. If cadence misses the end, save one terminal checkpoint with **zero extra optimizer steps**. VAL diagnostics only; TEST never selects. Common learned.py gains only the E07c checkpoint branch and reuses existing learned logging; earlier methods remain unchanged.

## Validation and access evidence

- E07c: 46 tests, **45 pass / 1 skip**, zero failures/errors. Skip: PyTorch/torchvision unavailable for real pinned custom_rn CPU construction; static source/interface tests pass.
- Static preflight: **PASS**, one auxiliary plan, three main plans, same future auxiliary identity. No runtime writes, inference, training, sampling or checkpoint creation.
- M6B: **PASS, 0 failures**.
- M6A8: **13/13**; M6C2b2: **29 pass / 1 Torch skip**; M6A7: **13/13**.
- M6C2b1: **49 pass / 1 Sim3DR_Cython skip**; M6A6: **4/4**; M6C2a: **72/72**; M6C1: **83/83**.
- Historical M0: **16/17**, only `test_manifests_are_stage_appropriate` fails on `manifests/artifact_probe_classes_v1.json`; historical failure unchanged.
- Full repository test suite: **NOT RUN**.

Audited E07c tests: 1053 open events, **0 benchmark-image opens, 0 manifest-image opens, 0 manifest opens**. Static preflight: 639 events, **0 benchmark-image opens, 0 manifest-image opens, 0 manifest opens**. Counts apply to those named processes. Canonical indexing and historical M0 inspect metadata; no result from those checks is benchmark evidence.

Observed development environment only: Python3.12.3, NumPy2.5.3, Pillow12.3.0, opencv-python-headless5.0.0.93. Torch, torchvision, SciPy and tensorfn unavailable. No dependencies installed; these versions are not final execution pins. Compatible dependencies, future runner integration and the one trained/hashed/frozen encoder remain required. No fake model execution claimed.

## Artifacts and ledger

New implementation: methods/difffas/{__init__,contract,source,seed_adapter,encoder,sampler,adapter}.py and source_traceability.md; three tests/test_m6c2b3_*.py modules; tools/m6c2b3_preflight.py; this audit and companion JSON. Modified support: methods/common/learned.py, E07c current row in configs/CONFIG_STATUS.md, ledger and canonical index. All other current method rows remain unchanged.

Append exactly one M6C2B3_E07C_IMPLEMENTATION record: 93 committed rows preserved byte-for-byte, final94. Current artifact SHA256s and committed-prefix SHA are in JSON. The canonical index excludes itself and the append-only ledger; it is rebuilt at finalization with unique paths, current sizes/hashes and CRLF.

No training, auxiliary/main training, benchmark image execution, diffusion sampling, synthetic bank, TEST data for method execution, GPU job, scientific checkpoint, frozen-config change, source modification, other baseline implementation, commit or push.
