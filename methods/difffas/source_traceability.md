# E07c — controlled encoder reconstruction

Identity: **DiffFAS-BIN-IDFREE (controlled encoder reconstruction)**,
`CONTROLLED_ADAPTATION`, `DEV-021`, with A1 ID-free and A3 encoder-objective
provenance. A6 corrects feature descriptions only; no channel adapter is added.

Authority: immutable M6B YAML/snapshot, A1 adaptation config/snapshot, A1/A2/A3
documents and A6 overlay/document. Effective feature shapes are returned in a
separate structure; the base config is never mutated. A6 shapes (HWC) are
32×32×256, 16×16×512, 8×8×512, followed by the discarded B×7 head.

Pinned source: `https://github.com/murphytju/DiffFAS` at
`23f40519ec25a833ebc06842aa6fbab74fad4d15`. Common source verification checks
repository, commit, tree, regular files, Git blobs and cited size/SHA256 values.
Source-cache weights are neither read nor restored.

| Contract | Exact source basis |
| --- | --- |
| Main training | `FAS_train.py:train/main`: frozen config values are translated into typed settings; unmodified `main` is not invoked |
| Seed correction | `FAS_train.py:seed_torch/main`: replace hard-coded seed 1 with the current main experiment seed across Python, launch-time PYTHONHASHSEED, NumPy, Torch CPU/CUDA and cuDNN |
| A1 unpaired path | `diffusion.py:GaussianDiffusion.training_losses`: `torch.cat([x_t],1)`, `x_cond=target_pose`, epsilon target from GT noise |
| Diffusion | `diffusion.py:create_gaussian_diffusion/make_beta_schedule`; `config/diffusion.conf` and `config/diffconfig.py` |
| Encoder architecture | `models/custom_rn.py:resnet18/ResNet`: BasicBlock `[3,4,6,3]`, A6 source-native channels; no torchvision replacement |
| Consumed features | `models/unet_autoenc.py:BeatGANsAutoencModel.encode/forward`: first three tensors consumed, fourth discarded |
| Auxiliary training | `models/pretrain_classifier.py`: fourth-output CE, SGD, loader and final whole-module save; this script is never imported because its top level trains |
| Whole-module load | `models/unet_autoenc.py:BeatGANsAutoencModel.encoder`: `torch.load(path)` then `.cuda()` |
| DDIM generation | `FAS_sample.py:main` loads `["model"]`; `diffusion.py:ddim_steps`; EMA is not substituted |

Training `content` is an inert API placeholder in the official unpaired loss:
it is not concatenated, conditioned on, paired by identity, or used as the target.
The metadata adapter extracts only recorded GT/guide IDs and binary provenance.
It never inspects subject IDs or attack labels and never samples another guide.
Frozen manifest identity, 8838 rows, SHA256 and raw-byte SHA256 ranking semantics
are bound from A1. Preparation does not open/decode the manifest. The future
runner must verify its bytes before consuming rows. No new guide algorithm is
implemented. `content_live_id` is not an identity reconstruction pairing.

The training-content rule does not remove the generation target live image:
the frozen official sampler initializes its noisy state through `q_sample` of
that generation target. This is described explicitly in the sampling plan; no
sampling is performed here.

The auxiliary objective is the unchanged A3 K7 TRAIN-only class order and counts.
Attack-macro labels apply only to auxiliary classifier pretraining, never to the
main diffusion target. The A3 disclosure about the taxonomy also being used by
ArtSim remains in each auxiliary plan. Class metadata path/hash are bound without
reading it; future data preparation must verify those bytes and class membership.

There is exactly one future auxiliary run at seed 42, 200 epochs, final-state
whole-module `encoder_final.pkl`. The explicit future `encoder_model` context
imports only the verified `custom_rn` module, constructs its CPU model with no
pretrained download, and replaces only `fc` with the A3 K7 head. Keeping this
context alive preserves the legacy `custom_rn.ResNet` pickle identity for a future
whole-module save. This task does not save or deserialize any checkpoint.

All three main plans (42/1337/2026) reference the same auxiliary path and
`RECORDED_AFTER_TRAINING` SHA state. Future consumption requires a frozen SHA and
byte verification; the helper hashes bytes without unpickling. No encoder per
main seed, candidate search, VAL selection, or TEST use is permitted.

Main plans retain 400 epochs, batch 4, AdamW lr=1e-5, cycle scheduler
lr=1e-5/n_iter=2400000/warmup=5000/decay=[linear,flat], EMA .9999 (0 before
warmup), 1000-step linear epsilon/learned-range diffusion, guidance .2,
means_size=5 and var_size=3. DDIM uses skip10, initial_noise250, 25 steps and
cond_scale2.0. Model tensor selection is explicit.

The existing common learned checkpoint/logging interface gains only E07c's
completed-epoch final-state rule: save at the end of epoch 400, with zero extra
optimizer steps if the 10000-iteration cadence misses the terminal state.
VAL is diagnostic only; TEST cannot select anything.

This milestone supplies static plans, source/contract validation, metadata
adapters and explicit future construction/seeding hooks. It does not execute an
auxiliary/main training runner, forward pass, sampling, GPU probe or checkpoint
write. Compatible Torch/torchvision/SciPy/tensorfn dependencies, a trained frozen
encoder and future runner integration remain required before execution.
