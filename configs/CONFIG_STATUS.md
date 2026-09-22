# Config file status (spec §23)

Policy: a config file is only created when every scientific value can be copied from the
frozen spec. No placeholder configs with invented defaults exist in `configs/`. Missing files
below are intentionally absent until their milestone.

| Config | Spec basis | Status | Blocking inputs |
|---|---|---|---|
| `configs/frozen/data_v1.yaml` | §3.2–3.4, §4, §23 | **CREATED M1** (spec values + M1 interpretations DEV-005/006, laptop root bindings); snapshotted | — (its SCRFD variant field is superseded by `preprocess_v1.yaml`, where Q-02 is resolved) |
| `configs/frozen/attack_map_v1.yaml` | §3.3 | **CREATED M1** — status PARTIAL_PENDING_OWNER_APPROVAL (MSU 3 + SiW 13 tokens mapped; CASIA 10 + SiW `Paper` pending) | Q-12, Q-13 |
| `configs/frozen/preprocess_v1.yaml` | §4, App. A | **CREATED M2A — FROZEN 2026-09-19** (owner decisions Q-02/03/18/19/22/23; promoted from `configs/proposed/preprocess_v1.proposed.yaml`, which is kept as history); snapshotted | none — no M2B-blocking field is null. `facexformer.parsing_class_names` stays null (Q-21, blocks M9 only) |
| `configs/frozen/split_v1.yaml` | §3.5 | NOT CREATED (M3) | allocator penalty weights/procedure (Q-01) |
| `configs/frozen/pairs_v1.yaml` | §6 | NOT CREATED (M4) | — (values fully in spec: 64 candidates; weights 0.50/0.30/0.20; seed 20260814) |
| `configs/frozen/downstream_resnet18.yaml` | §23.1 | **CREATED — spec verbatim** (extracted programmatically from DOCX) | — |
| `configs/frozen/downstream_dinov3_vits16.yaml` | §14.2 | NOT CREATED (M11) | Q-06 |
| `configs/frozen/artifact_probe.yaml` | §12.1 | **CREATED M5 pre-flight — FROZEN 2026-09-20** (owner resolutions D-M5-01…D-M5-08, Q-07, E-M5-01; promoted from `configs/proposed/artifact_probe.proposed.yaml`, which is kept as history); snapshotted | none — 0 unresolved execution-affecting fields. Execution still requires the GPU host preflight (E-M5-01 resolved to GPU_REQUIRED) |
| `configs/methods/e01_fas_aug.yaml` | §8.1 | **IMPLEMENTED_NOT_EXECUTED (M6C1)** — frozen YAML and snapshot present; recipe and official backend implemented; official synthetic pixel smoke PASS; **NOT BENCHMARK-EXECUTED** | No blocker. **E01-3** resolved M6A3b (A2-01): official asset directories enumerated in ascending lexicographic (bytewise) order, agreeing with git-tree order; runtime `os.listdir` order must not affect results. Operator set, magnitude distribution (`level=0` retained), asset contents and per-pair seed unchanged. **OBS-1** resolved M6A3b (A2-05): `global_seed` = current experiment seed `{42,1337,2026}`; preimage `UTF-8(pair_id + decimal_string(global_seed))` with **no separator**; `operator_seed = unsigned_big_endian(SHA256) mod 2^31` (`PTR000001`+`42` → `PTR00000142` → `795981663`). Also resolved: E01-1, E01-2 (M6A2). Source pinned `RizhaoCai/FAS-Aug@0da1dd79` |
| `configs/methods/e02_freqsub.yaml` | §8.2 | **IMPLEMENTED_NOT_EXECUTED (M6C1)** — frozen YAML and snapshot present; synthetic transform verified; **NOT BENCHMARK-EXECUTED** | No blocker. **E02-1/2/3** resolved M6A3b (A2-02/03/04). Rounding = **round-half-up** frozen as a rule: `k = (n_eligible + 2) // 4` → `k = 40` for `n_eligible = 159`. Selection = row-major eligible enumeration then `numpy.random.Generator(numpy.random.PCG64(pair_seed)).choice(n, k, replace=False)`; the **selected set** is the result, indices sorted ascending before substitution; `default_rng` without naming PCG64, `random.sample` and hash ranking are prohibited; the runtime freeze must pin NumPy. Seed = `unsigned_big_endian(SHA256(UTF-8("gpatbench.freqsub.block.v1\\\\\|" + pair_id + "\\\\\|" + decimal_string(global_seed))))`, full 32-byte digest, no truncation/modulo. E02-4 resolved M6A2 |
| `configs/methods/e03_stdn.yaml` | §8.3 | **IMPLEMENTED_NOT_EXECUTED (M6C2a)** — fidelity `FAITHFUL_OFFICIAL`; frozen config/snapshot unchanged; static adapter verified; **NOT TRAINED** | No blocker. **E03-2r** resolved M6A2 (FaceXFormer landmarks trained on 300-W = iBUG-68; STDN's `lm_reverse_list` is the canonical iBUG-68 flip permutation; flip NOT disabled). Also resolved: E03-1 (`BATCH_SIZE=2`), E03-2, seed propagation and serialized TF map hooks implemented (M6C2a); TF1 execution environment still required, checkpoint → `SOURCE_RESOLVED` (final/latest `ckpt-50`), which `BASELINE_FINAL_STATE_V1` does **not** override (A2-06 clause 1). Source pinned `yaojieliu/ECCV20-STDN@c79f1f8c` |
| `configs/methods/e04_physics_std.yaml` | §8.4 | **IMPLEMENTED_NOT_EXECUTED (M6C2b1)** — controlled geometry/depth adapter and static training plans; frozen base + A4 overlay; **NOT TRAINED** | **E04-2 RESOLVED M6A5b by OWNER DECISION option B — CONTROLLED RECONSTRUCTION (Amendment A3 §4).** `FAITHFUL_PAPER` is **no longer claimed**. Unchanged from the paper: `L_depth` **retained**, `α₁=100` and all coefficients, `K=32`, Eq. 21 L1 form, 150,000-iteration budget. **Paper-derived, not substituted:** Eq. 21 states the depth ground truth for a live face contains face-like shape and **for spoof must be zero**, so spoof `M₀` = all-zeros 32×32 and the reconstruction affects **live samples only**. **Geometry substitute:** `cleardusk/3DDFA_V2@1b6c6760` with `bfm_noneck_v3.pkl` (SHA256 `89ac9648…`, 24,393,598 B) — a **BFM2009 derivative**, i.e. the same model family the paper's `[61]` requires; fits a 3DMM (40-d shape + 10-d expression + weak-perspective pose) to a dense 38,365-vertex mesh; **Licence, factual:** the repo carries an MIT licence file and its `bfm/readme.md` **states academic-use terms** for the model asset; the external bytes are **not committed to or redistributed through** this repository; use must comply with the applicable upstream terms; **no licence gate was bypassed**; no independent legal interpretation is offered. **Fixed auxiliary assets:** the pinned commit, BFM-derived asset, 3DDFA checkpoint, Q=140 list and depth renderer are **reused unchanged for every E04 seed** (42/1337/2026) and are never retrained or re-derived per seed. MediaPipe was rejected on closeness grounds (landmark regressor, **no shape basis**), not convenience. **Q=140 substitute:** deterministic — 68 model-native iBUG anchor vertices + 72 farthest-point-sampling picks inside their convex hull on the canonical neutral mesh; ordered list SHA256 `1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f`, image-independent, reproducer `tools/m6a5_e04_q140_derive.py`. **Depth `M₀`:** official 3DDFA_V2 semantics (z-buffer, per-sample min–max → [0,1], background 0) in the canonical 256 frame, then uint8 INTER_AREA resize to 32×32, float32 cast, division by 255 and clip (A4). A4 Adam semantics use the pinned STDN predecessor; the optimizer choice is benchmark-defined. Assets live **outside** the repo, untracked. Provenance `outputs/audit/M6A5_TDDFA_ASSET_PROVENANCE.json`. **E04-3** remains resolved by `BASELINE_FINAL_STATE_V1` (terminal @ 150,000). α₀ non-blocking (scoring-only). **M6C2b1:** source/assets/Q140 verified; official C++ rasterizer synthetic tests pass; production Cython and framework execution environments remain pending. |
| `configs/methods/e05_pcgan.yaml` | §8.5 | **IMPLEMENTED_NOT_EXECUTED (M6C2b2)** — fidelity **`CONTROLLED_ADAPTATION`**; frozen base unchanged; A5 additive blur contract; **NOT TRAINED** | **Blur-operator gap resolved M6A7 (A5):** exact nonoverlapping 2×2 average pooling, stride 2, no padding, on both loss branches; benchmark-defined controlled reconstruction. M6C2b2 static adapter, pinned architecture mapping, A5 pooling hook, seed plans and checkpoint metadata implemented; no model/training execution. PyTorch execution environment and future runner integration remain pending. **E05-2** resolved M6A3b (A2-07): the pinned `[34]` `taesungp/swapping-autoencoder-pytorch@6baa180f` is the executable architecture basis, followed as it actually executes including its **StyleGAN2-style modulation/demodulation**; StyleGAN-v1 AdaIN is not separately ported for the ambiguous prose citation `[35]`. Fidelity is **`CONTROLLED_ADAPTATION`** and **must be disclosed**; the row must **not** be called "native PCGAN" or "faithful PCGAN". Method ID and result row remain **E05**. **E05-3** resolved by `BASELINE_FINAL_STATE_V1` (terminal @ iteration 4,000). **E05-1** resolved M6A2 (at 256: `sp=1`, `z_pat = 8×128×128`, blur ÷2) |
| `configs/methods/dsdg_binary.yaml` | §8.6 | NOT CREATED (M6) | superseded for Track A by E06c (Amendment A1) |
| **E06c** DSDG-BIN-IDFREE | §8.6 + A1 | **IMPLEMENTED_NOT_EXECUTED (M6C2a)** `configs/methods/e06c_dsdg_bin_idfree.yaml` — fidelity **`CONTROLLED_ADAPTATION`**; snapshotted; **NOT TRAINED** — fidelity **`CONTROLLED_ADAPTATION`** (provenance: `AMENDMENT_A1_IDFREE_ADAPTATION`; **not** native or faithful DSDG — A1 explicitly adapts DSDG to BIN-IDFREE semantics, DEV-020); adaptation semantics frozen in `configs/frozen/dsdg_bin_idfree_v1.yaml` | No blocker. **E06c-2** resolved M6A4 → `RESOLVED_EXTERNAL_WEIGHT_ACQUIRED_AND_PINNED`: the **original author's** LightCNN-29 v2 checkpoint (Google Drive id `1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS`, the **same id linked by BOTH** the upstream `AlfredXiangWu/LightCNN` README and the DSDG README, so no mirror substituted), **123,844,849 B**, SHA256 `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964` (observed pin — upstream publishes none). Stored outside the repo and outside `source_cache`, untracked. Inspected with a **restricted non-executing unpickler**; **60/60** `define_IP()` parameters matched with exact shapes, 0 missing, 0 mismatches, 1 unexpected (`module.fc2.weight [80013,256]`, dropped by the documented non-strict filter). **Never loaded for inference.** Resolved earlier: E06c-1 (`lr` 2e-4), seed/worker hooks and ID-free pair adapter implemented; effective batch 240 enforced; compatible PyTorch/CUDA environment still required, checkpoint `SOURCE_RESOLVED` (`netG` epoch 200), not overridden by `BASELINE_FINAL_STATE_V1` |
| `configs/methods/dsdg_native.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_binary.yaml` | §8.7 | NOT CREATED (M6) | superseded for Track A by E07c (Amendment A1) |
| **E07c** DIFFFAS-BIN-IDFREE | §8.7 + A1 + A3 + A6 | **CONTRACT_RESOLVED_NOT_IMPLEMENTED (M6A8)** `configs/methods/e07c_difffas_bin_idfree.yaml` — fidelity **`CONTROLLED_ADAPTATION`**; snapshotted; **NOT TRAINED** — fidelity **`CONTROLLED_ADAPTATION`** (provenance: `AMENDMENT_A1_IDFREE_ADAPTATION` **+** `AMENDMENT_A3_CONTROLLED_ENCODER_RECONSTRUCTION`); adaptation semantics in `configs/frozen/difffas_bin_idfree_v1.yaml` | **E07c-4 RESOLVED M6A5b by OWNER DECISION option B — CONTROLLED RECONSTRUCTION (Amendment A3 §5).** The encoder is **not removed, not disabled, not replaced** by a torchvision/ImageNet ResNet18. **Architecture NOT substituted:** the repo's own `custom_rn.resnet18` (a `[3,4,6,3]` **ResNet-34 topology despite its name**); **A6 source-traceability correction:** pinned interface `x32x32` 32×32×256, `x16x16` 16×16×512, `x8x8` 8×8×512, `embg` (B,7); prior first-two channel descriptions were erroneous. No projection or source modification authorized. M6C2b3 implementation has not resumed. **Containment fact:** `unet_autoenc` does `x32x32,x16x16,x8x8,_ = self.encode(...)` — the 4th output is **discarded** — so 17→7 classes changes **only the discarded head** and the consumed interface is shape-identical. **Substitute objective:** K=7 attack-macro classification over the frozen taxonomy (`manifests/artifact_probe_classes_v1.json`: live, makeup, mask_2d, mask_3d, partial, print, replay), **TRAIN split only**; all 7 classes non-empty (live 5629, print 2838, replay 2178, partial 1911, mask_3d 1056, makeup 759, mask_2d 96; total 14,467). **Official training values used verbatim:** 256×256 resize, `Normalize([0.5]*3)`, no augmentation, **no class balancing** (official has none), `CrossEntropyLoss`, SGD lr 2e-3/mom 0.9/wd 5e-3, no scheduler, batch 256, 200 epochs, final-state checkpoint, whole-module `torch.save`. **Benchmark-defined additions: exactly four** — K=7 label space, TRAIN data source, **`auxiliary_encoder_training_seed = 42`** (official sets none), and **`auxiliary_encoder_training_runs = 1`** (no candidate search; VAL diagnostics only; TEST never). **Auxiliary seed ≠ experiment seed:** the encoder is trained **exactly once** at seed 42, checkpointed by the frozen final-state rule, SHA256-recorded and frozen; **that same frozen encoder is reused for all three E07c main seeds 42/1337/2026**, which mirrors the unavailable original `PADISI.pkl` as one fixed external pretrained asset. **Three different conditioning encoders must NOT be trained**; only the DiffFAS main run seed varies. Encoder frozen and SHA256-recorded **before** DiffFAS consumes it. **Disclosed:** this taxonomy is also the ArtSim probe's label space — separate models, no shared weights, no TEST, so not leakage, but it could favour **E07c** on ArtSim. **E07c-5** remains under `BASELINE_FINAL_STATE_V1`; E07c-3 resolved M6A2 (DDIM, `DDIM_skip=10`, 25 steps, `cond_scale=2.0`) |
| `configs/methods/difffas_native.yaml` | §8.7 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/gpat_b0.yaml` | §23.2 | **CREATED — spec verbatim** (extracted programmatically from DOCX) | DEV-003 `lambda_dir` must be resolved before M7 |
| `configs/methods/gpat_b1.yaml` | §5.3, §10.5 | NOT CREATED (M7) | B0 + λ_type 0.2 + warmup; key naming for new fields not in spec |
| `configs/methods/gpat_b2.yaml` | §5.3, §9.2 | NOT CREATED (M7) | Q-05 |
| `configs/methods/gpat_b3.yaml` | §5.3 | NOT CREATED (M7) | Q-05 |

**M6A1 (2026-09-21) — baseline blocker status.** The Track-A rows above carry the blocker IDs from
`outputs/audit/M6_BASELINE_SOURCE_GAPS.json`. **Zero M6 methods are READY**, so no
`configs/methods/*.yaml` was created for them. No frozen config was changed by M6A1. Seeds: the
frozen spec mandates `[42, 1337, 2026]` and marks training seeds **not** changeable, which takes
precedence over the official hard-coded (E07c) or absent (E03, E06c) seeds; the propagating adapter
is **not** implemented in M6A1. E03 and E04 keep **distinct** method IDs and result rows despite their
disclosed publication lineage (`E03_E04_LINEAGE_OVERLAP`). **Checkpoint selection** is
`NOT_APPLICABLE` for E01/E02, `SOURCE_RESOLVED` for E03 and E06c, and a **blocking**
`CHECKPOINT_SELECTION_UNSPECIFIED` gap for E04, E05 and E07c (unchanged by M6A2) — the frozen spec fixes a generator
selection rule only for GPAT (§10.6) and for downstream evaluators (§14), never for baselines, and
**TEST is never used for checkpoint selection**. No checkpoint was chosen in M6A1.

**M6A2 (2026-09-21) — source-derivable blocker resolution.** Blocking gaps **17 → 11**. Resolved:
E01-1, E01-2 (`RESOLVED_BY_SPEC`), E02-4 (`RESOLVED_BY_SPEC`), E03-2r, E05-1
(`RESOLVED_BY_PINNED_UPSTREAM`), E07c-3 (`RESOLVED_BY_OFFICIAL_EXECUTION_PATH`). **E03 has no
remaining source-derivable blockers.** E06c-2 is `STILL_BLOCKED_REQUIRED_WEIGHT_NOT_FETCHED` — its
scientific identity is fully resolved (the DSDG README links the *same* Google Drive id as the
upstream LightCNN author's release) and only the bytes are absent. No method config was created and
no owner choice was frozen; the 11 residual decision surfaces are in
`outputs/audit/M6A2_BLOCKER_STATE.json`.

**M6A3b (2026-09-21) — owner-approved baseline execution decisions.** Blocking gaps **11 → 3**.
The owner explicitly approved the decisions recorded in
`outputs/audit/M6A3_OWNER_DECISIONS.md` / `.json` and `outputs/audit/M6A3_BLOCKER_STATE.json`, and
they are frozen additively in
`docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md`
(**Amendment A2**, OWNER-APPROVED). The frozen DOCX and Amendment A1 were **not** edited, and the
M6A3a decision packet, M6A1 and M6A2 artifacts were **not** rewritten. **Closed by owner contract
(8):** E01-3, E02-1, E02-2, E02-3, E04-3, E05-2, E05-3, E07c-5 — plus OBS-1, which was never counted
among the 11. **Still active (3):** **E04-2** (source/asset/scientific-route gap), **E06c-2**
(acquisition authorized, official LightCNN bytes not yet fetched/hash-pinned), **E07c-4** (official
PADISI encoder/source gap). **Zero M6 baselines are READY in the executable sense** and no
`configs/methods/*.yaml` was created — M6B has not started — but **E01, E02, E03 and E05 now have no
remaining contract blocker**, while **E04, E06c and E07c remain BLOCKED**. A new global checkpoint
policy **`BASELINE_FINAL_STATE_V1`** (A2-06) applies to exactly **E04-3, E05-3 and E07c-5**: use the
model state at the END of the frozen training budget, create a terminal checkpoint with **zero
additional optimizer steps** if periodic saving misses it, allow VAL only as diagnostics, never let
VAL select the checkpoint for governed methods, and **never** use TEST for checkpoint selection. It
**does not override** E03's `ckpt-50` or E06c's epoch-200 official rules. **E05 fidelity is
`CONTROLLED_ADAPTATION`** and must be disclosed in the final report. These decisions exist for
deterministic reproducibility, source-faithful execution where recoverable, no extra VAL-based
checkpoint tuning, no TEST-based selection and explicit disclosure of unavoidable adaptation — they
are not a weakening of any baseline. No training, no bank generation, no TEST access, no model
execution, no checkpoint and no weight download occurred in M6A3b.

**M6A4 (2026-09-21) — external asset and source resolution.** Blocking gaps **3 → 2**. No M6A3
owner decision was revisited. **RESOLVED: E06c-2** → `RESOLVED_EXTERNAL_WEIGHT_ACQUIRED_AND_PINNED`
— the owner-authorized (A2-08) acquisition of the **original author's** LightCNN-29 v2 checkpoint
succeeded (123,844,849 B, SHA256 `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964`),
was inspected **without executing anything** and statically verified **60/60 compatible** with
`define_IP()` (0 missing, 0 shape mismatches). The weight lives **outside the repository**, is
**untracked**, and was **never loaded for inference**; its provenance is
`outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json`. **STILL ACTIVE (2): E04-2** — all four
sub-gaps (BFM licence, Q=140 indices, depth-rendering of `M₀`, executable route) survived a search of
the **published TPAMI 2022** text, the official project page and the author's repositories; **E07c-4**
— the repository owner publicly acknowledged in 2024 that the pretrained encoder was missing and
never delivered it. **E01, E02, E03, E05 and E06c now have no remaining blocker; E04 and E07c remain
BLOCKED**, and **no `configs/methods/*.yaml` was created** because M6B has not started. `E04-2` and
`E07c-4` are **not** resolved, and no claim of successful E04/E07c reproduction is made. M6A4 did no
training, no optimizer steps, no bank generation, no TEST access, no model inference, and downloaded
**no** E04 (BFM) or E07c (PADISI) asset. Details: `outputs/audit/M6A4_EXTERNAL_ASSET_RESOLUTION.md`,
`.json`, `outputs/audit/M6A4_BLOCKER_STATE.json`.

**M6A5b (2026-09-21) — owner-approved controlled reconstruction of E04 and E07c.** Active M6
contract/source blockers **2 → 0**. The owner changed direction: **both** blocked rows become
executable via controlled reconstruction rather than being retained as N/A (the M6A5a Option-A
recommendation was **advisory only** and is superseded). Frozen additively in **Amendment A3**; the frozen DOCX, A1 and A2 are **unchanged**. **Authority vs
derivation:** the owner approved the **direction** (controlled reconstruction for both rows, as close to
the papers and official code as reasonably possible); every concrete technical value was **derived by the
benchmark audit** under that authorization — contract class
`BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION_UNDER_OWNER_AUTHORIZATION`. The owner did not independently
specify each value, and **none of these reconstruction details was specified by the original E04 or E07c
authors**. **E04-2** and **E07c-4** are both
`RESOLVED_BY_OWNER_CONTROLLED_RECONSTRUCTION`. **All seven Track-A rows (E01, E02, E03, E04, E05,
E06c, E07c) are now contract-ready for M6B**, but **no training has occurred** and **no
`configs/methods/*.yaml` exists** — M6B has not started. **Fidelity taxonomy (normalized):** `E01` = `FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION`,
`E02` = `SPEC_DEFINED`, `E03` = `FAITHFUL_OFFICIAL`, and **four** rows carry
**`CONTROLLED_ADAPTATION`** — **E04** (A3 geometry/depth reconstruction), **E05** (A2-07 architecture
resolution), **E06c** (A1 identity-free adaptation) and **E07c** (A1 identity-free adaptation **+** A3
controlled encoder reconstruction). E01 is **not** a controlled scientific adaptation and E02 is **not**
an adaptation of an external method; provenance strings such as "A1 adaptation semantics" are
provenance, **never** a fidelity class. These must be
reported as **controlled reconstructions, not exact reproductions** of the original authors'
unreleased pipelines, and must never be collapsed into "native baseline"; the wording *native*,
*official reproduction* and *faithful reproduction* is forbidden for them. Result tables must
distinguish source-faithful executable baselines from controlled reconstructions. One new
authoritative source was pinned (`cleardusk/3DDFA_V2@1b6c6760`); its model bytes live **outside** the
repository, untracked, and never in `source_cache`. Licence position is recorded **factually** (MIT
licence file for the code; `bfm/readme.md` states academic-use terms for the model asset; nothing
redistributed; upstream terms apply; no gate bypassed; no legal interpretation offered). No baseline
or encoder was trained, no bank generated, no TEST accessed, no benchmark data processed, and no
downstream model selection performed in M6A5b. Details: `outputs/audit/M6A5_OWNER_RECONSTRUCTION_DECISIONS.md`
/`.json`, `outputs/audit/M6A5_CONTROLLED_RECONSTRUCTION_CONTRACT.md`,
`outputs/audit/M6A5_BLOCKER_STATE.json`, `outputs/audit/M6A5_TDDFA_ASSET_PROVENANCE.json`,
`outputs/audit/M6A5_E04_Q140_VERTEX_SET.json`.

**M6B (2026-09-21) — baseline execution configs and run-logging contract FROZEN.** All seven Track-A
rows are now **`CONFIG_FROZEN`**: `e01_fas_aug`, `e02_freqsub`, `e03_stdn`, `e04_physics_std`,
`e05_pcgan`, `e06c_dsdg_bin_idfree`, `e07c_difffas_bin_idfree` under `configs/methods/`, each with a
byte-identical `frozen_config_snapshot/` copy. **NOT TRAINED and NOT COMPLETE** — M6B froze configs
and observability only: nothing was trained or generated, no model was executed, no checkpoint was
created, no GPU job ran and TEST was never touched. Every execution-affecting value is explicit;
nothing depends on cwd, `os.listdir` order, hostname, shell environment, `PYTHONHASHSEED`, automatic
checkpoint choice, TEST performance or a "best seed", and **nothing was retuned** (each config carries
a `field_provenance` block tagging `SPEC` / `PAPER` / `OFFICIAL_CODE` / `AMENDMENT_A1|A2|A3` /
`COMPUTED` / `M6B`). **Seeds:** every method runs `[42, 1337, 2026]` — E01/E02 are non-learned but
stochastic, since their per-pair seed derives from `global_seed`. **Primary result = mean ± std over
all three seeds** with per-seed values reported; `max`-over-seeds, best-seed, dropping a seed,
rerunning only a poor seed and TEST-based seed selection are all **FORBIDDEN**; a best seed may be used
only for an explicitly-labelled qualitative figure. **Checkpoint selection is WITHIN seed**: E03 keeps
official `ckpt-50` and E06c official generator epoch-200 (A2-06 clause 1 does **not** override them);
E04/E05/E07c use `BASELINE_FINAL_STATE_V1` at iteration 150,000 / iteration 4,000 / end of the
400-epoch budget. No VAL-based selection was invented and TEST never checkpoint-selects.
**E07c: one auxiliary encoder** — `auxiliary_encoder_training_seed = 42`,
`auxiliary_encoder_training_runs = 1`, trained exactly once (later), then the *same* frozen checkpoint
is reused for main seeds 42/1337/2026; three separately pretrained encoders are forbidden. **E04: fixed
auxiliary geometry** — the 3DDFA_V2 commit, model assets, Q=140 list and depth renderer are reused
unchanged across all three seeds and never re-derived per seed. The run-logging contract
`configs/run_logging_v1.yaml` (`run_logging_v1`) mandates full trajectory retention (append-only
`metrics.jsonl`, `keep_only_final_epoch` FORBIDDEN, resume appends or reconciles), a per-checkpoint
`checkpoint_index.json` with SHA256 and selection reason, and a `run_summary.json`; run outputs and
checkpoint bytes live under `<runtime_root>/runs/m6/<method_id>/seed_<seed>/` and are **never**
committed. `tools/m6b_validate_configs.py` (YAML-only; executes no model, reads no benchmark data)
passes with **0 failures**. Details: `outputs/audit/M6B_CONFIG_FREEZE.md` / `.json`,
`outputs/audit/M6B_LOGGING_CONTRACT.md`.

**M6C1 (2026-09-22 handoff continuation) — IMPLEMENTED, DRY-RUN VERIFIED,
NOT BENCHMARK-EXECUTED.** E01 and E02 are `IMPLEMENTED_NOT_EXECUTED`; E03–E07c
remain `CONFIG_FROZEN`. The existing recipe/seeding and E02 mathematics were retained.
E01 now binds unchanged pinned upstream pixel-operator bodies through
`OfficialFASAugBackend`; source commit/tree, cited source SHA256s, asset blobs/counts,
and operator names/ranges are checked. Toy use requires explicit synthetic-only mode.
The official synthetic pixel smoke is **PASS** after installing Pillow 12.3.0 in the
development .venv; all eight operators, deterministic repeats and level-zero no-ops pass. The CPU synthetic dry run passes, including
E02 determinism/Hermitian checks and append-only resume. Focused tests: 83/83 pass,
0 skipped. M6B validator: PASS, 0 failures. NumPy 2.5.3 is observed,
not yet an authoritative environment pin. Python 3.12.3/Pillow 12.3.0/NumPy 2.5.3
are the synthetic smoke environment only, not the final frozen execution environment.
Official-only smoke audit: 0 benchmark image opens, 0 manifest opens.

**Data-access disclosure:** the earlier 266-open audit applies only to that dry run.
The interrupted agent subsequently launched an inadvertent full regression command;
its benchmark-data access is **UNVERIFIED**. Its result is not benchmark evidence.
This continuation did not rerun the full suite. Its audited dry run recorded 454 opens,
0 benchmark image/video opens and 0 manifest opens. No learned training, bank, checkpoint,
GPU job or frozen-config change was performed by this continuation. Details:
`outputs/audit/M6C1_IMPLEMENTATION.md` / `.json`.

Verbatim copies of created configs are in `frozen_config_snapshot/` (spec §0.1 rule 1);
SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`. Tests assert the two stay byte-identical.
`configs/data_source_registry.yaml` records the selected dataset sources (M1 final); it is a registry, not a frozen config.
`configs/proposed/` holds superseded proposals kept as history; they are never edited after promotion.


**M6C2a (2026-09-22) — common learned support + E03/E06c IMPLEMENTED_NOT_EXECUTED.**
Source/asset verifiers, auditable frozen-config training plans, framework/worker seeding hooks,
learned logging and checkpoint metadata helpers are implemented additively. E03 retains the
pinned STDN graph, batch 2, normalized iBUG-68 landmarks and exact flip permutation, and
`OFFICIAL_LATEST_FINAL_CKPT_50`. E06c retains pinned DSDG components with A1 ID-free
pairs, one spoof class, `lambda_pair=0`, effective batch 240 and
`OFFICIAL_GENERATOR_EPOCH_200`. LightCNN bytes reverified without deserialization.
Static/synthetic validation only; no training launcher was invoked, no framework/model execution,
no benchmark images, no checkpoint, no bank. E04/E05/E07c remain CONFIG_FROZEN.
The final execution environment and actual training remain pending. Full audit:
`outputs/audit/M6C2A_IMPLEMENTATION.md` / `.json`.

**M6A6 (2026-09-22) — E04 contract resolved, implementation not started.**
Amendment A4 resolves the two M6C2b1 execution gaps additively: official
uint8-preserving depth resize/conversion order and Adam semantics inherited from
the pinned STDN predecessor. The frozen E04 YAML and snapshot remain unchanged;
At M6A6, E04 was **CONTRACT_RESOLVED_NOT_IMPLEMENTED**. Details:
`outputs/audit/M6A6_E04_EXECUTION_GAP_RESOLUTION.md` / `.json`.

**M6C2b1 — E04 IMPLEMENTED_NOT_EXECUTED.** The fixed geometry/depth adapter,
A4 conversion/optimizer mapping and three static seed plans are implemented.
Official-kernel tests use synthetic meshes only; no PhySTD model, training or
benchmark input has executed. E05 and E07c remain CONFIG_FROZEN. Details:
`outputs/audit/M6C2B1_IMPLEMENTATION.md` / `.json`.
