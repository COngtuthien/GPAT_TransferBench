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
| `configs/methods/fas_aug.yaml` | §8.1 | NOT CREATED (M6) — **BLOCKED** | **E01-3 only** (M6A2): no canonical official asset order, so `os.listdir` makes texture choice host-dependent. Resolved in M6A2: E01-1 (`changeLabel` never reaches `augment_fn`, so it is the authors' label policy, not the transformation; §8.1 fixes the bank label) and E01-2 (`level=0` is the official parameter grid; retained, ≈10 % no-op disclosed). Source pinned: `RizhaoCai/FAS-Aug@0da1dd79` |
| `configs/methods/freq_sub.yaml` | §8.2 | NOT CREATED (M6) — **BLOCKED** | Algorithm source is the frozen SPEC (`SPEC_DEFINED`); **three** execution-affecting spec-silent choices remain: E02-1 rounding (computed exactly: **159 eligible blocks**, ×0.25 = **39.75**, so the choice is real), E02-2 block-selection algorithm/RNG, E02-3 pair-seed formula scope. Resolved in M6A2: E02-4 — "conjugate-symmetric counterpart" is pixel-wise `p ↔ 256−p` (`p=0` Nyquist and `p=128` DC self-conjugate); no 16×16 block maps to a grid block, and only pixel-wise mirroring gives an exactly Hermitian spectrum. The contract is **not** complete |
| `configs/methods/stdn.yaml` | §8.3 | NOT CREATED (M6) — **NO REMAINING SOURCE BLOCKERS** | **E03-2r resolved in M6A2**: FaceXFormer's landmark head is trained on **300-W** (arXiv:2403.12960v3 §F.2, ref [82] = Sagonas et al. ICCVW 2013 = iBUG-68) and STDN's `lm_reverse_list` is byte-for-byte the canonical iBUG-68 flip permutation. Also resolved: E03-1 (`BATCH_SIZE=2`), E03-2 (68-pt geometry from the frozen M2 cache), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint → `SOURCE_RESOLVED` (final/latest `ckpt-50`). Config still **not created**: M6A2 forbids method YAMLs and the seed adapter is unimplemented. Source pinned: `yaojieliu/ECCV20-STDN@c79f1f8c` |
| `configs/methods/physics_std.yaml` | §8.4 | NOT CREATED (M6) — **BLOCKED** | **E04-2** `L_depth` (α₁ = 100) needs a depth map rendered from a 3DMM fit on 140 landmarks; the frozen cache has 68, no 3DMM, no depth, and refs [60]/[70] are unpinned. **E04-3** `CHECKPOINT_SELECTION_UNSPECIFIED` — no official code and the paper gives only a 150,000-iteration budget. M6A2 identified refs **[60]** `yaojieliu/ICCVW2017-DenseFaceAlignment@01cfb1aa` and **[70]** `1adrianb/face-alignment`, but the route stays blocked: license-gated Basel Face Model, unspecified Q=140 vertex set, undescribed depth rendering, MATLAB test-only release. α₀ is **explicitly non-blocking** — owner-confirmed 2026-09-21 as scoring-only, not a generator parameter. Q-09: no official release located after documented search as of 2026-09-21 |
| `configs/methods/pcgan.yaml` | §8.5 | NOT CREATED (M6) — **BLOCKED** | E05-1 (method defined end-to-end at 1024×1024 vs the 256×256 benchmark and §8.5 output), **E05-2** narrowed in M6A2 — [34] is now pinned (`taesungp/swapping-autoencoder-pytorch@6baa180f`, proven by exact `netE_num_downsampling_sp` default 4 and `spatial_code_ch` default 8 matches), but PCGAN's prose cites AdaIN [35] while [34] implements StyleGAN2 demodulation; **E05-3** `CHECKPOINT_SELECTION_UNSPECIFIED`. **E05-1 resolved in M6A2**: [34] officially runs at 256/512/1024, so 1024 is an experiment setting, not an architectural requirement — at 256 the route is `sp=1`, `z_pat = 8×128×128`, blur ÷2. Q-08: no official release located after documented search as of 2026-09-21 |
| `configs/methods/dsdg_binary.yaml` | §8.6 | NOT CREATED (M6) | superseded for Track A by E06c (Amendment A1) |
| **E06c** DSDG-BIN-IDFREE | §8.6 + A1 | adaptation semantics frozen in `configs/frozen/dsdg_bin_idfree_v1.yaml`; no method YAML — **BLOCKED** | **E06c-2 only**: LightCNN-29 v2 is `REQUIRED_OFFICIAL_EXTERNAL_WEIGHT_NOT_FETCHED` — the official README gives a download link and the path `./ip_checkpoint`, but M6A1 forbids fetching, so its local SHA256 is UNKNOWN. Resolved: E06c-1 → `RESOLVED_BY_OFFICIAL_EXECUTION_PATH` (README → `train_generator.sh` → argparse; `lr` falls through to 2e-4), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint selection → `SOURCE_RESOLVED` (`netG` epoch 200). Official **effective** batch 240 is a native hyperparameter, not implementation-only |
| `configs/methods/dsdg_native.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_binary.yaml` | §8.7 | NOT CREATED (M6) | superseded for Track A by E07c (Amendment A1) |
| **E07c** DIFFFAS-BIN-IDFREE | §8.7 + A1 | adaptation semantics frozen in `configs/frozen/difffas_bin_idfree_v1.yaml`; no method YAML — **BLOCKED** | **E07c-3 resolved in M6A2** (`FAS_sample.py` is the dedicated generation entrypoint → **DDIM**, `DDIM_skip=10`, `sample_initial_noise=250` ⇒ 25 steps, `cond_scale=2.0`; its `use_pair=False` matches DEV-021 and it parameterises the seed). Remaining: E07c-4 (PADISI conditioning encoder required every training step regardless of `use_pair`, not fetched, no official link), **E07c-5** `CHECKPOINT_SELECTION_UNSPECIFIED` (`--model_path` is `required=True` with no default; repo has zero releases and zero tags). Resolved: E07c-1 and E07c-2; seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED` |
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

Verbatim copies of created configs are in `frozen_config_snapshot/` (spec §0.1 rule 1);
SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`. Tests assert the two stay byte-identical.
`configs/data_source_registry.yaml` records the selected dataset sources (M1 final); it is a registry, not a frozen config.
`configs/proposed/` holds superseded proposals kept as history; they are never edited after promotion.

