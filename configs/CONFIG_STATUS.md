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
| `configs/methods/fas_aug.yaml` | §8.1 | NOT CREATED (M6) — **BLOCKED** | E01-1 (spec requires every output spoof-labeled, but official `changeLabel=False` for 3 of 8 operators), E01-2 (`level=0` returns an unmodified live copy), E01-3 (unsorted `os.listdir` makes texture choice host-dependent). Source pinned: `RizhaoCai/FAS-Aug@0da1dd79` |
| `configs/methods/freq_sub.yaml` | §8.2 | NOT CREATED (M6) — **BLOCKED** | Algorithm source is the frozen SPEC (`SPEC_DEFINED`), but four execution-affecting spec-silent choices remain: E02-1 rounding of "exactly 25 %", E02-2 block-selection algorithm/RNG, E02-3 pair-seed formula scope, E02-4 conjugate-counterpart index convention. The contract is **not** complete |
| `configs/methods/stdn.yaml` | §8.3 | NOT CREATED (M6) — **BLOCKED** | **E03-2r only**: `lm_reverse_list` assumes the iBUG-68 layout; M2 records only "68-point layout". Resolved: E03-1 (config precedence → `BATCH_SIZE=2`), E03-2 (68-pt geometry from the frozen M2 cache), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint selection → `SOURCE_RESOLVED` (final/latest `ckpt-50`). Source pinned: `yaojieliu/ECCV20-STDN@c79f1f8c` |
| `configs/methods/physics_std.yaml` | §8.4 | NOT CREATED (M6) — **BLOCKED** | **E04-2** `L_depth` (α₁ = 100) needs a depth map rendered from a 3DMM fit on 140 landmarks; the frozen cache has 68, no 3DMM, no depth, and refs [60]/[70] are unpinned. **E04-3** `CHECKPOINT_SELECTION_UNSPECIFIED` — no official code and the paper gives only a 150,000-iteration budget. α₀ is **explicitly non-blocking** — owner-confirmed 2026-09-21 as scoring-only, not a generator parameter. Q-09: no official release located after documented search as of 2026-09-21 |
| `configs/methods/pcgan.yaml` | §8.5 | NOT CREATED (M6) — **BLOCKED** | E05-1 (method defined end-to-end at 1024×1024 vs the 256×256 benchmark and §8.5 output), E05-2 (upstream architectures [34]/[35] unpinned and unnamed in spec §30), **E05-3** `CHECKPOINT_SELECTION_UNSPECIFIED`. Q-08: no official release located after documented search as of 2026-09-21 |
| `configs/methods/dsdg_binary.yaml` | §8.6 | NOT CREATED (M6) | superseded for Track A by E06c (Amendment A1) |
| **E06c** DSDG-BIN-IDFREE | §8.6 + A1 | adaptation semantics frozen in `configs/frozen/dsdg_bin_idfree_v1.yaml`; no method YAML — **BLOCKED** | **E06c-2 only**: LightCNN-29 v2 is `REQUIRED_OFFICIAL_EXTERNAL_WEIGHT_NOT_FETCHED` — the official README gives a download link and the path `./ip_checkpoint`, but M6A1 forbids fetching, so its local SHA256 is UNKNOWN. Resolved: E06c-1 → `RESOLVED_BY_OFFICIAL_EXECUTION_PATH` (README → `train_generator.sh` → argparse; `lr` falls through to 2e-4), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint selection → `SOURCE_RESOLVED` (`netG` epoch 200). Official **effective** batch 240 is a native hyperparameter, not implementation-only |
| `configs/methods/dsdg_native.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_binary.yaml` | §8.7 | NOT CREATED (M6) | superseded for Track A by E07c (Amendment A1) |
| **E07c** DIFFFAS-BIN-IDFREE | §8.7 + A1 | adaptation semantics frozen in `configs/frozen/difffas_bin_idfree_v1.yaml`; no method YAML — **BLOCKED** | E07c-3 (generation sampler undecided: `FAS_sample.py` defaults ddim, `FAS_train.py` ddpm), E07c-4 (PADISI conditioning encoder required every training step regardless of `use_pair`, not fetched, no official link), **E07c-5** `CHECKPOINT_SELECTION_UNSPECIFIED` (`--model_path` is `required=True` with no default). Resolved: E07c-1 and E07c-2; seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED` |
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
`CHECKPOINT_SELECTION_UNSPECIFIED` gap for E04, E05 and E07c — the frozen spec fixes a generator
selection rule only for GPAT (§10.6) and for downstream evaluators (§14), never for baselines, and
**TEST is never used for checkpoint selection**. No checkpoint was chosen in M6A1.

Verbatim copies of created configs are in `frozen_config_snapshot/` (spec §0.1 rule 1);
SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`. Tests assert the two stay byte-identical.
`configs/data_source_registry.yaml` records the selected dataset sources (M1 final); it is a registry, not a frozen config.
`configs/proposed/` holds superseded proposals kept as history; they are never edited after promotion.

