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
| `configs/frozen/artifact_probe.yaml` | §12.1 | NOT CREATED (M5) — `configs/proposed/artifact_probe.proposed.yaml` created at the M5 pre-flight (2026-09-20) with every unresolved field marked | Q-07 (extended), D-M5-01…D-M5-08; execution also blocked by E-M5-01 (CPU-only environment vs the required AMP) |
| `configs/methods/fas_aug.yaml` | §8.1 | NOT CREATED (M6) | official repo defaults at pinned commit |
| `configs/methods/freq_sub.yaml` | §8.2 | NOT CREATED (M6) | — |
| `configs/methods/stdn.yaml` | §8.3 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/physics_std.yaml` | §8.4 | NOT CREATED (M6) | paper extraction (Q-09) |
| `configs/methods/pcgan.yaml` | §8.5 | NOT CREATED (M6) | Q-08 |
| `configs/methods/dsdg_binary.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/dsdg_native.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_binary.yaml` | §8.7 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_native.yaml` | §8.7 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/gpat_b0.yaml` | §23.2 | **CREATED — spec verbatim** (extracted programmatically from DOCX) | DEV-003 `lambda_dir` must be resolved before M7 |
| `configs/methods/gpat_b1.yaml` | §5.3, §10.5 | NOT CREATED (M7) | B0 + λ_type 0.2 + warmup; key naming for new fields not in spec |
| `configs/methods/gpat_b2.yaml` | §5.3, §9.2 | NOT CREATED (M7) | Q-05 |
| `configs/methods/gpat_b3.yaml` | §5.3 | NOT CREATED (M7) | Q-05 |

Verbatim copies of created configs are in `frozen_config_snapshot/` (spec §0.1 rule 1);
SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`. Tests assert the two stay byte-identical.
`configs/data_source_registry.yaml` records the selected dataset sources (M1 final); it is a registry, not a frozen config.
`configs/proposed/` holds superseded proposals kept as history; they are never edited after promotion.

