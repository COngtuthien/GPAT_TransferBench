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
| `configs/methods/fas_aug.yaml` | §8.1 | NOT CREATED (M6) — **NO REMAINING CONTRACT BLOCKER** | **E01-3 resolved in M6A3b by OWNER DECISION (option A, Amendment A2 §A2-01)**: each official asset directory is enumerated in **ascending lexicographic (bytewise) order by filename**, which must agree with the canonical repository/git-tree order for the pinned assets; runtime `os.listdir` order must not affect results. Deterministic adaptation only — operator set, magnitude distribution (`level=0` retained), asset contents and per-pair operator seed all unchanged. **OBS-1 resolved in M6A3b** (A2-05, byte-layout clarification of the frozen §8.1 formula, **not** a change to it): `global_seed` = the current experiment seed from `{42,1337,2026}` (so the E01 bank **differs per seed**), preimage = `UTF-8(pair_id + decimal_string(global_seed))` with **no separator**, `operator_seed = unsigned_big_endian(SHA256(preimage)) mod 2^31`; e.g. `PTR000001` + `42` → `PTR00000142` → `795981663`. Still resolved from M6A2: E01-1, E01-2. Config **not created** because **M6B has not started**. Source pinned: `RizhaoCai/FAS-Aug@0da1dd79` |
| `configs/methods/freq_sub.yaml` | §8.2 | NOT CREATED (M6) — **NO REMAINING CONTRACT BLOCKER** | **E02-1/2/3 all resolved in M6A3b by OWNER DECISION (Amendment A2 §§A2-02/03/04).** **E02-1 → round-half-up**, frozen as a GENERAL rule and not as the number 40: `k = round_half_up(n_eligible × 0.25)`, implemented float-free as `k = (n_eligible + 2) // 4`; for the frozen geometry `k = (159 + 2) // 4 = 40` (audit-verified: 156→39, 157→39, 158→40, 159→40, 160→40 — round-half-up and ceil coincide at 159 but diverge at 157). **E02-2 → NumPy PCG64 without replacement**: eligible blocks enumerated row-major over the 16×16 grid (`block_row` ascending, then `block_col`), then `numpy.random.Generator(numpy.random.PCG64(pair_seed)).choice(n_eligible, size=k, replace=False)`; the **selected set** is the result — returned order carries no scientific meaning, and indices are **sorted ascending** before substitution if a processing order is needed. `default_rng` without naming PCG64, `random.sample` and hash ranking are **prohibited**; the runtime freeze must pin the NumPy version (audit observed 2.5.3 — **not** the authoritative pin). **E02-3 → dedicated namespace**: `pair_seed = unsigned_big_endian(SHA256(UTF-8("gpatbench.freqsub.block.v1\|" + pair_id + "\|" + decimal_string(global_seed))))`, the **full 32-byte digest** passed directly to `PCG64` — no truncation, no modulo, no hidden separator, no whitespace, no newline; `global_seed` = the current experiment seed from `{42,1337,2026}`. Intentionally distinct from the E01 namespace. Still resolved from M6A2: E02-4 (pixel-wise `p ↔ 256−p`). Config **not created** because **M6B has not started** |
| `configs/methods/stdn.yaml` | §8.3 | NOT CREATED (M6) — **NO REMAINING SOURCE BLOCKERS** | **E03-2r resolved in M6A2**: FaceXFormer's landmark head is trained on **300-W** (arXiv:2403.12960v3 §F.2, ref [82] = Sagonas et al. ICCVW 2013 = iBUG-68) and STDN's `lm_reverse_list` is byte-for-byte the canonical iBUG-68 flip permutation. Also resolved: E03-1 (`BATCH_SIZE=2`), E03-2 (68-pt geometry from the frozen M2 cache), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint → `SOURCE_RESOLVED` (final/latest `ckpt-50`). **M6A3b: `BASELINE_FINAL_STATE_V1` does NOT override the `ckpt-50` rule** — clause 1 of the policy preserves an already-resolved official checkpoint rule (Amendment A2 §A2-06). Config still **not created**: no method YAMLs before M6B, and the seed adapter is unimplemented. Source pinned: `yaojieliu/ECCV20-STDN@c79f1f8c` |
| `configs/methods/physics_std.yaml` | §8.4 | NOT CREATED (M6) — **BLOCKED by E04-2 only** | **E04-2 STILL BLOCKING after the M6A4 authoritative-source pass (2026-09-21)** → `STILL_BLOCKED_SOURCE_GAP`; `FAITHFUL_PAPER` preserved and **no controlled adaptation created**. M6A4 searched the **published TPAMI 2022 version** (Liu & Liu, *Spoof Trace Disentanglement for Generic Face Anti-Spoofing*, full 17-page text), arXiv:2012.05185, the official MSU CVLab project page, the author's GitHub account (7 repos) and homepage, and the pinned ECCV20-STDN + Dense Face Alignment trees. **New findings:** the published version renumbers 3DMM fitting to **[61]** (arXiv's [60]) and keeps **[70]** — the *same* works M6A2 pinned, so **no new source** and `source_pins.json` unchanged; the published paper **still** promises code "upon publication" and **none exists** (the project page's only lineage code link is `ECCV20-STDN`, whose 12-file tree contains **zero** occurrences of `140`/`vertex`/`3DMM`/`BFM`/depth code); the paper has **no supplementary section** and **never names the BFM** (that requirement reaches E04 only transitively via the DeFA README). **All four sub-gaps remain:** (1) **BFM** `LICENSE_GATED_NOT_ACQUIRED` — signed academic licence, exact version not identifiable, **not downloaded, licence not bypassed**; (2) **Q=140** `COUNT_STATED_INDICES_NEVER_PUBLISHED` — only the count and a rationale are published, no index list anywhere; (3) **depth rendering** `GROUND_TRUTH_RENDERING_NEVER_DESCRIBED` — the published Algorithm 1 takes `M₀` as a pre-computed **INPUT**; only the *network output* (32×32, normalized [0,1]) is specified, and camera/projection/visibility/normalization/rasterization/resize are all unstated (**nothing inferred**); (4) **executable route** `NO_EXECUTABLE_OFFICIAL_ROUTE` — DeFA is MATLAB/MatConvNet test-only *and* BFM-gated. **E04-3** remains resolved by `BASELINE_FINAL_STATE_V1` (terminal state at iteration 150,000). α₀ is non-blocking (owner-confirmed 2026-09-21, scoring-only) |
| `configs/methods/pcgan.yaml` | §8.5 | NOT CREATED (M6) — **NO REMAINING CONTRACT BLOCKER**; fidelity **`CONTROLLED_ADAPTATION`** | **E05-2 resolved in M6A3b by OWNER DECISION (option A, Amendment A2 §A2-07)**: the pinned `[34]` implementation `taesungp/swapping-autoencoder-pytorch@6baa180f1184ee79a6b967f9d80ee0e02a979ac7` is the **executable architecture basis**, followed as it actually executes — including its **StyleGAN2-style modulation/demodulation** and corresponding discriminator. StyleGAN-v1 AdaIN is **not** separately ported to chase the ambiguous prose citation `[35]`. **This discrepancy must not be hidden:** E05 fidelity is **`CONTROLLED_ADAPTATION`**, because PCGAN prose cites AdaIN/StyleGAN-v1 while its named executable base uses StyleGAN2 demodulation. Method ID and result row stay **E05** (not renamed, not removed); the final report **must disclose** this, and the row must **not** be called "native PCGAN" or "faithful PCGAN". **E05-3 RESOLVED in M6A3b by `BASELINE_FINAL_STATE_V1`** → terminal state at **iteration 4,000** (disclosed benchmark convention; no VAL-based and no TEST-based selection). **E05-1 resolved in M6A2**: [34] officially runs at 256/512/1024, so at 256 the route is `sp=1`, `z_pat = 8×128×128`, blur ÷2. Q-08: no official release located after documented search as of 2026-09-21 |
| `configs/methods/dsdg_binary.yaml` | §8.6 | NOT CREATED (M6) | superseded for Track A by E06c (Amendment A1) |
| **E06c** DSDG-BIN-IDFREE | §8.6 + A1 | adaptation semantics frozen in `configs/frozen/dsdg_bin_idfree_v1.yaml`; no method YAML — **NO REMAINING BLOCKER (asset-ready)** | **E06c-2 RESOLVED in M6A4** → `RESOLVED_EXTERNAL_WEIGHT_ACQUIRED_AND_PINNED`. The owner-authorized acquisition (A2-08) succeeded: the **original author's** LightCNN-29 v2 checkpoint was fetched from Google Drive file id `1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS` — the **same id linked by BOTH** the upstream `AlfredXiangWu/LightCNN` README (line 82) and the DSDG README (line 21), so no mirror was substituted. **123,844,849 bytes**, SHA256 `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964` (upstream publishes no checksum, so this is an **observed pin**). Stored **outside** the repo and outside `third_party/source_cache/`, **not git-tracked**: `/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/lightcnn/`. Inspected with a **restricted non-executing unpickler** (torch absent, so `weights_only=True` was unavailable — the method used is stronger, executing nothing): torch legacy format, magic `0x1950a86a20f9469cfc6c`, top-level keys `state_dict`/`epoch`/`arch`/`prec1` (`epoch=50`, `arch='LightCNN'`, `prec1=99.46`), **61 float32 tensors**, globals limited to `OrderedDict`/`torch.FloatStorage`/`torch._utils._rebuild_tensor` — **no unexpected executable objects**. **Statically verified against `define_IP()`**: **60/60 expected parameters matched with exact shapes, 0 missing, 0 shape mismatches**, 1 unexpected (`module.fc2.weight [80013, 256]`, the 80013-class head dropped by the documented non-strict filter — and itself corroborating identity, since the upstream README documents `num_classes=80013`). **Never loaded for inference.** Provenance: `outputs/audit/M6A4_LIGHTCNN_WEIGHT_PROVENANCE.json`. Resolved earlier: E06c-1 → `RESOLVED_BY_OFFICIAL_EXECUTION_PATH` (`lr` 2e-4), seed → `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED`, checkpoint → `SOURCE_RESOLVED` (`netG` epoch 200), which `BASELINE_FINAL_STATE_V1` does **not** override (A2-06 clause 1). Method YAML still **not created** — M6B has not started |
| `configs/methods/dsdg_native.yaml` | §8.6 | NOT CREATED (M6) | official repo at pinned commit |
| `configs/methods/difffas_binary.yaml` | §8.7 | NOT CREATED (M6) | superseded for Track A by E07c (Amendment A1) |
| **E07c** DIFFFAS-BIN-IDFREE | §8.7 + A1 | adaptation semantics frozen in `configs/frozen/difffas_bin_idfree_v1.yaml`; no method YAML — **BLOCKED by E07c-4 only** | **E07c-4 STILL BLOCKING after the M6A4 exhaustive search (2026-09-21)** → `STILL_BLOCKED_SOURCE_GAP`, dated conclusion **`NO_OFFICIAL_ENCODER_RELEASE_LOCATED_AS_OF_2026_09_21`** (absolute nonexistence **not** claimed). Faithful official route preserved: **no substitute trained, encoder not disabled, not replaced with ImageNet ResNet18**. Searched: the pinned tree, GitHub API **releases=0 / tags=0 / branches=1 / 5 commits / 1 issue (all 6 comments)**, the **ECCV 2024 official supplementary PDF** (5 pages — no `PADISI.pkl`, no encoder, no link), and a mirror search. **Decisive:** in issue #1 the repository **OWNER** promised on **2024-09-24** to "update the required pretrained models" — and never did: the last push (**2024-09-23**) *predates* the promise, there are 0 releases/0 tags, and four follow-ups through 2025-03-10 went unanswered. **No official asset or link was located, so nothing was downloaded for E07c** and the E06c authorization was **not** reused. **New M6A4 finding:** the official **training script exists** at `models/pretrain_classifier.py` (custom_rn `resnet18` — actually a `[3,4,6,3]` ResNet-34 topology — with `fc=Linear(512,17)`, 256×256 + `Normalize([0.5]*3)`, `ImageFolder`, batch 256, SGD lr 2e-3/mom 0.9/wd 5e-3, 200 epochs, `torch.save` of the **whole module**), so the protocol need **not** be invented — but the **weights**, the licence-gated **PADISI-USC** dataset (not a GPAT dataset) and the **17-class ImageFolder label mapping** are all unavailable, so even a faithful re-run is **not uniquely reproducible**. Training a replica would need a **new explicit owner authorization** and would change E07c's fidelity class. The dependency is unconditional (`training_losses` calls `encode()` every step; `use_pair=false` does not remove it). **E07c-5** remains resolved by `BASELINE_FINAL_STATE_V1`. E07c-3 resolved in M6A2 (DDIM, `DDIM_skip=10`, 25 steps, `cond_scale=2.0`) |
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

Verbatim copies of created configs are in `frozen_config_snapshot/` (spec §0.1 rule 1);
SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`. Tests assert the two stay byte-identical.
`configs/data_source_registry.yaml` records the selected dataset sources (M1 final); it is a registry, not a frozen config.
`configs/proposed/` holds superseded proposals kept as history; they are never edited after promotion.

