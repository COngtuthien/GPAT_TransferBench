# M8 global smoke test: S1 resolution, S2–S10 decision matrix and downstream readiness

Date: 2026-11-22. Branch `m6-baselines`. Starting HEAD = origin = `025d9e8993ff18ca7b7b16e23a15411f45f44683`.

**Status: `M8_SMOKE_CONTRACT_PARTIALLY_RESOLVED`.**

- S1 is owner-approved (`550d5cc`).
- S2–S10 are pending.
- The downstream evaluator contract has 14 open items (D1–D14).
- No smoke stage, detector, toy selection, bank or TEST access was run.

The companion files are:

| File | Content |
|---|---|
| `M8_DOWNSTREAM_READINESS_ASSESSMENT.json` | machine-readable record |
| `configs/amendments/m8_smoke_test_s1_owner_decision.yaml` | S1 ruling |
| `configs/amendments/m8_smoke_test_contract.DRAFT.yaml` | S2–S10, unchanged since `025d9e8` |
| `configs/amendments/m8_downstream_evaluator_contract.DRAFT.yaml` | D1–D14 |

## 1. S1 (approved)

S1 = `A_GLOBAL_END_TO_END`; the owner's label is `A_GLOBAL_END_TO_END_SMOKE_REQUIRED`.

- **The test:** one deterministic smoke test runs generator/transformation → bank → evaluator → metrics → plots. It must
  pass before any full M8+ execution.
- **No substitute:** the M6 writer qualification does not replace it. O3 is not amended.
- **Still blocked:** E01/E02 production banks remain blocked, and GPAT remains deferred.

## 2. S2–S10 decision matrix

None of these can be resolved by existing authority. The frozen text says only "2-subject-per-class toy subset" and
"generator → bank → evaluator → metrics → plots". Each answer therefore needs an owner ruling. Every ruling would be
recorded additively in `configs/amendments/m8_smoke_test_contract.yaml`, which is the "new amendment" in each row.

| ID | Scientific issue | Options | Recommended | Frozen evidence | Changes science? | Blocks impl.? | Blocks smoke? |
|---|---|---|---|---|---|---|---|
| S2 class | "class" may be the binary label or attack_macro, per dataset or pooled | BINARY_LABEL_PER_DATASET / ATTACK_MACRO_PER_DATASET / BINARY_LABEL_POOLED | BINARY_LABEL_PER_DATASET | §26; T01 "binary class"; Track A Binary is primary | no (smoke only) | toy selector | yes |
| S3 SiW-Mv2 unit | `subject_id_global` is null for all 9,507 TRAIN and 2,033 VAL SiW-Mv2 rows; the split is video-disjoint by `content_group_id` (DEV-012, "never described as subject-disjoint") | CONTENT_GROUP_ID (disclosed proxy) / EXCLUDE_SIWMV2 | CONTENT_GROUP_ID: it is the already-approved DEV-012 allocation unit, not an identity derived from filenames, and must be called "content group", not "subject" | `configs/frozen/split_v1.yaml` allocation_keys | no | toy selector | yes |
| S4 selection | deterministic choice of units | HASH_RANK / OWNER_LIST | HASH_RANK (§6 below) | none | no | toy selector | yes |
| S5 splits / TEST | the smoke's "apply threshold once to TEST" stage | TRAIN_VAL_ONLY_TEST_PROXY / TOY_TEST | TRAIN_VAL_ONLY_TEST_PROXY: §0.1 forbids TEST in any selection, and nothing authorizes TEST before the threshold freeze | §0.1, §15, §25 M10 | no | firewall config | yes |
| S6 bank pairs | the toy pair population | FROZEN_PAIRS_TARGET_IN_TOY / FROZEN_PAIRS_FIRST_N / REBUILD_PAIRS_ON_TOY | FROZEN_PAIRS_TARGET_IN_TOY: keeps frozen pairs and global positions, so smoke bytes equal future production bytes | §6, §26 | no | needs a bank-writer SMOKE mode | yes |
| S7 methods | which generators | ALL_METHODS_WITH_BANK_WRITERS / ALL_ACTIVE_METHODS | ALL_METHODS_WITH_BANK_WRITERS (today E01/E02). Note: this leaves E03–E06c each needing a bank-stage smoke later | §25 M8 "all synthetic banks" | no | scope | yes |
| S8 budget / env | smoke schedule, device, environment | SMOKE_SCHEDULE_CPU / SMOKE_SCHEDULE_GPU_AFTER_E04 | CPU, a fixed tiny schedule, new env (§8): a CPU run does not compete with E04 | §14 (40 epochs is the science budget, not the smoke) | no | evaluator runner, env | yes |
| S9 determinism | bitwise or procedural | BITWISE_ALL_STAGES / BITWISE_BANK_PROCEDURAL_EVALUATOR | BITWISE_ALL_STAGES if CPU (S8) | §26 "deterministic" | no | acceptance gates | yes |
| S10 seeds | which seeds | ALL_THREE_BANK_SEED42_EVALUATOR / SEED42_ONLY | ALL_THREE_BANK_SEED42_EVALUATOR | §0 seeds | no | acceptance gates | yes |

## 3. Downstream evaluator authority (spec §14–§16, §18–§23, §26)

| # | Item | Frozen authority | State |
|---|---|---|---|
| 1 | Architecture | torchvision ResNet-18, FC → Linear(512,1) | RESOLVED |
| 2 | Weights | IMAGENET1K_V1, `resnet18-f37072fd.pth` | RESOLVED |
| 3 | Weight provenance and SHA256 | `f37072fd47e8…07ec`, `models/registry.yaml` VERIFIED (M5); present in the GPU host torch cache | RESOLVED |
| 4 | Training input | 224×224 RGB, ImageNet normalization; RandomResizedCrop, flip, jitter (§14.1) | RESOLVED for training. **D2** (eval-time 256→224) and **D14** (augmentation order) are open |
| 5 | Synthetic input | 16 synthetic spoofs per mixed batch; canonical 256×256 uint8 (§7) | **D12** open: seed pairing and the pool draw |
| 6 | Labels | binary live/spoof only; no maps or q | RESOLVED; **D3** open: BCE positive class |
| 7 | TRAIN/VAL | train on TRAIN; VAL for checkpoint and threshold | **D5** open: per-epoch VAL protocol |
| 8 | Optimizer / LR | AdamW, 1e-4 / 5e-4, wd 1e-4, 3 warmup epochs, cosine to 1e-6, clip 1.0, AMP on CUDA | **D6** open: details |
| 9 | Budget | 40 epochs; batch 64 (32/32 or 32/16/16) | **D1** open: epoch definition and sampler |
| 10 | Seeds | 42/1337/2026; one detector per method/track/seed | RESOLVED; **D12** open: bank-seed pairing |
| 11 | Checkpoint | min VAL video ACER, then max VAL video AUC | **D5** open |
| 12 | Unit | video-level primary, frame-level diagnostics | RESOLVED; **D9** open: frame threshold |
| 13 | Aggregation | arithmetic mean of a canonical video's frame probabilities | RESOLVED (implemented) |
| 14 | Threshold | VAL video min ACER over unique midpoints plus {0,1}, nearest 0.5, applied once to TEST | RESOLVED (implemented); **D4** open: ≥ or >; a third-level tie raises |
| 15 | Metrics | APCER, BPCER, ACER, AUC, accuracy, balanced accuracy | RESOLVED (implemented). **D7** (EER) and **D8** (TPR@FPR, F1) open |
| 16 | ROC/DET | fig17 ROC, fig18 DET from predictions | operating points implemented; rendering blocked (no matplotlib) |
| 17 | Tables | T06, T07, T08, T13, T14 (§19) | schemas known; nothing measured |
| 18 | Dataset-wise | dataset ACER/AUC, attack_macro APCER, SiW raw 14-type APCER | first two implemented; **D10** open: SiW validity |
| 19 | Provenance | §0.1 rule 8; §22.2 / §22.3 | schemas implemented |
| 20 | TEST | never in training or any selection; synthetic never in TEST; one application of the frozen threshold | firewall implemented (TEST never allowed) |

**DINOv3 (§14.2) is not interchangeable with ResNet-18.** It is a separate "mandatory FULL profile" evaluator that
§25 schedules in M11, after M10 passes. Its state:

- `configs/frozen/downstream_dinov3_vits16.yaml`, which §23 lists, **does not exist**.
- The HF revision and weight SHA256 are unpinned (`models/registry.yaml` NOT_DOWNLOADED), and the model is
  license-gated.
- `transformers` is in no project environment.

This is a source gap for M11 (**D11**). The recommendation is ResNet-18 only in the smoke.

## 4. Implemented in this task (non-executing)

| File | What it implements |
|---|---|
| `gpatbench/downstream/metrics.py` | §15 aggregation and threshold sweep; §16 rates, AUC, ROC/DET points, subgroups; §18 seed mean and sample std. The decision operator is a required argument |
| `gpatbench/downstream/schemas.py` | §22.1, §22.2 and §22.3 schemas and validators; canonical table SHA256 |
| `gpatbench/downstream/firewall.py` | hash-checked split lookup; only TRAIN/VAL ever allowed |
| `gpatbench/downstream/bank_reader.py` | verified E01/E02 bank reader |
| `gpatbench/downstream/plots.py` | §26 placeholder rejection; template mode is explicit |
| `gpatbench/downstream/smoke_evidence.py` | a record passes only if the approved contract holds, all 5 O3 stages executed (not simulated) and passed, the replay matches, the firewall held, the tree was clean and no science result was produced |
| `tools/m8_smoke.py --check` | fail-closed readiness report; exits 3 and has no execution path |

**Not implemented (open items):** EER, TPR@FPR, F1, the frame-level threshold, SiW raw coverage, the toy selector, the
evaluator runner, the plot renderer and the bank-writer SMOKE mode.

## 5. Missing components and assets

1. **Approvals:** S2–S10 and D1–D14.
2. **Toy selector:** `gpatbench/downstream/toy_subset.py`.
3. **Bank-writer SMOKE mode.** `methods/common/bank_writer.py` has only PRODUCTION (all 8838 pairs) and QUALIFICATION
   (synthetic fixtures; refuses roots with real faces). A toy run on real TRAIN faces needs a third mode that:
   - reads only the toy pairs;
   - writes under a separate smoke root;
   - is labelled non-scientific.
4. **ResNet-18 evaluator runner:** dataloaders, sampler, training loop, VAL loop and prediction writer.
5. **Plot renderer:** needs matplotlib, which no locked project environment has.
6. **Evaluator environment lock:** `environments/m8_downstream.lock.json`.
7. **DINOv3:** config, pins, license and environment (M11).

## 6. Proposed deterministic toy selector (not executed; valid only if S2–S6 are approved as recommended)

**Inputs.** `manifests/split_v1.parquet`, hash-checked (`fb9aeb36…`). Rows are filtered to `split ∈ {TRAIN, VAL}`
before any other use. TEST rows are never counted, ranked or read further.

**Class.** `label_binary` (0 live, 1 spoof), within each of casia_fasd, msu_mfsd and siwmv2.

**Units.**
- CASIA/MSU use `subject_id_global`. The selector verifies it is non-null for every row of the dataset. Every
  CASIA/MSU subject has both classes (35/25 TRAIN, 8/5 VAL).
- SiW-Mv2 uses `content_group_id`, the DEV-012 allocation unit, and is disclosed as "content group, not subject". Each
  group is class-pure. No identity is derived from filenames or video ids.

**Ranking key.** `SHA256("gpatbench.smoke.v1|" + dataset + "|" + split + "|" + unit_id)`, ascending; class is added for
SiW-Mv2. The key is seed-free and recorded.

**Selection per dataset and split:**

| Dataset | TRAIN | VAL |
|---|---|---|
| CASIA/MSU | the first 2 eligible subjects, which cover both classes, so each class has exactly 2 subjects | ranks 1–2 are VAL-select (checkpoint and threshold); ranks 3–4 are VAL-holdout (the TEST stand-in, S5). MSU VAL has 5 subjects, enough for 4 |
| SiW-Mv2 | per class, the first 2 eligible content groups | VAL-select is the first 2 per class; VAL-holdout is the next 2 per class |

**Eligibility.** A TRAIN live unit must be the target of at least 1 frozen `pairs_train_v1` pair. Measured from
metadata: 60/60 CASIA/MSU TRAIN live subjects qualify, as do 543/549 SiW-Mv2 TRAIN live groups. An ineligible unit is
skipped in rank order, and the skip is recorded.

**Pairs (S6).** The toy pairs are the frozen pairs whose target live sample lies in a toy TRAIN unit. Each keeps its
global position, so E01 uses operator `position % 8` and every output equals the corresponding future production output.
Sources may lie outside the toy units; they are TRAIN by construction (firewall-checked).

**Leakage.** Units are already split-disjoint in split_v1. VAL-select and VAL-holdout are disjoint by rank. TRAIN, VAL and
holdout units never overlap.

**Labelling.** Toy outputs are written only under a dedicated smoke root (never `runs/m6/`), every record is marked
`SMOKE_NON_SCIENTIFIC`, and the evidence binds the contract SHA256.

**Fit to §26.** It gives 2 units per binary class per dataset, deterministic and TEST-free. The one departure is the
SiW-Mv2 content-group proxy, which the owner must accept (S3).

## 7. End-to-end smoke plan (each step lists the decisions it needs)

| # | Step | Needs |
|---|---|---|
| 1 | Verify the approved smoke and evaluator contracts (`tools/m8_smoke.py --check` all green) | S2–S10, D1–D14 |
| 2 | Hash-verify sources: spec, configs, manifests, env lock, ResNet-18 weights | env lock (§8) |
| 3 | Run the toy selector and write the toy manifest with hashes | S2–S6 |
| 4 | Generate the E01/E02 toy banks in SMOKE mode (gpat-m5, isolated clone, CPU), for seeds per S10 | SMOKE mode, S6, S7, S10 |
| 5 | Verify the banks (`bank_reader`), with a bitwise replay | — |
| 6 | Train the ResNet-18 toy evaluator: E00 real-only plus one detector per method and seed, on the toy TRAIN set; select the checkpoint and threshold on VAL-select | D1–D6, D12–D14, S8 |
| 7 | Predict on VAL-holdout once with the frozen threshold, writing §22.2 files | D2–D4 |
| 8 | Compute the metrics tables (T06, T08, T13, T14 subsets) from the prediction files | D7–D10 |
| 9 | Render plots (fig14–fig20, fig24) from those tables; placeholder rejection active | renderer, matplotlib |
| 10 | Run an independent second execution in a fresh root; compare banks, predictions, tables and plot inputs (S9) | S9 |
| 11 | TEST firewall: audit hook plus split checks, 0 TEST rows or files | — |
| 12 | Write `M8_SMOKE_TEST_RECORD.json`; it must pass `smoke_evidence.validate_smoke_record` | — |
| 13 | Independent re-derivation of the record from raw outputs; commit | — |
| 14 | Separate owner approval to bind `smoke_test_record` (E01/E02 ruling) and to state that O3 is completed | owner |

No toy number is a scientific result, and simulated evaluator outputs are not acceptable in step 6 or 7.

## 8. Environment assessment (read-only inspection)

**Laptop `.venv`** (Python 3.12.3): numpy, pyarrow, pandas, PIL, cv2 and yaml. It has no torch, torchvision,
matplotlib, scipy or scikit-learn. It is suitable only for the non-executing code above.

**GPU host `gpat-m5`** (Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, pyarrow, pandas; lock
`environments/m5_probe_gpu.lock.txt` `8f1e3d16…`):
- It already trained the ResNet-18 ArtifactProbeNet and is the E01/E02 production environment.
- It has **no matplotlib** and no `transformers`.
- It must not be modified: it is the qualified E01/E02 environment (`m6_core_gpu.lock.json`).

**Other project environments (`gpat-*`):** none has torch, torchvision, matplotlib and pyarrow together. `gpat-m6-e03`
has matplotlib but no torch (Python 3.8). Among the non-project environments, `stdn` has all four, but it is unlocked,
unqualified and not part of the project authority, so it is not admissible; neither are grbkd or yolo11.

**ResNet-18 weights:** present at `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` on the host. The registry path is
on the unmounted laptop drive.

**CPU feasibility:**
- The host has 20 cores and about 20 GiB available.
- From the TRAIN averages, a toy subset is about 96 live frames and about 256 real spoof frames:
  - live: CASIA 2 × 24, MSU 2 × 16, SiW 2 × 8;
  - spoof: CASIA 2 × 72, MSU 2 × 48, SiW 2 × 8.
  The frozen toy pairs add a few hundred synthetic outputs.
- ResNet-18 at 224 for a few epochs on that is minutes of CPU at `nice 19`.
- It does not compete with E04 on the GPU.

**Proposed plan (needs approval for installation):**
1. Create a new environment `gpat-m8-downstream` by cloning `gpat-m5` (same Python, torch, torchvision, numpy and Pillow),
   leaving `gpat-m5` untouched.
2. Add matplotlib only, pinned.
3. Capture `pip freeze` and `conda list --explicit` into `environments/m8_downstream.lock.json` with SHA256.
4. Qualify the environment by:
   - importing every module;
   - CPU-loading the hash-verified ResNet-18 weights;
   - running a deterministic CPU forward pass twice on synthetic tensors and comparing bitwise;
   - rendering one plot from a synthetic table.

DINOv3 (`transformers`, gated weights) is M11 work and is not needed if D11 = RESNET18_ONLY_IN_SMOKE.

## 9. E04 isolation

The checks at 2026-11-22T00:06Z were read-only `pgrep`, `tmux ls`, `git` and one `tail` line, plus a read-only package
listing of the conda environments. They showed:

- seed 2026 running: PID 435565, global_step 12257, no exit marker;
- checkout `18ba00e`, clean.

Nothing was started or modified on the host.

## 10. Next action before a legitimate smoke run

1. Owner rulings S2–S10, committed as `configs/amendments/m8_smoke_test_contract.yaml`.
2. Owner rulings D1–D14, committed as `configs/amendments/m8_downstream_evaluator_contract.yaml`. This is the D11 scope.
3. Owner approval to create and lock `gpat-m8-downstream` (gpat-m5 clone plus matplotlib).
4. Implement the toy selector, the bank-writer SMOKE mode, the ResNet-18 runner and the plot renderer, and qualify them
   on synthetic fixtures.
5. Run the smoke plan in §7. Only a record that passes `validate_smoke_record` and an independent re-derivation can
   then go to the owner for binding.
