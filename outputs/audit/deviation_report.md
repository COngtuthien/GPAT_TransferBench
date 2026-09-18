# Deviation Report — GPAT-TransferBench v1.0

Per spec §0.1 rule 9: every deviation from the frozen specification is recorded here with
reason, affected experiments and explicit status **APPROVED / UNAPPROVED**. Unapproved
deviations cannot be used for main paper results. Only the project owner may change a
status to APPROVED. This file is append-only in spirit: entries are never deleted; status
changes are appended as dated updates under the entry.

## Effective status summary (latest update wins; history kept below)

| ID | Effective spec status | Owner classification | Updated |
|---|---|---|---|
| DEV-001 | APPROVED | INFRASTRUCTURE_ADAPTATION | 2026-09-18 |
| DEV-002 | APPROVED | INFRASTRUCTURE_ADAPTATION | 2026-09-18 |
| DEV-003 | UNAPPROVED | UNRESOLVED — must be resolved before M7 | 2026-09-18 |
| DEV-004 | APPROVED | APPROVED_OPERATIONAL_DEVIATION (with conditions) | 2026-09-18 |

---

## DEV-001 — Project root path differs from spec §0.2

- **Spec says (§0.2 "Fixed project root and directory contract"):** `D:\AI on IOT\Anti_spoofing\GPAT_TransferBench` (Windows path).
- **Implemented:** `/home/cong/GPAT_TransferBench` (Linux laptop), per explicit owner instruction for this M0 session. The planned GPU mirror is `/home/sparc/workdir/longnm/GPAT_TransferBench` (not created in M0).
- **Observation:** the Windows `D:` volume appears to correspond to `/media/cong/Data` on this laptop (the spec file and an `AI on IOT/Anti_spoofing/` tree exist there). This is an inference, not verified. No `GPAT_TransferBench` directory exists there.
- **Directory contract inside the root:** preserved (all §0.2 subdirectories created, including `data/raw/{casia_fasd,msu_mfsd,siwmv2}` and `data/processed/{frames,faces_256}`).
- **Scientific impact:** none expected (path only); all manifests must use root-relative paths so the tree is portable.
- **Affected experiments:** none directly.
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: INFRASTRUCTURE_ADAPTATION. Linux root `/home/cong/GPAT_TransferBench` replaces the Windows root example of spec §0.2. No change to the scientific protocol.
  - Supersedes the UNAPPROVED line above (kept for history).

## DEV-002 — Additional directories/files beyond the §0.2 contract

- **Added:** `models/` (spec §4 requires `models/registry.yaml`), `frozen_config_snapshot/` (spec §0.1 rule 1), `gpatbench/` (spec §24 CLI `python -m gpatbench.cli`), `environments/`, `outputs/exploratory/`, `docs/spec/`, `tools/`, `configs/data_source_registry.yaml`, `configs/CONFIG_STATUS.md`.
- **Reason:** infrastructure required elsewhere in the spec or by the owner's M0 instructions. Purely additive; no spec directory removed or renamed.
- **Scientific impact:** none.
- **Affected experiments:** none.
- **Status:** UNAPPROVED — owner review required.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: INFRASTRUCTURE_ADAPTATION. Additional implementation/audit folders (e.g. `models/`, `environments/`, `gpatbench/`, `docs/`, `tools/`) are allowed provided no spec-mandated structure is removed or changes semantics.
  - Supersedes the UNAPPROVED line above (kept for history).

## DEV-003 — Spec internal inconsistency: `lambda_dir` (SPEC_INTERNAL_INCONSISTENCY / SOURCE_GAP)

- **Where:** §23.2 canonical GPAT-B0 YAML contains `lambda_dir: 0.5`.
- **Conflict:** no `λ_dir` term appears in the §10.3 loss-weight table nor in the §10.3 `L_G` formula. §10.2 `L_spec` contains an internal `0.5·||S_orient(x_hat) − S_orient(x_s)||_1` coefficient; whether `lambda_dir` denotes that coefficient or a separate loss is **not stated**. This session does not guess.
- **Action at M0:** `configs/methods/gpat_b0.yaml` is kept byte-verbatim from the spec (including `lambda_dir`). No code interprets it yet.
- **Affected experiments:** E08–E11 and ablations §17.1/§17.2 (M7).
- **Status:** UNAPPROVED / OPEN — must be resolved by the owner before M7; otherwise M7 is BLOCKED_BY_SOURCE_GAP for this term.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** UNAPPROVED
  - Owner classification: **UNRESOLVED**. Not approved and must not be resolved by the agent. The semantics and loss formula of `lambda_dir` must be traced before GPAT implementation/training (M7). Does not block M0.

## DEV-004 — M0 "environment locks": project environment not yet created

- **Spec M0 gate (§25):** "all dependencies/version hashes recorded; no method trained".
- **State:** no project Python environment exists yet. Laptop system Python 3.12.3 has no pip, and torch/torchvision/pandas/pyarrow/onnxruntime/scipy/scikit-learn/ptwt/transformers/timm are MISSING. GPU environment unknown (SSH needs interactive password). Therefore no environment lock hash can honestly be recorded.
- **What was recorded instead:** raw laptop state (`environments/laptop_environment_initial.txt`, SHA256 in `outputs/audit/ARTIFACT_INDEX.csv`) and the planned requirement list (`environments/PLANNED_ENVIRONMENT_REQUIREMENTS.md`) with every dependency marked AVAILABLE / MISSING / VERSION_UNVERIFIED / NOT_REQUIRED_AT_M0. No version hash is claimed for anything not installed.
- **Affected experiments:** all (a lock is required before any run, §0.1 rule 8).
- **Status:** UNAPPROVED — owner to accept this interpretation of the M0 gate or require environment creation first.
- **Status update 2026-09-18 (owner decision, recorded before M0 commit):**
  - **Status:** APPROVED
  - Owner classification: APPROVED_OPERATIONAL_DEVIATION (APPROVED_WITH_CONDITIONS).
  - Conditions:
    1. Dependencies that are not installed stay MISSING.
    2. Dependencies whose version is not verified stay VERSION_UNVERIFIED; nothing may be reported as VERIFIED without evidence.
    3. The environment a milestone needs must be created and frozen (lock + SHA256) before that milestone's scientific execution.
    4. The current laptop environment audit (`environments/laptop_environment_initial.txt`, `LAPTOP_ENVIRONMENT_AUDIT.md`) is kept unchanged as historical evidence.

---

## Open spec questions (not deviations yet — flagged for the milestone that needs them)

These were noticed while reading the full spec. Nothing below has been decided; each must be
resolved from the spec/source hierarchy or by the owner before the listed milestone, or be
marked BLOCKED_BY_SOURCE_GAP.

| ID | Milestone | Spec § | Question |
|---|---|---|---|
| Q-01 | M3 | 3.5 | Allocator objective names the terms (video-count deviation + penalties for missing binary class and attack_macro coverage) but not the penalty weights or search procedure (exact/greedy/random restarts). |
| Q-02 | M2 | 4 | SCRFD variant is not named (only "SCRFD ONNX, input 320"). Local candidates include `scrfd_10g_bnkps.onnx` and `det_2.5g.onnx`. |
| Q-03 | M2 | 4 | AdaFace IR-50 training-set variant (e.g. which pretraining dataset checkpoint) not named. |
| Q-04 | M1/M2 | 3.4 | Definition of a "valid frame" (decode success only, or face-detected) is not stated. |
| Q-05 | M7 | 5.3, 9.2, 10.3 | Identity adversary uses a GRL (coef 1.0) **and** `L_G` subtracts `λ_idadv·L_identity-adversary`; applying both could double-reverse the gradient. Intended sign convention needs confirmation. |
| Q-06 | M11 | 14.2 | DINOv3 table does not state warmup, scheduler/min LR, AMP or grad clip; only batch composition, augmentation, checkpoint/threshold are "same as ResNet-18". |
| Q-07 | M5 | 12.1 | ArtifactProbeNet input is 224×224 from 256×256 faces; resize vs crop method not stated. |
| Q-08 | M6 | 8.5, 30 | PCGAN (R7) has no code link in spec; status depends on code availability at implementation time. |
| Q-09 | M6 | 8.4, 30 | Physics-Guided STD (R2) has no code link; FAITHFUL_PAPER unless a verified official release is found. |
| Q-10 | M9 | 13 | LPIPS-Alex and KID (Inception feature extractor) implementations/weights not named. |
| Q-11 | M1 | 0 | Datasets are ≥ 3 local sources with mixed packaging (see `configs/data_source_registry.yaml`); whether raw data is referenced in place (read-only external path) or placed under `data/raw/` is not stated. |
