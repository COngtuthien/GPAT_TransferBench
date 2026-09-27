# GPAT-TransferBench v1.0 — E07c Auxiliary Conditioning Encoder Freeze Record (M6D6h)

| Field | Value |
|---|---|
| Record kind | **OWNER_DECISION / ASSET_FREEZE** (not an amendment; no A9) |
| Status | **OWNER_FROZEN**, **ADDITIVE**, **NON_DESTRUCTIVE** |
| Classification | `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION` |
| Method | E07c — DiffFAS-BIN-IDFREE |
| Role | `AUXILIARY_CONDITIONING_ENCODER` |
| Fidelity | `CONTROLLED_ADAPTATION` (unchanged) |
| Deviation | `DEV-021` (unchanged); no new deviation; no new fidelity class |
| Canonical record | `configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml` (SHA256 pinned in `methods/difffas/aux_checkpoint.py`) |
| Authority commit | `ee0a9b9166577cac606c142af5bc2e338d75eea1` (M6D6g recorded) |

## 1. Why this record exists

Amendment A3 §5.4b already fixed the rule: the controlled conditioning encoder is trained **exactly once**
(auxiliary seed 42) on the frozen GPAT TRAIN split; the checkpoint is chosen by the frozen **final-state**
rule; its **SHA256** is computed, recorded and **frozen**; and **that same frozen checkpoint** is reused by
**all** E07c main-training seeds.

M6D6g executed that single auxiliary run and **recorded** the final SHA256 with the historical status
`SHA256_RECORDED_PENDING_OWNER_FREEZE` (`authoritative_for_main_difffas = false`,
`owner_freeze_performed = false`). That status remains historically correct for M6D6g and is not edited.

M6D6h performs the pending **owner freeze** of exactly that identity. It introduces **no new scientific
rule**, rewrites **no historical config**, amendment, snapshot or evidence file, and performs **no training**.

## 2. The one frozen checkpoint

| Field | Value |
|---|---|
| Path contract (portable) | `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl` |
| Observed GPU path (provenance only) | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl` |
| SHA256 | `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c` |
| Size | `185136819` bytes |
| Format | `torch.save` of the **WHOLE** `nn.Module` (matches `torch.load(path).cuda()`) |
| Producer | M6D6g, auxiliary seed 42, ONE_LOGICAL_RUN, run_id `7014cdb366da52e4`, run_uuid `7031fd2f-92d7-465a-94d1-ab99bf8a3960` |
| Producer boundary | completed epoch 200, global step 11200 (11200 logical = physical optimizer steps, 0 superseded) |
| Selection | `FINAL_STATE_AFTER_EPOCH_200`; no VAL selection; no TEST use; no interruption; no resume |

The absolute GPU path is provenance only; it never replaces the `<runtime_root>` path contract.

**Exactly one checkpoint is authorized.** No alternate, fallback, "latest", CLI-override or
environment-variable-override checkpoint exists.

## 3. Main seeds

| E07c main seed | Conditioning encoder |
|---|---|
| 42 | frozen encoder `49a24a3a…fe107c` |
| 1337 | **the same** frozen encoder `49a24a3a…fe107c` |
| 2026 | **the same** frozen encoder `49a24a3a…fe107c` |

## 4. Consumption

The only authorized consumption seam is
`methods.difffas.aux_checkpoint.load_frozen_aux_encoder(runtime_root, recorded_sha256, config=None)`.
It (1) verifies the pinned canonical record; (2) refuses any caller SHA256 other than the frozen value
**before** the checkpoint file is opened; (3) resolves the configured external path under an absolute,
repository-external `runtime_root`; (4) requires the frozen size and verifies the asset through the
existing external-asset mechanism; then (5) keeps the M6D6c exact-byte order: `read_verified` →
SHA256 of the in-memory bytes → pinned source verification → `import torch` →
`torch.load(io.BytesIO(raw), weights_only=False)` → encoder identity check → device move.

No state_dict conversion, safetensors, TorchScript, `safe_globals` substitution or global `torch.load`
patch is introduced.

## 5. Scope

- Qualified consumer scope: **MAIN_DIFFFAS** authorization only. Main runner integration, main training
  graph, main resume and main production runner remain **unqualified**.
- Any future E07c consumer must refer to this same frozen identity, but **M8 remains unqualified**:
  M8 bank, M8 loader integration and the A7 M8 RNG-policy question are not decided here.

## 6. Failure policy

A missing checkpoint, a size mismatch, a SHA256 mismatch, or a caller-supplied SHA256 that differs from the
frozen value is **STOP_AND_REPORT**. There is **no automatic re-freeze and no automatic retraining**.
The checkpoint file is not chmod-ed, renamed, moved, copied, deleted, pruned or rewritten; protection is the
canonical path, frozen size, frozen SHA256 and fail-closed verification before deserialization.

## 7. The freeze operation itself

0 training runs; 0 optimizer steps; 0 backward calls; 0 checkpoint deserializations; no TRAIN, VAL or
TEST access; 0 M8 outputs; 0 main DiffFAS scientific runs.
