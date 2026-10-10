# GPAT-TransferBench v1.0 — M7D1-A1 adversarial curriculum amendment

Milestone M7D1-N6A, 2026-10-10. Status: **ADOPTED_FOR_FRESH_SCIENTIFIC_USE**.
Machine-readable record: `M7D1_A1_ADVERSARIAL_CURRICULUM_AMENDMENT.json`.

**Naming.** The identifier is **M7D1-A1**. `docs/spec/amendments/` already contains an unrelated
v1.0 *Amendment A1* (Fair ID-Free Main Track); M7D1-A1 is a different record. It was not part of
the original frozen specification and does not rewrite it.

## Authority

| Item | Value |
|---|---|
| Original frozen spec | `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx`, sha256 `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` (unmodified; historical authority for the original run) |
| Pre-amendment HEAD | `162bdfd4cca8e83926ef6f64ff46da2bef1bca66` |
| N3 (root cause) | `d88563a46757d171a5e2af359eebc4fc18c305eb` |
| N4 (repair) | `6fc875ef1c63f0d512e877643e8005879eb8c0cc` |
| N5 (stage 3) | `162bdfd4cca8e83926ef6f64ff46da2bef1bca66` |
| Owner approval | recorded in M7D1-N6A (decisions A and B below) |

## Changed field: `lambda_adv(u)` only

| Updates | ORIGINAL v1.0 | AMENDED M7D1-A1 |
|---|---|---|
| u1..u5525 | 0.0 | 0.0 |
| u5526..u6630 | 0.05 (**jump 0 → 0.05 at u5526**) | `0.05 * ((u - 5526) / 1104)` (one-epoch linear ramp, 1105 updates) |
| u6631..u16575 | 0.05 | 0.05 |
| u16576..u66300 | 0.10 | 0.10 (**stage-3 jump 0.05 → 0.10 at u16576 unchanged**) |

### Floating-point definition

The canonical expression is `0.05 * ((u - 5526) / (6630 - 5526))` in IEEE-754 binary64. It is implemented in
`methods/gpat/runtime_contract.py::lambda_adv_a1`, and the production `curriculum` is `curriculum_a1`. The value is
bitwise equal to the N4 R1 schedule on u5526..u16575.

Do not substitute the algebraically equal form `(0.05 * (u - 5526)) / 1104`. It differs in the last bit at 378 of
the 1105 ramp updates.

| u | lambda_adv (A1) | hex | original |
|---|---|---|---|
| 1 | 0.0 | 0x0.0p+0 | 0.0 |
| 5525 | 0.0 | 0x0.0p+0 | 0.0 |
| 5526 | 0.0 | 0x0.0p+0 | 0.05 |
| 5527 | 4.528985507246377e-05 | 0x1.7beb3922e017cp-15 | 0.05 |
| 6078 | 0.025 | 0x1.999999999999ap-6 | 0.05 |
| 6629 | 0.04995471014492754 | 0x1.993a9ecb50e1ap-5 | 0.05 |
| 6630 | 0.05 | 0x1.999999999999ap-5 | 0.05 |
| 6631 | 0.05 | 0x1.999999999999ap-5 | 0.05 |
| 16575 | 0.05 | 0x1.999999999999ap-5 | 0.05 |
| 16576 | 0.1 | 0x1.999999999999ap-4 | 0.1 |
| 66300 | 0.1 | 0x1.999999999999ap-4 | 0.1 |

### Unchanged

These items are unchanged:

- Curriculum values: the s_hf schedule, lambda_con, lambda_spec, and the stage boundaries.
- Optimisation: the main and attack-warmup learning rates, the optimizer, Adam betas, accumulation and batch size.
- Training controls: the AMP policy (ATOMIC_AMP_BACKOFF_RETRY), gradient clipping and EMA.
- Model: the G and D architectures, composition, and every loss term and fixed weight.
- Data: the pair manifest and data order.
- Evaluation: checkpoint eligibility, the VAL selection rule and the TEST policy.
- The mask parameterization.
- The frozen method configs `gpat_b0..b3.yaml` and their sha256. Their descriptive curriculum tables remain v1.0
  text, and the runtime schedule comes from `runtime_contract`.

The original schedule is preserved as `runtime_contract.curriculum_v1_0`, which equals `curriculum` at 162bdfd for
every u.

Every method (GPAT-B0..B3) calls `runtime_contract.curriculum(u)`, so the amendment applies to all of them in the
same way.

## Evidence chain

- **N3:** the lambda_adv jump at u5526, against a D that stage 1 had made overconfident, caused the seed42
  closed-mask identity collapse at u5781.
- **N4:** R1 (a one-epoch ramp to u6630) and R2 both repaired the known mechanism through u16575. R1 is preferred
  as the smaller deviation.
- **N5:** the original stage-3 transition S0 qualified through u27625, so no second ramp is required.

## Owner decisions (recorded in N6A)

- **A. N5 anchor accepted, for N5 diagnostic provenance only.**
  - Base: N4 `pre_update_16575.pt`, sha256 `ea26d463…2f741a612`.
  - One exact replay of u16575, bitwise equal to N4 R1, with no AMP event.
  - Result: `pre_update_16576.pt`, sha256 `99d06834…2c76`.
  - The anchor is never used to resume a scientific run.
- **B. The M7D1-A1 schedule is approved for a fresh scientific run starting at u1.**

## Run identity

- Every scientific run records `curriculum_id = GPAT-TransferBench-v1.0+M7D1-A1`. It is stored in the run manifest
  provenance and in the recovery and EMA-candidate metadata, and resume refuses a mismatch.
- Scientific roots moved to `<runtime_root>/runs/m7_a1/<E08..E11>/seed_<seed>`.
- The v1.0 roots under `runs/m7/` are immutable historical evidence. This includes B0/E08/seed42, run_id
  `7b799fbd6d0426da` at code `058e976`.

**Planned N6B run.** This run is not created in N6A.

| Field | Value |
|---|---|
| Method / experiment / seed | GPAT-B0 / E08 / 42 |
| Start | fresh from u1, with no snapshot or anchor |
| Root | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7_a1/E08/seed_42` |
| run_id | `sha256("GPAT-B0|42|0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a|<launch commit>")[:16]` |
| Launch commit | the M7D1-A1 adoption commit, or an owner-approved descendant |

## Checkpoint policy for N6B

- Train the full 60 epochs (66300 updates) unless a true hard failure occurs.
- The eligible candidates are the EMA checkpoints (E_art, G_res) for epochs 10..60.
- Selection is done per seed and uses VAL only, with the frozen spec §10.6 percentile-rank rule:
  - SelectionScore = 0.25·rank(ID) + 0.15·rank(Dice) + 0.25·rank(ArtSim) + 0.20·rank(−NME) + 0.15·rank(−LFErr).
  - Ties go to the higher ArtSim, then the earlier epoch.
- Use the best selected EMA checkpoint, not the final epoch by default.
- TEST is never used for selection.
- No selection is performed in N6A.

## Limitations

- The qualification covers seed42 only.
- The stage-3 qualification only reached u27625 of 66300.
- D keeps strengthening (D_total falls) and the gadv share of the G gradient rises.
- G clipping is frequent (about 100% in N5 stage 3).
- The mask saturates open (M_mean ≈ 1). This is recorded as an observation to monitor.
  - M7D1-A1 adds no mask floor, ceiling, regularizer, stopping criterion or loss threshold. Any of those would be a
    separate scientific change.
- There is no VAL/TEST quality evidence yet.

## Historical tools

The N3 and N4 diagnostic tools use `rc.curriculum` as their frozen base. To reproduce them on the v1.0 trajectory,
use their authority commits. N5 starts at u16576, where M7D1-A1 equals v1.0. No N3, N4 or N5 evidence was modified.
