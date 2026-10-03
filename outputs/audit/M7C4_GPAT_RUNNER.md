# M7C4 — GPAT production runner + real-TRAIN data path + checkpoint/resume qualification

Authority: `cff688dd35524bf7f01c39916ef87cd4987b6470` (M7C3). GPU: RTX 3090 (`gpat-m7-gpu`, lock `24c983eb…`).
**Qualification only.** The qualification seed was 70404, and every artifact lives under
`<runtime_root>/qualification/m7/M7C4/` and is labelled QUALIFICATION_ONLY · NOT_SCIENTIFIC · NOT_ELIGIBLE_FOR_BANK /
SELECTION / PAPER_RESULT. Nothing was run in scientific mode: no 60-epoch or 10-epoch training, no scientific EMA
candidate, no bank, no VAL and no TEST. Machine-readable evidence: `M7C4_GPAT_RUNNER_QUALIFICATION.json`.

## Runner (`methods/gpat/runner*.py`, CLI `train-generator`)

| Area | Implementation |
|---|---|
| TRAIN relation | `manifests/pairs_train_v1.parquet` (SHA `a5e4fdae…`, A10-bound), 8838 rows, 8-column allowlist `pair_id, dataset, source_spoof_id, target_live_id, attack_macro, source_subject, split, split_manifest_sha256`. TRAIN membership comes from the relation's own split column and its bound split SHA; `split_v1`, `val_pairs` and TEST metadata are never opened. |
| Reader / firewall | The M6D6e `CanonicalFaceReader` (O_NOFOLLOW, no enumeration) over relation ids only. Every access is checked and logged in the main process by `AccessLog`. Decode: `x = u8 / 127.5 − 1` fp32, no augmentation. |
| Epoch order | `PCG64(int(SHA256('GPAT-M7|<generator\|warmup>|<mode>|<seed>|<epoch>')[:16],16)).permutation(8838)`, fixed as index lists before any worker runs. It is independent of num_workers (0 = 4 shown), PYTHONHASHSEED and filesystem order. |
| Loader | num_workers 4, pin_memory, persistent workers, prefetch 2 (execution only); seeded worker init. |
| Layout | 2210 microbatches of 4 (tail 2), 1105 groups (tail [4, 2]), weights 1/2 and 1/2 (tail 4/6 and 2/6), 66,300 updates. Warmup: 139 × 64 (tail 6) × 10 = 1390 steps. |
| Identity map | (dataset, source_subject) of CASIA/MSU, UTF-8 sort, K = 60, mapping SHA `3211de12…`. SiW rows are masked. |
| Ownership | G_OPT Adam(0.5, 0.999), wd 0: B0 G_res + E_art; B1 + attack head; B2 + identity head; B3 + both. D_OPT is the same over D only. WARMUP_OPT AdamW(E_art + attack head, wd 1e-4). G_SCALER, D_SCALER and WARMUP_SCALER are separate. |
| Teachers | AdaFace, FaceXFormer (exact-forward / surrogate-backward adapter) and F_art (ImageNet ResNet-18 on HP, GAP layer4), all fp32, eval and frozen. x_t targets are computed under `no_grad`. ArtifactProbe is never loaded. |
| Step (N-04) | For each microbatch: one GPAT forward. G side: D frozen, x_hat not detached, `w_j · assemble(…)` plus λ_idadv · DEV-022 share (never multiplied by w_j). D side: `w_j · 0.5·(BCE(D(x_s),1) + BCE(D(x_hat.detach()),0))`. At the boundary: D unscale → clip 1 → step → update, then the same for G. EMA follows the G step when active. |
| Schedules | `main_lr(u)` and `curriculum(u)` are set per group, with u = (epoch−1)·1105 + group. D trains from u = 1. |
| EMA / candidates | EMA initialized from the live E_art/G_res at the end of epoch 5 and updated for u > 5525. One immutable candidate `ema_epoch_NN.pt` per epoch 10..60 (51), containing E_art EMA + G_res EMA + metadata, selected = false, no VAL field. |
| Recovery | `checkpoints/recovery/latest.pt` (and `warmup_latest.pt`), written only at optimizer boundaries, atomically (tmp → fsync → replace → fsync dir), `weights_only`-loadable. It holds all modules, optimizers, scalers, EMA, position, identity SHA, order authority, RNG and provenance. SIGINT/SIGTERM finish the current group, then save. |
| Run directory | `<rt>/runs/m7/<E08..E11>/seed_<42\|1337\|2026>/` with the run_logging_v1 file names, plus `identity_class_map.json` and `checkpoints/{ema_candidates,recovery}`. |
| CLI | `python -m gpatbench.cli train-generator --method GPAT-B<k> --seed <s>`. It refuses: non-scientific seeds (70404 included), a dirty tree or wrong branch, a missing PYTHONHASHSEED or launch environment, a wrong interpreter, asset hash mismatches, and an existing root without `--resume`. It was not executed. |

## Real-TRAIN qualification (final run; bitwise identical to the first run)

| Scenario | Result |
|---|---|
| B0 regular [4,4] (positions 0..7, u = 1) | All losses finite. G_res and E_art gradients are finite and non-zero (E_art 2.8e-5). D finite. Teacher grads None. lr = 0 at u = 1 as frozen. |
| B3 regular [4,4] | L_type and L_idadv finite, 6 labelled rows. Attack and identity heads receive gradients; the GRL path is active. |
| B3 tail [4,2] (positions 8832..8837, u = 1105) | Weights 4/6 and 2/6. The weighted microbatch means equal the full-6 batch means within 1.4e-6 absolute (bound 5e-6 + 1e-4·|v|, fp32 kernel noise). |
| DEV022_QUALIFICATION_GROUP | Labelled [2, 1]. L_idadv = 4.0573200 equals ΣCE/3 = 4.0573203, and differs from the sample-weighted 4.0560656. |
| Warmup batch 64 / tail 6 | lr 1e-4 and attack_warmup_lr(139). BN in train mode with buffers updated; AdamW state created; WARMUP_SCALER on. No teacher loaded and no target image opened. |
| Warmup handoff | E_art and attack head preserved. WARMUP_OPT and WARMUP_SCALER dropped. A fresh G_OPT (empty state, covers all G-owned params) and a fresh G_SCALER. |
| Generator resume (fresh process) | Bitwise equal to the uninterrupted path after group 2: all modules, BN buffers, optimizers, scalers, position, RNG, group-2 inputs, x_hat and losses. |
| Warmup resume (fresh process) | Bitwise equal after batch 2. |
| EMA candidate writer | Schema, weights-only load, E_art + G_res EMA only, selected = false, immutable, no tmp leftovers. Bytes deleted after hashing. |
| Firewall | 514 TRAIN image opens over 177 unique ids (SHA `d03bf3ba…`). VAL, TEST and non-TRAIN images and metadata: all 0. |
| VRAM (reserved, peak) | B0 group 3.96 GiB; B3 group 3.96 GiB; B3 tail 4.06 GiB; warmup 64 2.14 GiB; overall maximum 4.35 GiB. |
| Group wall time (first group, cold) | B0 2.81 s; B3 2.81 s; B3 tail 3.95 s; warmup 64 0.55 s. |

Qualification recovery checkpoints are kept, labelled, outside git: generator `48d15cdc…`, warmup `15898d0e…`.
They are byte-identical across the two runs.

## Owner decisions at M7C4 review

They are frozen in `configs/amendments/gpat_m7c4_runner_resolution.yaml` (ADDITIVE_RUNNER_IMPLEMENTATION_RESOLUTION; not
an amendment; no DEV).

- **Epoch order** (OWNER_IMPLEMENTATION_CLARIFICATION):
  - payload `GPAT-M7|{stage}|{MODE}|{seed}|{epoch}` in UTF-8, with MODE ∈ {`SCIENTIFIC`, `QUALIFICATION`}. The tokens are
    upper case, exactly as implemented and qualified; the owner confirmed this when the lower-case reading was found
    to change every recorded order;
  - `seed64 = int(sha256(payload).hexdigest()[:16], 16)`, i.e. 64 bits;
  - `numpy.random.Generator(PCG64(seed64)).permutation(8838)`.

  Recorded epoch-1 hashes:
  - seed 42: `2b8961ce…`
  - seed 1337: `4591f8e1…`
  - seed 2026: `11d4b830…`
  - seed 70404: `11821618…`
- **Asset config** `configs/execution/gpat_m7_assets_3090.yaml` is EXECUTION_ONLY_HOST_BINDING. Every entry carries a
  path, its authority SHA-256, an identity and a role. The runner resolves the path, requires the file, verifies the
  SHA-256, and only then loads it. A changed SHA is a hard failure.
- **FAIL_CLOSED_AMP_OVERFLOW.** The candidate's "GradScaler overflow = logged skipped step" behaviour was not approved
  and has been replaced:
  - A non-finite loss fails before backward.
  - D and G are both unscaled first, then every optimizer-owned gradient is scanned before clipping or stepping. Any
    NaN/Inf raises `AmpOverflowStop`: neither optimizer steps, no scaler update, no update-index or schedule advance,
    no EMA, and no recovery for the failed group. The same applies to warmup gradients.
  - A non-finite parameter after the steps raises `FAILED_NUMERICAL_POST_STEP`.
  - The failure record holds the type, epoch, attempted group, scales, optimizer and parameter names/count, and the
    hash of the last safe recovery. The process exits non-zero.
  - The previous safe recovery stays the only resumable authority, and the failed group is never retried or skipped
    automatically.
  - GradScaler settings are unchanged.
- **Atomic D/G group.** A group completes only when both the D and G steps succeed. Because the finiteness scan
  precedes either step, "D stepped and G skipped" (or the reverse) is structurally impossible.
- **Post-edit finite smoke.** B0 regular, the generator resume reference and warmup batch 64 were re-run with the
  final code. All are bitwise identical to the accepted qualification evidence (states, inputs, x_hat, losses,
  gradients), so finite execution is unchanged.
