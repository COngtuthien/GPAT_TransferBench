# M6D6jR — E07c science-launch owner decision (O3 smoke timing, O4 resume)

**Record kind:** OWNER_DECISION / SCIENCE_LAUNCH_POLICY · **Classification:** DETERMINISTIC_IMPLEMENTATION_CLARIFICATION ·
**Authority:** `ade5900db1dbff73917a2eb2280f8f2fe7afc418` (M6D6j) · no amendment, no A9.
**Record:** `configs/amendments/e07c_m6d6jr_science_launch_decision.yaml`
(SHA256 `2f7e5460314664fd0006b0b00a74707f79df5b427a6ccce084ec8dac3ddca830`).
E07c remains CONTROLLED_ADAPTATION, DEV-021 (`new_deviation = false`, `new_fidelity_class = false`); method status
IMPLEMENTED_NOT_EXECUTED.

## Evidence re-derived from committed bytes (stdlib; no GPU, no data)

| Fact | Source |
|---|---|
| Milestone order M6 Baselines (¶1258) < M7 GPAT (¶1261) < M8 Banks (¶1264) < M9 Generator eval (¶1267) < M10 ResNet downstream (¶1270) < M11 DINOv3 downstream (¶1273) | frozen spec `f7d23716…` |
| §26 "A full deterministic smoke test on a 2-subject-per-class toy subset must finish before full runs." (¶1295) | frozen spec |
| §32 "smoke test passes for generator → bank → evaluator → metrics → plots." (¶1426) | frozen spec |
| §27 "One seed fails — rerun same seed after technical fix; never replace with a different seed" (¶1315–1316) | frozen spec |
| CLI options exactly `--seed`, `--execution-config`, `--preflight-only`; every resume / pretrain flag refused; existing run root refused | `tools/run_e07c_main.py`, `main_runner_io.py` |
| Run context fresh only (`resume = False`, existing root refused) | `methods/difffas/main_runner.py` |
| M6D6j: MAIN_PRODUCTION_RUNNER qualified; MAIN_CHECKPOINT_RESUME unqualified; O3/O4 recorded as the pending pre-science blockers | M6D6j contract + evidence |

## Decisions

- **O3 = STAGED_MILESTONE_PRECEDENCE_CLARIFICATION → RESOLVED_STAGED_MILESTONE_PRECEDENCE.** M6/M7 generator training may
  proceed once the generator's production runner / data / checkpoint / firewall contracts are qualified. The smoke test is
  retained (not deleted, not waived): smoke_test_required = true, smoke_test_completed = false,
  blocks_E07c_M6_training = false, blocks_full_M8_plus_execution_until_completed = true. It must exercise
  generator → bank → evaluator → metrics → plots on the deterministic 2-subject-per-class toy subset. M6D6j is not that
  smoke test; no scientific result may come from it.
- **O4 = RESUME_NOT_REQUIRED_BEFORE_SCIENCE → RESOLVED_NOT_REQUIRED.** MAIN_CHECKPOINT_RESUME stays UNQUALIFIED; M6D6k is not a
  prerequisite. Scientific runs are FRESH_ONLY. Recovery policy FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX: preserve the
  failed run's evidence, diagnose, and after an authorized technical fix rerun the SAME seed from the beginning under the
  same frozen settings; never substitute a seed, never rerun for a better result.

## Launch status

FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION = false. E07c main scientific training is
AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED; nothing is marked completed. The run contract is unchanged: seeds
42 / 1337 / 2026; per seed 400 epochs, batch 4, drop_last false, 2210 iterations per epoch, 884000 optimizer iterations;
exact-source visualization every 1000 steps; periodic checkpoints every 10000; terminal at 884000 after the final
visualization; M6D6iR successor-gated pruning.

Qualified: E07c_MAIN_SCIENTIFIC_LAUNCH_POLICY_RESOLVED, E07c_MAIN_SCIENTIFIC_RUNS_AUTHORIZED_TO_LAUNCH. Remains qualified:
MAIN_PRODUCTION_RUNNER. Not qualified: MAIN_CHECKPOINT_RESUME, MAIN_DIFFFAS_SCIENTIFIC_TRAINING, M8_BANK.

## Operation

0 training runs, 0 scientific runs, 0 optimizer steps, 0 checkpoint writes, 0 TRAIN / VAL / TEST image reads; GPU not
contacted.

## Seed-42 launch plan (NOT EXECUTED; only after this decision is committed and the GPU repo is clean at that commit)

```
# on the GPU host, gpat-m6-e07c, clean worktree on m6-baselines at the committed M6D6jR SHA, inside tmux
cd /home/student20261/workdir/GPAT_TransferBench
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/student20261/miniconda3/envs/gpat-m6-e07c/bin/python tools/run_e07c_main.py --seed 42 \
    --execution-config configs/execution/m5_gpu_3090.yaml --preflight-only      # gates only, no torch
# then, if the preflight plan is clean, the same command without --preflight-only
```
