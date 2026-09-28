# M6D6jF: E07c SCIENTIFIC RunContext seed-metadata hotfix

Classification: implementation bugfix / DETERMINISTIC_IMPLEMENTATION_CORRECTION. Not an amendment; no A9; no new
deviation. Fidelity remains CONTROLLED_ADAPTATION / DEV-021.

- Authority: `8358d8b6fe468478ad86a715616becb81bb9b339` (M6D6jR: authorize E07c main scientific launch)
- Record: `configs/amendments/e07c_m6d6jf_scientific_context_hotfix.yaml`
  (sha256 `946ca134ef514650002f368a8d87715f16a80855d10725587d9e6e855747af13`)
- Only implementation file changed: `methods/difffas/main_runner.py` (class `E07cMainRunContext` only)

## Failed attempt (not a scientific result)

| field | value |
|---|---|
| seed / attempt | 42 / 1 |
| authority | 8358d8b6fe468478ad86a715616becb81bb9b339 |
| exit_status | 1 |
| failure_phase | RUN_CONTEXT_OPEN |
| scientific_optimizer_steps | 0 |
| scientific_checkpoint_writes | 0 |
| scientific_result | false |
| experiment_seed_consumed_as_completed_run | false |
| recovery | FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX |

Exception: `TypeError: dict.update() got multiple values for keyword argument 'experiment_seed'`, raised in
`E07cMainRunContext._manifest` via `RunContext._open_locked`, before `build_production`, TRAIN reads, model
construction or the training loop. The failed runtime root is preserved under the recovery namespace. No other runtime
facts are recorded, because none were supplied.

## Root cause and defect sites

In SCIENTIFIC mode `seed_field` is `'experiment_seed'`, so splicing `**{self.seed_field: self.run_seed}` next to an
explicit `experiment_seed=self.seed` supplies that key twice.

| method | defect | reached in launch |
|---|---|---|
| `_manifest` | duplicate kwarg -> TypeError | yes (observed) |
| `_open_locked` (checkpoint_index) | duplicate kwarg -> TypeError | no (latent) |
| `close` | summary carried `experiment_seed` -> RunContext.close raises `summary may not overwrite run identity` (also on the failure `__exit__` path) | no (latent) |
| `_resolved_config` | duplicate dict key, silent overwrite (same value) | n/a |
| `log_event` | duplicate dict key, silent overwrite (same value) | n/a |

## Fix

One canonical construction, `E07cMainRunContext._seed_metadata()`:

- SCIENTIFIC: `{"experiment_seed": <seed>}`
- QUALIFICATION: `{"experiment_seed": null, "qualification_seed": 60608}`

It is used by `_open_locked`, `_resolved_config`, `_manifest` and `log_event`. `close` passes only the
non-`experiment_seed` keys, because `RunContext.close` owns `experiment_seed`. The QUALIFICATION output (all run files,
with key order preserved) is identical to the authority implementation; a regression test checks this.

## Unchanged

Seeds 42/1337/2026, 400 epochs, batch 4, drop_last false, 2210 it/epoch, 884000 optimizer steps, the optimizer,
scheduler, EMA, RNG order, visualization every 1000 steps, DDPM, checkpoint cadence 10000, terminal ordering, M6D6iR
pruning, the encoder, the TRAIN manifest, the firewall, CLI fresh-only and resume status are all unchanged.
MAIN_CHECKPOINT_RESUME remains UNQUALIFIED. The M6D6j/M6D6jR records, evidence and tests are not rewritten; M6D6jR
remains historically correct at its commit.

## Regression

`tests/test_m6d6jf_e07c_scientific_context_hotfix.py` opens and closes the real `E07cMainRunContext` in a temporary
runtime root in both modes, without Torch, GPU or data.

## Rerun (NOT EXECUTED)

After the owner commits this hotfix, the plan is a fresh same-seed rerun on the GPU host from a clean worktree at the
M6D6jF commit. The scientific root `<rt>/runs/m6/E07c/seed_42` must not exist; the failed attempt stays in the recovery
namespace.

```
/home/student20261/miniconda3/envs/gpat-m6-e07c/bin/python tools/run_e07c_main.py --seed 42 \
    --execution-config configs/execution/m5_gpu_3090.yaml --preflight-only      # gates only, no torch
# then, if the preflight plan is clean, the same command without --preflight-only
```
