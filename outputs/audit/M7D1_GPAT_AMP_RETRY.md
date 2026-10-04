# M7D1-N1 — GPAT AMP overflow diagnosis and ATOMIC_AMP_BACKOFF_RETRY resolution

Authority: `6cf271ebd0d658dcc5384da00265326e7bac6476` (M7C4). GPU: RTX 3090 (`gpat-m7-gpu`, lock `24c983eb…`).
**Qualification only.** No scientific run was started, resumed or restarted. Every artifact lives under
`<runtime_root>/qualification/m7/M7D1_N1/` and is labelled QUALIFICATION_ONLY · NOT_SCIENTIFIC · NOT_ELIGIBLE_FOR_BANK /
SELECTION / PAPER_RESULT. Machine-readable evidence: `M7D1_GPAT_AMP_RETRY_QUALIFICATION.json`. Additive record:
`configs/amendments/gpat_m7d1_amp_retry_resolution.yaml` (ADDITIVE_RUNTIME_NUMERICAL_STABILITY_RESOLUTION).

## A — failed attempt 2 preserved

GPAT-B0 / E08 / seed 42, run_id `1215d39ebc184ab9`: completion_status FAIL_CLOSED_AMP_OVERFLOW, global_update 1970,
VAL 0 / TEST 0 (31532 TRAIN opens). The recovery `latest.pt` keeps SHA `a34e73d1…` (end of epoch 1, update 1105).
Every file of the run root was hashed before and after the milestone, and the tree hash is identical. The run root is only
read: the trainer loaded a byte-identical copy of the recovery placed in the qualification root.

## B/C — exact reproduction and pre-update-1971 snapshot

The runner was unmodified (M7C4 blob). The order was the scientific seed-42 epoch-2 order, with the determinism and
launch environment of `run_scientific`.
- Updates 1106..1970: all 865 optimizer_group records are **bitwise equal** to the scientific metrics, except
  utc / wall time / peak memory.
- At 1971, AmpOverflowStop fired: D_OPT only, `discriminator.model.0.weight` and `discriminator.model.2.weight`, G
  finite, both scales 65536. The failure record equals the scientific one. Exact reproduction confirmed.
- Snapshot (`snapshot/pre_update_1971.pt`, SHA `5e91e9f5…`): all modules with their buffers, G/D optimizers, both
  scalers, Python/NumPy/CPU/CUDA RNG, position, and the exact group tensors (pair indices [[1106, 5469, 5701, 3224],
  [430, 2940, 2660, 8000]]).
- The state was re-verified by digest after every reload.

## D — loss-scale probe (update 1971, G scale 65536, fresh snapshot state before every probe)

| D scale | D finite | non-finite D grads | D grad norm (unscaled) | G finite | G grad norm | D/G steps |
|---:|:--:|---|---:|:--:|---:|:--:|
| 65536 | no | model.0.weight (288 el.), model.2.weight (16 el.) | — | yes | 0.03109 | 0/0 |
| 32768 | no | model.0.weight (84 el.) | — | yes | 0.03109 | 0/0 |
| 16384 | yes | — | 43.562 | yes | 0.03109 | 1/1 |
| 8192 | yes | — | 43.563 | yes | 0.03109 | 1/1 |
| 4096 | yes | — | 43.563 | yes | 0.03109 | 1/1 |

The microbatch losses are identical at every scale: L_G 0.19105 / 0.21893, L_D 0.0053636 / 0.036739. Post-step
parameters were finite. The snapshot file was unaltered. **Classification: LOSS_SCALE_OVERFLOW_CONFIRMED.**

## E/F — ATOMIC_AMP_BACKOFF_RETRY and restore equivalence

`methods/gpat/runner.py` (additive). The production `Trainer` uses ATOMIC_AMP_BACKOFF_RETRY. GeneratorStep and
WarmupStep keep the M7C4 fail-closed policy as their default and as the terminal fallback.

How a group runs:
- `PreGroupState` saves every core buffer, the RNG and the scaler states before the group.
- Both scalers are unscaled and both are scanned before either optimizer steps.
- On a non-finite gradient: the saved state is restored and verified, gradients are zeroed, only the offending
  scaler(s) back off (×0.5, growth tracker 0), and the same group is recomputed at the same u.
- Each retry logs an `amp_retry` event.
- If the offending scale is already ≤ 1.0, the run stops with FAIL_CLOSED_AMP_OVERFLOW_FINAL.
- A recovery checkpoint is refused while a group is in flight.

What the qualification showed on update 1971:
- The failed attempt really mutated the E_art BN buffers. It changed no parameter and no optimizer moment or step
  count; it only re-set lr = main_lr(1971), which is a pure function of u. The GPAT step consumes no global RNG.
- Run A (retry policy): 65536 → D overflow → restore → 32768 → D overflow (model.0.weight) → restore → **16384
  accepted** (3 attempts, 1 D step, 1 G step).
- Run B: a single attempt from the original pre-group state at D = 16384.
- A and B are bitwise equal on trainable parameters, BN buffers, optimizer states, scaler states, RNG, the group record
  and the per-microbatch capture. Post-group digest: `d744004d…`.
- Negative control: retrying without the restore leaves different E_art BN buffers, so restoring the buffers is
  necessary.

## G — exact group plus 20 updates (real loader, production policy)

- The loader rebuilt group 866 and its tensors match the snapshot digest; the state at entry equals the snapshot.
- Update 1971 was accepted at D 16384 after 2 `amp_retry` events. Its record equals run A's.
- Updates 1971..1991 are contiguous with finite losses and gradient norms. D:G steps were 21:21.
- 336 TRAIN accesses match the plan exactly, with no duplicate group consumption. VAL/TEST/non-TRAIN access was 0.
- **Observation for the owner:** a second normal backoff happened at update 1985 (D 16384 → 8192). Unscaled D gradient
  norms rose to about 100–133 by 1985–1991 (clipped to 1.0), and D_total rose to about 0.74. Everything stayed finite.

Finite path under the new code: updates 1106..1125 replayed from the recovery copy are bitwise equal to the
scientific records, with no retry.

## H — tests

`tests/test_m7d1_gpat_amp_retry.py` covers:
- D, G and both overflowing;
- multiple backoffs, and the final stop at scale 1;
- BN and RNG restore;
- no step on a failed attempt and one D plus one G step on success;
- one advance each of the update index, the schedule and the EMA;
- access logged once per group;
- no checkpoint while a group is in flight;
- a recovery round trip that keeps the reduced scale;
- warmup retry and warmup final stop;
- hard stops (non-finite loss, post-step) never being retried;
- the step default staying M7C4 fail-closed.

## Harness disclosures

- Probe attempt 1 aborted on a harness bug before any probe ran. `Optimizer.load_state_dict` keeps the CPU Adam
  `step` tensors by reference, so a stepping probe altered the in-memory snapshot. The fix is a deep copy. The aborted
  root was moved aside as `_aborted_probe_attempt1_snapshot_alias_bug`.
- Equivalence was rerun after the failed-attempt optimizer check was refined to separate the state tensors from the
  per-u lr. Results were identical, with the same post-group digest. The first root is kept as
  `_superseded_equivalence_attempt1`.
- The assemble step ignores the metrics envelope keys (record_type / epoch / group) when it compares records.
