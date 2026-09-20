# M5 — Validation-Epoch Contract Correction

**Date:** 2026-09-21 · **Classification: `CONFIG_SEMANTIC_AMBIGUITY_FOUND_BEFORE_EXECUTION`**
**Not** an executed scientific defect and **not** a result invalidation — no model has ever been
trained in this project.

## 1. The ambiguous representation

The frozen ArtifactProbeNet config expressed the validation schedule as a two-element list:

```yaml
validation:
  checkpoint:
    evaluate_epochs: [1, 30]
```

Read as a range this means "validate at the end of every epoch from 1 to 30". Read as a literal
list — which is what the YAML actually is — it means "validate at epochs 1 and 30 only". Those are
30 validation passes versus 2, and they would select different checkpoints. The notation could not
distinguish them, so it had to go.

## 2. What the repository search showed

`grep -rn "evaluate_epochs"` over the whole repository (excluding `.git`, the virtualenvs and the
pinned third-party source cache) returned exactly two hits:

```
configs/frozen/artifact_probe.yaml:234:    evaluate_epochs: [1, 30]
frozen_config_snapshot/configs/frozen/artifact_probe.yaml:234:    evaluate_epochs: [1, 30]
```

The second is the byte-identical snapshot of the first, so the field existed in **one** place.
A matching search across `gpatbench/`, `tests/` and `tools/` returned **no** hits: **no code read
the field.** Nothing branched on it, nothing derived a loop bound from it, and no test asserted it.

## 3. Why nothing was invalidated

- The field was never consumed, so it could not have influenced any computation.
- `gpatbench/probe/train.py::run(dry_run=False)` raises `ProbeContractViolation` **before any
  optimization**; the authoritative training loop is not implemented yet.
- No checkpoint exists and `models/artifact_probe/` does not exist.
- M5 was and remains `NOT_STARTED`.

The ambiguity was therefore found **before execution**, which is exactly when a contract defect
costs nothing. It is recorded rather than quietly patched.

## 4. The owner contract, stated explicitly

ArtifactProbeNet is validated at the end of **every** epoch — `1, 2, 3, …, 30` — which is exactly
**30 validation passes**. The best checkpoint may come from **any** epoch in that range. Selection
replaces the best only when `new_macro_f1 > best_macro_f1` (strict `>`, never `>=`), so an exact tie
keeps the **earlier** epoch. TEST never participates.

## 5. The new representation

```yaml
validation:
  checkpoint:
    decision: RESOLVED_BY_OWNER_MAX_VAL_MACRO_F1_EARLIEST_TIE
    evaluate_every_epoch: true            # a validation pass at the END of every epoch
    epoch_start: 1
    epoch_end: 30                         # => 30 validation passes: 1, 2, 3, ..., 30
    evaluation_count: 30
    best_may_come_from_any_epoch_in_range: true
    rule: "replace best only when new_macro_f1 > best_macro_f1 (strict >, never >=)"
    tie_break: earlier_epoch
    epsilon_tie_window: NONE
    training_loss_as_tie_break: false
    test_involvement: NONE
    freeze_after_selection: true
    sequence_source: gpatbench.probe.contract.validation_epochs
    range_as_two_element_list: FORBIDDEN
```

Everything else in the checkpoint block is preserved unchanged.

## 6. The single source of the sequence

`gpatbench.probe.contract.validation_epochs()` derives the schedule from the frozen contract and
returns `tuple(range(epoch_start, epoch_end + 1))`. It hard-fails if `evaluate_every_epoch` is not
`true`, if `epoch_start != 1`, if `epoch_end != EPOCHS`, or if the resulting length is not `EPOCHS`.

**No second copy of `[1..30]` exists anywhere.** `train.preflight()` calls this helper and reports
`validation_epochs` / `validation_passes`, and the exact sequence is:

```
(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)        # 30 entries
```

`contract.select_best_epoch()` was added alongside it as a pure, side-effect-free implementation of
the D-M5-06 rule, so the selection behaviour can be tested without training anything. On a synthetic
sequence peaking at **epoch 17** it selects epoch 17 — an intermediate epoch is reachable, which is
the practical consequence of fixing the ambiguity.

## 7. Trainer status — unchanged and honestly stated

The authoritative training loop is **still not implemented**. `run(dry_run=False)` still refuses,
and on this CPU-only machine the E-M5-01 guard fires first. The module docstring and the refusal
message now state the per-epoch validation requirement explicitly, without pretending the loop
exists.

## 8. Config hashes

| | sha256 |
|---|---|
| before | `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` — **SUPERSEDED_BY_VALIDATION_EPOCH_CONTRACT_CORRECTION**, retained in the audit history |
| after | `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe` |
| snapshot | `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe` — byte-identical |

## 9. What did not change

No other scientific choice was touched: the 7-class population and counts, the class weights, the
high-pass transform, the ResNet contract and weight hash, AdamW, the cosine schedule, the
DataLoader, the macro-F1 definition, the strict tie-break and the embedding are all exactly as
frozen on 2026-09-20. The CPU-safe smoke still passes 22/22 against the corrected config.

## 10. Tests

Added to `tests/test_m5_probe_contract.py`: `validation_epochs()` equals `tuple(range(1, 31))`;
length 30; first 1; last 30; 2 and 29 present; every consecutive difference is 1; the parsed config
no longer carries an `evaluate_epochs` key; `evaluate_every_epoch` is `true` with `epoch_start = 1`
and `epoch_end = 30`; the helper rejects a wrong start, a wrong end and a false
`evaluate_every_epoch`; the replacement rule is `>` and not `>=`; exact ties keep the earlier epoch;
and a maximum at epoch 17 is selectable. **No model is trained by any of them.**
