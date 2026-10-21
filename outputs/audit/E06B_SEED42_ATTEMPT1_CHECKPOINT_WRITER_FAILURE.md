# E06b seed42 attempt 1: checkpoint-writer failure

**Classification:** `FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG`. This is not scientific instability, a numerical failure, OOM, a data failure, a protocol failure, or a completed seed.

**Disposition:** the run cannot be selected. It is never eligible for a bank, downstream evaluation, or reporting, and it is never resumed or used as a weight source. It is kept only as historical evidence. Seed42 is rerun fresh, from update 1, at the fix authority.

Machine-readable record: `E06B_SEED42_ATTEMPT1_CHECKPOINT_WRITER_FAILURE.json`. Root listing: `E06B_SEED42_ATTEMPT1_EVIDENCE_SHA256.txt`.

## Identity

| field | value |
|---|---|
| method / seed | E06b / 42 |
| run_id | `d3c0ddd5dc82a4a1` (run_uuid `ded46fb5-4a7b-43ba-9dd7-ebcdd98bc03d`) |
| launch authority | `24f7049b894330574b84d11baeb40ed4a7fb3c6b` (clean worktree) |
| config sha256 | `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| window | 2026-10-21T16:49:44Z to 16:51:08Z; tmux `m6_e06b_s42`; shell exit 1 |
| environment | gpat-m6-e06c, Python 3.11.16, torch 2.12.1+cu130, RTX 3090 |

## Failure

```
File "methods/dsdg/native_runner.py", line 443, in run_scientific
    trainer.checkpoint_event(epoch, ctx)
File "methods/dsdg/native_runner.py", line 331, in checkpoint_event
    require(rio.is_checkpoint_epoch(epoch) and ctx.mode == SCIENTIFIC, 'official checkpoint epoch, scientific only')
AttributeError: 'LearnedRunContext' object has no attribute 'mode'
```

`run_scientific` opens a `methods.common.learned_runlog.LearnedRunContext`, and that class has no `mode`. E06c's own `E06cRunContext` does have one. The trainer already holds the execution mode as `self.mode`. No test had called the E06b checkpoint writer with the real context class.

The crash happened on the first entry to `checkpoint_event`. That was epoch 1, under the official epoch-1 cadence exception, and it came after the epoch-1 visualization block.

State at failure: `completion_status` failed; 1 epoch done; global step 16 of 3200. There were **0 checkpoint files**, **0 checkpoint_index entries** and **0 resume sidecars**. The six epoch-1 visualization PNGs had been written.

## Epoch-1 health (all clean)

- 16/16 optimizer steps, split as 15 × 240 + 120, with micro-batches of 12 × 20 and a 6 × 20 tail.
- Each of the 3720 CASIA+MSU TRAIN spoof rows appeared exactly once.
- The independent predictor (`~/workdir/e06b_s42_predict.py`, re-run this session) matched all 16 step pair hashes and the epoch spoof order hash. The pair table it regenerated is byte-identical.
- Native same-subject pairing held for 3720/3720 pairs, and every partner is live. The only datasets were casia_fasd and msu_mfsd, and the only split was TRAIN.
- Classes: print=0, replay=1 (2080 / 1640).
- `loss_cls` was active and finite (7.40–9.75, λ_type-weighted CE). `loss_pair` was active and finite (0.0070–0.0130). Every loss was finite.
- All 55 owned gradients were finite at every step, and the netCls gradient was nonzero.
- There was 1 optimizer application per step. Mode was SCIENTIFIC, VAL fields were null, and there were no TEST keys.

## Firewall (stat-only atime, no image read)

The pre snapshot was taken at 16:49:29Z and the post snapshot at 16:51:34Z, covering 20 627 tracked files.

| split | files | atime-sensitive | changed |
|---|---|---|---|
| CASIA+MSU TEST | 992 | 992 | **0** |
| CASIA+MSU VAL | 1088 | 1088 | **0** |
| SiW (all splits) | 13 575 | 5 580 | **0** |
| CASIA+MSU TRAIN | 4960 | 287 | 279 |
| manifests | 12 | 10 | 0 |

TEST = 0, VAL = 0, SiW = 0. `test_split_accessed` is false.

## Root before archive

The root was `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06b/seed_42`. It held 14 files plus an empty `checkpoints/`.

| digest | value |
|---|---|
| path/size/sha256 listing (`LC_ALL=C` sort) | `8ce8d5a8be3ec6179ef4299a51ab7903383bb963307e809d50c7e2dd9d91f6ca` |
| path/size/mtime listing (`find . -type f -printf "%P %s %T@\n" \| LC_ALL=C sort \| sha256sum`) | `0498f9c924ac58acfad201fae08bb7b858289f7f4784fe7c3d9aa80973484dcc` |

Key file sha256:
- run_manifest `6ebe43f5…44df`
- run_summary `7e2fafad…ce58`
- metrics `24967f3a…949a`
- resolved_config `209e6821…06ff`
- checkpoint_index `4fa90afa…ec03`
- stdout / stderr: empty (`e3b0c442…b855`)

The planned archive path is `<runtime_root>/diagnostics/m6/E06B_SEED42_ATTEMPT1_d3c0ddd5dc82a4a1_CHECKPOINT_WRITER_FAILURE`, created by an atomic same-filesystem rename after the fix commit. The post-move digest check is recorded in the attempt-2 launch evidence.
