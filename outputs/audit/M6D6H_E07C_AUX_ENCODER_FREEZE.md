# M6D6h — E07c auxiliary conditioning-encoder OWNER FREEZE

**Result: PASS.** Record kind `OWNER_DECISION / ASSET_FREEZE` (ADDITIVE, NON_DESTRUCTIVE, OWNER_FROZEN);
classification `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`; fidelity `CONTROLLED_ADAPTATION`; deviation `DEV-021`;
no new deviation, no new fidelity class, no Amendment A9. Method status stays `IMPLEMENTED_NOT_EXECUTED`.
Authority: `ee0a9b9166577cac606c142af5bc2e338d75eea1` (M6D6g).

## Historical M6D6g vs M6D6h

| | M6D6g (historical, unchanged) | M6D6h (this record) |
|---|---|---|
| SHA256 | `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c` | **same** `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c` |
| Bytes | 185136819 | **same** 185136819 |
| Status | `SHA256_RECORDED_PENDING_OWNER_FREEZE` | `OWNER_FROZEN` |
| owner_freeze_performed | false | **true** |
| authoritative_for_main_difffas | false | **true** |

M6D6g evidence is byte-identical to `ee0a9b9166577cac606c142af5bc2e338d75eea1`; its pending status is historically correct and is
prospectively superseded here, not erased.

## Frozen asset

- Path contract: `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl`
- Observed GPU path (provenance only): `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl`
- Format: torch.save of the WHOLE nn.Module (matches torch.load(path).cuda())
- Producer: M6D6g, auxiliary seed 42, ONE_LOGICAL_RUN, run_id `7014cdb366da52e4`, run_uuid `7031fd2f-92d7-465a-94d1-ab99bf8a3960`,
  epoch 200, global step 11200.
- Main seeds bound to this one checkpoint: 42, 1337, 2026.
- Observed file mode 664 (disclosed; no chmod).

## Freeze record

- `configs/amendments/e07c_m6d6h_aux_encoder_freeze.yaml` SHA256 `6230e0b9531d47664f9eb14a7f87b24f67b0bf231a995adaba1294f944ec7693` (pinned as `FREEZE_RECORD_SHA256` in `methods/difffas/aux_checkpoint.py`)
- `docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Aux_Encoder_Freeze_Record_M6D6h.md` SHA256 `dffc0fbfb0da455c8b8bcb89980cbee71d50e33754bacc6b3a2e6b22e02fbd70`
- Bound: base E07c config, A1 adaptation, A3, A6 (doc + overlay), A7 (doc + overlay), A8 (doc + overlay),
  E07c environment lock, frozen specification, M6D6g final-checkpoint and scientific-run evidence; source pin `23f40519ec25a833ebc06842aa6fbab74fad4d15`.

## Secure loader binding

`methods.difffas.aux_checkpoint.load_frozen_aux_encoder(runtime_root, recorded_sha256, config=None)` (signature unchanged):
verify pinned freeze record -> caller SHA256 must equal the frozen SHA256 (else STOP_AND_REPORT **before the
checkpoint file is opened**) -> configured seed-42 path under an absolute, repository-external runtime root ->
frozen size 185136819 required -> existing external-asset SHA256 verification -> unchanged M6D6c exact-byte order
(`read_verified` -> SHA256 of in-memory bytes -> pinned source -> `import torch` ->
`torch.load(io.BytesIO(raw), weights_only=False)` -> identity -> device). Exactly one unsafe-load seam;
no state_dict, safetensors, TorchScript, safe_globals, fallback/latest/CLI/env override, or global patch.
`execution_policy.py` and `encoder.py` are unchanged.

## B1 — M6D6g test scope correction

`tests/test_m6d6g_e07c_aux_scientific_run.py`: `PROSPECTIVE_REGRESSION_HARNESS_SCOPE_CORRECTION`; scientific_history_changed = false.
Old SHA256 `a292d3bd8ad1185f0da8697f7c876372985aae4168521278fb2850d7b77d5bcf` -> new SHA256 `f84b36d914ac918d912472c50b5d781a37a96de754ab13982af34619fa7905ca`.
Only `test_17` changed: it now verifies the historical range `7642b23..ee0a9b9` (no scientific-tree diff; executed
files and spec identical at both ends) instead of locking the current HEAD/worktree forever. All other M6D6g
tests (including the per-file executed-code identity test_02) are unchanged; the M6D6g preflight, evidence and
ledger row are unchanged.

## Read-only GPU revalidation

`sha256sum` = `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c`; size 185136819; stat identical before/after hashing; all 9 small runtime files
equal the M6D6g recorded SHA256; run_summary `completion_status=completed`, `SHA256_RECORDED_PENDING_OWNER_FREEZE`; no E07c auxiliary or
main DiffFAS process. No torch.load, no chmod, no file mutation; GPU repo (at `7642b23`) not synced or modified.
Log: `outputs/audit/M6D6H_E07C_AUX_ENCODER_FREEZE_RUNTIME_LOG.txt`.

## Zero science

Training runs 0; scientific auxiliary runs 0; main DiffFAS runs 0; optimizer steps 0; backward calls 0;
TRAIN/VAL/TEST image reads 0; checkpoint deserializations 0 (the real checkpoint was never deserialized;
no synthetic 185-MB file); M8 outputs 0.

## Tests

| Scope | Ran | Skipped | Failures | Errors |
|---|---|---|---|---|
| M6D6h targeted (tests.test_m6d6h_e07c_aux_encoder_freeze) | 33 | 0 | 0 | 0 |
| E07c regression: m6d6h+m6d6g+m6d6f+m6d6e+m6d6d+m6d6c+m6d6b+m6d6a+m6c2b3 (3)+m6a8 (12 modules) | 272 | 12 | 0 | 0 |

Skips: torch-live tests (laptop has no torch).

## Statuses

**Qualified:**
- `E07c_AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`
- `AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`

**Still not qualified:**
- `MAIN_DIFFFAS_TRAINING_GRAPH`
- `MAIN_CHECKPOINT_RESUME`
- `MAIN_RUNNER_ENCODER_LOAD_INTEGRATION`
- `MAIN_PRODUCTION_RUNNER`
- `MAIN_DIFFFAS_SCIENTIFIC_TRAINING`
- `M8_BANK`

Next recommended milestone: M6D6i — main DiffFAS runner encoder-load integration through
`load_frozen_aux_encoder` (qualification only; no main scientific run).
