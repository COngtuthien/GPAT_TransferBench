# M5 ArtifactProbeNet — Completion Report

- Date: 2026-09-21
- Status: COMPLETE
- Authoritative training commit: ad27df2389041b0f008668b7936d5efffd148c35
- Branch: m5-gpu-3090
- Host: nd-System-Product-Name
- GPU: NVIDIA GeForce RTX 3090
- Training exit code: 0

## Scientific contract

- Frozen ArtifactProbeNet config SHA256: 3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe
- M3 split manifest SHA256: fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d
- ResNet18 IMAGENET1K_V1 SHA256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
- M5 GPU execution config SHA256: c9d22abf1b5e284593bfed21500e8113df7f14ff564b7434e3eda43cb7e7eb70
- Physical faces root: /home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256
- M2 COMPLETE faces verified before training: 20,615 expected, 20,615 found, 0 missing, 0 extra, 0 SHA mismatches.

## Authoritative result

- Epochs executed: 30
- Validation passes: 30
- Selected epoch: 14
- Best VAL macro-F1: 0.9693201750054498
- VAL samples: 3,121
- Checkpoint SHA256: b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff
- Checkpoint size: 44,802,507 bytes
- Checkpoint is intentionally not committed to Git.

## Selected-epoch VAL F1

- live: 0.971940
- makeup: 0.954248
- mask_2d: 1.000000
- mask_3d: 0.964444
- partial: 0.995062
- print: 0.940784
- replay: 0.958763

## Execution contract

- CUDA authoritative training: yes
- CUDA FP16 autocast: yes
- GradScaler: enabled
- VAL autocast: enabled
- TEST split used: no
- Synthetic samples used: no
- Checkpoint selection: maximum VAL macro-F1 with strict >; exact ties keep the earlier epoch.

## Post-run validation

The authoritative checkpoint and audit outputs were independently checked after training.

- Checkpoint SHA matches artifact_probe_v1.sha256: PASS
- Training code commit matches ad27df2389041b0f008668b7936d5efffd148c35: PASS
- Frozen-config hash: PASS
- Split-manifest hash: PASS
- ResNet18 weight hash: PASS
- GPU execution-config hash: PASS
- Resolved faces root: PASS
- 30 validation passes: PASS
- Epoch sequence 1..30: PASS
- 30 JSONL epoch records: PASS
- Checkpoint selected epoch and best macro-F1 agree with audit record: PASS
- TEST unused: PASS
- Synthetic data unused: PASS
- AMP contract: PASS
- VAL population = 3,121: PASS
- Overall post-run validation: PASS

## Provenance note

The CLI JSON reports `git_dirty=true` because `cmd_train_probe()` queries Git state after
`T.run()` returns. By then the authoritative run has created the checkpoint and runtime audit
artifacts. The repository was verified clean immediately before authoritative training, the
checkpoint records code commit ad27df2389041b0f008668b7936d5efffd148c35, and no tracked
source file changed during training.

## Earlier accidental test run

An earlier test-triggered checkpoint was quarantined and marked NON-AUTHORITATIVE before this
run. It is excluded from scientific results. Its best epoch and best VAL macro-F1 happened to
match this independently launched authoritative run exactly; it remains excluded regardless.

## Milestone disposition

M5 ArtifactProbeNet is COMPLETE. The checkpoint hash is frozen above before any synthetic-bank
evaluation. TEST has not been inspected or used by M5.
