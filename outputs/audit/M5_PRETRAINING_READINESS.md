# M5 ArtifactProbeNet — Pre-Training Readiness

- Date: 2026-09-21
- Branch: m5-gpu-3090
- Parent commit: 392884157dc85c376b2e987148b7f8cfecfd693e
- Status: READY_FOR_COMMIT; authoritative training NOT started yet.

## Frozen scientific contract
- artifact_probe.yaml SHA256: 3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe
- split manifest SHA256: fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d
- ResNet18 IMAGENET1K_V1 SHA256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec

## Execution relocation
- GPU execution config: configs/execution/m5_gpu_3090.yaml
- execution config SHA256: c9d22abf1b5e284593bfed21500e8113df7f14ff564b7434e3eda43cb7e7eb70
- faces root: /home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256
- owner-resolution SHA256: aaa28af29c4d6148ddffae94339dada6233c019e9c4c44fc2722da3c096d226c
- implementation full-patch SHA256: d9100043d3c0bb126256a0d33318fbd1b6d6132c926f4a6017b81906d0bf6ef3

## GPU evidence
- M2 COMPLETE face integrity: 20615 expected, 20615 found, 0 missing, 0 extra, 0 SHA mismatches.
- final CLI provenance tests: 8 passed.
- final GPU targeted M5 suite: 157 passed, 3 expected CUDA-host skips, 0 failed.
- test suite checkpoint side effect: none.
- dry-run checkpoint_written: false.
- TRAIN rows/batches: 14467 / 227.
- VAL rows/batches: 3121 / 49.
- forward-only VAL smoke: PASS on NVIDIA GeForce RTX 3090.
- forward input: (64,3,224,224) float32.
- forward logits: (64,7) float16, finite.
- deterministic algorithms: enabled.

## Incident disposition
- An earlier test-triggered checkpoint was quarantined outside the repository.
- It is NON-AUTHORITATIVE and EXCLUDED from scientific results.
- See outputs/audit/M5_ACCIDENTAL_TEST_RUN_INCIDENT.md.

## Training gate
- Authoritative training may start only after this implementation is committed and git status is clean.
- TEST has not been used.
- No authoritative M5 checkpoint exists at this stage.
