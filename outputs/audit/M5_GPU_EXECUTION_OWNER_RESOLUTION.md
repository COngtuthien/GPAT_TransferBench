# M5 — GPU Execution Owner Resolution

**Purpose.** To record, as an explicit owner approval, two M5 decisions taken on 2026-09-21: the
AMP scope for the ArtifactProbeNet validation pass, and the migration of the authoritative M5
execution host. This record documents **owner approval only**. It is **not** evidence of successful
execution, and nothing in it may be read as GPU verification.

| | |
|---|---|
| **Record date (UTC)** | 2026-09-21T04:15:47Z |
| **Status** | OWNER-APPROVED |
| **Starting scientific commit** | `392884157dc85c376b2e987148b7f8cfecfd693e` |
| **Frozen ArtifactProbeNet config** | `configs/frozen/artifact_probe.yaml` |
| **Frozen config SHA256** | `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe` |
| **Frozen config modified by this record** | **NO** |
| **Frozen snapshot modified by this record** | **NO** |

---

## 1. Execution-host deviation

**Classification: EXECUTION ENVIRONMENT DEVIATION ONLY — infrastructure/execution host migration.**

### Historical target, still recorded in the frozen config

```yaml
execution:
  remote_host: sparc5090
  remote_project_root: /home/sparc/workdir/longnm/GPAT_TransferBench
```

### Owner-approved target for the authoritative M5 run

| | |
|---|---|
| ssh user/host | `student20261@100.121.84.44` |
| project root | `/home/student20261/workdir/GPAT_TransferBench` |
| GPU | **NVIDIA GeForce RTX 3090** |

### The frozen config was NOT modified

`configs/frozen/artifact_probe.yaml` and
`frozen_config_snapshot/configs/frozen/artifact_probe.yaml` are **byte-unchanged** and both still
hash to `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe`. The historical
`sparc5090` values remain in the frozen config as history. This deviation record is the
authoritative statement of where the run will actually happen; the two must be read together.

The trainer's CUDA fail-closed message no longer names a host — it points at the frozen config's
`execution` block and at this pending deviation record, so the code asserts no host that the owner
has not approved.

### Reason

Infrastructure/execution host migration only. The previously planned host is not the machine the
run will use; the approved RTX 3090 box is.

### What this does NOT change

This deviation alters **nothing scientific**. Unchanged and still governed solely by the frozen
config:

- the scientific population (all real M2-COMPLETE TRAIN/VAL rows; 7 classes, `live … replay`)
- preprocessing (whole 256 canonical face, `INTER_AREA` 256→224, Gaussian 9×9 σ1.5
  `BORDER_REFLECT_101`, signed residual, no post-high-pass normalisation, no augmentation)
- the model (ResNet-18 `IMAGENET1K_V1`, `fc → Linear(512, 7)`, fully fine-tuned)
- the loss (class-weighted CE, `clip(N/(K·n_c), 0.5, 3.0)` in float64, TRAIN only)
- the optimizer (AdamW, lr 1e-4, weight decay 1e-4, batch 64, 30 epochs)
- the scheduler (`CosineAnnealingLR`, `T_max=30`, `eta_min=0.0`, once per epoch, no warmup)
- seeds (42) and the determinism contract
- the validation policy (a pass at the end of every epoch 1…30; fixed 7-class macro-F1)
- the checkpoint-selection policy (max VAL macro-F1, strict `>`, exact tie keeps the earlier epoch)
- the TEST prohibition
- the frozen config bytes

## 2. VAL autocast — owner resolution

**Resolved: keep validation under the same CUDA FP16 autocast regime as training.**

| pass | precision regime | GradScaler |
|---|---|---|
| TRAIN forward | `torch.autocast(device_type="cuda", dtype=torch.float16)` | — |
| TRAIN backward / update | — | **applies**: `scale(loss).backward()` → `step(optimizer)` → `update()` |
| VAL forward | `torch.autocast(device_type="cuda", dtype=torch.float16)` | **does not apply** |

Validation is **not** run in FP32. It runs under `torch.no_grad()`, so the GradScaler is never
invoked there — it governs the training backward/update only. The implementation keeps
`gpatbench.probe.train.VAL_AUTOCAST = True`, and the decision travels with the artifacts: it is
recorded as `amp.val_autocast` in the checkpoint payload and in the run provenance, so any later
reader can see which regime produced a reported macro-F1.

Rationale: the frozen config enables AMP for the run without scoping it to the training pass, and
keeping both passes in one precision regime avoids a train/validate mismatch in the selection
metric.

## 3. Storage relocation — execution-only, via `GPAT_M5_EXEC_CONFIG`

GPU verification found a real portability blocker: `gpatbench/probe/data.py` resolved the canonical
face store from a hard-coded laptop execution config, so on the RTX 3090 host it pointed at
`/media/cong/Data/GPAT_TransferBench_runtime/data/processed/faces_256`, which does not exist there.

**Owner-approved fix: an execution-only configuration selector.** No symlink is used.

| | |
|---|---|
| selector | environment variable **`GPAT_M5_EXEC_CONFIG`** |
| set | that YAML is the M5 physical-storage execution config; if it is missing or unparseable the run **fails closed** — there is no fallback |
| unset | the historical default `configs/execution/m2b_laptop_external_storage.yaml`, so laptop behaviour is unchanged |
| approved GPU execution config | **`configs/execution/m5_gpu_3090.yaml`** |
| approved physical faces root | **`/home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256`** |
| runtime root | `/home/student20261/workdir/GPAT_TransferBench_runtime` |
| symlink used | **NO** |
| direct faces-root override | **none** — there is no `GPAT_M5_FACES_ROOT`; relocation must go through a config that containment can check |

The selected config is always passed through `gpatbench.preprocess.m2b.resolve_roots()`, so the
runtime-root containment firewall still applies: a root that escapes by absolute path, by `..` or
through a symlink is refused rather than silently used. The GPU execution config declares only
`faces_256_root`; the M2 frames, geometry and identity caches are **deliberately absent**, because
they were not transferred and M5 does not read them — declaring them would assert storage that does
not exist.

### Fail-closed conditions for an authoritative run

The selected config must exist and parse; `resolve_roots()` containment must pass; `faces_256_root`
must exist and be a directory; and the selected path, its SHA-256 and the resolved faces root must
be recorded. No TEST path appears in any execution config.

### Provenance now carried by every M5 artifact

`m5_execution_config_path`, `m5_execution_config_sha256` and `faces_256_root_resolved` are written
into the run result, the checkpoint payload, the environment lock and the checkpoint SHA record.
For the approved GPU run these are expected to read:

```
GPAT_M5_EXEC_CONFIG        = configs/execution/m5_gpu_3090.yaml
m5_execution_config_path   = configs/execution/m5_gpu_3090.yaml
faces_256_root_resolved    = /home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256
```

`configs/execution/m5_gpu_3090.yaml` sha256 `c9d22abf1b5e284593bfed21500e8113df7f14ff564b7434e3eda43cb7e7eb70`
(as generated on the laptop; re-hash it on the GPU host after applying the patch).

### This changes physical location only

Sample membership, labels, splits, preprocessing, the model, the loss, the optimizer, the scheduler,
the seeds, the validation policy and the checkpoint-selection policy are **unchanged**. An execution
config is `kind: EXECUTION_INFRASTRUCTURE` and carries no scientific field; the frozen scientific
config `configs/frozen/artifact_probe.yaml` is untouched and still hashes to
`3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe`.

The GPU face store was **already independently SHA256-verified against all 20,615 M2 COMPLETE
samples** (`outputs/audit/M5_GPU_FACE_INTEGRITY.json` on the GPU host: 20,615 expected, 20,615
found, 0 missing, 0 extra, 0 SHA256 mismatches, `M5_FACE_INTEGRITY=PASS`). Relocation therefore
moves the same bytes, not a different dataset.

## 4. Execution status — still blocked

Authoritative M5 execution remains **blocked** until all of the following have happened on the
RTX 3090 box:

1. the implementation patch is applied to the GPU branch;
2. `tools/build_artifact_index.py` is regenerated there (that checkout has additional audit files);
   and `GPAT_M5_EXEC_CONFIG=configs/execution/m5_gpu_3090.yaml` is exported so the faces resolve to
   the GPU store;
3. the targeted and full test suites are run and compared against the saved GPU baselines;
4. the CUDA dry-run passes;
5. the CUDA-dependent items are actually measured — `torch.cuda` availability, real FP16 autocast,
   real `GradScaler` operation, deterministic CUDA kernels under
   `torch.use_deterministic_algorithms(True)` (including whether `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   suffices on the pinned build), and multi-worker loading.

**No part of this record is GPU-verified.** It states owner approval, not successful execution.

## 5. Standing constraints, restated

- **TEST remains NEVER.** It is not loaded, not evaluated and not inspected, in any mode.
- **The real 30-epoch training run has not been executed.** No scientific ArtifactProbeNet
  checkpoint exists; `models/artifact_probe/` does not exist and no M5 runtime audit output has been
  produced.
- Synthetic samples remain forbidden in probe training; augmentation remains forbidden.
- There remains exactly one authoritative training code path, `gpatbench.probe.train.run()`.

## 6. Cross-references

`configs/frozen/artifact_probe.yaml` (unchanged) ·
`outputs/audit/M5_ARTIFACT_PROBE_OWNER_DECISIONS.md` ·
`outputs/audit/M5_ARTIFACT_PROBE_FROZEN_CONTRACT.md` ·
`outputs/audit/M5_GPU_EXECUTION_PLAN.md` ·
`outputs/audit/M5_VALIDATION_EPOCH_CONTRACT_CORRECTION.md` ·
`configs/execution/m5_gpu_3090.yaml` (execution-only) ·
`gpatbench/probe/data.py` (`GPAT_M5_EXEC_CONFIG` selector)
