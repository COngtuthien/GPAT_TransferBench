# M6D2b — E03 STDN compatibility-first runtime qualification

Decision: **E03_ENVIRONMENT_QUALIFIED_WITH_COMPATIBILITY_RUNTIME**.
Method status remains **IMPLEMENTED_NOT_EXECUTED**. Execution fidelity disclosure:
**FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY**. No claim of bitwise reproduction
of the authors' original environment is made.

## Authority and repository synchronization

The explicit owner policy is recorded in
`docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Runtime_Compatibility_Addendum.md`.
Historical versions are reference provenance, not mandatory execution pins.
Earlier M6D2a evidence and the frozen config/snapshot were not rewritten.

Laptop HEAD and origin: `6aebf148bceb5e4d3692326dbabc6b3e2390fb9d`;
branch `m6-baselines`, initial clean worktree and divergence `0 0`.
GPU HEAD: `372d456dddb635d840e35c0659951f7d3716fc79` → `6aebf148bceb5e4d3692326dbabc6b3e2390fb9d`.
Ancestry proved and fast-forward distance exactly **1**; exact branch fetch and
`git merge --ff-only origin/m6-baselines` succeeded. Final GPU worktree is clean
and divergence is `0 0`. Full command output is in the paired JSON audit.

## Selected runtime and actual versions

Selected the official NVIDIA TensorFlow 1.15 wheel in the new isolated Conda prefix
`/home/student20261/miniconda3/envs/gpat-m6-e03`. Docker's daemon was inaccessible to this user and
`sudo -n docker info` required a password; no host permission change was attempted.
The wheel preserves the required TF1/contrib surface and successfully runs the
unchanged official scientific graph on the RTX 3090.

| Component | Actual evidence |
|---|---|
| Python | 3.8.20 |
| TensorFlow module | 1.15.5 |
| NVIDIA TensorFlow distribution | 1.15.5+nv22.12 |
| CUDA runtime package | 11.8.89 |
| Native CUDA runtime query | 11080 |
| cuDNN package | 8.9.7.29 |
| Native cuDNN query | 8907 |
| GPU / compute capability / driver / memory | NVIDIA GeForce RTX 3090, 8.6, 595.84, 24576 MiB |
| NumPy | 1.23.5 |
| OS/kernel/glibc | Ubuntu 24.04.4 / 7.0.0-29-generic / glibc 2.39 |

System nvcc is host provenance only; its output is preserved in runtime JSON.
NVIDIA's broad TensorRT dependency also installs separate CUDA 12/cuDNN 9 libraries.
The library mappings identify the actual CUDA 11/cuDNN 8 STDN execution path.
TensorRT, DALI, AMP and XLA are not used for STDN execution.
`pip check` passes. Every installed distribution is recorded; Conda explicit URLs
include package MD5, and pip artifact URLs/SHA256 plus the complete pip freeze are locked.
Install logs and exact commands are preserved in the paired JSON.
Only `gpat-m6-e03` was created/installed into. The complete lstat inventories of
`stdn` and `gpat-m5` match their pre-install fingerprints. No E01/E02 environment
or base/system Python package installation was performed.

The route was selected using NVIDIA's [TF1 compatibility documentation](https://github.com/NVIDIA/tensorflow)
and [22.12 release evidence](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-12.html),
then established by live probes. Actual wheel versions are measured facts, not an
assumption that the wheel exactly matches the published container. The upstream
NVIDIA repository is now archived; this is a reproducibly identified legacy runtime.

## Source and compatibility boundary

Pinned STDN: `yaojieliu/ECCV20-STDN` at
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`.
All 25 source-only cache files retain their recorded hashes on laptop and GPU.
The four original `data/pretrain/checkpoint`, `ckpt-50.data-00000-of-00001`,
`ckpt-50.index`, and `ckpt-50.meta` paths remain deleted. No weight was restored or loaded.
No source cache edit, generated patched scientific source, TF API alias, or contrib
replacement was needed. `tf.Session`, `tf.ConfigProto`, `tf.py_func`,
`tf.image.resize_images`, `tf.train.AdamOptimizer`, and original contrib functions remain native.

The machine-readable patch manifest reports empty source/API patch lists.
A new standalone Python 3.8 worker avoids importing the newer benchmark controller
package. Its local path containment helper uses `relative_to`/`ValueError` and is
tested for normal paths, siblings and symlink escape. Diagnostic observers forward
identical arguments/results to original Gen/Disc_s/get_train_op and are restored
after construction. Config class constants are accessed without calling its
filesystem-writing constructor; Dataset/main are never invoked. These are
qualification-controller arrangements, not scientific source substitutions.

## Progressive qualification

Stages ran in order in separate processes, using seed **42**. Stage F is a second
clean Stage E process with the same fixed seed and inputs. Original stages A–F retain their archived worker SHA256 and source text in the JSON.
The D-only extension uses the current worker, and preflight compares the unchanged
graph/forward construction path with that archived version. Supplemental
graph-only diagnostics add one output-observation statement, separately hashed and
validated by preflight.

| Stage | Result |
|---|---|
| A imports/APIs | Native TF1 graph mode and all required APIs/contrib available |
| B static source import | train/model/warp imported; zero graph ops created; no Dataset/Config constructors |
| C original graph construction | 1295827 parameters; 170 exact trainable variable names/shapes; generator 98 + discriminator 72 |
| D synthetic forward | Finite losses [108.60376739501953, 2.1818437576293945, 10.12548828125]; GPU trace contains 1717 executed nodes |
| E one synthetic optimizer application | Finite 170 G/D gradients; norm 284.5997619628906; G Adam completed; shared global step 0→1 |
| F clean-process repeat | Identical variable structure and initial weights; exact matching numerical diagnostics; supplemental graph topology comparison passed |

Synthetic input shape is `[2,2,256,256,3]`: batch 2 live/spoof pairs reshape to four
faces as the original source requires. Landmarks are 68 generated ellipse points;
offset maps use the original interpolation routine. Discriminator scales remain
256/160/40. Generator outputs retain ESR 32×32, global s/b, coarse trace 64×64 and
full trace 256×256. The map_size 32 contract is the ESR output, not the coarse trace.

The graph validates 170 Adam apply nodes, 170 weight-EMA variables, two loss-EMA
variables, 60 batch-normalization moving variables, empty regularization-loss
collection, and FP32 parameters/normalization. Batch normalization retains its
source epsilon 1e-5 (native fused-kernel minimum 1.001e-5, also present in TF 1.13.1) and inline update behavior (`updates_collections=None`).
Forward and gradient diagnostic fetches retain original BN/loss-EMA side effects.
No source update dependency or ordering is changed.

Exactly **one G optimizer application per independent trial** executes the official
G-only schedule branch. D optimizer construction and gradients are validated;
**D application was not executed at that initial stage**; the continuation below closes that gap. The G:D ratio 2 and full schedule are checked
statically against original source, not simulated by modifying the optimizer count.
There were two independent G one-step trials because Stage F repeats Stage E.
This continuation adds exactly two D-only trials, one application in each clean process.

Adam kernel inputs validate beta1=0.8999999761581421,
beta2=0.9990000128746033, epsilon=9.99999993922529e-09,
and initial learning rate=5.999999848427251e-05.
Official source/config hashes and focused tests preserve exponential decay 0.9 every
20000 shared steps with staircase, weight EMA 0.9999, loss EMA 0.9, original objectives,
50 epochs × 2000 steps, val_steps 500, flips and canonical 68-landmark permutation,
every-epoch checkpoint policy, final ckpt-50, seeds 42/1337/2026 and TEST prohibition.
No hyperparameter or frozen config value changed.

Graph structure SHA256: `27fd92cc7e36bd195e53d4744e0df29a64b0fa576949c5de0f05f47a077480fe`.
E-trial raw graph topology SHA256: `10ad4395db39676142bd94d77b9b89806fcce50e3d34a64902d675edd1f42543`.
The E/F raw hashes differ; full E/F topology lists were not retained. Two
supplemental graph-only captures demonstrate raw-hash variability from ordering of eight
control-prerequisite lists; their prerequisite sets and ordered data inputs match.
This comparison establishes a non-semantic source of hash variability, rather than
claiming direct normalization of the unavailable E/F topology lists. The E/F full
variable/observed graph structures and numerical diagnostics independently match.
Canonical supplemental topology SHA256:
`383fdccde898c509a9e47cd52741a35127051ae386bba2ae4d4b85b8d77dd2c3`.
Full compressed graph captures and differences are embedded in the JSON for independent
verification. Canonicalization affects comparison only, never the TensorFlow graph.
No optimizer applications were performed by those topology-only diagnostics.
Repeat diagnostics and exact differences: see JSON `qualification.repeatability`.
This smoke test is not proof of long-run or historical bitwise reproducibility.

## D optimizer execution coverage — continuation of M6D2b

The owner's follow-up explicitly authorizes one D-only synthetic application and
one repeat from a clean process. This updates M6D2b only. The original owner
addendum remains byte-identical; its original G-only scope is extended by this
recorded follow-up authority, not silently rewritten.

**G OPTIMIZER PATH VERIFIED. D OPTIMIZER PATH VERIFIED.**

| Check | Observed result in both D trials |
|---|---|
| Seed / controls | 42; same FP32/TF32/AMP/XLA/session settings |
| Synthetic inputs | `[2,2,256,256,3]` paired batch; 68 generated landmarks |
| D loss before application | 2.1818437576293945 — finite |
| D gradients | 72 checked, all finite; norm 17.33002652476326 |
| D applications | Exactly one per process; zero G applications |
| D trainable variables changed | 72 of 72 |
| G trainable variables unchanged | All 98, byte-identical before/after |
| G nontrainable BN state | 32 moving-stat variables changed through native forward dependencies |
| All global variables after application | Finite; no NaN/Inf |
| Shared global step | 0 → 1 |
| Graph before/after application | Serialized GraphDef SHA256 unchanged |
| Qualified graph structure/topology | Matches the existing G-qualified graph |
| Frozen Adam inputs | All 72 D kernels: LR6e-5, beta1=0.9, beta2=0.999, epsilon1e-8 (FP32 representations) |
| Clean-process repeat | Identical variables, canonical topology, diagnostics and final trainable-weight bytes |

The original `get_train_op` in `model/model.py` passes the shared global step to
`opt.apply_gradients` for both scopes. The observed single D increment is therefore
source-consistent. No G:D cycle, schedule alteration or hyperparameter change was used.
The original D operation executes directly, without feeding cached gradients or
bypassing its normalization/EMA dependencies.

**Generator invariance scope:** all G *trainable weights* remain unchanged. The
unqualified statement that *all* generator variables remain unchanged is false:
32 nontrainable G BN moving-stat variables update because the original training-mode
forward uses `updates_collections=None`. This is observed source behavior, not a G
optimizer application. Suppressing those updates would change scientific semantics;
no suppression or source patch was made. Their exact names are recorded in JSON.

Both D processes have identical loss and gradient norm (absolute differences zero),
within the existing rtol1e-5 / atol1e-6 criterion. The final D weights and all
trainable weights are byte-identical across processes. D-updated weight SHA256:
`8ec9f4449a9d53f645374845b3366976f6cdb0423745c38327beeb292f2e3a50`.
The canonical topology matches the existing supplemental graph SHA256
`383fdccde898c509a9e47cd52741a35127051ae386bba2ae4d4b85b8d77dd2c3`.

Before/after E03 environment lstat inventories match: 28467 entries,
SHA256 `30764b4faf6fcfd3c51f621552e61f7135eff22e17aa995ae5269ea91fcf338e`.
Packages, source cache hashes/deletions and GPU checkout also match. No package was
installed and no runtime environment was mutated in this continuation.

The environment lock is recomputed because it hashes the changed worker, tests and
preflight and now binds the D evidence. Its package/runtime selections did not change.
Conda explicit, pip freeze, runtime JSON, source/API patch manifest and owner addendum
hashes remain identical to the previous lock; JSON records that comparison.
The single existing M6D2b ledger record is updated in place; the committed 99-row
prefix is byte-identical and the final ledger remains 100 rows.

## Precision and execution firewall

FP32 tensors/parameters are checked. `NVIDIA_TF32_OVERRIDE=0`, TensorFlow AMP
switches 0, cuBLAS/cuDNN FP32 tensor-op switches 0, XLA auto-jit 0 and session JIT OFF
are applied before import. Grappler meta-optimizer is disabled and
`auto_mixed_precision=OFF`. Deterministic operation/cuDNN flags, fixed hash/random/
NumPy/TF seeds, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and one inter/intra-op thread
are recorded. Full environment variables and serialized ConfigProto are in the lock.
These controls and dtype evidence support FP32 intent; no hardware-instruction
trace or historical bitwise equivalence is claimed.

The worker's Python open-audit hook refuses benchmark manifests/data/runtime paths,
source data/weights, image and array files; writes are limited to temporary caches.
All scientific inputs are generated in memory. No loader, saver or training main
is constructed. All successful runs report zero denied accesses, zero benchmark
image decodes and no TEST access. This hook is not a kernel sandbox; source review
and the restricted execution path supply the remaining evidence.

## Validation, locks and audit integrity

E03-focused tests: **37 passed**, zero failures/errors/skips.
The paired JSON contains exact commands/output and any preliminary harness issues.
`tools/m6d2b_e03_runtime_preflight.py` validates source/config authority, GPU sync,
all stages/repeat diagnostics, lock hashes, protected environments, scope firewall,
ledger prefix and final CRLF artifact index. It now explicitly requires both G and D
optimizer execution PASS, and rejects missing D coverage or changed G trainable weights. `--live-gpu` adds metadata-only remote
identity/source/package/host checks without importing TensorFlow or executing data.

Environment lock SHA256:
`b2c6c2eadca1b93a0bdf6798990821cb889e199643a5bdb5368ed0b12fdc8c62`.
Owner addendum SHA256:
`f5133f81ed1a0567a7bb1426fc23e05d43e9d689d6478b0f37ec2377832d9d0e`.

Ledger: **99→100**, exactly one `M6D2B_E03_RUNTIME_QUALIFICATION` record.
The committed 251354-byte prefix remains byte-identical, SHA256
`bd59b8d596a9977fd7cc25bc2dd1e5bdd215e3737168e2d52a2116e96c014df9`.
The appended record hashes all new audit/runtime/code artifacts. The artifact index
is rebuilt after all artifacts and ledger are final, retaining CRLF; its final count
and hash are reported by the read-only preflight and final session report to avoid
self-referential hashes.

HISTORICAL PACKAGE VERSIONS WERE NOT TREATED AS MANDATORY.
SCIENTIFIC SEMANTICS WERE PRESERVED within this audited runtime qualification.
ORIGINAL PINNED SOURCE CACHE WAS NOT MODIFIED.
NO BENCHMARK IMAGE WAS DECODED. NO TEST SCIENTIFIC DATA WAS ACCESSED.
NO SCIENTIFIC TRAINING RUN WAS STARTED. NO SCIENTIFIC CHECKPOINT WAS CREATED.
NO SCIENTIFIC SYNTHETIC BANK. NO COMMIT. NO PUSH.

The remaining limitation is production benchmark-controller execution: this milestone
qualifies the unchanged official graph in the isolated runtime, not a real TRAIN/VAL
launch or end-to-end benchmark pipeline. E03 remains IMPLEMENTED_NOT_EXECUTED.
