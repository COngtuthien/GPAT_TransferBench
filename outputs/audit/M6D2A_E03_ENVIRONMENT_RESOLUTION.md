# M6D2a — E03 STDN environment compatibility resolution audit

**BLOCKED_BY_MULTIPLE_REASONS. Audit completed; E03 is not qualified.**
E03 remains `FAITHFUL_OFFICIAL`, `IMPLEMENTED_NOT_EXECUTED`, and `ENVIRONMENT_NOT_QUALIFIED`.
The source-tested version is contradictory and unavailable in the queried indexes;
the source-range legacy GPU stack has no established full RTX 3090 route. A later NVIDIA
TF1 build is a credible hardware research lead requiring fidelity review, not an
approved M6D2b candidate. No final environment pin was selected.

## Repository synchronization and provenance

Laptop HEAD and origin/m6-baselines: `372d456dddb635d840e35c0659951f7d3716fc79`.
Laptop branch m6-baselines; initial worktree clean; initial/final divergence 0 0.
GPU pre-HEAD: `7387a25f62b50ad748edd1a69e21616af44501b9`, branch m6-baselines, clean.
Remote target and GPU post-HEAD: `372d456dddb635d840e35c0659951f7d3716fc79`.
Ancestry true; distance exactly 1; post-sync divergence 0 0 and clean worktree.
The ancestry/distance were proved using laptop objects before fetch and repeated on
GPU after fetch and before merge. Only these Git mutation commands ran:

```text
git fetch --no-tags origin refs/heads/m6-baselines:refs/remotes/origin/m6-baselines
git merge --ff-only origin/m6-baselines
```

One preliminary GPU `ls-remote` failed on transient github.com DNS, before mutation;
a repeat verified the target and completed the gated sequence. SSH and public metadata
network calls needed sandbox escalation. No pull/reset/rebase/cherry-pick occurred.

STDN pin `c79f1f8c615d2b8471b3df29da881bb18dd54c90`, tree
`991a4f127990f98a108b8426113396f1c39f558d`, verified against source_pins.json.
GPU cache was absent; deployed only `third_party/source_cache/stdn`: 25 pinned working
files, with matching sizes/SHA256/Git blob IDs and no unexpected files. The transfer
contains 27 Git objects and deliberately omits checkpoint blobs even from its Git store.
Archive SHA256: `0196fe7601b1b1a1af56ba0e762f00ff06ff7aa330e59adccdcae6aeef7d0884`.
Pinned upstream example PNG/NPY files remain provenance cache contents; they were
hashed/transferred as opaque bytes, never decoded or deserialized. No benchmark data
was accessed. The four expected worktree deletions are exactly:

```text
data/pretrain/checkpoint
data/pretrain/ckpt-50.data-00000-of-00001
data/pretrain/ckpt-50.index
data/pretrain/ckpt-50.meta
```

No upstream weights were restored or loaded. The deployed detached shallow Git metadata
preserves the pin/tree/index but intentionally lacks these four blobs; do not use
checkout/fetch/repair to restore them. Laptop cache source bytes were not changed.

## Frozen E03 contract

Config and snapshot are byte-identical to the authoritative commit, SHA256
`08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0`.

| Field | Frozen value |
|---|---|
| method_id / fidelity_class / learned_method | E03 / FAITHFUL_OFFICIAL / true |
| source | yaojieliu/ECCV20-STDN @ c79f1f8c615d2b8471b3df29da881bb18dd54c90 |
| checkpoint rule | OFFICIAL_LATEST_FINAL_CKPT_50 |
| resolution / map_size / batch_size / G:D | 256 / 32 / 2 / 2 |
| max_epoch / steps_per_epoch / val_steps | 50 / 2000 / 500 |
| seeds | 42, 1337, 2026 |
| TEST | Forbidden |

## Exact source evidence

**SOURCE FACT — README.md:10**, verbatim:

> Tested on Python 3.6 & Tensorflow 1.13.0. As it uses the contrib package, the code should work with Tensorflow >1.8.0 and <1.13.0. The code should be easy to transfer to keras package.

The two numerical statements conflict mathematically: 1.13.0 does not satisfy `<1.13.0`.
Neither a typo correction, 1.13.1 substitution, nor an implied Keras migration is authorized.
The frozen config retains its exact framework description.

The pinned source directly imports `tensorflow.contrib.layers` at model/utils.py:17
and binds `tf.contrib.framework.arg_scope` at:18. Session/config/py_func/resize/Adam
are execution dependencies, not optional compatibility suggestions. Complete direct
TF reference inventory follows; exact matching lines, repeats, and source hashes are
retained in JSON. Reading `test.py` here is static source inspection, never TEST execution.

| API | Pinned source locations |
|---|---|
| `tf.AUTO_REUSE` | `model/utils.py:68`, `model/utils.py:82`, `model/utils.py:90`, `model/utils.py:107` (more locations in JSON) |
| `tf.ConfigProto` | `config.py:55`, `model/config.py:55` |
| `tf.GPUOptions` | `config.py:54`, `model/config.py:54` |
| `tf.Session` | `test.py:59`, `train.py:145` |
| `tf.abs` | `model/loss.py:21`, `model/loss.py:24` |
| `tf.app.run` | `test.py:93`, `train.py:210` |
| `tf.cast` | `model/utils.py:37`, `model/warp.py:45`, `model/warp.py:50`, `model/warp.py:56` (more locations in JSON) |
| `tf.ceil` | `model/warp.py:67` |
| `tf.clip_by_value` | `model/utils.py:24`, `model/warp.py:50` |
| `tf.concat` | `model/model.py:23`, `model/model.py:39`, `model/model.py:40`, `model/model.py:54` (more locations in JSON) |
| `tf.cond` | `train.py:73`, `train.py:74`, `train.py:75`, `train.py:79` (more locations in JSON) |
| `tf.constant_initializer` | `model/utils.py:70`, `model/utils.py:78`, `model/utils.py:102`, `model/utils.py:128` (more locations in JSON) |
| `tf.contrib.framework.arg_scope` | `model/utils.py:18` |
| `tf.contrib.layers.batch_norm` | `model/utils.py:85`, `model/utils.py:111`, `model/utils.py:137`, `model/utils.py:163` |
| `tf.contrib.layers.conv2d` | `model/utils.py:99`, `model/utils.py:109`, `model/utils.py:125`, `model/utils.py:135` |
| `tf.contrib.layers.conv2d_transpose` | `model/utils.py:151`, `model/utils.py:161` |
| `tf.contrib.layers.dropout` | `model/utils.py:95`, `model/utils.py:121`, `model/utils.py:147` |
| `tf.contrib.layers.fully_connected` | `model/utils.py:76`, `model/utils.py:83` |
| `tf.control_dependencies` | `model/model.py:114`, `model/model.py:125` |
| `tf.data.Dataset.from_tensor_slices` | `model/dataset.py:64`, `model/dataset.py:88` |
| `tf.ensure_shape` | `model/dataset.py:147`, `model/dataset.py:148`, `model/dataset.py:149`, `model/dataset.py:211` (more locations in JSON) |
| `tf.expand_dims` | `model/warp.py:26`, `model/warp.py:32` |
| `tf.float32` | `model/dataset.py:146`, `model/dataset.py:146`, `model/dataset.py:146`, `model/dataset.py:210` (more locations in JSON) |
| `tf.floor` | `model/warp.py:66` |
| `tf.gather_nd` | `model/warp.py:61` |
| `tf.get_variable` | `model/utils.py:69` |
| `tf.greater` | `train.py:73`, `train.py:74`, `train.py:75`, `train.py:78` |
| `tf.image.random_brightness` | `model/dataset.py:151`, `model/dataset.py:151` |
| `tf.image.resize_images` | `model/model.py:51`, `model/model.py:52`, `model/model.py:53`, `model/utils.py:31` (more locations in JSON) |
| `tf.image.rgb_to_yuv` | `model/model.py:23`, `model/model.py:63`, `model/model.py:82` |
| `tf.initializers.global_variables` | `test.py:69`, `train.py:155` |
| `tf.int32` | `model/warp.py:56` |
| `tf.math.reduce_mean` | `model/loss.py:21`, `model/loss.py:24`, `model/loss.py:30`, `model/loss.py:33` |
| `tf.math.reduce_sum` | `model/loss.py:22`, `model/loss.py:31` |
| `tf.meshgrid` | `model/warp.py:43` |
| `tf.name_scope` | `model/model.py:121` |
| `tf.nn.avg_pool` | `model/model.py:47` |
| `tf.nn.relu` | `model/utils.py:72` |
| `tf.nn.tanh` | `model/model.py:41`, `model/model.py:42`, `model/model.py:43` |
| `tf.no_op` | `model/model.py:126` |
| `tf.py_func` | `model/dataset.py:146`, `model/dataset.py:210`, `model/dataset.py:230` |
| `tf.random.uniform` | `train.py:65`, `train.py:66`, `train.py:67`, `train.py:68` (more locations in JSON) |
| `tf.random_normal_initializer` | `model/utils.py:77`, `model/utils.py:101`, `model/utils.py:127`, `model/utils.py:153` |
| `tf.range` | `model/warp.py:43`, `model/warp.py:43`, `model/warp.py:53` |
| `tf.reduce_mean` | `model/model.py:45`, `model/model.py:46`, `test.py:35`, `test.py:36` (more locations in JSON) |
| `tf.reduce_sum` | `model/loss.py:22`, `model/loss.py:31` |
| `tf.reshape` | `model/loss.py:21`, `model/loss.py:30`, `model/warp.py:22`, `model/warp.py:28` (more locations in JSON) |
| `tf.split` | `model/utils.py:29`, `model/utils.py:32`, `train.py:84`, `train.py:85` (more locations in JSON) |
| `tf.square` | `model/loss.py:30`, `model/loss.py:33` |
| `tf.stack` | `model/dataset.py:151`, `model/dataset.py:215`, `model/warp.py:44`, `model/warp.py:59` (more locations in JSON) |
| `tf.stop_gradient` | `train.py:76`, `train.py:77`, `train.py:108` |
| `tf.string` | `model/dataset.py:230` |
| `tf.tile` | `model/warp.py:27`, `model/warp.py:33` |
| `tf.train.AdamOptimizer` | `model/model.py:115` |
| `tf.train.ExponentialMovingAverage` | `model/model.py:111`, `model/model.py:122` |
| `tf.train.Saver` | `test.py:58`, `train.py:144` |
| `tf.train.exponential_decay` | `model/model.py:104` |
| `tf.train.get_checkpoint_state` | `test.py:61`, `train.py:147` |
| `tf.train.get_or_create_global_step` | `test.py:25`, `train.py:30` |
| `tf.trainable_variables` | `model/model.py:116`, `model/model.py:123` |
| `tf.uint8` | `model/utils.py:37` |
| `tf.variable_scope` | `model/utils.py:68` |

Object-method dependencies include Dataset shuffle/repeat/map/batch/prefetch and
`make_one_shot_iterator().get_next()` (dataset.py:32,64–71,88–91), TensorShape/get_shape
behavior, optimizer compute/apply gradients (model.py:116,118), EMA apply, Saver
save/restore and Session.run (train.py). None was invoked.

NumPy, OpenCV, Pillow and matplotlib.tri are unversioned direct imports. The benchmark
adapter additionally checks SciPy metadata, but no pinned source file directly imports
SciPy. The executable import closure is the unchanged train/model/config/dataset/utils/
loss/warp code. Root config.py is vestigial. Effective model/config.py batch size is 2.
No requirements/lock file in this pinned tree establishes ancillary package versions.

Release source metadata shows ensure_shape absent in TF 1.9/1.10, available in 1.11/1.12;
TF 1.11 math exports lack the source's reduce_mean/reduce_sum names. TF 1.12 exports all
highlighted symbols, including random.uniform and initializers.global_variables.
HTTP404 API-manifest paths are recorded as unavailable evidence, not proof of absent
contrib. The NVIDIA tag independently retains contrib layer definitions. No old
TensorFlow package was installed/imported to establish these static conclusions.

## Live existing environment and host

Environment `/home/student20261/miniconda3/envs/stdn`: Python 3.10.21, TensorFlow 2.21.0,
torch 2.5.1+cu121, torchvision 0.20.1+cu121. TensorFlow import succeeded. `tensorflow.contrib`
import failed with ModuleNotFoundError. Top-level Session, ConfigProto, GPUOptions,
py_func, image.resize_images, train.AdamOptimizer and set_random_seed are absent.
ensure_shape is present. Thus **EXISTING_ENV_NOT_SOURCE_COMPATIBLE** is derived from
the live API probe, not the environment's name or version alone.

No STDN source was imported into this environment. CUDA_VISIBLE_DEVICES=-1 limited
the probe to import/attribute/build-metadata inspection; no Session, tensor, model,
weight or GPU kernel was constructed. TensorFlow build info reports CUDA 12.5.1,
cuDNN major 9; this is compiled metadata, not proof of loaded native-library execution.

| HOST FACT | Live observation |
|---|---|
| GPU / capability / VRAM | NVIDIA GeForce RTX 3090 / 8.6 / 24576 MiB |
| Driver | 595.84 |
| System nvcc | 12.0.140 |
| OS / kernel / architecture | Ubuntu 24.04.4 LTS / 7.0.0-29-generic / x86_64 |
| glibc / Conda | 2.39 / 26.7.2 |

These are host facts, not future runtime pins. System nvcc need not match an isolated
framework's CUDA runtime.

The first guarded TF import blocked an attempted ~/.config/matplotlib mkdir before
mutation. The successful repeat redirected cache paths to /tmp. Both before/after
stdn fingerprints match: 76794 filesystem entries and 318 package-metadata files;
metadata tree SHA256 `085203ad7eccad0ee2421ec1de9d48bc0af0a14524cf7319761914a06170d3c6`;
package metadata content aggregate `b8720baf1aab53ce05a727761b55a58b5d21f9ef3c86e4a555b88d74e2e48bb3`.
The guard is Python-level and fingerprints cover paths/mode/size/mtime/link plus
package metadata bytes, not a full syscall audit or every environment binary's hash.
Temporary Matplotlib cache writes under /tmp are disclosed in JSON.

## Candidate compatibility matrix

| ID | Python / TensorFlow | Package evidence | CUDA / cuDNN | Unchanged STDN / contrib | RTX 3090 and fidelity decision |
|---|---|---|---|---|---|
| A | 3.6 / literal 1.13.0 | No exact release files in queried indexes | Exact build unknown | Author claim; binary unavailable | Blocked by contradictory version evidence, exact-package absence, and unproven GPU route |
| B older | 3.6 / 1.9, 1.10, 1.11 | cp36 Linux wheels exist | 9 / 7 upstream | No: missing required symbols; contrib alone insufficient | Patching forbidden; legacy GPU route unproven |
| B reference | 3.6 / 1.12.0 | cp36 wheels and Conda GPU records | 9 / 7 upstream; Conda 9.0 / >=7.1.0,<=8.0a0 or 9.2 / >=7.2.0,<=8.0a0 | Static API candidate; contrib present; execution untested | Best README-range reference; no supported full Ampere route established |
| C adjacent | 3.6 / 1.13.1 | cp36 wheels and Conda records | 10.0 / 7.4 upstream; Conda variants differ | TF1 API candidate; contrib present | Outside both literal version claims; review required; GPU route unproven |
| C upstream | 3.6–3.7 / 1.15.x | Linux wheels; 1.15.0 Conda records | 10.0 / 7.4 for upstream 1.15.0; Conda >=7.6.4,<8 | TF1 API candidate; contrib retained | Later TF1 is not source-authorized fidelity; old-runtime GPU route unproven |
| C NVIDIA | 3.8 / NVIDIA 1.15.5, image 23.03-tf1-py3 | Vendor documents prebuilt image; no pull/digest verified | 12.1.0 / 8.7.0 | Potentially unchanged STDN; inspected vendor contrib retained | Credible Ampere hardware candidate; needs version/fidelity and numerical-default review first |
| D | 3.10 / TF 2.21 + compat.v1 | Installed live; public wheel metadata | Build 12.5.1 / major 9 | No; contrib absent; source edits needed | SOURCE CODE / FRAMEWORK ADAPTATION; not authorized |


Evidence classes are explicit in JSON: SOURCE FACT, HOST FACT, PACKAGE-METADATA FACT,
and COMPATIBILITY INFERENCE. Exact requests, scripts, responses, public package
filenames/digests, Conda dependency metadata and API exports are retained there.
Queries used Python urllib HTTPS GETs to PyPI JSON, api.anaconda.org and pinned
raw GitHub source. No package manager command or solver ran; no wheel was downloaded.
Metadata scripts/results stayed under laptop /tmp before inclusion in the audit.
Package existence is not a full dependency-solver or execution success claim.

Upstream runtime figures come from the
[TensorFlow build table](https://www.tensorflow.org/install/source); Conda variants
are reported separately. [Ampere PTX compatibility](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)
is conditional, so the driver cannot establish a working old cuDNN stack.
[NVIDIA's 23.03 TF1 release](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-03.html)
provides credible hardware compatibility evidence but no STDN fidelity authorization.
[TF2 migration documentation](https://www.tensorflow.org/guide/migrate/upgrade)
confirms contrib cannot be restored simply by switching to compat.v1.

The committed benchmark wrapper adds a separate integration limitation:
methods/common/learned.py imports future annotations/importlib.metadata and uses
packages_distributions, str.removesuffix and Path.is_relative_to; config.py/upstream.py
also use is_relative_to. These are incompatible with a direct Python 3.6 execution
path; some remain unavailable in Python 3.8. No wrapper changes were made.
A future reviewed controller/worker boundary is only a proposal, never evidence
that the present adapter already runs under either legacy interpreter.

The [compatibility plan](../../environments/e03_stdn_compatibility_plan.md) records
candidate-specific patching/fidelity implications and gates. No candidate meets all
READY conditions. NVIDIA TF 1.15.5 is a conditional research lead after explicit
framework/fidelity review; TF 1.12 is the source-range reference. M6D2a recommends
no executable final pin and does not qualify E03.

## Validation, ledger and artifact firewall

Validator: `python -B tools/m6d2a_e03_environment_preflight.py --live-gpu`.
It validates identities, frozen config, source hashes/deletions, recorded live API
facts, live host/environment fingerprint, complete candidate dimensions, safety
records, the committed ledger prefix, six-path scope and final CRLF index. It never
imports a framework/model or runs scientific tests. Draft mode runs before append;
default mode requires finalized 99 rows and the rebuilt index. Evidence validation
success does not change the BLOCKED decision.

Ledger 98 -> 99: exactly one `M6D2A_E03_ENVIRONMENT_RESOLUTION` row. The original
248107-byte 98-row prefix is byte-identical, SHA256
`47735be1b3abd85c05927a6b2b54685619f04fc143b4d8c94e6ce77f19943102`.
The row binds all four finalized deliverables by SHA256. The canonical artifact
index builder runs LAST among repository writes, preserving CRLF; its final count
and SHA256 are reported by the read-only validator/completion report. This avoids
a hash cycle: index excludes itself and ledger by established policy.

Only the four requested additive files plus ledger/index may differ. Scientific
configs, amendments, methods/tests, previous audits and source pin records remain
byte-identical to HEAD. The GPU tracked worktree is clean. Canonical index hashing
may stream opaque tracked manifest bytes; it never interprets them or opens their
referenced images. No scientific test suite was run in M6D2a.

**NO ENVIRONMENT CREATED; NO PACKAGE INSTALLED; NO ENVIRONMENT MUTATED;
NO SOURCE PATCHED; NO MODEL CONSTRUCTED; NO WEIGHT LOADED;
NO BENCHMARK IMAGE DECODED; NO TEST SCIENTIFIC EXECUTION; NO TRAINING;
NO CHECKPOINT; NO SYNTHETIC BANK; NO COMMIT; NO PUSH.**
No optimizer step, backward, scientific manifest execution or scientific config change.
