# M6C2b1 — E04 controlled geometry/depth reconstruction

Starting HEAD: `2bf69b35858d0ae416547a4e320ff9ec9de93bc6`; branch `m6-baselines`;
clean starting tree. Status: **IMPLEMENTED_NOT_EXECUTED**. Static and synthetic
verification only; **NOT BENCHMARK-EXECUTED**.

Reporting name: **Physics-STD (controlled geometry/depth reconstruction)**.
Fidelity: **CONTROLLED_ADAPTATION**, never faithful/native exact reproduction.
Implemented scope is the fixed geometry/depth adapter and static future training
plans. No PhySTD training graph/model execution is claimed by this milestone.

## Authority and source

The base M6B E04 YAML remains byte-identical to its frozen snapshot and recorded
SHA256 `418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a`.
A3 and the A4 overlay are immutable execution inputs. Overlay SHA256:
`7579b7177e8b48513bb7a9057e6f700cf9b1f576b5c577887c07bbda9cf35355`.
The overlay and normative documents are checked against final committed M6A6
Git bytes, and the overlay's base-config hash is checked against the loader.

Historical metadata disclosure: M6A6's JSON retains a draft A4 Markdown hash
`991e85f5…`; the final committed document hash is
`98070d31417fda555d24ac80b7a4a2008e786550482757863be328a251a9e733`.
The difference is two removed trailing-space Markdown line breaks. The final
committed text governs. No scientific decision is reopened or historical file
rewritten.

Source: `cleardusk/3DDFA_V2@1b6c67601abffc1e9f248b291708aef0e43b55ae`.
Repository, commit, tree, required files, pinned Git blobs and cited hashes pass.
Common `verify_source` gains an opt-in E04 auxiliary-geometry branch using
`supports_method_ids`; existing method-source behavior is preserved.

The four frozen external assets all pass existence, byte-size and SHA256 checks
at the recorded runtime root. No download/copy/weight deserialization occurs in
preparation. Geometry tests use a restricted NumPy-only unpickler for the exact
hash-verified BFM and triangulation. The regressor checkpoint is never loaded.

## Geometry, Q140 and depth

`GeometryAdapter` binds the unchanged verified AST bodies of `TDDFA.recon_vers`,
`_parse_param` and `similar_transform`, avoiding framework imports when consuming
already fitted parameters. Shape/expression dimensions remain 40/10, with 38,365
vertices and 76,073 triangles. Synthetic parameters validate dense/Q140 shapes,
projection and repeat determinism. This is numeric reconstruction, not inference.

The frozen Q140 list is consumed, never re-derived: 140 unique indices, first 68
checked against actual model-native iBUG keypoints in order, followed by frozen
72 FPS indices. Ordered-list SHA256:
`1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f`.
Source, assets and list are identical across seeds 42, 1337 and 2026.

Live pipeline, exactly A3+A4:

1. Dense reconstruction and official projection.
2. Per-sample min-max vertex-z normalization in exact pinned `utils/depth.py`.
3. Official Sim3DR rasterization with larger-z-wins and its uint8 buffer.
4. Preserve uint8 at 256x256; resize with INTER_AREA while still uint8.
5. Cast float32, divide by 255.0, clip [0,1], return exactly (32,32).

The exact z-buffer written mask is retained separately at canonical resolution.
Official depth RGB channels are identical and collapse to one channel without
color conversion. Spoof targets return exact float32 zeros before any geometry
call. Constant-z meshes, zero-area projected triangles, nonfinite inputs and
invalid indices fail closed; no epsilon or substitute target is introduced.

The production path requires `Sim3DR_Cython` and never substitutes a test kernel.
The exact pinned Python depth/rasterize functions are bound unchanged. On this
laptop, tests execute the unchanged official C++ `_rasterize` through a small C
linkage shim compiled in /tmp. The shim changes no math. Its injection is marked
synthetic-only and rejected by the scientific live-target path. The missing
Cython-extension smoke is explicitly skipped; C++ testing is not presented as
that extension's execution.

Synthetic tests verify projection coverage, larger z winning, overlap triangle
order invariance, byte-identical repeats, zero background, interpolated uint8
quantization, resize-before-float ordering, output shape/dtype/range and guards.

## Plans, optimizer and final state

Three plans include base/overlay/A3/A4 hashes, source commit, asset hashes, Q140,
fixed-auxiliary reuse, losses, depth contract, seed settings, logging identity,
environment limitations and TEST/VAL-selection firewall.

Adam is `tf.train.AdamOptimizer(lr)` with beta1=0.9, beta2=0.999, epsilon=1e-8,
weight_decay=0. Provenance is
`BENCHMARK_DEFINED_NEAREST_OFFICIAL_PREDECESSOR_BEHAVIOR`, not a PhySTD paper fact.
The future optimizer hook maps lr=5e-5, divide by 10 every 45,000 iterations,
batch=8 and budget=150,000 from frozen inputs. All published loss coefficients
remain unchanged, including alpha1=100 and K=32. The hook's dispatch test is
mocked and is not reported as TensorFlow execution.

`BASELINE_FINAL_STATE_V1` selects iteration 150000 within each seed. If cadence
misses it, the future runner saves terminal state with zero extra optimizer
steps. Early selection, extra steps, VAL-best and TEST selection are rejected.
No checkpoint bytes or production run directories are created. The existing
common logging contract is reused; no second logger is added.

## Validation and limitations

- E04: 50 tests, 49 passed, 1 explicit missing-Sim3DR_Cython skip; no failures/errors.
- E04 preflight: PASS, three plans; 435 audited opens, zero benchmark-image opens,
  zero manifest opens, zero rejected attempts.
- Focused E04 tests: 926 audited opens; zero benchmark-image/manifest opens and
  zero rejected attempts.
- M6B validator: PASS, 0 failures.
- M6A6 regression: 4/4 PASS.
- M6C2a regression: 72/72 PASS.
- M6C1 regression: 83/83 PASS.
- Historical M0: 16/17 PASS; only the unchanged
  `test_manifests_are_stage_appropriate` failure on
  `manifests/artifact_probe_classes_v1.json`.

Observed environment only: Python 3.12.3, NumPy 2.5.3, OpenCV headless 5.0.0.93;
C++ compiler available. TensorFlow, PyTorch, torchvision and Cython unavailable.
No packages installed. These versions are not final benchmark pins. Future
execution needs the official extension build, compatible framework environments,
fixed regressor integration and main PhySTD graph/runner integration. No fake
model execution is claimed.

Open counts refer specifically to the audited Python processes. Git/compiler
internal reads are outside Python's audit hook and operate only on local source,
Git objects and temporary compiler inputs. The index/M0 may hash tracked manifest
bytes; no benchmark pixels or TEST records are executed. The full suite was not
run, and prior historical data-access disclosures remain untouched.

New source/test/tool/status hashes are in the JSON audit; audit-file hashes are
in the single appended ledger record and canonical index. The committed ledger
prefix is preserved. The index excludes itself and the ledger by convention.

NO TRAINING. NO BENCHMARK IMAGE EXECUTION. NO SYNTHETIC BANK GENERATION.
NO TEST DATA ACCESS FOR METHOD EXECUTION. NO GPU JOB. NO SCIENTIFIC CHECKPOINT.
NO FROZEN CONFIG CHANGE. NO E05/E07c IMPLEMENTATION. NO COMMIT. NO PUSH.
