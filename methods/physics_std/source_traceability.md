# E04 geometry/depth execution mapping

Reporting name: **Physics-STD (controlled geometry/depth reconstruction)**.
Fidelity: **CONTROLLED_ADAPTATION**. The original PhySTD network is not claimed
to have executed; this milestone implements auxiliary reconstruction/depth
interfaces and static future training plans.

| Contract | Executable binding |
|---|---|
| A3 dense 3DMM + weak-perspective projection | `geometry.GeometryAdapter`: exact AST bodies of pinned `TDDFA.recon_vers`, `_parse_param`, `similar_transform`; `dense_flag=True` |
| Fixed geometry bytes | `assets.validate_assets`: frozen external paths, sizes and SHA256s, no checkpoint deserialization |
| BFM basis and triangulation inputs | Explicit restricted NumPy-array loading of hash-verified BFM/tri only; float32 40/10 basis slices mirror `BFMModel`; source-relative asset paths are not used |
| Q140 | `contract.validate_q140`: committed file bytes and ordered-list hash, no FPS re-derivation; `GeometryAdapter` verifies first 68 against actual BFM keypoints |
| Eq.21 spoof M0 | `renderer.spoof_depth` / `depth_target`: exact float32 zeros, early return without geometry |
| Eq.21 live M0 / A3 | `DepthRenderer`: exact pinned `utils/depth.py::depth` and `Sim3DR.py::rasterize` bodies, official Cython kernel; written z-buffer mask captured without changing its operations |
| A4 output conversion | `convert_live_depth`: uint8 INTER_AREA, float32, divide by 255, clip; identical RGB depth channels collapse to one channel |
| A4 Adam | `PhysicsSTDAdapter.build_optimizer`: TensorFlow Adam with overlay beta1/beta2/epsilon; no weight decay; frozen LR and step-divide schedule |
| Eq.23/Eq.24 loss coefficients | Copied without change into plans from `config.losses`; network/loss graph execution is outside this preparation milestone |
| A2 final-state rule | `checkpoint_policy` / `validate_checkpoint_event`: within-seed iteration 150000, zero extra optimizer steps, no data-selected checkpoint |

The production renderer fails closed if `Sim3DR_Cython` is unavailable. Tests
compile the unchanged pinned C++ `_rasterize` into a temporary library using
a small C linkage shim, then invoke the exact Python depth/rasterize bodies.
This synthetic-only injected kernel is never accepted by the production
`depth_target` path. No alternative NumPy rasterizer is used as a fallback.
M6D3a builds and verifies the actual Cython extension in an external working
copy; the original source cache is unchanged. `runtime.py` resolves only paths,
checks the four frozen asset identities, verifies unchanged build inputs and
the compiled extension SHA256, and retains the `ExtensionFileLoader` gate.

Constant-z meshes, zero-area projected triangles, nonfinite vertices and invalid
triangle indices are rejected explicitly. No epsilon, replacement target, or
silent normalization is introduced. Official edge inclusion and z comparison
semantics remain unchanged.

Base/A3/A4 inputs are immutable; A4's document and overlay are verified against
the final M6A6 commit. The historical M6A6 JSON retained the pre-formatting A4
Markdown hash; the final committed document differs only by trailing spaces.
No historical artifact is rewritten or scientific resolution reinterpreted.

Plans use the common frozen-config/source/asset/seeding helpers and existing
`run_logging_v1` contract. No parallel logger or production run directory is
created. M6D3a qualifies the fixed geometry/depth environment and the pinned
MobileNet v1 regressor on prepared synthetic 120x120 inputs. `regressor.py`
executes the unchanged model and upstream loader/transforms; the frozen
checkpoint's unused `fc_lm.bias` and `fc_lm.weight` remain ignored exactly as
in that loader. All 164 inference-state tensors match the checkpoint exactly.
The benchmark crop/ROI integration and main PhySTD training graph/runner are
not qualified. E04 remains IMPLEMENTED_NOT_EXECUTED and CONTROLLED_ADAPTATION.
