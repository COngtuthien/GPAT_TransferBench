# E03 owner-approved runtime compatibility addendum — M6D2b

Authority: the owner's explicit M6D2b instruction in this session (2026-09-23).
This additive policy applies to E03 now. It supersedes M6D2a's requirement to
resolve exact historical runtime versions before compatibility engineering;
it does not retroactively rewrite the earlier evidence or frozen config.

Exact historical Python, TensorFlow, CUDA and cuDNN package versions are reference
provenance, not mandatory execution pins when unavailable or impractical on current
hardware. Scientific method semantics take precedence. Newer compatible runtimes,
including NVIDIA TensorFlow 1.x, and minimal runtime/API compatibility changes are
allowed when they do not alter the scientific method. Every compatibility change
must be disclosed and tested. Prefer unchanged upstream scientific source, then
minimal compatibility shims, then non-scientific wrapper/controller adaptations.

The source remains yaojieliu/ECCV20-STDN at
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`. The original source-only cache must remain
byte-verifiable with its four upstream checkpoint files excluded. Compatibility
changes belong in the adapter or a separately identified generated working copy.

Architecture/topology, discriminator scales [256,160,40], G:D ratio 2, resolution
256, map size 32, batch size 2, max_epoch 50, steps_per_epoch 2000, val_steps 500,
learning rate 6e-5, Adam beta1 0.9/beta2 0.999/epsilon 1e-8, staircase exponential
LR decay 0.9 every 20000 shared global steps, weight EMA 0.9999, loss EMA 0.9,
official losses, horizontal flip and canonical 68-landmark permutation remain
frozen. Checkpoints remain every epoch, final `ckpt-50` under
`OFFICIAL_LATEST_FINAL_CKPT_50`; seeds remain [42,1337,2026]; TEST remains forbidden.
No hyperparameter retuning or evaluation-protocol change is authorized.

Runtime modernization alone does not alter scientific fidelity. Following successful
semantic validation, the execution disclosure is
`FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`. This does not claim bitwise
reproduction of the authors' original environment. Frozen config history and its
original `FAITHFUL_OFFICIAL` field remain unchanged.

M6D2b authorizes an isolated E03 runtime and staged synthetic-only qualification:
imports, source import, graph construction, forward, one synthetic optimizer step,
and a clean-process repeat. Qualification exercises the source G-only schedule branch
(one Adam application per process); D optimizer construction and gradients are checked
without an additional D update. The G:D schedule itself remains frozen and is checked
against the original source. No benchmark schedule is executed. No benchmark image, TRAIN/VAL/TEST execution,
scientific sample generation, scientific checkpoint, commit or push is authorized.

Contrib APIs should remain unchanged when available. A substitute requires explicit
comparison of variables/shapes, update and normalization behavior, regularization,
and graph collections. A material layer/algorithm/optimizer/loss change requires
STOP_AND_REPORT. Preserve FP32 semantics; audit TF32, AMP, XLA and precision rewrites.
