# M6A6 E04 execution-gap resolution

Status: OWNER-APPROVED ADDITIVE CONTRACT · E04 remains NOT IMPLEMENTED

M6C2b1 correctly stopped before implementation because two execution-affecting
details were absent from the frozen E04 YAML: the conversion boundary between
the official uint8 depth renderer and INTER_AREA resizing, and the optimizer
algorithm. A4 resolves only those gaps.

The live-depth pipeline is now: dense 3DDFA reconstruction and projection;
pinned `utils/depth.py` vertex-z normalization; pinned Sim3DR rasterization;
preserve the uint8 256×256 renderer output; uint8 `cv2.INTER_AREA` resize to
32×32; cast to float32; divide by 255; clip to [0,1]; output (32,32).
Spoof M0 remains exact float32 zeros. Float conversion before resizing is
explicitly excluded because it is numerically non-equivalent.

The E04 optimizer is Adam with TensorFlow `tf.train.AdamOptimizer(lr)`
semantics: beta1 0.9, beta2 0.999, epsilon 1e-8, no weight decay. The
learning rate, /10 schedule, batch, initialization and 150,000-iteration
budget remain as frozen. This is benchmark-defined nearest official
predecessor behavior, not a PhySTD paper or author claim. The evidence is the
pinned STDN `model/model.py` at commit
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`, line 115.

The original E04 YAML and frozen snapshot were not edited. No benchmark data
was opened; no model, training, GPU job, checkpoint or bank was created.
The additive machine-readable input is
`configs/amendments/e04_a4_execution_resolution.yaml`.
