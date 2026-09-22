# M6A7 — E05 blur operator resolution

Status: **OWNER-APPROVED · ADDITIVE · NON-DESTRUCTIVE**. E05 is **CONTRACT_RESOLVED_NOT_IMPLEMENTED**.

Starting HEAD: `05111690087b8e4b723148ed96886478ba765333` on `m6-baselines`; clean preflight.

M6C2b2 stopped because the frozen 256→128 blur geometry did not identify the execution operator. No PCGAN implementation was started.

The [paper §3.1.2, Eq. 3](https://arxiv.org/html/2604.09018v1#S3.SS1.SSS2) describes target/mixed-image blur via 1024→512 downsampling but omits interpolation family, kernel, antialias flag, padding, coordinate convention and boundary mode. Eq. 3 typesets an L2 norm; this amendment resolves only blur and does not change the reconstruction distance or introduce a squared-distance rule.

A2-07 resolves inherited architecture only. Pinned Swapping Autoencoder commit `6baa180f1184ee79a6b967f9d80ee0e02a979ac7` does not implement PCGAN L_recblur. Source files were byte-verified against that commit:

- `third_party/source_cache/swapping_autoencoder/util/util.py`: resize2d_tensor (bilinear utility; visualization use). Not a PCGAN blur-loss binding.
- `third_party/source_cache/swapping_autoencoder/models/networks/loss.py`: VGG16Loss.__init__/vgg_forward ([1,2,1] VGG pooling replacement). Not a PCGAN blur-loss binding.
- `third_party/source_cache/swapping_autoencoder/models/networks/stylegan2_layers.py`: Downsample.__init__/forward (caller-supplied filter). Not a PCGAN blur-loss binding.
- `third_party/source_cache/swapping_autoencoder/models/swapping_autoencoder_model.py`: compute_generator_losses (no PCGAN L_recblur). Not a PCGAN blur-loss binding.

**Owner resolution:** `torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)`. N×C×256×256 → N×C×128×128; each output is the arithmetic mean of its disjoint 2×2 input block. Apply independently to both target and generated mixed image before the existing distance. No blur-induced detach, learnable parameters, rounding/quantization, padding, post-resize or alternative filter.

**Provenance:** `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION`; not a paper, official PCGAN, author-specified or official Swapping Autoencoder operator. This minimal parameter-free low-pass choice adds no sigma, coordinate mode, align_corners choice, antialias toggle or boundary extension. It does not purport to recover private author behavior.

All five frozen losses, unit weights, alpha=0.2, beta=1e-6 and all other scientific settings remain unchanged. Base YAML and snapshot are unmodified. A1–A4 are unmodified.

## Contract artifacts

- `configs/methods/e05_pcgan.yaml` — SHA256 `478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de`
- `frozen_config_snapshot/configs/methods/e05_pcgan.yaml` — SHA256 `478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de`
- `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A5_E05_Blur_Operator_Resolution.md` — SHA256 `05d6c6c271bf23aece3b2fcf9cdef5d94bd32517421a0ac83be26c5a9174d4dd`
- `configs/amendments/e05_a5_blur_operator_resolution.yaml` — SHA256 `af28a48b43828652a7423235aba6be239325e3fbfeeaf8fded9bbb57c11a7ca1`
- `tests/test_m6a7_e05_blur_contract.py` — SHA256 `e004cda10df32f14cca26a85c8a771d87978f00be16c2ba7ba835283d59a195e`

## Validation

Focused tests: **13/13 PASS**, zero failures/errors/skips. Synthetic NumPy reference only; no PyTorch execution or dependency installation. Tests check base/snapshot hashes, overlay scope, exact pooling arguments, constants, numeric means, disjoint windows, boundaries, determinism, both branches and prohibited fallbacks.
M6B validator: **PASS, 0 failures**. Full repository suite not run.
Observed validation environment: Python 3.12.3, NumPy 2.5.3, PyYAML 6.0.3; not final benchmark pins.
Existing StaticAccessAudit on the focused test process: benchmark image opens **0**, manifest opens **0**, rejected data opens **0**. Other work was source/config/audit metadata inspection only.

## Ledger and finalization

Append exactly one `M6A7_E05_BLUR_OPERATOR_RESOLUTION` record: 90 committed rows → 91 total. First 90 rows must remain byte-identical.
Committed prefix SHA256: `027158272934e6b4c0ac9504033afb67d97e7c5e3034caa47584bdb9097a0450`.
Rebuild current artifact index with `python tools/build_artifact_index.py`; verify unique paths, sizes, hashes and CRLF after final writes. Ledger/index hashes are reported externally to avoid self-referential hashes.

No training, benchmark image execution, PCGAN/E07c implementation, synthetic bank, GPU job, scientific checkpoint, frozen E05 config change, commit or push.
