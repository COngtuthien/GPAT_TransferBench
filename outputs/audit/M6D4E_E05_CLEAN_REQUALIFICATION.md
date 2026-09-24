# M6D4e — E05 clean synthetic requalification

**M6D4e CLEAN REQUALIFICATION PASS** — synthetic numerical/runtime status **PASS**; workflow scope **PASS**; `scope_exception = false`.

The already committed M6D4d implementation was executed unchanged in two new GPU processes. No scientific implementation, contract, precision, crop, noise, grid_sample, environment or build decision changed. E05 remains **IMPLEMENTED_NOT_EXECUTED**, **CONTROLLED_ADAPTATION**.

**M6D4D_SCOPE_EXCEPTION_RETAINED_AS_HISTORICAL_EVIDENCE.** M6D4d audits, implementation, tests, preflight, M6D4c resolution and ledger row 109 remain byte-identical. This milestone adds independent clean evidence; it does not relabel M6D4d.

## Authority and synchronization

Laptop start: HEAD = origin/m6-baselines = `32c8f979e14d1ea37ee2047cbd5489cb83c64b72`; divergence `0 0`; clean. The sole permitted exact `AGENTS.md` check was absent; no parent/instruction search followed.

GPU start: HEAD = origin/m6-baselines = `c27a110258f58f9bc035875aa40caf1ae681f9cb`; divergence `0 0`; clean. Ancestry proved locally and after the prescribed fetch: **one commit**. `git merge --ff-only origin/m6-baselines` succeeded. No cleanup was needed.

GPU after sync and after execution: HEAD = origin = `32c8f979e14d1ea37ee2047cbd5489cb83c64b72`; divergence `0 0`; clean. Execution used the committed checkout directly, not the prior temporary M6D4d copy. Exact JSON evidence transfers were SHA256-verified.

## Committed identity

| File | SHA256 |
| --- | --- |
| `methods/pcgan/training_losses.py` | `08f159622d41becb8df2502be297e3d4ffe05e5e4f3ff3bc6a074c1fbaa67202` |
| `methods/pcgan/training_runner.py` | `d249722dd7653e981ab7cd8dcf734ab8345bc89074ece5ba01e04489cefd92e5` |
| `methods/pcgan/training_diagnostics.py` | `68bed70699a7a6457ea480a9fd718ed5badf78c7078be61b3b76659fcb1619f4` |
| `methods/pcgan/training_qualification.py` | `617ac13b2373620028438ef0b2c370eb39895f46cf1a3b1a456863dfcda8a8b7` |

M6D4c overlay: `8eee350c5642f2de38c92af366b99e80f7b6dab4404433ff88ca71da87b92828`.

Environment lock: `849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50`.

Compatibility manifest: `b5d4d279c3332e6a3e3044e526d0b1304e0084cfc1450b38ce1a368a578052ac`. All 67 generated source files match the retained closure before/after. Pinned source commit `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`, tree `25a434f1282a0177d6afdcc68ebfcf40bd805621`.

`gpat-m6-e05`: Python 3.11.16, Torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, RTX 3090 / driver 595.84. Executable SHA256 `d3255ce6eac7a06a653a6f238d703ebf863423912b39ec957334ce2974364244`. Complete package, compiler, driver and launch-environment inventories equal M6D4d and are retained in each process JSON.

| Existing CUDA binary | SHA256 |
| --- | --- |
| fused_bias_act | `5afdb2369817ffea68530b1e88d75596a77ed08a397f83122b618a18225718ab` |
| upfirdn2d | `40ba9db73e1790cd3d904a1fed09c5441180c6995a9a396030cfaa52efd1d081` |

Existing real binaries and Ninja recipes were verified before/after; no rebuild or fallback. **FP32; TF32, AMP, autocast and cuDNN benchmark off; cuDNN deterministic flag on**, unchanged from M6D4d.

## Scope and process firewalls

Recursive filesystem search count: **0**. The following declarations all equal **false**:

- `recursive_filesystem_search_performed = false`
- `recursive_home_search = false`
- `recursive_media_search = false`
- `recursive_runtime_search = false`
- `generic_agents_discovery = false`
- `benchmark_filename_enumeration = false`
- `benchmark_data_access = false`

Operation classes: repository Git metadata operations; explicit known-path reads; explicit environment/build identity reads; explicit GPU repo commands; synthetic model execution. Exact temporary evidence writes/transfers, six named test modules, and ledger/index writes are detailed in the JSON audit. No generic test or instruction discovery.

Python import/package metadata directory operations are included in the counters below. The shell/session scope assertion is separate from Python hooks. Counters are retained as observed; benchmark/sample access counts are zero gate outcomes from the reviewed synthetic harness. These are not an OS-wide native/subprocess I/O trace.

| Execution | open | os.listdir | os.scandir | Denied actual operations |
| --- | ---: | ---: | ---: | ---: |
| Qualification 1: PASS | 3760 | 412 | 306 | 0 |
| Qualification 2: PASS | 3760 | 412 | 306 | 0 |
| gpu focused tests | 3668 | 320 | 306 | 0 |
| laptop focused tests | 1287 | 53 | 0 | 0 |

Both qualification processes used the identical committed `ReadOnlyRuntimeFirewall`; GPU tests used that firewall too. No benchmark data, TRAIN/VAL/TEST samples, pair-manifest records, image filenames or image bytes were accessed. Synthetic rejection tests call guards with fabricated paths; they perform no benchmark operation. Two sandbox-denied SSH connection attempts preceded successful authorized retries; neither executed remotely.

## Live model and optimizer inventories

| Component | Scalars | Parameter tensors |
| --- | ---: | ---: |
| Encoder | 929,960 | 17 |
| Generator | 3,699,233 | 42 |
| ImageD | 28,859,521 | 38 |
| PatchD | 24,522,145 | 42 |

**G = 4,629,193 scalars / 59 tensors; D = 53,381,666 / 80 tensors.** Exactly two Adam optimizers per qualification process; lr=1e-6, betas=(.9,.999), weight_decay=0. No duplicate identity or group overlap. Every identity, name, shape and numel is retained.

Each process seeded Python/NumPy/Torch CPU/CUDA with diagnostic seed **60401**. The committed deterministic in-memory FP32 source/target construction is unchanged. Source is `(xx, yy, sin(7xx)cos(5yy))`; target is `(cos(3xx), sin(4yy), xx*yy)`, from the same 256-point [-1,1] grids, shape `[1,3,256,256]`. No benchmark loader or image decoder is invoked.

## Exact measured losses

Every value below is identical in M6D4e processes 1 and 2 and both retained M6D4d processes (absolute difference **0**). Comparisons happen after execution; expected scalars are not injected into the runner.

| G term | Initial G probe | Actual post-D G |
| --- | ---: | ---: |
| L_rec | 241.71209716796875 | 241.71209716796875 |
| L_recblur | 130.07997131347656 | 130.07997131347656 |
| L_advrec | 0.6924521923065186 | 0.6924460530281067 |
| L_advmix | 0.6910748481750488 | 0.6910701990127563 |
| L_pat | 0.549139678478241 | 0.6085895299911499 |
| L_G_total | 373.7247009277344 | 373.7841491699219 |

| D term | Component probe | Actual D step |
| --- | ---: | ---: |
| L_D_real | 0.8617923259735107 | 0.8617923259735107 |
| L_D_rec | 0.6938426494598389 | 0.6938426494598389 |
| L_D_mix | 0.6952238082885742 | 0.6952238082885742 |
| L_D_image | 1.5563255548477173 | 1.5563255548477173 |
| L_D_patch_real | 0.6083027124404907 | 0.642215371131897 |
| L_D_patch_fake | 0.81983482837677 | 0.772175133228302 |
| L_D_patch | 1.4281375408172607 | 1.4143905639648438 |
| L_D_total | 2.9844632148742676 | 2.9707159996032715 |

Source-only unsquared Euclidean L_rec; exact A5 L_recblur; raw-logit softplus and qualified PatchD crop/feature interface. G weights 1/1/1/1/1; image-D 1/.5/.5; patch-D 1/1; total D=image+patch. External FP32 sums exactly reproduce totals. Alpha=.2 and beta=1e-6 retain their resolved PMN scope.

## Gradients, isolation, Adam and iteration

**D: 80/80 non-None, finite and nonzero gradient tensors. G: 59/59.** All per-loss connectivity gates pass; full input/code/component inventories remain in both process records.

D changes ImageD 38/38 and PatchD 42/42 parameter tensors, while Encoder 17/17 and Generator 42/42 remain byte-identical. G changes Encoder 17/17 and Generator 42/42 while ImageD/PatchD remain byte-identical. Inactive gradients are None. Counts describe tensors, not every scalar element.

D Adam owns exactly 80 states; G Adam 59. All steps equal 1, moments are finite FP32 with parameter-matching shapes and no foreign ownership. Both D fake paths detach. Fresh post-D G paths have two E and two G forward calls at t=0 using the same explicit source/target objects.

`before_D → D_forward → D_backward → after_D → before_G → G_forward → G_backward → after_G`; counter is 0 for the first seven events and 1 after G succeeds. Probes apply no optimizer step. **Per process D=1, G=1, total=2. M6D4e total D=2, G=2, total=4.** Regression tests block real Adam.step; no further applications.

## Measured nondeterminism

**445 / 1662 tensor hashes differ** between the new processes. Categories: 251 G-probe gradients, 2 D-probe input gradients, 56 G-step gradients, 111 G Adam moment records, 25 post-G parameters. D post-step parameters and Adam states are bitwise equal. Initial parameters, input tensors, model/group inventories, noise RNG sequence, events and scalar losses match.

| Summary statistic | Maximum absolute difference |
| --- | ---: |
| min | 9.5367431640625e-07 |
| max | 9.5367431640625e-07 |
| mean | 4.76837158203125e-07 |
| l2 | 9.815629660181457e-07 |

**These summary differences are not elementwise error bounds.** No tensor/checkpoint payloads were saved. Native CUDA backward accumulation including bilinear interpolation/grid_sample remains a plausible source, inferred rather than profiled. Full path-specific differences and all four new-to-historical comparisons are retained in the JSON audit. Scientific operations were not changed to force equality.

Peak CUDA allocation: 3734135296 bytes, 3734135296 bytes. No OOM/refactor or precision change.

## Tests, static boundary and evidence finalization

**GPU: 111 passed, zero skipped. Laptop: 95 passed, 16 skipped (Torch absent), 111 total.** All committed tests unchanged; no new tests or full repository suite. Both test launchers have zero actual firewall denials and zero real optimizer applications.

Checkpoint policy validated statically: BASELINE_FINAL_STATE_V1; terminal benchmark_iteration=4000; no VAL selection, TEST selection, best seed or extra applications. No torch.save scientific state, checkpoint bytes, pretrained load or synthetic bank.

Ledger **109 → 110**, exactly one `M6D4E_E05_CLEAN_REQUALIFICATION` row. First 109 rows byte-identical: **290581 bytes**, SHA256 `6af0ac0b6e9145addd996b4415a3bca118d1d2a09ffe2cba3c36612bbcf05357`. M6D4d row 109 is preserved.

Artifact index **566 → 572**, CRLF. Rebuilt **last** from committed metadata plus exactly the six specified new paths; all original rows remain byte-identical and unrelated indexed targets are not opened.

Static preflight (imports no Torch, executes no model) validates retained numerical/runtime gates and workflow-scope declarations independently, historical identity, ledger prefix and exact index bytes:

```sh
python3 -I -S -B tools/m6d4e_e05_clean_requalification_preflight.py --rebuild-index
python3 -I -S -B tools/m6d4e_e05_clean_requalification_preflight.py
```

Final laptop state: authority HEAD/origin unchanged; six new artifacts; only EXECUTION_LEDGER.jsonl and ARTIFACT_INDEX.csv modified. GPU clean. No commit or push.

**E05_ARCHITECTURE_RUNTIME_QUALIFIED**

**E05_TRAINING_RUNNER_CONTRACT_RESOLVED**

**E05_TRAINING_RUNNER_QUALIFIED**

**E05_CLEAN_REQUALIFICATION_PASS**

**E05 REMAINS IMPLEMENTED_NOT_EXECUTED**

**CONTROLLED_ADAPTATION PRESERVED**

**M6D4D_SCOPE_EXCEPTION_RETAINED_AS_HISTORICAL_EVIDENCE**

**NO RECURSIVE FILESYSTEM SEARCH. NO BENCHMARK FILENAME ENUMERATION. NO BENCHMARK DATA ACCESS. NO BENCHMARK TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
