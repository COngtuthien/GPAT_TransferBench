# M6D3e — E04 training graph and runtime qualification

**Decision: E04_TRAINING_GRAPH_AND_RUNTIME_QUALIFIED.** Synthetic qualification only. E04 remains **IMPLEMENTED_NOT_EXECUTED**, fidelity **CONTROLLED_ADAPTATION**.

Reporting label: Physics-STD (controlled geometry/depth reconstruction). Normative M6D3d contract and all frozen inputs/history are unchanged.

## Synchronization and runtime

Laptop HEAD/origin were `90263169f0731b397dfa459d7ac94eaafd2c69ec`, tracked-clean, divergence0/0. GPU was ancestor `374bd5dd7b56960e4ec3554dfbbd084061701cfd`, exactly3 commits behind. The prescribed fetch and fast-forward merge reached the authoritative HEAD/origin with tracked-clean status and divergence0/0 before new files were copied. A transient GitHub DNS failure required only a command-local SSH HostName override with HostKeyAlias verification. No reset, force checkout, commit or push.

Reused **gpat-m6-e03 read-only**, owned by E03. No environment was created or mutated. Explicit package metadata, interpreter SHA256, E03 lock and GPU identities are identical before/after. Python3.8.20; NVIDIA TensorFlow1.15.5+nv22.12; CUDA11.8.89 (native11080); cuDNN8.9.7.29 (native8907); RTX3090; driver595.84. FP32; TF32/AMP/XLA auto-jit and automatic precision rewrites disabled. Native bilinear sampling runs on CPU, convolutions on GPU. CUPTI kernel profiling was unavailable; executor placement evidence is retained.

E04 environment-reference lock SHA256: `a9629d46c2464f011fb68e0f071ecd4cc21cf0ede5af68faf9ab0d4595fefbdd`.

## Implemented files

- `methods/physics_std/training_contract.py`
- `methods/physics_std/network.py`
- `methods/physics_std/trace_transfer.py`
- `methods/physics_std/losses.py`
- `methods/physics_std/training_graph.py`
- `methods/physics_std/training_runtime.py`

The additive implementation record is `docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Training_Graph_Implementation_M6D3e.md`; it was written before optimizer execution. Tests, preflight, E04 environment references and both process-evidence JSON files are enumerated in the companion audit JSON. Existing method files, earlier tests, frozen configs and M6D3b/c/d artifacts were not edited.

## Measured structure

Production construction probe:11,315 operations, no Session/apply. Each full qualification graph including diagnostics: **11,671 operations**, **1,763,902 trainable parameters**,144 trainable variables and356 nontrainable variables (62 BN statistics,288 moment slots,4 beta powers,2 counters). All41 kernels use Normal(0,.02); biases0 and BN gamma1/beta0 are explicit standard exceptions.

| Component | Parameters |
|---|---:|
| D/D1 | 187297 |
| D/D2 | 187297 |
| D/D3 | 187297 |
| D/D4 | 187297 |
| G/decoder | 197773 |
| G/depth | 92813 |
| G/encoder | 724128 |

F1=`[8,128,128,64]`; F2=`[8,64,64,96]`; F3=`[8,32,32,128]`. Decoder raw=`[8,256,256,13]`, channel layout B3/C3/T3/P1/IP3. Depth=`[8,32,32,1]`, bounded[0,1]. D1–D4 are independent, input scales32/96/256/256 and output patches4/12/32/32. No STDN multiplicative component, RGB/YUV input transplant or G_D_RATIO.

Every output tensor name/shape, variable name/shape, initializer measurement and native-state hash is retained in each process JSON.

## Synthetic losses and gradients

Generated batch8 contains4 live and4 spoof patterns. Live depth is an analytic face-like fixture; spoof/hard-spoof depth is exact float32 zero. Geometry uses140 generated correspondences; no3DDFA or real depth was loaded.

| Loss | Initial value (both processes) | Connected variables |
|---|---:|---:|
| L_depth | 0.433335959911346 | 70 |
| L_G | 571.853088378906 | 58 |
| L_P | 51566.79296875 | 58 |
| L_R | 48921.9140625 | 58 |
| L_S | 59626.421875 | 58 |
| L_H | 0.515586793422699 | 70 |
| L_D | 1129.3291015625 | 72 |

Outside-graph NumPy FP32 recomputation equals graph values exactly: Step1=`54474.28125`, Step3=`596264.75`; absolute errors0. Rechecked before every apply. All individual losses, outputs and checked gradients are finite.

P0 is exactly zero; negative prior is a **Const with zero inputs**, value0, so no normalized quotient or epsilon is evaluated. The positive P term is active (`51566.79296875`); predicted P is nonzero and its primary gradient is nonzero. IP has a nonzero adversarial gradient. Synthesized inputs and matching trace targets are detached. Intentional disconnections are only the auxiliary depth branch for trace losses and the final trace projection for depth losses; all72 variables connect to each intended optimizer objective.

## Exactly one minibatch per clean process

| Step | Apply | Changed trainable variables | Unrelated trainables | Native BN changes | Iteration |
|---|---|---:|---|---|---|
| 1 | G Eq.23 | 72 | D unchanged | G30 then D32 | 0→0 |
| 2 | D L_D | 72 | G unchanged | G30 then D32 | 0→0 |
| 3 | G Eq.24 | 72 | D unchanged | G30 then shared G30 | 0→1 |

One G Adam owns the same m/v tensors and beta accumulators in Steps1/3; D owns separate slots. G beta1 power progresses .9→.81→.81→.729; D .9→.9→.81→.81 (FP32 measurements retained). G beta2 powers similarly advance twice, D once. No optimizer reset or extra apply. Both processes each execute exactly3 applies; total6. Native BN updates during the other group’s step are explicitly recorded and do not modify its trainable parameters.

| Complete iteration t | G LR | D LR |
|---:|---:|---:|
| 0 | 5e-5 | 2.5e-5 |
| 44999 | 5e-5 | 2.5e-5 |
| 45000 | 5e-6 | 2.5e-6 |
| 89999 | 5e-6 | 2.5e-6 |
| 90000 | 5e-7 | 2.5e-7 |
| 135000 | 5e-8 | 2.5e-8 |

All boundary checks pass at FP32 representation tolerance; D is exactly G/2. All three applies see t=0. Terminal benchmark iteration150000 and experiment seeds42/1337/2026 remain frozen; qualification-only seed314159 does not select a benchmark result.

## Repeatability and limits

Two clean processes have **byte-identical serialized GraphDefs**, hash `7e3e84ccc40f8390819b0121b417e5ebfea78de3efddb1e306d07e70bd4c27e2`. Variable inventory, parameter counts, initialized state, initial outputs/losses, every checked loss/step gradient, all G post-state, Step1 whole state and final losses match exactly. The auxiliary Python operation-list hash differs despite identical serialized GraphDefs; it is sensitive to native control-input enumeration order.

**Post-step state is not fully bitwise repeatable.** After Step2,51 D3 state tensors differ (13 trainable); the same differences persist through Step3. Maximum absolute differences between per-tensor trainable summary statistics are min1.06673e-6, max1.06671e-6, mean4.60513e-6, L2 norm4.72715e-6. Including Adam slots, the largest L2-norm summary difference is0.00906472453471. These are differences of scalar summaries, not elementwise-error bounds. Every differing tensor and its measured summaries is recorded in audit JSON.

The variance arises during native Step2 re-evaluation and is confined to D3; GPU convolution/reduction execution is the inferred cause. The exact kernel could not be established without CUPTI. This limitation is disclosed under the requested native-nondeterminism allowance; no scientific algorithm was changed to manufacture bitwise identity, and no extra optimizer steps ran.

## Tests, audit and final state

Both workers pass33/33 focused tests, no skips or failures. The laptop passes21/21 selected static E04 authority/runtime-rejection/contract regressions. All existing tests are byte-preserved. Preflight verifies evidence/hash consistency without importing TensorFlow or launching training.

Ledger:104→105, exactly one M6D3E row; first267,781 bytes (104 rows) preserved, SHA256 `5525fca5c6f3ee2d3ddb4bb9f02e0c50abfffc5e36493ab80bdf3a2b65e01eea`. ARTIFACT_INDEX:513→530, rebuilt last using CRLF and committed metadata plus17 explicit new artifacts; unchanged manifest/index targets were not opened. Final Git delta:17 new files, ledger/index modified; HEAD/origin unchanged, divergence0/0. Both repositories retain uncommitted reviewable work.

Final statuses:

- E04_GEOMETRY_RUNTIME_QUALIFIED
- E04_TRAINING_GRAPH_CONTRACT_RESOLVED
- E04_TRAINING_GRAPH_QUALIFIED
- E04_TRAINING_RUNTIME_QUALIFIED
- IMPLEMENTED_NOT_EXECUTED
- CONTROLLED_ADAPTATION

**NO BENCHMARK IMAGE DECODED. NO BENCHMARK IMAGE FILENAME ENUMERATION. NO TRAIN/VAL/TEST SAMPLE ACCESS OR SCIENTIFIC EXECUTION. NO PHY-STD BENCHMARK TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO runs/m6/E04 SCIENTIFIC RUN. NO COMMIT. NO PUSH.**

Firewall evidence covers Python open/listdir/scandir events with zero denied attempts, explicit-path command review and synthetic-only APIs. It is not claimed to be an OS-wide syscall trace.
