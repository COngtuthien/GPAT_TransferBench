# E04 Physics-STD S7 acceptance plan (prepared, NOT launched)

**Status:** `PREPARED_NOT_LAUNCHED`. Contract `configs/amendments/e04_bank_generation_contract.yaml` (`45240cd25e840c16`); fixture qualification `E04_BANK_ADAPTER_SYNTHETIC_QUALIFIED_S7_PENDING`.

## Checkpoints

| seed | run_id | identity |
|---|---|---|
| 42 | `40befa6004aee85c` | `96e89630150779848cc92c4c2756d01cbfeb48e6b0e8b87b815122e47dc2d30f` |
| 1337 | `a42fdd2a2accaf0a` | `4f183d2143d27d1d35326212f9ff9c0fb36e9d0d933e59c1df79436cc3492f2f` |
| 2026 | `e7100eb9a5ab0537` | `9b0b391bacfece7b1e4be3fb395f13f1c53e18b138ee7fda487df7a037a38d20` |

## Population

- 186 toy pairs (`50ac8fb8b9fc6938`) x seeds [42, 1337, 2026] x runs ['A', 'B'] = **1116 images**, batch 1.

## Workflow

1. owner commits configs/amendments/learned_bank_s7_launch_authority_E04.yaml (from configs/amendments/learned_bank_s7_launch_authority_E04.DRAFT.yaml) after the E04 S7 audit tool exists
2. git bundle of the execution commit -> ~/workdir/m8_learned_banks/bundles/; fresh clone ~/workdir/m8_learned_banks/clones/s7e04_<execution_commit>; mkdir -p third_party/source_cache and copy stdn from the GPU checkout (read-only); training checkout 18ba00e never used
3. resource preflight (S7 authority checks) and atime ledger of all TRAIN/VAL/TEST faces (stat only) -> ~/workdir/m8s7_E04_A_atime_pre.tsv
4. mkdir -p /home/student20261/workdir/m8_learned_banks/s7/<execution_commit>/E04 (outside the runtime root and the HDD bank root)
5. tmux m8s7_E04_A (remain-on-exit, pipe-pane log): run A command; exit code -> ~/workdir/m8s7_E04_A.exit; nvidia-smi --query-compute-apps sampled every 2 s
6. after A exits 0: tmux m8s7_E04_B: run B command; exit code -> ~/workdir/m8s7_E04_B.exit
7. compare command; atime ledger post; checkpoint re-hash
8. independent read-only host collector + laptop builder -> outputs/audit/LEARNED_BANK_S7_E04.json

## Commands

- **compare**: `env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES= NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 nice -n 10 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B tools/learned_bank.py s7-compare --method E04 --out-root /home/student20261/workdir/m8_learned_banks/s7/<execution_commit>/E04`
- **gates**: `env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES= NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 nice -n 10 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B tools/learned_bank.py gates --method E04`
- **run_A**: `env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 nice -n 10 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B tools/learned_bank.py s7 --method E04 --label A --out-root /home/student20261/workdir/m8_learned_banks/s7/<execution_commit>/E04 --runtime-root /home/student20261/workdir/GPAT_TransferBench_runtime`
- **run_B**: `env -i HOME=/home/student20261 PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TF_CUDNN_DETERMINISTIC=1 TF_DETERMINISTIC_OPS=1 TF_ENABLE_AUTO_MIXED_PRECISION=0 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0 TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS=--tf_xla_auto_jit=0 nice -n 10 /home/student20261/miniconda3/envs/gpat-m6-e03/bin/python -B tools/learned_bank.py s7 --method E04 --label B --out-root /home/student20261/workdir/m8_learned_banks/s7/<execution_commit>/E04 --runtime-root /home/student20261/workdir/GPAT_TransferBench_runtime`

## Acceptance gates

- checkpoint SHA256 of all 3 files per seed equal before run A and after run B (lb.verify_checkpoint / recheck_checkpoint_unchanged)
- frozen config, contract, frozen training sources and predecessor conversion lines verified (bank_adapter.verify_frozen_sources)
- environment lock verified on the GPU (production_runner.verify_environment, D9 driver policy)
- inference graph structure equal to the fixture-qualified identity (variable_names_sha256, variable_shapes_sha256, 15 FusedBatchNorm is_training=False, 0 update/gradient/optimizer/random ops)
- generator-only restore: 102 restored of 500 checkpoint variables; moving statistics finite
- population: 186 rows per seed equal to the toy manifest positions and the frozen relation
- 186 decodable 256x256 RGB PNGs per seed and run, 0 failures, all raw outputs finite
- reader: lb.read_bank ok with expect_mode S7_TOY, expect_rows 186, expect_checkpoint_identity
- every PNG re-decoded by the audit: pixel and file SHA256 equal to the manifest; conversion rule string E04-Q3; raw output finite with recorded min/max per row
- bitwise replay: lb.compare_runs(run_A, run_B) bitwise_identical for every seed (E04-Q5)
- TRAIN-only allowlist; reader record reads == 2 x 186 O_NOATIME atime-checked reads per seed and run
- atime ledger: no VAL/TEST face atime changed; TRAIN atimes unchanged
- provenance: authorization S7_TOY bound to the E04 S7 authority, the E04 contract and the family closure
- later (production audit): every toy row equals its production row byte for byte

**Failure:** any failed gate or non-bitwise replay STOPS the E04 family for an owner ruling; evidence is preserved; no automatic rerun into an existing root; never a CPU or tolerance fallback

## Remaining before launch

- E04 branch of the independent S7 audit (tools/learned_bank_s7_audit.py)
- owner activation of the E04 S7 launch authority

## Production unlock

- LEARNED_BANK_S7_E04.json PASS for all three seeds
- configs/amendments/learned_bank_storage_location_E04.yaml activated
- learned_bank_production_E04.yaml activated (template configs/amendments/learned_bank_production_E04.DRAFT.yaml)
- E04 branches of the production authority/audit tools
- live storage preflight at every launch

## Governance DRAFTs (not loadable)

- `configs/amendments/learned_bank_s7_launch_authority_E04.DRAFT.yaml` — status `DRAFT_PENDING_OWNER_DECISION`.
- `configs/amendments/learned_bank_storage_location_E04.DRAFT.yaml` — status `DRAFT_PENDING_OWNER_DECISION`; live HDD state 2026-11-25T12:57:14Z: preflight ok True, unexpected [], free 2991.1 GiB.
- `configs/amendments/learned_bank_production_E04.DRAFT.yaml` — production template, not approvable before S7.

**Resources:** single RTX 3090; batch-1 TF1 generator < 2 GiB expected; CPU warp + scipy offsets dominate (estimate 0.3-0.6 s per pair: about 1-2 min per seed per S7 run; 45-90 min per production seed); to be measured in run A
