# M8 learned-bank S7 toy acceptance: executable plan

Status: **PREPARED, NOT EXECUTED.** The machine-readable version is `LEARNED_BANK_S7_ACCEPTANCE_PLAN.json`.

Governing contract: `configs/amendments/learned_bank_generation_contract.yaml`, approved at `d0b4e4e`:

- C-Q3 `A_TOY_TWO_RUN_BITWISE`
- C-Q2 `A_GPU_FP32_BITWISE_REPLAY`
- C-Q5 `A_AFTER_E04_COMPLETES`

Code: adapters and infrastructure at `b09a1cc`. CPU fixture qualification at `fe57530` passed 125/125.

**No learned-method GPU inference may start while E04 seed 2026 trains.** On 2026-11-22T15:39Z it was at step 41546 of
150000.

## 1. Preconditions

The CLI checks P2–P4 itself and refuses with exit 3 if any is missing.

| Id | Gate |
|---|---|
| P1 | **Genuine E04 seed 2026 completion.** Runner exit marker `~/workdir/e04_s2026.exit` holds code 0, global_step is 150000 and the final checkpoint exists. tmux disappearance or a final checkpoint alone is not completion. |
| P2 | `outputs/audit/E04_SEED2026_COMPLETION.json` is committed: status `E04_SEED2026_COMPLETED_CLEANLY`, `failed_gates: []`, run `e7100eb9a5ab0537`. |
| P3 | `outputs/audit/E04_THREE_SEED_TRAINING_CLOSURE.json` is committed with status `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, or an owner-approved documented roadmap authority is named in the S7 authority. |
| P4 | The owner commits `configs/amendments/learned_bank_s7_launch_authority.yaml`. |
| P5 | Resource preflight passes immediately before each run A (details below). |
| P6 | A stat-only atime ledger of every canonical face is taken before run A and after run B, for E03 and E05. |

P4 fields:

- `schema_version: learned_bank_s7_launch_authority_v1`
- `status: OWNER_APPROVED`, `approved: true`
- `methods`
- `contract_sha256`
- `e04_seed_2026_completion_sha256`
- `resource_preflight.required: true`
- the code authority commit

P5 resource preflight:

- No `tools/run_e04.py` process is running.
- GPU UUID is `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`.
- Free GPU memory is at least the method estimate plus 2 GiB. Unrelated users' jobs are recorded, never touched.
- At least 2 GiB is free on the S7 filesystem.
- `tools/learned_bank.py gates` reports correctly and all 12 checkpoints re-hash correctly.

## 2. Setup on the GPU host

Use an isolated clone. Never use the E04 checkout `18ba00e`.

```bash
C=<S7 authority commit>
git clone -q <bundle> ~/workdir/m8_learned_banks/clones/s7_$C -b m6-baselines   # clean, at $C
cd ~/workdir/m8_learned_banks/clones/s7_$C
mkdir -p third_party/source_cache
for c in stdn swapping_autoencoder facexzoo; do
  cp -a ~/workdir/GPAT_TransferBench/third_party/source_cache/$c third_party/source_cache/$c
done
RT=/home/student20261/workdir/GPAT_TransferBench_runtime
S7=~/workdir/m8_learned_banks/s7/$C
mkdir -p $S7                         # outside $RT, never on the HDD bank root
```

## 3. Per method

Run order: E06c → E05 → E03 → E06b. Only one GPU job runs at a time. For each method:

1. Run A in tmux `m8s7_<M>_A` (remain-on-exit, pipe-pane log, exit code written to `~/workdir/m8s7_<M>_A.exit`).
2. Wait for run A to exit.
3. Run B in tmux `m8s7_<M>_B`, as a separate process.
4. Compare the two runs.

```bash
$LAUNCH_ENV nice -n 10 <env python> -B tools/learned_bank.py s7 --method <M> --label A --out-root $S7 --runtime-root $RT
$LAUNCH_ENV nice -n 10 <env python> -B tools/learned_bank.py s7 --method <M> --label B --out-root $S7 --runtime-root $RT
<env python> -B tools/learned_bank.py s7-compare --method <M> --out-root $S7
```

Each run writes `$S7/run_{A,B}/<M>/seed_{42,1337,2026}/`, with 186 rows and 186 PNGs per seed.

| | E06c | E05 | E03 | E06b |
|---|---|---|---|---|
| Environment | gpat-m6-e06c | gpat-m6-e05 | gpat-m6-e03 (Py 3.8, TF 1.15.5) | gpat-m6-e06c |
| Checkpoint | `netG_model_epoch_200_iter_0.pth` | `iteration_4000.pth` | `ckpt-50` (data/index/meta) | `netG_model_epoch_200_iter_0.pth` |
| Population | stream indices 0..185 | 186 toy pairs | 186 toy pairs | stream indices 0..185 |
| Face reads per seed | 0 | 372 (O_NOATIME) | 372 (O_NOATIME) | 0 |
| Determinism | torch flags, cudnn.benchmark off | `configure_precision` | M6D2b `precision_env` + `session_config` | torch flags |
| GPU memory (provisional) | 2–4 GiB | ≤ 3 GiB | ≤ 2 GiB | 2–4 GiB |

Population details:

- DSDG methods (E06c, E06b): 2 batches of 100, 14 samples discarded per seed.
- Pair methods (E03, E05): the 186 toy pairs of `outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json`
  (`50ac8fb8…`), at their global positions.

Launch environments:

- PyTorch (E05, E06b, E06c), `$LAUNCH_ENV`:

  ```
  env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  ```

- E03: `env -i` followed by exactly `methods.stdn.production_io.PRECISION_ENV`, with `CUDA_VISIBLE_DEVICES=0`. The JSON
  plan lists the full string.

Exact checkpoint SHA256, run IDs and DSDG stream seeds per seed are in the JSON plan and in
`methods/common/learned_bank.py::CHECKPOINTS`.

Profiling:

- Run A of each PyTorch method reports `torch.cuda.max_memory_allocated` per seed.
- For E03, sample `nvidia-smi --query-compute-apps` every 2 s during run A.
- The measured peak plus 2 GiB becomes the production preflight floor.

## 4. Audit gates

The S7 record for each method is `outputs/audit/LEARNED_BANK_S7_<M>.json`, built on the laptop from host-collected
evidence. It must contain `seeds [42, 1337, 2026]`, `failed_gates []` and `s7_acceptance_passed: true`. These are the
fields `load_production_authority` requires.

Gates:

1. Checkpoint SHA256 equals the approved table before run A and after run B, so no checkpoint was mutated.
2. Config SHA256 and pinned-source digests are unchanged.
3. The inference graph is the correct one:
   - E03: graph-structure record.
   - E05: runtime closure and strict E/G state load.
   - DSDG: full `load_model` coverage.
4. The population is correct:
   - pair methods: toy positions and rows equal the frozen relation;
   - DSDG: indices 0..185, synthetic IDs, null lineage.
5. Each seed and run has 186 rows, 186 files and 0 failures, with every raw output finite.
6. `lb.read_bank` passes with `expect_mode='S7_TOY'`, 186 rows and the approved checkpoint identity.
7. **Bitwise replay:** `lb.compare_runs(run_A, run_B)` shows identical manifest bytes and identical bytes for every PNG.
8. No optimizer, backward pass or save happened: torch guard counters are 0, and the TF graph is finalized.
9. Reads were TRAIN-allowlist only, 372 O_NOATIME reads per seed for pair methods. The atime ledger shows no VAL or TEST
   face touched.
10. Provenance binds mode `S7_TOY` to the S7 authority, the contract and the E04 record.
11. Later, each production audit checks that every toy row equals its production row byte for byte.

## 5. Failure behaviour

- **Any gate failure:** stop that method, leave both roots and logs untouched, never reuse a root, and report to the owner.
- **Replay not bitwise:** stop for owner review (C-Q2). The criterion is not weakened, and no architecture, checkpoint,
  graph or output change is made. Any further determinism setting is disclosed to the owner before activation.
- **Non-finite rows:** they are counted failures. S7 requires 0, so any failure stops the method.
- **GPU contention from other users:** never interfere. Repeat later in new roots.

## 6. Production unlock, per family and separately

Production for a method opens only when all of these hold:

1. The S7 record for that method passes for all three seeds.
2. The owner commits `configs/amendments/learned_bank_production_<M>.yaml` (`learned_bank_production_authority_v1`),
   binding the S7 record (path + SHA256) and the contract SHA256.
3. Storage is activated:
   - the owner has created `/media/ext_data/student20261_gpat_m8_banks` with `chmod 00700`;
   - the live preflight passes;
   - `configs/amendments/learned_bank_storage_location.yaml` is committed as OWNER_APPROVED with the bank-root registry.

   See `configs/amendments/learned_bank_storage_location.PENDING_ACTIVATION.yaml`.
4. A per-bank launch preflight passes, re-checking UUID, mount, owner, mode, free space, checkpoint re-hash and the
   absence of E04 or other learned jobs.
5. The production audit includes the toy-row equality check.
