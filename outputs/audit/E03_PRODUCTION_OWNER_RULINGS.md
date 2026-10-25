# E03 STDN: production owner rulings and investigation evidence

**Status:** `OWNER_RESOLVED`. The machine-readable record is `E03_PRODUCTION_OWNER_RULINGS.json`, the ruling text is `docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Production_Data_RNG_Owner_Resolution_Addendum.md`, and the companion is `configs/amendments/e03_production_data_rng_resolution.yaml`.

This commit records decisions and evidence only. It is **not** `E03_SCIENTIFIC_AUTHORITY`: no production code was written, no E03 root was created, and no benchmark or TEST data was opened. The frozen config (`08dde851…`), its snapshot, the M6D2b runtime addendum (`f5133f81…`), the lock (`b2c6c2ea…`) and the source cache are unchanged.

**Resolved execution fidelity:** `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY`.

| ruling | decision |
|---|---|
| V | `SEPARATE_DETERMINISTIC_SPLIT_RNG_STREAMS`: the TRAIN parser seed is `experiment_seed`, and the VAL parser seed is `1 + (u32_be(SHA256("GPAT_TRANSFERBENCH\|E03\|VAL_DATA\|<seed>"))[0:4] mod 2147483647)`. That gives 42 → 1355875380, 1337 → 1050417158 and 2026 → 1242808125 (qualification seed 60903 → 2050698081). This is a controlled benchmark determinism adaptation. |
| R1 | TRAIN relation is `pairs_train_v1` (`a5e4fdae…`, 8838 rows): target_live → live, source_spoof → spoof |
| R2 | VAL relation is `val_pairs_v1` (`84d12491…`, 1905 rows: 576 / 240 / 1089), with 500 diagnostic steps per epoch and no selection |
| R3 | The official diagnostic figures are kept: TRAIN at `step % 200 == 0` and VAL at `step % 50 == 0`, taken from the same `sess.run` of that step |

## Source behaviour (pinned c79f1f8c; line references)

- **Schedule** (`train.py:167-170`): an even step runs one `sess.run(losses + [g_op, d_op, fig])`, and an odd step runs `sess.run(losses + [g_op, fig])`.
- **Shared step:** both `get_train_op` calls use `apply_gradients(…, global_step=global_step)` (`model/model.py:118`), so the global_step is shared.
- **Learning rate:** `exponential_decay(6e-5, global_step, 20000, 0.9, staircase)` (`:101-106`). The loss EMA is 0.9 and the weight EMA 0.9999, with num_updates = global_step (`:111, :122`).
- **Epoch order:** `saver.save(…, global_step=epoch+1)` runs after the TRAIN loop and before VAL (`train.py:186`). The VAL graph is `_step(…, training_nn=False)` (`:141`). The upstream restore (`:146-150`) is not a qualified resume.
- **Upstream feed:**
  - directories are globbed, the shorter list is padded by `random.sample`, and the lists are zipped positionally, then `shuffle(len).repeat(-1)` (`model/dataset.py:44-65`);
  - map and prefetch use AUTOTUNE (`:67-71`);
  - frame choice uses global `random.randint` (`:107/126/169/189`) and flips use global `np.random.rand() > 0.5` (`:117/136/180/200`), shared by the TRAIN and VAL parsers;
  - TRAIN-only brightness is `random_brightness(…, 0.25)` (`:151`).

## Measured trace (non-scientific, GPU host, synthetic and bundled sample data only)

**Setup:**

- **Host:** `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d…`, driver 580.178.04, gpat-m6-e03 with Python 3.8.20, M6D2b precision environment, seed 60903.
- **Firewall:** an audit hook refused the benchmark runtime, `manifests/`, and any image or array outside the source repo's bundled `data/{train,val}`.

**Schedule:**

- global_step went 0 → 2 → 3 → 5 → 6.
- **G+D step:** changes 98 G and 72 D trainable variables, 344 Adam slots, 2 loss EMAs and 60 BN moving variables.
- **G-only step:** changes 98 G variables, 198 Adam slots, 98 weight-EMA shadows, 1 loss EMA and 60 BN moving variables.

**VAL graph:**

- 1996 reachable ops, none of them state-mutating; 48 FusedBatchNorm ops, all with `is_training=False`; 0 variables created.
- After 3 VAL runs, 0 of 747 variables changed.
- The next TRAIN step's full state is `7f38c10f…` both with and without VAL.

**Parser RNG:**

- VAL pulls changed the global Python and NumPy RNG state; the no-VAL control left it unchanged.
- Comparing 328 TRAIN flip bits, the first divergence is at element 56, after prefetch drained, and 134 bits diverge.
- Repeat runs are identical within each condition.

**Command:**

```
ssh <host> "cd /tmp && env -i <M6D2b precision env> PYTHONHASHSEED=60903 MPLCONFIGDIR=/tmp/e03_trace_mpl \
  <gpat-m6-e03 python> -I -B - --source <gpu repo>/third_party/source_cache/stdn --mode {graph|rng} [--no-val] [--post-train 80]" \
  < e03_source_trace.py
```

**Trace SHA256 values.** The raw traces are ephemeral scratch files, not scientific artifacts, and are not committed.

| file | sha256 |
|---|---|
| `e03_source_trace.py` (harness) | `e146643a334daa2bfc497a3290ed15cdc98c26b58ca17fab8500e356a9879879` |
| graph, with VAL (runs a and b identical) | `42ea7e6a90ee9d3419385b4e5856de2644117172283f3e290468f9259edbf205` |
| graph, no VAL | `8aa0786b9133a32d6a04675aaf5d5f3238dba3428e732226ddeba73b638ec73e` |
| rng, with VAL (runs a and b identical) | `d51ab24cc4be675e23d9306c32475ccfa09a6039ef0eb3126a0f2436551865fd` |
| rng, no VAL (runs c and d identical) | `3877af02dcf39dc83dc3c4411adbe9347aa1ef5fd75dd29818dcccb2ec7be3a6` |

## Conclusions

1. At the TF graph level the official VAL block does not mutate state.
2. The shared parser RNG is the only VAL → TRAIN coupling, and ruling V removes it.
3. The frozen E03 artifacts do not determine the upstream directory zip or frame choice. Rulings R1 and R2 bind the Track-A common pair relations instead.
