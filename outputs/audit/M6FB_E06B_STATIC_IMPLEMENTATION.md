# M6FB — E06b DSDG-NATIVE config freeze and static adapter (M6F-B)

| Field | Value |
|---|---|
| Milestone | M6FB, classification `M6FB_E06B_STATIC_IMPLEMENTATION` (static; no training, GPU or model) |
| Authority | `4ee7ac29ca16687aa4eb2a492d2a25554ce29253` (M6H, on top of M6F-A `e936ec2`), branch `m6-baselines` |
| Frozen config | `configs/methods/e06b_dsdg_native.yaml`, SHA256 `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| Snapshot | `frozen_config_snapshot/configs/methods/e06b_dsdg_native.yaml`, byte-identical |
| Static adapter | `methods/dsdg/native.py` |
| Freeze record | `outputs/audit/M6FB_E06B_STATIC_IMPLEMENTATION.json` (anchors `load_method_config('E06b')`) |
| Verifier | `tools/m6fb_e06b_static_preflight.py` (stdlib only) |

**Status:**
- `CONFIG_FROZEN`
- `STATIC_ADAPTER_IMPLEMENTED`
- `GPU_GRAPH_NOT_YET_QUALIFIED`
- `PRODUCTION_RUNNER_NOT_YET_QUALIFIED`
- `SCIENTIFIC_TRAINING_NOT_EXECUTED`

**Fidelity:** target `FAITHFUL_OFFICIAL` (spec §8.6). The final execution fidelity is
`PENDING_M6F_C_RUNTIME_QUALIFICATION`: not claimed, and not `CONTROLLED_ADAPTATION`.

## Frozen contract (from M6F-A; nothing retuned)

**Training population:** CASIA + MSU TRAIN. SiW is `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`, with 0 rows used.

| | Spoof | Live | Subjects | print | replay |
|---|---|---|---|---|---|
| CASIA | 2,520 | 840 | 35 | 1,680 | 840 |
| MSU | 1,200 | 400 | 25 | 400 | 800 |
| Total | 3,720 | 1,240 | 60 | 2,080 | 1,640 |

**Spoof type:** attack_macro {print: 0, replay: 1}, K = 2. Any unexpected macro is rejected, never remapped.

**Losses:** lambda_pair = 5 (`train_generator.sh:17`), plus lambda_mmd 50, lambda_ip 1000, lambda_type 10 and
lambda_ort 1. All seven official losses are kept, with the spoof-type and identity-pair losses active.

**Official values:**
- 200 epochs, effective batch 240, hdim 128, Adam at lr 2e-4, official warm-up;
- the epoch-200 generator checkpoint;
- seeds 42, 1337, 2026;
- the LightCNN-29 v2 weight (SHA256 `d0750746…`) fixed across seeds.

**Bank budget:** `n_syn_intended = DEFERRED_TO_M8`. No number is chosen here.

## Static adapter (`methods/dsdg/native.py`)

- **`NativeRelation`** takes CASIA/MSU TRAIN rows and indexes the spoof frames. Each spoof frame gets a same-subject
  live pool, keyed by `(dataset, subject_id_global)`. It rejects:
  - SiW rows, and any other dataset outside the scope;
  - non-TRAIN rows;
  - rows that are not M2-COMPLETE;
  - rows with no trustworthy subject (a missing or non-dataset ID; no pseudo-identity);
  - unexpected spoof macros, and label/macro inconsistencies;
  - spoof frames without a live pool.

  Spoof index and pool order are ascending bytewise `sample_id`, independent of input order. `draw_live` is the
  official `random.choice` over the subject's pool. Nothing is materialized, and
  `manifests/dsdg_identity_pairs_v1.parquet` is never written.
- **`NativePairDataset`** replaces `GenDataset_s`. It is indexed by spoof frame, redraws the live partner on every
  load (before the image read, as in the official code), and returns the official keys `0` / `1` / `type`.
- **`DSDGNativeAdapter`** subclasses the E06c `DSDGAdapter`. It reuses:
  - source verification;
  - LightCNN verification;
  - the argv mapping and the official component import;
  - the checkpoint plan.

  It replaces the A1 semantics with `native_semantics`, which validates the config against the M6FA contract by
  SHA256. `lambdas()` feeds the parameterized M6D5c graph and does not use the E06c frozen-lambda guard.
- **Loader contract:**
  - batch 240, shuffle with `torch.Generator().manual_seed(seed)`;
  - **8 workers**, `persistent_workers=False`, `drop_last=False`;
  - `seed_torch_worker` gives each worker a Python `random` seed of (per-epoch base seed + worker_id) mod 2^32;
  - batch b is served by worker b % 8, so the worker count changes the live draws.

  `simulate_live_draws` is a pure-Python model of this, and it is deterministic. M6F-C proves it against torch.
- **Batch plan: 15 × 240 + 120** gives 16 optimizer steps per epoch and 3,200 in total. At microbatch 20, a full batch
  is 12 chunks and the tail is 6, using the unchanged `epoch_batch_plan` / `chunk_slices_v2`. It remains an
  execution candidate, not yet qualified.
- `load_train_rows` checks the split SHA256 before parsing, filters to TRAIN + CASIA/MSU at read, and reads only
  allowlisted columns. It opens no image and materializes no TEST row.

## Shared code and registry (additive only; verified hunk by hunk against the authority)

- `methods/common/config.py`: adds `LATER_FROZEN_METHOD_FILES = {E06b: …}`, anchored to this freeze record.
  `METHOD_FILES`, the M6B set and `M6B_CONFIG_FREEZE.json` are unchanged.
- `methods/common/learned.py`: the `checkpoint_plan` official-epoch branch accepts `E06b` alongside `E06c`.
- `third_party/registry.yaml`: the E06b entry now points `config_file` at the new config and adds `m6_status` fields.
  - The shared pin `16b793a7` is unchanged; no new pin is created.
  - The M0-era `implementation_status` stays `NOT_STARTED`, because a historical test pins it for every method (as
    for E06c).
- Unchanged: the E06c config, its snapshot, the A1 semantics, the adapter, graph, runner and microbatch modules, the
  M6B validator, and A9.

## State

No GPU use, model, training, checkpoint, image read, TEST row or bank. **M6_CLOSED = false. M7 HAS NOT STARTED.**

Next, M6F-C: GPU synthetic qualification of the native graph (lambda_pair = 5, K = 2), the 120-row tail, microbatch
memory and the torch worker-seed stream.
