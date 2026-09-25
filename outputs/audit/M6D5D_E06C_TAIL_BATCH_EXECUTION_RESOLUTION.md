# M6D5d — E06c DSDG-BIN-IDFREE: variable tail-batch global-statistic microbatch execution resolution

**M6D5d PASS.** Execution mode: **GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2**
(classification CONTROLLED_EXECUTION_ADAPTATION, owner-approved, additive to V1).

| Status | |
|---|---|
| **E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED** | retained from M6D5c; V1 = V2 at B=240 re-verified |
| **E06c_GLOBAL_STATISTIC_MICROBATCH_QUALIFIED** | retained from M6D5c |
| **E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED** | this milestone |
| **E06c_EPOCH_BATCHING_QUALIFIED** | 36 × 240 + 198; 37 optimizer steps per epoch |
| **E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED** | M6D5b historical truth, unchanged |
| **E06c_PRODUCTION_RUNNER_NOT_YET_QUALIFIED** | |
| **E06c_FULL_TRAINING_NOT_YET_EXECUTED** | |
| **E06c REMAINS IMPLEMENTED_NOT_EXECUTED** | |
| **CONTROLLED_ADAPTATION PRESERVED** | Amendment A1, DEV-020 |

Reporting label: DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation). This path is not native, faithful or
official DSDG. Every optimizer step here is a synthetic qualification step; none of it is scientific training.

## Authority and GPU synchronization

Laptop HEAD = origin/m6-baselines = `666aba4803468c1d2888e347f8b09afa2ca47401` ("M6D5c: resolve E06c global batch 240
execution"). Divergence is 0 0 and the worktree was clean. `AGENTS.md` was checked with `test -f`: absent. The ledger had 113 rows and the index 604 rows. The committed
M6D5c files are byte-identical to the M6D5c evidence (`microbatch_execution.py` `67f2ae80…`, `microbatch_qualification.py`
`f85972bf…`). The GPU started at `0c62dbe`, clean and at 0 0. We fetched only `refs/heads/m6-baselines`, proved ancestry (distance 1) and ran `git merge --ff-only`,
which reached `666aba4` (0 0, clean). No reset, clean, stash, commit or push was used. The new files were staged on the GPU temporarily and removed
afterwards.

## Upstream DataLoader evidence (drop_last)

`third_party/source_cache/facexzoo` sits at `16b793a7564a4b9308cf94e62bdb2ffacb3a725a` (tree `0d2216bd…`), and
`train_generator.py` SHA256 is `5b7bf426…b370`, matching the lock. Lines 94–97 match verbatim:

    train_loader = torch.utils.data.DataLoader(
        GenDataset_s(img_root=args.img_root, list_file=args.train_list, attack_type=args.attack_type),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

This is the only `DataLoader(` in the file, and the string `drop_last` does not occur anywhere in it (`verify_pinned_dataloader` → `[]`, re-checked
in every GPU process). In the qualified runtime, `inspect.signature(torch.utils.data.DataLoader.__init__)` gives
`drop_last=False` as the default (recorded in all 4 processes). **Source-consistent final partial batch retained.** No `drop_last=True` was introduced.

## TRAIN row-count provenance

We used retained evidence only. No parquet file was parsed.
- `configs/frozen/dsdg_bin_idfree_v1.yaml` (A1, SHA-verified by the adapter): `training_relation.rows: 8838`,
  `authoritative_manifest_sha256: a5e4fdae…c243`, per dataset CASIA 2520 + MSU 1200 + SiW 5118 = 8838.
- `outputs/audit/pairs_train_v1.sha256` (M4 sidecar): `rows: 8838`, with the same SHA256 (`a5e4fdae…c243`).
- `outputs/audit/M4_COMMON_PAIR_AUDIT.json`: `expected_rows.TRAIN = 8838`, status PASS.

**Exactly what was opened in M6D5d.** On the laptop we read the three retained JSON/YAML files above. We ran one `ls manifests/`, which lists
manifest file names only, not benchmark sample files. We ran one `sha256sum manifests/pairs_train_v1.parquet`, which streams the raw
bytes through a hash without parsing them. It gave `a5e4fdae…c243`, equal to the frozen value. We also read the first 40 lines of
`manifests/pair_train_stats_v1.json` (TRAIN pose statistics per dataset; no sample rows) and ran `grep -l pairs_train_v1` over the
repository's `outputs/audit/*.json|*.md`. The row count was not taken from these. No other manifest, no VAL/TEST row, no image and no
benchmark data root was opened. The GPU processes opened no manifest (firewall: 0 denials; the audit hook denies `manifests/`
and `.parquet`).

## Epoch batch plan

`epoch_batch_plan(8838, 240, drop_last=False)`: `divmod(8838, 240) = (36, 198)`. The batch sizes are `[240]×36 + [198]`,
which covers 8838 rows, so the plan has **37 optimizer steps per epoch**. The final batch's chunks are `[20]×9 + [18]`. For 200 epochs the *planned*
count is 37 × 200 = 7400 optimizer steps. That is algebra only; nothing was executed.

## V1 ↔ V2 compatibility at B = 240 (process `v1compat`) — PASS, bitwise

Both paths used the same models (initial bytes restored and SHA-verified before each path), inputs equal to the M6D5b/M6D5c B=240
inputs, one shared epsilon draw [240,128], a fresh Adam of the same construction for each path, `set_to_none=False` and FP32. V1 ran
`microbatch_execution.run_global_batch`, and V2 ran `run_global_batch_v2`.

| Compared before the optimizer update | V1 | V2 |
|---|---|---|
| chunk inventory | 12 × 20 | 12 × 20 |
| backward calls | 12 | 12 |
| epoch-1 total | 14532.477670667766 | identical |
| all 7 global losses (relative difference) | | 0.0 |
| delta [128] | | bitwise equal (max abs diff 0.0) |
| ort_mean / ort sign | 0.53508872 / +1 | identical |
| MMD sign vector | | equal |
| gradients (45,370,182 owned) | | cosine 1.0, relative-L2 0.0, **bitwise equal** |
| gradient inventory | 55/55 owned non-None | identical |
| post-step parameters (reported, not gated) | | bitwise equal |

All 9 frozen compatibility gates pass. Accounting: 2 optimizer constructions, 2 applications, 24 backward calls.

## Remainder reference B = 7, chunks 3,3,1 (process `reference7`) — PASS

Path A is the **unchanged M6D5b full-batch graph** at B=7. Path B is V2 with max microbatch 3, giving chunks `[3,3,1]`. Both used identical initial
bytes (SHA-verified restore), identical inputs (14 distinct rows) and identical pre-drawn epsilon [7,128]. The epsilon draw is bitwise equal to the pinned
`reparameterize` draw.

| Gate | Threshold | Observed |
|---|---|---|
| scalar losses finite | all | all finite |
| epoch-1 total relative error | ≤ 1e-4 | 4.22e-8 (13920.35254 vs 13920.35313) |
| aggregate gradient cosine | ≥ 0.999 | 0.99999987 |
| aggregate gradient relative-L2 | ≤ 1e-2 | 5.08e-4 |
| owned gradient coverage | 55/55 | 55/55 both paths |

Per-term relative differences: rec 1.0e-8, kl 1.4e-8, mmd 0, ip 3.5e-7, ort 9.1e-8, cls 0, pair 0. The Adam update cosine is 0.99966
(reported, not gated; the first-step `lr·sign(g)` effect explained in M6D5c). The remainder comes from FP32 batch-shape kernel effects. On CPU in float64,
the new tests give gradient equality to 1e-9 for B/microbatch = 7/3 (3,3,1), 11/4 (4,4,3) and 5/20 (5), at epochs 1 and 2.
Formulas were not tuned. Accounting: 2 constructions, 2 applications, 4 backward calls.

## B = 198 final partial global batch: two fresh processes (seed 60504) — PASS

Inputs: `training_graph.synthetic_pair(198)` gives `x_spoof` and `x_live` of shape [198,3,256,256], FP32, in [0,1]. **All 396 rows are distinct.** `label_spoof` is all 0. No file,
decoder or manifest was used. Epsilon: `[198,128]` ×3, drawn once in the order cls → nir → vis, bitwise equal to the pinned draws, and never redrawn
(the RNG state is unchanged from after the draw through the step). Models, LightCNN (60/60 matched) and Adam (45,370,182 parameters / 55 tensors, 1 group,
lr 2e-4, no netCls/netIP) are the same as in M6D5c.

**Chunks:** `[20, 20, 20, 20, 20, 20, 20, 20, 20, 18]`. The local-mean weights are 20/198 = 0.10101 (×9) and 18/198 = 0.09091, summing to 1.
There was 1 `zero_grad(set_to_none=False)`, with 0 backward calls before it. There were **10 backward calls** and **1 optimizer application**.

| Global value (epoch-1 semantics) | Process 1 | Process 2 |
|---|---:|---:|
| loss_rec | 14501.305368134 | identical |
| loss_kl | 18.525230639 | identical |
| loss_mmd (pass-1 FP32; float64 recomputation 16.528747155) | 16.528745651 | identical |
| loss_ip | 6.356372568 | identical |
| loss_pair (λ=0) / loss_cls (one logit) | 0.0 / 0.0 | identical |
| loss_ort (ort_mean −0.140358895, sign −1) | 0.140358895 | identical |
| **epoch-1 total** | **14501.720875212** | identical |
| sum of the 10 executed chunk objectives | 14501.720703 | identical |
| post-warmup total (**algebraic only, not executed**) | 14542.856075888 | identical |

The delta has 65 positive, 63 negative and 0 zero coordinates. The surrogate sums match the global values: MMD 16.5287461 against 16.5287457, orthogonality 0.14035897 against
0.14035890. The pass-2 latents were bitwise equal to pass 1, with 0 sign mismatches. The naive chunk average (diagnostic, not used) would have been
MMD 21.27 and orthogonality 3.67. Because the orthogonality sign is negative here, the tail batch also exercises the negative-sign surrogate.

**Gradients:** netE_nir 17/17 (13,790,048 nonzero), netE_vis 17/17 (11,692,640), netG 21/21 (19,887,494). All are finite: **55/55**.
netCls has 2 tensors with zero gradient, as expected from the one-logit CE, and `pre_spoof` had a zero gradient in all 10 chunks, including the [18,1] tail chunk. netIP had no gradients. In every
chunk, the LightCNN target features were `requires_grad=False` and the reconstruction features kept their graph, with a nonzero gradient at `rec_nir`/`rec_vis`.
**Adam:** 55 state entries, step 1.0, owned parameters only, finite moments. **Parameters:** all 55 owned tensors changed (max |Δ| = 2.0000e-4 = lr).
netCls and netIP had 0 changed elements and unchanged bytes. netIP stayed frozen and in eval mode.

### CUDA memory (B=198, process 1; process 2 identical)

| Phase | Allocated | Peak allocated (window) | Peak reserved |
|---|---:|---:|---:|
| after model construction + LightCNN | 229,401,088 | 353,616,896 | 367,001,600 |
| after optimizer + B=198 inputs | 540,830,208 | 540,830,208 | 681,574,400 |
| after epsilon [198,128]×3 | 541,134,336 | 541,743,104 | 681,574,400 |
| pass-1 peak | 574,995,968 | 669,358,592 | 723,517,440 |
| pass-2 chunks 00–08 (m=20) | ≈797.7–797.9 M | 5,726,179,328 – 5,947,622,400 | ≤ 7,163,871,232 |
| pass-2 chunk 09 (m=18) | 797,939,200 | 5,566,324,736 | 7,163,871,232 |
| after optimizer step | 1,169,028,608 | 1,351,951,872 | 7,165,968,384 |
| **overall peak** | | **5,947,622,400 (5.54 GiB)** | **7,165,968,384 (6.67 GiB)** |

Wall time: pass 1 took 0.20 s and pass 2 (10 chunks) 3.25 s.

### Repeatability

Bitwise equality was not required, but it is what we observed. Between the two B=198 processes, the JSON differs only in `started_utc`/`ended_utc`, PIDs and
`phase_seconds`. Epsilon, delta, signs, all losses, 55/55 gradient SHA256, 55/55 Adam `exp_avg` SHA256, all 117 post-step parameter SHA256
and every memory figure are identical.

## Environment

The environment was `gpat-m6-e06c`, lock `91416a20…4e95`, verified in-process. All 14 non-launch identity fields matched: torch 2.12.1+cu130, CUDA 13.0,
cuDNN 92000, driver 595.84. It was identical before and after every process. FP32, TF32 off (plus `NVIDIA_TF32_OVERRIDE=0`), autocast off, matmul precision
`highest`, cuDNN deterministic. Source identity and the 46 pinned statements matched. LightCNN was `d0750746…9964`, the same before and after. The M6D5c V1 overlay
guard still passes, and the adapter's generic accumulation refusal is still in place. `compatibility_patch = NONE`. The GPU was clean before every
process (254 MiB, no compute processes) and read 254 MiB after each one. The only warning was the known pinned `reparameterize` deprecation.

## Process accounting (qualification only; none of this is training)

| Process | Global batch | Optimizer constructions | Applications | Backward calls |
|---|---|---:|---:|---:|
| reference7 | 7 (3,3,1), ×2 paths | 2 | 2 | 4 |
| v1compat | 240 (12×20), ×2 paths | 2 | 2 | 24 |
| B=198 process 1 | 198 (9×20+18) | 1 | 1 | 10 |
| B=198 process 2 | 198 (9×20+18) | 1 | 1 | 10 |
| **total** | | **6** | **6** | **48** |

B=198 qualification alone accounts for 2 optimizer applications and 20 backward calls. There were no autograd.grad calls, no torch.save calls and no activation checkpoints.

## Tests, preflight, ledger, index

The new file is `tests/test_m6d5d_e06c_tail_batch_execution.py`, with 10 static contract tests, 6 Torch formula tests (float64 CPU toy nets against the
unchanged M6D5b transcription) and 3 retained-evidence tests. Counts are in the audit JSON and the runtime log. The static preflight
`tools/m6d5d_e06c_tail_batch_preflight.py` imports no Torch, uses no CUDA and constructs no model, optimizer or image data. Ledger: **113 → 114**, with one
`M6D5D_E06C_TAIL_BATCH_EXECUTION_RESOLUTION` row; the first 113 rows are byte-identical. The artifact index was rebuilt last, with CRLF line endings.

nominal_global_batch = 240 · final_global_batch = 198 · max_microbatch = 20 · final_chunk = 18 · drop_last = false ·
benchmark_training = false · VAL_access = false · TEST_access = false · scientific_checkpoint = false.

**NO BENCHMARK TRAINING. NO VAL ACCESS. NO TEST ACCESS. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
