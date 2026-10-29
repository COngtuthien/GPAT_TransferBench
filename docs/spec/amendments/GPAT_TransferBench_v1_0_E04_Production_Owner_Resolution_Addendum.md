# GPAT-TransferBench v1.0: E04 Physics-STD production owner resolution (D1–D9)

**Status:** `OWNER_APPROVED`, 2026-10-29. Machine-readable companion: `configs/amendments/e04_production_resolution.yaml`.

**Scope:** production preparation and qualification only. **No E04 scientific seed (42, 1337, 2026) is authorized**,
and this ruling creates no E04 scientific authority.

**Binding text:** the owner approved D1–D8 exactly as specified in the draft committed at
`4112d164861bffa9adbe042f97833bc8ebaa697a`
(`docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum_DRAFT.md`,
sha256 `1611122e…`; companion `configs/amendments/e04_production_resolution.DRAFT.yaml`, sha256 `72ddac61…`). D9 is
approved conditionally (below). The draft, its evidence (`outputs/audit/E04_OWNER_RESOLUTION_INVESTIGATION.*`) and the
RNG test vectors (`outputs/audit/E04_RNG_TEST_VECTORS.json`) are kept unchanged as the approved specification. Where
this document adds to the draft, the additions only restate the owner's approval message.

The frozen authority is unchanged: the E04 config `41861794…` (equal to its snapshot), A3 §4, A4, A2-06
`BASELINE_FINAL_STATE_V1`, M6D3d and M6D3e. Fidelity stays `CONTROLLED_ADAPTATION`.

| | Approved resolution |
|---|---|
| D1 | The official 3DDFA_V2 landmark crop (`parse_roi_box_from_landmark`, `crop_img`, `INTER_LINEAR` to 120, official normalisation, `recon_vers` with the float ROI). It uses the frozen M2 FaceXFormer `landmarks_px256`, transposed to `[2, 68]` float32: zero-based pixel centres, no offset correction. The image is read through the qualified OpenCV BGR path. There is no fallback crop, alternative tracker or redetection. **Disclosed controlled adaptation:** the landmarks are supplied externally rather than by 3DDFA's own tracking. |
| D2 | A TRAIN-only pre-pass. The 3830 live faces get a full fit, depth `M₀`, face mask and Q140. The 8838 spoof faces get a fit and Q140 correspondences only. Spoof depth is never rendered, and the spoof target stays exact float32 zeros. `attack_macro` is never read. No TEST geometry or image is accessed. |
| D3 | CPU only, in `gpat-m6-e04-geometry`: one thread, deterministic algorithms, TF32 off, as qualified. No GPU-dependent generation. |
| D4 | `pairs_train_v1` (8838 rows, stored order = ascending `pair_id`, verified). One legacy `RandomState(pair_seed)` draws `rng.permutation(8838)` per pass. The passes are concatenated with no drop, padding or deduplication, and each iteration takes 4 consecutive rows: 600000 draws = 67 passes + 7854 rows. A repeated row across a pass boundary is permitted. |
| D5 | No data-pipeline augmentation. The model's frequency decomposition, trace warping, hard-sample perturbation and inpainting synthesis are retained. No E03/STDN flips or brightness. |
| D6 | No VAL loop. VAL fields are explicit null with reason `NO_VAL_PROTOCOL_IN_SOURCE_OR_FROZEN_CONTRACT`. The final checkpoint is `BASELINE_FINAL_STATE_V1` at iteration 150000. No VAL or TEST selection. |
| D7 | A separate legacy `RandomState(hard_seed)`. Before Step 1 of each iteration it draws `uniform(0,1,(4,1,1,3))` → float32 scales, then `uniform(0,1,(4,1,1,1)) ≥ 0.5` → float32 keep flags, and the same arrays are reused for all three steps. The TF graph is unchanged. PAIR_ORDER and HARD_SAMPLE are separate RNG objects. Their derived numeric seeds are **not guaranteed distinct**, and correctness does not rely on that. |
| D8 | RGB uint8 → float32 / 255 once, in the feed, for live and spoof alike. No second normalisation. |
| D9 | **Approved conditionally (narrow scope):** a driver-version-only compatibility resolution for the E04 training environment `gpat-m6-e03` (lock 595.84, observed 580.178.04). Every other qualified invariant still applies. The conditions follow below. |

Seed derivation for D4 and D7: `1 + (u32_be(SHA256("GPAT_TRANSFERBENCH|E04|<DOMAIN>|<seed>")[0:4]) mod 2147483647)` with
`DOMAIN ∈ {PAIR_ORDER, HARD_SAMPLE}`. Expected values: 42 → 679869707 / 320493565; 1337 → 1349475396 / 35554372;
2026 → 692890966 / 1053992267. The complete stream hashes are in `E04_RNG_TEST_VECTORS.json`. Qualification seed 314159
reproduces the M6D3e hard-sample fixture hashes.

## D9 conditions

1. Only the NVIDIA driver version field may differ. GPU model, UUID and every other lock field must be exact.
2. The resolution covers only E04 training in `gpat-m6-e03`. The geometry pre-pass stays CPU-only; its outputs
   independently matched their previously committed hashes.
3. It does not extend to any different TensorFlow, CUDA, cuDNN, graph, optimizer, input or training semantics.
4. No bitwise equivalence between the historical and current driver environments is claimed.
5. The four D3 BN `moving_mean` tensors whose recorded summaries differ by at most two float32 ulps are a documented
   runtime observation. They are not proof that every future long-run difference is irrelevant. The exact GPU
   kernel-level source remains unidentified.
6. Production qualification repeats the compatibility checks. Any new material numerical, state-transition, gradient
   or graph discrepancy fails closed.
