# E01 FAS-Aug / E02 Frequency Substitution: bank contract and M6/M8 milestone assessment

Date: 2026-11-22. Branch `m6-baselines`. Authority: laptop HEAD = origin = `1630d6a14af361f8975083cbc59ab967a73babab`.
GPU checkout `18ba00e94813f005876c8d6235b0b848226b2c1d` (pinned for E04, not moved).

**Status: `E01_E02_BLOCKED_BY_MILESTONE_RULING`.** No real bank output was generated. No E04, GPAT, TEST,
downstream or method-status change. Owner questions: `configs/amendments/e01_e02_bank_milestone_ruling.DRAFT.yaml`.

## 1. Live state at assessment time

- E04 Physics-STD seed 2026 is running: tmux `e04_physics_std_seed2026`, PID 435565 (`tools/run_e04.py --seed 2026`),
  run id `e7100eb9a5ab0537`, run root `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E04/seed_2026`,
  first iteration 2026-11-21T17:19:07Z, global_step 426 of 150000 at 00:33 +07, 4.4 GiB GPU. No exit marker.
- The 3090 is shared with three unrelated workloads that are not part of this project: a DDPM trainer, a BERT4Rec run and g_pilot.
  GPU use is 99 %, 17 GiB in use. Host: 20 cores, load average 3.8, 17 GiB RAM available.
- Contact with the host was read-only: `ps`, `tmux ls`, `nvidia-smi`, `ls` and a `tail` of `metrics.jsonl`. No attach, signal,
  instrumentation, checkout move, fetch or write.
- The laptop has no runtime root: `/media/cong/Data` is not mounted. Real `faces_256` images exist only on the GPU host.

## 2. Authority read

| Authority | SHA256 |
|---|---|
| Frozen spec docx (= `spec_sha256` in both configs) | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| `configs/methods/e01_fas_aug.yaml` (snapshot byte-identical) | `87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0` |
| `configs/methods/e02_freqsub.yaml` (snapshot byte-identical) | `fdf74776cc100d5b5c1ed5f58b84ebb0dfd534782fbb7fc34382919d38d67bf0` |
| Amendment A2 | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |
| M6D6jR decision (O3) md / yaml | `4707d135…cfae2c` / `2f7e5460…dca830` |
| Roadmap JSON (BASELINE-EXEC-P0) | `e30818e75d37fa54c63588c871f1d492448863f690ccfe12b378818846240e5d` |
| `method_status.csv` | `fd97e7cbac35af1ef02fd495d21ad3cfcebffb47702907ab01afd2d447003723` |
| `manifests/pairs_train_v1.parquet` (= config) | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| `manifests/split_v1.parquet` (= config) | `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d` |
| `manifests/m2_sample_accounting.parquet` (= ARTIFACT_INDEX) | `65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662` |
| `environments/m6_core_gpu.lock.json` (M6D1 E01/E02 lock) | `53d4910a17c541ad7ac57a1ac410c0fef8c2d1c63399b423c97e490817b335eb` |

Other sources read: spec §6–8.2, §25–27, §32 and Appendix A; M6_BASELINE_EXECUTION_CONTRACT_AUDIT; M6A2 resolution;
M6C1 and M6D1 records; PREVIOUS_WORK_RESULTS_AUDIT; `methods/fas_aug/{e01,official}.py`; `methods/freq_sub/e02.py`;
and `methods/common/{config,runlog,interface}.py`.

## 3. E01 FAS-Aug contract (spec §8.1; A2-01, A2-05; M6B config)

1. **Algorithm.** Official `RizhaoCai/FAS-Aug@0da1dd79…` operator bodies are loaded unchanged by `OfficialFASAugBackend`.
   There are 8 operators in the frozen order: Color_Diversity, Color_Distortion, Reflection, BN_Halftone, Moire_Pattern,
   SFC_Halftone, Hand_Trembling and Low_Resolution. Spec §8.1 overrides the official random choice: one operator per pair by
   **deterministic round-robin by pair_id**.
2. **Input.** The target live of the pair row, canonical 256×256 RGB uint8. The source spoof is not used.
3. **Fixed assets.** Pinned repository assets: 90 backgrounds, 48 noise textures, 190 moiré textures, 11 RGB and 7 CMYK ICC
   profiles. Each directory is enumerated in bytewise lexicographic order (A2-01). There are no weights.
4. **Randomness.** `operator_seed = int.from_bytes(SHA256(UTF-8(pair_id + str(seed))), "big") mod 2^31`. Check value:
   PTR000001/42 gives 795981663. The seed drives a Python `random.Random` instance. The magnitude level is
   `k = rng.randrange(10)`, with `level = k/9` and `mag = level·(hi−lo)+lo`. The official operators then make their own
   choice and crop draws, which are recorded.
5. **Output.** One spoof-labelled 256×256 RGB uint8 image per pair.
6. **Population.** All 8838 rows of `pairs_train_v1`, all TRAIN, with 3830 distinct targets.
7. **Count and pairing.** N_syn = 8838 per seed. Seeds are 42, 1337 and 2026, and every seed is primary evidence.
8. **Provenance.** Each output records the pair_id, the target_live_id and its M2 face PNG SHA256, the operator, the level,
   the magnitude, all sampled parameters, the asset filename and the output SHA256.
9. **Frozen parameters.** The ranges in the config are copied from the official code. NUM_MAG = 10. Level 0 is retained: it
   gives a byte-identical copy, about 10 % of pairs, and must be disclosed. The changeLabel divergence is disclosed.
10. **Fidelity.** `FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION`.
11. **TEST.** The TEST split is forbidden and has no code path.
12. **VAL.** VAL is used for diagnostics only. No selection exists.
13. **Output selection.** None. RAW-MATCHED keeps every valid output (spec §7).
14. **Runtime.** `gpat-m5` (M6D1 `QUALIFIED_EXECUTION_ENVIRONMENT_FOR_E01_E02_ONLY`), on CPU.
15. **Evidence.** Per-sample records as listed in the config's `logging` and `synthetic_bank_accounting`. The bank manifest
    is `<runtime_root>/runs/m6/E01/seed_<seed>/bank_manifest.parquet`.
16. **Qualification.** The official backend passed the synthetic smoke (M6C1) and the env qualification (M6D1). The
    real-TRAIN bank path is `NOT_QUALIFIED`.

**Spec-silent detail E01-RR (unresolved, affects output bytes).** "Round-robin by pair_id" does not fix the ordinal. The
manifest's pair_ids are exactly `PTR000001…PTR008838` in stored (ascending) order. The 0-based manifest position and the
numeric pair_id therefore differ by one, which shifts **every** pair's operator. The `E01Generator` default is the numeric
id. M6C1 recorded this as a known limitation: "pass canonical ordinal explicitly when consuming the frozen pair manifest".
No authority picks one. This is DRAFT question Q2.

## 4. E02 Frequency Substitution contract (spec §8.2; A2-02/03/04; M6B config)

1. **Algorithm.** Per-channel float64 `fft2` of target and source at 256×256, then `fftshift`. The target magnitude is
   replaced by the source magnitude in the selected blocks and in their pixel-wise conjugate counterparts
   (`p → (256−p) mod 256`; 0 and 128 are self-conjugate; a block-level mirror is forbidden). The target phase is kept.
   Then `A_mix·exp(jφ_t)`, `ifftshift`, `ifft2`, real part, clip to [0,255]. There is no spatial post-filter.
2. **Inputs.** The pair's target live **and** source spoof, both canonical 256×256 RGB uint8.
3. **Assets.** None.
4. **Randomness.** `pair_seed = int.from_bytes(SHA256("gpatbench.freqsub.block.v1|"+pair_id+"|"+str(seed)), "big")`, the
   full 256 bits. This seeds `Generator(PCG64(pair_seed)).choice(159, 40, replace=False)`, and the indices are sorted
   ascending. The NumPy version must be pinned at runtime (A2-03).
5. **Geometry.** 16×16 blocks. The eligible band is 0.20 ≤ r ≤ 0.50 and a block needs at least 75 % of its pixels in it.
   This gives n_eligible = 159 and k = (159+2)//4 = 40.
6. **Output.** One 256×256 RGB uint8 image per pair. Spec §8.2 also asks for the block mask to be saved as the edit map.
7. **Population, count, seeds and TEST.** Same as E01. Both target and source are TRAIN (12,668 distinct samples).
8. **Fidelity.** `SPEC_DEFINED` (the spec §8.2 tag is CONTROLLED_ADAPTATION; the gate is adapted).
9. **Numeric invariant.** The inverse-FFT imaginary residual must be ≤ 1e-8. The M6D1 maximum was 5.3e-12.
10. **Runtime and qualification.** Same as E01: gpat-m5 with NumPy 2.4.6. Synthetic qualification only.

**Spec-silent detail E02-Q (disclosure).** Neither the spec nor the config fixes the float-to-uint8 rounding. M6C1 implemented
clip → `numpy.rint` (round half to even) → uint8, and M6D1 qualified it. The writer does not change this. It is listed for
owner confirmation as DRAFT Q3.

## 5. M6-versus-M8 boundary

The conflicting requirements, quoted:

| # | Source | Requirement |
|---|---|---|
| 1 | Spec §25 | "M6 Baselines — method adapters + method_status.csv"; "M8 Banks — **all synthetic banks** — same intended budget; provenance complete; failure counts saved" |
| 2 | Spec §26 / §32 | "A full deterministic smoke test on a 2-subject-per-class toy subset must finish before full runs"; "smoke test passes for generator → bank → evaluator → metrics → plots" |
| 3 | M6D6jR O3 (owner) | "M6 and M7 generator scientific **training** MAY proceed"; the smoke test "is a mandatory HARD GATE before the first FULL M8+ end-to-end **bank** / evaluator / metrics / plots execution"; `smoke_test_completed = false` |
| 4 | E01/E02 M6B configs | `run_layout.root_template: <runtime_root>/runs/m6/E0x/seed_<seed>/`; `bank_manifest` under that root; `synthetic_bank_accounting.generated_in_m6b: false`; `training: NOT_APPLICABLE_NON_LEARNED_METHOD`. `field_provenance` tags run_layout and bank accounting as M6B logistics, not SPEC |
| 5 | Spec Appendix A | `train_method(...)` then `bank = generate_fixed_budget(method, pairs_train, N_syn=…)`. For E01/E02 the first call is empty |
| 6 | Roadmap P0 | E01/E02 readiness `NEEDS_BANK_PATH_IMPLEMENTATION_AND_OWNER_GATE_CLARIFICATION`; blocker "OWNER_CLARIFICATION: whether the E01 per-seed bank is M6 method execution or an M8 bank that waits for the deterministic smoke test" |

Analysis:

- E01 and E02 have no training. The only thing they execute is `generate_fixed_budget`, which yields exactly the 8838-image
  bank that §25 assigns to M8.
- O3 exempts generator *training* only, and it names "bank" among the gated stages.
- The `runs/m6/...` path in the M6B configs is a storage location. The same config says `generated_in_m6b: false`, so the path
  does not authorize generation.
- On the most direct reading, the full banks are M8 work and the incomplete smoke test blocks them. No owner record states
  this for non-learned methods, though, and the roadmap left it explicitly open.

The frozen authority therefore does **not** uniquely resolve the boundary. **`E01_E02_BLOCKED_BY_MILESTONE_RULING`.**

**Recommended owner ruling (DRAFT Q1 option A):**

- E01/E02 full banks are M8 bank work and remain behind the O3 smoke hard gate.
- M6 closes E01/E02 at "production writer qualified".
- When the smoke test passes, the banks are written at the M6B run-layout paths with the writer qualified here.

The alternative, option B, extends O3 so that the non-learned bank counts as M6 execution. Because O3 lists "bank" as gated,
option B needs an explicit amendment. In both options the smoke test still gates every downstream use.

**Real production execution is not authorized** by existing authority. Q2 (E01 ordinal) must also be answered before E01 can
produce a bank under either option. Q3 and Q4 are confirmations.

## 6. What remains permissible now

The following remain permissible and are carried out in later commits on this branch:

- writer implementation;
- synthetic-fixture qualification (the M6C1/M6D1 precedent, using in-memory or temporary synthetic images);
- metadata-only input-integrity checks over the frozen manifests.

The writer refuses production unless an approved `configs/amendments/e01_e02_bank_milestone_ruling.yaml` exists.
