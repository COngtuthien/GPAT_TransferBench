# M8 deterministic smoke test: authority assessment and owner decision package

Date: 2026-11-22. Branch `m6-baselines`, starting at `fa239adb9df6913c866cfeae05849aa6567f6b6d` (= origin).
Machine-readable record: `M8_SMOKE_TEST_AUTHORITY_ASSESSMENT.json`. Owner questions:
`configs/amendments/m8_smoke_test_contract.DRAFT.yaml`.

**Status: `M8_SMOKE_TEST_BLOCKED_BY_OWNER_DECISION`.**

- No smoke stage was executed, no real image was read and no bank was generated.
- No TEST row, file or metadata aggregate was used.
- E04 was untouched.

## 1. Frozen requirements

| Source | Requirement |
|---|---|
| Spec §26 | "A full deterministic smoke test on a 2-subject-per-class toy subset must finish before full runs." |
| Spec §32 | "smoke test passes for generator → bank → evaluator → metrics → plots." |
| Spec §25 | M8 Banks: "all synthetic banks"; M9 generator eval T04/T05 and plots; M10 ResNet-18 downstream T06 |
| M6D6jR O3 | `smoke_test_required: true`, `smoke_test_completed: false`, not waived. Hard gate before the first FULL M8+ bank / evaluator / metrics / plots execution. `must_exercise` = generator, bank, evaluator, metrics, plots. No scientific result may come from it |
| E01/E02 ruling Q1 (`656cc8a`) | Production waits for a committed, hash-verified smoke-test record bound in `smoke_test_record` |

**Scope finding.** Under the current authority, the smoke test is one global end-to-end gate. It is not an E01/E02-specific
test. The following are not equivalent to it:

- the M6 synthetic writer qualification (`6f15f8a`), which has no real data and no evaluator, metrics or plots;
- M6D6j, which O3 explicitly rules out.

## 2. What exists and what does not

| Stage | State |
|---|---|
| Generator → bank | E01/E02 only (`methods/common/bank_writer.py`, qualified). There is no bank writer for E03–E06c or GPAT |
| Evaluator | **Missing.** There is no downstream ResNet-18/DINOv3 code (`downstream/` holds only `.gitkeep`) and no T04/T05 generator-eval code. `gpatbench/cli.py` has no `generate-bank` / `evaluate-generator` / `train-downstream` command |
| Metrics | **Missing** for M9/M10. Only the ArtifactProbeNet macro-F1 exists |
| Plots | **Missing** (`outputs/plots/` holds only `.gitkeep`) |
| Evaluator environment | **Not created** (env-core in `PLANNED_ENVIRONMENT_REQUIREMENTS.md`; ResNet-18 weights missing) |
| Smoke contract | **Missing.** Nothing defines the toy subset, selection rule, budget or acceptance gates |

The evaluator stage is a smoke-scale run of the frozen downstream protocol (`downstream_resnet18.yaml`: training, VAL
checkpoint and threshold, a single application to TEST). It needs new code, a new environment (package installation) and
downstream detector training. **All three are outside this task's authorization.** The definition of the stage is also not
fixed (S5, S8).

## 3. Why the toy subset cannot be selected without the owner

These facts come from manifest metadata, TRAIN and VAL rows only.

- **"Class" is undefined.** It could be the binary label (T01 counts "binary class") or attack_macro (S2).
- **SiW-Mv2 has no subjects.** `subject_id_global` is null for all 9,507 TRAIN and 2,033 VAL SiW-Mv2 rows. The split is
  video-disjoint by `content_group_id` (DEV-012, "never described as subject-disjoint"), so "2 subjects" needs an owner
  proxy (S3).
- **Unit counts:**

  | Split | casia_fasd subjects | msu_mfsd subjects | siwmv2 content groups (live / spoof) |
  |---|---|---|---|
  | TRAIN | 35 | 25 | 549 / 641 |
  | VAL | 8 | 5 | 118 / 131 |

  Every CASIA/MSU subject has both live and spoof samples.
- **Pairs depend on the chosen subjects.** Frozen `pairs_train_v1` pairs use different target and source subjects, and
  whether any lie inside the toy subjects depends on which are chosen:
  - the first 2 subject ids contain 0 pairs (CASIA and MSU);
  - the best 2-subject sets contain at most 24 (CASIA) and 38 (MSU) pairs.

  The bank-stage pair population therefore needs a rule (S6). The recommended rule keeps the frozen pairs and their global
  positions: target in a toy unit, which gives 93 CASIA and 172 MSU pairs for the first-2 illustration. E01 operators then
  stay at global position % 8, and the smoke outputs equal the corresponding future production outputs, which gives a
  later cross-check.
- **TEST.** The smoke would apply a threshold to TEST, which conflicts with "TEST untouched before checkpoint/threshold
  freeze" unless the owner rules on it. The recommendation is a VAL-unit stand-in (S5).

## 4. E01/E02 contract (re-verified, unchanged)

The laptop `.venv` results:

- `test_e01_e02_*` 40/40 and `test_m6c1_*` 83/83 pass;
- the frozen configs equal their snapshots: E01 `87012ea7…`, E02 `fdf74776…`;
- the ruling is `4a89b4e1…`;
- the production gate refuses with `E01_E02_BLOCKED_BY_M8_SMOKE_GATE`.

The qualified semantics are those recorded at `6f15f8a`/`fa239ad`.

**E01:**
- 8 official operators in the pinned order;
- operator = position % 8;
- per-pair seed `SHA256(pair_id + str(seed)) mod 2^31`;
- NUM_MAG 10;
- level 0 is an unchanged copy, which is disclosed;
- Hand_Trembling varies across hosts by ≤ 1 LSB.

**E02:**
- per-channel float64 FFT and shift;
- 16×16 blocks;
- band 0.20–0.50 with ≥ 75 % coverage, which gives n_eligible 159 and k 40;
- the source magnitude replaces the target in the selected blocks and their pixel-wise conjugates, and the target phase is
  kept;
- real part → clip [0,255] → `numpy.rint` → uint8;
- the edit mask is Hermitian-closed;
- imaginary residual ≤ 1e-8.

Neither algorithm was changed.

**Correction to a prior record.** `E01_E02_M6_WRITER_READINESS_CLOSURE.md` (`fa239ad`) states that `--check-only` performs
the first real-face read. That statement is wrong: `--check-only` reads manifests and checks only that the faces root is a
directory, and it opens no face file. The historical record is left unchanged. A face-bytes preflight is proposed below.

## 5. Owner decisions required (DRAFT S1–S10)

| # | Question | Recommended |
|---|---|---|
| S1 | Global end-to-end gate, or a staged amendment where a per-method bank-stage smoke unlocks full M8 banks while the evaluator/metrics/plots smoke still gates M9+ | **A_GLOBAL_END_TO_END** (literal). B is the only route to E01/E02 banks before the evaluator exists, and it is an explicit O3 amendment the owner would have to make |
| S2 | What "class" means | BINARY_LABEL_PER_DATASET |
| S3 | SiW-Mv2 subject unit | CONTENT_GROUP_ID (disclosed proxy) |
| S4 | Selection rule | HASH_RANK `SHA256('gpatbench.smoke.v1\|dataset\|split\|class\|unit')`, first 2 |
| S5 | Splits and TEST | TRAIN_VAL_ONLY_TEST_PROXY (no TEST) |
| S6 | Bank pair population | FROZEN_PAIRS_TARGET_IN_TOY |
| S7 | Methods covered | ALL_METHODS_WITH_BANK_WRITERS (today E01/E02) |
| S8 | Evaluator budget, device and environment | owner choice. Implementing M9/M10 code and installing env-core both need explicit authorization |
| S9 | Determinism | bitwise banks; evaluator bitwise on CPU or procedural on GPU (E04 D12 precedent) |
| S10 | Seeds | banks 42/1337/2026; evaluator 42 |

## 6. Production binding requirements (not active)

**Current state.** `smoke_test_record` in `configs/amendments/e01_e02_bank_milestone_ruling.yaml` is `null`. No passing
record exists, and none was prepopulated.

**What the reader accepts.** `load_production_authorization` accepts a smoke record only if:

- the file is `.json` at `<repo>/<path>`;
- its SHA256 equals the bound `sha256`;
- it has top-level `smoke_test_passed: true`.

**Sequence to unlock E01/E02 production:**

1. The owner approves the smoke contract (`m8_smoke_test_contract.yaml`, separate commit). Under S1-A this includes
   authorizing evaluator implementation and environment creation.
2. Implement the missing stages, then run the smoke test and independently re-derive it. Commit
   `outputs/audit/M8_SMOKE_TEST_RECORD.json` with `smoke_test_passed: true`, all stage gates, commands and exit codes,
   hashes, environment and TEST-firewall evidence.
3. **A new owner approval**, recorded additively with an addendum, sets
   `smoke_test_record: {path: outputs/audit/M8_SMOKE_TEST_RECORD.json, sha256: <its SHA256>}` and names the production
   authority commit. A companion record states that O3 `smoke_test_completed` is now true; the M6D6jR file stays unchanged.
4. Recommended gate hardening, done in the same approval cycle: the reader also requires that the record names E01/E02
   with a passing bank stage and the approved contract SHA256. Today it checks only the file hash and the flag.

**Production launch checks** (Q4), once bound:

- run on the GPU host in an isolated clean clone at the authority commit (never the E04 checkout), on CPU, with
  nice/ionice;
- `verify_locked_environment` must pass (`gpat-m5`);
- `--check-only` for each method and seed;
- **new:** a face-bytes preflight (read and SHA256 all 12,668 referenced M2 faces, no decode);
- no existing run directory;
- after the run: file verification, full replay, and a cross-check of smoke-subset outputs against the production bytes;
- an M8 bank completion audit.

## 7. E04 isolation

The only access was a read-only `pgrep`, `tmux ls`, `git rev-parse`/`status`, one `tail` line and `nvidia-smi` query.
Results at 2026-11-21T23:46Z:

- E04 seed 2026 is running: PID 435565, `tools/run_e04.py --seed 2026`, tmux `e04_physics_std_seed2026`;
- global_step 11646, no exit marker;
- checkout `18ba00e`, clean.

Nothing was started on the host.
