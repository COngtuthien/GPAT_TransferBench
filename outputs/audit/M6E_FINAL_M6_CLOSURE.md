# M6E — Final M6 baseline closure

**M6_CLOSED = true**, at the implementation/qualification level that the frozen M6 gate defines.
- `baseline_full_scientific_execution_complete = false`
- `scientific_baseline_training_required_for_M6_gate = false`
- No method has completed a scientific seed. No scientific checkpoint, bank or evaluation exists or was created.
- **M7 HAS NOT STARTED.**

| Field | Value |
|---|---|
| Authority | `57c6a99035c2d45f47882c2104751a4cc0b923b1` (M6F-D), branch `m6-baselines` |
| Frozen specification | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` (DOCX, parsed directly) |
| Milestone id | M6E. Amendment A9 reserved this identifier for the real M6 closure. The withdrawn provisional artifacts (`M6E_BASELINE_CLOSURE.*`, `m6e_baseline_closure*`) are not revived. |
| `outputs/audit/method_status.csv` | 11 rows, SHA256 `fd97e7cbac35af1ef02fd495d21ad3cfcebffb47702907ab01afd2d447003723` |
| Preflight / test | `tools/m6e_final_closure_preflight.py`, `tests/test_m6e_final_closure.py` |
| Machine-readable evidence | `outputs/audit/M6E_FINAL_M6_CLOSURE.json` |

## 1. The M6 gate (spec §25)

"M6 Baselines | method adapters + method_status.csv | each row tagged faithful/adapted/blocked; no silent
simplification."

The gate labels are exactly `faithful`, `adapted` and `blocked`. No fourth label is used.

Spec §32 asks that `method_status.csv` identify the faithful/adapted/blocked state "for all methods". Spec §26 (the
deterministic smoke test) and full seeds, banks and evaluations belong to later milestones (§25 M8–M10). They are not
part of the M6 gate.

## 2. Canonical row universe: 11 rows

The universe is derived by the preflight from the frozen DOCX, Amendment A1 (`fair_track_v1.yaml`) and Amendment A9.
It is not assumed.

1. **§17 lists 14 experiment ids:** E00–E05, E06a, E06b, E07a, E07b, E08–E11.
2. **§8 defines the faithful/adapted/blocked vocabulary for third-party reimplementations only.** Its heading is
   "Third-party reimplementation policy and method cards", with cards §8.1–§8.7.
3. **E00 is not a row** (`NOT_AN_M6_BASELINE_NO_SYNTHETIC_METHOD`). Its §17 synthetic source is "None / real-only": it
   is the downstream lower bound for M10/M11 and has no generator to tag.
4. **E08–E11 are not rows** (`NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7`). They are GPAT-B0–B3, the frozen proposed
   method (§9). Their deliverable is the §25 M7 row, and the §8 third-party tags do not apply to them.
5. **The M6 baseline universe** is therefore the §17 third-party rows (E01–E05, E06a, E06b, E07a, E07b) plus A1's
   Track-A additions (E06c, E07c).
6. **Each row is classified by A1 and A9:**
   - **Superseded by adopted amendment:** E06a and E07a. They are §17 baselines that A1 places in **neither** track.
   - **Owner-excluded:** E07c and E07b, the A9 scope.
   - **Active M6 baselines:** the other seven.

**E06a (DSDG-BIN) and E07a (DiffFAS-BIN) keep visible rows with gate `blocked`.** This is the only non-misleading label
of the three:
- Their defining training relation is identity-dependent. §8.6 has DSDG pair live/spoof by same subject; §8.7 has
  DIFFFAS-BIN preserve the same-identity reconstruction-pair structure.
- A1 and `fair_track_v1` forbid identity in any Track-A `sampling_rule` and forbid dropping datasets there.
- A1 places neither row in Track B.
- So neither can be instantiated in the adopted protocol. E06c (DEV-020) and E07c (DEV-021) replace them.
- `faithful` or `adapted` would imply an adapter that was never built.
- The authority is DEV-019 (Amendment A1), which is already registered as APPROVED.

## 3. method_status rows

| Row | Track | Scope | Gate | Fidelity | Authority | Seeds | Latest |
|---|---|---|---|---|---|---|---|
| E01 FAS-Aug | A | active | faithful | FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION | A2-01; A2-05 | 0 | M6D1 |
| E02 Frequency Substitution | A | active | adapted | SPEC_DEFINED (spec §8.2 tag: CONTROLLED_ADAPTATION) | spec 8.2; A2-02/03/04 | 0 | M6D1 |
| E03 STDN | A | active | faithful | FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY | E03 runtime addendum | 0 | M6D2b |
| E04 Physics-STD | A | active | adapted | CONTROLLED_ADAPTATION | A3 §4; A4; A2-06 | 0 | M6D3e |
| E05 PCGAN | A | active | adapted | CONTROLLED_ADAPTATION | A2-07; A5; A2-06 | 0 | M6D4e |
| E06a DSDG-BIN | none | superseded | blocked | NOT_ASSESSED_NOT_IMPLEMENTED | DEV-019 (A1) | 0 | M4 (A1) |
| E06b DSDG-NATIVE | B | active | faithful | FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY | M6FD owner decision | 0 | M6FD |
| E06c DSDG-BIN-IDFREE | A | active | adapted | CONTROLLED_ADAPTATION | DEV-020 | 0 | M6D5e |
| E07a DiffFAS-BIN | none | superseded | blocked | NOT_ASSESSED_NOT_IMPLEMENTED | DEV-019 (A1) | 0 | M4 (A1) |
| E07b DiffFAS-NATIVE | B | owner-excluded | blocked | NOT_ASSESSED_NOT_IMPLEMENTED | A9 | 0 | M6A9 |
| E07c DiffFAS-BIN-IDFREE | A | owner-excluded | blocked | CONTROLLED_ADAPTATION | DEV-021; A9 | 0 | M6A9 |

**Honesty constraints that the preflight enforces:**
- E01–E05 are **not** claimed to have real-TRAIN production qualification. Their runtime qualification was synthetic.
- Only E06c (M6D5e), E06b (M6FD) and E07c (M6D6j) passed a production path on real TRAIN.
- `future_bank_eligible = true` only means "not excluded". Any bank still needs completed scientific seeds and the
  smoke gate.

**E06b** (`FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`, owner decision recorded in M6FD):
- These lifecycle disclosures are preserved: `SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED`,
  `CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED`, `RESUME_UNQUALIFIED_FRESH_ONLY` and
  `SCIENTIFIC_TRAINING_NOT_EXECUTED`.
- N_syn stays `DEFERRED_TO_M8` and does not block M6.
- The frozen config (`9d665dc2…`) is unchanged. Its `PENDING_M6F_C_RUNTIME_QUALIFICATION` field is immutable history.
- E06b is not owner-excluded.

**A9 (`OWNER_EXCLUDED_RESOURCE_CONSTRAINT`) covers only E07c and E07b.**

The E07c history is preserved exactly:
- Seed-42 attempt 1 failed before optimization (`RUN_CONTEXT_OPEN`, 0 steps).
- Seed-42 attempt 2 was owner-interrupted at last completed step 54,836 (epoch 25).
- 0 of 3 seeds completed. Seeds 1337 and 2026 were never started.
- `model_050000.pt` (`b4ae6325…`) is retained as NON_SCIENTIFIC_EVIDENCE_ONLY: selected = false, final = false,
  scientific eligibility false.
- There is no bank eligibility and no downstream scientific use.

## 4. Record reconciliation (additive)

- **Deviation register** (`outputs/audit/deviation_report.md`, the canonical register under spec §0.1 rule 9): an
  M6E section is appended.
  - It indexes every M6 amendment and addendum per method, with its own class and owner-approval status.
  - No new DEV number is invented; E04, E05 and E06b are identified by their amendment or decision ids.
  - Every earlier entry is byte-preserved.
  - `method_status.csv` cites the same identifiers.
- **Registry** (`third_party/registry.yaml`):
  - Only the current-state fields of E06b (M6FD state, gate faithful, environment `gpat-m6-e06c` /
    `environments/e06c.lock.json`), E07b (A9 exclusion replaces `DEFERRED_TO_M6_SECONDARY_TRACK`) and E07c (A9) are
    updated, plus a dated header note.
  - The M0-pinned `implementation_status: NOT_STARTED` fields are untouched.
  - Pins are unchanged: FaceX-Zoo `16b793a7…`, DiffFAS `23f40519…`.
  - There is no DiffFAS source-gap classification.
- **CONFIG_STATUS:** an M6E current-state section is appended. It supersedes the stale top-table readings
  (`dsdg_native.yaml` / `difffas_native.yaml` "NOT CREATED (M6)"), which stay as history.

## 5. Gates and authority that are unchanged

- **Smoke test:** `REQUIRED_NOT_COMPLETED`. It is a `HARD_GATE_BEFORE_FIRST_FULL_M8_PLUS_EXECUTION` (M6D6jR O3). It is
  not waived, and it does not block M6 closure or M7 generator training.
- **`outputs/audit/STAGE_STATE.json`** is byte-unchanged (`a8abf828…`, `STALE_HISTORICAL_FIXTURE`; historical tests pin
  it).
- **Current milestone authority** is the execution ledger, the adopted amendments, the milestone evidence and
  `outputs/audit/method_status.csv`.
- **Frozen scientific configs, snapshots, A1 and A9** are unchanged.

## 6. Known environmental test errors (full `tests/test_m6*` on the laptop)

There are exactly **4** errors. Each traceback is `PreparationError: required external asset unavailable:
/media/cong/Data/...`: the external drive is not mounted. All four are `PRE_EXISTING_ENVIRONMENTAL`.

- `test_m6c2a_dsdg.TestDSDG` (setUpClass): LightCNN checkpoint.
- `test_m6c2b1_physics_std.TestE04Geometry` (setUpClass): 3DDFA `bfm_noneck_v3.pkl`.
- `test_m6c2b1_physics_std.TestE04Plans` (setUpClass): 3DDFA `bfm_noneck_v3.pkl`.
- `test_m6c2b1_physics_std.TestE04SourcesAssets.test_actual_assets_verified_without_deserialization`: 3DDFA
  `bfm_noneck_v3.pkl`.

That is 1 LightCNN error plus 3 3DDFA errors, 4 in total.

## 7. After M6 (not M6 defects)

- Full scientific training: 3 seeds for each active baseline, including E06b's 200 epochs and its run-lifecycle,
  checkpoint-writer and resume exercise.
- Real-TRAIN production qualification for E01–E05.
- The E06b N_syn decision (M8).
- The T09 binary-vs-native interpretation (M12).
- The deterministic smoke test.
- M7 GPAT B0–B3, M8 banks and M9 evaluation.

**M7 entry blockers** (separate from M6):
- DEV-003 `lambda_dir` (UNAPPROVED);
- Q-05 (GRL identity-adversary semantics);
- the missing `gpat_b1/b2/b3.yaml`;
- the zero-residual, DWT→IDWT and VAL-only selection gates.

M6E performed no training, no optimizer step, no checkpoint write, no bank, no GPU contact and no TRAIN, VAL or TEST
access. **M7 HAS NOT STARTED.**
