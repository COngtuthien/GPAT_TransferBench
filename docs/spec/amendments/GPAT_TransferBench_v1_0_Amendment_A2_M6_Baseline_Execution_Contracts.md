# GPAT-TransferBench v1.0 — Amendment A2

## M6 Baseline Execution Contracts

**Status:** `OWNER-APPROVED`
**Date:** 2026-09-21
**Milestone:** M6A3b — record and freeze owner-approved baseline execution decisions
**Branch:** `m6-baselines`
**Amendment type:** ADDITIVE. This amendment adds execution-level contracts. It does **not**
edit, weaken or supersede any requirement of the frozen specification or of Amendment A1.

---

## 0. Authority and provenance

| Item | Value |
|---|---|
| Frozen specification | `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` |
| Frozen specification SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 | `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md` |
| Amendment A1 SHA256 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| M6A2 starting commit | `9dc2492750fff98fd2fa8676715eda58df3a9d46` |
| M6A2 result commit (= M6A3a/M6A3b starting commit) | `0d94d5f92f531d6ce22f59bebcd5d2444fe7e9fd` |
| Owner decision packet (input evidence, MD) | `outputs/audit/M6A3_OWNER_DECISION_PACKET.md` — `fcb453f655a6326deac74312aa2e5a7ac988a0ed5e27b714af23959f3f6cecc5` |
| Owner decision packet (input evidence, JSON) | `outputs/audit/M6A3_OWNER_DECISION_PACKET.json` — `d8d139fd2f78a7561d575b2748c9b25c9f92bcd06632796b49a1a31d7d703ef9` |

**Preservation clause.** Every requirement of the frozen specification and of Amendment A1 remains
in force **except** where this amendment records an explicit clarification or adaptation below. Where
this amendment is silent, the frozen specification governs. Where this amendment clarifies a
spec-silent execution detail, it **adds** determinism without changing any frozen scientific value
(operator sets, magnitude distributions, loss coefficients, training budgets, seeds `[42, 1337, 2026]`,
split definitions, or the §8 method definitions).

**TEST prohibition (reaffirmed, unconditional).** TEST **MUST NEVER** be used for checkpoint
selection, model selection, hyperparameter selection, early stopping, or any other selection
decision, for any baseline or for GPAT. No provision of this amendment may be read as relaxing
this prohibition.

---

## 1. Classification of the changes recorded here

Each decision below carries exactly one class:

| Class | Meaning |
|---|---|
| `SCIENTIFIC_CLARIFICATION` | Fixes the meaning of a spec expression that was ambiguous. Changes what is computed only by removing ambiguity. |
| `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION` | Fixes an execution detail the spec leaves open so that results are host-independent and reproducible. No scientific quantity is altered. |
| `CONTROLLED_ADAPTATION` | A documented, disclosed divergence from the original method description, adopted because the original description is internally inconsistent or unrecoverable. Must be disclosed in the final report. |
| `ACQUISITION_AUTHORIZATION` | Approval to obtain an external official asset in a later dedicated step. Resolves no execution state by itself. |
| `UNRESOLVED_SOURCE_GAP` | The owner recorded a direction, but the required scientific source is still missing. The method remains BLOCKED. |

---

## 2. A2-01 — E01 FAS-Aug deterministic asset enumeration order

**Blocker:** `E01-3` · **Method:** E01 (FAS-Aug) · **Class:** `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`

The official FAS-Aug code builds its texture/background/noise name lists with `os.listdir` at import
time and indexes into them with the per-pair RNG. `os.listdir` order is filesystem-dependent, so the
official code does not define a reproducible index → asset mapping.

**Frozen contract.**

1. Each official asset directory MUST be enumerated in **ascending lexicographic (bytewise) order by
   filename** before the per-pair RNG selects an index.
2. For the currently pinned official assets this ordering MUST agree with the canonical
   repository / git-tree ordering (git sorts tree entries bytewise).
3. Runtime filesystem enumeration order MUST NOT affect results.
4. Raw `os.listdir` ordering is **NOT** retained as the scientific contract.

**Explicitly unchanged:** the operator set (all 8 operators retained), the magnitude distribution
(`NUM_MAG = 10`, including `level = 0`), the asset file contents, and the per-pair operator seed.

This is a determinism adaptation only. The FAS-Aug authors' original host filesystem order is
unrecoverable by any means, so no choice can reproduce it; this contract makes the benchmark
reproducible instead of host-dependent.

---

## 3. A2-02 — E02 eligible-block count rule

**Blocker:** `E02-1` · **Method:** E02 (Frequency Substitution) · **Class:** `SCIENTIFIC_CLARIFICATION`

Spec §8.2 says "Select exactly 25 % of eligible blocks". For the frozen 256×256 geometry the audited
eligible-block count is `n_eligible = 159`, and `159 × 0.25 = 39.75`, so the rounding convention is a
real scientific choice.

**Frozen contract — the GENERAL rule, not the number 40.**

For integer `n_eligible`:

```
k = round_half_up(n_eligible * 0.25)
```

Because `0.25 = 1/4`, this MUST be implemented without floating-point ambiguity as:

```
k = (n_eligible + 2) // 4
```

For the frozen geometry:

```
k = (159 + 2) // 4 = 40
```

**Worked verification of the general rule** (audit-computed, M6A3b):

| `n_eligible` | `n/4` | floor | **round-half-up (frozen)** | ceil |
|---|---|---|---|---|
| 156 | 39.00 | 39 | **39** | 39 |
| 157 | 39.25 | 39 | **39** | 40 |
| 158 | 39.50 | 39 | **40** | 40 |
| 159 | 39.75 | 39 | **40** | 40 |
| 160 | 40.00 | 40 | **40** | 40 |

The rule is frozen as a rule because round-half-up and ceil coincide at `n = 159` but diverge at
`n = 157`; freezing only the number 40 would leave the contract undefined if the eligible count ever
changed.

---

## 4. A2-03 — E02 block-selection procedure and RNG

**Blocker:** `E02-2` · **Method:** E02 · **Class:** `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`

**Frozen canonical eligible-block enumeration.**

1. Scan the 16×16 block grid in **row-major** order.
2. `block_row` ascending first.
3. Within each row, `block_col` ascending.
4. Retain only eligible blocks, in that order. Their positions in this retained sequence are the
   indices `0 .. n_eligible-1`.

**Frozen selection procedure.**

```python
rng = numpy.random.Generator(numpy.random.PCG64(pair_seed))
selected_indices = rng.choice(n_eligible, size=k, replace=False)
```

Sampling is **without replacement**.

**Result semantics.** The **selected set** is the scientific result. No later scientific behaviour may
depend on the order in which `rng.choice` returned the indices. Where an implementation needs a
canonical processing order after selection, it MUST **sort the selected block indices ascending**
before applying substitution.

**Prohibited.** `numpy.random.default_rng` without explicitly naming `PCG64`; `random.sample`;
hash ranking.

**Version-pinning obligation.** The runtime environment MUST later pin the NumPy version used by the
authoritative implementation, because `Generator.choice` is a NumPy implementation detail. The
version observed during this audit-only verification was NumPy `2.5.3`; this observation is
**NOT** the authoritative pin and MUST be re-established by the runtime-environment freeze.

---

## 5. A2-04 — E02 pair-specific seed (dedicated namespace)

**Blocker:** `E02-3` · **Method:** E02 · **Class:** `SCIENTIFIC_CLARIFICATION`

`global_seed` means the **CURRENT EXPERIMENT SEED**, drawn from the frozen set `{42, 1337, 2026}`.

**Frozen contract.**

```
preimage = UTF-8( "gpatbench.freqsub.block.v1|" + pair_id + "|" + decimal_string(global_seed) )
digest   = SHA256(preimage).digest()
pair_seed = unsigned big-endian integer of the FULL 32-byte digest
```

`pair_seed` is passed **directly** to `numpy.random.PCG64(pair_seed)`.

- **No truncation.** The full 32-byte digest is used.
- **No modulo.**
- **No hidden separator**, beyond the two literal `|` characters shown.
- **No whitespace. No newline.**

This namespace is **intentionally different** from E01's §8.1 seed (A2-05), so that E01 and E02 do not
derive correlated randomness from the same preimage for the same pair. It follows the frozen M4
`pairs_v1` namespacing convention in style, but is a distinct namespace scoped to E02 block selection.

**Worked example** (audit-computed, `pair_id = "PTR000001"`):

| `global_seed` | preimage | `pair_seed` |
|---|---|---|
| 42 | `gpatbench.freqsub.block.v1\|PTR000001\|42` | `19354000309184887456743517179028239418116648094833629759175866349341962868907` |
| 1337 | `gpatbench.freqsub.block.v1\|PTR000001\|1337` | `100642993489923994340931747082706323367992294802124459081927424298639285382524` |
| 2026 | `gpatbench.freqsub.block.v1\|PTR000001\|2026` | `70582000648117090776289456477083767468449122475305887305661581023317195795819` |

---

## 6. A2-05 — E01 FAS-Aug seed byte layout (OBS-1)

**Observation:** `OBS-1` · **Method:** E01 · **Class:** `SCIENTIFIC_CLARIFICATION`

This section freezes the **byte-level interpretation** of the already-frozen §8.1 expression
`SHA256(pair_id + global_seed) mod 2^31`. It is **NOT** a change to the frozen §8.1 scientific
formula, which is preserved verbatim.

OBS-1 identified two open facets. Both are resolved here.

**Facet 1 — the meaning of `global_seed`.** For E01, `global_seed` is the **current experiment seed**,
drawn from `{42, 1337, 2026}`. Consequence: the E01 bank **differs per benchmark seed**; it is not
identical across the three seeds.

**Facet 2 — the preimage byte layout.**

```
preimage = UTF-8( pair_id + decimal_string(global_seed) )
```

There is **NO separator**, no whitespace and no newline.

```
digest        = SHA256(preimage).digest()
operator_seed = unsigned_big_endian_integer(digest) mod 2^31
```

**Example structure.** With `pair_id = "PTR000001"` and `global_seed = 42`, the preimage text is
exactly:

```
PTR00000142
```

**Worked example** (audit-computed, `pair_id = "PTR000001"`):

| `global_seed` | preimage text | `operator_seed` |
|---|---|---|
| 42 | `PTR00000142` | `795981663` |
| 1337 | `PTR0000011337` | `924982993` |
| 2026 | `PTR0000012026` | `1531213468` |

---

## 7. A2-06 — `BASELINE_FINAL_STATE_V1` global checkpoint policy

**Class:** `DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`
**Blockers closed:** `E04-3`, `E05-3`, `E07c-5`

**Purpose.** Resolve baseline checkpoint-selection gaps **without adding extra benchmark-side
checkpoint tuning**.

**Frozen policy.**

1. If an official method/source **explicitly defines** which checkpoint/state is used for generation,
   that official rule is **preserved**.
2. Otherwise, use the model state at the **END of the frozen training budget**.
3. If periodic upstream checkpoint saving does not save the exact terminal state, save **one
   additional TERMINAL checkpoint** immediately after the final scheduled training step/epoch, with
   **ZERO additional optimizer steps**.
4. VAL **MAY** be evaluated for diagnostics.
5. VAL **MUST NOT** select the checkpoint for methods governed by `BASELINE_FINAL_STATE_V1`, unless
   the original official method explicitly requires validation-based checkpoint selection.
6. TEST **MUST NEVER** be used for checkpoint selection.

**Non-override clause.** `BASELINE_FINAL_STATE_V1` MUST NOT override an already-resolved official
checkpoint rule. Specifically:

| Method | Already-resolved official rule | Effect of `BASELINE_FINAL_STATE_V1` |
|---|---|---|
| E03 (STDN) | latest/final `ckpt-50` (source-resolved) | **NOT overridden** — the official rule stands under clause 1 |
| E06c (DSDG-BIN-IDFREE) | generator epoch 200 (source-resolved) | **NOT overridden** — the official rule stands under clause 1 |

**Applications.**

| Blocker | Method | Governed terminal state |
|---|---|---|
| `E04-3` | E04 (Physics-Guided STD) | terminal state at **iteration 150,000** |
| `E05-3` | E05 (PCGAN) | terminal state at **iteration 4,000** |
| `E07c-5` | E07c (DiffFAS-BIN-IDFREE) | terminal model state at the end of the frozen **400-epoch** training budget. Upstream saves every 10,000 iterations, which does not in general coincide with the terminal state; a terminal checkpoint MUST therefore be created under clause 3, **without additional optimization**. |

**Disclosure.** Where a source states only a training budget and no selection rule, "end of budget" is
a **benchmark convention**, not a paper fact, and MUST be disclosed as such.

---

## 8. A2-07 — E05 PCGAN architecture resolution

**Blocker:** `E05-2` · **Method:** E05 · **Class:** `CONTROLLED_ADAPTATION`

**Frozen contract.** The pinned implementation `[34]` is the executable architecture basis:

```
taesungp/swapping-autoencoder-pytorch
commit 6baa180f1184ee79a6b967f9d80ee0e02a979ac7
```

E05 follows `[34]`'s **actual executable architecture** for the components inherited from `[34]`,
including its **StyleGAN2-style modulation/demodulation** and the corresponding discriminator
implementation. StyleGAN-v1 AdaIN MUST NOT be separately ported merely to follow the ambiguous PCGAN
prose citation `[35]`.

**Disclosure requirement — this discrepancy MUST NOT be hidden.**

E05 fidelity is recorded as:

```
CONTROLLED_ADAPTATION
```

**Reason.** PCGAN's prose references AdaIN / StyleGAN-v1 `[35]`, while its explicitly named executable
architectural base `[34]` uses StyleGAN2-style weight demodulation. StyleGAN2 explicitly replaced
AdaIN with demodulation, so the two cannot both be satisfied.

**Preserved.** Method ID **E05**; benchmark result row **E05**. The method is **NOT** renamed and
**NOT** removed.

**Reporting obligation.** The final report/paper MUST disclose this architecture-resolution choice.
The E05 row MUST NOT be described as "native PCGAN" or "faithful PCGAN".

---

## 9. A2-08 — E06c LightCNN-29 v2 acquisition authorization

**Blocker:** `E06c-2` · **Method:** E06c · **Class:** `ACQUISITION_AUTHORIZATION`

The owner **AUTHORISES** a later dedicated acquisition of the exact official LightCNN-29 v2 checkpoint
already identified in M6A2. The scientific identity is already resolved; only the bytes are absent.

| Item | Value |
|---|---|
| Expected official source identity | original LightCNN author release — the **same** Google Drive file id referenced by both upstream LightCNN and DSDG |
| Expected path/name from DSDG | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` |

**No bytes were fetched in M6A3b.** This amendment records acquisition approval only.

**This blocker is NOT yet execution-resolved.** Status:
`OWNER_AUTHORIZED_ACQUISITION_PENDING` — scientific contract resolved, acquisition approved, bytes
not fetched, local SHA256 unknown, execution still blocked.

**Obligations of the later dedicated acquisition step.** It MUST:

1. download from the exact official location;
2. compute SHA256;
3. inventory file size;
4. inspect checkpoint keys safely;
5. verify compatibility with `define_IP()`;
6. record provenance;
7. **never** silently substitute another identity model.

**E06c remains NOT executable until that later step succeeds.**

---

## 10. A2-09 — E04 Physics-Guided STD geometry/depth route

**Blocker:** `E04-2` · **Method:** E04 · **Class:** `UNRESOLVED_SOURCE_GAP`

**Frozen direction: preserve the `FAITHFUL_PAPER` intent.**

The benchmark MUST NOT invent:

- a substitute 140-vertex set;
- a new depth renderer;
- a FaceXFormer-68 → 140 conversion;
- a substitute 3DMM;
- a benchmark-created depth target.

**Permitted direction.** Official/licensed BFM acquisition may be pursued in a later dedicated asset
step; official authors/upstream sources may be contacted or searched for the missing 140-vertex
definition and depth-rendering route.

**E04-2 is NOT resolved.** Selecting this direction does not close the blocker. Status:

```
OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP
```

**Unresolved scientific items that remain:**

1. BFM licensed asset not acquired;
2. `Q = 140` vertex set not published/resolved;
3. depth-rendering procedure not published/resolved;
4. exact executable legacy geometry route not yet reproducibly frozen.

**E04 remains BLOCKED.** No claim of successful E04 reproduction may be made while these source
assets remain unavailable.

---

## 11. A2-10 — E07c DiffFAS PADISI conditioning encoder

**Blocker:** `E07c-4` · **Method:** E07c · **Class:** `UNRESOLVED_SOURCE_GAP`

**Frozen direction: preserve the FAITHFUL official DiffFAS route.**

The benchmark MUST NOT:

- train a benchmark-side substitute encoder;
- disable the conditioning encoder;
- replace it with an ImageNet ResNet18;
- invent PADISI training details.

**Permitted direction.** Attempt to obtain the official PADISI conditioning encoder, or exact
training/release information, from the original authors or an authoritative source, in a later
source-resolution step.

**E07c-4 is NOT resolved.** Status:

```
OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP
```

**E07c remains BLOCKED.** No claim of successful E07c reproduction may be made while the official
conditioning encoder remains unavailable.

---

## 12. Resulting blocker state

| | Count |
|---|---|
| Active blockers before these owner decisions (M6A2) | **11** |
| Resolved/closed by owner contract in M6A3b | **8** |
| **Active blockers after M6A3b** | **3** |

**Resolved by owner contract:** `E01-3`, `E02-1`, `E02-2`, `E02-3`, `E04-3`, `E05-2`, `E05-3`, `E07c-5`.

**Still active:**

| Blocker | Method | Why it is still active |
|---|---|---|
| `E04-2` | E04 | source/asset/scientific-route gap |
| `E06c-2` | E06c | acquisition authorized, but official LightCNN bytes not yet fetched/hash-pinned |
| `E07c-4` | E07c | official PADISI encoder/source gap |

**Method-level state.**

| Method | State |
|---|---|
| E01 | contract-ready for implementation |
| E02 | contract-ready for implementation |
| E03 | contract-ready for implementation |
| E04 | **BLOCKED** by `E04-2` |
| E05 | contract-ready for implementation — fidelity `CONTROLLED_ADAPTATION` |
| E06c | **BLOCKED** pending authorized LightCNN acquisition |
| E07c | **BLOCKED** by `E07c-4` |

**Not all M6 baselines are READY.** Three blockers remain active and three methods remain blocked.

---

## 13. Scientific rationale

These decisions were selected to make the benchmark deterministic and fair. The rationale of record is:

- **deterministic reproducibility** — host-dependent behaviour is eliminated (A2-01, A2-03);
- **source-faithful execution where recoverable** — the pinned executable source governs where it is
  unambiguous (A2-07), and faithful intent is preserved rather than substituted where the source is
  missing (A2-09, A2-10);
- **no extra validation-based checkpoint tuning** when the source does not specify it (A2-06);
- **no TEST-based selection**, anywhere, ever;
- **explicit disclosure** when controlled adaptation is unavoidable (A2-07).

These decisions are **not** to be described as making baselines weaker, helping GPAT win, or
handicapping competitors.

---

## 14. Scope boundaries of this amendment

- The original frozen DOCX is **NOT** edited.
- Amendment A1 is **NOT** edited.
- No existing `docs/spec` file is edited; this amendment is a new additive file.
- No `configs/methods/*.yaml` is created by this amendment.
- No training, synthetic-bank generation, TEST access, model execution, checkpoint creation or weight
  download is authorized or performed by this amendment.
