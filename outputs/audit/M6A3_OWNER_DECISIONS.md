# M6A3b — Owner-Approved Baseline Execution Decisions

**Status:** `OWNER_APPROVED` · **Date:** 2026-09-21 · **Branch:** `m6-baselines`
**Starting commit:** `0d94d5f92f531d6ce22f59bebcd5d2444fe7e9fd`
**M6A2 starting commit:** `9dc2492750fff98fd2fa8676715eda58df3a9d46`

This document **records** owner decisions. It does not recommend, reconsider or replace them.
Every `owner_choice` below was issued explicitly by the owner. The `recommended_option` fields in the
M6A3a packet are **inputs only** and were **not** used to decide — in several cases the owner's choice
differs from the packet's recommendation, and the owner's choice governs.

This is **M6 baseline-contract work**, not GPAT implementation. Nothing was trained, generated,
downloaded or executed.

## Provenance

| Item | Value |
|---|---|
| Frozen specification SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 SHA256 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| Amendment A2 (created here) | `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A2_M6_Baseline_Execution_Contracts.md` |
| Input evidence — packet MD | `fcb453f655a6326deac74312aa2e5a7ac988a0ed5e27b714af23959f3f6cecc5` |
| Input evidence — packet JSON | `d8d139fd2f78a7561d575b2748c9b25c9f92bcd06632796b49a1a31d7d703ef9` |

The two M6A3a packet files are **input evidence** and were not rewritten. M6A1 and M6A2 historical
artifacts were not rewritten; `M6A2_BLOCKER_STATE.json` was **not** mutated.

---

## 0. Summary

| Blocker | Method | Owner choice | Class | Status after M6A3b | Active |
|---|---|---|---|---|---|
| `E01-3` | E01 | **A** — lexicographic enumeration | deterministic impl. clarification | `RESOLVED_BY_OWNER_DECISION` | no |
| `E02-1` | E02 | **B** — round-half-up | scientific clarification | `RESOLVED_BY_OWNER_DECISION` | no |
| `E02-2` | E02 | **B** — PCG64 without replacement | deterministic impl. clarification | `RESOLVED_BY_OWNER_DECISION` | no |
| `E02-3` | E02 | **B** — dedicated E02 namespace | scientific clarification | `RESOLVED_BY_OWNER_DECISION` | no |
| `OBS-1` | E01 | approve byte-layout clarification | scientific clarification | `RESOLVED_BY_OWNER_DECISION` | no (never a counted blocker) |
| `E04-2` | E04 | **A** — preserve `FAITHFUL_PAPER` | unresolved source gap | `OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP` | **YES** |
| `E04-3` | E04 | `BASELINE_FINAL_STATE_V1` | deterministic impl. clarification | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` | no |
| `E05-2` | E05 | **A** — pinned `[34]` as executable basis | **controlled adaptation** | `RESOLVED_BY_OWNER_CONTROLLED_ADAPTATION` | no |
| `E05-3` | E05 | `BASELINE_FINAL_STATE_V1` | deterministic impl. clarification | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` | no |
| `E06c-2` | E06c | **A** — authorize later acquisition | acquisition authorization | `OWNER_AUTHORIZED_ACQUISITION_PENDING` | **YES** |
| `E07c-4` | E07c | **A** — preserve faithful DiffFAS route | unresolved source gap | `OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP` | **YES** |
| `E07c-5` | E07c | `BASELINE_FINAL_STATE_V1` | deterministic impl. clarification | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` | no |

**Blocker count: 11 → 3.** Not all M6 baselines are READY.

---

## 1. E01-3 — FAS-Aug asset enumeration · option **A**

Deterministic asset ordering is frozen as:

- enumerate each official asset directory in **ascending lexicographic order by filename**;
- for the currently pinned official assets this MUST agree with the canonical repository/git-tree
  ordering;
- runtime filesystem enumeration order MUST NOT affect results.

Raw `os.listdir` ordering is **not** retained as the scientific contract.

This is a **deterministic execution adaptation only**. Unchanged: the operator set, the magnitude
distribution, the asset contents, and the per-pair operator seed.

**Status:** `RESOLVED_BY_OWNER_DECISION`

---

## 2. E02-1 — number of eligible blocks · option **B** (ROUND-HALF-UP)

Known: `n_eligible = 159` for the frozen 256×256 geometry.

The **GENERAL rule** is frozen, not just the number 40. For integer `n_eligible`:

```
k = round_half_up(n_eligible * 0.25)
```

Because `0.25 = 1/4`, implement without floating ambiguity as:

```
k = (n_eligible + 2) // 4
```

For the frozen geometry: `k = (159 + 2) // 4 = 40`.

Audit-computed verification of the rule: `n = 156 → 39`, `157 → 39`, `158 → 40`, `159 → 40`,
`160 → 40`. Round-half-up and ceil coincide at 159 but diverge at 157, which is why the rule is
frozen as a rule.

**Status:** `RESOLVED_BY_OWNER_DECISION`

---

## 3. E02-2 — block-selection RNG · option **B**

NumPy PCG64 sampling **without replacement**.

**Canonical eligible-block enumeration:** scan the 16×16 block grid in **row-major** order,
`block_row` ascending first, `block_col` ascending within each row; retain only eligible blocks in
that order.

```python
rng = numpy.random.Generator(numpy.random.PCG64(pair_seed))
selected_indices = rng.choice(n_eligible, size=k, replace=False)
```

The **selected set** is the scientific result. No later scientific behaviour may depend on the
returned order. Where a canonical processing order is needed after selection, **sort the selected
block indices ascending** before applying substitution.

**Prohibited:** `default_rng` without explicitly naming `PCG64`; Python `random.sample`; hash ranking.

**Runtime obligation:** the runtime environment MUST later pin the NumPy version used by the
authoritative implementation. The version observed during this audit-only verification was `2.5.3`;
this is **not** the authoritative pin.

**Status:** `RESOLVED_BY_OWNER_DECISION`

---

## 4. E02-3 — pair-specific seed · option **B** (dedicated E02 namespace)

`global_seed` means the **CURRENT EXPERIMENT SEED**, from `{42, 1337, 2026}`.

```
preimage  = UTF-8("gpatbench.freqsub.block.v1|" + pair_id + "|" + decimal_string(global_seed))
digest    = SHA256(preimage).digest()
pair_seed = unsigned big-endian integer of the FULL 32-byte digest
```

`pair_seed` is passed **directly** to `numpy.random.PCG64(pair_seed)`.

No truncation. No modulo. No hidden separator. No whitespace. No newline.

This namespace is **intentionally different** from E01's.

**Worked example** (`pair_id = "PTR000001"`, audit-computed):

| `global_seed` | preimage | `pair_seed` |
|---|---|---|
| 42 | `gpatbench.freqsub.block.v1\|PTR000001\|42` | `19354000309184887456743517179028239418116648094833629759175866349341962868907` |
| 1337 | `gpatbench.freqsub.block.v1\|PTR000001\|1337` | `100642993489923994340931747082706323367992294802124459081927424298639285382524` |
| 2026 | `gpatbench.freqsub.block.v1\|PTR000001\|2026` | `70582000648117090776289456477083767468449122475305887305661581023317195795819` |

**Status:** `RESOLVED_BY_OWNER_DECISION`

---

## 5. OBS-1 — E01 FAS-Aug seed byte layout

The owner approved freezing the **byte-level interpretation** of the already-frozen §8.1 expression
`SHA256(pair_id + global_seed) mod 2^31`. This is recorded as a **byte-layout clarification**, NOT as
a change to the frozen §8.1 scientific formula.

OBS-1 raised two facets; both are resolved:

1. **`global_seed`** = the current experiment seed, from `{42, 1337, 2026}`. Consequence: the E01 bank
   **differs per benchmark seed**.
2. **Byte layout:**

```
preimage = UTF-8(pair_id + decimal_string(global_seed))
```

There is **NO separator**.

```
digest        = SHA256(preimage).digest()
operator_seed = unsigned_big_endian_integer(digest) mod 2^31
```

**Example structure.** `pair_id = "PTR000001"`, `global_seed = 42` → preimage text is exactly
`PTR00000142`.

**Worked example** (`pair_id = "PTR000001"`, audit-computed):

| `global_seed` | preimage text | `operator_seed` |
|---|---|---|
| 42 | `PTR00000142` | `795981663` |
| 1337 | `PTR0000011337` | `924982993` |
| 2026 | `PTR0000012026` | `1531213468` |

OBS-1 was never counted among the 11 blockers, so this resolution does not change the blocker
arithmetic.

---

## 6. E04-2 — Physics-Guided STD geometry/depth route · option **A**

The `FAITHFUL_PAPER` intent is **preserved**. The benchmark MUST NOT invent a substitute 140-vertex
set, a new depth renderer, a FaceXFormer-68 → 140 conversion, a substitute 3DMM, or a
benchmark-created depth target.

**Direction:** official/licensed BFM acquisition may be pursued in a later dedicated asset step;
official authors/upstream sources may be contacted or searched for the missing 140-vertex definition
and depth-rendering route.

**E04-2 is NOT resolved** merely because the owner selected A. E04 remains BLOCKED until the required
scientific route is actually resolved.

**Status:** `OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP`

**Unresolved scientific items:** (1) BFM licensed asset not acquired; (2) `Q = 140` vertex set not
published/resolved; (3) depth-rendering procedure not published/resolved; (4) exact executable legacy
geometry route not yet reproducibly frozen.

---

## 7. `BASELINE_FINAL_STATE_V1` — global checkpoint policy

**Purpose:** resolve baseline checkpoint-selection gaps **without adding extra benchmark-side
checkpoint tuning**.

1. If an official method/source explicitly defines which checkpoint/state is used for generation,
   preserve that official rule.
2. Otherwise, use the model state at the **END of the frozen training budget**.
3. If periodic upstream checkpoint saving does not save the exact terminal state, save one additional
   **TERMINAL** checkpoint immediately after the final scheduled training step/epoch, with **ZERO**
   additional optimizer steps.
4. VAL **may** be evaluated for diagnostics.
5. VAL **MUST NOT** select the checkpoint for methods governed by `BASELINE_FINAL_STATE_V1`, unless
   the original official method explicitly requires validation-based checkpoint selection.
6. TEST **MUST NEVER** be used for checkpoint selection.

**This policy does NOT override an already-resolved official checkpoint rule:**

| Method | Official rule | Overridden? |
|---|---|---|
| E03 | official/source-resolved latest/final `ckpt-50` | **No** |
| E06c | official/source-resolved generator epoch 200 | **No** |

**Applications:**

| Blocker | Method | Terminal state | Status |
|---|---|---|---|
| `E04-3` | E04 | iteration **150,000** | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` |
| `E05-3` | E05 | iteration **4,000** | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` |
| `E07c-5` | E07c | end of the frozen **400-epoch** budget; if upstream periodic saving (every 10,000 iterations) does not coincide exactly, create a terminal checkpoint **without additional optimization** | `RESOLVED_BY_OWNER_DECISION_BASELINE_FINAL_STATE_V1` |

Where a source gives only a training budget and no selection rule, "end of budget" is a **benchmark
convention**, not a paper fact, and must be disclosed as such.

---

## 8. E05-2 — PCGAN architecture · option **A**

The pinned `[34]` implementation is the executable architecture basis:

```
taesungp/swapping-autoencoder-pytorch
commit 6baa180f1184ee79a6b967f9d80ee0e02a979ac7
```

E05 follows its **actual executable architecture** for the components inherited from `[34]`, including
its StyleGAN2-style modulation/demodulation and the corresponding discriminator implementation.
StyleGAN-v1 AdaIN is **not** separately ported merely to follow the ambiguous PCGAN prose citation
`[35]`.

**This discrepancy MUST NOT be hidden.** E05 fidelity is recorded as:

```
CONTROLLED_ADAPTATION
```

**Reason:** PCGAN prose references AdaIN / StyleGAN-v1, while its explicitly named executable
architectural base `[34]` uses StyleGAN2-style demodulation.

**Preserved:** method ID **E05**, benchmark result row **E05**. The method is not renamed and not
removed. The final report/paper **must disclose** this architecture-resolution choice, and the E05 row
must **not** be called "native PCGAN" or "faithful PCGAN".

**Status:** `RESOLVED_BY_OWNER_CONTROLLED_ADAPTATION`

---

## 9. E06c-2 — LightCNN-29 v2 checkpoint · option **A**

The owner **AUTHORISES** a later dedicated acquisition of the exact official LightCNN-29 v2 checkpoint
already identified in M6A2. Scientific identity is already resolved.

| Item | Value |
|---|---|
| Expected official source identity | original LightCNN author release; the **same** Google Drive file id referenced by both upstream LightCNN and DSDG |
| Expected path/name from DSDG | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` |

**IT WAS NOT DOWNLOADED IN M6A3b.** M6A3b records acquisition approval only.

**Status:** `OWNER_AUTHORIZED_ACQUISITION_PENDING` — meaning: scientific contract resolved,
acquisition approved, **bytes not fetched**, **local SHA256 unknown**, **execution still blocked**.

The later dedicated acquisition step must: download from the exact official location; compute SHA256;
inventory file size; inspect checkpoint keys safely; verify compatibility with `define_IP()`; record
provenance; and **never** silently substitute another identity model.

**E06c remains NOT executable until that later step succeeds.**

---

## 10. E07c-4 — DiffFAS PADISI conditioning encoder · option **A**

The **FAITHFUL official DiffFAS route** is preserved. The benchmark MUST NOT train a benchmark-side
substitute encoder, disable the conditioning encoder, replace it with an ImageNet ResNet18, or invent
PADISI training details.

**Direction:** attempt to obtain the official PADISI conditioning encoder, or exact training/release
information, from the original authors / an authoritative source in a later source-resolution step.

**E07c-4 is NOT resolved.** E07c remains BLOCKED meanwhile.

**Status:** `OWNER_DIRECTION_RECORDED_STILL_BLOCKED_SOURCE_GAP`

---

## 11. Resulting active blocker state

| | Count |
|---|---|
| Active blockers before owner decisions (M6A2) | **11** |
| Resolved/closed by owner contract | **8** |
| **Active after M6A3b** | **3** |

**Resolved by owner contract:** `E01-3`, `E02-1`, `E02-2`, `E02-3`, `E04-3`, `E05-2`, `E05-3`, `E07c-5`.

**Still active:**

| Blocker | Why |
|---|---|
| `E04-2` | source/asset/scientific-route gap |
| `E06c-2` | acquisition authorized, but official LightCNN bytes not yet fetched/hash-pinned |
| `E07c-4` | official PADISI encoder/source gap |

**Method state:**

| Method | State |
|---|---|
| E01 | contract-ready for implementation |
| E02 | contract-ready for implementation |
| E03 | contract-ready for implementation |
| E04 | **BLOCKED** by `E04-2` |
| E05 | contract-ready for implementation — fidelity `CONTROLLED_ADAPTATION` |
| E06c | **BLOCKED** pending authorized LightCNN acquisition |
| E07c | **BLOCKED** by `E07c-4` |

**Not all M6 baselines are READY.** No `configs/methods/*.yaml` was created; M6B has not started.

---

## 12. Scientific boundaries

These decisions were selected to make the benchmark **deterministic and fair**. The rationale of
record is:

- deterministic reproducibility;
- source-faithful execution where recoverable;
- no extra validation-based checkpoint tuning when the source does not specify it;
- no TEST-based selection;
- explicit disclosure when controlled adaptation is unavoidable.

They are **not** to be described as "making baselines weaker", "helping GPAT win", or "handicapping
competitors".

For **E05** specifically: the controlled adaptation must not be called "native PCGAN" or "faithful
PCGAN". For **E04** and **E07c**: no claim of successful reproduction may be made while the required
source assets remain unavailable.

---

## 13. Execution boundaries observed in M6A3b

No training. No synthetic bank generation. No TEST access. No model execution. No checkpoint created.
No weight download. No method config created. No commit. No push. `STAGE_STATE` M6 was not changed to
COMPLETE. The frozen DOCX, Amendment A1, `configs/frozen`, `frozen_config_snapshot`, `manifests` and
`gpatbench` are unchanged.
