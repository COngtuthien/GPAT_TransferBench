# M6A5a — Owner Decision Packet: Unreproducible Baselines (E04, E07c)

**Date:** 2026-09-21 · **Branch:** `m6-baselines`
**Starting commit:** `6e81f797e767d44f0d02c8910919fbc03133e0cc`

**Nothing is decided or resolved in this packet.** It exists only to put a precise, evidence-backed
set of choices in front of the owner. Neither blocker is resolved here, and no recommendation below
is an owner decision.

| Item | SHA256 |
|---|---|
| Frozen specification | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| Amendment A2 | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |

Active blockers at entry: **`E04-2`**, **`E07c-4`**.
No remaining blocker: **E01, E02, E03, E05, E06c**. Blocked: **E04, E07c**.

---

## 0. The single most important finding

**`BLOCKED_BY_SOURCE_GAP` is already one of the four status tags defined by the frozen
specification (§8).** Option A is therefore not an invention — it is the disposition the spec itself
prescribes.

| Status tag (§8) | Meaning |
|---|---|
| `FAITHFUL_OFFICIAL` | Official code/config used; only dataset paths/adapters and output wrapper changed |
| `FAITHFUL_PAPER` | No usable official code; architecture/equations reconstructed from paper without method-changing assumptions |
| `CONTROLLED_ADAPTATION` | Paper idea reimplemented under an explicitly frozen adaptation defined here |
| **`BLOCKED_BY_SOURCE_GAP`** | **A required method detail is missing and no override is defined; stop this method rather than guess** |

And §8.4 — the Physics-STD section itself — gives a direct instruction for exactly E04's situation:

> "If a tensor dimension, architecture block, loss coefficient or optimization setting **that changes
> the scientific method** is missing, mark `BLOCKED_BY_SOURCE_GAP` **instead of choosing one**."

Reinforced by the §0.1 hard rules: never substitute a named model (rule 3); never invent a missing
third-party detail (rule 4); **never fabricate a result, interpolate a missing method, or copy a
reported paper result into an "our reimplementation" table** (rule 7).

`N/A` is likewise already spec-native: §19.1's paper-ready templates carry `N/A` / `N/A*` cells
(FAS-Aug ArtSim, DSDG pairwise metrics), the M9 acceptance check is literally *"N/A used correctly"*,
and the spec already rules *"VAL class/attack absent | metric shown N/A; no resampling from TEST"*.

---

## 1. Fairness principle governing this packet

> A baseline must **not** be made artificially weak merely to favor GPAT.
> A baseline must **not** be silently reconstructed with undocumented components and then called faithful.

**Inability to execute is categorically different from poor baseline performance.** `N/A` due to an
external source gap is **not** a loss, **not** a score, and **not** a performance result.

**Direction-of-bias note, stated plainly.** The GPAT-favorable risk in this decision sits with
**option B**, not option A. A hastily substituted depth target (E04) or a benchmark-trained
substitute encoder (E07c) would most plausibly **underperform** the true method while still carrying
that method's name — an artificially weak baseline that flatters GPAT invisibly. Option A carries the
opposite profile: it removes comparison points, which is visible and disclosed rather than silent.
This asymmetry is recorded so the owner can weigh it directly.

---

## 2. Matrix-integrity analysis

**E04 is a §17 required row:** `| E04 | Physics-STD | Binary=native | Both | Physics/frequency trace
transfer |`.

**E07c is not.** §17 lists E07a (DiffFAS-BIN) and E07b (DiffFAS-NATIVE); **E07c** (DIFFFAS-BIN-IDFREE,
DEV-021) is introduced by **Amendment A1** as the Track-A identity-free main-track variant. Its
disposition touches A1's Track-A table, not §17 directly. E07c is also **already**
`CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH`, so an option-B substitute encoder would be
a **second** adaptation layer stacked on the first.

### 2.1 Does N/A retention require a new additive amendment?

**Yes — recommended, but for a narrower reason than it first appears.**

**The status itself does not need one.** Marking these rows `BLOCKED_BY_SOURCE_GAP` and reporting
`N/A` is already authorized: it is a §8 status tag, §8.4 explicitly instructs it for E04's situation,
and §0.1 rules 3/4/7 forbid every alternative.

**An amendment is still required because:**

1. §17 is titled "**Required** experiment matrix" and lists E04 with evaluator "Both". Shipping that
   row unexecuted is a deviation from the matrix as written, and **§0.1 rule 9** requires every
   deviation to be recorded with an explicit `APPROVED`/`UNAPPROVED` status — *"Unapproved deviations
   cannot be used as main paper results."*
2. The spec defines `N/A` cells but **nowhere defines how N/A rows are excluded from numeric
   aggregates**, means, CIs, Pareto fronts, the normalized radar (fig09) or the method-ranking
   parallel-coordinates plot (fig27, sourced from T04/T06/T15). That rule is currently undefined and
   must be frozen **before** any table is produced.
3. E07c's disposition modifies A1's Track-A table, so it needs an additive instrument of equal standing.

**Minimum sufficient alternative:** an `APPROVED` entry in `outputs/audit/deviation_report.md` (per
§0.1 rule 9) is the minimum that makes option A usable in main paper results — but it would leave the
aggregate-exclusion rule unfrozen. **An additive Amendment A3 is the cleaner vehicle** (additive only;
it would not edit the frozen DOCX, A1 or A2).

### 2.2 Row preserved, executable status changed

Option A keeps E04 and E07c **visible as rows** in the matrix and in every result table, and changes
only their **executable status** to "not executed — external source gap". The matrix's scientific
claim (which methods the benchmark intends to compare) is preserved; only the claim "we measured this
one" is withdrawn — which is the honest position.

### 2.3 N/A, not zero, not missing

| Must use | Must **not** use |
|---|---|
| `N/A` with an explicit reason token and dated provenance, e.g. `N/A_BLOCKED_BY_SOURCE_GAP` (as of 2026-09-21) | `0` / `0.0`; a blank or empty cell; a deleted row; an em-dash placeholder; a paper-reported number copied in; an interpolated/imputed value; a "worst observed" or chance-level stand-in |

§0.1 rule 7 forbids fabricating, interpolating or copying a reported result. A zero would additionally
read as a **measured performance floor** — precisely the inability-vs-performance confusion the
fairness principle forbids. Note that §19.1's em-dash `—` already means *"not yet measured"*, so `N/A`
must be a **distinct token** meaning *"cannot be measured"*.

### 2.4 Effect on aggregate statistics — yes, material

Affected: T04 generator pair metrics; T06 primary downstream video metrics (mean ± std + CI); T15
compute; fig09 normalized radar; fig10 Pareto; fig25 compute tradeoff; fig27 method-ranking parallel
coordinates; and any "best method" or rank-order claim.

- **N/A rows MUST be excluded** from every numerical average, standard deviation, confidence
  interval, normalization range, percentile rank, Pareto front and rank summary.
- **Every aggregate must state its denominator explicitly** — e.g. *"mean over the 5 executed
  baselines; E04 and E07c excluded (`BLOCKED_BY_SOURCE_GAP`)"*. A silently smaller *n* is itself a
  reporting defect.
- **Normalization caution:** radar and parallel-coordinate plots derive min/max from the participating
  set, so excluding two rows changes the normalization range for **all remaining rows**. The excluded
  set must appear in the figure caption, not only in the table.

### 2.5 Silent removal is forbidden

Neither method may be silently removed from any table. Removal without disclosure would
misrepresent the intended comparison set and would be indistinguishable, to a reader, from a
benchmark that never intended to include a strong competitor.

---

## 3. E04 — Physics-Guided STD

**Current:** `STILL_BLOCKED_SOURCE_GAP` · **Spec target:** `FAITHFUL_PAPER` unless a verified
official code release is found · **Prior owner direction (A2-09):** preserve `FAITHFUL_PAPER`.
**M6A4 confirmed 0 of 4 sub-gaps closed.**

Unresolved: licence-gated **BFM** (version not even identifiable); exact **Q=140 vertex indices**
never published; **depth-target rendering** never published (`M₀` is an *input* to the published
Algorithm 1); **no complete reproducible executable official route**.

**Scientific weight of the gap:** `L_depth` carries **α₁ = 100**, the dominant coefficient in the
paper's total loss (Eq. 23). The missing component is not peripheral — it is the single largest
supervision term, and its target cannot be produced at all without the geometry route.

### Option A — `RETAIN_ROW_NOT_EXECUTED_EXTERNAL_SOURCE_GAP`

Keep E04 in the required matrix and all result tables; do not train or synthesize; report
`N/A` / `NOT_EXECUTED` because required official scientific ingredients are unavailable or
unspecified. The row is a **transparency/disclosure row** and must **not** be presented as a
completed baseline.

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Highest available.** Nothing is claimed that was not done; `FAITHFUL_PAPER` intent preserved intact; no method-changing assumption introduced. |
| Fairness | **Neutral-positive.** E04 is neither weakened nor flattered. Avoids option B's risk of shipping a degraded pseudo-Physics-STD under the original name. |
| Reproducibility | **Fully reproducible as a statement.** Any third party following M6A4's documented search reaches the same conclusion. Nothing unreproducible is published. |
| Engineering cost | **Lowest.** No implementation, assets or training; cost confined to reporting plumbing. |
| Review risk | **Low-moderate.** A reviewer may ask why a listed baseline has no numbers — fully answerable with dated evidence. Far lower than defending an undocumented substitute. |
| Effect on required matrix | Row **preserved**; status → `NOT_EXECUTED`. Because §17 says "required", this is a deviation needing `APPROVED` status per §0.1 rule 9. |
| Effect on GPAT comparison | GPAT compared against **5** executed baselines instead of 6. This **removes** a comparison point rather than creating a favorable one; it narrows breadth and must be disclosed. |

### Option B — `CONTROLLED_ADAPTATION`

Permit a reproducible substitute geometry/depth pipeline under a **new explicit technical
specification** (3DMM choice, vertex-selection rule, depth rendering, normalization, projection,
rasterization, target construction). The row **must** be labelled `CONTROLLED_ADAPTATION`, not
faithful Physics-STD. **Not designed in M6A5a.**

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Substantially reduced, exactly where it matters most.** The substitute supplies the target of the α₁ = 100 dominant term, so the row optimizes a *different* dominant objective. Unlike E05 — whose controlled adaptation resolves a contradiction between a paper's prose and its **own named executable base** — an E04 depth target would be anchored to **nothing**. |
| Fairness | **Highest-risk option, in the GPAT-favorable direction.** A benchmark-designed depth target is unlikely to match an unpublished author-designed one; the row would plausibly underperform while carrying the method's name — invisible to a reader. |
| Reproducibility | **High internally, zero externally.** Reproduces the *benchmark's invention*, not the published method. |
| Engineering cost | **Highest by a large margin.** Licensed BFM acquisition (manual + legal), 3DMM fitting, vertex selection, depth renderer, normalization/projection conventions, plus full E04 training × 3 seeds — with no authoritative acceptance test. |
| Review risk | **High.** "How do you know your depth target resembles theirs?" has no answer. An underperforming adapted baseline can reasonably be read as a straw man. |
| Effect on required matrix | Row executed but its **scientific identity changes**; requires a new additive amendment and mandatory relabelling everywhere. |
| Effect on GPAT comparison | Adds a comparison point of uncertain, probably downward-biased quality. A GPAT win over it would be weakly defensible. |

### Option C — `DEFER_M6_UNTIL_FAITHFUL_ASSETS_RESOLVED`

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Maximal in principle, unrealizable in practice.** Conditioned on assets promised "upon publication" and still unreleased, plus a licence-gated third-party model. |
| Fairness | Neutral — nothing is measured. |
| Reproducibility | Not applicable — no results produced. |
| Engineering cost | Zero now, **unbounded schedule cost**; no known resolution date. |
| Review risk | **Not a review risk but a publication blocker.** Also blocks the 11 rows with no blocker at all. |
| Effect on required matrix | Entire matrix stays unexecuted. |
| Effect on GPAT comparison | No comparison exists; GPAT itself cannot be evaluated. |

---

## 4. E07c — DIFFFAS-BIN-IDFREE

**Current:** `STILL_BLOCKED_SOURCE_GAP` ·
**`NO_OFFICIAL_ENCODER_RELEASE_LOCATED_AS_OF_2026_09_21`** ·
**Existing classification:** `CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH` (A1 / DEV-021).

Known: the official training path requires the PADISI conditioning encoder **every** step; official
weights were publicly promised by the repository **owner** on 2024-09-24 and never released (last push
2024-09-23 *predates* the promise; 0 releases, 0 tags; four follow-ups through 2025-03-10 unanswered);
PADISI-USC is not GPAT benchmark data and is itself licence-gated; the exact released encoder bytes
are unavailable. M6A4 newly established the official **training script** exists
(`models/pretrain_classifier.py`), so the protocol need not be invented — but the **17-class
ImageFolder label mapping is unpublished**, so even a faithful re-run is not uniquely reproducible.

### Option A — `RETAIN_ROW_NOT_EXECUTED_EXTERNAL_SOURCE_GAP`

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Highest available.** No undocumented component enters the pipeline; the A1 adaptation is not compounded. |
| Fairness | **Neutral-positive.** Neither weakened nor flattered; the maintainer's own acknowledgement makes the cause verifiably external. |
| Reproducibility | **Fully reproducible as a statement**, with unusually strong evidence — a dated, public, first-party acknowledgement rather than mere absence. |
| Engineering cost | **Lowest.** Reporting-layer only. |
| Review risk | **Low.** A public issue in which the repository owner says the models are forthcoming and never delivers is among the most defensible justifications possible. |
| Effect on required matrix | E07c preserved in A1's Track-A table, status `NOT_EXECUTED`. (E07a/E07b remain §17 Track-B rows with their own, separately unassessed status.) |
| Effect on GPAT comparison | Removes the diffusion comparison point from Track A. A genuine loss of breadth — diffusion baselines are topical — to be disclosed plainly, not glossed. |

### Option B — `CONTROLLED_ADAPTATION`

Authorize a benchmark-trained substitute conditioning encoder under a **new explicit specification**
(training dataset, labels/classes, architecture, initialization, optimization, checkpoint rule, seed
policy). Result **must** be labelled `CONTROLLED_ADAPTATION`, not faithful/native DiffFAS.
**Not designed or trained in M6A5a.**

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Reduced and compounded.** E07c is *already* a controlled adaptation; a substitute encoder stacks a second layer, making the composite's relationship to published DiffFAS hard to state. **Mitigating factor, recorded honestly:** the official recipe is known, so architecture/optimization need not be invented — but the training **data** would differ and the **17-class label semantics** are unpublished, so the learned conditioning space differs in an uncharacterizable way. |
| Fairness | **Moderate-to-high risk, GPAT-favorable direction.** Different data + different label space plausibly yields weaker conditioning, under the DiffFAS name. |
| Reproducibility | **High internally**; again reproduces the benchmark's construction, not the published one. |
| Engineering cost | **Moderate** — lower than E04-B because the recipe exists, but still requires choosing a substitute dataset and label space, training the encoder, then DiffFAS × 3 seeds; every choice benchmark-authored. |
| Review risk | **Moderate-high.** "What did the 17 classes become, and why is that a fair stand-in for PADISI attack types?" has no authoritative answer. |
| Effect on required matrix | Row executed with a **compounded** adaptation label; needs a new amendment and wording distinguishing it from both native and A1-adapted DiffFAS. |
| Effect on GPAT comparison | Restores a diffusion point of uncertain, probably downward-biased quality; a GPAT win over it would be weakly defensible. |

### Option C — `REMOVE_CONDITIONING_PATH_CONTROLLED_ADAPTATION`

**Execution-feasibility verdict: NOT VIABLE. Disabling the encoder fundamentally changes BOTH the
training objective and the architecture.** Evidence from the official code:

- `models/unet_autoenc.py::forward()` — when `cond is None` it calls `self.encode(x_cond, encoder)`
  and builds `cond` as an **18-entry list**: 9 **zero** tensors (`x256`, `x128`, `x64`, three each)
  **plus 9 entries of real encoder features** (`[x32x32]*3 + [x16x16]*3 + [x8x8]*3`).
- That list **is** the conditioning signal: `enc_cond_emb = cond` is injected into **every** input
  block (`cond=enc_cond_emb[k]`), `mid_cond_emb = cond[-1]` drives the middle block, and
  `dec_cond_emb = cond` drives the decoder. The encoder supplies exactly the **9 deepest**
  (32×32, 16×16, 8×8) conditioning entries.
- Removing the encoder forces those 9 entries to zero too, making `cond` **entirely zeros**;
  `mid_cond_emb` becomes `0` and the network **degenerates from a conditional to an unconditional
  diffusion model**.
- **Classifier-free guidance also collapses.** `forward_with_cond_scale()` builds its conditional and
  null branches by masking the encoder **input** (`x_cond = cond_mask * x_cond`) — the null branch is
  `encoder(0)`, not "no encoder". With the encoder removed both branches become identical, so
  `null + (logits − null) · cond_scale` reduces to `logits` and the frozen **`cond_scale = 2.0`**
  contract (resolved in M6A2 as part of the official sampler route) becomes a **no-op**.
- An unconditional model **cannot perform the benchmark's source-conditioned transfer task at all** —
  there is no path by which the source spoof style influences the output.

This is not a parameter change but a **redefinition of the model class**.

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Lowest of all options. Never faithful** under any reading — the conditioning mechanism *is* the method's scientific core. |
| Fairness | **Unacceptable risk.** The archetype of an artificially weak baseline: a deliberately crippled model under a strong method's name. Would materially favor GPAT. |
| Reproducibility | Reproducible but scientifically meaningless — it reproduces a model the authors never proposed. |
| Engineering cost | Low to implement, but the output has no scientific standing, so the cost is entirely wasted. |
| Review risk | **Severe.** A reviewer noticing the removal could regard the comparison as invalid and question the benchmark's good faith. |
| Effect on required matrix | Row executed but scientifically void; would need such heavy caveating that it conveys less than an honest `N/A`. |
| Effect on GPAT comparison | Would likely produce a large, **entirely artifactual** GPAT advantage — recorded explicitly as a reason to **reject**, not adopt. |

### Option D — `DEFER_M6_UNTIL_OFFICIAL_ENCODER_OBTAINED`

| Dimension | Assessment |
|---|---|
| Scientific fidelity | **Maximal in principle, unrealizable in practice.** The maintainer promised the asset in Sept 2024 and has not delivered in the two years since, with four unanswered requests on record. |
| Fairness | Neutral — nothing is measured. |
| Reproducibility | Not applicable. |
| Engineering cost | Zero now, **unbounded schedule cost**, with concrete evidence the condition is unlikely to resolve. |
| Review risk | **Publication blocker**, not a review risk; also blocks 5 ready baselines and all GPAT rows over 1 unresolved row. |
| Effect on required matrix | Entire matrix stays unexecuted. |
| Effect on GPAT comparison | No comparison exists. |

---

## 5. Recommendations — `INFERENCE, NOT AUTHORITATIVE`

**These are not owner decisions.** Basis: **reproducibility and scientific defensibility only.**
Expected GPAT performance was **not** a factor — and the packet records above that the GPAT-favorable
risk lies with the options **not** recommended.

### E04-2 → **Option A** (`RETAIN_ROW_NOT_EXECUTED_EXTERNAL_SOURCE_GAP`)

1. **§8.4 instructs exactly this.** If a component that *changes the scientific method* is missing,
   "mark `BLOCKED_BY_SOURCE_GAP` instead of choosing one." The missing geometry/depth route supplies
   the target of `L_depth`, whose **α₁ = 100** is the dominant loss term. This is the paradigm case
   the rule addresses.
2. **Option B has no anchor.** It would require the benchmark to author the most heavily weighted
   supervision signal of a method it did not invent, with **no authoritative artifact to validate
   against**. E05's controlled adaptation is anchored to an official executable base; an E04 depth
   target would be anchored to nothing.
3. **Option C** blocks 11 unblocked rows on 1 unresolved row, conditioned on assets unreleased since
   publication.
4. **Why N/A beats inventing a substitute:** an `N/A` cell with dated evidence makes the limitation
   **visible and checkable** — a reader can locate the exact missing ingredients. A substituted
   baseline **hides** the same limitation inside a number that looks like a measurement, and every
   downstream comparison silently inherits an unvalidatable assumption. The honest gap is also
   **recoverable**: if the authors release the assets, E04 can be executed and the row filled. A
   published fake number is not recoverable.

### E07c-4 → **Option A** (`RETAIN_ROW_NOT_EXECUTED_EXTERNAL_SOURCE_GAP`)

1. **The evidence is unusually strong** — a dated, public, first-party acknowledgement by the
   repository owner that the models were required and forthcoming, never delivered. The cause is
   verifiably external to this benchmark.
2. **E07c is already a controlled adaptation** (DEV-021). Stacking a benchmark-trained encoder on
   different data with an unpublished 17-class label space would make the row's relationship to
   published DiffFAS effectively unstateable — and A1 already forbids calling this row native or
   faithful.
3. **Option C is architecturally incoherent**, not merely undesirable (see §4C).
4. **Option D** is conditioned on an asset unreleased for two years despite four unanswered public
   requests.
5. **Why N/A beats a substitute encoder:** the substitute's conditioning space would differ from the
   authors' in a way nobody can characterize, so an E07c number would not measure DiffFAS — it would
   measure a benchmark artifact wearing the DiffFAS name, most plausibly biased **downward in GPAT's
   favor**.

### Cross-cutting

If option A is chosen for either row, the owner should also settle the §2 reporting questions — the
`N/A` token, the aggregate-exclusion rule, explicit-*n* reporting, and whether approval is recorded
via additive **Amendment A3** or an `APPROVED` `deviation_report.md` entry. Without that, §0.1 rule 9
leaves the deviation **`UNAPPROVED`** and therefore **unusable in main paper results**.

---

## 6. Owner answer template

```
E04-2:
E07c-4:
```

Valid values — **E04-2**: `A` (retain, not executed) · `B` (controlled adaptation) · `C` (defer M6).
**E07c-4**: `A` (retain, not executed) · `B` (controlled adaptation) · `C` (remove conditioning path)
· `D` (defer M6).

If `A` is chosen for either, please also indicate: **Amendment A3** or **`deviation_report.md`
APPROVED entry**.

---

## 7. Constraints observed in M6A5a

No training. No synthetic-bank generation. No TEST access. No asset download. No method
implementation. No method config created. No frozen spec or existing amendment modified. Neither
blocker resolved automatically. No historical audit artifact rewritten. Only the two permitted files
created. Not committed, not pushed.
