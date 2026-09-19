# M3 — Allocator Design (Q-01 owner resolution)

Frozen contract: `configs/frozen/split_v1.yaml`. Implementation: `gpatbench/split/allocator.py`.
Tests: `tests/test_m3_allocator.py`. Feasibility on real metadata: `M3_ALLOCATOR_FEASIBILITY.md`.

**No split exists.** This pass freezes *how* the split will be computed. `manifests/split_v1.parquet`
is not created here.

## 1. What Q-01 actually was

Spec §3.5 fixes the ratios (70/15/15), the seed (20260814), the grouping (group-disjoint inside each
dataset, then pool) and names the allocator's *terms*: "minimize absolute deviation from target video
counts plus penalties for missing binary class and attack_macro coverage". It does not give the
penalty weights, the relative importance of the terms, or the search procedure. Choosing those
silently would have been a hidden scientific decision, which is why M1 recorded Q-01 as OPEN and
`dataset_protocol_policy_v1.yaml` left `allocator.objective` explicitly `null`.

The owner resolved it as a **lexicographic** contract rather than one weighted sum, so that no
trade-off between priorities is buried in a weight nobody chose.

## 2. Population contract

Two populations are deliberately distinct:

| | |
|---|---|
| **Allocation population** | canonical videos, grouped by the dataset's allocation key. Each canonical video weighs exactly **one** video. |
| **Usable sample population** | only rows with M2 `final_status == COMPLETE`. |

M2's 25 `SCRFD_NO_FACE` failures are frame-level technical failures inside
23 otherwise healthy videos (at most
2 frames each; minimum
6 usable frames remaining, and
**0** videos left with zero usable frames). They therefore do not
change any video's group identity or its weight in the split objective — a video with 7 surviving
frames still counts as one video of that subject/content, exactly like a video with 8. Balancing on
surviving *frame counts* instead would let a detector failure quietly reweight the benchmark.

Failed rows stay in provenance (`manifests/m2_sample_accounting.parquet`) and never become TRAIN,
VAL or TEST image rows. A canonical video with zero usable frames has no owner policy, so the
allocator raises `UnresolvedPolicyError` instead of dropping it; the audit measured zero such cases,
so that path is a guard rather than a live branch.

## 3. Allocation keys (unchanged from the frozen policy)

| dataset | key | grouping | fidelity |
|---|---|---|---|
| CASIA-FASD | `subject_id_global` | SUBJECT | subject-disjoint |
| MSU-MFSD | `subject_id_global` | SUBJECT | subject-disjoint |
| SiW-Mv2 | `content_group_id` (`siwmv2::sha256:<raw sha256>`) | exact raw-content group | **canonical-video / content-group-disjoint**, not subject-disjoint (DEV-012) |

No pseudo-subject identity is created for SiW, and it is never described as subject-disjoint.

## 4. The lexicographic objective

```
P0  hard grouping / leakage feasibility        (constraint, never a penalty)
P1  total canonical-video ratio                E1 = Σ_s |100·n_s − pct_s·N|
P2  binary live/spoof distribution             E2 = Σ_c Σ_s W_c·|100·n_cs − pct_s·N_c|
P3  attack_macro distribution                  E3 = same form over attack_macro
P4  attack_raw distribution                    E4 = same form over attack_raw
P5  deterministic seeded tie-break
```

Each level is minimized, then **pinned as an exact equality constraint** before the next is
minimized, so a later priority can never worsen an earlier one. That is the whole point of the
lexicographic form: P2 cannot buy a better class balance with a worse video ratio.

`W_c = SCALE // N_c` gives every category equal weight, so a majority category cannot dominate
simply by being larger.

## 5. Normalization: why fixed point, stated honestly

The idealised objective is the rational `Σ_c Σ_s |…| / N_c`. Writing that exactly in integers needs a
common denominator of `lcm(N_c)`, which is fine for CASIA/MSU (≤450) but reaches **≈1.2·10^17** for
the SiW `attack_raw` level — beyond exact float64 representation, and numerically hostile to any
MILP backend.

The frozen objective is therefore the **fixed-point** form with `SCALE = 10^6` and `W_c = SCALE // N_c`
(floor). This is *the definition*, not a runtime approximation of something else: every coefficient
and every deviation is an integer, so each objective value is an exact integer, every coefficient is
≤ 10^6, and every observed objective is < 5·10^9 — far inside float64's exact integer range (2^53).
No rounding can occur inside the solve, and the allocator independently recomputes each objective in
exact Python integers and refuses the result if it disagrees with the solver.

What the fixed point costs is stated rather than hidden: `|E/SCALE − E_rational| < (Σ deviations)/SCALE`.
Near-ties that differ by less than that bound could order differently than the pure rational form.

## 6. Why an exact global optimum is cheap here

Groups carrying an identical multiset of `(label_binary, attack_macro, attack_raw)` are
interchangeable for every P1–P4 term, so they collapse into **profile classes** and the decision
becomes a per-class count vector. On the real data:

| dataset | allocation groups | profile classes | integer variables |
|---|---|---|---|
| casia_fasd | 50 | 1 | 3 |
| msu_mfsd | 35 | 1 | 3 |
| siwmv2 | 1,694 | 16 | 48 |

That is why HiGHS proves global optimality at every level in seconds instead of wrestling with 5,082
binaries.

## 7. Determinism

Two things could otherwise be arbitrary, and both are pinned:

1. **Which optimal count vector.** After P1–P4 are pinned, each count column is minimized one at a
   time and pinned, in an order given by
   `sha256("gpatbench.split_v1.tie.v1|canon|<dataset>|<class>|<split>|20260814")`. The solver is never
   left to pick among equally optimal vectors, and the *seed* — not the code layout — decides which
   split absorbs an unavoidable remainder. (Iterating splits in their natural order would have
   systematically starved VAL.)
2. **Which concrete groups.** Inside a class, groups are ordered by
   `sha256("gpatbench.split_v1.tie.v1|<dataset>|<group_id>|20260814")` as lowercase hex, ties broken
   by `group_id`, then handed out TRAIN → VAL → TEST.

Tests cover permuted input rows, repeated execution, and three different `PYTHONHASHSEED` values in
fresh interpreters.

## 8. Rare categories

Coverage in all three splits is **not** a hard constraint: grouping is never broken to improve it. A
category carried by fewer than three independent allocation groups simply cannot occupy three splits,
and that is reported rather than forced. On the real data no category at any level has fewer than
three groups, so this is not currently a live limitation.

## 9. What this pass deliberately did not do

No split was created, no membership was persisted, `manifests/split_v1.parquet` does not exist, and
M3 remains `NOT_STARTED`. The allocator was run against the real metadata only to measure feasibility
and optimality; the assignment it produced was discarded.
