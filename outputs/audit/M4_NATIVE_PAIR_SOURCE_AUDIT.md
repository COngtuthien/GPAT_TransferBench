# M4 — Native Pair Construction: Source-of-Truth Audit

> **SUPERSEDED IN PART by Owner Protocol Amendment A1 (2026-09-20).** The blocker described below
> was real and is preserved as history. Two things have since changed: the owner **authorized
> pinning** both repositories (FaceX-Zoo `16b793a7564a4b9308cf94e62bdb2ffacb3a725a`,
> murphytju/DiffFAS `23f40519ec25a833ebc06842aa6fbab74fad4d15`; see `third_party/source_pins.json`),
> so §2 and §6 option (A) are now resolved; and the owner **removed the native manifests from the
> main-track M4 dependency** altogether. DSDG-NATIVE and DiffFAS-NATIVE are now **Track B
> (SECONDARY)** and deferred to M6; they no longer block the main comparison or M5. The questions in
> §3 remain open and will be answered from the now-pinned source during M6. Nothing below was
> deleted or rewritten.
>
> Current status: `M4_FAIR_TRACK_AMENDMENT_REPORT.md`, `M4_NATIVE_PAIR_COVERAGE.md`.

**Date:** 2026-09-20 · **Outcome:** `BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP`
**Affected manifests:** `manifests/dsdg_identity_pairs_v1.parquet`,
`manifests/difffas_recon_pairs_v1.parquet` — **neither created**
**Not affected:** the common TRAIN/VAL pair manifests and `pair_train_stats_v1.json`, which are
complete, audited and frozen.

## 1. What the source-of-truth rule requires

The construction hierarchy for a native manifest is, in order:

1. official repository **at a pinned commit**
2. official supplementary material
3. paper architecture / equations
4. already-approved benchmark adaptation

Level 4 (`configs/frozen/pairs_v1.yaml`, Q-28/Q-29) fixes **which datasets may appear** and **what a
row means**. It does not define *how many* rows exist, *which* live frame pairs with *which* spoof
frame, or whether the official pipeline enumerates a finite pair set at all. Those are
execution-affecting choices, and the frozen config explicitly does not grant permission to invent a
new stochastic pairing algorithm.

## 2. What the project has actually registered

`third_party/registry.yaml`, read verbatim:

| method_id | method | official_repo | pinned_commit | url_verification | local source |
|---|---|---|---|---|---|
| E06b | DSDG-NATIVE | `https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DSDG` | **null** | UNVERIFIED | none |
| E07b | DiffFAS-NATIVE | `https://github.com/murphytju/DiffFAS` | **null** | UNVERIFIED | none |

Verified on disk:

- `methods/dsdg/` contains only `.gitkeep`. `methods/difffas/` contains only `.gitkeep`.
- `third_party/` contains only `registry.yaml`; no repository is vendored.
- A filesystem search of `/home/cong` and `/media/cong/Data` finds no FaceX-Zoo, DSDG or DiffFAS
  checkout.
- The registry's own header states the rule: *"pinned_commit: null until pinned at first setup
  (**M6**, spec §8.1 'pin the repository commit on first setup')"*, and
  *"url_verification: UNVERIFIED = copied from spec, remote not contacted"*.

**There is therefore no pinned official source to inspect.** Level 1 of the hierarchy is empty, and
level 2 and 3 cannot substitute for it here: the questions below are about data-loader behaviour,
which papers do not specify.

## 3. Questions that cannot be answered without the source

### DSDG (prompt §29)

| Question | Answerable now? |
|---|---|
| What exactly constitutes a live/spoof identity training tuple? | **No** |
| Is pairing materialized once, or sampled online per batch/iteration? | **No** |
| How many tuples / iterations are expected? | **No** |
| Are all live × spoof combinations used? | **No** |
| May a live or a spoof frame repeat? | **No** |
| Is there a random pairing procedure, and what seed controls it? | **No** |

### DiffFAS (prompt §32)

| Question | Answerable now? |
|---|---|
| Reconstruction-pair sampling semantics | **No** |
| Guide-spoof sampling semantics | **No** |
| Style-pool construction | **No** |
| Reuse rules | **No** |
| Online / random behaviour and seed behaviour | **No** |
| Can a finite deterministic manifest faithfully encode the official relationship? | **No** |

Spec §8.7 does constrain the *guide*: "sampled from the same `style_id` and a different subject
when possible". It does not define the reconstruction-pair enumeration, the per-identity row count,
or the fallback when no different-subject guide exists. Prompt §33 requires any fallback to be
"official/source-supported only", and no such source is available to support one.

Answering any of these from the method name, from the paper alone, or from a plausible-looking
convention is exactly what §32 and §34 forbid.

## 4. What was deliberately NOT done

Per §34, none of the following was used to make M4 pass:

- first-lexical-sample selection
- random sampling under a project-chosen seed
- the full live × spoof Cartesian product
- one-to-one matching
- cyclic matching
- reusing the common-pair Q-24 hash ranking for a native manifest it was never frozen for

No native manifest file was created, not even an empty one.

## 5. What is not the blocker

- **Not connectivity.** Outbound HTTPS works (`https://github.com` returned HTTP 200 on
  2026-09-20). The gap is a provenance decision, not a network failure.
- **Not dataset coverage.** CASIA (35/35 identities) and MSU (25/25) fully support the row
  semantics; see `M4_NATIVE_PAIR_COVERAGE.md`.
- **Not Q-28/Q-29.** Those are resolved. They fixed dataset scope, which is a necessary but not
  sufficient input to construction.

## 6. What would unblock it

An owner decision on **one** of these, recorded before construction:

- **(A)** Authorize pinning the two repositories now — pulling the M6 "pin the repository commit on
  first setup" action forward into M4. Then `pinned_commit` and `url_verification` are filled in,
  the data-loader semantics in §3 are read from the pinned code, and the manifests are materialized
  faithfully or the construction is shown to be irreducibly online.
- **(B)** Keep pinning at M6 as the registry currently states, leave M4 at
  `IN_PROGRESS / COMMON_PAIRS_COMPLETE_NATIVE_PAIR_SOURCE_BLOCKED`, and construct both native
  manifests during M6 once the environments are created.

Option (A) is the smaller change if native rows are wanted before M6; option (B) matches the
project's existing frozen plan. Either way the common pair artifacts stand unchanged.

## 7. SiW-Mv2 is a separate, already-settled matter

SiW-Mv2 is `NOT_INSTANTIABLE_MISSING_SUBJECT_ID` for both methods under Q-28/Q-29 and would remain
so even with the sources pinned: it carries no trustworthy subject identity (Q-14), and identity is
never fabricated. `content_group_id` and `video_id` are not identity, and DEV-013 is a
different-video / different-exact-content guarantee for the **common** rule only — never
same-identity evidence. SiW coverage is recorded as 0 in the coverage table, never as manifest rows.
