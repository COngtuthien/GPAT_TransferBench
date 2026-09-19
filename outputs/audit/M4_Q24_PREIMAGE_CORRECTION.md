# Q-24 Candidate Preimage — Pre-Execution Contract Correction

**Date:** 2026-09-20 · **Classification:** `PRE_EXECUTION_CONTRACT_IMPLEMENTATION_CORRECTION`
**Base commit:** `e4d167b9ea3502cebbb97f730a5499060bb4ce26` · **M4 status:** NOT_STARTED

## 1. The authoritative contract

```
source_seed_digest_bytes =
    SHA256( UTF8("gpatbench.pair.source_seed.v1|" + source_sample_id + "|" + str(split_seed)) ).digest()
        -> RAW 32 BYTES, never rendered as hex / base64 / decimal / text

candidate_rank_digest =
    SHA256( source_seed_digest_bytes || UTF8("|gpatbench.pair.candidate.v1|" + target_sample_id) ).digest()

rank key = ( int.from_bytes(candidate_rank_digest, "big", signed=False), target_sample_id )   ascending
keep     = first 64
```

The second element of the rank key is a defensive tie-break against a digest collision, not a
routine path.

## 2. What was actually wrong

The owner's Q-24 contract **did not change**. Checking each carrier of the contract against the
text above:

| Carrier | State at `e4d167b` | Action |
|---|---|---|
| `gpatbench/pairs/common.py` | **Already correct.** `candidate_rank_digest()` hashes `seed_digest + UTF8("\|" + namespace + "\|" + target_sample_id)` where `seed_digest` is `hashlib.sha256(...).digest()` — raw bytes. | none needed |
| `tests/test_m4_pair_preflight.py` | **Already correct.** `test_frozen_digest_construction` rebuilt both preimages by hand from `hashlib` and asserted equality. | strengthened (§4) |
| `configs/frozen/pairs_v1.yaml` | **Already correct but under-specified in prose:** it stated the concatenation and `output: raw_32_byte_digest`, but did not label the source-digest *representation* or name the forbidden variant. | corrected (§5) |
| M4 owner decision report §3 (chat deliverable, 2026-09-19) | **Wrong.** Summarised the rule as "candidate digest over (namespace, source_seed_hex, target_sample_id)". That prose describes a different function from the one implemented and frozen. | corrected here |

So the defect was a **reporting/specification-wording defect**, not an implementation defect. The
forbidden hex formulation was never present in executable code: a repository-wide search for
`source_seed_hex`, `seed_hex`, `hexdigest` and `lowercase_hex` across `gpatbench/`, `configs/`,
`tests/` and `outputs/audit/` returns no Q-24 hit in any module or test. The only two textual
occurrences are deliberate and non-executable:

- `configs/proposed/pairs_v1.proposed.yaml` — the **pre-decision proposal**, preserved byte-unchanged
  as history. It proposed a different rule again (a single-stage
  `SHA256(namespace|source_sample_id|target_sample_id|split_seed)` with a `lowercase_hex` digest),
  which the owner's two-stage raw-byte contract superseded. History is not rewritten.
- `configs/frozen/pairs_v1.yaml` — the new `forbidden_variants` block, which states the hex
  formulation **in order to forbid it** and points at the test that keeps it out.

There was consequently no second variant to remove from the codebase and no flag to delete.

## 3. Measured candidate-membership impact

`tools/m4_q24_preimage_audit.py`, on the same deterministic sample the frozen diagnostic uses
(salt `gpatbench.m4_frozen_diag.v1|`, 40 sources per dataset × split, 240 sources total):

| cell | sources | sets changed vs `e4d167b` | sets changed vs forbidden hex variant |
|---|---|---|---|
| casia_fasd TRAIN | 40 | 0 | 40 (100%) |
| casia_fasd VAL | 40 | 0 | 40 (100%) |
| msu_mfsd TRAIN | 40 | 0 | 40 (100%) |
| msu_mfsd VAL | 40 | 0 | 0 (0%) |
| siwmv2 TRAIN | 40 | 0 | 40 (100%) |
| siwmv2 VAL | 40 | 0 | 40 (100%) |
| **total** | **240** | **0 (0.0000%)** | **200 (83.3333%)** |

The comparison against `e4d167b` is not an assertion: the tool imports
`gpatbench/pairs/common.py` **as committed at that SHA** (blob sha256
`8f2faaa3feeefad3e86b5309df4adc0ec716ec783e69f80a7495c32eb2eb48ee`) and re-runs the selection.

- **0 / 240** is the real effect of this pass. Candidate membership is unchanged because the
  implementation already matched the owner contract.
- **200 / 240** is a counterfactual on a formulation that was never implemented. It quantifies what
  the ambiguous prose would have cost had anyone coded from it, which is why the variant is now
  named and test-guarded rather than merely avoided.
- MSU VAL is 0% for a structural reason, not a hash reason: every MSU VAL source has **exactly 64**
  eligible targets, so all 64 are kept under any ranking. The cap is reached there but is not
  selective.

No membership was persisted. Nothing in this measurement was allowed to influence the frozen rule.

## 4. Test guard added

`TestQ24Preimage` (in `tests/test_m4_pair_preflight.py`) now:

1. rebuilds `source_seed_digest` byte-for-byte from `hashlib` for a fixed fixture and asserts
   equality, including that the result is exactly 32 `bytes` and not `str`;
2. rebuilds `candidate_rank_digest` as `sha256(expected_source_digest_bytes + expected_suffix_utf8)`
   and asserts equality;
3. asserts the forbidden hex formulation yields a **different** digest and a **different** rank key
   for the same fixture;
4. asserts the forbidden formulation appears nowhere in `gpatbench/`;
5. re-asserts permutation invariance of the selected 64 after the correction.

This test exists specifically so the two formulations cannot be confused again.

## 5. Config supersession

| | SHA256 |
|---|---|
| previous `configs/frozen/pairs_v1.yaml` | `16ff68036a9ed3a315df6944697394359cb2ea53e0b5623721d036a3cf170398` — **SUPERSEDED_BY_Q24_PREIMAGE_CORRECTION**, retained in audit history |
| corrected `configs/frozen/pairs_v1.yaml` | `f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985` |
| `frozen_config_snapshot/…/pairs_v1.yaml` | `f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985` (byte-identical) |

Content changes: explicit `source_digest_representation: RAW_32_BYTE_SHA256_DIGEST`, a part-by-part
`preimage_parts` breakdown with the encoding of each part, a `forbidden_variants` block naming both
`HEX_TEXT_SOURCE_SEED` and `AMBIGUOUS_PROSE`, and the new Q-28/Q-29 `native_manifests` scope
section. **Q-25, Q-26, Q-27, the weights, the cap, the seed, the tie-break and DEV-018 are
byte-unchanged in meaning and were not reopened.**

## 6. Why no scientific result is invalidated

- The owner contract did not change.
- The implementation already matched it, and this is proven by re-execution against the committed
  blob, not asserted.
- No authoritative pair manifest, no `pair_train_stats_v1.json` and no native manifest had been
  created — M4 execution has not started.
- Therefore no frozen pair membership, no distance statistic and no downstream result existed that
  could be invalidated.

This is not a new scientific decision and must not be recorded as one. It is a pre-execution
correction of how an already-approved contract was written down.
