# M4 — Native Pair Requirements (pre-flight)

The common manifest serves source-conditioned methods. Two methods additionally need a **native** pair structure that the common manifest does not provide, and both depend on subject identity.

## TRAIN identity coverage

| dataset | TRAIN rows | subject ids | identities | with both live+spoof | coverage |
|---|---|---|---|---|---|
| casia_fasd | 3,360 | yes | 35 | 35 | 100.0 % |
| msu_mfsd | 1,600 | yes | 25 | 25 | 100.0 % |
| siwmv2 | 9,507 | **none** | — | **0** | **0 %** |

## DSDG (spec §8.6)

> "Training live/spoof identity pairing uses only TRAIN identities that possess both live and spoof samples. Log usable identity coverage by dataset."

- CASIA: **SUPPORTED** (35/35 identities)
- MSU: **SUPPORTED** (25/25 identities)
- SiW-Mv2: **NOT_INSTANTIABLE (no subject identity, Q-14)**

SiW carries no trustworthy subject identity (Q-14), so it contributes **zero** identity pairs. The spec fixes the rule but not what a partially-covered pooled benchmark should do with it — DSDG must still 'generate exactly N_syn samples', where N_syn counts SiW sources too. Whether DSDG-native trains on CASIA+MSU identities while generating the full pooled budget, or is marked blocked for SiW, is **Q-28** (owner).

## DiffFAS (spec §8.7)

> "Native training: construct same-dataset, same-identity live/spoof reconstruction pairs from TRAIN. … If the local dataset lacks enough same-ID live/spoof pairs for a dataset, report the coverage. Do not synthesize identity labels or pair different identities for the reconstruction target merely to increase count."

- CASIA: **SUPPORTED**, 9 style_ids, guide with a different subject ALWAYS_POSSIBLE
- MSU: **SUPPORTED**, 3 style_ids, guide with a different subject ALWAYS_POSSIBLE
- SiW-Mv2: **NOT_INSTANTIABLE (no subject identity, Q-14)**, 14 style_ids, guide selection NOT_DECIDABLE (no subject identity)

Here the spec *does* say what not to do: coverage is reported and identity is never fabricated. DEV-013 resolved the **common** pairing rule for SiW and must not be stretched to cover a same-identity reconstruction requirement — that would be a different and much stronger claim. The remaining scope question (run DiffFAS-native on CASIA+MSU only, or treat it as blocked) is **Q-29** (owner).

### Binary variants

DIFFFAS-BIN collapses `style_id` into one spoof style but *preserves the reconstruction-pair structure*, and DSDG-BIN collapses the spoof-type target while keeping the rest of the architecture. Neither collapse creates identity labels, so **binary variants do not rescue SiW**: the missing information is identity, not style.

## Native manifests M4 should eventually create

| manifest | method | scope | row unit | identity metadata | supported |
|---|---|---|---|---|---|
| `dsdg_identity_pairs_v1.parquet` | DSDG / DSDG-BIN | TRAIN | one live/spoof identity pair | `subject_id_global` | CASIA + MSU only |
| `difffas_recon_pairs_v1.parquet` | DiffFAS / DIFFFAS-BIN | TRAIN | one same-ID live/spoof reconstruction pair + guide | `subject_id_global`, `style_id` | CASIA + MSU only |

Neither is created in this pass.
