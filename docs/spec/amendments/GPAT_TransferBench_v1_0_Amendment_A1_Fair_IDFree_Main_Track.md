# GPAT-TransferBench v1.0 — Amendment A1
## Main Fair Identity-Free Track for Common Dataset Comparison

| | |
|---|---|
| **Amendment ID** | A1 |
| **Title** | Main Fair Identity-Free Track for Common Dataset Comparison |
| **Status** | ADOPTED (owner scientific protocol amendment) |
| **Adopted** | 2026-09-20 |
| **Authority** | Project owner |
| **Amends** | `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` (sha256 `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`) |
| **Spec sections touched** | §2, §5.1, §5.2, §8.6, §8.7, §17, and the M4 acceptance rule |
| **Deviations opened** | DEV-019, DEV-020, DEV-021 |
| **Decision timing** | **Before** M5 probe training, M6 baseline training, M7 GPAT training, any synthetic generation and any inspection of any result. No model has been trained in this project and no TEST performance has ever been observed. |

---

## 1. The original specification remains historically authoritative

The frozen specification is **not edited**. It stays byte-identical and remains the authoritative
record of what was originally specified. This amendment is an additive, versioned, hashed document
that changes the **primary comparison contract** from this point forward. Every later report must
be able to distinguish "as originally specified" from "as amended by A1".

## 2. Why the amendment is required

The benchmark's primary scientific question is a **comparison between artifact-transfer methods**.
For that comparison to be fair, every compared method must be trained and evaluated on the **same
data population** under the **same supervision constraints**. The project's pooled population is:

- CASIA-FASD
- MSU-MFSD
- SiW-Mv2

Two methods in the original specification require **subject identity** to instantiate their native
training relation:

- **DSDG** (spec §8.6) — "training live/spoof identity pairing uses only TRAIN identities that
  possess both live and spoof samples";
- **DiffFAS** (spec §8.7) — "construct same-dataset, same-identity live/spoof reconstruction pairs
  from TRAIN".

**SiW-Mv2 carries no trustworthy subject identity in this project** (Q-14, measured: 0 of 9,507 SiW
TRAIN rows have a `subject_id_global`, against 35/35 CASIA and 25/25 MSU identities with both live
and spoof). This is a property of the available data, not a processing choice, and it is not
recoverable: no authoritative video→person mapping exists locally or in the official sources
(`SIW_PROTOCOL_NAME_SEMANTICS.md`).

The original M4 rule therefore forced an unacceptable choice:

- **(a)** drop SiW from DSDG and DiffFAS, which makes the primary table compare methods trained on
  *different dataset populations* — CASIA+MSU for two methods, CASIA+MSU+SiW for the rest — so any
  difference confounds method with data; or
- **(b)** fabricate identity for SiW, which is forbidden and scientifically false; or
- **(c)** block the whole benchmark on a metadata field that the primary question does not need.

M4 execution on 2026-09-20 hit exactly this wall and correctly stopped
(`BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP`, M4 `IN_PROGRESS`). The owner's resolution is
neither (a), (b) nor (c): **remove the identity-dependent assumption from the primary comparison**,
and keep identity-dependent native behaviour as a clearly labelled secondary analysis.

## 3. The amended contract: two tracks

### Track A — FAIR / COMMON / IDENTITY-FREE (**PRIMARY**)

| | |
|---|---|
| Role | The primary comparison and the primary fairness table |
| Datasets | **CASIA-FASD + MSU-MFSD + SiW-Mv2** for *every* Track-A method, with no exceptions |
| Population | The same frozen inventory, M2 usable population, M3 split, TRAIN/VAL boundaries, common source population, common pair population and downstream evaluator protocol |
| Synthetic budget | `N_syn = 8,838` (the number of TRAIN spoof frames) for every Track-A method |
| Identity supervision | **FORBIDDEN** |
| Attack-type supervision | **FORBIDDEN** (binary/adapted track) |

A Track-A method may **not** consume, in any model input, loss, class target or sampling rule:

- `subject_id_global`
- any pseudo or derived subject identity
- person labels or identity class labels derived from dataset metadata

`subject_id_global` may still appear in audit tables for provenance. It must never cross the
model-adapter boundary. Tests null CASIA/MSU subject ids at that boundary and prove the Track-A
construction is unchanged.

### Track B — NATIVE / FULL (**SECONDARY**)

| | |
|---|---|
| Role | Secondary analysis of identity-dependent native behaviour. **Not** the primary fairness table |
| Datasets | May differ per method |
| Coverage | May be partial; a missing native metadata field yields `N/A`, `NOT_INSTANTIABLE` or a controlled method-specific adaptation |

Track B is preserved, not deleted. `DSDG-NATIVE` (E06b) and `DIFFFAS-NATIVE` (E07b) keep their
definitions and move to Track B with coverage CASIA + MSU and SiW
`NOT_INSTANTIABLE_MISSING_SUBJECT_ID`. Track-B rows may never appear in a main result table without
an explicit track column.

## 4. Identity is removed, never fabricated

For SiW-Mv2 the following remain **forbidden** in both tracks:

`video_id` as person identity · `content_group_id` as person identity · any pseudo subject id ·
face-embedding clustering as subject id · filename-derived identity · manually guessed identity ·
**DEV-013 as a same-person guarantee**.

DEV-013 resolves the *common* source→target pairing rule for SiW — different video **and** different
exact-content group. It is a different-video / different-exact-content guarantee and is never
evidence that two frames show the same person or different people.

The correct fix for Track A is to **remove the identity-dependent assumption**, which is exactly
what DEV-020 and DEV-021 do.

## 5. Track-A adapted methods

Both adaptations were designed from the **official source at a pinned commit**
(`third_party/source_pins.json`), source only — no model weights were downloaded. Pinning does not
start M6 training.

### DSDG-BIN-IDFREE (E06c) — DEV-020 — `CONTROLLED_ADAPTATION`

The official `GenDataset_s` indexes spoof frames and draws the live partner with
`random.choice(make_pair_dict[label]['1'])`, where `label = video_name[4:6]` is the OULU-NPU **user
id**. The official relation is therefore same-subject, online and random.

Amendment A1 replaces that relation with the **frozen common fair pair**
(`manifests/pairs_train_v1.parquet`): each training relationship is
`source_spoof_id → target_live_id`, same dataset, same frozen TRAIN split, same common
target-selection protocol, for all three datasets.

One loss term is disabled: `lambda_pair = 0`. The official
`loss_pair = lambda_pair * MSE(rec_spoof_identity_feature, rec_live_identity_feature)` forces the
identity representations of the two reconstructions to coincide, which is only meaningful when the
two images show the same person. Optimising it across different people would be actively wrong.
All other official losses (`loss_rec`, `loss_kl`, `loss_mmd`, `loss_ip`, `loss_cls`, `loss_ort`) are
**retained** — `loss_ip` in particular compares each reconstruction with its *own* input and
therefore carries no cross-pair identity requirement. Spoof-type supervision is collapsed to a
single binary spoof class, as DSDG-BIN already specified.

### DIFFFAS-BIN-IDFREE (E07c) — DEV-021 — `CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH`

The official training step concatenates the content image into the diffusion input only when
`use_pair=True`. With `use_pair=False` the model input is `x_t` alone, the conditioning is
`style_spoof` alone, and the target is the epsilon of `GT`. The content image is then a dataset-API
placeholder with no path into the model input, the conditioning, the target, the noise or the
variational term. Amendment A1 uses that **official unpaired code path**, which removes the
same-identity content-conditioning requirement without modifying official model code.

The official style guide is drawn with `random.choice` over the GT's style folder. The benchmark
needs reproducibility, so A1 freezes **one deterministic guide per TRAIN spoof GT** by raw-byte
SHA-256 ranking over the binary spoof pool of the same dataset. `style_id = SPOOF_BINARY`;
`attack_raw` is never a class target.

Neither adaptation may be described as native or faithful. Required wording: **controlled benchmark
adaptation**, **identity-free fair-track variant**, **source-based architecture/code-path
adaptation**.

## 6. Amended M4 acceptance rule

**Original M4 rule (spec):** M4 requires the common pair manifest **and** the native pair manifests.

**Amended main-track M4 rule (A1):** M4 main-track acceptance requires

1. common TRAIN pair manifest frozen;
2. common VAL pair manifest frozen;
3. TRAIN pair statistics frozen;
4. DSDG-BIN-IDFREE Track-A training relation frozen;
5. DIFFFAS-BIN-IDFREE Track-A training relation frozen;
6. all three datasets represented in both Track-A adapted methods;
7. deterministic reruns byte-identical;
8. source pins and adaptation provenance frozen;
9. no subject-ID dependency anywhere in Track A;
10. all tests passing.

The **native** DSDG/DiffFAS Track-B manifests are **deferred to M6** secondary-method
implementation and no longer block M5.

This is a deliberate owner protocol amendment. It was **not** the original M4 rule, and the audit
trail preserves that distinction: original native requirement → blocker discovered and reported →
owner scientific objective clarified → Amendment A1 → Track-A identity-free adaptation frozen.

## 7. Claim limits

Track A **supports**: "all compared methods were trained/adapted on the same pooled CASIA-FASD +
MSU-MFSD + SiW-Mv2 TRAIN population and evaluated under the same frozen protocol."

Track A **does not support**: "DSDG-BIN-IDFREE reproduces official DSDG training exactly", or
"DIFFFAS-BIN-IDFREE is native DiffFAS", or any same-person claim about SiW pairs.

## 8. No result cherry-picking

This amendment is adopted before any probe training, any baseline or GPAT training, any synthetic
generation and any downstream result inspection. It is justified solely by protocol fairness and
data availability. No model result influenced it, and TEST rows have been used only for the
integrity counts frozen in earlier stages.

## 9. Artifacts this amendment governs

`configs/frozen/fair_track_v1.yaml` · `configs/frozen/dsdg_bin_idfree_v1.yaml` ·
`configs/frozen/difffas_bin_idfree_v1.yaml` · `manifests/difffas_bin_idfree_train_v1.parquet` ·
`third_party/source_pins.json` · `outputs/audit/M4_DSDG_IDFREE_SOURCE_ANALYSIS.md` ·
`outputs/audit/M4_DIFFFAS_IDFREE_SOURCE_ANALYSIS.md` ·
`outputs/audit/M4_TRACK_A_METHOD_MATRIX.csv`

The three previously frozen common M4 artifacts are **unchanged** by this amendment:
`pair_train_stats_v1.json` `a7ccabb0…`, `pairs_train_v1.parquet` `a5e4fdae…`,
`val_pairs_v1.parquet` `84d12491…`.
