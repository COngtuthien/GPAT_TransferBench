# M4 — Common Pair Feasibility (pre-flight, no pairs written)

Source: `manifests/split_v1.parquet` (sha256 `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d`, 20,615 usable M2 COMPLETE rows). Only COMPLETE rows can be a source or a target; the 25 M2 FAILED rows are absent from the split manifest and therefore cannot enter a pair.

Every TRAIN spoof frame is one source row, so the intended synthetic budget is **N_syn,intended = 8,838**.

| split | dataset | spoof sources | live targets | eligible min | median | max | zero-candidate | 1-63 | >=64 |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | casia_fasd | 2,520 | 840 | 816 | 816 | 816 | **0** | 0 | 2,520 |
| TRAIN | msu_mfsd | 1,200 | 400 | 384 | 384 | 384 | **0** | 0 | 1,200 |
| TRAIN | siwmv2 | 5,118 | 4,389 | 4,389 | 4,389 | 4,389 | **0** | 0 | 5,118 |
| **TRAIN** | **all** | **8,838** | | | | | **0** | | |
| VAL | casia_fasd | 576 | 192 | 168 | 168 | 168 | **0** | 0 | 576 |
| VAL | msu_mfsd | 240 | 80 | 64 | 64 | 64 | **0** | 0 | 240 |
| VAL | siwmv2 | 1,089 | 944 | 944 | 944 | 944 | **0** | 0 | 1,089 |
| **VAL** | **all** | **1,905** | | | | | **0** | | |

Eligibility rules applied:

- CASIA / MSU: same dataset, same split, target LIVE, `subject_id_global` differs.
- SiW-Mv2: same dataset, same split, target LIVE, **different canonical video AND different exact-content group** (DEV-013). This is a different-video / different-exact-content guarantee — never a different-person claim.

**Zero-candidate sources: 0.** No spoof source would have to be skipped, so there is no feasibility blocker.

### The 64-candidate cap binds everywhere

Every source in every dataset and both splits has at least 64 eligible targets (the tightest case is MSU VAL at exactly 64). The candidate sampler is therefore not a corner case: it selects the evaluated set for **every single pair**, which is why its exact rule (Q-24) blocks M4 rather than being a detail.
