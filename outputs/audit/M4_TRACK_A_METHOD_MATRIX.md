# M4 — Track-A Fairness Matrix (Amendment A1)

**Date:** 2026-09-20 · **Violations: 0** · Machine-readable: `M4_TRACK_A_METHOD_MATRIX.csv`
Generator: `tools/m4_track_a_matrix.py` (fails non-zero on any violation).

Every Track-A method trains and validates on the **same pooled population** and carries the **same
synthetic budget**. Identity and attack-type supervision are forbidden for all of them.

| id | method | TRAIN datasets | VAL datasets | subject ID | attack type | source-conditioned | common pair | N_syn |
|---|---|---|---|---|---|---|---|---|
| E01 | FAS-Aug | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | target_live_id; pair_id seeds the operator (§8.1) | 8,838 |
| E02 | Frequency Substitution | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | source and target | 8,838 |
| E03 | STDN | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | source and target | 8,838 |
| E04 | Physics-Guided STD | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | source and target | 8,838 |
| E05 | PCGAN | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | source and target | 8,838 |
| **E06c** | **DSDG-BIN-IDFREE** | CASIA+MSU+SiW | CASIA+MSU+SiW | **NO** | **NO** | NO | **the common TRAIN pair IS its training relation** | 8,838 |
| **E07c** | **DIFFFAS-BIN-IDFREE** | CASIA+MSU+SiW | CASIA+MSU+SiW | **NO** | **NO** | YES | generation `target_live_id` + `source_spoof_id` | 8,838 |
| E08 | GPAT-B0 | CASIA+MSU+SiW | CASIA+MSU+SiW | NO | NO | YES | source and target | 8,838 |

## Notes on the two structurally different rows

- **E06c (DSDG-BIN-IDFREE)** is a *generative* method: after training it samples from the latent
  prior, so there is no per-output source→target relation. It is therefore marked
  `source_conditioned = NO`. Its **generation budget is still exactly 8,838**, and its *training*
  relation is the common TRAIN pair manifest itself, so it shares the population and the pairing
  protocol with every other Track-A method. Pairwise target-to-synthetic identity metrics remain
  N/A for it, exactly as spec §8.6 already said.
- **E07c (DIFFFAS-BIN-IDFREE)** trains from its own Track-A manifest
  (`difffas_bin_idfree_train_v1.parquet`: GT spoof + deterministic binary style guide) but
  **generates** against the same common pair universe as every other source-conditioned method. It
  may not substitute another target population.

## Methods deliberately outside Track A

| id | method | why not Track A |
|---|---|---|
| E00 | Real-only | downstream lower bound; no synthetic generation |
| E06a | DSDG-BIN | collapses only the spoof-type target and still inherits the official same-subject pairing — superseded for Track A by E06c |
| E07a | DiffFAS-BIN | collapses `style_id` but keeps the same-ID reconstruction structure — superseded for Track A by E07c |
| E06b | DSDG-NATIVE | Track B; needs subject identity |
| E07b | DiffFAS-NATIVE | Track B; needs subject identity |
| E09 | GPAT-B1 | uses attack-type supervision (λ_type = 0.2) |
| E10 | GPAT-B2 | uses an identity adversary (λ_idadv = 0.1) |
| E11 | GPAT-B3 | uses both |

Track-B and ablation rows may never appear in a main result table without an explicit track column.
