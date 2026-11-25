# E04 Physics-STD bank readiness

**Status:** `E04_BANK_BLOCKED_BY_SCIENTIFIC_AUTHORITY` (10/10 record gates).

The three E04 generators are closed and verified. Bank inference is not yet defined by any authority: 6 owner questions are open, 4 of them pixel-affecting (E04-Q1_normalization_statistics, E04-Q2_synthesis_variant, E04-Q3_float_to_uint8, E04-Q5_device_and_reproducibility). The DRAFT is `configs/amendments/e04_bank_generation_contract.DRAFT.yaml` (`dc04e4845524f738`).

## Final checkpoints

| seed | run_id | checkpoint identity | data sha256 |
|---|---|---|---|
| 42 | `40befa6004aee85c` | `ff4c5065757b7169d3acafe8da56dde3636494108f0b9de9da265590b401c540` | `55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6` |
| 1337 | `a42fdd2a2accaf0a` | `01e58c806bbda7f290043e7da5764075b8462f3948b3ca73fb7f13e534b4afef` | `fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5` |
| 2026 | `e7100eb9a5ab0537` | `d18f48e4057a0c14fcfd1974e8ef15dcb9454ec61432732f259739cc97502d54` | `089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2` |

## Semantics classification

| point | class | basis | open item |
|---|---|---|---|
| generator graph and components | FROZEN | M6D3e network.generator (B, C, T, P, I_P heads; depth head unused) | implementation qualification of the inference construction |
| source-spoof and target-live inputs | FROZEN | PAPER Eq.8/13-15; training_graph.py:30-39; pairs_train_v1 roles | — |
| geometry / landmarks / auxiliary tensors | FROZEN | D1-D3 pre-pass, D2 offsets dense_offsets(source=spoof, target=live), e04_feed_pack_v1 | — |
| preprocessing | FROZEN | D8 RGB uint8/255 once; D5 no augmentation | — |
| output synthesis equation | PAPER_AND_FROZEN_GRAPH | PAPER Eq.8/13-15 realized as synthesized_spoof | E04-Q2 confirms plain vs harder-sample synthesis |
| model inference mode | OWNER_RULING | not frozen; PhySTD silent | E04-Q1 |
| batch-normalization behaviour | OWNER_RULING | frozen graph uses batch statistics (training only); predecessor test uses moving statistics | E04-Q1 |
| checkpoint restoration | FROZEN_PLUS_QUALIFICATION | BASELINE_FINAL_STATE_V1 iteration_150000; raw variables (no EMA shadows) | restore generator variables only |
| randomness and seed derivation | OWNER_RULING | none under E04-Q2 A; a new RNG domain only under E04-Q2 B | E04-Q2 |
| output tensor interpretation | FROZEN | D8 scale: RGB float32 nominally [0, 1], additive traces may leave it | — |
| float-to-uint8 conversion | OWNER_RULING | no PhySTD writer; predecessor clip + truncating cast is reference only | E04-Q3 |
| PNG encoding | OWNER_RULING | learned-bank common encoding does not yet cover E04 | E04-Q6 |
| auxiliary outputs | OWNER_RULING | spec 7 needs only RGB | E04-Q4 |
| row identities | FROZEN | spec 6: pair_id, target_live_id, source_spoof_id, dataset | — |
| expected population | FROZEN | spec 6-7: 8838 TRAIN pairs per seed, 26514 per family | — |
| determinism and replay | OWNER_RULING | D12 is a training standard only | E04-Q5 |
| S7 toy acceptance | OWNER_RULING | existing S7 authority covers E03/E05/E06b/E06c only | E04-Q5, E04-Q6 and a new E04 S7 launch authority |
| scientific fidelity disclosures | FROZEN_PLUS_NEW_DISCLOSURE | CONTROLLED_ADAPTATION; new: inference statistics and conversion provenance | disclosures follow the approved options |

## Recommended options (DRAFT, not approved)

- `E04-Q1_normalization_statistics`: `A_MOVING_STATISTICS_PER_PAIR`
- `E04-Q2_synthesis_variant`: `A_PAPER_SPOOF_SYNTHESIS`
- `E04-Q3_float_to_uint8`: `A_PREDECESSOR_CLIP_TRUNCATE`
- `E04-Q4_auxiliary_outputs`: `A_NONE`
- `E04-Q5_device_and_reproducibility`: `A_GPU_FP32_BITWISE_REPLAY`
- `E04-Q6_governance_inclusion`: `A_ADDITIVE_E04_EXTENSION`

## Storage

- `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, ext4, mode 0o700; preflight checks all true: True.
- Free 2991.1 GiB of 3666.3 GiB; 243606759 free inodes; entries ['E03', 'E05', 'E06b', 'E06c'].
- not covered: the activated storage authority registers E03, E05, E06b, E06c only, and the live preflight (bank_root_contents_ok) refuses any other top-level name, so creating <bank_root>/E04 now would block every learned bank. An additive E04 storage record plus a registry change in methods.common.learned_bank are required before any E04 directory exists.

## Existing banks

- Re-verified 2026-11-25T12:20:31Z: all 12 learned banks (8838 PNG each) and 6 E01/E02 manifests equal the final verification of 2026-11-25T11:00:09Z; six family closures `..._COMPLETED_AND_AUDITED`.

## E05

- Owner reporting decision recorded (`configs/amendments/e05_near_collapse_reporting_decision.yaml`): Retain E05 as the original RAW-MATCHED baseline with a documented near-collapse caveat. Do not retrain, filter, enhance, replace or cherry-pick its outputs.

## S7 proposal (conditional)

- status: PROPOSAL_CONDITIONAL_ON_OWNER_RULINGS
- toy population: {"manifest": "outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json", "sha256": "50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690", "pairs": 186, "rule": "the approved M8 toy pairs (TRAIN only, 0 TEST rows) for every seed; E04 is a pair method, so rows keep pair_id / target_live_id / source_spoof_id"}
- environment: gpat-m6-e03 (E04 training lock, D9 driver policy, PRECISION_ENV, qualified session configuration), approved RTX 3090; fresh clean clone from a git bundle at the S7 authority commit, never the training checkout 18ba00e
- commands after implementation: ["tools/learned_bank.py s7 --method E04 --label A --seed <s> (one OS process per seed and run; to be implemented only after approval)", "tools/learned_bank.py s7 --method E04 --label B --seed <s>", "tools/learned_bank.py s7-compare --method E04 --seed <s>"]
- runs: two independent runs (A then B) per seed, sequential, one GPU job at a time, tmux with exit markers, no automatic rerun into an existing root
- outputs: {"per_run_per_seed": 186, "total": 1116}
- reproducibility standard: per E04-Q5 (recommended A: bitwise-identical manifests and PNG bytes between runs A and B; failure stops for an owner ruling)
- checkpoint integrity: full SHA256 of the 3 files per seed before run A and after run B; no write
- atime protection: TRAIN faces read with O_NOATIME after SHA256 check; atime ledger of all TRAIN, VAL and TEST faces before and after: zero changes required
- firewall: feed-pack TRAIN closure only; any VAL/TEST id refused; no attack_macro read
- independent audit: per-seed audit on the host by a separate read-only collector and a laptop builder: population equals the toy manifest, 186 decodable 256x256 RGB PNGs, finite raw outputs, approved conversion re-applied, graph construction record, no optimizer state, reader round trip, A/B comparison
- gpu resources: single RTX 3090; inference generator at batch 1 needs well under 2 GiB; CPU warp and scipy offsets dominate (estimate 0.3-0.6 s per pair, to be measured in S7: about 1-2 min per S7 run, about 45-90 min per production seed); must not interfere with the unrelated GPU jobs present on the host
- required owner authority: ["configs/amendments/e04_bank_generation_contract.yaml (approves this DRAFT)", "configs/amendments/learned_bank_s7_launch_authority_E04.yaml"]
- production unlock: ["E04 S7 record PASS for all three seeds", "configs/amendments/learned_bank_storage_location_E04.yaml (additive registry)", "configs/amendments/learned_bank_production_E04.yaml binding the S7 record", "live storage preflight at every launch; production audit includes toy-row byte equality"]

## Gates

| gate | pass |
|---|---|
| `draft_not_approved` | PASS |
| `draft_frozen_inputs` | PASS |
| `evidence_collected` | PASS |
| `existing_banks_unchanged_since_final_verification` | PASS |
| `existing_bank_counts` | PASS |
| `production_clones_clean` | PASS |
| `six_family_closures` | PASS |
| `storage_preflight_live` | PASS |
| `storage_e04_not_silently_added` | PASS |
| `e05_reporting_decision` | PASS |

Not done: E04 adapter, E04 S7 GPU inference, E04 production, <bank_root>/E04, downstream training, TEST access, DINOv3, GPAT.
