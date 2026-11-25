# E04 Physics-STD: production readiness after S7

**S7:** `E04_S7_ALL_THREE_SEEDS_QUALIFIED` (180/180 gates, `138e1ef988970a7c8643f544b26850fbde41f7d1b63a54c86da9ccb5ac4b63a6`). **Production:** `E04_PRODUCTION_NOT_AUTHORIZED_PENDING_OWNER_ACTIVATIONS_AND_TOOLING`.

Nothing below was activated, created or launched.

## HDD state (2026-11-25T13:47:46Z)

- Bank root `/media/ext_data/student20261_gpat_m8_banks`: preflight ok True, entries {'E03': ['seed_1337', 'seed_2026', 'seed_42'], 'E05': ['seed_1337', 'seed_2026', 'seed_42'], 'E06b': ['seed_1337', 'seed_2026', 'seed_42'], 'E06c': ['seed_1337', 'seed_2026', 'seed_42']}, unexpected [], free 2.92 TiB; E04 directory exists: False.
- HDD free space fell by ~361 MB between 11:00Z and 13:47Z (shared filesystem; nothing of E04 S7 is on the HDD; the bank-root contents check is unchanged)

## Existing families

- E03/E05/E06b/E06c (12 banks: manifests, 8838 PNGs, provenance, clones, scripts, markers) and the E01/E02 production manifests identical to outputs/audit/LEARNED_BANK_PRODUCTION/final_host_verification.json except repo path, timestamp and HDD free space.
- no committed record of E01/E02/E03/E05/E06b/E06c and no E04 training/closure record changed since 2caff0d; E05 near-collapse disclosure and RAW-MATCHED decision untouched.

## Resources (measured in S7)

- 582 MiB GPU per process; [12.1, 12.1, 11.7, 11.6, 11.9, 11.7] s per 186 pairs; ~0.65 GiB PNG per seed, ~1.95 GiB for 26514 images.

## DRAFTs

- `configs/amendments/learned_bank_storage_location_E04.DRAFT.yaml` (`DRAFT_PENDING_OWNER_DECISION`): structurally complete: binds the unchanged base storage authority (f0df1b08...), the same HDD root/UUID/ext4 and the <bank_root>/E04/seed_<seed>/ layout; a copy with status OWNER_APPROVED, approved true and activation_verification.ok true loads with methods.common.learned_bank.load_e04_storage_extension (simulated in a temporary root only). Activation needs a live lb.storage_preflight on the host at activation time. Its size estimate (0.81 GB/seed, from E03) exceeds the measured E04 0.65 GiB/seed.
- `configs/amendments/learned_bank_production_E04.DRAFT.yaml` (`DRAFT_TEMPLATE_NOT_APPROVABLE_BEFORE_S7`): S7 has now passed; an activation must set s7_acceptance_record.sha256 to the LEARNED_BANK_S7_E04.json sha256 above. A copy so activated, with the activated storage extension, loads with load_production_authority("E04", seed, live_storage=False) (simulated in a temporary root only); the real repository still refuses (LEARNED_BANK_BLOCKED_BY_PRODUCTION_AUTHORITY).

## Implementation remaining before production

| item | state |
|---|---|
| E04 storage registry | IMPLEMENTED (92ef74c: load_e04_storage_extension, registered_bank_methods, production_bank_dir); needs only the owner activation record |
| production authority loader | IMPLEMENTED (load_production_authority E04 records); tools/learned_bank_production_authority.py has no E04 branch (uses CONTRACT_PATH / PAIR_METHODS of the four original methods) and must derive the E04 authority |
| production writer | IMPLEMENTED (tools/learned_bank.py production --method E04 --seed S; same adapter/allowlist path qualified by S7); tools/learned_bank_production_scripts/m8prod_launch.sh has no E04 case and lacks the E04 precision environment (load_real refuses without it) - needs an E04 launcher (the S7 launch line generator can be reused) |
| independent production audit | MISSING E04 branch in tools/learned_bank_production_audit.py: E04 checkpoints and env probe in preflight; the E04 gate set at 8838 rows (graph/BN/coverage/moving-statistics identity, conversion, PNG structure, lineage); the owner-ruling split_v1/M2 feed-pack cross-check (mandatory per e04_s7_input_allowlist_ruling.yaml) |
| full-population manifest checks | MISSING for E04: rows 0..8837 equal e04_feed_pack_v1 / pairs_train_v1, every face TRAIN with its M2 SHA256 |
| S7-to-production byte equivalence | NEEDS ADAPTATION: s7_equivalence reads outputs[seed].run_A_dir/run_B_dir; the E04 S7 record stores outputs[seed].run_dirs {A, B} (per-seed layout <S7>/E04/seed_<s>/run_<L>/E04/seed_<s>) |
| E04 family closure | MISSING E04 branch (close --method E04) and an E04 entry in tools/learned_bank_production_summary.py METHODS |

## Owner decisions remaining

1. activate configs/amendments/learned_bank_storage_location_E04.yaml (after a live storage preflight)
2. authorize implementation and qualification of the E04 production authority / launcher / audit / closure branches
3. activate configs/amendments/learned_bank_production_E04.yaml binding the S7 record sha256
4. authorize E04 production execution (3 banks x 8838 = 26514 images, one process per seed, 42 -> 1337 -> 2026)

## Observations

- outputs are non-degenerate: 186 distinct images per seed, no identical output across seeds; typical raw range about [0, 1] with clipping on roughly half the rows (expected under E04-Q3)
- seed 42 has one pair with an outlier raw range (min -1.113, max 1.677); seeds 1337/2026 stay within [-0.05, 1.10]. Disclosed, not a contract question; no pixel-level contrast diagnostic was run

Fidelity: CONTROLLED_ADAPTATION - not an official Physics-STD inference reproduction.
