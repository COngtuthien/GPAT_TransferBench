# E06c DSDG-BIN-IDFREE: three-seed scientific training closure

**Status: `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`.** All 62 family gates pass. Evidence: `E06C_THREE_SEED_TRAINING_CLOSURE.json` and `E06C_THREE_SEED_FINAL_CHECKPOINTS.csv`, built by `tools/e06c_three_seed_training_closure.py`. The build re-derives each per-seed `E06C_SEED<s>_COMPLETION.json` with its own closure tool, checks those files byte-for-byte against the per-seed closure commits, and takes a read-only host recheck (`tools/e06c_family_host_recheck.py`). It embeds that recheck, so the family evidence re-derives from committed files alone.

**Scope:** E06c *generator scientific training* is complete for the three frozen experiment seeds. This status does **not** mean the synthetic bank is generated, that downstream FAS evaluation is done, or that the benchmark is complete. None of those has started.

| Item | Value |
|---|---|
| Method | E06c DSDG-BIN-IDFREE |
| Fidelity / deviation | CONTROLLED_ADAPTATION / DEV-020 |
| Scientific training authority (all three seeds) | `84809868acf472af5ba8cf6ab47165ae038e1012`. The GPU checkout was clean at this commit for every launch, and `git_dirty` is false. |
| Config SHA256 | `7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002` |
| Frozen spec SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| TRAIN relation SHA256 | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| Execution | GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2: nominal global 240, max microbatch 20, drop_last false; 200 epochs × 37 steps = 7400 optimizer steps; 36 × 240 + 198 = 8838 TRAIN pairs per epoch |
| Checkpoint rule | `OFFICIAL_GENERATOR_EPOCH_200`, applied independently within each seed |

## Final checkpoints (one row per seed)

| Seed | run_id | Final netG (`checkpoints/netG_model_epoch_200_iter_0.pth`, 79 577 600 B) SHA256 | metrics.jsonl SHA256 | Run-root listing digest | Seed closure |
|---|---|---|---|---|---|
| 42 | `27dcaf23aac18d7a` | `665684fddf11ec28b95e2a98122c85bfc5352bb8ea69ecb60caac9418c0871d9` | `12fb184b…` | `31609e65…` | `cf5ed6c` |
| 1337 | `49a179a4dca77f41` | `43ffa30dc7eaa568bc0b573a34978f897f46156232ee1a7e71d7f994beb8f336` | `c06ec5c0…` | `87a0bed5…` | `9281c79` |
| 2026 | `86b631403e3c4ee1` | `dc6c54b4357a51546c3abab3b957dd4b19044ae0188dde417df03446fc951878` | `0ee1d2a7…` | `eda61bb7…` | `81a8ef6` |

For all three seeds: completion_status is completed, epoch 200, step 7400, launch authority 84809868, TEST access 0, and VAL selection 0. Full digests are in the CSV.

## Cross-seed consistency
- **Common, verified equal:** method, fidelity, deviation, authority, config SHA, frozen spec SHA, TRAIN relation SHA, and the full `identities` block. That block covers the source commit/tree pin, environment lock, LightCNN SHA, logging contract, A1 adaptation and the M6D5c/d/e overlays. Also equal: dependency fingerprint and environment `gpat-m6-e06c` (torch 2.12.1+cu130), the M6D5e file hashes, the V2 execution mode, the batch semantics, 200 epochs / 7400 steps, the 21-event checkpoint schedule, the checkpoint rule and the firewall policy. Every `run_manifest` field is byte-equal across seeds except those that differ by design.
- **Different by design:** experiment seed, run_id (= sha256("E06c|seed|config|authority")[:16]), run_uuid, start/end time, the `--seed` argument, data shuffle and random draws (all three metrics streams differ), and the learned checkpoint bytes (all three final netG hashes are distinct).
- `resolved_config.yaml` differs only in three seed-derived lines: `experiment_seed`, `run_dir` (`…/seed_<s>`) and `generator_seed`. This was checked from the read-only host text, and each text's SHA256 equals its run manifest.
- Gate semantics are identical. The three per-seed closure tools each run the same 86 gates; the only difference is the run-root parent-listing gate, which reflects which seeds existed at each closure.

## Family composition and selection
- The family is exactly seeds {42, 1337, 2026}: 3/3 completed scientific runs. There is no fourth, replacement, best, preferred or dropped seed. The host's `runs/m6/E06c/` holds exactly these three roots.
- No seed is selected. The final checkpoint is the official epoch-200 netG, chosen independently within each seed by the frozen rule. No VAL-based selection exists at any level.
- Future applicable evaluation metrics will be reported under the frozen primary result rule `MEAN_AND_STD_OVER_ALL_THREE_SEEDS`. No performance metric exists yet. The per-seed epoch-mean training losses in the JSON (total_loss at epoch 200: 774.2 / 749.0 / 748.3) are descriptive only and are not performance claims. Wall-clock time is not a gate or a comparison metric.

## Firewall
- **TEST was not used.** In every seed's run, all 3027 TEST faces were relatime-sensitive for the whole window and their atime was unchanged. `test_split_accessed` is false.
- **VAL was not used for checkpoint selection.** No run had a VAL pass, `val_split_accessed` is false, and all 3121 VAL faces were unchanged in every run.
- The upstream code block named "test" writes TRAIN-sample visualization PNGs. It is not benchmark TEST.

## Immutability recheck at family closure (2026-10-21T16:41Z, read-only)
- For all three run roots, the final netG SHA256 and size, the metrics.jsonl SHA256, the 219-file count and the C-collation listing digest (`find . -type f -printf "%P %s %T@\n" | LC_ALL=C sort | sha256sum`) equal the per-seed closure evidence. The digests were stable through the recheck, so the seed42 and seed1337 roots remain exactly as closed.

## Confirmations
- No synthetic bank generation, downstream evaluation, E06b, other baseline, GPAT work or additional E06c seed was started.
- No checkpoints, images or runtime logs are committed. E06c remains CONTROLLED_ADAPTATION / DEV-020.
