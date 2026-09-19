# M2A Determinism Report — frozen contract (2026-09-19)

- **Contract:** same host (laptop, CPU), same M2 environment (`environments/m2_preprocess_laptop.lock.txt`),
  same code, same manifest, same models, 4 torch intra-op threads, 1 ORT thread,
  `torch.use_deterministic_algorithms(True)`, `inference_mode`, `eval()`.
- **Method:** two separate processes (`runC`, `runD`) start from clean output directories, with the FINAL
  code of this pass (SCRFD selection, AdaFace original/BGR adapter, Q-22 square zero padding, Q-23 lossless
  logit shards). Every produced file is compared byte-wise (`tools/m2a_smoke.py compare`), and the results
  table is compared with a real CSV parser, excluding only wall-clock columns.
- The previous pass's `run3`/`run4` comparison is **not** reused as proof, because the contract changed.

| Artifact | Files | Byte-identical |
|---|---|---|
| Frame PNGs (MSU/SiW) | 16 | 16 |
| Canonical face PNGs | 24 | 24 |
| FaceXFormer arrays (.npy: parsing_logits, parsing_mask, landmarks ×3, pose) | 144 | 144 |
| AdaFace embeddings (.npy) | 24 | 24 |
| Parsing-logit shard (`parsing_logits-00000.bin`, 24 compressed blocks) | 1 | 1 |
| Shard index (`logit_shards/index.csv`: shard, offset, length, block/npy sha256) | 1 | 1 |
| **Total** | **210** | **210** |
| Results table (SCRFD metadata, crop/padding metadata, geometry, identity, storage flags; timing excluded) | – | identical |

- **Max float difference:** 0 — exact byte identity, so `max_abs_diff` is empty and no numeric comparison
  was needed. Decompressed float32 parsing logits are covered twice: the shard bytes are identical, and each
  run independently verified `np.array_equal(original, decoded)` plus `.npy` byte equality per sample.
- **Final status: PASS** (same host/device/environment). Machine-readable: `M2A_DETERMINISM_COMPARE.json`.
- **Cross-device:** not tested and not claimed. Full M2B on another host or device must repeat this smoke and
  lock (rule 35). Laptop hashes are not evidence of remote determinism.
- **Tool correction:** the comparison helper previously split the results CSV on commas, which mis-aligns the
  JSON geometry columns and so failed to exclude the timing columns. It now uses `csv.DictReader`. This
  affected only the results-table check, never the file-level byte comparison. Recorded in the ledger.
