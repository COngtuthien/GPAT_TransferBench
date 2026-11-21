"""Synthetic qualification fixture for the E01/E02 bank writer.

Builds N synthetic TRAIN-shaped pairs over synthetic 256x256 RGB PNG "faces" in a caller-owned directory. Images
come from numpy PCG64(fixture_seed): a smooth gradient plus bounded noise, so both FAS-Aug operators and the FFT
substitution have non-trivial content. No benchmark image, manifest or runtime root is opened. The fixture's
outputs are never E01/E02 benchmark evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .bank_writer import BankInputs, PairRow, encode_png

FIXTURE_DATASETS = ("fixture_a", "fixture_b")


def synthetic_face(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
    base = np.stack([128 + 90 * np.sin((xx * rng.uniform(0.01, 0.05)) + c) * np.cos(yy * rng.uniform(0.01, 0.05))
                     for c in range(3)], axis=2)
    noise = rng.normal(0, 12, size=(256, 256, 3))
    return np.clip(np.rint(base + noise), 0, 255).astype(np.uint8)


def build_fixture(directory: str | Path, n_pairs: int, *, fixture_seed: int = 20261122) -> BankInputs:
    """Write 2*n_pairs synthetic faces under <directory>/<dataset>/<sample_id>.png and return verified inputs."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(fixture_seed))
    pairs, face_sha, split, dataset = [], {}, {}, {}
    for i in range(n_pairs):
        ds = FIXTURE_DATASETS[i % len(FIXTURE_DATASETS)]
        ids = []
        for role in ("live", "spoof"):
            sid = hashlib.sha256(f"fixture|{fixture_seed}|{i}|{role}".encode()).hexdigest()
            raw = encode_png(synthetic_face(rng), "RGB")
            (root / ds).mkdir(exist_ok=True)
            (root / ds / f"{sid}.png").write_bytes(raw)
            face_sha[sid] = hashlib.sha256(raw).hexdigest()
            split[sid], dataset[sid] = "TRAIN", ds
            ids.append(sid)
        pairs.append(PairRow(i, f"PTR{i + 1:06d}", ids[0], ids[1], ds))
    return BankInputs(pairs=pairs, faces_root=root, face_sha256=face_sha, sample_split=split, sample_dataset=dataset,
                      provenance={"kind": "SYNTHETIC_QUALIFICATION_FIXTURE", "fixture_seed": fixture_seed,
                                  "pairs": n_pairs, "faces_root": str(root),
                                  "note": "synthetic images; never benchmark evidence"})
