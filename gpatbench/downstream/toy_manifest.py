"""M8 smoke toy manifest: the exact frames (with M2 face hashes) of every role and the frozen toy pairs.

Two builders with one output schema:
- build_real: the approved toy plan (toy_subset.plan) plus TRAIN/VAL metadata and M2 face hashes, each read through a
  parquet row filter restricted to the needed sample ids, so no TEST row reaches Python. Face files are only stat-ed
  (existence), never opened here.
- build_synthetic_world: a synthetic qualification world (synthetic PNG faces written under a caller-owned root) with
  the same schema, sized so the sampler remainder/reshuffle path is exercised. Never benchmark evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import toy_subset
from .firewall import read_permitted_rows

SCHEMA = "m8_toy_manifest_v1"
M2_PATH = "manifests/m2_sample_accounting.parquet"
M2_SHA256 = "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662"
ROLE_KEYS = {"train_live": "TRAIN/train/0", "train_spoof": "TRAIN/train/1", "val_select_live": "VAL/select/0",
             "val_select_spoof": "VAL/select/1", "val_holdout_live": "VAL/holdout/0", "val_holdout_spoof": "VAL/holdout/1"}
FRAME_FIELDS = ("sample_id", "dataset", "subject_id_global", "content_group_id", "video_id", "frame_index", "y_true",
                "attack_raw", "attack_macro", "face_rel_path", "face_sha256")


class ToyManifestError(RuntimeError):
    pass


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _m2_hashes(root: Path, ids: list[str]) -> dict:
    import pyarrow.parquet as pq
    p = root / M2_PATH
    if hashlib.sha256(p.read_bytes()).hexdigest() != M2_SHA256:
        raise ToyManifestError("M2 accounting SHA256 differs")
    t = pq.read_table(p, columns=["sample_id", "face_png_sha256", "final_status"],
                      filters=[("sample_id", "in", ids)]).to_pylist()
    out = {}
    for r in t:
        if r["final_status"] != "COMPLETE":
            raise ToyManifestError(f"M2 face not COMPLETE: {r['sample_id']}")
        out[r["sample_id"]] = r["face_png_sha256"]
    if set(out) != set(ids):
        raise ToyManifestError("toy samples missing from M2")
    return out


def build_real(root: str | Path, faces_root: str | Path) -> dict:
    root, faces_root = Path(root), Path(faces_root)
    plan = toy_subset.plan(root)
    if plan["shortfalls"]:
        raise ToyManifestError(f"toy plan shortfall {plan['shortfalls']}")
    rows = read_permitted_rows(root, ["sample_id", "dataset", "subject_id_global", "content_group_id", "video_id",
                                      "frame_index", "label_binary", "attack_raw", "attack_macro"])
    unit = lambda r: r["content_group_id"] if r["dataset"] == "siwmv2" else r["subject_id_global"]
    roles = {}
    for name, key in ROLE_KEYS.items():
        sp = key.split("/")[0]
        c = int(key.split("/")[2])
        chosen = {(ds, u) for ds, d in plan["roles"][key].items() for u in d["units"]}
        roles[name] = sorted((r for r in rows if r["split"] == sp and r["label_binary"] == c
                              and (r["dataset"], unit(r)) in chosen), key=lambda r: r["sample_id"])
    train_ids = {r["sample_id"] for r in rows if r["split"] == "TRAIN"}
    sources = sorted({p["source_spoof_id"] for p in plan["toy_pairs"]} | {p["target_live_id"] for p in plan["toy_pairs"]})
    if not set(sources) <= train_ids:
        raise ToyManifestError("a toy pair references a non-TRAIN sample")
    meta = {r["sample_id"]: r for r in rows if r["sample_id"] in set(sources)}
    needed = sorted({r["sample_id"] for v in roles.values() for r in v} | set(sources))
    m2 = _m2_hashes(root, needed)

    def frame(r):
        f = {"sample_id": r["sample_id"], "dataset": r["dataset"], "subject_id_global": r["subject_id_global"],
             "content_group_id": r["content_group_id"], "video_id": r["video_id"], "frame_index": r["frame_index"],
             "y_true": r["label_binary"], "attack_raw": r["attack_raw"], "attack_macro": r["attack_macro"],
             "face_rel_path": f"{r['dataset']}/{r['sample_id']}.png", "face_sha256": m2[r["sample_id"]]}
        if not os.path.isfile(faces_root / f["face_rel_path"]):        # stat only; no open, no atime change
            raise ToyManifestError(f"face file missing: {f['face_rel_path']}")
        return f

    out = {"schema": SCHEMA, "kind": "REAL_TOY", "label": "SMOKE_NON_SCIENTIFIC",
           "siwmv2_unit_disclosure": plan["siwmv2_unit_disclosure"],
           "roles": {k: [frame(r) for r in v] for k, v in roles.items()},
           "pair_samples": [frame(meta[s]) for s in sources],
           "toy_pairs": [{k: p[k] for k in ("position", "pair_id", "dataset", "target_live_id", "source_spoof_id")}
                         for p in plan["toy_pairs"]],
           "plan_sha256": canonical_sha256({k: plan[k] for k in ("roles", "pooled", "toy_pairs", "feasibility")}),
           "test_rows_read": 0}
    out["allowlist"] = sorted({f["face_rel_path"] for v in out["roles"].values() for f in v}
                              | {f["face_rel_path"] for f in out["pair_samples"]})
    return out


def build_synthetic_world(world_root: str | Path, *, seed: int = 20261122, n_pairs: int = 20) -> dict:
    """Synthetic faces + manifest. Roles: 12 live / 7 spoof TRAIN videos (56 frames: pools of 16 and 32 leave remainders); VAL select and holdout 6 live / 8 spoof videos
    each; 8 frames per video; three datasets; siwmv2 spoof videos carry real attack_map tokens."""
    from methods.common.bank_fixture import synthetic_face
    from methods.common.bank_writer import encode_png
    world_root = Path(world_root)
    faces = world_root / "faces_256"
    faces.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    datasets = ("casia_fasd", "msu_mfsd", "siwmv2")
    macro = {"casia_fasd": ("print", None), "msu_mfsd": ("replay", None), "siwmv2": ("replay", "Replay")}
    siw_tokens = [("Replay", "replay"), ("Paper", "print"), ("Makeup_Cosmetic", "makeup"), ("Partial_Eye", "partial")]
    roles = {k: [] for k in ROLE_KEYS}
    plan = {"train_live": 12, "train_spoof": 7, "val_select_live": 6, "val_select_spoof": 8,
            "val_holdout_live": 6, "val_holdout_spoof": 8}
    vid = 0
    for role, n_videos in plan.items():
        y = 0 if role.endswith("live") else 1
        for v in range(n_videos):
            ds = datasets[v % 3]
            vid += 1
            if y == 0:
                am, ar = "live", None
            elif ds == "siwmv2":
                ar, am = siw_tokens[v % len(siw_tokens)]
            else:
                am, ar = macro[ds]
            for fi in range(8):
                sid = hashlib.sha256(f"synthetic|{seed}|{role}|{v}|{fi}".encode()).hexdigest()
                arr = synthetic_face(rng)
                if y == 1:                                   # a weak class signal so training has something to fit
                    arr = np.clip(arr.astype(np.int16) + 12, 0, 255).astype(np.uint8)
                raw = encode_png(arr, "RGB")
                (faces / ds).mkdir(exist_ok=True)
                (faces / ds / f"{sid}.png").write_bytes(raw)
                roles[role].append({"sample_id": sid, "dataset": ds,
                                    "subject_id_global": None if ds == "siwmv2" else f"{ds}::syn{role}{v}",
                                    "content_group_id": f"{ds}::cg{vid}", "video_id": f"synvid{vid:04d}",
                                    "frame_index": fi, "y_true": y, "attack_raw": ar, "attack_macro": am,
                                    "face_rel_path": f"{ds}/{sid}.png", "face_sha256": hashlib.sha256(raw).hexdigest()})
    for k in roles:
        roles[k].sort(key=lambda r: r["sample_id"])
    positions = sorted(rng.choice(8838, size=n_pairs, replace=False).tolist())
    live = roles["train_live"]
    spoof = roles["train_spoof"]
    pairs = []
    for i, pos in enumerate(positions):
        t = live[(i * 7) % len(live)]
        same = [x for x in spoof if x["dataset"] == t["dataset"]]          # frozen pairs are within one dataset
        s = same[(i * 5) % len(same)]
        pairs.append({"position": int(pos), "pair_id": f"PTR{pos + 1:06d}", "dataset": t["dataset"],
                      "target_live_id": t["sample_id"], "source_spoof_id": s["sample_id"]})
    by_id = {r["sample_id"]: r for v in roles.values() for r in v}
    pair_samples = sorted({p["target_live_id"] for p in pairs} | {p["source_spoof_id"] for p in pairs})
    out = {"schema": SCHEMA, "kind": "SYNTHETIC_QUALIFICATION", "label": "SMOKE_NON_SCIENTIFIC",
           "siwmv2_unit_disclosure": "synthetic world", "roles": roles,
           "pair_samples": [by_id[s] for s in pair_samples], "toy_pairs": pairs, "plan_sha256": None, "test_rows_read": 0}
    out["allowlist"] = sorted({f["face_rel_path"] for v in roles.values() for f in v})
    return out


def write_manifest(manifest: dict, path: str | Path) -> str:
    path = Path(path)
    if path.exists():
        raise ToyManifestError(f"refusing to overwrite {path}")
    data = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
