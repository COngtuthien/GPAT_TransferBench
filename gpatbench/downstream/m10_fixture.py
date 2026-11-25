"""NON-SCIENTIFIC synthetic fixtures for qualifying the M10 launcher and the independent auditor.

Random 256x256 RGB PNGs only (no real face, no bank image, no split data). Layout:
  <dir>/fixture.json                 label NON_SCIENTIFIC_FIXTURE; TRAIN/VAL pools with face_rel_path + face_sha256
  <dir>/faces/fixture/<sid>.png      TRAIN live (L*), TRAIN spoof (S*), VAL (V*) and TEST decoys (T*, never in a pool:
                                     the launcher's firewall must refuse them)
  <dir>/bank/<sid>.png, manifest.json  a fixture "synthetic bank" (for non-E00 groups)
Fixture metrics are never scientific findings.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_fixture(d, *, group: str, seed: int, n_live=96, n_spoof=64, n_syn=64, n_val_videos=16,
                  frames_per_video=2, n_test_decoys=4, img_seed=20261126) -> dict:
    import numpy as np
    from PIL import Image
    d = Path(d)
    (d / "faces" / "fixture").mkdir(parents=True)
    (d / "bank").mkdir()
    rng = np.random.Generator(np.random.PCG64(img_seed))

    def png(path: Path, base, sigma=40):
        a = np.clip(rng.normal(base, sigma, size=(256, 256, 3)), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a, "RGB").save(buf, format="PNG", compress_level=1)
        path.write_bytes(buf.getvalue())
        return _sha(buf.getvalue())

    def frame(sid, y, video, split, fidx, base, sigma=40):
        rel = f"fixture/{sid}.png"
        return {"sample_id": sid, "dataset": "fixture", "subject_id_global": "fixture::" + video, "video_id": video,
                "frame_index": fidx, "y_true": y, "attack_raw": None if y == 0 else "fixture",
                "attack_macro": "live" if y == 0 else "print", "split": split, "face_rel_path": rel,
                "face_sha256": png(d / "faces" / rel, base, sigma)}
    live = [frame("L%04d" % i, 0, "tl%02d" % (i // 8), "TRAIN", i % 8, 100) for i in range(n_live)]
    spoof = [frame("S%04d" % i, 1, "ts%02d" % (i // 8), "TRAIN", i % 8, 150) for i in range(n_spoof)]
    val = []
    for v in range(n_val_videos):
        y = v % 2
        for f in range(frames_per_video):
            val.append(frame("V%02d_%d" % (v, f), y, "vv%02d" % v, "VAL", f, int(rng.integers(80, 170)), 20 + 30 * y))
    for i in range(n_test_decoys):
        png(d / "faces" / "fixture" / ("T%04d.png" % i), 120)
    syn = []
    if group != "E00":
        for i in range(n_syn):
            sid = "Y%04d" % i
            syn.append({"sample_id": sid, "pair_id": sid, "path": sid + ".png",
                        "output_sha256": png(d / "bank" / (sid + ".png"), 140)})
        (d / "bank" / "manifest.json").write_text(json.dumps({"method_id": group, "seed": seed, "samples": syn},
                                                            sort_keys=True) + "\n")
    pools = {"train_live": live, "train_real_spoof": spoof, "val": val}
    body = json.dumps(pools, sort_keys=True).encode()
    doc = {"label": "NON_SCIENTIFIC_FIXTURE", "group": group, "seed": seed, "pools": pools,
           "synthetic": {"method_id": group, "seed": seed, "samples": syn} if syn else None,
           "test_decoys": ["fixture/T%04d.png" % i for i in range(n_test_decoys)],
           "split_manifest_sha256": _sha(b"fixture-split|" + body), "m2_sha256": _sha(b"fixture-m2|" + body)}
    (d / "fixture.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def expectations(fixture_dir, *, run_root, group, seed, epochs, warmup_epochs, env, gpu, code_sha256, source_commit,
                 owner_decisions_sha256, precision_pin_sha256, qualification_overrides=None, grad_scaler=None,
                 policy_name="CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", firewall_required=True) -> dict:
    """Auditor expectations (profile 'qualification') derived from the fixture files, not from the run outputs."""
    fx = Path(fixture_dir)
    doc = json.loads((fx / "fixture.json").read_text())
    pools = doc["pools"]
    syn = (doc["synthetic"] or {}).get("samples", [])
    allow = sorted({f["face_rel_path"] for v in pools.values() for f in v})
    o1 = policy_name == "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"
    scaler = grad_scaler if grad_scaler is not None else (
        {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "enabled": True}
        if o1 else None)
    bank = None
    if group != "E00":
        man = fx / "bank" / "manifest.json"
        bank = {"manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(), "manifest_file": str(man),
                "samples_sha256": _sha("\n".join(f"{s['sample_id']}|{s['output_sha256']}" for s in syn).encode())}
    return {"profile": "qualification", "label": "NON_SCIENTIFIC_QUALIFICATION", "group": group, "seed": seed,
            "track": {"E00": "binary", "E06b": "B_NATIVE_FULL_SECONDARY"}.get(group, "A_FAIR_COMMON_IDENTITY_FREE"),
            "epochs": epochs, "warmup_epochs": warmup_epochs, "run_root": str(run_root), "check_group_dir": False,
            "owner_decisions_sha256": owner_decisions_sha256, "precision_pin_sha256": precision_pin_sha256,
            "source_commit": source_commit, "code_sha256": code_sha256, "env": env, "gpu": gpu,
            "pools": {"train_live": [f["sample_id"] for f in pools["train_live"]],
                      "train_real_spoof": [f["sample_id"] for f in pools["train_real_spoof"]],
                      "val": [f["sample_id"] for f in pools["val"]], "synthetic": [s["sample_id"] for s in syn]},
            "pool_sha256": {k: _sha("\n".join(f"{f['sample_id']}|{f['face_sha256']}" for f in v).encode())
                            for k, v in pools.items()},
            "split_sha256": doc["split_manifest_sha256"], "m2_sha256": doc["m2_sha256"],
            "val_videos": len({(f["dataset"], f["video_id"]) for f in pools["val"]}),
            "allowlist_sha256": _sha("\n".join(allow).encode()), "allowlist_size": len(allow),
            "firewall_required": firewall_required, "bank": bank, "qualification_overrides": qualification_overrides,
            "grad_scaler": scaler, "expect_o1_policy": o1, "policy_name": policy_name, "exit_marker_required": False}
