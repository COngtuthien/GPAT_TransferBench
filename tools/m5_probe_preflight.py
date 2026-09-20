"""M5 pre-flight data audit for ArtifactProbeNet (spec §12.1). READ-ONLY. Trains nothing.

    <m2 venv>/bin/python tools/m5_probe_preflight.py

Computes the probe population, the attack_macro class inventory under both candidate populations,
every plausible class-weight formula, and the feasibility numbers. Writes
outputs/audit/M5_ARTIFACT_PROBE_PREFLIGHT.json. No model is constructed for training, no checkpoint
is written, and TEST rows are counted for integrity only -- never loaded or used.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"
SPLIT = ROOT / "manifests/split_v1.parquet"

# spec §3.2 attack_macro enum, verbatim and in spec order
SPEC_ATTACK_MACRO_ENUM = ["live", "print", "replay", "mask_2d", "mask_3d", "makeup", "partial",
                          "other_spoof"]
CLIP_LO, CLIP_HI = 0.5, 3.0
BATCH = 64


def weight_variants(counts: dict) -> dict:
    """Every plausible reading of 'inverse-frequency weights clipped to [0.5, 3.0]' (spec §12.1)."""
    classes = sorted(counts)
    n = np.array([counts[c] for c in classes], dtype=np.float64)
    N, K = n.sum(), len(classes)
    raw = 1.0 / n                                   # w_c = 1 / n_c
    balanced = N / (K * n)                          # sklearn 'balanced'
    mean_one = raw / raw.mean()                     # raw rescaled to mean 1
    max_one = raw / raw.max()                       # raw rescaled to max 1
    median_over = np.median(n) / n                  # median frequency balancing
    out = {}
    for name, w in (("raw_inverse", raw), ("normalized_inverse_N_over_Kn", balanced),
                    ("mean_one_inverse", mean_one), ("max_one_inverse", max_one),
                    ("median_frequency", median_over)):
        clipped = np.clip(w, CLIP_LO, CLIP_HI)
        out[name] = {
            "formula": {"raw_inverse": "w_c = 1 / n_c",
                        "normalized_inverse_N_over_Kn": "w_c = N / (K * n_c)",
                        "mean_one_inverse": "w_c = (1/n_c) / mean_c(1/n_c)",
                        "max_one_inverse": "w_c = (1/n_c) / max_c(1/n_c)",
                        "median_frequency": "w_c = median_c(n_c) / n_c"}[name],
            "unclipped": {c: float(v) for c, v in zip(classes, w)},
            "clipped": {c: float(v) for c, v in zip(classes, clipped)},
            "all_clipped_to_lower_bound": bool(np.all(clipped == CLIP_LO)),
            "all_clipped_to_upper_bound": bool(np.all(clipped == CLIP_HI)),
            "n_hitting_lower_bound": int((clipped == CLIP_LO).sum()),
            "n_hitting_upper_bound": int((clipped == CLIP_HI).sum()),
            "n_unclipped": int(((w >= CLIP_LO) & (w <= CLIP_HI)).sum()),
            "clipping_destroys_all_information": bool(len(set(np.round(clipped, 12))) == 1),
            "max_over_min_after_clip": float(clipped.max() / clipped.min()),
        }
    return {"class_order": classes, "counts": {c: int(counts[c]) for c in classes},
            "variants": out}


def main() -> int:
    rows = pq.read_table(SPLIT).to_pylist()
    rep = {
        "read_only": True, "trained": False, "checkpoint_created": False,
        "split_manifest_sha256": E.sha256_file(SPLIT),
        "spec_attack_macro_enum": SPEC_ATTACK_MACRO_ENUM,
        "clip": [CLIP_LO, CLIP_HI], "batch_size": BATCH,
    }

    # ---------------- population ----------------
    pop = {}
    for split in ("TRAIN", "VAL", "TEST"):
        sub = [r for r in rows if r["split"] == split]
        live = [r for r in sub if int(r["label_binary"]) == 0]
        spoof = [r for r in sub if int(r["label_binary"]) == 1]
        pop[split] = {
            "total": len(sub), "live": len(live), "spoof": len(spoof),
            "by_dataset": dict(sorted(collections.Counter(r["dataset"] for r in sub).items())),
            "live_by_dataset": dict(sorted(collections.Counter(r["dataset"] for r in live).items())),
            "spoof_by_dataset": dict(sorted(collections.Counter(r["dataset"] for r in spoof).items())),
            "attack_macro": dict(sorted(collections.Counter(r["attack_macro"] for r in sub).items())),
            "attack_macro_live_rows": dict(sorted(collections.Counter(
                r["attack_macro"] for r in live).items())),
            "attack_macro_spoof_rows": dict(sorted(collections.Counter(
                r["attack_macro"] for r in spoof).items())),
            "attack_raw_tokens": len({r["attack_raw"] for r in sub if r["attack_raw"]}),
            "null_or_empty_attack_macro": sum(1 for r in sub if not r["attack_macro"]),
        }
    rep["population"] = pop
    rep["test_usage"] = ("counted for integrity only; TEST rows are never loaded, never trained on "
                         "and never used for checkpoint selection")

    # ---------------- class inventory for both candidate populations ----------------
    def inventory(pred, name, desc):
        tr = [r for r in rows if r["split"] == "TRAIN" and pred(r)]
        va = [r for r in rows if r["split"] == "VAL" and pred(r)]
        tc = collections.Counter(r["attack_macro"] for r in tr)
        vc = collections.Counter(r["attack_macro"] for r in va)
        classes = sorted(set(tc) | set(vc))                # deterministic: lexical ascending
        per_class = {}
        for c in classes:
            ds_tr = sorted({r["dataset"] for r in tr if r["attack_macro"] == c})
            ds_va = sorted({r["dataset"] for r in va if r["attack_macro"] == c})
            per_class[c] = {
                "train": int(tc.get(c, 0)), "val": int(vc.get(c, 0)),
                "train_datasets": ds_tr, "val_datasets": ds_va,
                "attack_raw_tokens_train": sorted({r["attack_raw"] for r in tr
                                                   if r["attack_macro"] == c and r["attack_raw"]}),
                "in_train": c in tc, "in_val": c in vc,
            }
        return {
            "name": name, "description": desc,
            "class_order": classes, "class_order_rule": "lexical class token ascending",
            "num_classes": len(classes),
            "train_samples": len(tr), "val_samples": len(va),
            "per_class": per_class,
            "train_only_classes": sorted(set(tc) - set(vc)),
            "val_only_classes": sorted(set(vc) - set(tc)),
            "rarest_train_class": min(classes, key=lambda c: tc.get(c, 0)) if classes else None,
            "min_train_count": int(min(tc.get(c, 0) for c in classes)) if classes else None,
            "max_train_count": int(max(tc.get(c, 0) for c in classes)) if classes else None,
            "imbalance_ratio_train": (float(max(tc.get(c, 0) for c in classes) /
                                            max(1, min(tc.get(c, 0) for c in classes)))
                                      if classes else None),
            "class_weights": weight_variants({c: tc.get(c, 0) for c in classes}),
            "batches_per_epoch_drop_last_false": -(-len(tr) // BATCH),
            "batches_per_epoch_drop_last_true": len(tr) // BATCH,
        }

    rep["candidate_populations"] = {
        "A_spoof_only": inventory(lambda r: int(r["label_binary"]) == 1, "A",
                                  "SPOOF attack_macro classes only (live excluded)"),
        "B_live_plus_spoof": inventory(lambda r: True, "B",
                                       "all real rows; the spec enum includes a `live` token"),
    }

    # ---------------- spec-enum coverage ----------------
    seen = {r["attack_macro"] for r in rows}
    rep["enum_coverage"] = {
        "spec_enum": SPEC_ATTACK_MACRO_ENUM,
        "present_in_split": sorted(seen),
        "spec_tokens_absent_from_data": [c for c in SPEC_ATTACK_MACRO_ENUM if c not in seen],
        "data_tokens_absent_from_spec_enum": sorted(seen - set(SPEC_ATTACK_MACRO_ENUM)),
        "live_token_present": "live" in seen,
        "live_token_only_on_live_rows": all(
            (int(r["label_binary"]) == 0) == (r["attack_macro"] == "live") for r in rows),
    }

    # ---------------- feasibility ----------------
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=None)
    params = sum(p.numel() for p in m.parameters())
    rep["feasibility"] = {
        "resnet18_params_default_1000_classes": int(params),
        "param_bytes_fp32": int(params * 4),
        "checkpoint_estimate_bytes_weights_only": int(params * 4),
        "checkpoint_estimate_bytes_with_adamw_states": int(params * 4 * 3),
        "penultimate_feature_dim": int(m.fc.in_features),
        "input_tensor_bytes_per_sample_fp32_3x224x224": 3 * 224 * 224 * 4,
        "input_tensor_bytes_per_batch64_fp32": 64 * 3 * 224 * 224 * 4,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "weight_url": ResNet18_Weights.IMAGENET1K_V1.url,
        "note": ("No training was run, so no wall-clock or GPU-memory figure is reported. "
                 "Parameter counts are from model construction only."),
    }
    rep["status"] = "PREFLIGHT_ONLY"
    (AUDIT / "M5_ARTIFACT_PROBE_PREFLIGHT.json").write_bytes(E.canonical_json_bytes(rep))
    print(json.dumps({
        "TRAIN": pop["TRAIN"], "VAL": {k: pop["VAL"][k] for k in ("total", "live", "spoof")},
        "A_classes": rep["candidate_populations"]["A_spoof_only"]["class_order"],
        "B_classes": rep["candidate_populations"]["B_live_plus_spoof"]["class_order"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
