"""CPU-safe ArtifactProbeNet smoke (M5 owner-resolution pass §27). NOT a training run.

    <m2 venv>/bin/python tools/m5_probe_smoke.py

Exercises the frozen contract end to end without optimizing the network on real data: config and
hashes, class map, the frozen transform on real canonical faces, model construction and a single
forward pass, the weighted loss, the fixed-class macro-F1, the cosine LR sequence, the embedding,
and every refusal. No checkpoint is written and no performance is claimed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E            # noqa: E402
from gpatbench.probe import contract as C           # noqa: E402
from gpatbench.probe import data as D               # noqa: E402
from gpatbench.probe import metrics as MET          # noqa: E402
from gpatbench.probe import model as MODEL          # noqa: E402
from gpatbench.probe import preprocess as PP        # noqa: E402
from gpatbench.probe import train as T              # noqa: E402

AUDIT = ROOT / "outputs/audit"
SALT = "gpatbench.m5_smoke.v1|"
N_PER_SPLIT = 8


def main() -> int:
    out = {"authoritative_training": False, "checkpoint_written": False,
           "performance_claimed": False, "checks": {}}

    # ---- contract -------------------------------------------------------
    cfg = C.load_config()
    pop = C.verify_population(cfg)
    w = C.verify_class_weights(cfg, pop["train"])
    out["config_sha256"] = C.sha256_file(C.FROZEN_CONFIG)
    out["snapshot_byte_identical"] = (
        C.FROZEN_SNAPSHOT.read_bytes() == C.FROZEN_CONFIG.read_bytes())
    out["classes"] = list(C.CLASSES)
    out["class_weights"] = [float(x) for x in w]
    out["train_counts"], out["val_counts"] = pop["train"], pop["val"]
    out["checks"]["config_frozen_and_hashed"] = True
    out["checks"]["counts_match_split_manifest"] = True
    out["checks"]["weights_recomputed_from_manifest"] = True

    # ---- transform on real faces ---------------------------------------
    samples = {}
    for split in ("TRAIN", "VAL"):
        rows = D.load_rows(split)
        rows = sorted(rows, key=lambda r: hashlib.sha256(
            (SALT + r["sample_id"]).encode()).hexdigest())[:N_PER_SPLIT]
        ds = D.ProbeDataset(split, rows=rows)
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x); ys.append(y)
        a = np.stack(xs)
        samples[split] = {
            "n": len(ds), "shape": list(a.shape[1:]), "dtype": str(a.dtype),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "signed": bool(a.min() < 0 < a.max()),
            "labels": ys,
            "label_tokens": [C.CLASSES[i] for i in ys],
        }
    out["transform"] = samples
    out["checks"]["transform_shape_3x224x224"] = all(
        s["shape"] == [3, 224, 224] for s in samples.values())
    out["checks"]["transform_signed_residual"] = all(s["signed"] for s in samples.values())
    out["checks"]["transform_float32"] = all(s["dtype"] == "float32" for s in samples.values())

    # deterministic repeat on one real face
    ds = D.ProbeDataset("TRAIN", rows=D.load_rows("TRAIN")[:1])
    x1, _ = ds[0]
    x2, _ = ds[0]
    out["checks"]["transform_deterministic"] = bool(np.array_equal(x1, x2))
    # the frozen path is exactly the candidate path under the frozen options
    out["checks"]["frozen_path_equals_explicit_options"] = bool(np.array_equal(
        PP.frozen_probe_input(np.zeros((256, 256, 3), np.uint8) + 17),
        PP.probe_input(np.zeros((256, 256, 3), np.uint8) + 17, **PP.FROZEN)))

    # ---- model + one forward pass --------------------------------------
    T.seed_everything()
    model = MODEL.build(C.SEED)
    rep = MODEL.trainable_report(model)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.stack([samples_x for samples_x in
                                        [D.ProbeDataset("VAL", rows=D.load_rows("VAL")[:4])[i][0]
                                         for i in range(4)]]))
        logits = model(xb)
        emb = MODEL.embed(model, xb)
    out["model"] = rep
    out["forward"] = {"input_shape": list(xb.shape), "logits_shape": list(logits.shape),
                      "logits_finite": bool(torch.isfinite(logits).all()),
                      "embedding_shape": list(emb.shape),
                      "embedding_l2_norms": [float(v) for v in emb.norm(dim=1)]}
    out["checks"]["seven_logits"] = list(logits.shape) == [4, C.K]
    out["checks"]["fully_trainable_backbone"] = rep["fully_trainable"]
    out["checks"]["embedding_512"] = list(emb.shape) == [4, C.EMBED_DIM]
    out["checks"]["embedding_unit_norm"] = bool(
        torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-6))

    # ---- loss ------------------------------------------------------------
    crit = T.build_loss(w)
    with torch.no_grad():
        loss = crit(logits, torch.tensor([0, 1, 2, 3]))
    # The weights are COMPUTED in float64 (frozen contract) and the tensor is float32 to match the
    # model dtype, which PyTorch requires. The cast is element-wise, so the frozen class ORDER is
    # what must be asserted, against the float64 values cast the same way.
    expect32 = np.asarray(w, dtype=np.float64).astype(np.float32)
    got32 = crit.weight.detach().cpu().numpy()
    out["loss"] = {"class": type(crit).__name__, "reduction": crit.reduction,
                   "weight_dtype": str(got32.dtype), "computed_dtype": "float64",
                   "weight_order_matches_classes": bool(np.array_equal(got32, expect32)),
                   "max_abs_float32_cast_error": float(np.max(np.abs(
                       got32.astype(np.float64) - np.asarray(w, dtype=np.float64)))),
                   "value_finite": bool(torch.isfinite(loss))}
    out["checks"]["weighted_ce_in_frozen_class_order"] = \
        out["loss"]["weight_order_matches_classes"]

    # ---- macro-F1 --------------------------------------------------------
    perfect = MET.confusion_matrix(list(range(C.K)), list(range(C.K)))
    out["metric"] = {
        "perfect_macro_f1": MET.macro_f1(perfect),
        "empty_class_gives_zero_f1": bool(MET.per_class_f1(
            MET.confusion_matrix([0, 0, 1], [0, 0, 1]))[2] == 0.0),
        "unpredicted_class_not_dropped": len(MET.per_class_f1(
            MET.confusion_matrix([0], [0]))) == C.K,
        "tie_keeps_earlier_epoch": not MET.is_better(0.5, 0.5),
        "strict_improvement_replaces": MET.is_better(0.5 + 1e-12, 0.5),
    }
    out["checks"]["macro_f1_fixed_seven_classes"] = (
        out["metric"]["perfect_macro_f1"] == 1.0
        and out["metric"]["unpredicted_class_not_dropped"]
        and out["metric"]["empty_class_gives_zero_f1"])
    out["checks"]["checkpoint_tie_keeps_earlier_epoch"] = out["metric"]["tie_keeps_earlier_epoch"]

    # ---- scheduler -------------------------------------------------------
    analytic = C.lr_schedule()
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.T_MAX, eta_min=C.ETA_MIN)
    torch_seq = []
    for _ in range(C.EPOCHS):
        torch_seq.append(opt.param_groups[0]["lr"])
        sch.step()
    rel = max(abs(a - b) / max(1e-30, b) for a, b in zip(torch_seq, analytic))
    out["scheduler"] = {"implementation": "torch.optim.lr_scheduler.CosineAnnealingLR",
                        "T_max": C.T_MAX, "eta_min": C.ETA_MIN, "step": "once_per_epoch",
                        "epoch_1_lr": torch_seq[0], "epoch_30_lr": torch_seq[-1],
                        "lr_after_30_steps": opt.param_groups[0]["lr"],
                        "max_relative_difference_vs_analytic": rel,
                        "exact_matches": sum(1 for a, b in zip(torch_seq, analytic) if a == b),
                        "warmup": None}
    out["checks"]["epoch_1_starts_at_1e_4"] = torch_seq[0] == C.LR
    out["checks"]["cosine_reaches_eta_min"] = opt.param_groups[0]["lr"] == C.ETA_MIN
    with (AUDIT / "M5_ARTIFACT_PROBE_LR_SCHEDULE.csv").open("w", newline="",
                                                            encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["epoch", "lr_in_force_during_epoch_torch", "lr_analytic_closed_form"])
        for i, (a, b) in enumerate(zip(torch_seq, analytic), start=1):
            wri.writerow([i, repr(a), repr(b)])

    # ---- firewalls -------------------------------------------------------
    refusals = {}
    try:
        D.load_rows("TEST")
        refusals["test_split"] = False
    except C.ProbeContractViolation:
        refusals["test_split"] = True
    try:
        D.ProbeDataset("TEST")
        refusals["test_dataset"] = False
    except C.ProbeContractViolation:
        refusals["test_dataset"] = True
    try:
        T.run(dry_run=False)
        refusals["cpu_authoritative"] = False
    except C.ProbeContractViolation:
        refusals["cpu_authoritative"] = True
    try:
        C.load_config(ROOT / "configs/proposed/artifact_probe.proposed.yaml")
        refusals["non_frozen_config"] = False
    except C.ProbeContractViolation:
        refusals["non_frozen_config"] = True
    out["refusals"] = refusals
    out["checks"].update({f"refuses_{k}": v for k, v in refusals.items()})
    out["checks"]["no_synthetic_code_path"] = all(
        s.synthetic_samples == 0 for s in (D.ProbeDataset("TRAIN", rows=[]),
                                           D.ProbeDataset("VAL", rows=[])))

    out["checks"] = {k: bool(v) for k, v in out["checks"].items()}    # no numpy scalars in JSON
    out["metric"] = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                         else bool(v)) for k, v in out["metric"].items()}
    out["status"] = "PASS" if all(out["checks"].values()) else "FAIL"
    out["failed_checks"] = [k for k, v in out["checks"].items() if not v]
    (AUDIT / "M5_ARTIFACT_PROBE_SMOKE.json").write_bytes(E.canonical_json_bytes(out))
    print(json.dumps({"status": out["status"], "failed": out["failed_checks"],
                      "checks": len(out["checks"])}, indent=1))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
