"""M8 smoke METRICS stage: tables computed only from prediction files (spec 16, 19 formats; owner-approved D7-D10).

Holdout prediction files (the TEST stand-in) give the T06/T08/T13/T14-shaped smoke tables; the training logs give the
fig24 input. Rates are written both as exact rationals and as float reprs. Undefined cells are "N/A" with a reason;
nothing is filled in. Every row carries label SMOKE_NON_SCIENTIFIC. The subject bootstrap CI is out of smoke scope.
"""
from __future__ import annotations

import csv
import io
import json
from fractions import Fraction
from pathlib import Path

from . import metrics as M

RULE = "SPOOF_IF_SCORE_GE_THRESHOLD"
LABEL = "SMOKE_NON_SCIENTIFIC"
METHODS = ("E00", "E01", "E02")


def fmt(v) -> tuple[str, str]:
    if v is None:
        return "N/A", "N/A"
    if isinstance(v, Fraction):
        return repr(float(v)), f"{v.numerator}/{v.denominator}"
    return repr(float(v)), repr(float(v))


def read_predictions(path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def siw_token_map(root: Path) -> dict:
    import yaml
    doc = yaml.safe_load((root / "configs/frozen/attack_map_v1.yaml").read_text())
    m = {tok: v["attack_macro"] for tok, v in doc["mapped"]["siwmv2"].items()}
    if len(m) != 14:
        raise ValueError(f"attack_map_v1 siwmv2 has {len(m)} tokens, expected 14")
    return m


def video_metrics(videos, threshold) -> dict:
    y = [v["y_true"] for v in videos]
    s = [v["video_score"] for v in videos]
    pred = M.predict_spoof(s, threshold, rule=RULE)
    r = M.rates_from_counts(M.confusion_counts(y, pred))
    both = 0 in y and 1 in y
    r.update(AUC=M.auc(y, s) if both else None, EER=M.eer_linear_interpolation(y, s, rule=RULE) if both else None,
             F1_spoof=M.f1_spoof(y, pred), **{"TPR@FPR=1%": M.tpr_at_fpr(y, s, "0.01", rule=RULE) if both else None,
                                               "TPR@FPR=5%": M.tpr_at_fpr(y, s, "0.05", rule=RULE) if both else None})
    return r


def frame_metrics(frames, threshold) -> dict:
    r = M.frame_level_rates(frames, threshold, rule=RULE)
    y = [f["y_true"] for f in frames]
    s = [f["p_frame"] for f in frames]
    both = 0 in y and 1 in y
    r.update(AUC=M.auc(y, s) if both else None, EER=M.eer_linear_interpolation(y, s, rule=RULE) if both else None)
    return r


def to_csv(rows: list[dict], columns: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue().encode("utf-8")


def build_tables(evaluator_dir: Path, root: Path) -> dict[str, bytes]:
    """{filename: csv bytes}. evaluator_dir holds <method>/predictions_holdout.parquet, steps.jsonl, epochs.json."""
    tokens = siw_token_map(root)
    T06, T08, T13, T13b, T14, CURVE, TRAIN = [], [], [], [], [], [], []
    keys = ["APCER", "BPCER", "ACER", "AUC", "EER", "accuracy", "balanced_accuracy", "F1_spoof", "TPR@FPR=1%", "TPR@FPR=5%"]
    for m in METHODS:
        rows = read_predictions(evaluator_dir / m / "predictions_holdout.parquet")
        thr = {r["threshold"] for r in rows}
        if len(thr) != 1:
            raise ValueError("one frozen threshold per prediction file")
        t = thr.pop()
        videos = M.aggregate_videos(rows)
        vm = video_metrics(videos, t)
        row = {"method_id": m, "evaluator_id": "resnet18", "seed": 42, "n_seeds": 1, "split": "VAL_HOLDOUT_TEST_PROXY",
               "threshold": repr(t), "videos": len(videos), "label": LABEL, "sample_std": "N/A (n_seeds = 1)",
               "ci95_acer": "N/A (subject bootstrap out of smoke scope)"}
        for k in keys:
            row[k], row[k + "_exact"] = fmt(vm[k])
        T06.append(row)
        fm = frame_metrics(rows, t)
        frow = {"method_id": m, "seed": 42, "threshold_source": "frozen video threshold (D9)", "frames": len(rows), "label": LABEL}
        for k in ("APCER", "BPCER", "ACER", "accuracy", "balanced_accuracy", "F1_spoof", "AUC", "EER"):
            frow[k], frow[k + "_exact"] = fmt(fm[k])
        T08.append(frow)
        sub = M.apply_frozen_threshold(videos, t, rule=RULE)
        for am, v in sub["apcer_by_attack_macro"].items():
            a, ae = fmt(v)
            n = sum(1 for x in videos if x["y_true"] == 1 and x.get("attack_macro") == am)
            T13.append({"method_id": m, "attack_macro": am, "APCER": a, "APCER_exact": ae, "spoof_videos": n, "label": LABEL})
        raw = M.siw_raw_type_table(videos, t, tokens, rule=RULE)
        for r in raw["rows"]:
            a, ae = fmt(r["APCER"])
            T13b.append({"method_id": m, "table_status": raw["status"], "attack_raw": r["attack_raw"], "APCER": a,
                         "APCER_exact": ae, "numerator": "N/A" if r["numerator"] is None else r["numerator"],
                         "denominator": "N/A" if r["denominator"] is None else r["denominator"],
                         "reason": r["reason"] or "", "label": LABEL})
        for ds, r in sub["by_dataset"].items():
            row = {"method_id": m, "dataset": ds, "label": LABEL,
                   "dataset_unit_note": "siwmv2 units are content groups, not subjects" if ds == "siwmv2" else ""}
            for k in ("APCER", "BPCER", "ACER", "AUC"):
                row[k], row[k + "_exact"] = fmt(r.get(k))
            T14.append(row)
        for p in M.curve_points([v["y_true"] for v in videos], [v["video_score"] for v in videos], rule=RULE):
            CURVE.append({"method_id": m, "threshold": repr(p["threshold"]), "APCER": fmt(p["APCER"])[0],
                          "BPCER": fmt(p["BPCER"])[0], "FPR": fmt(p["FPR"])[0], "TPR": fmt(p["TPR"])[0], "label": LABEL})
        for s in (json.loads(l) for l in (evaluator_dir / m / "steps.jsonl").read_text().splitlines()):
            TRAIN.append({"method_id": m, "kind": "train_step", "step": s["step"], "epoch": s["epoch"],
                          "loss": repr(float.fromhex(s["loss_hex"])), "val_acer": "", "val_auc": "", "label": LABEL})
        for e in json.loads((evaluator_dir / m / "epochs.json").read_text()):
            TRAIN.append({"method_id": m, "kind": "val_epoch", "step": "", "epoch": e["epoch"], "loss": "",
                          "val_acer": repr(float(Fraction(e["val_acer"]))), "val_auc": repr(float(Fraction(e["val_auc"]))),
                          "label": LABEL})
    vcols = ["method_id", "evaluator_id", "seed", "n_seeds", "split", "threshold", "videos"] + \
        [c for k in keys for c in (k, k + "_exact")] + ["sample_std", "ci95_acer", "label"]
    fcols = ["method_id", "seed", "threshold_source", "frames"] + \
        [c for k in ("APCER", "BPCER", "ACER", "accuracy", "balanced_accuracy", "F1_spoof", "AUC", "EER")
         for c in (k, k + "_exact")] + ["label"]
    return {
        "T06_smoke_downstream_resnet18_video.csv": to_csv(T06, vcols),
        "T08_smoke_downstream_frame.csv": to_csv(T08, fcols),
        "T13_smoke_attackwise_apcer.csv": to_csv(T13, ["method_id", "attack_macro", "APCER", "APCER_exact", "spoof_videos", "label"]),
        "T13b_smoke_siwmv2_raw_type_apcer.csv": to_csv(T13b, ["method_id", "table_status", "attack_raw", "APCER", "APCER_exact",
                                                              "numerator", "denominator", "reason", "label"]),
        "T14_smoke_datasetwise_metrics.csv": to_csv(T14, ["method_id", "dataset", "APCER", "APCER_exact", "BPCER", "BPCER_exact",
                                                          "ACER", "ACER_exact", "AUC", "AUC_exact", "dataset_unit_note", "label"]),
        "curves_roc_det_points.csv": to_csv(CURVE, ["method_id", "threshold", "APCER", "BPCER", "FPR", "TPR", "label"]),
        "training_curves.csv": to_csv(TRAIN, ["method_id", "kind", "step", "epoch", "loss", "val_acer", "val_auc", "label"]),
    }
