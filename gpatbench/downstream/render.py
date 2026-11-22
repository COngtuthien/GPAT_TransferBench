"""M8 smoke PLOTS stage: figures rendered only from the metrics-stage CSV tables (spec 20; no typed values).

Every input table passes plots.validate_plot_table(template_mode=False) first. Matplotlib Agg, fixed size/dpi, PNG
metadata stripped ({'Software': None}) so equal inputs give equal bytes in the locked environment. Figure names follow
the spec catalog with a 'smoke_' prefix; every figure is titled SMOKE_NON_SCIENTIFIC.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from statistics import NormalDist

from .plots import validate_plot_table

LABEL = "SMOKE_NON_SCIENTIFIC"


def _read(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _num(v):
    return None if v in ("N/A", "") or v is None else float(v)


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 100, "font.family": "DejaVu Sans", "path.simplify": False})
    return plt


def _save(plt, fig, path: Path):
    if path.exists():
        raise FileExistsError(path)
    fig.tight_layout()
    fig.savefig(path, format="png", metadata={"Software": None})
    plt.close(fig)


def _bar(plt, rows, key, title, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = [r["method_id"] for r in rows]
    ys = [_num(r[key]) for r in rows]
    ax.bar(range(len(xs)), [0.0 if y is None else y for y in ys], color="#4C72B0")
    for i, y in enumerate(ys):
        ax.text(i, 0.0 if y is None else y, "N/A" if y is None else f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(xs)), xs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(key)
    ax.set_title(f"{title}\n{LABEL}", fontsize=9)
    _save(plt, fig, path)


def render_all(metrics_dir: Path, out_dir: Path) -> dict:
    metrics_dir, out_dir = Path(metrics_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    plt = _fig()
    t06 = _read(metrics_dir / "T06_smoke_downstream_resnet18_video.csv")
    t13 = _read(metrics_dir / "T13_smoke_attackwise_apcer.csv")
    t14 = _read(metrics_dir / "T14_smoke_datasetwise_metrics.csv")
    cur = _read(metrics_dir / "curves_roc_det_points.csv")
    trn = _read(metrics_dir / "training_curves.csv")
    checks = {"T06": validate_plot_table(t06, ["ACER", "AUC", "EER"], template_mode=False),
              "T13": validate_plot_table(t13, ["APCER"], template_mode=False),
              "T14": validate_plot_table(t14, ["ACER", "AUC"], template_mode=False),
              "curves": validate_plot_table(cur, ["APCER", "BPCER", "FPR", "TPR"], template_mode=False),
              "training": validate_plot_table([r for r in trn if r["kind"] == "train_step"], ["loss"], template_mode=False)}
    _bar(plt, t06, "ACER", "fig14 video ACER (VAL holdout, seed 42)", out_dir / "smoke_fig14_downstream_acer_bar.png")
    _bar(plt, t06, "AUC", "fig15 video AUC", out_dir / "smoke_fig15_downstream_auc_bar.png")
    _bar(plt, t06, "EER", "fig16 video EER", out_dir / "smoke_fig16_downstream_eer_bar.png")
    methods = [r["method_id"] for r in t06]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for m in methods:
        pts = sorted(((float(r["FPR"]), float(r["TPR"])) for r in cur if r["method_id"] == m))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", label=m)
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    ax.set_xlabel("FPR (BPCER)")
    ax.set_ylabel("TPR (1 - APCER)")
    ax.legend(fontsize=8)
    ax.set_title(f"fig17 ROC\n{LABEL}", fontsize=9)
    _save(plt, fig, out_dir / "smoke_fig17_downstream_roc.png")
    nd = NormalDist()
    clip = lambda v: nd.inv_cdf(min(max(v, 1e-3), 1 - 1e-3))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for m in methods:
        pts = [(clip(float(r["BPCER"])), clip(float(r["APCER"]))) for r in cur if r["method_id"] == m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", label=m)
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999]
    ax.set_xticks([nd.inv_cdf(t) for t in ticks], [str(t) for t in ticks], fontsize=6, rotation=45)
    ax.set_yticks([nd.inv_cdf(t) for t in ticks], [str(t) for t in ticks], fontsize=6)
    ax.set_xlabel("BPCER (normal deviate scale)")
    ax.set_ylabel("APCER (normal deviate scale)")
    ax.legend(fontsize=8)
    ax.set_title(f"fig18 DET\n{LABEL}", fontsize=9)
    _save(plt, fig, out_dir / "smoke_fig18_det_curve.png")
    macros = sorted({r["attack_macro"] for r in t13})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        vals = {r["attack_macro"]: _num(r["APCER"]) for r in t13 if r["method_id"] == m}
        ax.bar([j + i * w for j in range(len(macros))], [vals.get(a) or 0.0 for a in macros], width=w, label=m)
    ax.set_xticks([j + w * (len(methods) - 1) / 2 for j in range(len(macros))], macros, fontsize=7)
    ax.set_ylabel("APCER")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(f"fig19 attack-wise APCER\n{LABEL}", fontsize=9)
    _save(plt, fig, out_dir / "smoke_fig19_attackwise_apcer.png")
    dss = sorted({r["dataset"] for r in t14})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, key in zip(axes, ("ACER", "AUC")):
        for i, m in enumerate(methods):
            vals = {r["dataset"]: _num(r[key]) for r in t14 if r["method_id"] == m}
            ax.bar([j + i * w for j in range(len(dss))], [vals.get(d) or 0.0 for d in dss], width=w, label=m)
        ax.set_xticks([j + w * (len(methods) - 1) / 2 for j in range(len(dss))], dss, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(key)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"fig20 dataset-wise ACER / AUC ({LABEL})", fontsize=9)
    _save(plt, fig, out_dir / "smoke_fig20_datasetwise_metrics.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for m in methods:
        st = [r for r in trn if r["method_id"] == m and r["kind"] == "train_step"]
        axes[0].plot([int(r["step"]) for r in st], [float(r["loss"]) for r in st], marker="o", label=m)
        ep = [r for r in trn if r["method_id"] == m and r["kind"] == "val_epoch"]
        axes[1].plot([int(r["epoch"]) for r in ep], [float(r["val_acer"]) for r in ep], marker="o", label=m)
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("VAL-select video ACER")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"fig24 training curves ({LABEL})", fontsize=9)
    _save(plt, fig, out_dir / "smoke_fig24_training_curves.png")
    return {"input_checks": checks, "figures": sorted(p.name for p in out_dir.glob("*.png"))}
