"""Read-only diagnostics + evidence collector for one finished M10 ResNet-18 detector run (stdlib only).

Runs on the GPU host (python3 -I -B tools/m10_run_diagnostics.py --group G --seed S --out-dir DIR). It never writes
under the run root: it reads the run root and the host-side markers, and writes into --out-dir:
  diagnostics.json          duration, throughput, loss / LR / GradScaler / gradient histories, VAL history, selection,
                            VRAM samples (external nvidia-smi probe), applied/skipped optimizer steps
  run_root_sha256.txt       sha256sum of every file under the run root (LC_ALL=C byte order, paths relative)
  evidence/                 copies of the small run-root records, steps.jsonl.gz (gzip -n), the selected epoch's
                            VAL predictions parquet, and the host-side command / marker / pane-log / VRAM files
The checkpoint bytes are never copied (O3: the single copy stays under the HDD detector root).
This tool is a reporting aid; it is not the independent auditor (tools/m10_run_auditor.py).
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path

M10_ROOT = "/media/ext_data/student20261_gpat_m10_detectors"
SMALL_RECORDS = ("run_manifest.json", "run_status.json", "exit_status.json", "launch_preflight.json",
                 "firewall_report.json", "detector/epochs.json", "detector/selection.json", "detector/detector.json",
                 "detector/best_so_far.json")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fromhex(x):
    return None if x is None else float.fromhex(x)


def quantiles(v):
    v = sorted(v)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]      # noqa: E731
    return {"min": v[0], "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": v[-1], "mean": math.fsum(v) / len(v)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--workdir", default=os.path.expanduser("~/workdir"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    rr = Path(M10_ROOT) / a.group / f"seed_{a.seed}"
    tag = f"m10_{a.group}_s{a.seed}"
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    if os.path.realpath(out).startswith(M10_ROOT):
        raise SystemExit("refusing: out-dir under the detector root")
    (out / "evidence" / "detector").mkdir(parents=True)

    # run-root listing --------------------------------------------------------------------------------------------
    files = sorted((p.relative_to(rr).as_posix() for p in rr.rglob("*") if p.is_file()), key=lambda s: s.encode())
    listing = "".join(f"{sha(rr / f)}  {f}\n" for f in files)
    (out / "run_root_sha256.txt").write_text(listing)

    M = json.loads((rr / "run_manifest.json").read_text()) if (rr / "run_manifest.json").is_file() else None
    E = json.loads((rr / "detector/epochs.json").read_text())
    SEL = json.loads((rr / "detector/selection.json").read_text()) if (rr / "detector/selection.json").is_file() else None
    epochs = E["epochs"] if isinstance(E, dict) and "epochs" in E else E

    # steps --------------------------------------------------------------------------------------------------------
    steps = [json.loads(l) for l in (rr / "detector/steps.jsonl").read_text().splitlines() if l.strip()]
    skipped = [s["step"] for s in steps if not s["optimizer_step_applied"]]
    scale_changes = [{"step": s["step"], "before": s["grad_scale_before"], "after": s["grad_scale"]}
                     for s in steps if s["grad_scale"] != s["grad_scale_before"]]
    gpre = [fromhex(s["grad_norm_hex"]) for s in steps if s["optimizer_step_applied"]]
    gpost = [fromhex(s["grad_norm_post_clip_hex"]) for s in steps if s["optimizer_step_applied"]]
    loss = [fromhex(s["loss_hex"]) for s in steps]
    lr_marks = sorted({0, 1, 524, 525, 526, len(steps) - 1} | {e * 175 for e in range(41) if e * 175 < len(steps)})
    lr_hist = [{"step": steps[k]["step"], "epoch": steps[k]["epoch"], "lr_backbone": fromhex(steps[k]["lr_backbone_hex"]),
                "lr_head": fromhex(steps[k]["lr_head_hex"])} for k in lr_marks if k < len(steps)]

    # epochs -------------------------------------------------------------------------------------------------------
    hist, prev_wall = [], 0.0
    for e in epochs:
        w = e.get("wall_seconds")
        hist.append({"epoch": e["epoch"], "train_loss_mean": e["train_loss_mean"], "train_loss_last": e["train_loss_last"],
                     "optimizer_steps_applied": e["optimizer_steps_applied"],
                     "optimizer_steps_skipped": e["optimizer_steps_skipped"],
                     "grad_norm_pre_clip_mean": e["grad_norm_pre_clip_mean"],
                     "grad_norm_pre_clip_max": e["grad_norm_pre_clip_max"], "grad_scale_end": e["grad_scale_end"],
                     "lr_backbone_last": fromhex(e.get("lr_backbone_last_hex")),
                     "lr_head_last": fromhex(e.get("lr_head_last_hex")),
                     "val_acer": e["val_acer"], "val_auc": e["val_auc"], "threshold": fromhex(e["threshold_hex"]),
                     "threshold_tie_break": e["threshold_tie_break"], "became_best": e["became_best"],
                     "val_videos": e["val_videos"], "val_frames": e["val_frames"],
                     "predictions_sha256": e["predictions_sha256"], "state_sha256": e["state_sha256"],
                     "wall_seconds_cumulative": w,
                     "epoch_wall_seconds": None if w is None else w - prev_wall})
        prev_wall = w if w is not None else prev_wall
    ew = [h["epoch_wall_seconds"] for h in hist if h["epoch_wall_seconds"] is not None]

    # host markers -------------------------------------------------------------------------------------------------
    wd = Path(a.workdir)
    rd = lambda n: (wd / f"{tag}.{n}").read_text().strip() if (wd / f"{tag}.{n}").is_file() else None  # noqa: E731
    vram = []
    if (wd / f"{tag}.vram.tsv").is_file():
        for l in (wd / f"{tag}.vram.tsv").read_text().splitlines():
            parts = l.replace("\\t", "\t").split("\t")
            if len(parts) >= 3 and parts[1].strip().isdigit():
                tot = parts[2].split(",")
                vram.append({"utc": parts[0], "proc_mib": int(parts[1]), "gpu_used_mib": int(tot[0]),
                             "gpu_util_pct": int(tot[1]) if len(tot) > 1 else None})

    sel_epoch = SEL["selected_epoch"] if SEL else None
    diag = {
        "schema": "m10_run_diagnostics_v1", "group": a.group, "seed": a.seed, "run_root": str(rr),
        "run_id": (M or {}).get("run_id"), "execution_commit": (M or {}).get("execution_commit"),
        "markers": {"launch_time": rd("launch_time"), "end_time": rd("end_time"), "exit": rd("exit"),
                    "command_sha256": sha(wd / f"{tag}.command") if (wd / f"{tag}.command").is_file() else None},
        "manifest_times": {"start_utc": (M or {}).get("start_utc"), "end_utc": (M or {}).get("end_utc")},
        "training_wall_seconds": prev_wall,
        "throughput": {"epoch_wall_seconds": quantiles(ew),
                       "steps_per_second_mean": (len(steps) / prev_wall) if prev_wall else None,
                       "images_per_second_mean": (64 * len(steps) / prev_wall) if prev_wall else None,
                       "note": "epoch wall includes the VAL pass and checkpoint events"},
        "steps": {"recorded": len(steps), "expected": 7000, "applied": len(steps) - len(skipped),
                  "skipped": len(skipped), "skipped_steps": skipped, "grad_scale_changes": scale_changes,
                  "grad_scale_first": steps[0]["grad_scale_before"] if steps else None,
                  "grad_scale_last": steps[-1]["grad_scale"] if steps else None,
                  "loss": quantiles(loss), "loss_first": loss[0] if loss else None, "loss_last": loss[-1] if loss else None,
                  "grad_norm_pre_clip": quantiles(gpre), "grad_norm_post_clip": quantiles(gpost),
                  "fraction_clipped": (sum(g > 1.0 for g in gpre) / len(gpre)) if gpre else None,
                  "batch_compositions": sorted({(s["n_live"], s["n_real_spoof"], s["n_synthetic"]) for s in steps}),
                  "lr_history_marks": lr_hist},
        "epochs": hist,
        "selection": SEL,
        "vram_external_probe": {"samples": len(vram),
                                "proc_mib_max": max((v["proc_mib"] for v in vram), default=None),
                                "proc_mib_median": statistics.median([v["proc_mib"] for v in vram]) if vram else None,
                                "gpu_used_mib_max": max((v["gpu_used_mib"] for v in vram), default=None),
                                "gpu_util_pct_median": statistics.median([v["gpu_util_pct"] for v in vram
                                                                          if v["gpu_util_pct"] is not None]) if vram else None,
                                "note": "nvidia-smi per-process used memory every 15 s (includes CUDA context and "
                                        "caching-allocator reserve); GPU shared with unrelated jobs"},
        "run_root_listing": {"files": len(files), "sha256": hashlib.sha256(listing.encode()).hexdigest()},
    }
    # evidence copies ----------------------------------------------------------------------------------------------
    for r in SMALL_RECORDS:
        if (rr / r).is_file():
            shutil.copyfile(rr / r, out / "evidence" / r)
    with open(rr / "detector/steps.jsonl", "rb") as src, open(out / "evidence/detector/steps.jsonl.gz", "wb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0) as gz:
            shutil.copyfileobj(src, gz)
    if sel_epoch is not None:
        pn = f"detector/predictions_val_epoch{sel_epoch:02d}.parquet"
        shutil.copyfile(rr / pn, out / "evidence" / pn)
    (out / "evidence/host").mkdir()
    for n in ("command", "launch_time", "end_time", "exit", "pane.log", "vram.tsv"):
        if (wd / f"{tag}.{n}").is_file():
            shutil.copyfile(wd / f"{tag}.{n}", out / "evidence/host" / f"{tag}.{n}")
    if (wd / f"{tag}_preflight.json").is_file():
        shutil.copyfile(wd / f"{tag}_preflight.json", out / "evidence/host" / f"{tag}_preflight.json")
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"ok": True, "out": str(out), "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
