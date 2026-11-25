"""M10 downstream ResNet-18 science-runner qualification on SYNTHETIC FIXTURES (non-scientific; no real face, no
synthetic bank, no VAL/TEST image is read). GPU host, approved env gpat-m8-downstream, fresh clean clone:

  env -i HOME=$HOME PATH=/usr/bin:/bin LANG=C.UTF-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONNOUSERSITE=1 ~/workdir/envs/gpat-m8-downstream/bin/python -B tools/m10_downstream_qualification.py \
      --out <new dir outside the runtime root and the bank root> --weights ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

Q1 augmentation: science_runner.draw_aug_params + apply_aug == resnet_runner.train_transform bitwise (300 images).
Q2 CPU equivalence: science runner (CPU_FP32_DETERMINISTIC, workers 0 and 4) == qualified smoke runner bitwise (every
   step's lr / loss / grad norm / batch ids, every epoch's state sha256, threshold and selected epoch) for E00 and a
   mixed method.
Q3 CUDA policies: each candidate precision policy completes, is replayed bitwise in a second run, and its selected
   checkpoint reloads to its state sha256; fp16 GradScaler skipped steps are counted.
Q4 throughput (provisional resource evidence): GPU train step at batch 64 per policy, VAL inference rate, and the CPU
   decode + augmentation cost of a 64-image batch for worker counts 0/4/8 on 256x256 PNGs.
Q5 plots: per-epoch VAL ACER/AUC and VAL ROC from the qualification outputs, inputs checked by plots.validate_plot_table.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture(n_live=96, n_spoof=64, n_syn=64, n_val_videos=32, frames_per_video=2, seed=20261125):
    """Deterministic random 256x256 RGB PNG images in memory (live/spoof differ in mean intensity so training moves)."""
    import numpy as np
    from PIL import Image
    rng = np.random.Generator(np.random.PCG64(seed))
    store = {}

    def png(sid, base, sigma=40):
        a = np.clip(rng.normal(base, sigma, size=(256, 256, 3)), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a, "RGB").save(buf, format="PNG", compress_level=6)
        store[sid] = buf.getvalue()

    def frame(sid, y, video, split, ds="fixture"):
        return {"sample_id": sid, "dataset": ds, "subject_id_global": "fixture::" + video, "video_id": video,
                "frame_index": 0, "y_true": y, "attack_raw": None if y == 0 else "fixture",
                "attack_macro": "live" if y == 0 else "print", "split": split}
    live = []
    for i in range(n_live):
        png("L%04d" % i, 100)
        live.append(frame("L%04d" % i, 0, "tl%02d" % (i // 8), "TRAIN"))
    spoof = []
    for i in range(n_spoof):
        png("S%04d" % i, 150)
        spoof.append(frame("S%04d" % i, 1, "ts%02d" % (i // 8), "TRAIN"))
    syn = []
    for i in range(n_syn):
        png("Y%04d" % i, 140)
        syn.append({"sample_id": "Y%04d" % i, "pair_id": "Y%04d" % i})
    val = []
    for v in range(n_val_videos):
        y = v % 2
        for f in range(frames_per_video):
            sid = "V%02d_%d" % (v, f)
            png(sid, int(rng.integers(80, 170)), 20 + 30 * y)   # label-independent brightness, label-dependent texture
            val.append(dict(frame(sid, y, "vv%02d" % v, "VAL"), frame_index=f))
    return store, live, spoof, syn, val


def loader(store):
    from PIL import Image

    def load(item):
        with Image.open(io.BytesIO(store[item["sample_id"]])) as im:
            return im.copy()
    return load


def q1(R, S):
    import numpy as np
    import torch
    from PIL import Image
    rng = np.random.Generator(np.random.PCG64(7))
    g1, g2 = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    bad = 0
    for _ in range(300):
        img = Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), "RGB")
        a = R.train_transform(img, g1)
        b = S.apply_aug(img, S.draw_aug_params(g2))
        bad += not torch.equal(a, b)
    return {"images": 300, "mismatches": bad, "generator_states_equal": torch.equal(g1.get_state(), g2.get_state()),
            "pass": bad == 0 and torch.equal(g1.get_state(), g2.get_state())}


def steps_of(d):
    return [json.loads(x) for x in (Path(d) / "steps.jsonl").read_text().splitlines()]


def q2(R, S, out, weights):
    from gpatbench.downstream import metrics as M
    res, skipped = {}, []
    for m in ("E00", "EQ1"):
        for fseed in range(20261125, 20261135):
            store, live, spoof, syn, val = fixture(seed=fseed)
            load = loader(store)
            sy = [] if m == "E00" else syn
            try:
                R.train_detector(method_id=m, seed=42, live=live, real_spoof=spoof, synthetic=sy, val_select=val,
                                 holdout=val, load_real=load, load_synthetic=load, weights_path=weights, epochs=2,
                                 warmup_epochs=1, out_dir=out / ("smoke_%s_f%d" % (m, fseed)),
                                 split_manifest_sha256="0" * 64)
                break
            except M.UnresolvedThresholdTie as exc:
                skipped.append({"method": m, "fixture_seed": fseed, "reason": str(exc)})
        (out / ("smoke_" + m)).symlink_to(out / ("smoke_%s_f%d" % (m, fseed)))
        smoke_steps = steps_of(out / ("smoke_" + m))
        smoke_epochs = json.loads((out / ("smoke_" + m) / "epochs.json").read_text())
        smoke_det = json.loads((out / ("smoke_" + m) / "detector.json").read_text())
        for w in (0, 4):
            d = out / ("science_cpu_%s_w%d" % (m, w))
            r = S.train_detector(mode="NON_SCIENTIFIC_QUALIFICATION", method_id=m, seed=42, live=live, real_spoof=spoof,
                                 synthetic=sy, val=val, load_real=load, load_synthetic=load, weights_path=weights,
                                 out_dir=d, split_manifest_sha256="0" * 64, policy=S.POLICIES["CPU_FP32_DETERMINISTIC"],
                                 epochs=2, warmup_epochs=1, workers=w, bank_seed=None if m == "E00" else 42)
            sci = steps_of(d)
            keys = ("step", "epoch", "lr_backbone_hex", "lr_head_hex", "loss_hex", "grad_norm_hex", "n_live",
                    "n_real_spoof", "n_synthetic", "batch_ids_sha256")
            res["%s_w%d" % (m, w)] = {
                "steps_equal": [{k: s[k] for k in keys} for s in sci] == [{k: s[k] for k in keys} for s in smoke_steps],
                "epoch_states_equal": [e["state_sha256"] for e in r["epochs"]] == [e["state_sha256"] for e in smoke_epochs],
                "thresholds_equal": [e["threshold_hex"] for e in r["epochs"]] == [e["threshold_hex"] for e in smoke_epochs],
                "logits_equal": [e["logits_sha256"] for e in r["epochs"]] == [e["logits_sha256"] for e in smoke_epochs],
                "selected_epoch_equal": r["selection"]["selected_epoch"] == smoke_det["selected_epoch"],
                "steps": len(sci)}
    res["pass"] = all(all(val for kk, val in v.items() if kk != "steps") and v["steps"] > 0 for v in res.values())
    res["fixture_seeds_skipped_for_threshold_tie"] = skipped
    return res


def q3(S, out, weights):
    import torch
    from gpatbench.downstream import metrics as M
    res, skipped = {}, []
    for name in ("CUDA_FP32_DETERMINISTIC", "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "CUDA_AMP_BF16_DETERMINISTIC"):
        for fseed in range(20261125, 20261135):
            store, live, spoof, syn, val = fixture(seed=fseed)
            load = loader(store)
            runs = []
            try:
                for rep in ("a", "b"):
                    d = out / ("science_%s_f%d_%s" % (name, fseed, rep))
                    r = S.train_detector(mode="NON_SCIENTIFIC_QUALIFICATION", method_id="EQ1", seed=1337, live=live,
                                         real_spoof=spoof, synthetic=syn, val=val, load_real=load, load_synthetic=load,
                                         weights_path=weights, out_dir=d, split_manifest_sha256="0" * 64,
                                         policy=S.POLICIES[name], epochs=2, warmup_epochs=1, workers=4, bank_seed=1337)
                    runs.append((r, steps_of(d), d))
                break
            except M.UnresolvedThresholdTie as exc:
                skipped.append({"policy": name, "fixture_seed": fseed, "reason": str(exc)})
        (out / ("science_%s_a" % name)).symlink_to(runs[0][2])
        (ra, sa, da), (rb, sb, db) = runs
        res[name] = {"fixture_seed": fseed,
                     "replay_steps_bitwise": sa == sb,
                     "replay_states_bitwise": [e["state_sha256"] for e in ra["epochs"]] == [e["state_sha256"] for e in rb["epochs"]],
                     "replay_val_logits_bitwise": [e["logits_sha256"] for e in ra["epochs"]] == [e["logits_sha256"] for e in rb["epochs"]],
                     "checkpoint_reloaded": ra["selection"]["checkpoint_state_sha256"] == ra["epochs"][ra["selection"]["selected_epoch"]]["state_sha256"],
                     "checkpoint_file_sha256_equal": ra["selection"]["checkpoint_file_sha256"] == rb["selection"]["checkpoint_file_sha256"],
                     "finite": all(s["loss_hex"] not in ("nan", "inf") for s in sa),
                     "non_finite_grad_steps": sum(1 for s in sa if s["grad_norm_hex"] in ("inf", "nan", "-inf")),
                     "grad_scale_last": sa[-1]["grad_scale"], "selected_epoch": ra["selection"]["selected_epoch"],
                     "label": ra["selection"]["label"]}
    res["pass"] = all(v["replay_steps_bitwise"] and v["replay_states_bitwise"] and v["replay_val_logits_bitwise"]
                      and v["checkpoint_reloaded"] and v["checkpoint_file_sha256_equal"] and v["finite"]
                      and v["label"] == "NON_SCIENTIFIC_QUALIFICATION" for k, v in res.items())
    res["fixture_seeds_skipped_for_threshold_tie"] = skipped
    res["cuda_device"] = torch.cuda.get_device_name(0)
    return res


def q4(R, S, weights):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    out = {"gpu_train_step_seconds": {}, "note": "synthetic tensors / PNGs; shared GPU (unrelated jobs recorded); "
           "provisional resource evidence only"}
    for name in ("CUDA_FP32_DETERMINISTIC", "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "CUDA_AMP_BF16_DETERMINISTIC"):
        pol = S.POLICIES[name]
        model = R.build_model(weights, 0).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda") if pol.grad_scaler else None
        x = torch.randn(64, 3, 224, 224, device="cuda")
        y = (torch.arange(64, device="cuda") >= 32).float()
        times = []
        for i in range(40):
            torch.cuda.synchronize()
            t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with S._autocast(pol):
                lo = model(x).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(lo.float(), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out["gpu_train_step_seconds"][name] = {"median_after_10": float(np.median(times[10:])),
                                               "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2 ** 20}
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                with S._autocast(pol):
                    model(x)
            torch.cuda.synchronize()
        out.setdefault("gpu_eval_64_seconds", {})[name] = (time.perf_counter() - t) / 10
        del model, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.Generator(np.random.PCG64(3))
    pngs = []
    for _ in range(64):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), "RGB").save(buf, "PNG", compress_level=6)
        pngs.append(buf.getvalue())
    g = torch.Generator().manual_seed(1)

    def one(j):
        raw, p = j
        with Image.open(io.BytesIO(raw)) as im:
            img = im.copy()
        hashlib.sha256(raw).hexdigest()
        return S.apply_aug(img, p)
    out["cpu_batch64_decode_augment_seconds"] = {}
    for w in (0, 4, 8):
        pool = ThreadPoolExecutor(w) if w else None
        ts = []
        for _ in range(5):
            jobs = [(r, S.draw_aug_params(g)) for r in pngs]
            t = time.perf_counter()
            torch.stack(S._map(pool, one, jobs))
            ts.append(time.perf_counter() - t)
        if pool:
            pool.shutdown()
        out["cpu_batch64_decode_augment_seconds"][str(w)] = float(np.median(ts))
    out["note_noise"] = "random-noise PNGs compress worse than faces; real decode cost is lower or equal"
    return out


def q5(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gpatbench.downstream import metrics as M, plots, tables
    d = out / "science_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC_a"
    ep = json.loads((d / "epochs.json").read_text())
    rows = [{"epoch": e["epoch"], "val_acer": float(M.Fraction(e["val_acer"])), "val_auc": float(M.Fraction(e["val_auc"]))}
            for e in ep]
    plots.validate_plot_table(rows, ["val_acer", "val_auc"], template_mode=False)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([r["epoch"] for r in rows], [r["val_acer"] for r in rows], marker="o", label="VAL ACER")
    ax.plot([r["epoch"] for r in rows], [r["val_auc"] for r in rows], marker="s", label="VAL AUC")
    ax.set_title("NON-SCIENTIFIC fixture qualification")
    ax.legend()
    fig.savefig(out / "qualification_val_curves.png", dpi=80)
    sel = json.loads((d / "selection.json").read_text())
    pred = tables.read_predictions(d / ("predictions_val_epoch%02d.parquet" % sel["selected_epoch"]))
    videos = M.aggregate_videos(pred)
    pts = M.curve_points([v["y_true"] for v in videos], [v["video_score"] for v in videos], rule="SPOOF_IF_SCORE_GE_THRESHOLD")
    roc = [{"FPR": float(p["FPR"]), "TPR": float(p["TPR"])} for p in pts]
    plots.validate_plot_table(roc, ["FPR", "TPR"], template_mode=False)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot([r["FPR"] for r in roc], [r["TPR"] for r in roc], drawstyle="steps-post")
    ax.set_title("NON-SCIENTIFIC fixture VAL ROC")
    fig.savefig(out / "qualification_val_roc.png", dpi=80)
    return {"plots": sorted(p.name for p in out.glob("*.png")), "pass": len(list(out.glob("*.png"))) == 2}


def q6():
    """The frozen section-15 rule is incomplete for degenerate VAL scores: constant or fully inverted video scores
    leave thresholds 0 and 1 tied on ACER 1/2 and equally close to 0.5 (owner decision required)."""
    from gpatbench.downstream import metrics as M
    out = {}
    for name, scores in (("constant", [0.5] * 8), ("inverted", [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])):
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        try:
            M.select_threshold(y, scores, rule="SPOOF_IF_SCORE_GE_THRESHOLD")
            out[name] = "selected"
        except M.UnresolvedThresholdTie as exc:
            out[name] = "UnresolvedThresholdTie: %s" % exc
    out["finding"] = "spec 15 / selection.threshold gives no tie-break beyond 'closest to 0.5'; a degenerate epoch " \
                     "aborts the run (fail closed) until the owner pins a rule"
    out["pass"] = all(str(v).startswith("UnresolvedThresholdTie") for k, v in out.items() if k in ("constant", "inverted"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=False, exist_ok=False)
    from gpatbench.downstream import resnet_runner as R, science_runner as S
    S.configure(S.POLICIES["CPU_FP32_DETERMINISTIC"])
    import torch
    doc = {"label": "NON_SCIENTIFIC_QUALIFICATION (synthetic fixtures; no real or bank image, no VAL/TEST data)",
           "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "cuda": torch.version.cuda,
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    doc["Q1_augmentation_equivalence"] = q1(R, S)
    doc["Q2_cpu_equivalence_with_smoke_runner"] = q2(R, S, out, a.weights)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    doc["Q3_cuda_policies_replay"] = q3(S, out, a.weights)
    doc["Q4_throughput"] = q4(R, S, a.weights)
    doc["Q5_plots"] = q5(S, out)
    doc["Q6_threshold_tie_finding"] = q6()
    doc["pass"] = all(doc[k]["pass"] for k in ("Q1_augmentation_equivalence", "Q2_cpu_equivalence_with_smoke_runner",
                                               "Q3_cuda_policies_replay", "Q5_plots"))
    print(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
