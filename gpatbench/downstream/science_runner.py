"""ResNet-18 downstream detector runner for the M10 scientific matrix (E00 + seven previous-work baselines).

Frozen semantics are those of the qualified smoke runner (gpatbench/downstream/resnet_runner.py, owner-approved
m8_downstream_evaluator_contract.yaml D1-D14, spec 14-16, configs/frozen/downstream_resnet18.yaml); this module reuses
its functions unchanged (permutation, Pool, lr_at, eval_transform, build_model, state_sha256, prediction_rows,
write_parquet, metrics) and adds only what a full scientific run needs:

  - device / precision: a named PrecisionPolicy. D13 fixes 'Science: CUDA with AMP' but leaves the AMP dtype and
    GradScaler policy to be pinned before M10; SCIENTIFIC mode therefore refuses unless an owner-activated pin
    (configs/amendments/m10_resnet18_science_precision.yaml) names the policy. NON_SCIENTIFIC_QUALIFICATION accepts any
    policy and labels every output non-scientific.
  - augmentation: the D14 parameters are drawn from the dedicated generator in sample order exactly as
    resnet_runner.train_transform draws them (draw_aug_params), then applied (apply_aug), so decoding/augmentation of a
    batch may run in a thread pool without changing any random draw (qualified bitwise against train_transform).
  - evaluation: the full VAL population after every epoch (D5), never TEST; there is no TEST / holdout input.
  - checkpoints: every epoch's state SHA256 is recorded; retention SELECTED_ONLY keeps the best-so-far state in memory
    and writes the selected checkpoint once, ALL_EPOCHS writes every epoch (owner retention decision); the written
    selected checkpoint is reloaded (weights_only) and its state SHA256 re-verified.
  - selection record: selected epoch, checkpoint file and state SHA256, VAL ACER / AUC, the VAL threshold (frozen for a
    later, separately authorized TEST stage), metric implementation identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import metrics as M
from . import resnet_runner as R

MODES = ("SCIENTIFIC", "NON_SCIENTIFIC_QUALIFICATION")
# prediction-row track_id (spec 22.2): the A1 track of each method from outputs/audit/method_status.csv; E00 = spec 17
# "Binary" (the smoke runner's fixed value). The detector itself is always the binary live/spoof ResNet-18.
TRACKS = {"E00": "binary", "E01": "A_FAIR_COMMON_IDENTITY_FREE", "E02": "A_FAIR_COMMON_IDENTITY_FREE",
          "E03": "A_FAIR_COMMON_IDENTITY_FREE", "E04": "A_FAIR_COMMON_IDENTITY_FREE", "E05": "A_FAIR_COMMON_IDENTITY_FREE",
          "E06b": "B_NATIVE_FULL_SECONDARY", "E06c": "A_FAIR_COMMON_IDENTITY_FREE"}
PRECISION_PIN_PATH = "configs/amendments/m10_resnet18_science_precision.yaml"
RETENTION = ("SELECTED_ONLY", "ALL_EPOCHS")
FROZEN_EPOCHS, FROZEN_WARMUP = 40, 3
METRIC_IMPLEMENTATION = {"module": "gpatbench/downstream/metrics.py", "threshold": "select_threshold",
                         "auc": "auc (exact Mann-Whitney, ties 1/2)", "rule": R.RULE}


class ScienceRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    device: str                  # "cpu" | "cuda"
    autocast_dtype: str | None   # None | "float16" | "bfloat16"
    grad_scaler: bool
    deterministic: bool          # torch.use_deterministic_algorithms + cudnn deterministic, benchmark off

    def record(self):
        return dict(self.__dict__)


POLICIES = {
    "CPU_FP32_DETERMINISTIC": PrecisionPolicy("CPU_FP32_DETERMINISTIC", "cpu", None, False, True),   # = smoke D13
    "CUDA_FP32_DETERMINISTIC": PrecisionPolicy("CUDA_FP32_DETERMINISTIC", "cuda", None, False, True),
    "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC": PrecisionPolicy("CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "cuda",
                                                              "float16", True, True),
    "CUDA_AMP_BF16_DETERMINISTIC": PrecisionPolicy("CUDA_AMP_BF16_DETERMINISTIC", "cuda", "bfloat16", False, True),
}
SCIENCE_ELIGIBLE = ("CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC", "CUDA_AMP_BF16_DETERMINISTIC")   # spec 14: AMP on CUDA


def load_science_precision(repo: str | Path) -> PrecisionPolicy:
    """The owner-activated science precision pin (D13); refuses when absent, unapproved or not an AMP-on-CUDA policy."""
    p = Path(repo) / PRECISION_PIN_PATH
    if not p.is_file():
        raise ScienceRunError(f"no owner precision pin at {PRECISION_PIN_PATH} (D13: AMP dtype / GradScaler must be "
                              "pinned before M10)")
    d = json.loads(p.read_text())
    if d.get("status") != "OWNER_APPROVED" or d.get("approved") is not True or d.get("policy") not in SCIENCE_ELIGIBLE:
        raise ScienceRunError("precision pin is not an owner-approved AMP-on-CUDA policy")
    return POLICIES[d["policy"]]


def configure(policy: PrecisionPolicy, threads: int = R.THREADS):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if policy.deterministic:
        torch.use_deterministic_algorithms(True)
    if policy.device == "cuda":
        if not torch.cuda.is_available():
            raise ScienceRunError("CUDA requested but unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = policy.deterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------------------------------- D14 augmentation
def draw_aug_params(g, w: int = 256, h: int = 256) -> dict:
    """Every random draw of resnet_runner.train_transform for a w x h image, in the same order, from generator g."""
    import torch
    area = w * h
    lo_r, hi_r = math.log(0.95), math.log(1.05)
    crop = None
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(0.92, 1.00, generator=g).item()
        aspect = math.exp(torch.empty(1).uniform_(lo_r, hi_r, generator=g).item())
        cw, ch = int(round(math.sqrt(target_area * aspect))), int(round(math.sqrt(target_area / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            i = int(torch.randint(0, h - ch + 1, (1,), generator=g).item())
            j = int(torch.randint(0, w - cw + 1, (1,), generator=g).item())
            crop = (i, j, ch, cw)
            break
    if crop is None:                                          # torchvision fallback: central crop
        r = w / h
        if r < 0.95:
            cw, ch = w, int(round(w / 0.95))
        elif r > 1.05:
            ch, cw = h, int(round(h * 1.05))
        else:
            cw, ch = w, h
        crop = ((h - ch) // 2, (w - cw) // 2, ch, cw)
    flip = torch.rand(1, generator=g).item() < 0.5
    jitter = None
    if torch.rand(1, generator=g).item() < 0.5:
        order = torch.randperm(4, generator=g).tolist()
        b = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        c = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        s = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        hu = torch.empty(1).uniform_(-0.01, 0.01, generator=g).item()
        jitter = (order, b, c, s, hu)
    return {"crop": crop, "flip": flip, "jitter": jitter, "size": (w, h)}


def apply_aug(img, p: dict):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    if img.size != p["size"]:
        raise ScienceRunError("augmentation parameters were drawn for another image size")
    i, j, ch, cw = p["crop"]
    img = TF.resized_crop(img, i, j, ch, cw, [224, 224], interpolation=InterpolationMode.BICUBIC)
    if p["flip"]:
        img = TF.hflip(img)
    if p["jitter"] is not None:
        order, b, c, s, hu = p["jitter"]
        for fn in order:
            img = (TF.adjust_brightness(img, b) if fn == 0 else TF.adjust_contrast(img, c) if fn == 1
                   else TF.adjust_saturation(img, s) if fn == 2 else TF.adjust_hue(img, hu))
    return TF.normalize(TF.to_tensor(img), R.MEAN, R.STD)


def _map(pool, fn, items):
    return list(pool.map(fn, items)) if pool is not None else [fn(x) for x in items]


def _predict(model, frames, load, policy, pool, batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            x = torch.stack(_map(pool, lambda f: R.eval_transform(load(f)), chunk)).to(policy.device)
            with _autocast(policy):
                logits = model(x).squeeze(1).float().cpu().tolist()
            for f, z in zip(chunk, logits):
                if not math.isfinite(z):
                    raise ScienceRunError(f"non-finite VAL logit for {f['sample_id']}")
                out.append({**f, "logit_hex": float(z).hex(), "p_frame": M.sigmoid(float(z))})
    return out


def _autocast(policy):
    import contextlib
    import torch
    if policy.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=policy.device, dtype=getattr(torch, policy.autocast_dtype))


def _rank(e):
    return (Fraction(e["val_acer"]), -Fraction(e["val_auc"]), e["epoch"])      # D5: min ACER, max AUC, earliest


def train_detector(*, mode: str, method_id: str, seed: int, live, real_spoof, synthetic, val, load_real,
                   load_synthetic, weights_path, out_dir, split_manifest_sha256: str, policy: PrecisionPolicy,
                   epochs: int = FROZEN_EPOCHS, warmup_epochs: int = FROZEN_WARMUP, retention: str = "SELECTED_ONLY",
                   workers: int = 0, bank_seed: int | None = None, track_id: str | None = None) -> dict:
    """One detector (method, detector seed). `val` = VAL frames only; there is no TEST / holdout argument."""
    import torch
    import torch.nn.functional as F
    if mode not in MODES or retention not in RETENTION or policy.name not in POLICIES:
        raise ScienceRunError("mode / retention / policy")
    if mode == "SCIENTIFIC":
        if (epochs, warmup_epochs) != (FROZEN_EPOCHS, FROZEN_WARMUP):
            raise ScienceRunError("SCIENTIFIC runs use the frozen 40-epoch / 3-warmup schedule")
        if policy.name not in SCIENCE_ELIGIBLE:
            raise ScienceRunError("SCIENTIFIC runs need an AMP-on-CUDA policy (spec 14)")
        if any(f.get("split") not in (None, "VAL") for f in val) or any(
                f.get("split") not in (None, "TRAIN") for f in list(live) + list(real_spoof)):
            raise ScienceRunError("TRAIN pools must be TRAIN and selection frames VAL")
    if mode == "SCIENTIFIC" and (method_id not in TRACKS or track_id != TRACKS[method_id]):
        raise ScienceRunError("SCIENTIFIC runs carry the method's A1 track id (TRACKS)")
    track_id = track_id or TRACKS.get(method_id, "binary")
    real_only = method_id == "E00"
    if real_only and synthetic:
        raise ScienceRunError("E00 is real-only")
    if not real_only and (not synthetic or bank_seed != seed):
        raise ScienceRunError("D12 SEED_MATCHED_RAW: detector seed s uses the bank of generator seed s")
    if not val or not ({f["y_true"] for f in val} == {0, 1}):
        raise ScienceRunError("VAL selection needs live and spoof frames")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    t0, start_utc = time.monotonic(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    comp = R.FROZEN["real_only"] if real_only else R.FROZEN["mixed"]
    steps_per_epoch = len(live) // comp["live"]
    if steps_per_epoch < 1:
        raise ScienceRunError("fewer live frames than one full batch")
    W, T = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    model = R.build_model(weights_path, seed).to(policy.device)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    bases = [R.FROZEN["lr_backbone"], R.FROZEN["lr_head"]]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 0.0}, {"params": head, "lr": 0.0}],
                            weight_decay=R.FROZEN["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda") if policy.grad_scaler else None
    gen = torch.Generator().manual_seed(R.sha_int(f"gpatbench.downstream.augment.v1|resnet18|{method_id}|{seed}") % 2 ** 63)
    spoof_pool = R.Pool(real_spoof, method_id, seed, "real_spoof")
    syn_pool = None if real_only else R.Pool(synthetic, method_id, seed, "synthetic")
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
    steps, epochs_log, k = [], [], 0
    best_state, best = None, None
    try:
        for epoch in range(epochs):
            order = R.permutation(method_id, seed, "live", epoch, len(live))
            for b in range(steps_per_epoch):
                lv = [live[i] for i in order[b * comp["live"]:(b + 1) * comp["live"]]]
                rs = spoof_pool.take(comp["real_spoof"])
                sy = [] if real_only else syn_pool.take(comp["synthetic"])
                params = [draw_aug_params(gen) for _ in range(len(lv) + len(rs) + len(sy))]     # sample order
                jobs = [(load_real, f, p) for f, p in zip(lv + rs, params)] + \
                       [(load_synthetic, s, p) for s, p in zip(sy, params[len(lv) + len(rs):])]
                x = torch.stack(_map(pool, lambda j: apply_aug(j[0](j[1]), j[2]), jobs)).to(policy.device)
                y = torch.tensor([0.0] * len(lv) + [1.0] * (len(rs) + len(sy)), device=policy.device)
                lrs = [R.lr_at(k, base, W, T, R.FROZEN["min_lr"]) for base in bases]
                for g_, lr in zip(opt.param_groups, lrs):
                    g_["lr"] = lr
                model.train()
                opt.zero_grad(set_to_none=True)
                with _autocast(policy):
                    logits = model(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
                if not torch.isfinite(loss):
                    raise ScienceRunError(f"non-finite loss at step {k}")
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), R.FROZEN["grad_clip_norm"])
                    scaler.step(opt)
                    scaler.update()
                    scale = float(scaler.get_scale())
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), R.FROZEN["grad_clip_norm"])
                    opt.step()
                    scale = None
                ids = [f["sample_id"] for f in lv + rs] + [s["sample_id"] for s in sy]
                steps.append({"step": k, "epoch": epoch, "lr_backbone_hex": lrs[0].hex(), "lr_head_hex": lrs[1].hex(),
                              "loss_hex": float(loss.item()).hex(), "grad_norm_hex": float(gn.item()).hex(),
                              "grad_scale": scale, "n_live": len(lv), "n_real_spoof": len(rs), "n_synthetic": len(sy),
                              "batch": len(ids), "batch_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})
                k += 1
            state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
            ssha = R.state_sha256(state)
            scored = _predict(model, val, load_real, policy, pool)
            videos = M.aggregate_videos(scored)
            yv, sv = [v["y_true"] for v in videos], [v["video_score"] for v in videos]
            sel = M.select_threshold(yv, sv, rule=R.RULE)          # UnresolvedThresholdTie fails the run closed
            auc = M.auc(yv, sv)
            rows = R.prediction_rows(scored, sel["threshold"], method_id=method_id, seed=seed, checkpoint_sha256=ssha,
                                     split_manifest_sha256=split_manifest_sha256)
            for r_ in rows:
                r_["track_id"] = track_id
            psha = R.write_parquet(rows, out_dir / f"predictions_val_epoch{epoch:02d}.parquet")
            e = {"epoch": epoch, "state_sha256": ssha, "threshold_hex": float(sel["threshold"]).hex(),
                 "threshold": sel["threshold"], "val_acer": str(sel["val_acer"]), "val_auc": str(auc),
                 "val_videos": len(videos), "val_frames": len(scored), "predictions_sha256": psha,
                 "logits_sha256": hashlib.sha256("\n".join(r["logit_hex"] for r in scored).encode()).hexdigest(),
                 "wall_seconds": time.monotonic() - t0}
            if retention == "ALL_EPOCHS":
                e["checkpoint_file_sha256"] = _save(state, out_dir / f"checkpoint_epoch{epoch:02d}.pt")
            epochs_log.append(e)
            if best is None or _rank(e) < _rank(best):
                best, best_state = e, state
    finally:
        if pool is not None:
            pool.shutdown()
    ck_name = f"checkpoint_epoch{best['epoch']:02d}.pt"
    ck_sha = best.get("checkpoint_file_sha256") or _save(best_state, out_dir / ck_name)
    reloaded = torch.load(out_dir / ck_name, map_location="cpu", weights_only=True)
    if R.state_sha256(reloaded) != best["state_sha256"]:
        raise ScienceRunError("selected checkpoint does not reload to its state sha256")
    selection = {"label": "SCIENTIFIC" if mode == "SCIENTIFIC" else "NON_SCIENTIFIC_QUALIFICATION",
                 "method_id": method_id, "detector_seed": seed, "bank_seed": bank_seed,
                 "selected_epoch": best["epoch"], "checkpoint_file": ck_name, "checkpoint_file_sha256": ck_sha,
                 "checkpoint_state_sha256": best["state_sha256"], "val_acer": best["val_acer"],
                 "val_auc": best["val_auc"], "frozen_val_threshold": best["threshold"],
                 "frozen_val_threshold_hex": best["threshold_hex"], "selection_rule": "D5 EPOCH_OWN_VAL_THRESHOLD: "
                 "min VAL video ACER, then max VAL video AUC, then earliest epoch (spec 14, downstream_resnet18 "
                 "selection.checkpoint min_val_video_acer_then_max_auc)", "threshold_rule": "spec 15: min VAL video "
                 "ACER over unique midpoints plus 0 and 1, tie-break nearest 0.5 (selection.threshold "
                 "min_val_video_acer_then_nearest_0.5); D4 spoof iff score >= threshold",
                 "selection_source": "VAL only (split_v1 VAL frames, video score = mean frame probability)",
                 "metric_implementation": METRIC_IMPLEMENTATION, "test_used": False}
    record = {"label": selection["label"], "method_id": method_id, "track_id": track_id, "detector_seed": seed,
              "bank_seed": bank_seed,
              "epochs": epochs, "warmup_epochs": warmup_epochs, "steps_per_epoch": steps_per_epoch, "total_steps": k,
              "batch": comp, "precision_policy": policy.record(), "retention": retention, "workers": workers,
              "pool_passes": {"real_spoof": spoof_pool.pass_index, "synthetic": None if real_only else syn_pool.pass_index},
              "pool_discarded": {"real_spoof": spoof_pool.discarded, "synthetic": None if real_only else syn_pool.discarded},
              "weights_sha256": R.WEIGHTS_SHA256, "torch": torch.__version__, "cuda": torch.version.cuda,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "start_utc": start_utc, "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "wall_seconds": time.monotonic() - t0}
    (out_dir / "steps.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in steps))
    (out_dir / "epochs.json").write_text(json.dumps(epochs_log, indent=1, sort_keys=True) + "\n")
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=1, sort_keys=True) + "\n")
    (out_dir / "detector.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return {"record": record, "selection": selection, "epochs": epochs_log}


def _save(state, path: Path) -> str:
    import torch
    if path.exists():
        raise ScienceRunError(f"refusing to overwrite {path}")
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
