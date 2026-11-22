"""ResNet-18 downstream SMOKE runner (owner-approved m8_downstream_evaluator_contract.yaml; CPU, no AMP).

Frozen (spec 14 / downstream_resnet18.yaml): torchvision ResNet-18 IMAGENET1K_V1 (resnet18-f37072fd.pth, hash-checked),
fc -> Linear(512, 1), BCEWithLogitsLoss, batches 32 live + 16 real spoof + 16 synthetic (E00: 32 + 32), AdamW with
backbone lr 1e-4 / head lr 5e-4, weight decay 1e-4, warmup then cosine to 1e-6, gradient clip 1.0, train augmentation
RandomResizedCrop(224, (0.92, 1.0), (0.95, 1.05), BICUBIC) -> HorizontalFlip 0.5 -> ColorJitter(0.05, 0.05, 0.05, 0.01)
with p 0.5 -> ImageNet normalization.

Approved pins: D1 LIVE_PASS sampler (fresh live permutation per epoch; spoof/synthetic pools without replacement,
remainder discarded and next permutation at the boundary; batch order live, real spoof, synthetic), D2 full bicubic
resize to 224 for VAL/holdout, D3 target = label_binary, D4 spoof iff score >= threshold, D5 VAL after every epoch at
the epoch's own section-15 threshold (min ACER, max AUC, earliest), D6 lr_g(k) = base_g*k/W in warmup then cosine,
D13 CPU deterministic settings, D14 order with every random draw from one dedicated generator, rng_scheme for model,
sampler and augmentation. The smoke schedule (2 epochs, 1 warmup epoch) is passed in; it never replaces the frozen
40-epoch schedule.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import metrics as M

RULE = "SPOOF_IF_SCORE_GE_THRESHOLD"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHTS_FILE = "resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
FROZEN = {"batch_size": 64, "mixed": {"live": 32, "real_spoof": 16, "synthetic": 16},
          "real_only": {"live": 32, "real_spoof": 32}, "lr_backbone": 1.0e-4, "lr_head": 5.0e-4,
          "weight_decay": 1.0e-4, "min_lr": 1.0e-6, "grad_clip_norm": 1.0}
THREADS = 8


class RunnerError(RuntimeError):
    pass


def configure_determinism():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)


def sha_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def permutation(method: str, seed: int, pool: str, pass_index: int, n: int) -> list[int]:
    g = np.random.Generator(np.random.PCG64(
        sha_int(f"gpatbench.downstream.sampler.v1|resnet18|{method}|{seed}|{pool}|{pass_index}")))
    return [int(i) for i in g.permutation(n)]


class Pool:
    """Without-replacement draws; when fewer than k remain, the remainder is discarded and the next permutation drawn."""

    def __init__(self, items, method, seed, name):
        if not items:
            raise RunnerError(f"empty pool {name}")
        self.items, self.method, self.seed, self.name = list(items), method, seed, name
        self.pass_index, self.ptr = 0, 0
        self.order = permutation(method, seed, name, 0, len(self.items))
        self.discarded = 0

    def take(self, k):
        if k > len(self.items):
            raise RunnerError(f"pool {self.name} has {len(self.items)} < {k} items")
        if len(self.order) - self.ptr < k:
            self.discarded += len(self.order) - self.ptr
            self.pass_index += 1
            self.order, self.ptr = permutation(self.method, self.seed, self.name, self.pass_index, len(self.items)), 0
        out = [self.items[i] for i in self.order[self.ptr:self.ptr + k]]
        self.ptr += k
        return out


def lr_at(k: int, base: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    """D6: base*k/W for k < W (first step lr 0); then min_lr + (base-min_lr)*0.5*(1+cos(pi*(k-W)/(T-W)))."""
    if k < warmup_steps:
        return base * k / warmup_steps
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (k - warmup_steps) / (total_steps - warmup_steps)))


def train_transform(img, g):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    w, h = img.size
    area = w * h
    lo_r, hi_r = math.log(0.95), math.log(1.05)
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(0.92, 1.00, generator=g).item()
        aspect = math.exp(torch.empty(1).uniform_(lo_r, hi_r, generator=g).item())
        cw, ch = int(round(math.sqrt(target_area * aspect))), int(round(math.sqrt(target_area / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            i = int(torch.randint(0, h - ch + 1, (1,), generator=g).item())
            j = int(torch.randint(0, w - cw + 1, (1,), generator=g).item())
            break
    else:                                                     # torchvision fallback: central crop
        r = w / h
        if r < 0.95:
            cw, ch = w, int(round(w / 0.95))
        elif r > 1.05:
            ch, cw = h, int(round(h * 1.05))
        else:
            cw, ch = w, h
        i, j = (h - ch) // 2, (w - cw) // 2
    img = TF.resized_crop(img, i, j, ch, cw, [224, 224], interpolation=InterpolationMode.BICUBIC)
    if torch.rand(1, generator=g).item() < 0.5:
        img = TF.hflip(img)
    if torch.rand(1, generator=g).item() < 0.5:
        order = torch.randperm(4, generator=g).tolist()
        b = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        c = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        s = torch.empty(1).uniform_(0.95, 1.05, generator=g).item()
        hu = torch.empty(1).uniform_(-0.01, 0.01, generator=g).item()
        for fn in order:
            img = (TF.adjust_brightness(img, b) if fn == 0 else TF.adjust_contrast(img, c) if fn == 1
                   else TF.adjust_saturation(img, s) if fn == 2 else TF.adjust_hue(img, hu))
    return TF.normalize(TF.to_tensor(img), MEAN, STD)


def eval_transform(img):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    return TF.normalize(TF.to_tensor(TF.resize(img, [224, 224], interpolation=InterpolationMode.BICUBIC)), MEAN, STD)


def state_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build_model(weights_path: str | Path, seed: int):
    import torch
    import torchvision
    raw = Path(weights_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
        raise RunnerError("ResNet-18 weight SHA256 differs from the frozen value")
    torch.manual_seed(seed)                                   # rng_scheme: seeds the new fc head
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    if model.fc.in_features != 512:
        raise RunnerError("unexpected ResNet-18 head")
    model.fc = torch.nn.Linear(512, 1)
    return model


def _predict(model, frames, load, batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            x = torch.stack([eval_transform(load(f)) for f in chunk])
            logits = model(x).squeeze(1).tolist()
            for f, z in zip(chunk, logits):
                out.append({**f, "logit_hex": float(z).hex(), "p_frame": M.sigmoid(float(z))})
    return out


def prediction_rows(scored, threshold, *, method_id, seed, checkpoint_sha256, split_manifest_sha256):
    videos = {(v["dataset"], v["video_id"]): v for v in M.aggregate_videos(scored)}
    rows = []
    for f in sorted(scored, key=lambda r: r["sample_id"]):
        v = videos[(f["dataset"], f["video_id"])]
        rows.append({"sample_id": f["sample_id"], "dataset": f["dataset"], "subject_id_global": f["subject_id_global"],
                     "video_id": f["video_id"], "frame_index": f["frame_index"], "y_true": f["y_true"],
                     "attack_raw": f["attack_raw"], "attack_macro": f["attack_macro"], "p_frame": f["p_frame"],
                     "threshold": threshold, "pred_frame": M.predict_spoof([f["p_frame"]], threshold, rule=RULE)[0],
                     "video_score": v["video_score"],
                     "pred_video": M.predict_spoof([v["video_score"]], threshold, rule=RULE)[0],
                     "method_id": method_id, "track_id": "binary", "evaluator_id": "resnet18", "seed": seed,
                     "checkpoint_sha256": checkpoint_sha256, "split_manifest_sha256": split_manifest_sha256})
    return rows


def write_parquet(rows, path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    if path.exists():
        raise RunnerError(f"refusing to overwrite {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_detector(*, method_id: str, seed: int, live, real_spoof, synthetic, val_select, holdout, load_real,
                   load_synthetic, weights_path, epochs: int, warmup_epochs: int, out_dir: Path,
                   split_manifest_sha256: str) -> dict:
    """Train one smoke detector and write its evidence into out_dir (which must not exist)."""
    import torch
    import torch.nn.functional as F
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    real_only = method_id == "E00"
    comp = FROZEN["real_only"] if real_only else FROZEN["mixed"]
    if real_only and synthetic:
        raise RunnerError("E00 is real-only")
    if not real_only and not synthetic:
        raise RunnerError(f"{method_id} needs synthetic samples")
    steps_per_epoch = len(live) // comp["live"]
    if steps_per_epoch < 1:
        raise RunnerError("fewer live frames than one full batch")
    W, T = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    model = build_model(weights_path, seed)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    bases = [FROZEN["lr_backbone"], FROZEN["lr_head"]]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 0.0}, {"params": head, "lr": 0.0}],
                            weight_decay=FROZEN["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(sha_int(f"gpatbench.downstream.augment.v1|resnet18|{method_id}|{seed}") % 2 ** 63)
    spoof_pool = Pool(real_spoof, method_id, seed, "real_spoof")
    syn_pool = None if real_only else Pool(synthetic, method_id, seed, "synthetic")
    steps, epochs_log, states, k = [], [], [], 0
    for epoch in range(epochs):
        order = permutation(method_id, seed, "live", epoch, len(live))
        for b in range(steps_per_epoch):
            lv = [live[i] for i in order[b * comp["live"]:(b + 1) * comp["live"]]]
            rs = spoof_pool.take(comp["real_spoof"])
            sy = [] if real_only else syn_pool.take(comp["synthetic"])
            imgs = [train_transform(load_real(f), gen) for f in lv + rs] + [train_transform(load_synthetic(s), gen) for s in sy]
            y = torch.tensor([0.0] * len(lv) + [1.0] * (len(rs) + len(sy)))
            x = torch.stack(imgs)
            lrs = [lr_at(k, base, W, T, FROZEN["min_lr"]) for base in bases]
            for g_, lr in zip(opt.param_groups, lrs):
                g_["lr"] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            if not torch.isfinite(loss):
                raise RunnerError(f"non-finite loss at step {k}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN["grad_clip_norm"])
            opt.step()
            ids = [f["sample_id"] for f in lv + rs] + [s["pair_id"] for s in sy]
            steps.append({"step": k, "epoch": epoch, "lr_backbone_hex": lrs[0].hex(), "lr_head_hex": lrs[1].hex(),
                          "loss_hex": float(loss.item()).hex(), "grad_norm_hex": float(gn.item()).hex(),
                          "n_live": len(lv), "n_real_spoof": len(rs), "n_synthetic": len(sy), "batch": len(ids),
                          "batch_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})
            k += 1
        state = {n: t.detach().clone() for n, t in model.state_dict().items()}
        ssha = state_sha256(state)
        scored = _predict(model, val_select, load_real)
        videos = M.aggregate_videos(scored)
        yv, sv = [v["y_true"] for v in videos], [v["video_score"] for v in videos]
        sel = M.select_threshold(yv, sv, rule=RULE)
        auc = M.auc(yv, sv)
        rows = prediction_rows(scored, sel["threshold"], method_id=method_id, seed=seed, checkpoint_sha256=ssha,
                               split_manifest_sha256=split_manifest_sha256)
        psha = write_parquet(rows, out_dir / f"predictions_val_select_epoch{epoch}.parquet")
        epochs_log.append({"epoch": epoch, "state_sha256": ssha, "threshold_hex": float(sel["threshold"]).hex(),
                           "val_acer": str(sel["val_acer"]), "val_auc": str(auc), "val_videos": len(videos),
                           "val_frames": len(scored), "predictions_sha256": psha,
                           "logits_sha256": hashlib.sha256("\n".join(r["logit_hex"] for r in scored).encode()).hexdigest()})
        states.append(state)
    best = min(range(epochs), key=lambda e: (M.Fraction(epochs_log[e]["val_acer"]), -M.Fraction(epochs_log[e]["val_auc"]), e))
    model.load_state_dict(states[best])
    threshold = float.fromhex(epochs_log[best]["threshold_hex"])
    scored = _predict(model, holdout, load_real)
    rows = prediction_rows(scored, threshold, method_id=method_id, seed=seed,
                           checkpoint_sha256=epochs_log[best]["state_sha256"], split_manifest_sha256=split_manifest_sha256)
    hsha = write_parquet(rows, out_dir / "predictions_holdout.parquet")
    record = {"method_id": method_id, "seed": seed, "label": "SMOKE_NON_SCIENTIFIC", "epochs": epochs,
              "warmup_epochs": warmup_epochs, "steps_per_epoch": steps_per_epoch, "total_steps": k, "batch": comp,
              "selected_epoch": best, "frozen_threshold_hex": float(threshold).hex(), "holdout_predictions_sha256": hsha,
              "holdout_frames": len(rows), "pool_passes": {"real_spoof": spoof_pool.pass_index,
                                                          "synthetic": None if real_only else syn_pool.pass_index},
              "pool_discarded": {"real_spoof": spoof_pool.discarded, "synthetic": None if real_only else syn_pool.discarded},
              "weights_sha256": WEIGHTS_SHA256, "amp": False, "device": "cpu",
              "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    (out_dir / "steps.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in steps))
    (out_dir / "epochs.json").write_text(json.dumps(epochs_log, indent=1, sort_keys=True) + "\n")
    (out_dir / "detector.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    (out_dir / "timing.json").write_text(json.dumps({"wall_seconds": time.monotonic() - t0}) + "\n")
    return record
