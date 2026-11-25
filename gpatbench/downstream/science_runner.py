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
  - checkpoints (owner decision O3 A_SELECTED_ONLY_HDD_NEW_ROOT): every epoch's state SHA256 is recorded; whenever an
    epoch becomes the D5 best so far its state is written durably (temp file, fsync, atomic rename), reloaded
    (weights_only) and its state SHA256 verified, the best_so_far.json pointer is atomically replaced, and only then is
    the previous best-so-far checkpoint removed. At the end exactly the selected checkpoint remains (ALL_EPOCHS is
    non-scientific only).
  - threshold (owner decision O2 A_SMALLER_THRESHOLD): m10_selection.select_threshold_o2 (spec 15 completed by the
    smaller-threshold final tie-break); metrics.select_threshold is unchanged.
  - precision (owner decision O1): SCIENTIFIC needs CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC with the pinned GradScaler
    (65536, 2, 0.5, 2000) and determinism flags; unscale_ before clip_grad_norm_. Every iteration records the grad
    scale before/after, whether the optimizer update was applied and the pre/post-clip gradient norms. The D6 schedule
    index advances per iteration; a skipped update never changes LR, batch or iteration count. Fail closed on a
    non-finite loss, a non-finite gradient at grad scale <= 1, or an epoch without an applied update.
  - evidence is written incrementally (steps.jsonl fsynced per epoch, epochs.json atomically per epoch) and a failure
    writes failure.json without removing anything.
  - selection record: selected epoch, checkpoint file and state SHA256, VAL ACER / AUC, the VAL threshold (frozen for a
    later, separately authorized TEST stage), the selection reasoning, metric implementation identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import m10_selection as S10
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
METRIC_IMPLEMENTATION = {"module": "gpatbench/downstream/metrics.py + gpatbench/downstream/m10_selection.py",
                         "threshold": "m10_selection.select_threshold_o2 (spec 15 + O2 A_SMALLER_THRESHOLD)",
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
O1_POLICY = "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"     # owner decision O1 (m10_resnet18_owner_decisions.yaml)


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
    if torch.get_num_interop_threads() != 1:          # settable once per process
        torch.set_num_interop_threads(1)
    if policy.deterministic:
        torch.use_deterministic_algorithms(True)
    if policy.device == "cuda":
        if not torch.cuda.is_available():
            raise ScienceRunError("CUDA requested but unavailable")
        if policy.deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ScienceRunError("deterministic CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (O1)")
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


def _predict(model, frames, load, policy, pool, batch=64, obs=None):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            x = torch.stack(_map(pool, lambda f: R.eval_transform(load(f)), chunk)).to(policy.device)
            with _autocast(policy):
                out_t = model(x).squeeze(1)
            if obs is not None and "val_logits_dtype" not in obs:
                obs["val_logits_dtype"] = str(out_t.dtype)
            logits = out_t.float().cpu().tolist()
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
    return S10.checkpoint_rank(e)                     # D5: min ACER, max AUC, earliest


# ------------------------------------------------------------------------------------- durable evidence writes
def _fsync_dir(d: Path):
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes):
    """temp file in the same directory -> fsync -> os.replace -> fsync(dir)."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise ScienceRunError(f"stale temporary file {tmp}")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, obj):
    _atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _persist_checkpoint(state: dict, state_sha: str, epoch: int, d: Path) -> dict:
    """Write one epoch state durably (temp + fsync + atomic rename), reload it weights_only and verify its state
    SHA256. Raises before anything else is touched if the new checkpoint is not valid."""
    import torch
    name = f"checkpoint_epoch{epoch:02d}.pt"
    final, tmp = d / name, d / (name + ".tmp")
    if final.exists() or tmp.exists():
        raise ScienceRunError(f"refusing to overwrite {final}")
    with open(tmp, "xb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    _fsync_dir(d)
    fsha = _file_sha256(final)
    reloaded = torch.load(final, map_location="cpu", weights_only=True)
    if R.state_sha256(reloaded) != state_sha:
        raise ScienceRunError(f"checkpoint {name} does not reload to its state sha256")
    return {"epoch": epoch, "checkpoint_file": name, "checkpoint_file_sha256": fsha, "state_sha256": state_sha,
            "reload_verified": True}


def _scaler_config(scaler) -> dict | None:
    if scaler is None:
        return None
    sd = scaler.state_dict()
    return {"init_scale": float(sd["scale"]), "growth_factor": float(sd["growth_factor"]),
            "backoff_factor": float(sd["backoff_factor"]), "growth_interval": int(sd["growth_interval"]),
            "enabled": scaler.is_enabled()}


def determinism_record() -> dict:
    import torch
    return {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


O1_DETERMINISM = {"use_deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                  "matmul_allow_tf32": False, "cudnn_allow_tf32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
O1_SCALER = {"init_scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "enabled": True}
QUALIFICATION_OVERRIDES = ("grad_scaler_init_scale", "nonfinite_loss_at_step", "nonfinite_grad_at_step")


def _post_clip_norm(params) -> float:
    import torch
    norms = [torch.linalg.vector_norm(p.grad.detach().float()) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0


def train_detector(*, mode: str, method_id: str, seed: int, live, real_spoof, synthetic, val, load_real,
                   load_synthetic, weights_path, out_dir, split_manifest_sha256: str, policy: PrecisionPolicy,
                   epochs: int = FROZEN_EPOCHS, warmup_epochs: int = FROZEN_WARMUP, retention: str = "SELECTED_ONLY",
                   workers: int = 0, bank_seed: int | None = None, track_id: str | None = None,
                   qualification_overrides: dict | None = None) -> dict:
    """One detector (method, detector seed). `val` = VAL frames only; there is no TEST / holdout argument.

    Evidence in out_dir (created here, must not exist): steps.jsonl (appended and fsynced every epoch), epochs.json and
    best_so_far.json (atomically replaced every epoch), predictions_val_epochNN.parquet (every epoch), the retained
    checkpoint(s), selection.json and detector.json at the end, failure.json on any failure (partial evidence kept).
    """
    import torch
    import torch.nn.functional as F
    if mode not in MODES or retention not in RETENTION or policy.name not in POLICIES:
        raise ScienceRunError("mode / retention / policy")
    ov = dict(qualification_overrides or {})
    if ov and (mode == "SCIENTIFIC" or set(ov) - set(QUALIFICATION_OVERRIDES)):
        raise ScienceRunError("qualification overrides are refused in SCIENTIFIC mode / unknown override")
    if mode == "SCIENTIFIC":
        if (epochs, warmup_epochs) != (FROZEN_EPOCHS, FROZEN_WARMUP):
            raise ScienceRunError("SCIENTIFIC runs use the frozen 40-epoch / 3-warmup schedule")
        if policy.name != O1_POLICY:
            raise ScienceRunError(f"SCIENTIFIC runs use the owner-approved O1 policy {O1_POLICY}")
        if retention != "SELECTED_ONLY":
            raise ScienceRunError("O3: SCIENTIFIC runs retain the selected checkpoint only")
        if determinism_record() != O1_DETERMINISM:
            raise ScienceRunError(f"O1 determinism settings not in force: {determinism_record()}")
        if any(f.get("split") != "VAL" for f in val) or any(
                f.get("split") != "TRAIN" for f in list(live) + list(real_spoof)):
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
    out_dir.mkdir(parents=False, exist_ok=False)
    t0, start_utc = time.monotonic(), _utc()
    comp = R.FROZEN["real_only"] if real_only else R.FROZEN["mixed"]
    steps_per_epoch = len(live) // comp["live"]
    if steps_per_epoch < 1:
        raise ScienceRunError("fewer live frames than one full batch")
    W, T = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    det_before = determinism_record()
    model = R.build_model(weights_path, seed).to(policy.device)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    names = {id(p): n for n, p in model.named_parameters()}
    bases = [R.FROZEN["lr_backbone"], R.FROZEN["lr_head"]]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 0.0}, {"params": head, "lr": 0.0}],
                            weight_decay=R.FROZEN["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    optimizer_record = {"class": type(opt).__name__, "groups": [
        {"name": gname, "base_lr": base, "n_tensors": len(g["params"]),
         "n_parameters": int(sum(p.numel() for p in g["params"])),
         "names_sha256": hashlib.sha256("\n".join(names[id(p)] for p in g["params"]).encode()).hexdigest(),
         **{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k not in ("params", "lr")}}
        for gname, base, g in zip(("backbone", "head"), bases, opt.param_groups)],
        "head_names": [names[id(p)] for p in head], "grad_clip_norm": R.FROZEN["grad_clip_norm"],
        "zero_grad_set_to_none": True}
    scheduler_record = {"rule": "D6 PER_STEP_FROM_ZERO", "warmup_steps": W, "total_steps": T,
                        "min_lr": R.FROZEN["min_lr"], "bases": {"backbone": bases[0], "head": bases[1]}}
    scaler = None
    if policy.grad_scaler:
        scaler = torch.amp.GradScaler("cuda", init_scale=float(ov.get("grad_scaler_init_scale", 65536.0)))
        if mode == "SCIENTIFIC" and _scaler_config(scaler) != O1_SCALER:
            raise ScienceRunError("O1 GradScaler configuration")
    scaler_record = _scaler_config(scaler)
    gen = torch.Generator().manual_seed(R.sha_int(f"gpatbench.downstream.augment.v1|resnet18|{method_id}|{seed}") % 2 ** 63)
    spoof_pool = R.Pool(real_spoof, method_id, seed, "real_spoof")
    syn_pool = None if real_only else R.Pool(synthetic, method_id, seed, "synthetic")
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
    epochs_log, k = [], 0
    best, best_ptr = None, None
    amp_observed = {}
    steps_fh = open(out_dir / "steps.jsonl", "x")
    retained = []
    stage = {"epoch": None, "step": None, "phase": "setup"}
    try:
        for epoch in range(epochs):
            stage.update(epoch=epoch, phase="train")
            order = R.permutation(method_id, seed, "live", epoch, len(live))
            e_loss, e_applied, e_skipped, e_gn = [], 0, 0, []
            for b in range(steps_per_epoch):
                stage["step"] = k
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
                if k == 0:
                    amp_observed["train_logits_dtype"] = str(logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
                if ov.get("nonfinite_loss_at_step") == k:
                    loss = loss * float("nan")
                if not torch.isfinite(loss):
                    raise ScienceRunError(f"non-finite loss at step {k}")
                if scaler is not None:
                    s_before = float(scaler.get_scale())
                    scaler.scale(loss).backward()
                    if ov.get("nonfinite_grad_at_step") == k:
                        next(iter(model.parameters())).grad.view(-1)[0] = float("inf")
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), R.FROZEN["grad_clip_norm"])
                    gn_v = float(gn.item())
                    post = _post_clip_norm(model.parameters()) if math.isfinite(gn_v) else None
                    scaler.step(opt)
                    scaler.update()
                    s_after = float(scaler.get_scale())
                    applied = s_after >= s_before
                    if applied != math.isfinite(gn_v):
                        raise ScienceRunError(f"step {k}: GradScaler skip and gradient finiteness disagree")
                    if not applied and s_before <= 1.0:
                        raise ScienceRunError(f"non-finite gradient at grad scale {s_before} <= 1 (step {k})")
                else:
                    s_before = s_after = None
                    loss.backward()
                    if ov.get("nonfinite_grad_at_step") == k:
                        next(iter(model.parameters())).grad.view(-1)[0] = float("inf")
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), R.FROZEN["grad_clip_norm"])
                    gn_v = float(gn.item())
                    if not math.isfinite(gn_v):
                        raise ScienceRunError(f"non-finite gradient at step {k} (no GradScaler)")
                    post = _post_clip_norm(model.parameters())
                    opt.step()
                    applied = True
                loss_v = float(loss.item())
                e_loss.append(loss_v)
                e_applied += applied
                e_skipped += not applied
                if applied:
                    e_gn.append(gn_v)
                ids = [f["sample_id"] for f in lv + rs] + [s["sample_id"] for s in sy]
                steps_fh.write(json.dumps({
                    "step": k, "epoch": epoch, "lr_backbone_hex": lrs[0].hex(), "lr_head_hex": lrs[1].hex(),
                    "loss_hex": loss_v.hex(), "grad_norm_hex": gn_v.hex(),
                    "grad_norm_post_clip_hex": None if post is None else float(post).hex(),
                    "grad_scale_before": s_before, "grad_scale": s_after, "optimizer_step_applied": applied,
                    "n_live": len(lv), "n_real_spoof": len(rs), "n_synthetic": len(sy), "batch": len(ids),
                    "batch_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}, sort_keys=True) + "\n")
                k += 1
            steps_fh.flush()
            os.fsync(steps_fh.fileno())
            if e_applied == 0:
                raise ScienceRunError(f"epoch {epoch}: no optimizer update was applied")
            stage["phase"] = "val"
            state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
            ssha = R.state_sha256(state)
            scored = _predict(model, val, load_real, policy, pool, obs=amp_observed)
            videos = M.aggregate_videos(scored)
            yv, sv = [v["y_true"] for v in videos], [v["video_score"] for v in videos]
            sel = S10.select_threshold_o2(yv, sv)
            auc = M.auc(yv, sv)
            rows = R.prediction_rows(scored, sel["threshold"], method_id=method_id, seed=seed, checkpoint_sha256=ssha,
                                     split_manifest_sha256=split_manifest_sha256)
            logit_of = {r["sample_id"]: r["logit_hex"] for r in scored}
            for r_ in rows:
                r_["track_id"] = track_id
                r_["logit_hex"] = logit_of[r_["sample_id"]]
            psha = R.write_parquet(rows, out_dir / f"predictions_val_epoch{epoch:02d}.parquet")
            e = {"epoch": epoch, "state_sha256": ssha, "threshold_hex": float(sel["threshold"]).hex(),
                 "threshold": sel["threshold"], "threshold_tie_break": sel["tie_break"],
                 "min_acer_thresholds_hex": [float(t).hex() for t in sel["min_acer_thresholds"]],
                 "val_acer": str(sel["val_acer"]), "val_auc": str(auc), "val_videos": len(videos),
                 "val_frames": len(scored), "predictions_sha256": psha,
                 "logits_sha256": hashlib.sha256("\n".join(r["logit_hex"] for r in scored).encode()).hexdigest(),
                 "train_steps": len(e_loss), "train_loss_mean": math.fsum(e_loss) / len(e_loss),
                 "train_loss_last": e_loss[-1], "optimizer_steps_applied": e_applied, "optimizer_steps_skipped": e_skipped,
                 "grad_norm_pre_clip_mean": math.fsum(e_gn) / len(e_gn), "grad_norm_pre_clip_max": max(e_gn),
                 "grad_scale_end": None if scaler is None else float(scaler.get_scale()),
                 "lr_backbone_last_hex": lrs[0].hex(), "lr_head_last_hex": lrs[1].hex(),
                 "wall_seconds": time.monotonic() - t0}
            became_best = best is None or _rank(e) < _rank(best)
            e["became_best"] = became_best
            stage["phase"] = "checkpoint"
            if retention == "ALL_EPOCHS" or became_best:
                ptr = _persist_checkpoint(state, ssha, epoch, out_dir)
                retained.append(ptr["checkpoint_file"])
                e["checkpoint_event"] = {"written": ptr["checkpoint_file"], "file_sha256": ptr["checkpoint_file_sha256"],
                                         "reload_verified": True}
            if became_best:
                old = best_ptr
                best, best_ptr = e, ptr
                _atomic_json(out_dir / "best_so_far.json", dict(best_ptr, val_acer=e["val_acer"], val_auc=e["val_auc"],
                                                               threshold_hex=e["threshold_hex"]))
                if retention == "SELECTED_ONLY" and old is not None:
                    os.unlink(out_dir / old["checkpoint_file"])      # only after the new best is persisted + verified
                    _fsync_dir(out_dir)
                    retained.remove(old["checkpoint_file"])
                    e["checkpoint_event"]["removed_previous_best"] = old["checkpoint_file"]
            epochs_log.append(e)
            _atomic_json(out_dir / "epochs.json", epochs_log)
        stage["phase"] = "final"
        steps_fh.close()
        final = S10.select_epoch(epochs_log)
        if final["epoch"] != best["epoch"] or best_ptr["epoch"] != best["epoch"]:
            raise ScienceRunError("running best-so-far differs from the D5 selection over all epochs")
        ck = out_dir / best_ptr["checkpoint_file"]
        ck_sha = _file_sha256(ck)
        if ck_sha != best_ptr["checkpoint_file_sha256"] or \
                R.state_sha256(torch.load(ck, map_location="cpu", weights_only=True)) != best["state_sha256"]:
            raise ScienceRunError("selected checkpoint changed or does not reload to its state sha256")
        if retention == "SELECTED_ONLY" and sorted(p.name for p in out_dir.glob("checkpoint_epoch*.pt")) != [ck.name]:
            raise ScienceRunError("SELECTED_ONLY: exactly the selected checkpoint must remain")
    except BaseException as exc:
        if pool is not None:
            pool.shutdown()
            pool = None
        if not steps_fh.closed:
            steps_fh.flush()
            os.fsync(steps_fh.fileno())
            steps_fh.close()
        import traceback
        _atomic_json(out_dir / "failure.json", {"label": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                                                "stage": stage, "completed_epochs": len(epochs_log), "steps_done": k,
                                                "retained_checkpoints": retained, "utc": _utc(),
                                                "traceback": traceback.format_exc()})
        raise
    finally:
        if pool is not None:
            pool.shutdown()
    applied_total = sum(e["optimizer_steps_applied"] for e in epochs_log)
    selection = {"label": "SCIENTIFIC" if mode == "SCIENTIFIC" else "NON_SCIENTIFIC_QUALIFICATION",
                 "method_id": method_id, "track_id": track_id, "detector_seed": seed, "bank_seed": bank_seed,
                 "selected_epoch": best["epoch"], "checkpoint_file": best_ptr["checkpoint_file"],
                 "checkpoint_file_sha256": ck_sha, "checkpoint_state_sha256": best["state_sha256"],
                 "val_acer": best["val_acer"], "val_auc": best["val_auc"], "frozen_val_threshold": best["threshold"],
                 "frozen_val_threshold_hex": best["threshold_hex"], "threshold_tie_break": best["threshold_tie_break"],
                 "selection_rule": S10.CHECKPOINT_RULE_TEXT, "threshold_rule": S10.THRESHOLD_RULE_TEXT,
                 "selection_reasoning": {
                     "epochs_evaluated": len(epochs_log),
                     "min_val_acer": best["val_acer"],
                     "epochs_at_min_acer": [e["epoch"] for e in epochs_log if Fraction(e["val_acer"]) == Fraction(best["val_acer"])],
                     "epochs_at_min_acer_and_max_auc": [e["epoch"] for e in epochs_log if _rank(e)[:2] == _rank(best)[:2]],
                     "best_so_far_trajectory": [e["epoch"] for e in epochs_log if e["became_best"]]},
                 "selection_source": "VAL only (split_v1 VAL frames, video score = mean frame probability)",
                 "metric_implementation": METRIC_IMPLEMENTATION, "test_used": False}
    record = {"label": selection["label"], "method_id": method_id, "track_id": track_id, "detector_seed": seed,
              "bank_seed": bank_seed, "epochs": epochs, "warmup_epochs": warmup_epochs,
              "steps_per_epoch": steps_per_epoch, "total_steps": k, "optimizer_steps_applied": applied_total,
              "optimizer_steps_skipped": k - applied_total, "batch": comp, "precision_policy": policy.record(),
              "grad_scaler": scaler_record, "amp_observed": amp_observed, "determinism_before": det_before,
              "determinism_after": determinism_record(), "optimizer": optimizer_record, "scheduler": scheduler_record,
              "retention": retention, "workers": workers, "qualification_overrides": ov or None,
              "pool_passes": {"real_spoof": spoof_pool.pass_index, "synthetic": None if real_only else syn_pool.pass_index},
              "pool_discarded": {"real_spoof": spoof_pool.discarded, "synthetic": None if real_only else syn_pool.discarded},
              "weights_sha256": R.WEIGHTS_SHA256, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "start_utc": start_utc, "end_utc": _utc(), "wall_seconds": time.monotonic() - t0}
    if record["determinism_after"] != det_before:
        raise ScienceRunError("determinism settings changed during the run")
    _atomic_json(out_dir / "selection.json", selection)
    _atomic_json(out_dir / "detector.json", record)
    return {"record": record, "selection": selection, "epochs": epochs_log}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
