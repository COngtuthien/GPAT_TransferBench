"""Numeric proof that the DiffFAS content image is inert under use_pair=false (Amendment A1 §20).

    <m2 venv>/bin/python tools/m4_difffas_content_inertness.py

Imports the OFFICIAL `diffusion.py` at the pinned commit and runs `GaussianDiffusion.training_losses`
twice with everything held fixed except the `content` tensor. Under `use_pair=False` the model input
and every loss term must be bit-identical; under `use_pair=True` they must differ, which proves the
test can actually detect a dependency. Results are frozen in
outputs/audit/M4_DIFFFAS_CONTENT_INERTNESS.json.

Device shim: the official line `img[~cond_mask] = torch.zeros(1,3,256,256).cuda()` hard-codes CUDA.
This tool neutralises device placement only (`Tensor.cuda -> self`). No scientific line is altered,
and no model weights are loaded: the diffusion model is replaced by a deterministic stub whose only
job is to record the tensor it is handed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E    # noqa: E402
from gpatbench.pairs import track_a as T     # noqa: E402

AUDIT = ROOT / "outputs/audit"
SRC = ROOT / "third_party/source_cache/difffas"
SIZE = 256                   # the official code hard-codes 256x256 in the CFG dropout line
BATCH = 2


def load_official():
    torch.Tensor.cuda = lambda self, *a, **k: self          # device shim only
    # `diffusion.py` imports tqdm only for progress bars; stub it so the module imports on a machine
    # without it. Nothing in the traced code path uses it.
    if "tqdm" not in sys.modules:
        import importlib.machinery
        import types
        stub = types.ModuleType("tqdm")
        stub.__spec__ = importlib.machinery.ModuleSpec("tqdm", None)
        stub.tqdm = lambda x=None, *a, **k: x
        sys.modules["tqdm"] = stub
    sys.path.insert(0, str(SRC))
    import importlib
    return importlib.import_module("diffusion")


def run_once(diffusion_mod, use_pair: bool, content: torch.Tensor, gt: torch.Tensor,
             style: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, betas: torch.Tensor):
    seen = {}

    def stub(x=None, encoder=None, t=None, cond_mask=None, x_cond=None, prob=None, **kw):
        seen["x"] = x.detach().clone()
        seen["x_cond"] = None if x_cond is None else x_cond.detach().clone()
        c = gt.shape[1]
        # A deterministic function of EVERY input channel, so a content channel appearing in `x`
        # must move the loss. That is what makes the negative control meaningful.
        pooled = x.mean(dim=1, keepdim=True).expand(-1, c, -1, -1)
        base = (x[:, :c] + pooled) * 0.5 + 0.25
        return torch.cat([base, torch.zeros_like(base)], dim=1)

    diff = diffusion_mod.create_gaussian_diffusion(betas.numpy(), predict_xstart=False)
    terms = diff.training_losses(stub, None, x_start=gt.clone(), t=t,
                                 betas=betas.clone(), cond_input=[content.clone(), style.clone()],
                                 prob=1, means_size=5, var_size=3, noise=noise.clone(),
                                 use_pair=use_pair)
    return {"model_input": seen["x"], "x_cond": seen["x_cond"],
            "loss": terms["loss"].detach().clone(), "mse": terms["mse"].detach().clone(),
            "vb": terms["vb"].detach().clone() if "vb" in terms else None}


def main() -> int:
    if not (SRC / "diffusion.py").is_file():
        print(json.dumps({"status": "SOURCE_CACHE_ABSENT", "path": str(SRC)}))
        return 1
    mod = load_official()

    torch.manual_seed(0)
    gt = torch.randn(BATCH, 3, SIZE, SIZE)
    style = torch.randn(BATCH, 3, SIZE, SIZE)
    noise = torch.randn(BATCH, 3, SIZE, SIZE)
    content_a = torch.randn(BATCH, 3, SIZE, SIZE)
    content_b = torch.randn(BATCH, 3, SIZE, SIZE)
    t = torch.tensor([7, 123])
    betas = torch.from_numpy(
        np.linspace(1e-4, 0.02, 1000, dtype=np.float64)).float()

    out = {"pinned_commit": T.official_source_commit("difffas"),
           "diffusion_py_sha256": E.sha256_file(SRC / "diffusion.py"),
           "batch": BATCH, "spatial_size": SIZE,
           "device_shim": "Tensor.cuda -> self (device placement only)",
           "import_shim": "tqdm stubbed (progress bars only; unused in the traced path)",
           "model": "deterministic stub; no official weights loaded",
           "cases": {}}

    for use_pair in (False, True):
        a = run_once(mod, use_pair, content_a, gt, style, noise, t, betas)
        b = run_once(mod, use_pair, content_b, gt, style, noise, t, betas)
        out["cases"][f"use_pair={use_pair}"] = {
            "model_input_channels": int(a["model_input"].shape[1]),
            "model_input_identical": bool(torch.equal(a["model_input"], b["model_input"])),
            "x_cond_identical": bool(torch.equal(a["x_cond"], b["x_cond"])),
            "x_cond_is_style": bool(torch.equal(a["x_cond"], style)),
            "loss_identical": bool(torch.equal(a["loss"], b["loss"])),
            "mse_identical": bool(torch.equal(a["mse"], b["mse"])),
            "vb_identical": (None if a["vb"] is None else bool(torch.equal(a["vb"], b["vb"]))),
            "max_abs_loss_diff": float((a["loss"] - b["loss"]).abs().max()),
        }

    off = out["cases"]["use_pair=False"]
    on = out["cases"]["use_pair=True"]
    checks = {
        "unpaired_model_input_is_x_t_only": off["model_input_channels"] == 3,
        "paired_model_input_concatenates_content": on["model_input_channels"] == 6,
        "unpaired_content_does_not_change_model_input": off["model_input_identical"],
        "unpaired_content_does_not_change_loss": off["loss_identical"] and off["mse_identical"]
                                                 and off["vb_identical"] is not False,
        "conditioning_is_style_not_content": off["x_cond_is_style"] and on["x_cond_is_style"],
        "test_can_detect_a_dependency": (not on["model_input_identical"])
                                        and (not on["loss_identical"]),
    }
    out["checks"] = checks
    out["content_training_role"] = ("INERT_API_PLACEHOLDER" if all(checks.values())
                                    else "DEPENDENCY_DETECTED")
    out["status"] = "PASS" if all(checks.values()) else "FAIL"
    (AUDIT / "M4_DIFFFAS_CONTENT_INERTNESS.json").write_bytes(E.canonical_json_bytes(out))
    print(json.dumps({"status": out["status"], "role": out["content_training_role"],
                      "checks": checks}, indent=1))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
