"""Frozen auxiliary model adapters: FaceXFormer (geometry) and AdaFace IR-50 (identity).

Model code is imported read-only from a code directory whose files must match the official
repository file hashes pinned here (no vendoring into this repo). Weights are hash-checked.
All inference: model.eval(), torch.inference_mode(), no gradients, no augmentation.

FaceXFormer (VERIFIED weights): Kartik-3004/facexformer @ 10fe8291 (code), HF kartiknarayan/facexformer
  @ fd12148d ckpts/model.pt. Official input transform (inference.py): PIL RGB -> Resize(224x224, BICUBIC)
  -> ToTensor -> Normalize(ImageNet mean/std). One forward per task (0 parsing, 1 landmarks, 2 headpose).
  The decoder computes every head each call; `tasks` only filters rows (verified in facexformer.py).
  Constructor calls swin_b(weights='IMAGENET1K_V1'); we build it with weights=None (no download) and then
  load the checkpoint with strict=True, so every parameter/buffer comes from the checkpoint.

AdaFace (owner-resolved Q-03/Q-18/Q-19): weights = the ORIGINAL mk-minchul/AdaFace release
  "R50 / WebFace4M" (official README table, repo @ c60eaa78), file adaface_ir50_webface4m.ckpt,
  sha256 52cca7c6...; loaded as the official inference.py does (torch.load(...)["state_dict"],
  keys starting with "model." with that prefix stripped, strict=True) into the official net.py
  build_model("ir_50"), whose forward returns the L2-normalised feature and its norm.
  The CVLFace export (sha256 43bd2d57...) is NOT selected: it is the same trained model with the
  input convolution's channel axis reversed for RGB input, kept in models/registry.yaml as
  historical candidate evidence only, and refused by this adapter.
  Frozen input contract (no parameters, no alternatives):
    canonical 256x256 RGB uint8 -> cv2.resize 112x112 INTER_AREA -> RGB->BGR reversal ->
    ((x/255) - 0.5)/0.5 exactly as official inference.py to_input -> float32 [1,3,112,112].
  No MTCNN, no landmark alignment and no second face detector run inside this adapter: the
  benchmark's frozen canonical face is the common geometry frame (DEV-014).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from .logit_store import mask_from_logits

FACEXFORMER_CODE_SHA256 = {   # official Kartik-3004/facexformer @ 10fe8291f8a64e2ca1daf938e3e0007bd860303b
    "network/__init__.py": "31b03a6fb9c51b9c56d6482604b6e4a97ee4638719f7c3e0c32764e90a6f6c29",
    "network/models/__init__.py": "d107e7865cd12a2f2d55a2c36a499ec481230066466a478d68ec1088d70aa3fc",
    "network/models/facexformer.py": "007076ec43c718858d75c755f940776c9f17bac07173b65316b38388db9f8d90",
    "network/models/transformer.py": "df8ac36e5bd9d367837efc3012b9c2e675b75ab6983a2ea25e2a85dba6d0b237",
}
ADAFACE_CODE_SHA256 = {       # official mk-minchul/AdaFace @ c60eaa786a42c03444f3df7096dbaf9d57ae010d net.py
    "adaface_net.py": "b4db4eb0174a385fd29e5f616391b50d443f455990c8b88dcab1f8021af8ba4c",
}
FACEXFORMER_WEIGHT_SHA256 = "327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d"
# Owner selection (Q-03): the ORIGINAL AdaFace repository release R50 / WebFace4M.
ADAFACE_WEIGHT_SHA256 = "52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8"
ADAFACE_SELECTED = {"variant": "IR-50 (R50)", "training_dataset": "WebFace4M",
                    "source": "https://github.com/mk-minchul/AdaFace README -> adaface_ir50_webface4m.ckpt",
                    "code_commit": "c60eaa786a42c03444f3df7096dbaf9d57ae010d"}
# Investigated but NOT SELECTED for the final M2 identity cache (kept so it cannot silently return).
ADAFACE_NOT_SELECTED_SHA256 = {
    "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496":
        "CVLFace export hf:minchul/cvlface_adaface_ir50_webface4m (RGB input); NOT_SELECTED_FOR_FINAL_M2",
}
ADAFACE_COLOR_ORDER = "BGR"                                  # Q-18 RESOLVED_BY_OWNER
ADAFACE_GEOMETRIC = "RESIZE_256_TO_112_INTER_AREA"           # Q-19 RESOLVED_BY_OWNER
ADAFACE_INPUT = 112
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FX_INPUT = 224
FX_TASKS = {"parsing": 0, "landmarks": 1, "headpose": 2}


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_code(code_dir, expected: dict) -> None:
    for rel, h in expected.items():
        got = _sha(Path(code_dir) / rel)
        if got != h:
            raise ValueError(f"code fingerprint mismatch for {rel}: {got} != {h}")


def set_deterministic(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class FaceXFormerAdapter:
    def __init__(self, code_dir, weight_path, device="cpu"):
        import torch
        import torchvision.models as tvm
        verify_code(code_dir, FACEXFORMER_CODE_SHA256)
        if _sha(weight_path) != FACEXFORMER_WEIGHT_SHA256:
            raise ValueError("FaceXFormer weight hash mismatch")
        sys.path.insert(0, str(code_dir))
        import network.models.facexformer as fxm
        fxm.swin_b = lambda weights=None, **k: tvm.swin_b(weights=None, **k)   # no ImageNet download
        self.model = fxm.FaceXFormer()
        ck = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict_backbone"], strict=True)
        self.model.eval().to(device)
        self.device = device

    def preprocess(self, rgb_uint8: np.ndarray):
        """Official transform on the canonical RGB face (canonical image is not modified)."""
        import torch
        from PIL import Image
        from torchvision import transforms as T
        t = T.Compose([T.Resize(size=(FX_INPUT, FX_INPUT), interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        assert rgb_uint8.dtype == np.uint8 and rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
        return t(Image.fromarray(rgb_uint8)).unsqueeze(0).to(self.device, torch.float32)

    def __call__(self, rgb_uint8: np.ndarray) -> dict:
        import torch
        x = self.preprocess(rgb_uint8)
        out = {}
        with torch.inference_mode():
            for name, t in FX_TASKS.items():
                lm, hp, _a, _v, _ag, _g, _r, seg = self.model(x, None, torch.tensor([t], device=self.device))
                if name == "parsing":
                    out["parsing_logits"] = seg[0].float().cpu().numpy()                 # (11, 224, 224)
                elif name == "landmarks":
                    out["landmarks_norm"] = lm.view(-1, 68, 2)[0].float().cpu().numpy()  # [-1, 1], 224 input
                else:
                    out["pose_pitch_yaw_roll_rad"] = hp[0].float().cpu().numpy()         # (3,) radians
        out["parsing_mask"] = mask_from_logits(out["parsing_logits"])   # official argmax OF the stored logits (Q-23)
        # official denorm_points (align_corners=False) to the 224 input grid, then linear map to canonical 256
        lm224 = ((out["landmarks_norm"] + 1.0) * FX_INPUT - 1.0) / 2.0
        out["landmarks_px224"] = lm224.astype(np.float32)
        out["landmarks_px256"] = ((lm224 + 0.5) * (256.0 / FX_INPUT) - 0.5).astype(np.float32)
        return out


class AdaFaceAdapter:
    """Frozen identity adapter. The colour order and the geometric adapter are OWNER-FROZEN
    module constants, not arguments: there is no code path to RGB input, to another checkpoint,
    to MTCNN/landmark alignment or to a second face detection."""

    def __init__(self, code_dir, weight_path, device="cpu"):
        import torch
        verify_code(code_dir, ADAFACE_CODE_SHA256)
        got = _sha(weight_path)
        if got in ADAFACE_NOT_SELECTED_SHA256:
            raise ValueError(f"refusing a checkpoint that the owner did not select: {ADAFACE_NOT_SELECTED_SHA256[got]}")
        if got != ADAFACE_WEIGHT_SHA256:
            raise ValueError(f"AdaFace weight hash mismatch: {got} != {ADAFACE_WEIGHT_SHA256}")
        sys.path.insert(0, str(code_dir))
        import adaface_net
        self.model = adaface_net.build_model("ir_50")
        ck = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = {k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}   # official inference.py
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.color_order, self.geometric, self.device = ADAFACE_COLOR_ORDER, ADAFACE_GEOMETRIC, device
        self.weight_sha256 = got

    def preprocess(self, rgb_uint8: np.ndarray):
        """Canonical 256 RGB uint8 -> official AdaFace input tensor.

        Numerically identical to the official inference.py `to_input` applied to the 112x112
        image: the scaling is done in float64 (as in the official code) and the tensor is float32.
        """
        import cv2
        import torch
        if rgb_uint8.shape != (256, 256, 3) or rgb_uint8.dtype != np.uint8:
            raise ValueError(f"AdaFace expects the canonical 256x256x3 uint8 face, got {rgb_uint8.shape} {rgb_uint8.dtype}")
        img = cv2.resize(rgb_uint8, (ADAFACE_INPUT, ADAFACE_INPUT), interpolation=cv2.INTER_AREA)   # Q-19
        bgr = np.asarray(img)[:, :, ::-1]                                     # Q-18: RGB -> BGR (official)
        x = ((bgr / 255.0) - 0.5) / 0.5                                       # official to_input arithmetic
        return torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1))[None]).float().to(self.device)

    def __call__(self, rgb_uint8: np.ndarray) -> dict:
        import torch
        with torch.inference_mode():
            feat, norm = self.model(self.preprocess(rgb_uint8))
        f = feat[0].float().cpu().numpy().astype(np.float32)
        f = (f / np.linalg.norm(f)).astype(np.float32)                                 # explicit L2 (idempotent)
        return {"embedding": f, "raw_norm": float(norm[0, 0])}
