"""ArtifactProbeNet model (owner decision D-M5-04: full fine-tune, 7-logit head)."""
from __future__ import annotations

from .contract import EMBED_DIM, K, L2_EPS, ProbeContractViolation


def build(seed: int = 42, *, pretrained: bool = True):
    """ResNet-18 IMAGENET1K_V1 with `fc` replaced by Linear(512, 7). Every parameter is trainable.

    The seed is set BEFORE the classifier is constructed so its initialisation is reproducible; the
    classifier is newly initialised while the backbone carries the pretrained weights.
    """
    import random

    import numpy as np
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if model.fc.in_features != EMBED_DIM:
        raise ProbeContractViolation(f"expected {EMBED_DIM}-D penultimate, got {model.fc.in_features}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.fc = torch.nn.Linear(EMBED_DIM, K)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def trainable_report(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    return {"total_parameters": int(total), "trainable_parameters": int(trainable),
            "frozen_parameter_names": frozen, "fully_trainable": total == trainable,
            "out_features": int(model.fc.out_features),
            "in_features": int(model.fc.in_features)}


def embed(model, x):
    """Penultimate 512-D feature after global average pooling, L2-normalised (eps 1e-12).

    Never the classifier logits, and no projection head.
    """
    import torch
    import torch.nn.functional as F
    modules = list(model.children())[:-1]          # everything up to, but not including, fc
    backbone = torch.nn.Sequential(*modules)
    feat = torch.flatten(backbone(x), 1)
    if feat.shape[1] != EMBED_DIM:
        raise ProbeContractViolation(f"embedding dim {feat.shape[1]} != {EMBED_DIM}")
    return F.normalize(feat, p=2, dim=1, eps=L2_EPS)
