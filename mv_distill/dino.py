"""DinoV2 (vitl14_reg) patch features of camera images, with the exact
preprocessing GAP's deploy_policy applies to the head image (the collector
stores the head features; the wrist ones are extracted here).

Weights come from the local torch hub: ~/.cache/torch/hub/facebookresearch_dinov2_main
and ~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .parts import DINO_HW


def load_dino(device) -> torch.nn.Module:
    # fp32 matmuls: features as the collector extracted them
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hub = Path.home() / ".cache/torch/hub"
    model = torch.hub.load(str(hub / "facebookresearch_dinov2_main"), "dinov2_vitl14_reg", source="local",
                           pretrained=True, weights=str(hub / "checkpoints/dinov2_vitl14_reg4_pretrain.pth"))
    return model.to(device).eval()


@torch.no_grad()
def dino_features(model: torch.nn.Module, rgb: np.ndarray, device) -> torch.Tensor:
    """[B, 240, 320, 3] uint8 RGB -> [B, 391, 1024] float32: bilinear resize to
    238x322, ImageNet normalization, x_norm_patchtokens."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(0, 3, 1, 2).float() / 255.0
    img = F.interpolate(img, size=DINO_HW, mode="bilinear", align_corners=False)
    return model.forward_features((img - mean) / std)["x_norm_patchtokens"].float()
