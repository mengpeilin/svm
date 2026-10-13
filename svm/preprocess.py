"""Raw observation -> model batch.

Conventions (all must match training, see README "Inputs"):
  images   240x320 RGB uint8, one per camera (head, left wrist, right wrist)
  depth    240x320 float, MILLIMETRES, 0 where nothing was rendered
  K        3x3 pinhole intrinsic
  E        3x4 extrinsic, world -> camera (OpenCV: x right, y down, z forward)
  state    16-D [L xyz, L quat(wxyz), L grip, R xyz, R quat(wxyz), R grip], world frame
  track    [391, 16, 3] world-frame displacement of each head patch's 3D point
           from t to t+h+1 (h = 0..15), metres; `track_valid` [391] marks the
           patches whose point is tracked                       (track model)
  action   [16, 16] commanded EEF poses of the chunk, each in the `state`
           format                                               (action model)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PATCH_GRID = (17, 23)
N_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]  # 391
IMG_HW = (240, 320)
DINO_HW = (PATCH_GRID[0] * 14, PATCH_GRID[1] * 14)  # 238 x 322
ARM_DIM = 8


def patch_pixel_centres() -> np.ndarray:
    """[391, 2] native-pixel (x, y) centres of the 17x23 DINOv2 patches: patch
    centres on the 238x322 resized grid, scaled back to 240x320."""
    ph, pw = PATCH_GRID
    gx, gy = np.meshgrid((np.arange(pw) + 0.5) * 14, (np.arange(ph) + 0.5) * 14)
    scale = np.array([IMG_HW[1] / DINO_HW[1], IMG_HW[0] / DINO_HW[0]])
    return np.stack([gx.ravel(), gy.ravel()], -1) * scale


PATCH_PX = patch_pixel_centres().astype(np.float32)


def patch_centres(depth_mm: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World xyz [391, 3] of each patch centre (bilinear depth at the centre
    pixel, unprojected with K, then camera -> world with E) and its validity [391]."""
    x, y = PATCH_PX[:, 0], PATCH_PX[:, 1]
    d = cv2.remap(depth_mm.astype(np.float32), x[None], y[None], cv2.INTER_LINEAR)[0] / 1000.0
    ok = (d > 0) & np.isfinite(d)
    cam = np.stack([(x - K[0, 2]) * d / K[0, 0], (y - K[1, 2]) * d / K[1, 1], d], -1)
    world = (cam - E[:, 3]) @ E[:, :3]
    world[~ok] = 0.0
    return world.astype(np.float32), ok


def load_dino(device) -> torch.nn.Module:
    """DINOv2 ViT-L/14 with registers (the GAP policy's image encoder)."""
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device).eval()


@torch.no_grad()
def dino_features(model: torch.nn.Module, rgb: np.ndarray, device) -> torch.Tensor:
    """[B, 240, 320, 3] uint8 -> [B, 391, 1024] float32 patch tokens: bilinear
    resize to 238x322, ImageNet normalization, `x_norm_patchtokens`. Run in fp32
    (no TF32) to reproduce the training features exactly."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(0, 3, 1, 2).float() / 255.0
    img = F.interpolate(img, size=DINO_HW, mode="bilinear", align_corners=False)
    return model.forward_features((img - mean) / std)["x_norm_patchtokens"].float()


class Normalizer:
    """State / action: per-dim min/max -> [-1, 1]. Tracks: one isotropic scale
    x / rms. DINO features and patch centres are used raw (centres feed RoPE in
    metres). Each checkpoint stores the statistics of the inputs it reads."""

    def __init__(self, state_lo, state_hi, track_rms=None, action_lo=None, action_hi=None):
        self.state_lo = np.asarray(state_lo, dtype=np.float32)
        self.state_hi = np.asarray(state_hi, dtype=np.float32)
        self.track_rms = track_rms
        self.action_lo = np.asarray(action_lo, dtype=np.float32) if action_lo is not None else None
        self.action_hi = np.asarray(action_hi, dtype=np.float32) if action_hi is not None else None

    def state_norm(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.state_lo) / (self.state_hi - self.state_lo) - 1.0

    def action_norm(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.action_lo) / (self.action_hi - self.action_lo) - 1.0

    def track_norm(self, x: np.ndarray) -> np.ndarray:
        return x / self.track_rms


def build_batch(obs: dict[str, np.ndarray], dino: torch.Tensor, norm: Normalizer, device,
                inputs: str) -> dict[str, torch.Tensor]:
    """Model batch for B decision steps.

    `obs` holds per-step arrays with a leading batch dim B: `{cam}_depth`,
    `{cam}_K`, `{cam}_E` for cam in (head, left_wrist, right_wrist), `state`
    [B, 16], and for `inputs="track"` `track` [B, 391, 16, 3] + `track_valid`
    [B, 391], for `inputs="action"` `action` [B, 16, 16].
    `dino` is [B, 3, 391, 1024] (head, left wrist, right wrist)."""
    B = len(obs["state"])
    centres = np.zeros((B, 3, N_PATCHES, 3), np.float32)
    centre_ok = np.zeros((B, 3, N_PATCHES), bool)
    for v, cam in enumerate(("head", "left_wrist", "right_wrist")):
        for i in range(B):
            centres[i, v], centre_ok[i, v] = patch_centres(obs[f"{cam}_depth"][i], obs[f"{cam}_K"][i], obs[f"{cam}_E"][i])
    state = np.asarray(obs["state"], np.float32)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)  # noqa: E731
    batch = {
        "dino": dino.to(device=device, dtype=torch.float32),
        "centres": t(centres),
        "centre_ok": t(centre_ok, torch.bool),
        "state": t(norm.state_norm(state)),
        "state_xyz": t(arm_xyz(state)),
    }
    if inputs == "track":
        batch["track"] = t(norm.track_norm(np.asarray(obs["track"], np.float32)))
        batch["valid"] = t(obs["track_valid"], torch.bool)
    else:
        action = np.asarray(obs["action"], np.float32)
        batch["action"] = t(norm.action_norm(action))
        batch["action_xyz"] = t(arm_xyz(action[:, 0]))
    return batch


def arm_xyz(vec16: np.ndarray) -> np.ndarray:
    """[B, 16] -> [B, 2, 3] raw world-frame L/R EEF xyz (RoPE coordinates)."""
    return np.stack([vec16[:, 0:3], vec16[:, ARM_DIM : ARM_DIM + 3]], axis=1)
