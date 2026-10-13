"""3D Rotary Position Embedding (RoPE) over (x, y, z) world-frame coordinates.

Adapted from GAP's `gap_policy/model/common/rope4d.py` with the time axis
removed. The discriminator scores ONE frame at a time: there is no action chunk
to index and no diffusion timestep, so a 4D RoPE would spend a quarter of every
head on `t = 0` -- an identity rotation. Splitting the same head_dim three ways
puts all of it to work.

Tokens without a natural coordinate (CLS, depth-invalid patches) are excluded via
`rope_mask`; their q/k pass through unrotated. We use an explicit mask rather
than zero coordinates because (0, 0, 0) is a real location in the robot
workspace.

Coordinate sources (world frame, raw metres, from the tracks dataset):
  - patch tokens:      image_patch_centre_3d_position
  - per-arm state:     that arm's current EEF xyz  (robot_state[0:3], [8:11])
  - per-arm action:    that arm's commanded EEF xyz (action[0:3], [8:11])
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE pairing: (x1, x2) -> (-x2, x1) on interleaved pairs."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE3D(nn.Module):
    """Per-axis inverse frequencies; builds cos/sin tables from token
    coordinates and applies the rotation to q/k shaped [B, n_heads, N, head_dim].

    head_dim is split across (x, y, z) into even groups, remainder to z. With
    head_dim=128 the split is 42/42/44.
    """

    def __init__(
        self,
        head_dim: int,
        spatial_base: float = 100.0,
        spatial_scale: float = 0.01,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim

        per_axis = head_dim // 3
        per_axis -= per_axis % 2
        dims = [per_axis, per_axis, head_dim - 2 * per_axis]
        assert all(d > 0 and d % 2 == 0 for d in dims), f"Bad axis split {dims}"
        self.axis_dims = dims

        # spatial_scale normalizes raw metres to ~O(1) before the rotary phase.
        # GAP uses 0.5 (workspace spans ~±0.5 m), which leaves patch centres a
        # few centimetres apart nearly phase-identical. Task-relevant motion
        # here is ~1 cm (mean |track displacement| = 0.0104 m), so 0.01 puts one
        # centimetre at ~1 rad on the top frequency.
        self.spatial_scale = spatial_scale

        for name, d in zip(("x", "y", "z"), dims):
            inv_freq = 1.0 / (spatial_base ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer(f"inv_freq_{name}", inv_freq, persistent=False)

    def build_cos_sin(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, N, 3] world-frame (x, y, z) in metres

        Returns:
            cos, sin: [B, N, head_dim]
        """
        assert coords.shape[-1] == 3, f"Expected 3D coords, got {tuple(coords.shape)}"
        normed = coords / self.spatial_scale  # [B, N, 3]

        parts_cos, parts_sin = [], []
        for i, name in enumerate(("x", "y", "z")):
            inv_freq = getattr(self, f"inv_freq_{name}")  # [d_i/2]
            phase = normed[..., i : i + 1] * inv_freq     # [B, N, d_i/2]
            phase = phase.repeat_interleave(2, dim=-1)    # [B, N, d_i]
            parts_cos.append(phase.cos())
            parts_sin.append(phase.sin())
        return torch.cat(parts_cos, dim=-1), torch.cat(parts_sin, dim=-1)

    def apply(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rotate q or k.

        Args:
            x: [B, n_heads, N, head_dim]
            cos, sin: [B, N, head_dim] from build_cos_sin
            rope_mask: [B, N] bool, True where rotation applies. Tokens with
                False pass through unrotated.
        """
        cos = cos.unsqueeze(1).to(x.dtype)  # [B, 1, N, head_dim]
        sin = sin.unsqueeze(1).to(x.dtype)
        rotated = x * cos + _rotate_half(x) * sin
        if rope_mask is None:
            return rotated
        keep = rope_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        return torch.where(keep, rotated, x)
