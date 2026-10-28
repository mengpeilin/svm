"""Multi-view value network V(s[, track]): head camera in self-attention, the
two wrist cameras read through cross-attention, every token grounded with 3D
RoPE in world metres. One architecture, two inputs:

  state        the teacher: CLS, per-arm state L/R, head DinoV2 x391       (394 tokens)
  track_fused  the student: the same, with each head patch's DinoV2 feature
               concatenated with the embedding of its 16-step 3D track and
               projected back to 1024                                      (394 tokens)

Every layer: pre-norm self-attention over these tokens, cross-attention to the
2 x 391 wrist patches (a fixed memory: Q = head-side tokens, K/V = wrist
patches), FFN. q and k are rotated by their OWN 3D coordinates (head patches
and wrist patches by their unprojected world xyz, arm state tokens by the EEF
xyz), so a q.k score depends on the 3D offset between the two tokens; CLS and
depth-invalid patches pass unrotated. Additive 2D patch-grid encodings (+ a
per-wrist view embedding) go into Q and K only. Output: one logit (CLS ->
3-layer MLP).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INPUTS = ("state", "track_fused")
D_MODEL = 1024        # = DinoV2 dim: patch features enter unprojected
N_HEADS = 8
D_FF = 3200
TRACK_TEMPORAL_DIM = 512
HEAD_HIDDEN = 256
N_WRIST = 2


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE3D(nn.Module):
    """Rotary embedding over (x, y, z) world metres: head_dim split into even
    groups per axis (42/42/44 for 128), coordinates scaled by `spatial_scale`
    (0.01: one centimetre is ~1 rad on the top frequency)."""

    def __init__(self, head_dim: int, spatial_base: float = 100.0, spatial_scale: float = 0.01):
        super().__init__()
        per_axis = head_dim // 3
        per_axis -= per_axis % 2
        dims = [per_axis, per_axis, head_dim - 2 * per_axis]
        self.spatial_scale = spatial_scale
        for name, d in zip("xyz", dims):
            self.register_buffer(f"inv_freq_{name}", 1.0 / (spatial_base ** (torch.arange(0, d, 2).float() / d)),
                                 persistent=False)

    def build_cos_sin(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        """[B, N, 3] metres -> cos, sin [B, N, head_dim]."""
        normed = coords / self.spatial_scale
        cos, sin = [], []
        for i, name in enumerate("xyz"):
            phase = (normed[..., i:i + 1] * getattr(self, f"inv_freq_{name}")).repeat_interleave(2, dim=-1)
            cos.append(phase.cos())
            sin.append(phase.sin())
        return torch.cat(cos, -1), torch.cat(sin, -1)

    def apply(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """Rotate q or k [B, heads, N, head_dim] where `mask` [B, N]."""
        rotated = x * cos.unsqueeze(1).to(x.dtype) + _rotate_half(x) * sin.unsqueeze(1).to(x.dtype)
        return torch.where(mask[:, None, :, None], rotated, x)


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.nhead, self.head_dim, self.dropout = nhead, d_model // nhead, dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, rope: RoPE3D, q_rope, k_rope) -> Tensor:
        """`q_rope` / `k_rope`: ((cos, sin), mask)."""
        q = rope.apply(self._split(self.q_proj(query)), *q_rope[0], q_rope[1])
        k = rope.apply(self._split(self.k_proj(key)), *k_rope[0], k_rope[1])
        out = F.scaled_dot_product_attention(q, k, self._split(self.v_proj(value)),
                                             dropout_p=self.dropout if self.training else 0.0)
        B, _, N, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


class MVLayer(nn.Module):
    """Pre-norm self-attn -> wrist cross-attn -> FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.dropout, self.dropout1 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, pos, mem, mem_pos, rope, x_rope, mem_rope) -> Tensor:
        h = self.norm1(x)
        qk = h + pos
        x = x + self.dropout1(self.self_attn(qk, qk, h, rope, x_rope, x_rope))
        h, m = self.norm2(x), self.norm_mem(mem)
        x = x + self.dropout2(self.cross_attn(h + pos, m + mem_pos, m, rope, x_rope, mem_rope))
        h = self.norm3(x)
        return x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(h)))))


class TrackTemporalEncoder(nn.Module):
    """[B, N, 16, 3] displacement trajectory -> one token per point (flatten + 2-layer MLP)."""

    def __init__(self, horizon: int, temporal_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(horizon * 3, temporal_dim), nn.GELU(), nn.Linear(temporal_dim, out_dim))

    def forward(self, tracks: Tensor) -> Tensor:
        B, N, T, _ = tracks.shape
        return self.mlp(tracks.reshape(B, N, T * 3))


def sinusoidal_2d(height: int, width: int, dim: int, device) -> Tensor:
    """2D sin-cos grid encoding [height * width, dim]."""
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000.0 ** (torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4 - 1)))
    y, x = y.flatten()[:, None] * omega[None], x.flatten()[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class MultiViewDiscriminator(nn.Module):
    """Batch keys (`data.Frames.inputs`): dino [B, 3, 391, 1024] (head, left,
    right wrist), centres [B, 3, 391, 3] (raw m), centre_ok [B, 3, 391], state
    [B, 16] (normalized), state_xyz [B, 2, 3] (raw m); for track_fused also
    track [B, 391, 16, 3] (normalized) and valid [B, 391]."""

    def __init__(self, inputs: str, n_layers: int = 6, dropout: float = 0.1, rope_spatial_scale: float = 0.01,
                 arm_dim: int = 8, patch_grid: tuple[int, int] = (17, 23), horizon: int = 16):
        super().__init__()
        if inputs not in INPUTS:
            raise ValueError(f"inputs must be one of {INPUTS}, got {inputs!r}")
        self.inputs, self.patch_grid, self.arm_dim = inputs, patch_grid, arm_dim
        d = D_MODEL
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cls_pos = nn.Embedding(1, d)
        self.state_encoder = nn.Linear(arm_dim, d)
        self.state_pos = nn.Embedding(2, d)
        self.arm_embed = nn.Embedding(2, d)          # left = 0, right = 1
        self.wrist_view_embed = nn.Embedding(N_WRIST, d)
        if inputs == "track_fused":
            self.track_temporal_encoder = TrackTemporalEncoder(horizon, TRACK_TEMPORAL_DIM, d)
            self.invalid_embed = nn.Parameter(torch.zeros(d))
            self.track_fusion_proj = nn.Linear(2 * d, d)
        self.layers = nn.ModuleList([MVLayer(d, N_HEADS, D_FF, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, HEAD_HIDDEN), nn.ReLU(inplace=True),
                                  nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.ReLU(inplace=True),
                                  nn.Linear(HEAD_HIDDEN, 1))
        self.rope = RoPE3D(d // N_HEADS, spatial_scale=rope_spatial_scale)
        self._grid: Optional[Tensor] = None
        for p in [self.cls_token] + ([self.invalid_embed] if inputs == "track_fused" else []):
            nn.init.trunc_normal_(p, std=0.02)
        for emb in (self.cls_pos, self.state_pos, self.arm_embed, self.wrist_view_embed):
            nn.init.trunc_normal_(emb.weight, std=0.02)

    def _grid_pos(self, device, dtype) -> Tensor:
        if self._grid is None or self._grid.device != device:
            self._grid = sinusoidal_2d(*self.patch_grid, D_MODEL, device)
        return self._grid.to(dtype)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        dino, centres, centre_ok = batch["dino"], batch["centres"], batch["centre_ok"]
        B, device, dtype = dino.shape[0], dino.device, dino.dtype
        grid = self._grid_pos(device, dtype)
        head = dino[:, 0]
        if self.inputs == "track_fused":
            valid = batch["valid"]
            track_tok = self.track_temporal_encoder(batch["track"] * valid[..., None, None])
            head = self.track_fusion_proj(torch.cat([head, track_tok], -1)) + (~valid)[..., None] * self.invalid_embed
        state_tok = self.state_encoder(batch["state"].reshape(B, 2, self.arm_dim)) + self.arm_embed.weight[None]

        x = torch.cat([self.cls_token.expand(B, -1, -1).to(dtype), state_tok, head], 1)
        pos = torch.cat([self.cls_pos.weight, self.state_pos.weight, grid], 0)[None]
        coords = torch.cat([x.new_zeros(B, 1, 3, dtype=torch.float32), batch["state_xyz"].float(),
                            centres[:, 0].float()], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device),
                          torch.ones(B, 2, dtype=torch.bool, device=device), centre_ok[:, 0]], 1)
        x_rope = (self.rope.build_cos_sin(coords), mask)

        mem = dino[:, 1:].reshape(B, N_WRIST * grid.shape[0], -1)
        mem_pos = (grid[None] + self.wrist_view_embed.weight.to(dtype)[:, None]).reshape(1, -1, D_MODEL)
        mem_rope = (self.rope.build_cos_sin(centres[:, 1:].reshape(B, -1, 3).float()), centre_ok[:, 1:].reshape(B, -1))
        for layer in self.layers:
            x = layer(x, pos, mem, mem_pos, self.rope, x_rope, mem_rope)
        return self.head(self.final_norm(x[:, 0])).squeeze(-1)


def build(config: dict) -> MultiViewDiscriminator:
    """`config`: {"arch": "dino", **MultiViewDiscriminator kwargs}."""
    assert config["arch"] == "dino", config
    return MultiViewDiscriminator(**{k: v for k, v in config.items() if k != "arch"})


def load(path, device) -> tuple[nn.Module, dict]:
    """A checkpoint written by train_teacher / train_student: (model in eval
    mode, checkpoint dict); config["arch"] picks the DinoV2 model or the
    Qwen-3D student (`qwen_model.py`)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["config"]["arch"] == "qwen3d":
        from .qwen_model import from_checkpoint
        return from_checkpoint(ck, device).eval(), ck
    model = build(ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict(model: nn.Module, batches) -> np.ndarray:
    """Logits over an iterable of model batches (bf16 autocast, as in training)."""
    was = model.training
    model.eval()
    out = []
    for b in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(b).float().cpu().numpy())
    model.train(was)
    return np.concatenate(out)
