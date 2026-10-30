"""Multi-view SVM discriminators f(s, a). Two variants differ only in how "a"
enters the head-side token sequence:

  MultiViewTrackSVM   a = 3D point tracks of the head patches
      self-attention   [CLS], state L/R, head dino x391, track x391     = 785
  MultiViewActionSVM  a = 16-step commanded EEF pose chunk
      self-attention   [CLS], state L/R, action L/R, head dino x391     = 396

Head camera tokens form the self-attention sequence; the two wrist cameras are
a memory every layer cross-attends to:

  cross-attention  left wrist dino x391, right wrist dino x391          = 782

Every token is grounded with a 3D RoPE over world-frame metres: state tokens at
their arm's EEF xyz, action tokens at their arm's first commanded xyz, head
dino and track tokens at the head patch centre, wrist patches at the world xyz
unprojected from the wrist depth. In cross-attention q
and k are rotated by their own coordinates, so q.k depends on the 3D offset
between a head-side token and a wrist patch. Tokens without a valid coordinate
(CLS, depth-invalid patches) pass through unrotated.

The readout is the CLS token through a 3-layer MLP; its output is the logit
log f/(1-f) of "this decision step belongs to a successful episode".
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope3d import RoPE3D

N_WRIST = 2


class TrackTemporalEncoder(nn.Module):
    """[B, N, T, 3] displacement trajectory -> one token per point (flatten T*3, 2-layer MLP)."""

    def __init__(self, track_horizon: int, temporal_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(track_horizon * 3, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, out_dim),
        )

    def forward(self, tracks: Tensor) -> Tensor:
        B, N, T, _ = tracks.shape
        return self.mlp(tracks.reshape(B, N, T * 3))


def sinusoidal_2d(height: int, width: int, dim: int, device, temperature: float = 10000.0) -> Tensor:
    """2D sin-cos patch-grid encoding, [height*width, dim]."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class RoPEAttention(nn.Module):
    """Multi-head attention with 3D RoPE applied to q and k by their own
    coordinate tables and masks; serves both self- and cross-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, rope: RoPE3D, q_rope, k_rope) -> Tensor:
        """`q_rope` / `k_rope`: ((cos, sin) [B, N, head_dim], mask [B, N])."""
        q = rope.apply(self._split(self.q_proj(query)), *q_rope[0], rope_mask=q_rope[1])
        k = rope.apply(self._split(self.k_proj(key)), *k_rope[0], rope_mask=k_rope[1])
        v = self._split(self.v_proj(value))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        B, _, N, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


class MVLayer(nn.Module):
    """Pre-norm self-attn -> wrist cross-attn -> FFN. Additive position
    encodings go into Q and K only."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, pos, mem, mem_pos, rope, x_rope, mem_rope) -> Tensor:
        h = self.norm1(x)
        qk = h + pos
        x = x + self.dropout1(self.self_attn(qk, qk, h, rope, x_rope, x_rope))

        h = self.norm2(x)
        m = self.norm_mem(mem)
        x = x + self.dropout2(self.cross_attn(h + pos, m + mem_pos, m, rope, x_rope, mem_rope))

        h = self.norm3(x)
        return x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(h)))))


class _MultiViewSVM(nn.Module):
    """Shared trunk: CLS, state tokens, wrist memory, layers and readout.
    Subclasses add the action-side tokens in `_head_side`."""

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 3200,
        dropout: float = 0.1,
        rope_spatial_scale: float = 0.01,
        rope_spatial_base: float = 100.0,
        arm_dim: int = 8,
        patch_grid: tuple[int, int] = (17, 23),
        horizon: int = 16,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_grid = tuple(patch_grid)
        self.arm_dim = arm_dim
        self.horizon = horizon

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_pos = nn.Embedding(1, d_model)
        self.state_encoder = nn.Linear(arm_dim, d_model)
        self.state_pos = nn.Embedding(2, d_model)
        self.arm_embed = nn.Embedding(2, d_model)  # left=0, right=1
        self.wrist_view_embed = nn.Embedding(N_WRIST, d_model)

        self.layers = nn.ModuleList([MVLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(inplace=True),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )
        self.rope = RoPE3D(head_dim=d_model // n_heads, spatial_base=rope_spatial_base,
                           spatial_scale=rope_spatial_scale)
        self._grid_pos_cache: Optional[Tensor] = None

    def _init_weights(self) -> None:
        """Training init: learned tokens (CLS, track type / invalid) and every
        embedding table ~ trunc_normal(std 0.02); all else PyTorch defaults.
        Subclasses call it once their own parameters exist."""
        for p in (*self._parameters.values(), *(m.weight for m in self.modules() if isinstance(m, nn.Embedding))):
            nn.init.trunc_normal_(p, std=0.02)

    def _grid_pos(self, device, dtype) -> Tensor:
        if self._grid_pos_cache is None or self._grid_pos_cache.device != device:
            self._grid_pos_cache = sinusoidal_2d(*self.patch_grid, self.d_model, device)
        return self._grid_pos_cache.to(dtype)

    def _head_side(self, batch: dict[str, Tensor], grid: Tensor, head_dino: Tensor, head_c: Tensor,
                   head_ok: Tensor) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Segments after CLS + state: (tokens [B,n,D], pos [n,D], coords [B,n,3], rope_ok [B,n])."""
        raise NotImplementedError

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """`batch` (see `preprocess.build_batch`):
            dino       [B, 3, 391, 1024]  head, left wrist, right wrist
            centres    [B, 3, 391, 3]     patch centre world xyz, metres
            centre_ok  [B, 3, 391]        bool, depth-valid patch centre
            state      [B, 16]            normalized robot state
            state_xyz  [B, 2, 3]          raw L/R EEF xyz, metres
        plus the variant's action inputs. Returns logits [B]."""
        dino, centres, centre_ok = batch["dino"], batch["centres"], batch["centre_ok"]
        B = dino.shape[0]
        device, dtype = dino.device, dino.dtype
        grid = self._grid_pos(device, dtype)

        state_tok = self.state_encoder(batch["state"].reshape(B, 2, self.arm_dim)) + self.arm_embed.weight
        segs = [
            (self.cls_token.expand(B, -1, -1).to(dtype), self.cls_pos.weight,
             torch.zeros(B, 1, 3, device=device), torch.zeros(B, 1, dtype=torch.bool, device=device)),
            (state_tok, self.state_pos.weight, batch["state_xyz"], torch.ones(B, 2, dtype=torch.bool, device=device)),
            *self._head_side(batch, grid, dino[:, 0], centres[:, 0], centre_ok[:, 0]),
        ]
        x = torch.cat([s[0] for s in segs], dim=1)
        pos = torch.cat([s[1] for s in segs], dim=0).unsqueeze(0)
        x_rope = (self.rope.build_cos_sin(torch.cat([s[2].float() for s in segs], 1)),
                  torch.cat([s[3] for s in segs], 1))

        mem = dino[:, 1:].reshape(B, N_WRIST * grid.shape[0], -1)
        mem_pos = (grid.unsqueeze(0) + self.wrist_view_embed.weight.to(dtype)[:, None]).reshape(1, -1, self.d_model)
        mem_rope = (self.rope.build_cos_sin(centres[:, 1:].reshape(B, -1, 3).float()),
                    centre_ok[:, 1:].reshape(B, -1))

        for layer in self.layers:
            x = layer(x, pos, mem, mem_pos, self.rope, x_rope, mem_rope)
        return self.head(self.final_norm(x[:, 0])).squeeze(-1)


class MultiViewTrackSVM(_MultiViewSVM):
    """f(s, track) -> logit. Extra batch keys: track [B,391,16,3] (normalized),
    valid [B,391] bool. Track tokens sit at their head patch's grid position
    and 3D coordinate."""

    def __init__(self, track_temporal_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.track_temporal_encoder = TrackTemporalEncoder(self.horizon, track_temporal_dim, self.d_model)
        self.invalid_embed = nn.Parameter(torch.zeros(self.d_model))
        self.track_type_embed = nn.Parameter(torch.zeros(self.d_model))
        self._init_weights()

    def _head_side(self, batch, grid, head_dino, head_c, head_ok):
        valid = batch["valid"]
        track_tok = self.track_temporal_encoder(batch["track"] * valid[..., None, None])
        track_tok = track_tok + (~valid)[..., None] * self.invalid_embed + self.track_type_embed
        return [(head_dino, grid, head_c, head_ok), (track_tok, grid, head_c, head_ok)]


class MultiViewActionSVM(_MultiViewSVM):
    """f(s, action chunk) -> logit. Extra batch keys: action [B,16,16]
    (normalized commanded poses), action_xyz [B,2,3] (raw L/R xyz of the first
    action, metres). One token per arm = MLP(flatten(16 x 8))."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.horizon * self.arm_dim, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        self.action_pos = nn.Embedding(2, self.d_model)
        self._init_weights()

    def _head_side(self, batch, grid, head_dino, head_c, head_ok):
        B = head_dino.shape[0]
        chunk = batch["action"].reshape(B, self.horizon, 2, self.arm_dim).transpose(1, 2)
        act_tok = self.action_encoder(chunk.reshape(B, 2, self.horizon * self.arm_dim)) + self.arm_embed.weight
        act_ok = torch.ones(B, 2, dtype=torch.bool, device=head_dino.device)
        return [(act_tok, self.action_pos.weight, batch["action_xyz"], act_ok), (head_dino, grid, head_c, head_ok)]
