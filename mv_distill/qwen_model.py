"""The Qwen-3D student V(s[, track]): Qwen-3D (Qwen2.5-VL-3B + LoRA, pretrained on
3D grounding / QA; github.com/ll220/qwen3d, HF katefgroup/Qwen-3D
qwen3d_3b.pth) finetuned with a fresh LoRA to forecast the task teacher's value.

Observation tokens (t = 0), from the FROZEN Qwen ViT run online on the RGB:
  head         240 x 320 resized to 476 x 644 (2x the DinoV2 input 238 x 322):
               17 x 23 = 391 merged tokens, one per DinoV2 patch, at the cached
               DinoV2 patch centres
  wrists       native size (Qwen's smart_resize: 252 x 308): 9 x 11 = 99 merged
               tokens each, at the cached wrist Qwen token centres
  -> voxel pooling as in Qwen-3D: the tokens with a valid centre inside the
     `workspace` box (config.WORKSPACE: above the tabletop, x / y within its edges) are
     binned by their world xyz into `voxel`-metre cells, features and xyz
     averaged per cell.

Future tokens, from the 16-step GT track of each head patch in the workspace:
for k in 4, 8, 12, 16, every patch that has moved by step k (displacement
above `move_thresh` at some step <= k; resting tracked points carry sub-mm sim
jitter) is put back into the sequence at its moved position centre +
track[k - 1] and time t = k, carrying its own ViT feature plus
TrackEncoder(its track up to step k). A zero track moves nothing, so the
static input V(s, 0) is the observation alone.

Language, in Qwen-3D's chat template around the point tokens:
  "Task: {full_description}. Estimate how likely the task is to succeed."
V = MLP(final hidden state of the prompt's last token).

Positions: Qwen-3D's 4D RoPE [t, x, y, z] (raw world metres) through the
vendored LLM (`qwen3d_vendor`): point tokens (t, x, y, z), text tokens (t, 0, 0,
0) numbered as Qwen-3D does (prefix 0..P-1, suffix from P on). Attention is
bidirectional; rows are padded and masked, so batches of any size work
(upstream Qwen-3D: batch size 1).

Trainable: the new LoRA (LLM linear layers), the track encoder and the value
head. Frozen: the ViT and the LLM with Qwen-3D's LoRA merged in. A checkpoint
stores only the trainable weights; the base comes from config["qwen3d_ckpt"].
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CHUNK, IMG_HW, N_PATCHES, PATCH_GRID, TASK_TEXT, WORKSPACE
from .build_centres import WRIST_Q_GRID

FUTURE_KS = (4, 8, 12, 16)
HEAD_HW = (PATCH_GRID[0] * 28, PATCH_GRID[1] * 28)          # 476 x 644
WRIST_HW = (WRIST_Q_GRID[0] * 28, WRIST_Q_GRID[1] * 28)     # 252 x 308
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)              # Qwen2.5-VL preprocessor_config.json
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>",
          "<|vision_end|>{sentence}<|im_end|>\n<|im_start|>assistant\n")   # Qwen-3D TEXT_INPUT_GEN_3D
SENTENCE = "Task: {desc}. Estimate how likely the task is to succeed."
TRACK_HIDDEN = 512
HEAD_HIDDEN = 256
QWEN3D_LORA = (8, 16)           # rank, alpha of Qwen-3D's own LoRA (its config.py)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ------------------------------------------------------------------ base model


def load_base(qwen_dir: str, ckpt: str):
    """Qwen-3D's Qwen2.5-VL (bf16, on CPU) with its LoRA merged into the base weights."""
    from transformers.modeling_utils import no_init_weights

    from .qwen3d_vendor.config_qwen2_5_vl_modified import Qwen2_5_3D_VLConfig
    from .qwen3d_vendor.modeling_qwen2_5_vl_modified import Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_3D_VLConfig.from_pretrained(qwen_dir)
    config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with no_init_weights():
            model = Qwen2_5_VLForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default)
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["model"]
    prefix = "qwen_model.base_model.model."
    r, alpha = QWEN3D_LORA
    merged = {}
    for k, v in sd.items():
        if not k.startswith(prefix) or ".lora_" in k or k.startswith(prefix + "lm_head."):
            continue
        if k.endswith(".base_layer.weight"):
            a = sd[k.replace(".base_layer.weight", ".lora_A.default.weight")].float()
            b = sd[k.replace(".base_layer.weight", ".lora_B.default.weight")].float()
            v = v.float() + (alpha / r) * (b @ a)
        merged[k[len(prefix):].replace(".base_layer.", ".")] = v.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    # lm_head (tied to embed_tokens in Qwen2.5-VL-3B) is not used
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    model.lm_head = None
    model.model.rotary_emb_4d.dtype = torch.bfloat16
    model.requires_grad_(False)
    return model


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B(A(dropout(x))); B starts at zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> Tensor:          # read by the vendored attention's dtype check
        return self.base.weight

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def add_lora(llm: nn.Module, r: int, alpha: float, dropout: float) -> None:
    for layer in llm.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                if hasattr(parent, name):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, dropout))


# ------------------------------------------------------------------ ViT


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def patchify(img: Tensor) -> Tensor:
    """[n, 3, H, W] normalized -> [n, H/14 * W/14, 1176] in Qwen2VLImageProcessor's
    order (2 x 2 merge units contiguous, the image repeated over the 2 temporal steps)."""
    n, c, H, W = img.shape
    x = img.unsqueeze(1).expand(n, 2, c, H, W)
    x = x.reshape(n, 2, c, H // 28, 2, 14, W // 28, 2, 14)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(n, (H // 14) * (W // 14), c * 2 * 14 * 14)


class BatchedViT(nn.Module):
    """Qwen2.5-VL's ViT forward over a batch of SAME-SIZE images as a batch
    dimension (upstream packs all images into one sequence with an O(N^2)
    block mask). Same math (fp32: equal to upstream to 1e-5 relative): window
    attention over the windows padded to 64 patches, full attention at
    `fullatt_block_indexes`, 2 x 2 patch merger."""

    def __init__(self, visual: nn.Module, hw: tuple[int, int]):
        super().__init__()
        self.visual = visual
        gh, gw = hw[0] // 14, hw[1] // 14
        grid = torch.tensor([[1, gh, gw]])
        self.L = gh * gw
        widx, cu = visual.get_window_index(grid)
        cu = torch.unique_consecutive(torch.tensor(cu))
        lens = cu.diff()
        wmax = int(lens.max())
        slot = torch.arange(wmax)[None] < lens[:, None]             # [n_windows, wmax]
        win_idx = torch.where(slot, cu[:-1, None] + torch.arange(wmax)[None], 0)
        rot = visual.rot_pos_emb(grid)                          # [L, head_dim / 2]
        rot = rot.view(self.L // 4, 4, -1)[widx].reshape(self.L, -1)
        emb = torch.cat([rot, rot], -1)
        self.register_buffer("widx", widx, persistent=False)
        self.register_buffer("unwidx", torch.argsort(widx), persistent=False)
        self.register_buffer("win_idx", win_idx.flatten(), persistent=False)     # window slot -> patch
        self.register_buffer("win_slot", slot, persistent=False)                    # real (not padding) slots
        self.register_buffer("win_keep", slot.flatten().nonzero().squeeze(1), persistent=False)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def _window_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """[n, heads, L, d] in window order -> attention within each window."""
        n, h, _, d = q.shape
        nw, wmax = self.win_slot.shape
        split = lambda t: t[:, :, self.win_idx].view(n, h, nw, wmax, d).transpose(1, 2).reshape(n * nw, h, wmax, d)  # noqa: E731
        mask = self.win_slot[None, :, None, None, :].expand(n, nw, 1, 1, wmax).reshape(n * nw, 1, 1, wmax)
        o = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
        return o.view(n, nw, h, wmax, d).transpose(1, 2).reshape(n, h, nw * wmax, d)[:, :, self.win_keep]

    @torch.no_grad()
    def forward(self, img: Tensor) -> Tensor:
        """[n, 3, H, W] normalized -> [n, H/28 * W/28, 2048] merged tokens, row-major."""
        v = self.visual
        n, L = img.shape[0], self.L
        x = v.patch_embed(patchify(img).reshape(n * L, -1)).view(n, L, -1)
        x = x.view(n, L // 4, 4, -1)[:, self.widx].reshape(n, L, -1)
        for i, blk in enumerate(v.blocks):
            h = blk.norm1(x)
            heads = blk.attn.num_heads
            q, k, val = blk.attn.qkv(h).view(n, L, 3, heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
            q = (q.float() * self.cos + _rotate_half(q.float()) * self.sin).to(val.dtype)
            k = (k.float() * self.cos + _rotate_half(k.float()) * self.sin).to(val.dtype)
            if i in v.fullatt_block_indexes:
                o = F.scaled_dot_product_attention(q, k, val)
            else:
                o = self._window_attention(q, k, val)
            x = x + blk.attn.proj(o.transpose(1, 2).reshape(n, L, -1))
            x = x + blk.mlp(blk.norm2(x))
        x = v.merger(x).view(n, L // 4, -1)
        return x[:, self.unwidx]


def preprocess(rgb: Tensor, hw: tuple[int, int]) -> Tensor:
    """[n, H, W, 3] uint8 RGB -> [n, 3, h, w] Qwen-normalized at size `hw` (bicubic)."""
    x = rgb.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False, antialias=True).clamp(0, 255)
    mean = x.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x / 255.0 - mean) / std


# ------------------------------------------------------------------ student


class TrackEncoder(nn.Module):
    """(track [M, 16, 3] normalized, its steps after k zeroed; k one-hot) -> [M, d].
    The last layer starts at zero: a future token starts as its patch's feature."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(CHUNK * 3 + len(FUTURE_KS), TRACK_HIDDEN), nn.GELU(),
                                 nn.Linear(TRACK_HIDDEN, d))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, track: Tensor, kk: Tensor) -> Tensor:
        steps = torch.arange(CHUNK, device=track.device)
        k = track.new_tensor(FUTURE_KS)[kk]
        x = track * (steps[None, :, None] < k[:, None, None])
        return self.mlp(torch.cat([x.flatten(1), F.one_hot(kk, len(FUTURE_KS)).to(x.dtype)], 1))


class QwenValue(nn.Module):
    """Batch keys (`data.Frames.qwen_inputs`): rgb [B, 3, 240, 320, 3] u8,
    centres [B, 391, 3] (head, raw m), centre_ok [B, 391], qcentres [B, 2, 99, 3]
    (wrists, raw m), qcentre_ok [B, 2, 99], task [B]; track [R, 391, 16, 3]
    (normalized), track_m [R, 391, 16, 3] (m), valid [R, 391] with R a multiple
    of B (row r reads observation r % B). Returns V [R]."""

    def __init__(self, config: dict, base: nn.Module):
        super().__init__()
        self.config = config
        self.qwen = base
        self.llm = base.model
        d = self.llm.config.hidden_size
        add_lora(self.llm, config["lora_r"], config["lora_alpha"], config["lora_dropout"])
        self.head_vit = BatchedViT(base.visual, HEAD_HW)
        self.wrist_vit = BatchedViT(base.visual, WRIST_HW)
        self.track_encoder = TrackEncoder(d)
        self.value_head = nn.Sequential(nn.Linear(d, HEAD_HIDDEN), nn.ReLU(inplace=True),
                                        nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.ReLU(inplace=True),
                                        nn.Linear(HEAD_HIDDEN, 1))
        for k in ("prefix_ids", "suffix_ids"):
            self.register_buffer(k, torch.tensor(config[k]), persistent=False)

    def trainable_state(self) -> dict[str, Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def _in_workspace(self, xyz: Tensor) -> Tensor:
        box = xyz.new_tensor(self.config["workspace"])
        return ((xyz >= box[:, 0]) & (xyz <= box[:, 1])).all(-1)

    def _observation(self, b: dict) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Head tokens [B, 391, d] and per-observation voxel features / xyz."""
        rgb = b["rgb"]
        B = rgb.shape[0]
        head = self.head_vit(preprocess(rgb[:, 0], HEAD_HW))
        wrist = self.wrist_vit(preprocess(rgb[:, 1:].flatten(0, 1), WRIST_HW)).view(B, -1, head.shape[-1])
        feat = torch.cat([head, wrist], 1).float()
        xyz = torch.cat([b["centres"], b["qcentres"].flatten(1, 2)], 1)
        ok = torch.cat([b["centre_ok"], b["qcentre_ok"].flatten(1, 2)], 1) & self._in_workspace(xyz)
        bi, ti = ok.nonzero(as_tuple=True)
        cell = torch.floor(xyz[bi, ti] / self.config["voxel"]).long() + 2 ** 15     # |xyz| < 2^15 voxels
        key = ((bi * 2 ** 16 + cell[:, 0]) * 2 ** 16 + cell[:, 1]) * 2 ** 16 + cell[:, 2]
        uniq, inv = torch.unique(key, return_inverse=True)
        cnt = torch.zeros(len(uniq), device=rgb.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
        vf = torch.zeros(len(uniq), feat.shape[-1], device=rgb.device).index_add_(0, inv, feat[bi, ti]) / cnt[:, None]
        vx = torch.zeros(len(uniq), 3, device=rgb.device).index_add_(0, inv, xyz[bi, ti]) / cnt[:, None]
        per = torch.bincount(uniq // 2 ** 48, minlength=B).tolist()     # unique keys are sorted, so by batch row
        return head, list(vf.split(per)), list(vx.split(per))

    def _future(self, b: dict, head: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Per row: future token features, xyz, t."""
        track, track_m, valid = b["track"], b["track_m"], b["valid"]
        R, B = track.shape[0], head.shape[0]
        obs = torch.arange(R, device=track.device) % B
        ok = valid & b["centre_ok"][obs] & self._in_workspace(b["centres"][obs])
        moved = torch.cumsum(track_m.norm(dim=-1) > self.config["move_thresh"], -1) > 0   # [R, 391, 16] by step j + 1
        ks = torch.tensor(FUTURE_KS, device=track.device)
        sel = moved[:, :, ks - 1] & ok[..., None]                                # [R, 391, 4]
        ri, pi, kk = sel.nonzero(as_tuple=True)
        f = head[obs[ri], pi].float() + self.track_encoder(track[ri, pi], kk).float()
        xyz = b["centres"][obs[ri], pi] + track_m[ri, pi, ks[kk] - 1]
        t = ks[kk].float()
        per = torch.bincount(ri, minlength=R).tolist()
        return list(f.split(per)), list(xyz.split(per)), list(t.split(per))

    def sequences(self, b: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Per row: the input embeddings [n, d] (prefix text, voxels, future
        tokens, suffix text) and their 4D positions [n, 4] (t, x, y, z)."""
        head, vox_f, vox_x = self._observation(b)
        fut_f, fut_x, fut_t = self._future(b, head)
        R, B = b["track"].shape[0], head.shape[0]
        dev = head.device
        tasks = b["task"].tolist()
        emb_tab = self.llm.embed_tokens
        P = self.prefix_ids.shape[0]
        prefix = emb_tab(self.prefix_ids).float()
        suffixes = emb_tab(self.suffix_ids).float()
        pre_pos = F.pad(torch.arange(P, device=dev, dtype=torch.float)[:, None], (0, 3))
        suf_pos = F.pad(P + torch.arange(self.suffix_ids.shape[1], device=dev, dtype=torch.float)[:, None], (0, 3))
        rows_e, rows_p = [], []
        for r in range(R):
            o = r % B
            n_suf = self.config["suffix_len"][tasks[o]]
            rows_e.append(torch.cat([prefix, vox_f[o], fut_f[r], suffixes[tasks[o], :n_suf]]))
            rows_p.append(torch.cat([pre_pos, F.pad(vox_x[o], (1, 0)), torch.cat([fut_t[r][:, None], fut_x[r]], 1),
                                     suf_pos[:n_suf]]))
        return rows_e, rows_p

    def forward(self, b: dict) -> Tensor:
        rows_e, rows_p = self.sequences(b)
        R, dev = len(rows_e), rows_e[0].device
        dtype = self.llm.embed_tokens.weight.dtype          # bf16 (fp32 only in qwen_check)
        out = torch.empty(R, device=dev)
        lens = torch.tensor([len(e) for e in rows_e])
        for chunk in torch.argsort(lens).tensor_split(min(2, R)):     # 2 length buckets: less padding
            idx = chunk.tolist()
            L = int(math.ceil(int(lens[chunk].max()) / 16) * 16)
            emb = torch.nn.utils.rnn.pad_sequence([rows_e[r] for r in idx], batch_first=True)
            pos = torch.nn.utils.rnn.pad_sequence([rows_p[r] for r in idx], batch_first=True)
            emb, pos = F.pad(emb, (0, 0, 0, L - emb.shape[1])), F.pad(pos, (0, 0, 0, L - pos.shape[1]))
            n = lens[chunk].to(dev)
            keep = torch.arange(L, device=dev)[None] < n[:, None]
            mask = torch.where(keep, 0.0, torch.finfo(dtype).min).to(dtype)
            mask = mask[:, None, None, :].expand(len(idx), 1, L, L)
            h = self.llm(inputs_embeds=emb.to(dtype), attention_mask=mask,
                         position_ids=pos.permute(2, 0, 1), rope_type="custom_txyz", use_cache=False,
                         return_dict=True).last_hidden_state
            last = h[torch.arange(len(idx), device=dev), n - 1]
            out[chunk.to(dev)] = self.value_head(last.float()).squeeze(-1).float()
        return out


def make_config(tasks: tuple[str, ...], qwen_dir: str, qwen3d_ckpt: str, voxel: float, move_thresh: float,
                lora_r: int, lora_alpha: float, lora_dropout: float) -> dict:
    """The student's config, with the prompt token ids of each task."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen_dir)
    prefix = tok(PROMPT[0], add_special_tokens=False)["input_ids"]
    suffixes = [tok(PROMPT[1].format(sentence=SENTENCE.format(desc=TASK_TEXT[t].rstrip("."))),
                    add_special_tokens=False)["input_ids"] for t in tasks]
    n = max(len(s) for s in suffixes)
    return {"arch": "qwen3d", "tasks": list(tasks), "qwen_dir": qwen_dir, "qwen3d_ckpt": qwen3d_ckpt,
            "voxel": voxel, "workspace": [list(r) for r in WORKSPACE], "move_thresh": move_thresh, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "prefix_ids": prefix, "suffix_ids": [s + [0] * (n - len(s)) for s in suffixes],
            "suffix_len": [len(s) for s in suffixes]}


def build(config: dict, device) -> QwenValue:
    model = QwenValue(config, load_base(config["qwen_dir"], config["qwen3d_ckpt"]))
    return model.to(device)


def from_checkpoint(ck: dict, device) -> QwenValue:
    model = build(ck["config"], device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not unexpected and not (trainable - set(ck["model"])), (unexpected, trainable - set(ck["model"]))
    return model
