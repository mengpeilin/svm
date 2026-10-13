"""End-to-end scorer: raw multi-view observation + track -> SVM logit / dense reward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .model import MultiViewTrackSVM
from .preprocess import Normalizer, build_batch, dino_features, load_dino

CAMERAS = ("head", "left_wrist", "right_wrist")


def svm_reward(logit: torch.Tensor | np.ndarray, lam: float = 0.1, beta: float = 20.0):
    """r = lam * clip(logit, -beta, beta); the logit already is log f/(1-f).
    The outcome reward r_out(s) of the SVM paper is added by the RL code."""
    return lam * logit.clip(-beta, beta)


class MVTrackSVMReward:
    def __init__(self, ckpt_dir: str | Path, device: str | None = None, dino: torch.nn.Module | None = None):
        """`ckpt_dir` holds `model.safetensors` + `config.json`. Pass `dino`
        to share a DINOv2-reg ViT-L/14 you already have loaded (e.g. the GAP
        policy's); otherwise one is loaded from torch.hub."""
        ckpt_dir = Path(ckpt_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = json.loads((ckpt_dir / "config.json").read_text())
        self.model = MultiViewTrackSVM(**self.config["model_config"]).to(self.device).eval()
        self.model.load_state_dict(load_file(ckpt_dir / "model.safetensors"))
        self.norm = Normalizer(**self.config["normalizer"])
        self.dino = dino if dino is not None else load_dino(self.device)

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs) -> "MVTrackSVMReward":
        from huggingface_hub import snapshot_download

        return cls(snapshot_download(repo_id), **kwargs)

    @torch.no_grad()
    def logits(self, obs: dict[str, np.ndarray], head_dino: np.ndarray | None = None, chunk: int = 32) -> np.ndarray:
        """[B] logits for B decision steps.

        `obs` keys (leading dim B): `{cam}_rgb` [240,320,3] uint8, `{cam}_depth`
        [240,320] mm, `{cam}_K` [3,3], `{cam}_E` [3,4] world->cam for cam in
        CAMERAS; `state` [16]; `track` [391,16,3] m; `track_valid` [391] bool.
        `head_dino` [B,391,1024] skips re-extracting the head features when the
        policy already computed them."""
        out = []
        for s in range(0, len(obs["state"]), chunk):
            o = {k: v[s : s + chunk] for k, v in obs.items()}
            feats = [torch.as_tensor(np.asarray(head_dino[s : s + chunk], np.float32), device=self.device)
                     if head_dino is not None else dino_features(self.dino, o["head_rgb"], self.device)]
            feats += [dino_features(self.dino, o[f"{c}_rgb"], self.device) for c in CAMERAS[1:]]
            batch = build_batch(o, torch.stack(feats, 1), self.norm, self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                out.append(self.model(batch).float().cpu().numpy())
        return np.concatenate(out)

    def rewards(self, obs: dict[str, np.ndarray], lam: float = 0.1, beta: float = 20.0, **kwargs) -> np.ndarray:
        return svm_reward(self.logits(obs, **kwargs), lam, beta)
