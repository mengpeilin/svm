"""The student's data: `data.Frames` over the cotrained tasks plus each task's
teacher label on every frame.

The student V(s_t, track) reads the observation BEFORE a chunk and the 16-step
3D track the policy then executes, and is trained on two inputs per sample
frame t, both with the task's teacher logits as targets:

  static   s_t, zero track           -> T_t
  forward  s_t, the executed track_t -> T_{t+16}

(T of the episode's last frame when the episode ends inside the chunk). Its
progress reward of a chunk is r = V(s_t, track_t) - V(s_t, 0) and its forecast
of the next decision's value is V(s_t, track_t). Each task has its own teacher
(runs/<teacher>/labels_<task>.npz), so logit scales differ between tasks.

Two student architectures read different observations: "dino" (`model.py`,
`Frames.inputs`) and "qwen3d" (`qwen_model.py`, `Frames.qwen_inputs`).
"""

from __future__ import annotations

import numpy as np
import torch

from .batching import prefetched
from .config import CHUNK, FAILURE, SUCCESS, run_dir
from .data import Frames, Normalizer
from .model import predict

TRACKS = ("zero", "real", "mirrored")
PAIR_KEYS = ("track", "track_m", "valid", "target")   # per input; every other key is the observation


def double(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Repeat the observation of a `pair_batch` (read once) for its static and forward halves."""
    return {k: v if k in PAIR_KEYS else torch.cat([v, v]) for k, v in batch.items()}


class StudentData:
    def __init__(self, teachers: dict[str, str], arch: str):
        """`teachers`: task -> teacher run dir name; `arch`: "dino" | "qwen3d"."""
        self.teachers = dict(teachers)
        self.fr = fr = Frames(tuple(self.teachers))
        self.obs = {"dino": fr.inputs, "qwen3d": fr.qwen_inputs}[arch]
        labels = []
        for k, task in enumerate(fr.tasks):
            logit = np.load(run_dir(self.teachers[task]) / f"labels_{task}.npz")["logit"]
            assert len(logit) == (fr.task == k).sum(), f"labels_{task}.npz must label every frame of every root"
            labels.append(logit)
        self.teacher = np.concatenate(labels)
        self.next = np.minimum(np.arange(len(self.teacher)) + CHUNK, fr.episode_end - 1)
        self.next_logit = self.teacher[self.next]

    def train_frames(self, label: int) -> np.ndarray:
        return self.fr.frames(label, "train", "sample")

    def eval_frames(self) -> np.ndarray:
        """Every sample frame of the eval episodes, sorted."""
        return np.sort(np.concatenate([self.fr.frames(label, "eval", "sample") for label in (SUCCESS, FAILURE)]))

    def pair_batch(self, frames: np.ndarray, norm: Normalizer, device) -> tuple[dict, torch.Tensor]:
        """Training batch of `frames`: the observation once (`double` repeats it)
        with the tracks [zero; real], and the targets [T_t; T_{t+16}]."""
        track, valid = self.fr.tracks(frames)
        b = self.obs(frames, norm, device, np.concatenate([np.zeros_like(track), track]),
                     np.concatenate([valid, valid]))
        target = np.concatenate([self.teacher[frames], self.next_logit[frames]])
        return b, torch.as_tensor(target, dtype=torch.float32)

    def values(self, model, frames: np.ndarray, track: str, norm: Normalizer, device, chunk: int = 64) -> np.ndarray:
        """V(s_t, track) on `frames` with the zero, real or mirrored (-track_t) track."""
        def batches():
            for s in range(0, len(frames), chunk):
                f = frames[s:s + chunk]
                tr, valid = self.fr.tracks(f)
                tr = {"zero": 0.0 * tr, "real": tr, "mirrored": -tr}[track]
                yield self.obs(f, norm, device, tr, valid)
        return predict(model, prefetched(batches()))
