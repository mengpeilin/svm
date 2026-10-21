"""The student's data: `data.Frames` plus the teacher's label on every frame.

The student V(s_t, track) reads the observation BEFORE a chunk and the 16-step
3D track the policy then executes, and is trained on two inputs per sample
frame t, both with the teacher's logits as targets:

  static   s_t, zero track           -> T_t
  forward  s_t, the executed track_t -> T_{t+16}

(T of the episode's last frame when the episode ends inside the chunk). Its
progress reward of a chunk is r = V(s_t, track_t) - V(s_t, 0) and its forecast
of the next decision's value is V(s_t, track_t).
"""

from __future__ import annotations

import numpy as np
import torch

from .config import CHUNK, FAILURE, SUCCESS, WRIST_STRIDE, run_dir
from .data import Frames, Normalizer
from .model import predict

TRACKS = ("zero", "real", "mirrored")


class StudentData:
    def __init__(self, teacher: str, datasets: tuple[str, ...] = ("scan", "branch")):
        self.fr = fr = Frames(datasets)
        labels = []
        for c in fr.caches:
            logit = np.load(run_dir(teacher) / f"labels_{c.name}.npz")["logit"]
            assert len(logit) == len(c), f"runs/{teacher}/labels_{c.name}.npz must label every frame"
            labels.append(logit)
        self.teacher = np.concatenate(labels)
        self.next = np.minimum(np.arange(len(self.teacher)) + CHUNK, fr.episode_end - 1)
        self.next_logit = self.teacher[self.next]

    def train_frames(self, label: int) -> np.ndarray:
        """Sample frames of the train episodes (+ branches) with t % WRIST_STRIDE == 0."""
        f = self.fr.frames(label, "train", "sample")
        f = f[(f - self.fr.episode_start[f]) % WRIST_STRIDE == 0]
        assert (self.fr.wrist_row[f] >= 0).all(), "train frames without wrist rows"
        return f

    def eval_frames(self) -> np.ndarray:
        """Every sample frame of the eval episodes, sorted."""
        return np.sort(np.concatenate([self.fr.frames(label, "eval", "sample") for label in (SUCCESS, FAILURE)]))

    def pair_batch(self, frames: np.ndarray, norm: Normalizer, device) -> tuple[dict, torch.Tensor]:
        """Training batch [static of `frames`, forward of `frames`] and its targets [T_t, T_{t+16}]."""
        both = np.concatenate([frames, frames])
        track, valid = self.fr.tracks(both)
        track[:len(frames)] = 0.0
        target = np.concatenate([self.teacher[frames], self.next_logit[frames]])
        return self.fr.inputs(both, norm, device, track, valid), torch.as_tensor(target, dtype=torch.float32)

    def values(self, model, frames: np.ndarray, track: str, norm: Normalizer, device, chunk: int = 64) -> np.ndarray:
        """V(s_t, track) on `frames` with the zero, real or mirrored (-track_t) track."""
        def batches():
            for s in range(0, len(frames), chunk):
                f = frames[s:s + chunk]
                tr, valid = self.fr.tracks(f)
                tr = {"zero": 0.0 * tr, "real": tr, "mirrored": -tr}[track]
                yield self.fr.inputs(f, norm, device, tr, valid)
        return predict(model, batches())
