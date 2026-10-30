"""Training data: the decision-step cache written by `svm.cache`, its train /
held-out split, normalizer fitting and batch assembly.

One sample is one policy decision: the observation at frame c * CHUNK of an
episode with the CHUNK-step track / commanded action chunk from there. Indices
are FRAME indices into the cache's flat episode axis; `rows` maps a decision
frame to its row in the per-decision arrays.

Split: cache episodes are seed-sorted per class, and the last `holdout`
episodes (highest seeds) of each class are held out. They select the
checkpoint and give the reported numbers. Failure episodes are cut to their
first `failure_cap` frames in every pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .cache import FILES
from .preprocess import CHUNK, Normalizer, arm_xyz

RANGE_EPS = 1e-4


class SVMData:
    """Memmapped cache plus the (label, pool) -> decision-frame pools, label
    1 = success / 0 = failure, pool "train" / "heldout". Nothing is normalized
    here; RoPE reads the raw world-frame metres in centres / state / action."""

    def __init__(self, cache: str | Path, holdout: int = 50, failure_cap: int | None = 250):
        cache = Path(cache)
        if not (cache / "dino_done.json").exists():
            raise FileNotFoundError(f"{cache}: no dino_done.json; run `svm.cache geometry`, then `svm.cache dino`")
        m = {k: np.load(cache / f, mmap_mode="r") for k, (f, _, _) in FILES.items() if k != "rgb"}
        self.dino, self.tracks, self.track_valid = m["dino"], m["tracks"], m["track_valid"]
        self.centres, self.centre_ok = m["centres"], m["centre_ok"]
        vec = np.load(cache / "vectors.npz")
        self.state, self.action, self.rows = vec["robot_state"], vec["action"], vec["rows"]
        ends = vec["episode_ends"]
        self.bounds = np.stack([np.concatenate([[0], ends[:-1]]), ends], 1)
        self.episode_end = np.repeat(ends, np.diff(np.concatenate([[0], ends])))
        self.success = vec["label"][self.bounds[:, 0]].astype(bool)
        self.failure_cap = failure_cap

        n_succ = int(self.success.sum())
        by_class = {1: np.arange(n_succ), 0: np.arange(n_succ, len(ends))}
        if not self.success[:n_succ].all():
            raise ValueError(f"{cache}: episodes are not ordered successes first")
        self.episodes, self.pools = {}, {}
        for label, eps in by_class.items():
            if len(eps) <= holdout:
                raise ValueError(f"{cache}: {len(eps)} episodes of class {label}, holdout {holdout}")
            for pool, sel in (("train", eps[:-holdout]), ("heldout", eps[-holdout:])):
                self.episodes[(label, pool)] = sel
                self.pools[(label, pool)] = np.concatenate([self.steps(e) for e in sel])

    def __repr__(self) -> str:
        n = {k: len(v) for k, v in self.pools.items()}
        return (
            f"<SVMData {len(self.bounds)} episodes, {len(self.state)} frames, {len(self.rows)} decision rows; "
            f"decision steps train {n[(1, 'train')]}+/{n[(0, 'train')]}-, "
            f"held-out {n[(1, 'heldout')]}+/{n[(0, 'heldout')]}->"
        )

    def steps(self, ep: int, capped: bool = True) -> np.ndarray:
        """Decision frames of episode `ep` (a failure cut to `failure_cap` frames unless not `capped`)."""
        s, e = self.bounds[ep]
        if capped and self.failure_cap is not None and not self.success[ep]:
            e = min(e, s + self.failure_cap)
        return np.arange(s, e, CHUNK)

    def row(self, idx: np.ndarray) -> np.ndarray:
        r = np.searchsorted(self.rows, idx)
        if not np.array_equal(self.rows[r], idx):
            raise IndexError("frame indices must be decision frames")
        return r

    def action_chunk(self, idx: np.ndarray) -> np.ndarray:
        """[B, CHUNK, 16] actions executed from each decision frame; an episode
        that ends mid-chunk holds its last action."""
        t = idx[:, None] + np.arange(CHUNK)[None]
        return self.action[np.minimum(t, self.episode_end[idx, None] - 1)]


def limits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (min, max); a dim with no range gets a unit range so it maps to a constant."""
    lo, hi = x.min(0), x.max(0)
    hi = np.where(hi - lo < RANGE_EPS, lo + 1.0, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def fit_normalizer(data: SVMData, inputs: str, max_track_frames: int = 1500) -> Normalizer:
    """Fit on the TRAIN decision frames: state (and action) min/max -> [-1, 1];
    tracks one isotropic RMS over the valid components of up to
    `max_track_frames` evenly spaced frames (zero stays zero, directions are
    not sheared)."""
    idx = np.sort(np.concatenate([data.pools[(1, "train")], data.pools[(0, "train")]]))
    state_lo, state_hi = limits(data.state[idx])
    if inputs == "action":
        action_lo, action_hi = limits(data.action_chunk(idx).reshape(-1, data.action.shape[1]))
        return Normalizer(state_lo, state_hi, action_lo=action_lo, action_hi=action_hi)
    sq_sum, n_elem = 0.0, 0
    for r in data.row(idx[np.linspace(0, len(idx) - 1, min(max_track_frames, len(idx))).astype(int)]):
        v = np.asarray(data.tracks[r][data.track_valid[r]], dtype=np.float64)
        sq_sum += float((v**2).sum())
        n_elem += v.size
    rms = max(float(np.sqrt(sq_sum / max(n_elem, 1))), 1e-4)
    print(f"[normalizer] track rms = {rms * 1000:.2f} mm ({n_elem} valid components)")
    return Normalizer(state_lo, state_hi, track_rms=rms)


def assemble(data: SVMData, idx: np.ndarray, norm: Normalizer, inputs: str, device) -> dict[str, torch.Tensor]:
    """Model batch (the keys of `preprocess.build_batch`) for the decision
    frames `idx`, in SORTED order (memmap reads)."""
    idx = np.sort(idx)
    r = data.row(idx)
    state = data.state[idx]
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)  # noqa: E731
    out = {
        "dino": t(np.asarray(data.dino[r], dtype=np.float32)),
        "centres": t(data.centres[r]),
        "centre_ok": t(data.centre_ok[r], torch.bool),
        "state": t(norm.state_norm(state)),
        "state_xyz": t(arm_xyz(state)),
    }
    if inputs == "track":
        out["track"] = t(norm.track_norm(np.asarray(data.tracks[r], dtype=np.float32)))
        out["valid"] = t(data.track_valid[r], torch.bool)
    else:
        action = data.action_chunk(idx)
        out["action"] = t(norm.action_norm(action))
        out["action_xyz"] = t(arm_xyz(action[:, 0]))
    return out
