"""The caches, the episode splits, and model batches.

A dataset's cache (`build_cache.py`, `build_wrist.py`) under cache/<dataset>/:

  dino_f16.npy        f16  [N, 391, 1024]     head DinoV2 features, every frame
  tracks_f16.npy      f16  [N, 391, 16, 3]    GT 16-step 3D track of each head patch
  vectors.npz         robot_state [N, 16], action [N, 16], patch_centres [N, 391, 3]
                      (head, world m), valid_point_mask [N, 391], label [N],
                      episode_ends [E]
  episodes.npz        the parts' per-episode datasets (seed, success, branch_frame, ...)
  wrist_frames.npy    int64 [R]               frames with wrist rows (sorted)
  wrist_dino_f16.npy  f16  [R, 2, 391, 1024]  left, right wrist DinoV2 features
  wrist_centres.npy   f32  [R, 2, 391, 3]     wrist patch centres, world m
  wrist_ok.npy        bool [R, 2, 391]        wrist depth validity

`Frames` puts several datasets on ONE flat frame axis ([scan | branch]) and
assembles the multi-view model's batches. Nothing in the caches is
normalized: the model's state / track inputs go through `Normalizer`, while
the 3D RoPE coordinates (patch centres, EEF xyz) stay in raw world metres.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import ARM_DIM, CHUNK, DATASETS, FAILURE, FAILURE_CAP, REPLAY_TOL_MM, SUCCESS, cache_dir

WRIST_FILES = {"frames": "wrist_frames.npy", "dino": "wrist_dino_f16.npy",
               "centres": "wrist_centres.npy", "ok": "wrist_ok.npy"}


def bounds_of(ends: np.ndarray) -> np.ndarray:
    """[E, 2] episode [start, end) frames from episode_ends."""
    ends = ends.astype(np.int64)
    return np.stack([np.concatenate([[0], ends[:-1]]), ends], 1)


def episode_bounds(dataset: str) -> np.ndarray:
    with np.load(cache_dir(dataset) / "vectors.npz") as v:
        return bounds_of(v["episode_ends"])


def split_episodes(dataset: str, labels: np.ndarray, meta: dict) -> dict[tuple[int, str], list[int]]:
    """{(label, "train" | "eval"): episodes}. Per class, the last `n_eval`
    episodes (highest seeds) are eval; branches whose replay missed their
    parent's state are dropped."""
    n_eval = DATASETS[dataset].n_eval
    use = np.ones(len(labels), bool)
    if DATASETS[dataset].branch:
        use = meta["replay_err_mm"] <= REPLAY_TOL_MM
    out = {}
    for label in (SUCCESS, FAILURE):
        eps = np.flatnonzero(labels == label)
        cut = len(eps) - n_eval
        out[(label, "train")] = [int(e) for e in eps[:cut] if use[e]]
        out[(label, "eval")] = [int(e) for e in eps[cut:]]
    return out


class Normalizer:
    """Model-input normalization: robot state per dimension to [-1, 1] by its
    train min / max (a constant dimension gets a unit range), tracks by ONE
    global RMS over valid components (zero offset, so zero motion stays zero
    and directions are undistorted)."""

    RANGE_EPS = 1e-4

    def __init__(self, state_lo, state_hi, track_rms):
        self.state_lo = np.asarray(state_lo, dtype=np.float32)
        self.state_hi = np.asarray(state_hi, dtype=np.float32)
        self.track_rms = float(track_rms)

    @classmethod
    def fit(cls, states: np.ndarray, track_rows: list[np.ndarray]) -> "Normalizer":
        """`states` [n, 16]; `track_rows` [391, 16, 3] tracks with invalid points removed."""
        lo, hi = states.min(0), states.max(0)
        hi = np.where(hi - lo < cls.RANGE_EPS, lo + 1.0, hi)
        sq = sum(float((np.asarray(t, np.float64) ** 2).sum()) for t in track_rows)
        n = sum(t.size for t in track_rows)
        return cls(lo, hi, max(float(np.sqrt(sq / max(n, 1))), 1e-4))

    def state_norm(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.state_lo) / (self.state_hi - self.state_lo) - 1.0

    def track_norm(self, x: np.ndarray) -> np.ndarray:
        return x / self.track_rms

    def state_dict(self) -> dict:
        return {"state_lo": self.state_lo.tolist(), "state_hi": self.state_hi.tolist(), "track_rms": self.track_rms}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["state_lo"], d["state_hi"], d["track_rms"])


def arm_xyz(state: np.ndarray) -> np.ndarray:
    """[B, 16] -> [B, 2, 3] raw world xyz of the left / right EEF (RoPE coordinates)."""
    return np.stack([state[:, 0:3], state[:, ARM_DIM:ARM_DIM + 3]], 1)


class Cache:
    """One dataset's memmapped cache."""

    def __init__(self, dataset: str, wrist: bool = True):
        d = cache_dir(dataset)
        self.name = dataset
        self.dino = np.load(d / "dino_f16.npy", mmap_mode="r")
        self.tracks = np.load(d / "tracks_f16.npy", mmap_mode="r")
        with np.load(d / "vectors.npz") as v:
            self.state, self.action = v["robot_state"], v["action"]
            self.centres, self.valid, self.label = v["patch_centres"], v["valid_point_mask"], v["label"]
            self.bounds = bounds_of(v["episode_ends"])
        with np.load(d / "episodes.npz") as m:
            self.meta = dict(m)
        self.splits = split_episodes(dataset, self.label[self.bounds[:, 0]], self.meta)
        self.wrist = None
        if wrist:
            self.wrist = {k: np.load(d / f, mmap_mode=None if k == "frames" else "r") for k, f in WRIST_FILES.items()}

    def __len__(self) -> int:
        return len(self.label)


class Frames:
    """Datasets on one flat frame axis: frame i of dataset k is offset[k] + i.

    Per-frame vectors are concatenated (state, centres, valid, label,
    episode_start / episode_end as global frames); the heavy rows (head dino,
    tracks, wrist rows) are gathered from each dataset's memmaps. `episodes`
    lists (dataset, episode, global start, global end); `splits` maps (label,
    "train" | "eval") to indices into it.
    """

    def __init__(self, datasets: tuple[str, ...] = ("scan", "branch"), wrist: bool = True):
        self.caches = [Cache(n, wrist) for n in datasets]
        self.offsets = np.cumsum([0] + [len(c) for c in self.caches])
        cat = lambda k: np.concatenate([getattr(c, k) for c in self.caches])  # noqa: E731
        self.state, self.centres, self.valid, self.label = cat("state"), cat("centres"), cat("valid"), cat("label")
        self.episodes, self.splits = [], {(label, s): [] for label in (SUCCESS, FAILURE) for s in ("train", "eval")}
        start, end = np.empty(len(self.label), np.int64), np.empty(len(self.label), np.int64)
        for c, lo in zip(self.caches, self.offsets):
            for key, eps in c.splits.items():
                self.splits[key] += [len(self.episodes) + e for e in eps]
            for s, e in c.bounds:
                start[lo + s:lo + e], end[lo + s:lo + e] = lo + s, lo + e
            self.episodes += [(c.name, ep, lo + s, lo + e) for ep, (s, e) in enumerate(c.bounds)]
        self.episode_start, self.episode_end = start, end
        if wrist:
            self.wrist_row = np.full(len(self.label), -1, np.int64)
            for c, lo in zip(self.caches, self.offsets):
                self.wrist_row[lo + c.wrist["frames"]] = np.arange(len(c.wrist["frames"]))

    def branch_frame(self, episode: int) -> int:
        """Episode-relative frame where a noise branch leaves its parent (0 for a scan episode)."""
        name, ep, _, _ = self.episodes[episode]
        cache = self.caches[[c.name for c in self.caches].index(name)]
        return int(cache.meta["branch_frame"][ep]) if DATASETS[name].branch else 0

    def frames(self, label: int, split: str, kind: str) -> np.ndarray:
        """Sample frames of a split, episode by episode:

          decision  every CHUNK-th frame from the episode start (from the branch
                    frame for a branch: earlier frames replay the parent) -- the
                    teacher's training steps
          sample    every frame with a full chunk ahead (t <= end - 16), from
                    branch_frame - 15 for a branch (earlier frames see the
                    parent's observation AND track) -- the student's

        Failures stop at FAILURE_CAP frames from their start."""
        out = []
        for e in self.splits[(label, split)]:
            _, _, s, t = self.episodes[e]
            bf = self.branch_frame(e)
            if kind == "decision":
                stop = min(t, s + FAILURE_CAP) if label == FAILURE else t
                out.append(np.arange(s + bf, stop, CHUNK))
            else:
                stop = t - (CHUNK - 1)
                if label == FAILURE:
                    stop = min(stop, s + FAILURE_CAP)
                out.append(np.arange(s + max(bf - (CHUNK - 1), 0), stop))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def _gather(self, frames: np.ndarray, rows) -> None:
        """Call rows(cache, sel, local) for each dataset with the positions
        `sel` of its frames in `frames` and their dataset-local indices, in
        sorted order (memmap reads)."""
        for c, lo in zip(self.caches, self.offsets):
            sel = np.flatnonzero((frames >= lo) & (frames < lo + len(c)))
            if len(sel):
                sel = sel[np.argsort(frames[sel], kind="stable")]
                rows(c, sel, frames[sel] - lo)

    def tracks(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GT track [B, 391, 16, 3] (zeroed where invalid) and valid [B, 391]."""
        tr = np.empty((len(frames), *self.caches[0].tracks.shape[1:]), np.float32)

        def rows(c, sel, local):
            tr[sel] = c.tracks[local]
        self._gather(frames, rows)
        valid = self.valid[frames]
        return tr * valid[..., None, None], valid

    def inputs(self, frames: np.ndarray, norm: Normalizer, device, track: np.ndarray | None = None,
               valid: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        """Model batch of `frames`: the three cameras' dino / centres / validity
        and the robot state, plus `track` [B, 391, 16, 3] with `valid` for the
        student (None: the state-only teacher)."""
        B = len(frames)
        dino = np.empty((B, 3, *self.caches[0].dino.shape[1:]), np.float32)
        centres = np.empty((B, 3, *self.centres.shape[1:]), np.float32)
        ok = np.empty((B, 3, self.centres.shape[1]), bool)
        centres[:, 0] = self.centres[frames]
        ok[:, 0] = np.abs(centres[:, 0]).sum(-1) > 1e-6

        def rows(c, sel, local):
            r = self.wrist_row[frames[sel]]
            assert (r >= 0).all(), "frame without a wrist row"
            dino[sel, 0] = c.dino[local]
            dino[sel, 1:], centres[sel, 1:], ok[sel, 1:] = c.wrist["dino"][r], c.wrist["centres"][r], c.wrist["ok"][r]
        self._gather(frames, rows)
        t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
        state = self.state[frames]
        out = {"dino": t(dino), "centres": t(centres), "centre_ok": t(ok, torch.bool),
               "state": t(norm.state_norm(state)), "state_xyz": t(arm_xyz(state))}
        if track is not None:
            out["track"] = t(norm.track_norm(track))
            out["valid"] = t(valid, torch.bool)
        return out
