"""The staged tasks, the episode splits, and model batches.

Per task and staged root, from <root>/<task>/ (`fm_stage.py`, read only):

  rgb_u8.npy        u8   [N, 3, 240, 320, 3] head, left, right wrist (RGB)
  dino_f16.npy      f16  [N, 3, 391, 1024]  DinoV2 patch features of the same cameras
  tracks_f32.npy    f32  [N, 391, 16, 3]    GT 3D track of each head patch: track[:, :, j]
                                            is its displacement (m) after j + 1 steps
  robot_state.npy   f32  [N, 16]
  label.npy         i64  [N]                1 = success episode
  episodes.npz      episode_ends, seed, success, ...

and from $MV_DISTILL_WORK/cache/<root name>/<task>/ (`build_centres.py`):

  centres_f32.npy        f32  [N, 3, 391, 3]  DinoV2 patch centres of the three cameras, world m
  centre_ok.npy          bool [N, 3, 391]
  valid.npy              bool [N, 391]        head patch has a GT track
  qcentres_wrist_f32.npy f32  [N, 2, 99, 3]   Qwen token centres of the two wrists (9 x 11 grid)
  qcentre_wrist_ok.npy   bool [N, 2, 99]

`Frames` puts several tasks (each over all staged roots, base first) on ONE
flat frame axis and assembles the model batches: `inputs` for the DinoV2
models, `qwen_inputs` for the Qwen-3D student. Nothing stored is normalized:
the state / track inputs go through `Normalizer`, while 3D RoPE coordinates
(patch centres, EEF xyz, track positions) stay in raw world metres.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from .config import ARM_DIM, CHUNK, FAILURE, N_EVAL, SUCCESS, cache_dir, data_dir, roots

MAX_TRACK_FRAMES = 1500   # per task and root, for the track RMS


def bounds_of(ends: np.ndarray) -> np.ndarray:
    """[E, 2] episode [start, end) frames from episode_ends."""
    ends = ends.astype(np.int64)
    return np.stack([np.concatenate([[0], ends[:-1]]), ends], 1)


def split_episodes(success: np.ndarray, base: bool) -> dict[tuple[int, str], list[int]]:
    """{(label, "train" | "eval"): episodes}. In the base root, the last
    `N_EVAL` episodes of each class (highest seeds) are eval; every episode of
    a later root is train."""
    out = {}
    for label in (SUCCESS, FAILURE):
        eps = np.flatnonzero(success == label)
        cut = len(eps) - N_EVAL if base else len(eps)
        out[(label, "train")] = eps[:cut].tolist()
        out[(label, "eval")] = eps[cut:].tolist()
    return out


def task_episodes(root: Path, task: str, base: bool) -> tuple[np.ndarray, dict]:
    """(episode bounds [E, 2], splits) of a staged task."""
    with np.load(data_dir(root, task) / "episodes.npz") as m:
        return bounds_of(m["episode_ends"]), split_episodes(m["success"], base)


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
    """One task's memmapped data in one staged root."""

    def __init__(self, root: Path, task: str, base: bool):
        self.dir, self.cache = data_dir(root, task), cache_dir(root, task)
        if not (self.cache / "centres.json").exists():
            raise SystemExit(f"{root.name}/{task}: run build_centres first")
        self.task, self.root, self.base = task, root, base
        self.tracks = np.load(self.dir / "tracks_f32.npy", mmap_mode="r")
        self.centres = np.load(self.cache / "centres_f32.npy", mmap_mode="r")
        self.centre_ok = np.load(self.cache / "centre_ok.npy", mmap_mode="r")
        self.valid = np.load(self.cache / "valid.npy")
        self.state = np.load(self.dir / "robot_state.npy")
        self.label = np.load(self.dir / "label.npy")
        self.bounds, self.splits = task_episodes(root, task, base)
        assert self.bounds[-1, 1] == len(self.label) == len(self.tracks) == len(self.centres)
        assert (self.label[self.bounds[:, 0]] == np.load(self.dir / "episodes.npz")["success"]).all()

    def __len__(self) -> int:
        return len(self.label)

    @cached_property
    def dino(self) -> np.ndarray:
        return np.load(self.dir / "dino_f16.npy", mmap_mode="r")

    @cached_property
    def rgb(self) -> np.ndarray:
        return np.load(self.dir / "rgb_u8.npy", mmap_mode="r")

    @cached_property
    def qcentres(self) -> tuple[np.ndarray, np.ndarray]:
        """(wrist Qwen token centres [N, 2, 99, 3], ok [N, 2, 99])."""
        if not (self.cache / "wrist_q.json").exists():
            raise SystemExit(f"{self.root.name}/{self.task}: run build_centres --products wrist_q first")
        return (np.load(self.cache / "qcentres_wrist_f32.npy", mmap_mode="r"),
                np.load(self.cache / "qcentre_wrist_ok.npy", mmap_mode="r"))


class Frames:
    """Tasks on one flat frame axis, each task's roots in order (base first):
    frame i of cache k is offsets[k] + i, and a task's frames are contiguous.

    Per-frame vectors are concatenated (state, valid, label, task index,
    episode_start / episode_end as global frames, episode_id); the heavy rows
    (dino, rgb, centres, tracks) are gathered from the memmaps. `episodes`
    lists (task, episode within the task over its roots, global start, global
    end); `splits` maps (label, "train" | "eval") to indices into it.
    """

    def __init__(self, tasks: tuple[str, ...]):
        self.tasks = tuple(tasks)
        self.caches = [Cache(root, t, i == 0) for t in self.tasks for i, root in enumerate(roots())]
        self.offsets = np.cumsum([0] + [len(c) for c in self.caches])
        cat = lambda k: np.concatenate([getattr(c, k) for c in self.caches])  # noqa: E731
        self.state, self.valid, self.label = cat("state"), cat("valid"), cat("label")
        self.task = np.concatenate([np.full(len(c), self.tasks.index(c.task), np.int64) for c in self.caches])
        self.episodes, self.splits = [], {(label, s): [] for label in (SUCCESS, FAILURE) for s in ("train", "eval")}
        n = len(self.label)
        start, end, eid = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
        first = {}
        for c, lo in zip(self.caches, self.offsets):
            ep0 = sum(1 for e in self.episodes if e[0] == c.task)
            for key, eps in c.splits.items():
                self.splits[key] += [len(self.episodes) + e for e in eps]
            for s, e in c.bounds:
                start[lo + s:lo + e], end[lo + s:lo + e] = lo + s, lo + e
                eid[lo + s:lo + e] = len(self.episodes)
                self.episodes.append((c.task, ep0, lo + s, lo + e))
                ep0 += 1
            first.setdefault(c.task, lo)
        self.episode_start, self.episode_end, self.episode_id = start, end, eid
        self.task_offsets = np.array([first[t] for t in self.tasks])

    def frames(self, label: int, split: str, kind: str) -> np.ndarray:
        """Sample frames of a split, episode by episode:

          decision  every CHUNK-th frame from the episode start -- the
                    teacher's training steps
          sample    every frame with a full chunk ahead (t <= end - 16) -- the
                    student's"""
        out = []
        for e in self.splits[(label, split)]:
            _, _, s, t = self.episodes[e]
            out.append(np.arange(s, t, CHUNK) if kind == "decision" else np.arange(s, t - (CHUNK - 1)))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def _gather(self, frames: np.ndarray, rows) -> None:
        """Call rows(cache, sel, local) for each cache with the positions `sel`
        of its frames in `frames` and their cache-local indices, in sorted
        order (memmap reads)."""
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
        """DinoV2-model batch of `frames`: the three cameras' dino / centres /
        validity and the robot state, plus `track` [B', 391, 16, 3] with
        `valid` for the student (None: the state-only teacher)."""
        B = len(frames)
        dino = np.empty((B, *self.caches[0].dino.shape[1:]), np.float32)
        centres = np.empty((B, *self.caches[0].centres.shape[1:]), np.float32)
        ok = np.empty((B, *self.caches[0].centre_ok.shape[1:]), bool)

        def rows(c, sel, local):
            dino[sel], centres[sel], ok[sel] = c.dino[local], c.centres[local], c.centre_ok[local]
        self._gather(frames, rows)
        t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
        state = self.state[frames]
        out = {"dino": t(dino), "centres": t(centres), "centre_ok": t(ok, torch.bool),
               "state": t(norm.state_norm(state)), "state_xyz": t(arm_xyz(state))}
        if track is not None:
            out["track"] = t(norm.track_norm(track))
            out["valid"] = t(valid, torch.bool)
        return out

    def qwen_inputs(self, frames: np.ndarray, norm: Normalizer, device, track: np.ndarray,
                    valid: np.ndarray) -> dict[str, torch.Tensor]:
        """Qwen-3D student batch of `frames`: the three cameras' RGB, the head
        patch centres, the wrist Qwen token centres and the task index, plus
        `track` [B', 391, 16, 3] (m) as `track` (normalized, the track
        encoder's input) and `track_m` (m, the future tokens' positions) with
        `valid`."""
        B = len(frames)
        c0 = self.caches[0]
        rgb = np.empty((B, *c0.rgb.shape[1:]), np.uint8)
        centres = np.empty((B, *c0.centres.shape[2:]), np.float32)
        ok = np.empty((B, c0.centre_ok.shape[2]), bool)
        qc = np.empty((B, *c0.qcentres[0].shape[1:]), np.float32)
        qok = np.empty((B, *c0.qcentres[1].shape[1:]), bool)

        def rows(c, sel, local):
            rgb[sel], centres[sel], ok[sel] = c.rgb[local], c.centres[local, 0], c.centre_ok[local, 0]
            qc[sel], qok[sel] = c.qcentres[0][local], c.qcentres[1][local]
        self._gather(frames, rows)
        t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
        return {"rgb": t(rgb, torch.uint8), "centres": t(centres), "centre_ok": t(ok, torch.bool),
                "qcentres": t(qc), "qcentre_ok": t(qok, torch.bool), "task": t(self.task[frames], torch.int64),
                "track": t(norm.track_norm(track)), "track_m": t(track), "valid": t(valid, torch.bool)}


def fit_normalizer(fr: Frames, frames: np.ndarray) -> Normalizer:
    """State limits over `frames`; track RMS over up to MAX_TRACK_FRAMES of them per cache."""
    frames = np.sort(frames)
    tracks = []
    for c, lo in zip(fr.caches, fr.offsets):
        own = frames[(frames >= lo) & (frames < lo + len(c))] - lo
        if not len(own):
            continue
        for i in own[np.linspace(0, len(own) - 1, min(MAX_TRACK_FRAMES, len(own))).astype(int)]:
            if c.valid[i].any():
                tracks.append(np.asarray(c.tracks[i][c.valid[i]]))
    norm = Normalizer.fit(fr.state[frames], tracks)
    print(f"normalizer: track rms {norm.track_rms * 1000:.2f} mm")
    return norm
