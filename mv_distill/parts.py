"""Reading the collector parts: episode order, per-episode metadata, and the
cameras' patch geometry.

Each part (HDF5) concatenates a few episodes along a flat frame axis
(``episode_ends``) and records, per frame, the head camera's stored DinoV2
features (``dino_features``), GT 16-step 3D point tracks (``3d_track``,
``valid_point_mask``), head patch centres (``image_patch_centre_3d_position``),
robot state / action, and the three cameras' RGB-D + intrinsics / extrinsics;
per episode ``episode_seed``, ``episode_success`` and, for noise branches,
``episode_parent_seed`` / ``episode_branch_frame`` / ``episode_noise_cm`` /
``episode_replay_err_mm``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

from .config import PATCH_GRID

IMG_HW = (240, 320)
DINO_HW = (PATCH_GRID[0] * 14, PATCH_GRID[1] * 14)   # 238 x 322, GAP's resize
# Per camera (head, left wrist, right wrist): rgb, depth [mm], intrinsic, extrinsic [world -> camera].
CAMERA_KEYS = (
    ("image", "depth", "camera_intrinsic", "camera_extrinsic"),
    *((f"{w}_image", f"{w}_depth", f"{w}_intrinsic", f"{w}_extrinsic") for w in ("left_wrist", "right_wrist")),
)


def episode_plan(parts: list[Path]) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Cache episode order -- successes sorted by seed, then failures sorted by
    seed -- as [(part index, start, end)], and the cache ``episode_ends``."""
    eps = []
    for k, p in enumerate(parts):
        with h5py.File(p, "r") as f:
            ends = f["episode_ends"][:]
            starts = np.concatenate([[0], ends[:-1]])
            for s, e, seed, ok in zip(starts, ends, f["episode_seed"][:], f["episode_success"][:]):
                eps.append((int(seed), int(ok), k, int(s), int(e)))
    seeds = [e[0] for e in eps]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate episode seeds across parts")
    eps = sorted(e for e in eps if e[1]) + sorted(e for e in eps if not e[1])
    ends = np.cumsum([e - s for *_, s, e in eps])
    return [(k, s, e) for _, _, k, s, e in eps], ends


def episode_meta(parts: list[Path], plan: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """The parts' per-episode datasets (``episode_<name>`` -> ``<name>``) in plan order."""
    rows = {}
    for k, p in enumerate(parts):
        with h5py.File(p, "r") as f:
            keys = [n for n in f if n.startswith("episode_") and n != "episode_ends"]
            starts = np.concatenate([[0], f["episode_ends"][:-1]])
            for i, s in enumerate(starts):
                rows[(k, int(s))] = {n[len("episode_"):]: f[n][i] for n in keys}
    first = rows[plan[0][:2]]
    return {n: np.array([rows[(k, s)][n] for k, s, _ in plan]) for n in first}


def _patch_pixel_centres() -> np.ndarray:
    """[391, 2] native-pixel (x, y) centres of the 17x23 DinoV2 patches (centres
    on the 238x322 resized grid, scaled back to 240x320), as the collector."""
    ph, pw = PATCH_GRID
    gx, gy = np.meshgrid((np.arange(pw) + 0.5) * 14, (np.arange(ph) + 0.5) * 14)
    scale = np.array([IMG_HW[1] / DINO_HW[1], IMG_HW[0] / DINO_HW[0]])
    return (np.stack([gx.ravel(), gy.ravel()], -1) * scale).astype(np.float32)


PATCH_PX = _patch_pixel_centres()


def patch_centres(depth_mm: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World xyz [391, 3] of each patch centre (bilinear depth at the centre,
    unprojected with intrinsic K and world -> camera extrinsic E, OpenCV
    convention) and its validity [391] (RoboTwin renders depth 0 where nothing
    is; those patches are 0 and invalid). The collector's head-camera rule,
    applied to every camera."""
    x, y = PATCH_PX[:, 0], PATCH_PX[:, 1]
    d = cv2.remap(depth_mm.astype(np.float32), x[None], y[None], cv2.INTER_LINEAR)[0] / 1000.0
    ok = (d > 0) & np.isfinite(d)
    cam = np.stack([(x - K[0, 2]) * d / K[0, 0], (y - K[1, 2]) * d / K[1, 1], d], -1)
    world = (cam - E[:, 3]) @ E[:, :3]
    world[~ok] = 0.0
    return world.astype(np.float32), ok
