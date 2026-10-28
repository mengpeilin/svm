"""Token geometry of the three cameras on every frame of a staged task, from its
depth (mm) and camera intrinsics / extrinsics (world -> camera, OpenCV), with
the collector's rule: the centre of each token's pixel cell, scaled back to the
native 240 x 320 image, bilinear depth there, unprojected to world metres.
RoboTwin renders depth 0 where nothing is; those tokens are 0 and not ok.

Two products, each over every staged root (MV_DISTILL_DATA):

  centres   the 17 x 23 DinoV2 patches of all three cameras (on the 238 x 322
            resized grid; the head's 2x Qwen image has the same 17 x 23 merged
            tokens). Reproduces the collector's stored head centres
            (`image_patch_centre_3d_position`, max error 1e-7 m on scan_object):
              centres_f32.npy   f32  [N, 3, 391, 3]
              centre_ok.npy     bool [N, 3, 391]
              valid.npy         bool [N, 391]   the head patch has a GT track
                                                (track_valid[:, :, 0] = the collector's valid_point_mask)
  wrist_q   the 9 x 11 merged Qwen tokens of the two wrists at native size
            (Qwen's smart_resize of 240 x 320 is 252 x 308, 28 px per token):
              qcentres_wrist_f32.npy  f32  [N, 2, 99, 3]
              qcentre_wrist_ok.npy    bool [N, 2, 99]

in cache/<root name>/<task>/, with <product>_done.txt (finished row blocks, a
rerun resumes) and <product>.json (written once every block is done).

    python -m mv_distill.build_centres build --tasks all --workers 32
    python -m mv_distill.build_centres check --tasks all
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_HW, N_PATCHES, N_VIEWS, PATCH_GRID, cache_dir, data_dir, parse_tasks, roots

BLOCK = 256
WRIST_Q_GRID = (9, 11)                     # merged Qwen tokens of a 240 x 320 image (smart_resize: 252 x 308)
WRIST_Q_HW = (WRIST_Q_GRID[0] * 28, WRIST_Q_GRID[1] * 28)
N_WRIST_Q = WRIST_Q_GRID[0] * WRIST_Q_GRID[1]


def _cell_centres(grid: tuple[int, int], cell: int, resized_hw: tuple[int, int]) -> np.ndarray:
    """[h * w, 2] native-pixel (x, y) centres of the cells of a resized image, row-major."""
    gh, gw = grid
    gx, gy = np.meshgrid((np.arange(gw) + 0.5) * cell, (np.arange(gh) + 0.5) * cell)
    scale = np.array([IMG_HW[1] / resized_hw[1], IMG_HW[0] / resized_hw[0]])
    return (np.stack([gx.ravel(), gy.ravel()], -1) * scale).astype(np.float32)


PATCH_PX = _cell_centres(PATCH_GRID, 14, (PATCH_GRID[0] * 14, PATCH_GRID[1] * 14))   # 238 x 322, GAP's resize
WRIST_Q_PX = _cell_centres(WRIST_Q_GRID, 28, WRIST_Q_HW)


def unproject(px: np.ndarray, depth_mm: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World xyz [n, 3] at the native pixels `px` [n, 2] (bilinear depth) and validity [n]."""
    x, y = px[:, 0], px[:, 1]
    d = cv2.remap(depth_mm.astype(np.float32), x[None], y[None], cv2.INTER_LINEAR)[0] / 1000.0
    ok = (d > 0) & np.isfinite(d)
    cam = np.stack([(x - K[0, 2]) * d / K[0, 0], (y - K[1, 2]) * d / K[1, 1], d], -1)
    world = (cam - E[:, 3]) @ E[:, :3]
    world[~ok] = 0.0
    return world.astype(np.float32), ok


def _compute_centres(d: Path, a: int, b: int) -> list[np.ndarray]:
    depth = np.load(d / "depth_f32.npy", mmap_mode="r")[a:b]
    K = np.load(d / "intrinsic.npy", mmap_mode="r")[a:b]
    E = np.load(d / "extrinsic.npy", mmap_mode="r")[a:b]
    valid = np.load(d / "track_valid.npy", mmap_mode="r")[a:b, :, 0]
    centres = np.empty((b - a, N_VIEWS, N_PATCHES, 3), np.float32)
    ok = np.empty((b - a, N_VIEWS, N_PATCHES), bool)
    for i in range(b - a):
        for v in range(N_VIEWS):
            centres[i, v], ok[i, v] = unproject(PATCH_PX, depth[i, v], K[i, v], E[i, v])
    return [centres, ok, np.ascontiguousarray(valid)]


def _compute_wrist_q(d: Path, a: int, b: int) -> list[np.ndarray]:
    depth = np.load(d / "depth_f32.npy", mmap_mode="r")[a:b, 1:]
    K = np.load(d / "intrinsic.npy", mmap_mode="r")[a:b, 1:]
    E = np.load(d / "extrinsic.npy", mmap_mode="r")[a:b, 1:]
    centres = np.empty((b - a, 2, N_WRIST_Q, 3), np.float32)
    ok = np.empty((b - a, 2, N_WRIST_Q), bool)
    for i in range(b - a):
        for v in range(2):
            centres[i, v], ok[i, v] = unproject(WRIST_Q_PX, depth[i, v], K[i, v], E[i, v])
    return [centres, ok]


# product -> (compute, [(file, per-frame shape, dtype)])
PRODUCTS = {
    "centres": (_compute_centres, [("centres_f32", (N_VIEWS, N_PATCHES, 3), np.float32),
                                   ("centre_ok", (N_VIEWS, N_PATCHES), bool), ("valid", (N_PATCHES,), bool)]),
    "wrist_q": (_compute_wrist_q, [("qcentres_wrist_f32", (2, N_WRIST_Q, 3), np.float32),
                                   ("qcentre_wrist_ok", (2, N_WRIST_Q), bool)]),
}


def _write(job) -> int:
    product, d, out, a, b = job
    compute, files = PRODUCTS[product]
    for (name, _, _), arr in zip(files, compute(Path(d), a, b)):
        m = np.load(Path(out) / f"{name}.npy", mmap_mode="r+")
        m[a:b] = arr
        m.flush()
    return b - a


def build(product: str, root: Path, task: str, workers: int) -> None:
    out, d = cache_dir(root, task), data_dir(root, task)
    tag = f"{root.name}/{task} {product}"
    if (out / f"{product}.json").exists():
        print(f"{tag}: done already")
        return
    depth = np.load(d / "depth_f32.npy", mmap_mode="r")
    n = len(depth)
    assert depth.shape[1:] == (N_VIEWS, *IMG_HW), depth.shape
    for name, shape, dtype in PRODUCTS[product][1]:
        if not (out / f"{name}.npy").exists():
            np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=(n, *shape))
    done_path = out / f"{product}_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    jobs = [(product, str(d), str(out), a, min(a + BLOCK, n)) for a in range(0, n, BLOCK) if str(a) not in done]
    print(f"{tag}: {n} frames, {len(jobs)} blocks to do", flush=True)
    t0, k = time.time(), 0
    with Pool(workers) as pool, open(done_path, "a") as log:
        for i, (job, m) in enumerate(zip(jobs, pool.imap(_write, jobs, chunksize=1))):
            log.write(f"{job[3]}\n")
            log.flush()
            k += m
            if i % 100 == 0:
                print(f"  {k} frames, {k / (time.time() - t0):.0f} fps", flush=True)
    files = {name: np.load(out / f"{name}.npy", mmap_mode="r") for name, _, _ in PRODUCTS[product][1]}
    if product == "centres":
        stats = {"frames": n, "ok_frac": [float(files["centre_ok"][::97, v].mean()) for v in range(N_VIEWS)],
                 "valid_frac": float(files["valid"][::97].mean())}
    else:
        stats = {"frames": n, "ok_frac": [float(files["qcentre_wrist_ok"][::97, v].mean()) for v in range(2)]}
    (out / f"{product}.json").write_text(json.dumps(stats, indent=1))
    print(f"{tag}: {stats} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def check(product: str, root: Path, task: str, n_check: int) -> None:
    """Recompute sampled frames and compare exactly."""
    out, d = cache_dir(root, task), data_dir(root, task)
    tag = f"{root.name}/{task} {product}"
    if not (out / f"{product}.json").exists():
        raise SystemExit(f"{tag}: not built")
    compute, spec = PRODUCTS[product]
    files = [np.load(out / f"{name}.npy", mmap_mode="r") for name, _, _ in spec]
    rng = np.random.default_rng(0)
    for a in rng.choice(len(files[0]), n_check, replace=False):
        for m, ref in zip(files, compute(d, int(a), int(a) + 1)):
            assert np.array_equal(m[a:a + 1], ref), (tag, a)
    print(f"{tag}: {n_check} sampled frames match; {json.loads((out / f'{product}.json').read_text())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("build", "check"))
    p.add_argument("--tasks", default="all")
    p.add_argument("--products", default="centres,wrist_q")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-check", type=int, default=64)
    args = p.parse_args()
    for product in args.products.split(","):
        for root in roots():
            for task in parse_tasks(args.tasks):
                if args.cmd == "build":
                    build(product, root, task, args.workers)
                else:
                    check(product, root, task, args.n_check)


if __name__ == "__main__":
    main()
