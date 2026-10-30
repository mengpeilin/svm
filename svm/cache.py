"""Build the training cache from RoboTwin collector parts (HDF5 with head +
left/right wrist RGB-D, per-frame camera K/E, GT tracks and commanded actions).

Training samples decision steps at random, and the parts are gzip-compressed
HDF5, so everything the model reads is first copied into uncompressed memmaps
holding DECISION frames only (every CHUNK-th frame of an episode):

  geometry  (CPU, one worker per part) head DINOv2 features and GT tracks as
            the collector stored them, the three RGB images, and every
            camera's patch centres unprojected from its own depth
            (`preprocess.patch_centres`). Checks that the head centres
            reproduce the collector's `image_patch_centre_3d_position`, and
            how often wrist centres agree with the head depth map.
  dino      (GPU) DINOv2 features of the two wrist images; first re-extracts a
            sample of head images and checks them against the stored features.

Cache layout (`--cache DIR`):

  vectors.npz      per FRAME: robot_state, action, label (1 = success);
                   per episode: episode_ends, episode_seed; per decision row:
                   rows (its frame index)
  dino_f16.npy     [R, 3, 391, 1024]  head, left wrist, right wrist
  centres.npy      [R, 3, 391, 3]     patch centre world xyz (m)
  centre_ok.npy    [R, 3, 391]        depth-valid patch centre
  tracks_f16.npy   [R, 391, 16, 3]    head-patch GT tracks (m)
  track_valid.npy  [R, 391]
  rgb_u8.npy       [R, 3, 240, 320, 3]  only read by the dino stage; delete after

Episode order: successes sorted by seed, then failures sorted by seed (the
split in `svm.data` holds out the highest seeds of each class).

    python -m svm.cache --cache cache/scan_object geometry --parts parts/scan_wdepth-*.hdf5
    python -m svm.cache --cache cache/scan_object dino
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

from .preprocess import CHUNK, IMG_HW, N_PATCHES, dino_features, load_dino, patch_centres

DINO_DIM = 1024
N_VIEWS = 3  # 0 head, 1 left wrist, 2 right wrist
WRIST = ("left_wrist", "right_wrist")

# Per view: (rgb, depth [mm], intrinsic, extrinsic [world -> camera]) datasets of a part.
CAMERA_KEYS = (
    ("image", "depth", "camera_intrinsic", "camera_extrinsic"),
    *((f"{w}_image", f"{w}_depth", f"{w}_intrinsic", f"{w}_extrinsic") for w in WRIST),
)
VEC_KEYS = {"robot_state": "robot_state", "action": "action", "label": "success_cond_uniform"}

FILES = {
    "dino": ("dino_f16.npy", np.float16, (N_VIEWS, N_PATCHES, DINO_DIM)),
    "tracks": ("tracks_f16.npy", np.float16, (N_PATCHES, CHUNK, 3)),
    "track_valid": ("track_valid.npy", np.bool_, (N_PATCHES,)),
    "centres": ("centres.npy", np.float32, (N_VIEWS, N_PATCHES, 3)),
    "centre_ok": ("centre_ok.npy", np.bool_, (N_VIEWS, N_PATCHES)),
    "rgb": ("rgb_u8.npy", np.uint8, (N_VIEWS, *IMG_HW, 3)),
}


def open_out(cache: Path, name: str, mode: str, n: int | None = None):
    fname, dtype, shape = FILES[name]
    if mode == "w+":
        return np.lib.format.open_memmap(cache / fname, mode="w+", dtype=dtype, shape=(n, *shape))
    return np.load(cache / fname, mmap_mode=mode)


# ------------------------------------------------------------------ geometry


def head_agreement(world, ok, head_depth_mm, K, E, tol=0.01):
    """Cross-camera check of a wrist camera's extrinsic: of its grounded patches
    that land inside the head image, the fraction whose head-camera depth agrees
    with the head depth map there to `tol` m. Occluded points disagree, so this
    is a visibility-limited rate, not an error rate; a convention bug drives it
    to ~0."""
    cam = world[ok] @ E[:, :3].T + E[:, 3]
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    inside = (z > 0) & (u >= 0) & (u < IMG_HW[1] - 1) & (v >= 0) & (v < IMG_HW[0] - 1)
    if not inside.any():
        return 0, 0
    d = head_depth_mm[v[inside].astype(int), u[inside].astype(int)] / 1000.0
    return int(inside.sum()), int((np.abs(d - z[inside]) < tol).sum())


def episode_plan(parts: list[Path]):
    """[(part index, src start, src end, seed, success)] in cache order."""
    eps = []
    for k, p in enumerate(parts):
        with h5py.File(p, "r") as f:
            ends = f["episode_ends"][:]
            starts = np.concatenate([[0], ends[:-1]])
            for s, e, seed, ok in zip(starts, ends, f["episode_seed"][:], f["episode_success"][:]):
                eps.append((k, int(s), int(e), int(seed), int(ok)))
    seeds = [e[3] for e in eps]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate episode seeds across parts")
    return sorted((e for e in eps if e[4]), key=lambda e: e[3]) + sorted(
        (e for e in eps if not e[4]), key=lambda e: e[3]
    )


def geometry_worker(job):
    """Copy one part's episodes: per-frame vectors, then per decision frame the
    head fields, the three RGB images and the three cameras' patch centres."""
    cache, part, eps, check_every = job
    out = {k: open_out(cache, k, "r+") for k in FILES}
    vec = {k: np.load(cache / f"_{k}.npy", mmap_mode="r+") for k in VEC_KEYS}
    stats = {"rows": 0, "head_conv_err": 0.0, "agree": np.zeros((2, 2), np.int64)}
    t0 = time.time()
    with h5py.File(part, "r") as f:
        for s, e, dst_frame, dst_row in eps:
            for k, h in VEC_KEYS.items():
                vec[k][dst_frame : dst_frame + e - s] = f[h][s:e]
            src = np.arange(s, e, CHUNK)
            rows = slice(dst_row, dst_row + len(src))
            out["dino"][rows, 0] = f["dino_features"][src][:, 0]
            out["tracks"][rows] = f["3d_track"][src].astype(np.float16)
            out["track_valid"][rows] = f["valid_point_mask"][src]
            stored = f["image_patch_centre_3d_position"][src]
            cams = [tuple(f[k][src] for k in keys) for keys in CAMERA_KEYS]
            for i in range(len(src)):
                r = dst_row + i
                for v, (img, depth, K, E) in enumerate(cams):
                    out["rgb"][r, v] = img[i]
                    out["centres"][r, v], out["centre_ok"][r, v] = patch_centres(depth[i], K[i], E[i])
                ok0 = out["centre_ok"][r, 0]
                stats["head_conv_err"] = max(
                    stats["head_conv_err"], float(np.abs(out["centres"][r, 0][ok0] - stored[i][ok0]).max(initial=0.0))
                )
                if r % check_every == 0:
                    for v in (1, 2):
                        stats["agree"][v - 1] += head_agreement(
                            out["centres"][r, v], out["centre_ok"][r, v], cams[0][1][i], cams[0][2][i], cams[0][3][i]
                        )
            stats["rows"] += len(src)
    for m in (*out.values(), *vec.values()):
        m.flush()
    print(f"  {part.name}: {stats['rows']} rows ({time.time() - t0:.0f}s)", flush=True)
    return stats


def build_geometry(cache: Path, parts: list[Path], workers: int, check_every: int) -> None:
    if (cache / "vectors.npz").exists():
        raise SystemExit(f"{cache / 'vectors.npz'} exists")
    cache.mkdir(parents=True, exist_ok=True)
    parts = sorted(parts)
    eps = episode_plan(parts)
    n_succ = sum(e[4] for e in eps)
    ends = np.cumsum([e[2] - e[1] for e in eps])
    starts = np.concatenate([[0], ends[:-1]])
    n_dec = np.array([-(-(e[2] - e[1]) // CHUNK) for e in eps])
    row_starts = np.concatenate([[0], np.cumsum(n_dec)[:-1]])
    rows = np.concatenate([np.arange(s, s + (e[2] - e[1]), CHUNK) for s, e in zip(starts, eps)])
    total, R = int(ends[-1]), int(n_dec.sum())
    print(f"{len(eps)} episodes ({n_succ} success), {total} frames, {R} decision rows from {len(parts)} parts")

    for k in FILES:
        open_out(cache, k, "w+", R)
    with h5py.File(parts[0], "r") as f:
        for k, h in VEC_KEYS.items():
            np.lib.format.open_memmap(cache / f"_{k}.npy", mode="w+", dtype=f[h].dtype, shape=(total, *f[h].shape[1:]))

    jobs = []
    for k, p in enumerate(parts):
        mine = [(e[1], e[2], int(starts[i]), int(row_starts[i])) for i, e in enumerate(eps) if e[0] == k]
        jobs.append((cache, p, mine, check_every))
    with Pool(workers) as pool:
        all_stats = pool.map(geometry_worker, jobs, chunksize=1)

    vec = {k: np.load(cache / f"_{k}.npy") for k in VEC_KEYS}
    lab = vec["label"][starts]
    if not (lab[:n_succ].all() and not lab[n_succ:].any()):
        raise SystemExit("per-frame labels disagree with episode_success")
    np.savez(cache / "vectors.npz", episode_ends=ends.astype(np.int64), rows=rows,
             episode_seed=np.array([e[3] for e in eps]), **vec)
    for k in VEC_KEYS:
        (cache / f"_{k}.npy").unlink()

    ok = open_out(cache, "centre_ok", "r")
    agree = sum(s["agree"] for s in all_stats)
    stats = {
        "coverage": {v: float(ok[:, i].mean()) for i, v in enumerate(("head", *WRIST))},
        "head_convention_max_err_m": max(s["head_conv_err"] for s in all_stats),
        "wrist_head_depth_agreement": {w: float(agree[i, 1] / max(agree[i, 0], 1)) for i, w in enumerate(WRIST)},
    }
    (cache / "geometry_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if stats["head_convention_max_err_m"] > 1e-4:
        raise SystemExit("head patch centres do not reproduce the collector's")


# ---------------------------------------------------------------------- dino


def build_dino(cache: Path, device: str, batch: int, n_check: int) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rgb = open_out(cache, "rgb", "r")
    dino = open_out(cache, "dino", "r+")
    R = len(rgb)
    model = load_dino(device)

    sample = np.linspace(0, R - 1, n_check).astype(int)
    got = dino_features(model, rgb[sample, 0], device)
    ref = torch.from_numpy(np.asarray(dino[sample, 0], dtype=np.float32)).to(device)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    print(f"head re-extraction vs stored: cos-sim min {cos.min():.5f} mean {cos.mean():.5f}")
    if cos.min() < 0.999:
        raise SystemExit("re-extracted head features disagree with the stored ones")

    t0 = time.time()
    for s in range(0, R, batch):
        e = min(s + batch, R)
        for v in (1, 2):
            dino[s:e, v] = dino_features(model, rgb[s:e, v], device).cpu().numpy()
        if (s // batch) % 20 == 0:
            print(f"  dino {e}/{R} ({e / (time.time() - t0):.0f} rows/s)", flush=True)
    dino.flush()
    (cache / "dino_done.json").write_text(json.dumps({"head_check_cos_min": float(cos.min()), "rows": R}))
    print(f"wrote wrist features into {cache / FILES['dino'][0]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="cache directory")
    sub = p.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--parts", nargs="+", type=Path, required=True)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--check-every", type=int, default=50, help="wrist/head depth cross-check every N rows")
    d = sub.add_parser("dino")
    d.add_argument("--device", default="cuda")
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--n-check", type=int, default=256)
    args = p.parse_args()
    if args.stage == "geometry":
        build_geometry(args.cache, args.parts, args.workers, args.check_every)
    else:
        build_dino(args.cache, args.device, args.batch, args.n_check)


if __name__ == "__main__":
    main()
