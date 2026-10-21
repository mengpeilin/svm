"""Build a dataset's every-frame head cache from its collector parts
(cache/<dataset>/: dino_f16.npy, tracks_f16.npy, vectors.npz, episodes.npz;
see `data.py`). The parts are gzip HDF5 chunked for sequential writes, so a
random frame is expensive to read; training reads memmaps instead. One process
per part writes its episodes into a disjoint frame range.

    python -m mv_distill.build_cache --dataset scan
    python -m mv_distill.build_cache --dataset branch
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import h5py
import numpy as np

from .config import CHUNK, DINO_DIM, N_PATCHES, cache_dir, parts
from .parts import episode_meta, episode_plan

BLOCK = 512
VEC_KEYS = {"robot_state": "robot_state", "action": "action", "patch_centres": "image_patch_centre_3d_position",
            "valid_point_mask": "valid_point_mask", "label": "success_cond_uniform"}


def copy_part(job) -> int:
    out, part, eps = job
    dino = np.load(out / "dino_f16.npy", mmap_mode="r+")
    tracks = np.load(out / "tracks_f16.npy", mmap_mode="r+")
    vec = {k: np.load(out / f"_{k}.npy", mmap_mode="r+") for k in VEC_KEYS}
    t0, n = time.time(), 0
    with h5py.File(part, "r") as f:
        for s, e, d in eps:
            for a in range(s, e, BLOCK):
                b, o = min(a + BLOCK, e), d + a - s
                dino[o:o + b - a] = f["dino_features"][a:b][:, 0]
                tracks[o:o + b - a] = f["3d_track"][a:b].astype(np.float16)
                for k, h in VEC_KEYS.items():
                    vec[k][o:o + b - a] = f[h][a:b]
            n += e - s
    for m in (dino, tracks, *vec.values()):
        m.flush()
    print(f"  {part.name}: {n} frames ({n / (time.time() - t0):.0f} fps)", flush=True)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("scan", "branch"), required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    out, srcs = cache_dir(args.dataset), parts(args.dataset)
    if (out / "vectors.npz").exists():
        raise SystemExit(f"{out / 'vectors.npz'} exists")
    plan, ends = episode_plan(srcs)
    starts = np.concatenate([[0], ends[:-1]])
    total = int(ends[-1])
    print(f"{args.dataset}: {len(plan)} episodes, {total} frames from {len(srcs)} parts")

    np.lib.format.open_memmap(out / "dino_f16.npy", mode="w+", dtype=np.float16, shape=(total, N_PATCHES, DINO_DIM))
    np.lib.format.open_memmap(out / "tracks_f16.npy", mode="w+", dtype=np.float16, shape=(total, N_PATCHES, CHUNK, 3))
    with h5py.File(srcs[0], "r") as f:
        for k, h in VEC_KEYS.items():
            np.lib.format.open_memmap(out / f"_{k}.npy", mode="w+", dtype=f[h].dtype, shape=(total, *f[h].shape[1:]))
    jobs = [(out, part, [(s, e, int(d)) for (k, s, e), d in zip(plan, starts) if k == i])
            for i, part in enumerate(srcs)]
    with Pool(args.workers) as pool:
        assert sum(pool.map(copy_part, jobs, chunksize=1)) == total

    vec = {k: np.load(out / f"_{k}.npy") for k in VEC_KEYS}
    np.savez(out / "vectors.npz", episode_ends=ends.astype(np.int64), **vec)
    np.savez(out / "episodes.npz", **episode_meta(srcs, plan))
    for k in VEC_KEYS:
        (out / f"_{k}.npy").unlink()
    lab = vec["label"][starts]
    n = int(lab.sum())
    assert lab[:n].all() and not lab[n:].any(), "successes must come first"
    print(f"wrote {out}: {n} success + {len(plan) - n} failure episodes, {total} frames")


if __name__ == "__main__":
    main()
