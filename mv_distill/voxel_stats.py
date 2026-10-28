"""Choose the Qwen-3D student's voxel size: on sampled frames of every task,
pool the 391 head + 2 x 99 wrist tokens with a valid centre (`qwen_model.py`'s
observation tokens) by world xyz at each candidate size and report

  tokens      voxels per frame (the observation's sequence length)
  cross-view  share of voxels holding tokens of more than one camera
  mixed       share of the tracked head patches (GT track: objects / robot)
              whose voxel also holds an untracked head token

Reads the cached head centres; the wrist Qwen token centres are recomputed
from depth (`build_centres.wrist_q`), so it runs before that product is built.

    python -m mv_distill.voxel_stats --frames 200 --sizes 0.005,0.01,0.015,0.02,0.03,0.05
"""

from __future__ import annotations

import argparse

import numpy as np

from .build_centres import _compute_wrist_q
from .config import TASKS, cache_dir, data_dir, roots


def pool(xyz: np.ndarray, size: float) -> np.ndarray:
    """Voxel index of each point."""
    _, inv = np.unique(np.floor(xyz / size).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=200, help="per task")
    p.add_argument("--sizes", default="0.005,0.01,0.015,0.02,0.03,0.05")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sizes = [float(s) for s in args.sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    root = roots()[0]
    stats = {s: {"tokens": [], "cross": [], "mixed": []} for s in sizes}
    n_valid = []
    for task in TASKS:
        c, d = cache_dir(root, task), data_dir(root, task)
        centres = np.load(c / "centres_f32.npy", mmap_mode="r")
        ok = np.load(c / "centre_ok.npy", mmap_mode="r")
        valid = np.load(c / "valid.npy", mmap_mode="r")
        for i in np.sort(rng.choice(len(centres), args.frames, replace=False)):
            qc, qok = _compute_wrist_q(d, int(i), int(i) + 1)
            xyz = np.concatenate([centres[i, 0], qc[0].reshape(-1, 3)])
            keep = np.concatenate([ok[i, 0], qok[0].ravel()])
            cam = np.concatenate([np.zeros(391, int), np.repeat([1, 2], 99)])[keep]
            tracked = np.concatenate([valid[i] & ok[i, 0], np.zeros(198, bool)])[keep]
            head = cam == 0
            n_valid.append(int(tracked.sum()))
            for s in sizes:
                v = pool(xyz[keep], s)
                nv = v.max() + 1
                cams = np.zeros((nv, 3), bool)
                cams[v, cam] = True
                untracked_head = np.zeros(nv, bool)
                untracked_head[v[head & ~tracked]] = True
                st = stats[s]
                st["tokens"].append(nv)
                st["cross"].append(float((cams.sum(1) > 1).mean()))
                if tracked.any():
                    st["mixed"].append(float(untracked_head[v[tracked]].mean()))
        print(f"  {task} done", flush=True)
    print(f"\n{len(n_valid)} frames; tokens before pooling: 391 head + 198 wrist (valid centres only); "
          f"tracked head patches per frame: mean {np.mean(n_valid):.0f}\n")
    print("| voxel (cm) | tokens mean | p50 | p90 | max | cross-view voxels | tracked patches mixed with untracked |")
    print("|---|---|---|---|---|---|---|")
    for s in sizes:
        t = np.array(stats[s]["tokens"])
        print(f"| {s * 100:g} | {t.mean():.0f} | {np.median(t):.0f} | {np.percentile(t, 90):.0f} | {t.max()} | "
              f"{np.mean(stats[s]['cross']):.3f} | {np.mean(stats[s]['mixed']):.3f} |")


if __name__ == "__main__":
    main()
