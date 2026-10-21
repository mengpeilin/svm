"""Label EVERY frame of a dataset with the teacher: T_t = f(s_t), the value the
student distills. Each frame's inputs are built from the parts as the caches
build them -- head DinoV2 features as stored, wrist features by
`dino.dino_features`, every camera's patch centres from its own depth -- so the
labels do not depend on which frames have wrist rows.

Writes runs/<teacher>/labels_<dataset>.npz: logit [n_frames] (cache frame order).

    python -m mv_distill.label --teacher teacher --dataset scan
    python -m mv_distill.label --teacher teacher --dataset branch
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from .config import N_PATCHES, parts, run_dir
from .data import Normalizer, arm_xyz, episode_bounds
from .dino import dino_features, load_dino
from .model import load
from .parts import CAMERA_KEYS, episode_plan, patch_centres

BLOCK = 64


def read_block(job) -> dict:
    """Frames [a, b) of one part: stored head features, wrist RGB, the three
    cameras' patch centres and validity, robot state."""
    part, a, b = job
    with h5py.File(part, "r") as f:
        cams = [tuple(f[k][a:b] for k in keys) for keys in CAMERA_KEYS]
        centres = np.zeros((b - a, 3, N_PATCHES, 3), np.float32)
        ok = np.zeros((b - a, 3, N_PATCHES), bool)
        for i in range(b - a):
            for v, (_, depth, K, E) in enumerate(cams):
                centres[i, v], ok[i, v] = patch_centres(depth[i], K[i], E[i])
        return {"head": f["dino_features"][a:b][:, 0], "wrist": np.stack([cams[1][0], cams[2][0]], 1),
                "centres": centres, "ok": ok, "state": f["robot_state"][a:b]}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="teacher", help="runs/<teacher>/teacher.pt")
    p.add_argument("--dataset", choices=("scan", "branch"), required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = args.device
    dino = load_dino(device)
    model, ck = load(run_dir(args.teacher) / "teacher.pt", device)
    assert ck["config"]["inputs"] == "state", ck["config"]
    norm = Normalizer.from_state_dict(ck["normalizer"])

    srcs = parts(args.dataset)
    plan, ends = episode_plan(srcs)
    if not np.array_equal(ends, episode_bounds(args.dataset)[:, 1]):
        raise SystemExit(f"parts do not reproduce the {args.dataset} cache's episodes")
    jobs, dst = [], []
    for (k, s, e), d in zip(plan, np.concatenate([[0], ends[:-1]])):
        for a in range(s, e, BLOCK):
            jobs.append((srcs[k], a, min(a + BLOCK, e)))
            dst.append(int(d) + a - s)
    n = int(ends[-1])
    logit = np.full(n, np.nan, dtype=np.float32)
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (blk, d) in enumerate(zip(pool.imap(read_block, jobs, chunksize=1), dst)):
            m = len(blk["state"])
            wrist = [dino_features(dino, blk["wrist"][:, v], device) for v in (0, 1)]
            batch = {"dino": torch.stack([t(blk["head"]), *wrist], 1), "centres": t(blk["centres"]),
                     "centre_ok": t(blk["ok"], torch.bool), "state": t(norm.state_norm(blk["state"])),
                     "state_xyz": t(arm_xyz(blk["state"]))}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit[d:d + m] = model(batch).float().cpu().numpy()
            if i % 200 == 0:
                print(f"  {d + m}/{n} frames ({(d + m) / (time.time() - t0):.0f} fps)", flush=True)
    assert np.isfinite(logit).all()
    out = run_dir(args.teacher) / f"labels_{args.dataset}.npz"
    np.savez(out, logit=logit)
    print(f"{n} frames, mean logit {logit.mean():+.2f} -> {out}")


if __name__ == "__main__":
    main()
