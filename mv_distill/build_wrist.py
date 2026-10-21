"""Wrist-camera rows of a dataset's cache: per selected frame, both wrist
cameras' DinoV2 features (`dino.dino_features`) and patch centres
(`parts.patch_centres`), from the parts' wrist RGB-D.

A row is 1.6 MB, so not every frame is kept: every WRIST_STRIDE-th frame of
each episode (t % 4 == 0 from the episode start: all teacher decision frames
and all student train frames) plus EVERY frame of the eval episodes (the
student is scored on every frame).

  init   create the files (cache/<dataset>/wrist_*.npy, see `data.py`)
  fill   write shard k of n of the rows (one process per GPU); first
         re-extracts a sample of HEAD frames and requires them to match the
         features the collector stored (same DinoV2 weights / preprocessing)
  check  every shard finished and every row written

    python -m mv_distill.build_wrist init --dataset scan
    python -m mv_distill.build_wrist fill --dataset scan --shard 0 --shards 2 --device cuda:0
    python -m mv_distill.build_wrist fill --dataset scan --shard 1 --shards 2 --device cuda:1
    python -m mv_distill.build_wrist check --dataset scan --shards 2
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from .config import DINO_DIM, FAILURE, N_PATCHES, SUCCESS, WRIST_STRIDE, cache_dir, parts
from .data import WRIST_FILES, Cache, episode_bounds
from .dino import dino_features, load_dino
from .parts import CAMERA_KEYS, episode_plan, patch_centres

BLOCK = 64
SHAPES = {"dino": (np.float16, (2, N_PATCHES, DINO_DIM)), "centres": (np.float32, (2, N_PATCHES, 3)),
          "ok": (np.bool_, (2, N_PATCHES))}


def selected_frames(dataset: str) -> np.ndarray:
    """Every WRIST_STRIDE-th frame of each episode and every frame of the eval episodes."""
    c = Cache(dataset, wrist=False)
    b = c.bounds
    t_in = np.arange(len(c)) - np.repeat(b[:, 0], b[:, 1] - b[:, 0])
    sel = t_in % WRIST_STRIDE == 0
    for label in (SUCCESS, FAILURE):
        for e in c.splits[(label, "eval")]:
            sel[b[e, 0]:b[e, 1]] = True
    return np.flatnonzero(sel)


def plan_jobs(dataset: str, frames: np.ndarray) -> list[tuple]:
    """(part, src_start, src_end, part frames to keep, their rows) per BLOCK of an episode."""
    srcs = parts(dataset)
    plan, ends = episode_plan(srcs)
    if not np.array_equal(ends, episode_bounds(dataset)[:, 1]):
        raise SystemExit(f"parts do not reproduce the {dataset} cache's episodes")
    jobs = []
    for (k, s, e), d in zip(plan, np.concatenate([[0], ends[:-1]])):
        for a in range(s, e, BLOCK):
            b = min(a + BLOCK, e)
            lo, hi = np.searchsorted(frames, [d + a - s, d + b - s])
            if hi > lo:
                jobs.append((srcs[k], a, b, frames[lo:hi] - d + s, np.arange(lo, hi)))
    return jobs


def read_block(job) -> dict:
    part, a, b, keep, rows = job
    with h5py.File(part, "r") as f:
        cams = [tuple(f[k][a:b] for k in keys) for keys in CAMERA_KEYS[1:]]
    i = keep - a
    centres = np.zeros((len(i), 2, N_PATCHES, 3), np.float32)
    ok = np.zeros((len(i), 2, N_PATCHES), bool)
    for j, fi in enumerate(i):
        for v, (_, depth, K, E) in enumerate(cams):
            centres[j, v], ok[j, v] = patch_centres(depth[fi], K[fi], E[fi])
    return {"rgb": np.stack([cams[0][0][i], cams[1][0][i]], 1), "centres": centres, "ok": ok, "rows": rows}


def head_check(dataset: str, model, device, n: int = 256) -> float:
    """Min cosine similarity of re-extracted vs stored head features on n frames."""
    srcs = parts(dataset)
    with h5py.File(srcs[0], "r") as f:
        idx = np.linspace(0, len(f["robot_state"]) - 1, min(n, len(f["robot_state"]))).astype(int)
        rgb, ref = f["image"][idx], f["dino_features"][idx][:, 0]
    got = dino_features(model, rgb, device)
    cos = torch.nn.functional.cosine_similarity(got, torch.as_tensor(ref, dtype=torch.float32, device=device), dim=-1)
    return float(cos.min())


def init(args) -> None:
    d = cache_dir(args.dataset)
    if (d / WRIST_FILES["frames"]).exists():
        raise SystemExit(f"{d / WRIST_FILES['frames']} exists")
    frames = selected_frames(args.dataset)
    for name, (dtype, shape) in SHAPES.items():
        np.lib.format.open_memmap(d / WRIST_FILES[name], mode="w+", dtype=dtype, shape=(len(frames), *shape))
    np.save(d / WRIST_FILES["frames"], frames)
    print(f"{args.dataset}: {len(frames)} wrist rows, "
          f"{len(frames) * 2 * N_PATCHES * DINO_DIM * 2 / 2**30:.0f} GiB of features")


@torch.no_grad()
def fill(args) -> None:
    d = cache_dir(args.dataset)
    model = load_dino(args.device)
    cos = head_check(args.dataset, model, args.device)
    print(f"head re-extraction vs stored features: min cos {cos:.5f}")
    if cos < 0.999:
        raise SystemExit("DinoV2 features disagree with the collector's (weights / preprocessing)")
    frames = np.load(d / WRIST_FILES["frames"])
    jobs = plan_jobs(args.dataset, frames)[args.shard::args.shards]
    out = {k: np.load(d / WRIST_FILES[k], mmap_mode="r+") for k in SHAPES}
    n, done, t0 = sum(len(j[4]) for j in jobs), 0, time.time()
    with Pool(args.workers) as pool:
        for i, blk in enumerate(pool.imap(read_block, jobs, chunksize=1)):
            r = blk["rows"]
            out["dino"][r] = np.stack([dino_features(model, blk["rgb"][:, v], args.device).cpu().numpy()
                                       for v in (0, 1)], 1).astype(np.float16)
            out["centres"][r], out["ok"][r] = blk["centres"], blk["ok"]
            done += len(r)
            if i % 100 == 0:
                print(f"  shard {args.shard}: {done}/{n} rows ({done / (time.time() - t0):.0f} rows/s)", flush=True)
    for m in out.values():
        m.flush()
    with open(d / f"wrist_done_{args.shard}.json", "w") as f:
        json.dump({"rows": done, "shards": args.shards, "head_check_cos_min": cos, "seconds": time.time() - t0}, f)
    print(f"shard {args.shard}/{args.shards}: {done} rows")


def check(args) -> None:
    d = cache_dir(args.dataset)
    missing = [k for k in range(args.shards) if not (d / f"wrist_done_{k}.json").exists()]
    if missing:
        raise SystemExit(f"unfinished shards: {missing}")
    rows = sum(json.loads((d / f"wrist_done_{k}.json").read_text())["rows"] for k in range(args.shards))
    frames = np.load(d / WRIST_FILES["frames"])
    dino = np.load(d / WRIST_FILES["dino"], mmap_mode="r")
    ok = np.load(d / WRIST_FILES["ok"], mmap_mode="r")
    sample = np.linspace(0, len(frames) - 1, min(2000, len(frames))).astype(int)
    blank = int((np.abs(dino[sample].astype(np.float32)).sum((-1, -2)) == 0).any(-1).sum())
    stats = {"rows": len(frames), "rows_written": rows, "blank_rows_in_sample": blank,
             "wrist_coverage": [float(ok[::50, v].mean()) for v in (0, 1)]}
    print(json.dumps(stats, indent=2))
    if rows != len(frames) or blank:
        raise SystemExit("wrist rows incomplete")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("scan", "branch"), required=True)
    sub.add_parser("init", parents=[common])
    q = sub.add_parser("fill", parents=[common])
    q.add_argument("--shard", type=int, required=True)
    q.add_argument("--shards", type=int, required=True)
    q.add_argument("--workers", type=int, default=12)
    q.add_argument("--device", default="cuda:0")
    q = sub.add_parser("check", parents=[common])
    q.add_argument("--shards", type=int, required=True)
    args = p.parse_args()
    {"init": init, "fill": fill, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
