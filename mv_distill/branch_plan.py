"""Plan the noise branches (the "branch" dataset) for the collector
(`collect_svm_rollouts.py --branch_plan`, on grogu).

Every branch re-plays one TRAIN success episode of "scan" up to a chunk c
drawn uniformly among the chunks fully inside the episode (16c + 16 <=
length), executes that chunk with a smooth per-arm xyz offset delta ~ N(0,
sigma^2 I3) (step i of the chunk gets (i+1)/16 of it), then lets the policy
continue. Branch k of a parent gets seed 10 * parent_seed + k; sigmas cycle
over the shuffled rows (equal counts per sigma).

Stored per branch: seed, parent_seed, parent_ep (scan cache episode), k,
branch_frame (16c), noise_cm (sigma), delta [2, 3] m, and the parent's
recorded actions[0 : 16c+16] and robot_state[0 : 16c+1] (flattened, with
offsets): the collector replays the former and checks the replay against the
latter. Rows are shuffled so workers taking every n-th row get a mix.

Writes branch_plans/<name>.npz.

    python -m mv_distill.branch_plan --name train --k 0,1 --sigmas 2,4,8
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from .config import CHUNK, SUCCESS, parts, plan_dir
from .data import Cache
from .parts import episode_plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--k", required=True, help="branch indices per parent, e.g. 0,1")
    p.add_argument("--sigmas", required=True, help="xyz offset sigmas in cm, e.g. 2,4,8")
    p.add_argument("--n-parents", type=int, default=None, help="random subset of the train success episodes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    sigmas = np.array([float(x) for x in args.sigmas.split(",")])
    rng = np.random.default_rng(args.seed)

    c = Cache("scan", wrist=False)
    seeds, bounds = c.meta["seed"], c.bounds
    parents = np.array(c.splits[(SUCCESS, "train")])
    if args.n_parents is not None:
        parents = np.sort(rng.choice(parents, args.n_parents, replace=False))

    # The replay source is the cache's actions: spot-check them against the parts.
    srcs = parts("scan")
    plan, _ = episode_plan(srcs)
    for ep in parents[:3]:
        k, s, e = plan[ep]
        with h5py.File(srcs[k], "r") as f:
            assert np.array_equal(f["action"][s:e], c.action[bounds[ep, 0]:bounds[ep, 1]]), ep

    rows = [(ep, k) for ep in parents for k in ks]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    sig = sigmas[np.arange(len(rows)) % len(sigmas)]
    out = {k: [] for k in ("seed", "parent_seed", "parent_ep", "k", "branch_frame", "noise_cm", "delta")}
    actions, states = [], []
    for (ep, k), sigma in zip(rows, sig):
        s, e = bounds[ep]
        f = int(rng.integers(0, (e - s) // CHUNK)) * CHUNK
        out["seed"].append(10 * seeds[ep] + k)
        out["parent_seed"].append(seeds[ep])
        out["parent_ep"].append(ep)
        out["k"].append(k)
        out["branch_frame"].append(f)
        out["noise_cm"].append(sigma)
        out["delta"].append(rng.normal(0.0, sigma / 100.0, size=(2, 3)))
        actions.append(c.action[s:s + f + CHUNK])
        states.append(c.state[s:s + f + 1])
    res = {k: np.asarray(v) for k, v in out.items()}
    res["delta"] = res["delta"].astype(np.float32)
    res["actions"] = np.concatenate(actions).astype(np.float32)
    res["action_offsets"] = np.concatenate([[0], np.cumsum([len(a) for a in actions])])
    res["parent_state"] = np.concatenate(states).astype(np.float32)
    res["state_offsets"] = np.concatenate([[0], np.cumsum([len(a) for a in states])])
    assert len(np.unique(res["seed"])) == len(rows)
    dst = plan_dir() / f"{args.name}.npz"
    np.savez(dst, **res)
    bf = res["branch_frame"]
    print(f"{len(rows)} branches of {len(parents)} parents, k {ks}, sigma {sigmas.tolist()} cm; "
          f"branch frame median {np.median(bf):.0f} (max {bf.max()}) -> {dst}")


if __name__ == "__main__":
    main()
