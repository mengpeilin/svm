"""Paths, dataset layout and constants.

Two environment variables locate everything:

  MV_DISTILL_RAW   the collector parts (`collect_svm_rollouts.py --wrist`):
                     $MV_DISTILL_RAW/scan/scan_wdepth-*.hdf5     scan_object rollouts
                     $MV_DISTILL_RAW/branch/scan_branch-*.hdf5   their noise branches
  MV_DISTILL_WORK  everything this package writes:
                     cache/<dataset>/      memmapped inputs (build_cache, build_wrist)
                     runs/<run>/           checkpoints, labels, eval frames, videos
                     branch_plans/         noise-branch plans (branch_plan)

Datasets: "scan" (GAP scan_object rollouts, successes then failures, each
class sorted by seed) and "branch" (noise branches of the scan train
successes; train only). The last `n_eval` episodes of each scan class are
held out: they select checkpoints AND give every reported number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CHUNK = 16            # GAP executes 16 actions per decision; tracks are 16 steps
PATCH_GRID = (17, 23)
N_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]   # 391 DinoV2 patches per camera
DINO_DIM = 1024
ARM_DIM = 8           # robot_state = [L pose7, L grip, R pose7, R grip]
SUCCESS, FAILURE = 1, 0
FAILURE_CAP = 250     # failure episodes are used up to this frame
REPLAY_TOL_MM = 5.0   # a branch whose replay misses its parent's EEF by more is not used
WRIST_STRIDE = 4      # wrist rows (and student train frames): every 4th frame


@dataclass(frozen=True)
class Dataset:
    pattern: str          # part file glob under $MV_DISTILL_RAW/<name>/
    n_eval: int           # held-out episodes per class (the last ones)
    branch: bool = False  # noise branches: frames before the branch frame replay the parent


DATASETS = {
    "scan": Dataset("scan_wdepth-*.hdf5", n_eval=50),
    "branch": Dataset("scan_branch-*.hdf5", n_eval=0, branch=True),
}


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see README.md)")
    return Path(value)


def parts(dataset: str) -> list[Path]:
    found = sorted((_env("MV_DISTILL_RAW") / dataset).glob(DATASETS[dataset].pattern))
    if not found:
        raise SystemExit(f"no {DATASETS[dataset].pattern} under $MV_DISTILL_RAW/{dataset}")
    return found


def cache_dir(dataset: str) -> Path:
    d = _env("MV_DISTILL_WORK") / "cache" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run: str) -> Path:
    d = _env("MV_DISTILL_WORK") / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def plan_dir() -> Path:
    d = _env("MV_DISTILL_WORK") / "branch_plans"
    d.mkdir(parents=True, exist_ok=True)
    return d
