"""Paths, tasks and constants.

Two environment variables locate everything:

  MV_DISTILL_DATA  colon-separated roots of staged SVM rollouts (`fm_stage.py`,
                   read only), e.g. /scratch/pmeng/svm_fm:/scratch/pmeng/svm_fm_rest.
                   Each root holds <task>/ with 3-view RGB-D, tracks, DinoV2
                   features. The FIRST root is the base staging (250 success +
                   250 failure episodes per task): the eval episodes come from
                   it. Every episode of the later roots (`fm_stage.py plan
                   --rest`: the remaining unique-seed rollouts) is train.
  MV_DISTILL_WORK  everything this package writes:
                     cache/<root name>/<task>/   patch centres (build_centres)
                     runs/<run>/                 checkpoints, labels, eval frames, videos

Per task and root, episodes are successes then failures, each class sorted by
seed; failures are already truncated to the task's p90 success length. The
last `N_EVAL` base episodes of each class are held out: they select
checkpoints AND give every reported number.
"""

from __future__ import annotations

import os
from pathlib import Path

CHUNK = 16            # GAP executes 16 actions per decision; tracks are 16 steps
PATCH_GRID = (17, 23)
N_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]   # 391 DinoV2 patches per camera
DINO_DIM = 1024
N_VIEWS = 3           # head, left wrist, right wrist
ARM_DIM = 8           # robot_state = [L pose7, L grip, R pose7, R grip]
SUCCESS, FAILURE = 1, 0
N_EVAL = 25           # held-out base episodes per class and task (the highest seeds)
IMG_HW = (240, 320)   # staged camera resolution
# Qwen-3D student token filters (world metres). RoboTwin's table: top 1.2 x 0.7 m centred at the
# origin (x +-0.6, y +-0.35), at z 0.74 (the wall's face is the plane y = 0.4). Only tokens above the
# tabletop are kept: everything beyond its x / y edges (wall, floor, robot base) or below it is
# dropped. A tracked patch counts as moved once its displacement exceeds MOVE_THRESH (resting tracked
# points carry < 0.3 mm of sim jitter).
WORKSPACE = ((-0.6, 0.6), (-0.35, 0.35), (0.73, 1.5))
MOVE_THRESH = 0.001

# RoboTwin's task_instruction/<task>.json "full_description" (verbatim)
TASK_TEXT = {
    "handover_block": "use the left arm to grasp the red block on the table, handover it to the right arm "
                      "and place it on the blue pad",
    "hanging_mug": "Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle "
                   "of the table, use the right arm to pick the mug and hang it onto the rack.",
    "pick_diverse_bottles": "pick up one bottle with one arm, and pick up another bottle with the other arm",
    "pick_dual_bottles": "pick up one bottle with one arm, and pick up another bottle with the other arm",
    "place_burger_fries": "Use dual arm to pick the hamburg and frenchfries and put them onto the tray.",
    "place_cans_plasticbox": "Use dual arm to pick and place cans into plasticbox",
    "place_object_basket": "use one arm to grab the target object and put it in the basket, then use the other "
                           "arm to grab the basket, and finally move the basket slightly away",
    "stack_blocks_two": "there are two blocks on the table, the color of the blocks is <red, green>, <move the "
                        "blocks to the center of the table>, and <stack the geen block on the red block>",
    "stack_bowls_three": "stack the three bowls on top of each other",
    "stack_bowls_two": "stack the two bowls on top of each other",
}
TASKS = tuple(TASK_TEXT)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see README.md)")
    return value


def roots() -> tuple[Path, ...]:
    """The staged roots, base first."""
    return tuple(Path(r) for r in _env("MV_DISTILL_DATA").split(":"))


def data_dir(root: Path, task: str) -> Path:
    d = root / task
    if not (d / "episodes.npz").exists():
        raise SystemExit(f"no staged task at {d}")
    return d


def cache_dir(root: Path, task: str) -> Path:
    d = Path(_env("MV_DISTILL_WORK")) / "cache" / root.name / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run: str) -> Path:
    d = Path(_env("MV_DISTILL_WORK")) / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_tasks(s: str) -> tuple[str, ...]:
    """'all' or a comma-separated subset of TASKS."""
    tasks = TASKS if s == "all" else tuple(s.split(","))
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown tasks {sorted(unknown)}")
    return tasks
