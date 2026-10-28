"""Score a student's eval frames (`infer.py`) against the task teachers, on
frames t >= 16 (`metrics.py`, class-balanced), pooled over all tasks and per
task:

  RMSE         of the forecast V(s_t, track_t) vs the teacher's T_{t+16}
  reverse acc  P(V(s_t, track_t) > V(s_t, -track_t)) on the chunks with g = T_{t+16} - T_t > 0.5

Each task's values are in its own teacher's logit units; the references
(predicting the mean of T_{t+16}, holding the teacher's T_t) give each task's
scale. Writes runs/<run>/metrics.json.

    python -m mv_distill.evaluate --run student
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import CHUNK, SUCCESS, run_dir
from .metrics import PROGRESS, balanced, reverse_acc, rmse


def scores(t_now, t_next, forward, mirrored, lab) -> dict:
    g = t_next - t_now
    return {"rmse": rmse(forward, t_next, lab), "reverse_acc": reverse_acc(forward, mirrored, g, lab),
            # references: predicting the mean of T_{t+16}, and holding the teacher's T_t (no change)
            "balanced_std_T_next": rmse(np.average(t_next, weights=balanced(lab)), t_next, lab),
            "rmse_hold_T": rmse(t_now, t_next, lab),
            "frames": int(len(g)), "success_frames": int((lab == SUCCESS).sum()),
            "progress_frames": int((g > PROGRESS).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="student")
    args = p.parse_args()
    z = np.load(run_dir(args.run) / "eval_frames.npz")
    m = z["t"] >= CHUNK
    cols = [z[k][m] for k in ("T", "T_next", "forward", "mirrored", "label")]
    task = z["task"][m]
    res = {"pooled": scores(*cols), "tasks": {str(name): scores(*(c[task == k] for c in cols))
                                              for k, name in enumerate(z["tasks"])}}
    print(f"runs/{args.run}  (frames t >= 16, class-balanced)\n\n"
          f"| task | RMSE | std T_t+16 | holding T_t | reverse acc | frames | g > {PROGRESS} |\n|---|---|---|---|---|---|---|")
    for name, s in [*res["tasks"].items(), ("pooled", res["pooled"])]:
        print(f"| {name} | {s['rmse']:.3f} | {s['balanced_std_T_next']:.2f} | {s['rmse_hold_T']:.2f} | "
              f"{s['reverse_acc']:.3f} | {s['frames']} | {s['progress_frames']} |")
    with open(run_dir(args.run) / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
