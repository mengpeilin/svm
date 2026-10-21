"""Score a student's eval frames (`infer.py`) against the teacher, on frames
t >= 16 (`metrics.py`, class-balanced):

  RMSE         of the progress reward r = V(s_t, track_t) - V(s_t) vs g = T_{t+16} - T_t
  reverse acc  P(r(real track) > r(-track)) on the chunks with g > 0.5

Writes runs/<run>/metrics.json.

    python -m mv_distill.evaluate --run student
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import CHUNK, SUCCESS, run_dir
from .metrics import PROGRESS, balanced, reverse_acc, rmse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="student")
    args = p.parse_args()
    z = np.load(run_dir(args.run) / "eval_frames.npz")
    m = z["t"] >= CHUNK
    lab, g = z["label"][m], (z["T_next"] - z["T"])[m]
    r = (z["forward"] - z["static"])[m]
    res = {"rmse": rmse(r, g, lab), "reverse_acc": reverse_acc(z["forward"][m], z["mirrored"][m], g, lab),
           "balanced_std_g": rmse(np.average(g, weights=balanced(lab)), g, lab),  # RMSE of predicting the mean
           "frames": int(m.sum()), "success_frames": int((lab == SUCCESS).sum()),
           "progress_frames": int((g > PROGRESS).sum())}
    print(f"runs/{args.run}: RMSE {res['rmse']:.3f} (balanced std of g {res['balanced_std_g']:.2f}), "
          f"reverse acc {res['reverse_acc']:.3f}  [{res['frames']} frames t >= 16, "
          f"{res['progress_frames']} with g > {PROGRESS}]")
    with open(run_dir(args.run) / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
