"""Run the student on every eval sample frame (each eval episode's frames with a
full chunk ahead, failures capped at 250) with the zero, the real and the
mirrored (-track) track.

Writes runs/<run>/eval_frames.npz (sorted cache frames of the scan dataset):

  frames, episode, t, label    cache frame, its episode, frame in episode, 1 / 0
  T, T_next                    teacher logit T_t and T_{t+16}
  static, forward, mirrored    V(s_t, 0), V(s_t, track_t), V(s_t, -track_t)

so the progress reward is r = forward - static (GT g = T_next - T) and the
forecast of the value 16 frames ahead is forward (GT T_next).

    python -m mv_distill.infer --run student
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import run_dir
from .data import Normalizer
from .model import load
from .student import TRACKS, StudentData


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="student")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    model, ck = load(run_dir(args.run) / "student.pt", args.device)
    norm = Normalizer.from_state_dict(ck["normalizer"])
    print(f"runs/{args.run}: epoch {ck['epoch']}, val loss {ck['metrics']['val_loss']:.4f}, teacher {ck['teacher']}")
    sd = StudentData(ck["teacher"], datasets=("scan",))
    frames = sd.eval_frames()
    episode = np.searchsorted(sd.fr.caches[0].bounds[:, 0], frames, side="right") - 1
    out = {"frames": frames, "episode": episode, "t": frames - sd.fr.episode_start[frames],
           "label": sd.fr.label[frames], "T": sd.teacher[frames], "T_next": sd.next_logit[frames]}
    for track, key in zip(TRACKS, ("static", "forward", "mirrored")):
        t0 = time.time()
        out[key] = sd.values(model, frames, track, norm, args.device)
        print(f"  {key}: {len(frames)} frames ({time.time() - t0:.0f}s)", flush=True)
    dst = run_dir(args.run) / "eval_frames.npz"
    np.savez(dst, **out)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
