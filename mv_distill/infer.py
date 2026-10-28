"""Run the student on every eval sample frame of its tasks (each eval episode's
frames with a full chunk ahead) with the zero, the real and the mirrored
(-track) track.

Writes runs/<run>/eval_frames.npz (sorted global frames over the student's tasks):

  tasks                        task names; `task` indexes them
  frames, task, episode, t     global frame, its task, episode within the task (= within its
                               base root: eval episodes are base episodes), frame in episode
  label                        1 / 0
  T, T_next                    the task teacher's logit T_t and T_{t+16}
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
    p.add_argument("--chunk", type=int, default=64, help="frames per forward")
    args = p.parse_args()
    model, ck = load(run_dir(args.run) / "student.pt", args.device)
    norm = Normalizer.from_state_dict(ck["normalizer"])
    print(f"runs/{args.run}: epoch {ck['epoch']}, val loss {ck['metrics']['val_loss']:.4f}, "
          f"{len(ck['teachers'])} tasks")
    sd = StudentData(ck["teachers"], ck["config"]["arch"])
    fr = sd.fr
    frames = sd.eval_frames()
    task = fr.task[frames]
    episode = np.array([fr.episodes[e][1] for e in fr.episode_id[frames]])
    out = {"tasks": np.array(fr.tasks), "frames": frames, "task": task, "episode": episode,
           "t": frames - fr.episode_start[frames], "label": fr.label[frames],
           "T": sd.teacher[frames], "T_next": sd.next_logit[frames]}
    for track, key in zip(TRACKS, ("static", "forward", "mirrored")):
        t0 = time.time()
        out[key] = sd.values(model, frames, track, norm, args.device, args.chunk)
        print(f"  {key}: {len(frames)} frames ({time.time() - t0:.0f}s)", flush=True)
    dst = run_dir(args.run) / "eval_frames.npz"
    np.savez(dst, **out)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
