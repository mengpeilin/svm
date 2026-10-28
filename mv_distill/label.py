"""Label EVERY frame of a task (over all staged roots, base first) with its
teacher: T_t = f(s_t), the value the student distills.

Writes runs/<teacher>/labels_<task>.npz: logit [n_frames] (the task's frame
order on `data.Frames`' axis: its roots in MV_DISTILL_DATA order).

    python -m mv_distill.label --task hanging_mug --teacher teacher2_hanging_mug
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .batching import prefetched
from .config import FAILURE, SUCCESS, TASKS, run_dir
from .data import Frames, Normalizer
from .model import load, predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--teacher", required=True, help="run dir of the task's teacher")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    teacher = args.teacher
    model, ck = load(run_dir(teacher) / "teacher.pt", args.device)
    assert ck["config"]["inputs"] == "state" and ck["task"] == args.task, (ck["config"], ck["task"])
    norm = Normalizer.from_state_dict(ck["normalizer"])

    fr = Frames((args.task,))
    n = len(fr.label)
    t0 = time.time()

    def batches():
        for s in range(0, n, args.batch):
            if s // args.batch % 200 == 0:
                print(f"  {s}/{n} frames ({s / (time.time() - t0):.0f} fps)", flush=True)
            yield fr.inputs(np.arange(s, min(s + args.batch, n)), norm, args.device)

    logit = predict(model, prefetched(batches()))
    assert len(logit) == n and np.isfinite(logit).all()
    out = run_dir(teacher) / f"labels_{args.task}.npz"
    np.savez(out, logit=logit)
    print(f"{n} frames ({(time.time() - t0) / 60:.1f} min), mean logit success {logit[fr.label == SUCCESS].mean():+.2f} "
          f"/ failure {logit[fr.label == FAILURE].mean():+.2f} -> {out}")


if __name__ == "__main__":
    main()
