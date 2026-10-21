"""Distill the teacher into the multi-view track student (`student.py`).

Recipe (the best model of the scan_object ablations):

  loss      the "minus" loss on both inputs of every sample frame t:
                L = MSE(V_static, T_t) + MSE(V_forward - V_static, T_{t+16} - T_t)
            so the progress reward V_forward - V_static is supervised directly
  data      the scan train episodes + their noise branches (each from its
            branch frame - 15 on), frames t % 4 == 0 (those with wrist rows)
  sampling  symmetric batches (half success, half failure frames); in each
            half, `--change-frac` of the frames come from those whose teacher
            value changes over the chunk, |T_{t+16} - T_t| > `--change-thresh`,
            the rest uniformly
  init      from scratch; the normalizer is the teacher's

Checkpoint = the epoch with the lowest val loss (the same two terms) on every
`--val-stride`-th eval frame; early stop after `--patience` epochs.
Writes runs/<run>/student.pt and train_history.json.

    python -m mv_distill.train_student --teacher teacher --run student
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from .batching import Prefetcher
from .config import FAILURE, SUCCESS, run_dir
from .data import Normalizer
from .metrics import roc_auc
from .model import build
from .student import StudentData


def val_metrics(sd: StudentData, model, frames, norm, device) -> dict:
    static = sd.values(model, frames, "zero", norm, device)
    forward = sd.values(model, frames, "real", norm, device)
    r, g = forward - static, sd.next_logit[frames] - sd.teacher[frames]
    m = {"mse_static": float(((static - sd.teacher[frames]) ** 2).mean()), "mse_delta": float(((r - g) ** 2).mean()),
         "auc_static": roc_auc(static, sd.fr.label[frames]), "dcorr": float(np.corrcoef(r, g)[0, 1])}
    m["val_loss"] = m["mse_static"] + m["mse_delta"]
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="teacher", help="runs/<teacher>/: teacher.pt (normalizer), labels_*.npz")
    p.add_argument("--run", default="student")
    p.add_argument("--change-thresh", type=float, default=0.5)
    p.add_argument("--change-frac", type=float, default=0.75)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64,
                   help="frames per step (x2 inputs), half success / half failure")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="3 steps, a 1/40 val subset, one checkpoint")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

    sd = StudentData(args.teacher)
    norm = Normalizer.from_state_dict(torch.load(run_dir(args.teacher) / "teacher.pt", map_location="cpu",
                                                 weights_only=False)["normalizer"])
    config = {"inputs": "track_fused", "n_layers": args.n_layers, "dropout": args.dropout, "rope_spatial_scale": 0.01}
    model = build(config).to(device)
    print(f"student: {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M params -> runs/{args.run}")

    half = args.batch_size // 2
    n_change = round(args.change_frac * half)
    pools, change = {}, {}
    for label in (SUCCESS, FAILURE):
        f = pools[label] = sd.train_frames(label)
        change[label] = f[np.abs(sd.next_logit[f] - sd.teacher[f]) > args.change_thresh]
        print(f"  {'success' if label == SUCCESS else 'failure'}: {len(f)} train frames, {len(change[label])} change")

    def make_batch():
        frames = np.concatenate([np.concatenate([rng.choice(change[label], size=n_change, replace=False),
                                                 rng.choice(pools[label], size=half - n_change, replace=False)])
                                 for label in (SUCCESS, FAILURE)])
        b, b["target"] = sd.pair_batch(frames, norm, "cpu")
        return b

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.max_epochs * args.steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        q = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(q, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    val = sd.eval_frames()[::args.val_stride]
    if args.smoke:
        val = val[::40]
    prefetch = Prefetcher(make_batch)
    out = run_dir(args.run)
    history, best, since, step = [], float("inf"), 0, 0
    try:
        for epoch in range(args.max_epochs):
            t0, losses = time.time(), []
            for _ in range(3 if args.smoke else args.steps_per_epoch):
                b = {k: v.to(device, non_blocking=True) for k, v in prefetch.next().items()}
                tgt = b.pop("target")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(b).float()
                (s, f), (ts, tf) = pred.chunk(2), tgt.chunk(2)
                loss = ((s - ts) ** 2).mean() + (((f - s) - (tf - ts)) ** 2).mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
                step += 1
                losses.append(loss.item())
            m = val_metrics(sd, model, val, norm, device)
            m.update(epoch=epoch, step=step, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0])
            history.append(m)
            print(f"epoch {epoch:3d}  loss {m['loss']:.4f}  val {m['val_loss']:.4f} (static {m['mse_static']:.3f} "
                  f"delta {m['mse_delta']:.3f})  AUC {m['auc_static']:.4f}  dcorr {m['dcorr']:+.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if m["val_loss"] < best - args.min_delta:
                best, since = m["val_loss"], 0
                torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "config": config, "normalizer": norm.state_dict(), "teacher": args.teacher,
                            "epoch": epoch, "metrics": m, "args": vars(args)}, out / "student.pt")
            else:
                since += 1
            if args.smoke:
                print(f"smoke ok: peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
                break
            if since >= args.patience:
                break
    finally:
        prefetch.close()
    with open(out / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val {best:.4f} -> {out / 'student.pt'}")


if __name__ == "__main__":
    main()
