"""Train ONE task's teacher: a state-only multi-view SVM discriminator f(s)
(no action, no track), so its logit T_t = V(s_t) is a state value.

SVM discriminator objective (BCE, success = 1 / failure = 0) on decision
frames (every 16th frame) of the task's train episodes, with symmetric
batches: half success, half failure frames, each drawn uniformly from its
class's frames. Checkpoint = the epoch with the lowest class-balanced BCE on
the eval decision frames (early stop after `--patience` epochs without
improvement). The normalizer is fit on the train decision frames.

Logs to wandb (project `--wandb-project`, group teacher, run name = `--run`):
the train loss every 10 steps, the eval AUC / BCE every epoch; `--wandb-project
""` turns it off.

Frames come from every staged root (MV_DISTILL_DATA); the eval decision
frames are the base root's held-out episodes. Writes runs/<run>/teacher.pt and
train_history.json.

    python -m mv_distill.train_teacher --task hanging_mug --run teacher2_hanging_mug
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .batching import Prefetcher
from .config import FAILURE, SUCCESS, TASKS, run_dir
from .data import Frames, fit_normalizer
from .metrics import roc_auc
from .model import build, predict


def evaluate(model, fr: Frames, frames: dict[int, np.ndarray], norm, device, chunk: int = 32) -> tuple[float, float]:
    """(AUC, class-balanced BCE) on the eval decision frames."""
    bce = nn.BCEWithLogitsLoss()
    logits = {label: predict(model, (fr.inputs(f[s:s + chunk], norm, device) for s in range(0, len(f), chunk)))
              for label, f in frames.items()}
    loss = 0.5 * sum(float(bce(torch.from_numpy(v), torch.full((len(v),), float(label))))
                     for label, v in logits.items())
    scores = np.concatenate([logits[SUCCESS], logits[FAILURE]])
    labels = np.r_[np.ones(len(logits[SUCCESS])), np.zeros(len(logits[FAILURE]))]
    return roc_auc(scores, labels), loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256, help="half success, half failure")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=4, help="batch-building threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="mv_distill", help="'' = no wandb")
    p.add_argument("--smoke", action="store_true", help="3 steps, a 1/20 eval subset, one checkpoint")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    fr = Frames((args.task,))
    train = {label: fr.frames(label, "train", "decision") for label in (SUCCESS, FAILURE)}
    held = {label: fr.frames(label, "eval", "decision")[::20 if args.smoke else 1] for label in (SUCCESS, FAILURE)}
    print(f"decision frames: train {len(train[SUCCESS])} success / {len(train[FAILURE])} failure, "
          f"eval {len(held[SUCCESS])} / {len(held[FAILURE])}")
    norm = fit_normalizer(fr, np.concatenate(list(train.values())))
    config = {"arch": "dino", "inputs": "state", "n_layers": args.n_layers, "dropout": args.dropout,
              "rope_spatial_scale": 0.01}
    model = build(config).to(device)
    print(f"teacher: {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M params -> runs/{args.run}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.max_epochs * args.steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        q = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(q, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    bce = nn.BCEWithLogitsLoss()
    half = args.batch_size // 2
    target = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)

    def make_batch(rng):
        frames = np.concatenate([np.sort(rng.choice(train[label], half, replace=False))
                                 for label in (SUCCESS, FAILURE)])
        return {k: v.pin_memory() for k, v in fr.inputs(frames, norm, "cpu").items()}

    prefetch = Prefetcher(make_batch, depth=2 * args.threads, threads=args.threads, seed=args.seed)
    out = run_dir(args.run)
    wb = None
    if args.wandb_project and not args.smoke:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.run, group="teacher", config=vars(args))
    history, best, since, step = [], float("inf"), 0, 0
    try:
        for epoch in range(args.max_epochs):
            t0, losses, wait = time.time(), [], 0.0
            for _ in range(3 if args.smoke else args.steps_per_epoch):
                tw = time.time()
                batch = {k: v.to(device, non_blocking=True) for k, v in prefetch.next().items()}
                wait += time.time() - tw
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                loss = bce(logits.float(), target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
                step += 1
                losses.append(loss.item())
                if wb is not None and step % 10 == 0:
                    wb.log({"train/loss": float(np.mean(losses[-10:])), "train/lr": sched.get_last_lr()[0],
                            "epoch": epoch}, step=step)
            train_s = time.time() - t0
            auc, eval_loss = evaluate(model, fr, held, norm, device)
            history.append({"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "eval_auc": auc,
                            "eval_loss": eval_loss, "lr": sched.get_last_lr()[0], "train_s": train_s,
                            "data_wait_s": wait})
            if wb is not None:
                wb.log({"epoch": epoch, "eval/auc": auc, "eval/bce": eval_loss, "time/train_s": train_s,
                        "time/data_wait_frac": wait / train_s}, step=step)
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  eval AUC {auc:.4f}  eval loss {eval_loss:.5f}  "
                  f"(train {train_s:.0f}s, data wait {100 * wait / train_s:.0f}%, total {time.time() - t0:.0f}s)",
                  flush=True)
            if eval_loss < best - args.min_delta:
                best, since = eval_loss, 0
                torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "config": config, "normalizer": norm.state_dict(), "epoch": epoch,
                            "eval_auc": auc, "eval_loss": eval_loss, "task": args.task, "args": vars(args)}, out / "teacher.pt")
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
    if wb is not None:
        wb.summary.update({"best/eval_bce": best})
        wb.finish()
    print(f"best eval loss {best:.5f} -> {out / 'teacher.pt'}")


if __name__ == "__main__":
    main()
