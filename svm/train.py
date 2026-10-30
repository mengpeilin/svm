"""Train a multi-view SVM discriminator f(s, a) on a cache built by `svm.cache`.

The SVM discriminator objective (Eq. 2) in the offline setting: BCE between
decision steps of successful (label 1) and failed (label 0) episodes, one
discriminator over all timesteps, SYMMETRIC batches (half D+, half D-; sampling
in proportion to pool size turns f into a Q-function, SVM Fig. 23).

AdamW, linear warmup then cosine decay over max_epochs x steps_per_epoch, bf16
autocast. After every epoch: balanced BCE (mean of the per-class means) and
step AUC on the held-out steps; the checkpoint is the epoch with the lowest
held-out loss (AUC keeps rising after the reward has started to overfit), and
training stops after `--patience` epochs without improvement.

Writes to `--out`: model.safetensors + config.json (loadable by
`MVTrackSVMReward` / `MVActionSVMReward`) and train_history.json.

    python -m svm.train --cache cache/scan_object --inputs track --out runs/track
    python -m svm.train --cache cache/scan_object --inputs action --out runs/action
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from .data import SVMData, assemble, fit_normalizer
from .metrics import roc_auc
from .reward_model import MODELS


class Prefetcher:
    """Runs `fn()` on a background thread, keeping `depth` batches ready, so the
    memmap gather overlaps the GPU step."""

    def __init__(self, fn, depth: int = 4):
        self.fn = fn
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.q.put(self.fn(), timeout=1.0)
            except queue.Full:
                continue
            except Exception as exc:  # surface worker failures on the main thread
                self.q.put(exc)
                return

    def next(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.stop.set()


@torch.no_grad()
def evaluate(model, data: SVMData, norm, inputs: str, device: str, chunk: int = 32) -> tuple[float, float]:
    """(step AUC, balanced BCE) over the held-out decision steps."""
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, scores, labels = {}, [], []
    for label in (1, 0):
        idx, out = data.pools[(label, "heldout")], []
        for s in range(0, len(idx), chunk):
            batch = assemble(data, idx[s : s + chunk], norm, inputs, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(model(batch).float())
        logits = torch.cat(out)
        losses[label] = bce(logits, torch.full_like(logits, float(label))).mean().item()
        scores.append(logits.cpu().numpy())
        labels.append(np.full(len(logits), label))
    model.train()
    return roc_auc(np.concatenate(scores), np.concatenate(labels)), 0.5 * (losses[1] + losses[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--inputs", choices=tuple(MODELS), required=True)
    p.add_argument("--out", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--holdout", type=int, default=50, help="held-out episodes per class (highest seeds)")
    p.add_argument("--failure-cap", type=int, default=250, help="frames kept of each failure episode")
    p.add_argument("--max-epochs", type=int, default=100)
    # scan_object: ~4.5k success / ~7.3k failure train steps, so 150 x 32
    # positives is ~1 pass over D+ per epoch.
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64, help="total; half success, half failure")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rope-spatial-scale", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="", help="log train / held-out curves to this wandb project")
    args = p.parse_args()

    hparams = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    data = SVMData(args.cache, args.holdout, args.failure_cap)
    print(data)
    norm = fit_normalizer(data, args.inputs)
    model_config = {"n_layers": args.n_layers, "dropout": args.dropout, "rope_spatial_scale": args.rope_spatial_scale}
    model = MODELS[args.inputs](**model_config).to(device)
    print(f"{type(model).__name__}: {sum(q.numel() for q in model.parameters()) / 1e6:.1f} M params, device {device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_epochs * args.steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    bce = nn.BCEWithLogitsLoss()
    half = args.batch_size // 2
    target = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)

    def make_batch():
        parts = [assemble(data, rng.choice(data.pools[(label, "train")], half, replace=False), norm, args.inputs, "cpu")
                 for label in (1, 0)]
        return {k: torch.cat([b[k] for b in parts]) for k in parts[0]}

    wb = None
    if args.wandb_project:
        import wandb

        wb = wandb.init(project=args.wandb_project, name=args.out.name, config=hparams)

    history, best, since_best, step = [], {"val_loss": float("inf")}, 0, 0
    prefetch = Prefetcher(make_batch)
    try:
        for epoch in range(args.max_epochs):
            t0, losses = time.time(), []
            for _ in range(args.steps_per_epoch):
                batch = {k: v.to(device, non_blocking=True) for k, v in prefetch.next().items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(batch)
                loss = bce(logits.float(), target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
                step += 1
                losses.append(loss.item())
                if wb is not None and step % 10 == 0:
                    wb.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]}, step=step)

            auc, val_loss = evaluate(model, data, norm, args.inputs, device)
            rec = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "val_auc": auc, "val_loss": val_loss}
            history.append(rec)
            if wb is not None:
                wb.log({"train/epoch_loss": rec["loss"], "heldout/loss": val_loss, "heldout/step_auc": auc}, step=step)
            print(f"epoch {epoch:3d}  loss {rec['loss']:.4f}  held-out AUC {auc:.4f}  loss {val_loss:.5f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

            if val_loss < best["val_loss"] - args.min_delta:
                best, since_best = rec, 0
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                          args.out / "model.safetensors")
                config = {
                    "model_config": model_config,
                    "normalizer": norm.state_dict(),
                    "training": {"inputs": args.inputs, "epoch": epoch, "val_auc": auc, "val_loss": val_loss,
                                 "hparams": hparams},
                }
                (args.out / "config.json").write_text(json.dumps(config, indent=1))
            else:
                since_best += 1
                if since_best >= args.patience:
                    print(f"no held-out loss improvement > {args.min_delta} for {args.patience} epochs")
                    break
    finally:
        prefetch.close()
        if wb is not None:
            wb.finish()

    (args.out / "train_history.json").write_text(json.dumps({"best": best, "history": history}, indent=1))
    print(f"selected epoch {best['epoch']}: held-out loss {best['val_loss']:.5f}, step AUC {best['val_auc']:.4f}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
