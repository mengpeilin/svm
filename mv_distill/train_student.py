"""Distill the per-task teachers into ONE multi-view track student cotrained
on all tasks (`student.py`).

Recipe:

  arch      "dino"   the DinoV2 multi-view track_fused transformer (`model.py`),
                     from scratch
            "qwen3d" Qwen-3D finetuned with a fresh LoRA (`qwen_model.py`)
  loss      the "minus" loss on both inputs of every sample frame t:
                L = MSE(V_static, T_t) + MSE(V_forward - V_static, T_{t+16} - T_t)
            so the progress reward V_forward - V_static is supervised directly
            (T from the frame's task teacher, runs/<teacher-pattern>)
  data      every sample frame of the train episodes of `--tasks` (all staged roots)
  sampling  symmetric batches (half success, half failure frames), each half
            drawn from the class's frames pooled over the tasks (so a task
            weighs by its frame count); in each half, `--change-frac` of the
            frames come from those whose teacher value changes over the chunk,
            |T_{t+16} - T_t| > `--change-thresh`, the rest uniformly
  schedule  linear warmup, cosine decay over max_epochs x steps_per_epoch
  init      the normalizer is fit on the train decision frames

Checkpoint = the epoch with the lowest val loss (the same two terms, pooled
over the tasks' frames) on every `--val-stride`-th eval frame; early stop
after `--patience` epochs. Writes runs/<run>/student.pt, every `--save-every`
epochs also runs/<run>/student_ep<epoch>.pt (epoch counted from 1), and
train_history.json.

Logs to wandb (project `--wandb-project`, run name = `--run`): the train loss
every `--log-every` steps, the val metrics (pooled and per task) and timings
every epoch; `--wandb-project ""` turns it off.

Multi-GPU (run.sbatch NPROC > 1: RANK / LOCAL_RANK / WORLD_SIZE per process): every rank draws its own `--batch-size` frames
(global batch = batch x world size) and evaluates a slice of the val frames.

    python -m mv_distill.train_student --run student2
    NPROC=2 sbatch --gres=gpu:2 run.sbatch mv_distill.train_student --arch qwen3d --run qwen3d
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .batching import Prefetcher
from .config import FAILURE, MOVE_THRESH, SUCCESS, parse_tasks, run_dir
from .data import fit_normalizer
from .metrics import roc_auc
from .student import StudentData, double


def gather(x: np.ndarray, world: int) -> list[np.ndarray]:
    if world == 1:
        return [x]
    out = [None] * world
    dist.all_gather_object(out, x)
    return out


def val_metrics(sd: StudentData, model, frames, norm, device, rank: int, world: int, chunk: int) -> dict:
    """Val metrics on `frames`; each rank scores frames[rank::world], all ranks get the result."""
    mine = frames[rank::world]
    parts = [gather(sd.values(model, mine, track, norm, device, chunk), world) for track in ("zero", "real")]
    static, forward = (np.concatenate([p[r] for r in range(world)]) for p in parts)
    frames = np.concatenate([frames[r::world] for r in range(world)])
    r, g = forward - static, sd.next_logit[frames] - sd.teacher[frames]
    m = {"mse_static": float(((static - sd.teacher[frames]) ** 2).mean()), "mse_delta": float(((r - g) ** 2).mean()),
         "auc_static": roc_auc(static, sd.fr.label[frames]), "dcorr": float(np.corrcoef(r, g)[0, 1])}
    m["val_loss"] = m["mse_static"] + m["mse_delta"]
    task = sd.fr.task[frames]
    m["task_val_loss"] = {name: float(((static - sd.teacher[frames]) ** 2)[task == k].mean()
                                      + ((r - g) ** 2)[task == k].mean()) for k, name in enumerate(sd.fr.tasks)}
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=("dino", "qwen3d"), default="dino")
    p.add_argument("--tasks", default="all", help="'all' or a comma-separated subset (one task: a specialist)")
    p.add_argument("--teacher-pattern", default="teacher2_{task}", help="runs/<pattern>/labels_<task>.npz")
    p.add_argument("--run", default="student2")
    p.add_argument("--change-thresh", type=float, default=0.5)
    p.add_argument("--change-frac", type=float, default=0.75)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=192,
                   help="frames per step and GPU (x2 inputs), half success / half failure")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm (0: off)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=0, help="also keep student_ep<N>.pt every N epochs (0: off)")
    p.add_argument("--val-stride", type=int, default=16)
    p.add_argument("--val-chunk", type=int, default=64)
    p.add_argument("--threads", type=int, default=4, help="batch-building threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="mv_distill", help="'' = no wandb")
    p.add_argument("--log-every", type=int, default=20, help="steps between train-loss logs")
    p.add_argument("--smoke", action="store_true", help="--smoke-steps steps, a 1/40 val subset, one checkpoint")
    p.add_argument("--smoke-steps", type=int, default=3)
    g = p.add_argument_group("dino")
    g.add_argument("--n-layers", type=int, default=6)
    g.add_argument("--dropout", type=float, default=0.1)
    g = p.add_argument_group("qwen3d")
    g.add_argument("--qwen-dir", default="/scratch/pmeng/hf/Qwen2.5-VL-3B-Instruct", help="config + tokenizer")
    g.add_argument("--qwen3d-ckpt", default="/scratch/pmeng/hf/Qwen-3D/qwen3d_3b.pth")
    g.add_argument("--voxel", type=float, default=0.02, help="voxel pooling cell (m); voxel_stats: 2 cm")
    g.add_argument("--move-thresh", type=float, default=MOVE_THRESH, help="a tracked patch moves above this (m)")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=float, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.0)
    g.add_argument("--no-grad-ckpt", action="store_true")
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    main_rank = rank == 0
    log = print if main_rank else (lambda *a, **k: None)

    sd = StudentData({t: args.teacher_pattern.format(task=t) for t in parse_tasks(args.tasks)}, args.arch)
    norm = fit_normalizer(sd.fr, np.concatenate([sd.fr.frames(label, "train", "decision")
                                                 for label in (SUCCESS, FAILURE)]))
    if args.arch == "dino":
        from .model import build
        config = {"arch": "dino", "inputs": "track_fused", "n_layers": args.n_layers, "dropout": args.dropout,
                  "rope_spatial_scale": 0.01}
        model = build(config).to(device)
    else:
        from .qwen_model import build, make_config
        config = make_config(sd.fr.tasks, args.qwen_dir, args.qwen3d_ckpt, args.voxel, args.move_thresh,
                             args.lora_r, args.lora_alpha, args.lora_dropout)
        model = build(config, device)
        if not args.no_grad_ckpt:
            model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [q for q in model.parameters() if q.requires_grad]
    log(f"student ({args.arch}): {sum(q.numel() for q in model.parameters()) / 1e6:.1f} M params, "
        f"{sum(q.numel() for q in params) / 1e6:.2f} M trainable -> runs/{args.run}  "
        f"({world} GPU x {args.batch_size} frames)")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device]) if world > 1 else model
    wb = None
    if main_rank and args.wandb_project and not args.smoke:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.run, group=f"student-{args.arch}",
                        config={**vars(args), "world_size": world, "trainable_params": sum(q.numel() for q in params),
                                "tasks": list(sd.fr.tasks), "teachers": sd.teachers})

    half = args.batch_size // 2
    n_change = round(args.change_frac * half)
    pools, change = {}, {}
    for label in (SUCCESS, FAILURE):
        f = pools[label] = sd.train_frames(label)
        change[label] = f[np.abs(sd.next_logit[f] - sd.teacher[f]) > args.change_thresh]
        log(f"  {'success' if label == SUCCESS else 'failure'}: {len(f)} train frames, {len(change[label])} change")

    def make_batch(rng):
        frames = np.concatenate([np.concatenate([rng.choice(change[label], size=n_change, replace=False),
                                                 rng.choice(pools[label], size=half - n_change, replace=False)])
                                 for label in (SUCCESS, FAILURE)])
        b, b["target"] = sd.pair_batch(frames, norm, "cpu")
        return {k: v.pin_memory() for k, v in b.items()}

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
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
    prefetch = Prefetcher(make_batch, depth=2 * args.threads, threads=args.threads, seed=args.seed * 1000 + rank)
    out = run_dir(args.run)
    history, best, since, step = [], float("inf"), 0, 0
    try:
        for epoch in range(args.max_epochs):
            t0, losses, wait = time.time(), [], 0.0
            n_steps = args.smoke_steps if args.smoke else args.steps_per_epoch
            for i in range(n_steps):
                if args.smoke and i == 1:      # time steps after the first (allocator, kernels warm)
                    torch.cuda.synchronize()
                    t_warm = time.time()
                tw = time.time()
                b = {k: v.to(device, non_blocking=True) for k, v in prefetch.next().items()}
                wait += time.time() - tw
                if args.arch == "dino":
                    b = double(b)
                tgt = b.pop("target")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(b).float()
                (s, f), (ts, tf) = pred.chunk(2), tgt.chunk(2)
                loss = ((s - ts) ** 2).mean() + (((f - s) - (tf - ts)) ** 2).mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                optim.step()
                sched.step()
                step += 1
                losses.append(loss.item())
                if wb is not None and step % args.log_every == 0:
                    wb.log({"train/loss": float(np.mean(losses[-args.log_every:])), "train/lr": sched.get_last_lr()[0],
                            "epoch": epoch}, step=step)
            train_s = time.time() - t0
            if args.smoke:
                torch.cuda.synchronize()
                log(f"smoke: {(time.time() - t_warm) / max(n_steps - 1, 1):.2f} s/step after the first, "
                    f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, losses {np.round(losses, 3)}")
            tv = time.time()
            m = val_metrics(sd, model, val, norm, device, rank, world, args.val_chunk)
            m.update(epoch=epoch, step=step, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0],
                     train_s=train_s, data_wait_s=wait, val_s=time.time() - tv)
            history.append(m)
            if wb is not None:
                wb.log({"epoch": epoch, "train/epoch_loss": m["loss"],
                        **{f"val/{k}": m[k] for k in ("val_loss", "mse_static", "mse_delta", "auc_static", "dcorr")},
                        **{f"val_task/{t}": v for t, v in m["task_val_loss"].items()},
                        "time/train_s": train_s, "time/data_wait_frac": wait / train_s, "time/val_s": m["val_s"]},
                       step=step)
            log(f"epoch {epoch:3d}  loss {m['loss']:.4f}  val {m['val_loss']:.4f} (static {m['mse_static']:.3f} "
                f"delta {m['mse_delta']:.3f})  AUC {m['auc_static']:.4f}  dcorr {m['dcorr']:+.3f}  "
                f"(train {train_s:.0f}s, data wait {100 * wait / train_s:.0f}%, val {m['val_s']:.0f}s)", flush=True)
            names = []
            if m["val_loss"] < best - args.min_delta:
                best, since = m["val_loss"], 0
                names.append("student.pt")
                if wb is not None:
                    wb.summary.update({"best/val_loss": best, "best/epoch": epoch})
            else:
                since += 1
            if args.save_every and (epoch + 1) % args.save_every == 0:
                names.append(f"student_ep{epoch + 1}.pt")
            if main_rank and names:
                state = (model.trainable_state() if args.arch == "qwen3d" else
                         {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                for name in names:
                    torch.save({"model": state, "config": config, "normalizer": norm.state_dict(),
                                "teachers": sd.teachers, "epoch": epoch, "metrics": m, "args": vars(args)},
                               out / name)
            if args.smoke:
                log(f"smoke ok: peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
                break
            if since >= args.patience:
                break
    finally:
        prefetch.close()
    if main_rank:
        with open(out / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
    if wb is not None:
        wb.finish()
    log(f"best val {best:.4f} -> {out / 'student.pt'}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
