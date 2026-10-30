"""Score every decision step of the held-out episodes of a cache with a
checkpoint (local dir or Hugging Face repo id) and report `svm.metrics`.

Writes `<out>/rewards.npz` ("{success|failure}|{episode}|{logit|reward}", one
value per decision step of the WHOLE episode) and `<out>/metrics.json`
(failures cut to `--failure-cap` frames, as in training).

    python -m svm.evaluate --cache cache/scan_object --model track --ckpt runs/track
    python -m svm.evaluate --cache cache/scan_object --model action \
        --ckpt Mercury007/mv-action-svm-scan-object --out eval/action
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import SVMData, assemble
from .metrics import reward_metrics
from .preprocess import CHUNK
from .reward_model import MODELS, load_model, svm_reward


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--ckpt", required=True, help="checkpoint dir, or HF repo id")
    p.add_argument("--out", type=Path, default=None, help="default: the checkpoint dir")
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--failure-cap", type=int, default=250)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--chunk", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        from huggingface_hub import snapshot_download

        ckpt = Path(snapshot_download(args.ckpt))
    out = args.out or ckpt
    out.mkdir(parents=True, exist_ok=True)
    model, norm, _ = load_model(ckpt, args.model, device)
    data = SVMData(args.cache, args.holdout, args.failure_cap)
    print(data)

    rewards, series = {}, {1: [], 0: []}
    cap = -(-args.failure_cap // CHUNK)
    with torch.no_grad():
        for label, tag in ((1, "success"), (0, "failure")):
            for ep in data.episodes[(label, "heldout")]:
                steps = data.steps(ep, capped=False)
                logits = []
                for s in range(0, len(steps), args.chunk):
                    batch = assemble(data, steps[s : s + args.chunk], norm, args.model, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        logits.append(model(batch).float().cpu().numpy())
                logit = np.concatenate(logits)
                rewards[f"{tag}|{ep}|logit"] = logit
                rewards[f"{tag}|{ep}|reward"] = svm_reward(logit, args.lam, args.beta)
                series[label].append(logit if label else logit[:cap])

    np.savez(out / "rewards.npz", **rewards)
    m = reward_metrics([svm_reward(x, args.lam, args.beta) for x in series[1]],
                       [svm_reward(x, args.lam, args.beta) for x in series[0]],
                       np.concatenate(series[1] + series[0]), args.beta)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    for k, v in m.items():
        print(f"  {k:<20s} {v:.4f}" if isinstance(v, float) else f"  {k:<20s} {v}")
    print(f"wrote {out / 'rewards.npz'}, {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
